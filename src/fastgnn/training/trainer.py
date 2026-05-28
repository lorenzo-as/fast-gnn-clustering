"""
Training loop for HGCAL GNN with Object Condensation loss using tf.GradientTape.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import cast
import warnings

import numpy as np
import tensorflow as tf

from fastgnn.data.base import CaloDataset

from .objectcondensation import (
    batch_and_mask_to_flat,
    calc_LV_Lbeta,
    formatted_loss_components_string,
)

logger = logging.getLogger(__name__)


def train(
    model: tf.keras.Model,
    train_dataset: CaloDataset,
    val_dataset: CaloDataset,
    cfg: dict,
    output_dir: str | Path,
) -> None:
    """
    Main training loop.

    Args:
        model:          QGravNet (or any model returning (beta, cluster_coords))
        train_dataset:  canonical CaloDataset split="train"
        val_dataset:    canonical CaloDataset split="val"
        cfg:            full config dict (model + training sections)
        output_dir:     where to save checkpoints and logs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    optimizer = _build_optimizer(train_cfg)

    best_val_loss = float("inf")
    history = {
        "train": [],
        "val": [],
        "train_components": [],
        "val_components": [],
    }
    warned_padding = False

    for epoch in range(train_cfg["n_epochs"]):
        t0 = time.time()

        # ------------------------------------------------------------------
        # Training pass
        # ------------------------------------------------------------------
        train_losses = []
        train_components = []
        for batch in train_dataset.batches(
            max_vertices=model_cfg["max_vertices"],
            batch_size=train_cfg["batch_size"],
            feature_names=model_cfg["feature_names"],
            shuffle=True,
            seed=cfg.get("seed", 42) + epoch,
            truncate=train_cfg.get("truncate", "first"),
            normalize_features=train_cfg.get("normalize_features", True),
        ):
            if not warned_padding and not np.all(batch["mask"]):
                warnings.warn(
                    "Batch contains padded vertices. qgravnet does not currently "
                    "mask padded vertices inside neighbour aggregation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_padding = True
            loss, components = _train_step(model, batch, optimizer, train_cfg)
            train_losses.append(float(loss))
            train_components.append(components)

        mean_train_loss = np.mean(train_losses)
        mean_train_components = _mean_components(train_components)

        # ------------------------------------------------------------------
        # Validation pass
        # ------------------------------------------------------------------
        val_losses = []
        val_components = []
        for batch in val_dataset.batches(
            max_vertices=model_cfg["max_vertices"],
            batch_size=train_cfg["batch_size"],
            feature_names=model_cfg["feature_names"],
            shuffle=False,
            seed=0,
            truncate=train_cfg.get("truncate", "first"),
            normalize_features=train_cfg.get("normalize_features", True),
        ):
            loss, components = _eval_step(model, batch, train_cfg)
            val_losses.append(float(loss))
            val_components.append(components)

        mean_val_loss = np.mean(val_losses)
        mean_val_components = _mean_components(val_components)

        history["train"].append(mean_train_loss)
        history["val"].append(mean_val_loss)
        history["train_components"].append(mean_train_components)
        history["val_components"].append(mean_val_components)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch + 1:03d}/{train_cfg['n_epochs']}  "
            f"train={mean_train_loss:.4f}  val={mean_val_loss:.4f}  "
            f"({elapsed:.1f}s)"
        )
        if train_losses and (epoch + 1) % train_cfg.get("log_components_every", 10) == 0:
            logger.info("  train components:")
            logger.info(formatted_loss_components_string(mean_train_components))
            logger.info("  val components:")
            logger.info(formatted_loss_components_string(mean_val_components))

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            model.save(str(output_dir / "best_model.keras"))
            logger.info("  saved best model (val_loss=%.4f)", best_val_loss)

        if (epoch + 1) % train_cfg.get("save_every", 10) == 0:
            model.save(str(output_dir / f"checkpoint_epoch{epoch + 1:03d}.keras"))

        np.save(str(output_dir / "history.npy"), np.asarray(history, dtype=object))

    model.save(str(output_dir / "final_model.keras"))
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Outputs saved to: %s", output_dir)


def _train_step(
    model: tf.keras.Model,
    batch: dict,
    optimizer: tf.keras.optimizers.Optimizer,
    train_cfg: dict,
) -> tuple[tf.Tensor, dict]:
    """Single training step with GradientTape."""
    batch_tensors = {k: tf.constant(v) for k, v in batch.items()}
    hit_object_id = tf.cast(batch_tensors["hit_object_id"], tf.int32)

    with tf.GradientTape() as tape:
        outputs = model(batch_tensors["features"], training=True)
        beta = tf.sigmoid(outputs[..., 0])
        cluster_coords = outputs[..., 1:]
        flat, batch_idx = batch_and_mask_to_flat(
            {
                "beta": beta,
                "cluster_coords": cluster_coords,
                "hit_object_id": hit_object_id,
                "mask": batch_tensors["mask"],
            }
        )
        _assert_no_padding_labels(flat["hit_object_id"])
        components = calc_LV_Lbeta(
            beta=flat["beta"],
            cluster_space_coords=flat["cluster_coords"],
            cluster_index_per_event=tf.cast(flat["hit_object_id"], tf.int32),
            batch=batch_idx,
            qmin=train_cfg.get("qmin", 1.0),
            s_B=train_cfg.get("s_B", 0.1),
            beta_stabilizing=train_cfg.get("beta_stabilizing", "soft_q_scaling"),
            beta_term_option=train_cfg.get("beta_term_option", "paper"),
            return_components=True,
        )
        components = cast(dict[str, tf.Tensor], components)
        loss = components["L_V"] + components["L_beta"]

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=True))
    return loss, components


def _eval_step(
    model: tf.keras.Model,
    batch: dict,
    train_cfg: dict,
) -> tuple[tf.Tensor, dict]:
    """
    Single validation step (no gradient).

    Returns:
        Tuple of (loss, components) where components is a dictionary of loss components.
    """
    batch_tensors = {k: tf.constant(v) for k, v in batch.items()}
    hit_object_id = tf.cast(batch_tensors["hit_object_id"], tf.int32)

    outputs = model(batch_tensors["features"], training=False)
    flat, batch_idx = batch_and_mask_to_flat(
        {
            "beta": tf.sigmoid(outputs[..., 0]),
            "cluster_coords": outputs[..., 1:],
            "hit_object_id": hit_object_id,
            "mask": batch_tensors["mask"],
        }
    )
    _assert_no_padding_labels(flat["hit_object_id"])
    components = calc_LV_Lbeta(
        beta=flat["beta"],
        cluster_space_coords=flat["cluster_coords"],
        cluster_index_per_event=tf.cast(flat["hit_object_id"], tf.int32),
        batch=batch_idx,
        qmin=train_cfg.get("qmin", 1.0),
        s_B=train_cfg.get("s_B", 0.1),
        beta_stabilizing=train_cfg.get("beta_stabilizing", "soft_q_scaling"),
        beta_term_option=train_cfg.get("beta_term_option", "paper"),
        return_components=True,
    )
    components = cast(dict[str, tf.Tensor], components)
    loss = components["L_V"] + components["L_beta"]
    return loss, components


def _mean_components(components_per_batch: list[dict[str, tf.Tensor]]) -> dict[str, float]:
    """Average loss component dictionaries over batches."""
    if not components_per_batch:
        return {}

    keys = components_per_batch[0].keys()
    return {
        key: float(np.mean([_to_float(components[key]) for components in components_per_batch]))
        for key in keys
    }


def _to_float(value: tf.Tensor | float) -> float:
    return float(value.numpy()) if hasattr(value, "numpy") else float(value)


def _assert_no_padding_labels(hit_object_id: tf.Tensor) -> None:
    tf.debugging.assert_greater_equal(
        tf.cast(hit_object_id, tf.int32),
        tf.constant(0, dtype=tf.int32),
        message="Padding label -1 reached the OC loss. Check batch mask handling.",
    )


def _build_optimizer(train_cfg: dict) -> tf.keras.optimizers.Optimizer:
    lr = train_cfg.get("lr", 1e-3)
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    optimizer_kwargs = {}
    if train_cfg.get("global_clipnorm") is not None:
        optimizer_kwargs["global_clipnorm"] = train_cfg.get("global_clipnorm")

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr, **optimizer_kwargs)
    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=train_cfg.get("momentum", 0.9),
            **optimizer_kwargs,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
