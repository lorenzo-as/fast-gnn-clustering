"""
Training loop for HGCAL GNN with Object Condensation loss using tf.GradientTape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
import time
from typing import Any, cast
import warnings

import numpy as np
import tensorflow as tf

from fastgnn.data.base import CaloDataset

from .objectcondensation_loss import (
    batch_and_mask_to_flat,
    calc_LV_Lbeta,
    calc_payload_correction_loss,
    calc_payload_loss,
    formatted_loss_components_string,
)
from .oc_outputs import OCOutputLayout, OCOutputSlices, split_oc_outputs
from .schedules import scheduled_scalar_value

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
    feature_names = cfg["data"]["feature_names"]
    normalization_method = cfg["data"].get("normalization", "zscore")
    output_layout = OCOutputLayout.from_config(model_cfg)
    payload_quantities = _payload_quantities(train_cfg)

    optimizer = _build_optimizer(train_cfg)

    best_val_loss = float("inf")
    history = {
        "train": [],
        "val": [],
        "val_reference": [],
        "train_components": [],
        "val_components": [],
        "qmin": [],
        "lr": [],
    }
    warned_padding = False

    for epoch in range(train_cfg["n_epochs"]):
        t0 = time.time()
        qmin = scheduled_scalar_value(
            train_cfg.get("qmin", 1.0),
            train_cfg.get("qmin_schedule"),
            epoch,
        )
        qmin_reference = float(train_cfg.get("qmin_reference", train_cfg.get("qmin", 1.0)))

        # ------------------------------------------------------------------
        # Training pass
        # ------------------------------------------------------------------
        train_losses = []
        train_components = []
        for batch in train_dataset.batches(
            max_vertices=model_cfg["max_vertices"],
            batch_size=train_cfg["batch_size"],
            feature_names=feature_names,
            payload_quantities=payload_quantities,
            shuffle=True,
            seed=cfg.get("seed", 42) + epoch,
            truncate=train_cfg.get("truncate", "first"),
            normalize_features=train_cfg.get("normalize_features", True),
            normalization_method=normalization_method,
        ):
            if not warned_padding and not np.all(batch["mask"]):
                warnings.warn(
                    "Batch contains padded vertices. qgravnet does not currently "
                    "mask padded vertices inside neighbour aggregation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_padding = True
            loss, components = _train_step(
                model,
                batch,
                optimizer,
                train_cfg,
                output_layout,
                qmin=qmin,
                epoch=epoch,
            )
            train_losses.append(float(loss))
            train_components.append(components)

        mean_train_loss = np.mean(train_losses)
        mean_train_components = _mean_components(train_components)

        # ------------------------------------------------------------------
        # Validation pass
        # ------------------------------------------------------------------
        val_losses = []
        val_reference_losses = []
        val_components = []
        for batch in val_dataset.batches(
            max_vertices=model_cfg["max_vertices"],
            batch_size=train_cfg["batch_size"],
            feature_names=feature_names,
            payload_quantities=payload_quantities,
            shuffle=False,
            seed=0,
            truncate=train_cfg.get("truncate", "first"),
            normalize_features=train_cfg.get("normalize_features", True),
            normalization_method=normalization_method,
        ):
            loss, components = _eval_step(
                model,
                batch,
                train_cfg,
                output_layout,
                qmin=qmin,
                epoch=epoch,
            )
            val_losses.append(float(loss))
            val_components.append(components)
            if qmin_reference == qmin:
                val_reference_losses.append(float(loss))
            else:
                reference_loss, _ = _eval_step(
                    model,
                    batch,
                    train_cfg,
                    output_layout,
                    qmin=qmin_reference,
                    epoch=epoch,
                )
                val_reference_losses.append(float(reference_loss))

        mean_val_loss = np.mean(val_losses)
        mean_val_reference_loss = np.mean(val_reference_losses)
        mean_val_components = _mean_components(val_components)

        history["train"].append(mean_train_loss)
        history["val"].append(mean_val_loss)
        history["val_reference"].append(mean_val_reference_loss)
        history["train_components"].append(mean_train_components)
        history["val_components"].append(mean_val_components)
        history["qmin"].append(qmin)
        history["lr"].append(_optimizer_learning_rate(optimizer))

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch + 1:03d}/{train_cfg['n_epochs']}  "
            f"train={mean_train_loss:.4f}  val={mean_val_loss:.4f}  "
            f"val_reference={mean_val_reference_loss:.4f}  "
            f"qmin={qmin:.4g}  lr={history['lr'][-1]:.4g}  "
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
        min_ckpt_epoch = train_cfg.get("best_model_start_epoch", 0)
        if epoch >= min_ckpt_epoch and mean_val_reference_loss < best_val_loss:
            best_val_loss = mean_val_reference_loss
            model.save(str(output_dir / "best_model.keras"))
            logger.info("  saved best model (val_reference_loss=%.4f)", best_val_loss)

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
    output_layout: OCOutputLayout | None = None,
    *,
    qmin: float | None = None,
    epoch: int = 0,
) -> tuple[tf.Tensor, dict]:
    """Single training step with GradientTape."""
    batch_tensors = {k: tf.constant(v) for k, v in batch.items()}
    with tf.GradientTape() as tape:
        loss, components = _forward_pass(
            model, batch_tensors, train_cfg, output_layout, qmin=qmin, epoch=epoch, training=True
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=True))
    return loss, components


def _eval_step(
    model: tf.keras.Model,
    batch: dict,
    train_cfg: dict,
    output_layout: OCOutputLayout | None = None,
    *,
    qmin: float | None = None,
    epoch: int = 0,
) -> tuple[tf.Tensor, dict]:
    """Single validation step (no gradient)."""
    batch_tensors = {k: tf.constant(v) for k, v in batch.items()}
    return _forward_pass(
        model, batch_tensors, train_cfg, output_layout, qmin=qmin, epoch=epoch, training=False
    )


def _forward_pass(
    model: tf.keras.Model,
    batch_tensors: dict[str, tf.Tensor],
    train_cfg: dict,
    output_layout: OCOutputLayout | None,
    *,
    qmin: float | None,
    epoch: int,
    training: bool,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    hit_object_id = tf.cast(batch_tensors["hit_object_id"], tf.int32)
    outputs = model(batch_tensors["features"], training=training)
    if output_layout is None:
        output_layout = OCOutputLayout.from_output_dim(int(outputs.shape[-1]))
    output_slices = split_oc_outputs(outputs, output_layout)
    flat, batch_idx = batch_and_mask_to_flat(
        {
            "beta": output_slices.beta,
            "cluster_coords": output_slices.cluster_coords,
            "hit_object_id": hit_object_id,
            **_payload_flat_inputs(output_slices, output_layout, batch_tensors),
            "mask": batch_tensors["mask"],
        }
    )
    _assert_no_padding_labels(flat["hit_object_id"])
    components = calc_LV_Lbeta(
        beta=flat["beta"],
        cluster_space_coords=flat["cluster_coords"],
        cluster_index_per_event=tf.cast(flat["hit_object_id"], tf.int32),
        batch=batch_idx,
        qmin=train_cfg.get("qmin", 1.0) if qmin is None else qmin,
        beta_stabilizing=train_cfg.get("beta_stabilizing", "soft_q_scaling"),
        beta_term_option=train_cfg.get("beta_term_option", "paper"),
        return_components=True,
    )
    components = cast(dict[str, tf.Tensor], components)
    _add_payload_components(components, flat, batch_idx, output_layout, train_cfg)
    _weight_oc_components(components, train_cfg, epoch=epoch)
    return components["L_total"], components


def _payload_flat_inputs(
    output_slices: OCOutputSlices,
    output_layout: OCOutputLayout,
    batch_tensors: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    if output_layout.payload is None:
        return {}
    if "payload_targets" not in batch_tensors:
        raise KeyError(
            "Configured training.payload.quantities but batch is missing payload_targets"
        )
    if output_slices.payload is None:
        raise ValueError("Configured payload layout but split outputs did not produce payload")
    flat_inputs = {
        "payload_predictions": output_slices.payload,
        "payload_targets": batch_tensors["payload_targets"],
    }
    if "payload_seeds" in batch_tensors:  # correction mode
        flat_inputs["payload_seeds"] = batch_tensors["payload_seeds"]
    return flat_inputs


def _add_payload_components(
    components: dict[str, tf.Tensor],
    flat: dict,
    batch_idx: tf.Tensor,
    output_layout: OCOutputLayout,
    train_cfg: dict,
) -> None:
    payload_quantities = _payload_quantities(train_cfg)
    if not payload_quantities:
        return
    if output_layout.payload is None:
        raise ValueError("training.payload.quantities requires model.payload_output_dim > 0")
    payload_cfg = train_cfg.get("payload") or {}
    common = {
        "payload_predictions": flat["payload_predictions"],
        "payload_targets": flat["payload_targets"],
        "cluster_index_per_event": tf.cast(flat["hit_object_id"], tf.int32),
        "batch": batch_idx,
        "payload_specs": payload_quantities,
        "huber_delta": float(payload_cfg.get("huber_delta", 1.0)),
        "return_components": True,
    }
    if str(payload_cfg.get("mode", "direct")) == "correction":
        payload_components = calc_payload_correction_loss(
            payload_seeds=flat["payload_seeds"],
            beta=flat["beta"],
            max_log_corr=float(payload_cfg.get("max_log_corr", 3.0)),
            weighting=str(payload_cfg.get("weighting", "beta2")),
            beta_detach=bool(payload_cfg.get("beta_detach", False)),
            **common,
        )
    else:
        payload_components = calc_payload_loss(**common)
    components.update(cast(dict[str, tf.Tensor], payload_components))


def _payload_quantities(train_cfg: dict) -> list:
    payload_cfg = train_cfg.get("payload") or {}
    return list(payload_cfg.get("quantities") or [])


def _weight_oc_components(
    components: dict[str, tf.Tensor],
    train_cfg: dict,
    *,
    epoch: int = 0,
) -> None:
    """Scale OC component dict in place to match the optimized objective."""
    weights = _oc_loss_weights(train_cfg, epoch=epoch)

    components["L_V_attractive"] = weights["L_V_attractive"] * components["L_V_attractive"]
    components["L_V_repulsive"] = weights["L_V_repulsive"] * components["L_V_repulsive"]
    components["L_beta_sig"] = weights["L_beta_sig"] * components["L_beta_sig"]
    components["L_beta_noise"] = weights["L_beta_noise"] * components["L_beta_noise"]

    for key in ("L_beta_norms_term", "L_beta_logbeta_term"):
        if key in components:
            components[key] = weights["L_beta_sig"] * components[key]

    components["L_V"] = components["L_V_attractive"] + components["L_V_repulsive"]
    components["L_beta"] = components["L_beta_sig"] + components["L_beta_noise"]
    if "L_payload" in components:
        for key in list(components):
            if key.startswith("L_payload"):
                components[key] = weights["L_payload"] * components[key]
    components["L_total"] = (
        components["L_V"]
        + components["L_beta"]
        + components.get(
            "L_payload",
            tf.constant(0.0, dtype=components["L_V"].dtype),
        )
    )


def _oc_loss_weights(train_cfg: dict, *, epoch: int = 0) -> dict[str, float]:
    weights = train_cfg.get("loss_weights") or {}
    return {
        "L_V_attractive": _scheduled_loss_weight(weights, "L_V_attractive", 1.0, epoch),
        "L_V_repulsive": _scheduled_loss_weight(weights, "L_V_repulsive", 1.0, epoch),
        "L_beta_sig": _scheduled_loss_weight(weights, "L_beta_sig", 1.0, epoch),
        "L_beta_noise": _scheduled_loss_weight(weights, "L_beta_noise", 0.1, epoch),
        "L_payload": _scheduled_loss_weight(weights, "L_payload", 1.0, epoch),
    }


def _scheduled_loss_weight(weights: dict, key: str, default: float, epoch: int) -> float:
    value = weights.get(key, default)
    if _is_sequence_schedule(value):
        return scheduled_scalar_value(default, {"points": value}, epoch)
    if isinstance(value, Mapping):
        return scheduled_scalar_value(default, value, epoch)
    return float(value)


def _is_sequence_schedule(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


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
    return float(value.numpy()) if isinstance(value, tf.Tensor) else float(value)


def _assert_no_padding_labels(hit_object_id: tf.Tensor) -> None:
    tf.debugging.assert_greater_equal(
        tf.cast(hit_object_id, tf.int32),
        tf.constant(0, dtype=tf.int32),
        message="Padding label -1 reached the OC loss. Check batch mask handling.",
    )


def _build_optimizer(train_cfg: dict) -> tf.keras.optimizers.Optimizer:
    lr = _build_learning_rate(train_cfg)
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


def _build_learning_rate(
    train_cfg: dict,
) -> float | tf.keras.optimizers.schedules.LearningRateSchedule:
    schedule_cfg = train_cfg.get("lr_schedule")
    if schedule_cfg is None:
        return train_cfg.get("lr", 1e-3)
    return tf.keras.optimizers.schedules.deserialize(
        {
            "class_name": schedule_cfg["class_name"],
            "config": dict(schedule_cfg.get("config", {})),
        }
    )


def _optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    return _to_float(optimizer.learning_rate)
