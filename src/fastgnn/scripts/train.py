"""Hydra training entrypoint for fast-gnn-clustering."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastgnn.data import CaloDataset
from fastgnn.data.base import _payload_quantity_dim
from fastgnn.utils import get_project_root, resolve_project_path

logger = logging.getLogger(__name__)


def main() -> None:
    """Run a configured training job."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = _compose_config(sys.argv[1:])
    output_dir = _resolve_output_dir(cfg, sys.argv[1:])
    cfg.output_dir = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    _set_seeds(int(cfg.seed))

    train_ds, val_ds = _load_datasets(cfg)
    model = _build_gravnet_model(cfg, n_features=len(cfg.data.feature_names))
    _validate_training_contract(model, train_ds, val_ds, cfg)

    from fastgnn.training.trainer import train

    train(model, train_ds, val_ds, cfg, output_dir)


def _compose_config(overrides: list[str]) -> DictConfig:
    config_dir = get_project_root() / "configs"
    configs = sorted(p.stem for p in config_dir.glob("*.yaml"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train", choices=configs, help="Config to use.")
    parser.add_argument("--name", default=None, help="Human-readable run name.")
    args, hydra_overrides = parser.parse_known_args(overrides)

    logger.info("Using config file: %s", config_dir / f"{args.config}.yaml")

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=args.config, overrides=hydra_overrides)
    if args.name is not None:
        cfg.run_name = args.name
    return cfg


def _resolve_output_dir(cfg: DictConfig, overrides: list[str]) -> Path:
    output_overridden = any(override.startswith("output_dir=") for override in overrides)
    hydra_run_dir = _override_value(overrides, "hydra.run.dir")
    output_dir = str(cfg.output_dir)
    if hydra_run_dir is not None and not output_overridden:
        output_dir = hydra_run_dir
    if "${now:" in output_dir:
        output_dir = _resolve_now(output_dir)
    return resolve_project_path(output_dir)


def _override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            return override[len(prefix) :]
    return None


def _resolve_now(value: str) -> str:
    now = datetime.now()
    return value.replace("${now:%Y-%m-%d}", now.strftime("%Y-%m-%d")).replace(
        "${now:%H-%M-%S}", now.strftime("%H-%M-%S")
    )


def _set_seeds(seed: int) -> None:
    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)  # noqa: NPY002 - intentionally seed legacy global state.
    tf.random.set_seed(seed)


def _load_datasets(cfg: DictConfig):
    from fastgnn.data import CaloDataset

    if not str(cfg.data.name).startswith("cmssw_processed"):
        raise NotImplementedError(
            f"Only cmssw_processed* datasets are supported, got {cfg.data.name!r}"
        )

    dataset_dir = resolve_project_path(cfg.data.dataset_dir)
    max_events = cfg.data.get("max_events")
    train_ds = CaloDataset(dataset_dir, split="train", max_events=max_events)
    val_ds = CaloDataset(dataset_dir, split="val", max_events=max_events)
    return train_ds, val_ds


def _build_gravnet_model(cfg: DictConfig, n_features: int):
    from qgravnet import GravNetFactory, QGravNetFactory

    from fastgnn.training.oc_outputs import OCOutputLayout

    model_cfg = cfg.model
    OCOutputLayout.from_config(model_cfg)
    quantized = model_cfg.get("quantized")
    if not isinstance(quantized, bool):
        raise ValueError(
            "cfg.model.quantized must be set to a boolean to select "
            "GravNetFactory or QGravNetFactory"
        )

    dense_layer_dims = _optional_container(model_cfg.get("dense_layer_dims"))
    gravnet_cfg = _optional_container(model_cfg.get("gravnet_cfg")) or {}
    selector_cfg = _optional_container(model_cfg.get("selector_cfg")) or {}

    factory_cls = QGravNetFactory if quantized else GravNetFactory
    factory_kwargs = {
        "n_blocks": int(model_cfg.n_blocks),
        "n_neighbours": int(model_cfg.n_neighbours),
        "n_dimensions": int(model_cfg.n_dimensions),
        "n_filters": int(model_cfg.n_filters),
        "n_propagate": int(model_cfg.n_propagate),
        "n_postgn_dense_blocks": int(model_cfg.n_postgn_dense_blocks),
        "output_dim": int(model_cfg.output_dim),
        "output_head": str(model_cfg.get("output_head", "oc")),
        "distance_metric": str(model_cfg.distance_metric),
        "neighbour_selector": str(model_cfg.neighbour_selector),
        "gravnet_cfg": gravnet_cfg,
        "selector_cfg": selector_cfg,
        "dense_layer_dims": dense_layer_dims,
    }
    if quantized:
        quant_cfg = model_cfg.get("quantization", {})
        factory_kwargs.update(
            dense_kernel_quantizer=quant_cfg.get("dense_kernel_quantizer"),
            dense_bias_quantizer=quant_cfg.get("dense_bias_quantizer"),
        )

    factory = factory_cls(**factory_kwargs)
    return factory.create_keras_model(
        n_vertices=int(model_cfg.max_vertices),
        n_features=n_features,
    )


def _validate_training_contract(
    model, train_dataset: CaloDataset, val_dataset: CaloDataset, cfg: DictConfig
) -> None:
    """Fail before training when configured model inputs and dataset fields disagree."""
    feature_names = list(cfg.data.feature_names)
    normalize = bool(cfg.training.get("normalize_features", True))
    payload_quantities = list((cfg.training.get("payload") or {}).get("quantities") or [])
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        try:
            dataset.validate_feature_names(feature_names, normalize=normalize)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{split} dataset feature validation failed: {exc}") from exc
        if payload_quantities:
            _validate_payload_quantity_fields(split, dataset, payload_quantities)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(f"Expected one model input, got input shapes {input_shape}")
        input_shape = input_shape[0]
    if not input_shape or input_shape[-1] != len(feature_names):
        raise ValueError(
            "Model input feature dimension does not match cfg.data.feature_names: "
            f"model.input_shape={input_shape}, configured features={feature_names}"
        )
    if payload_quantities:
        from fastgnn.training.oc_outputs import OCOutputLayout

        layout = OCOutputLayout.from_config(cfg.model)
        target_dim = sum(_payload_quantity_dim(quantity) for quantity in payload_quantities)
        if layout.payload_dim != target_dim:
            raise ValueError(
                "Configured payload target dimension does not match model.output_layout.payload.dim: "
                f"payload targets={target_dim}, model payload dim={layout.payload_dim}"
            )


def _validate_payload_quantity_fields(
    split: str,
    dataset: CaloDataset,
    payload_quantities: list,
) -> None:
    from fastgnn.data.base import _payload_is_correction
    from fastgnn.data.object_properties import PROPERTY_NAMES

    if _payload_is_correction(payload_quantities):
        # Correction mode: seeds are per-hit features; targets are computed object aggregates.
        hit_fields = set(dataset.fields.get("hits", []))
        bad_seeds = sorted(
            {str(q["seed"]) for q in payload_quantities if str(q["seed"]) not in hit_fields}
        )
        bad_targets = sorted(
            {
                str(q["target"])
                for q in payload_quantities
                if str(q["target"]) not in set(PROPERTY_NAMES)
            }
        )
        if bad_seeds:
            raise ValueError(
                f"{split} dataset is missing payload seed hit field(s): {', '.join(bad_seeds)}. "
                f"Available hit fields: {', '.join(sorted(hit_fields)) or '<none>'}"
            )
        if bad_targets:
            raise ValueError(
                f"{split} payload target(s) are not computable object properties: "
                f"{', '.join(bad_targets)}. Available: {', '.join(PROPERTY_NAMES)}"
            )
        return

    available = set(dataset.fields.get("truth.objects", []))
    missing = sorted(
        {
            str(quantity["field"])
            for quantity in payload_quantities
            if str(quantity["field"]) not in available
        }
    )
    if missing:
        available_msg = ", ".join(sorted(available)) or "<none>"
        raise ValueError(
            f"{split} dataset is missing payload truth.objects field(s): "
            f"{', '.join(missing)}. Available fields: {available_msg}"
        )


def _optional_container(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    container = OmegaConf.to_container(value, resolve=True)

    if not isinstance(container, Mapping):
        msg = f"Expected OmegaConf container to be a mapping, got {type(container).__name__}"
        raise TypeError(msg)

    return dict(container)


if __name__ == "__main__":
    main()
