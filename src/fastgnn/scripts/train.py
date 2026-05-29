"""Hydra training entrypoint for fast-gnn-clustering."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastgnn.utils import get_project_root, resolve_project_path


def main() -> None:
    """Run a configured CMSSW Object Condensation training job."""
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

    from fastgnn.training.trainer import train

    train(model, train_ds, val_ds, cfg, output_dir)


def _compose_config(overrides: list[str]) -> DictConfig:
    config_dir = get_project_root() / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="train", overrides=overrides)


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

    if cfg.data.name != "cmssw_processed":
        raise NotImplementedError(
            f"Only data.name='cmssw_processed' is supported, got {cfg.data.name!r}"
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
