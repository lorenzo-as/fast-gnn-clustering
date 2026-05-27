"""Hydra entrypoint for CMSSW ROOT to canonical Parquet conversion."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from fastgnn.utils import get_project_root, resolve_project_path

_CONTROL_KEYS = {"input_files", "output_dir", "max_events", "overwrite"}


def main() -> None:
    """Convert configured CMSSW ROOT files into a CaloDataset directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = _compose_config(sys.argv[1:])
    convert_from_config(cfg)


def _compose_config(overrides: list[str]) -> DictConfig:
    config_dir = get_project_root() / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="convert/cmssw", overrides=overrides)


def convert_from_config(cfg: DictConfig) -> Path:
    """Run CMSSW conversion from a composed Hydra config."""
    from fastgnn.data.cmssw import convert_cmssw_root

    input_files = [resolve_project_path(path) for path in cfg.input_files]
    output_dir = resolve_project_path(cfg.output_dir)
    conversion_config = _conversion_config(cfg)

    return convert_cmssw_root(
        input_files=input_files,
        output_dir=output_dir,
        config=conversion_config,
        max_events=cfg.get("max_events"),
        overwrite=bool(cfg.get("overwrite", False)),
    )


def _conversion_config(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, Mapping):
        msg = f"Expected conversion config to be a mapping, got {type(container).__name__}"
        raise TypeError(msg)
    return {str(key): value for key, value in container.items() if key not in _CONTROL_KEYS}


if __name__ == "__main__":
    main()
