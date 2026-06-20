"""Hydra entrypoint for CMSSW ROOT to canonical Parquet conversion."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import re
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import yaml

from fastgnn.utils import get_project_root, resolve_project_path

logger = logging.getLogger(__name__)

_CONTROL_KEYS = {"input_files", "output_dir", "max_events", "overwrite", "num_workers"}
_DEFAULT_CONFIG_NAME = "convert/cmssw"
_GLOB_CHARS = "*?["


def main() -> None:
    """Convert configured CMSSW ROOT files into a CaloDataset directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = _compose_config(sys.argv[1:])
    convert_from_config(cfg)
    logger.info("Finished CMSSW conversion.")


def _compose_config(overrides: list[str]) -> DictConfig:
    config_name, hydra_overrides = _split_config_name(overrides)
    config_dir = get_project_root() / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=hydra_overrides)


def convert_from_config(cfg: DictConfig) -> Path:
    """Run CMSSW conversion from a composed Hydra config."""
    from fastgnn.data.cmssw import convert_cmssw_root

    input_files = _resolve_input_files(cfg.input_files)
    output_dir = resolve_project_path(cfg.output_dir)
    conversion_config = _conversion_config(cfg)

    logger.info("Starting CMSSW conversion of %d ROOT files.", len(input_files))
    logger.debug("Input files: %s\nConfig: %s", input_files, yaml.safe_dump(conversion_config))

    return convert_cmssw_root(
        input_files=input_files,
        output_dir=output_dir,
        config=conversion_config,
        max_events=cfg.get("max_events"),
        overwrite=bool(cfg.get("overwrite", False)),
        num_workers=int(cfg.get("num_workers", 1) or 1),
    )


def _conversion_config(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, Mapping):
        msg = f"Expected conversion config to be a mapping, got {type(container).__name__}"
        raise TypeError(msg)
    return {str(key): value for key, value in container.items() if key not in _CONTROL_KEYS}


def _split_config_name(args: list[str]) -> tuple[str, list[str]]:
    config_name = _DEFAULT_CONFIG_NAME
    overrides: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"--config-name", "--config"}:
            if i + 1 >= len(args):
                raise ValueError(f"{arg} requires a config name")
            config_name = args[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        overrides.append(arg)
        i += 1
    return config_name, overrides


def _resolve_input_files(input_files: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for input_file in input_files:
        raw_path = str(input_file)
        if not any(char in raw_path for char in _GLOB_CHARS):
            paths.append(resolve_project_path(raw_path))
            continue

        pattern = Path(raw_path)
        resolved_pattern = pattern if pattern.is_absolute() else get_project_root() / pattern
        # Find the deepest directory prefix with no glob characters to use as base.
        raw = str(resolved_pattern)
        split = min((raw.find(c) for c in _GLOB_CHARS if c in raw), default=len(raw))
        base = Path(raw[:split]).parent
        rel = raw[len(str(base)) + 1 :]
        matches = sorted(base.glob(rel), key=_natural_path_key)
        if not matches:
            raise FileNotFoundError(f"No input files matched glob pattern: {raw_path}")
        paths.extend(matches)
    return paths


def _natural_path_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


if __name__ == "__main__":
    main()
