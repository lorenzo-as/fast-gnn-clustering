"""Tiny Polars index over processed dataset metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml


class DatasetRegistry:
    """Scan ``metadata.yaml`` sidecars into a queryable in-memory table."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._table: pl.DataFrame | None = None

    def table(self, *, refresh: bool = False) -> pl.DataFrame:
        if self._table is None or refresh:
            rows = [
                _row(path.parent, self.root)
                for path in sorted(self.root.glob("**/metadata.yaml"))
                if (path.parent / "events.parquet").exists()
            ]
            self._table = pl.DataFrame(rows) if rows else pl.DataFrame()
        return self._table

    def filter(self, **metadata: Any) -> pl.DataFrame:
        table = self.table()
        for column, value in metadata.items():
            table = table.filter(pl.col(column) == value)
        return table

    def paths(self, **metadata: Any) -> list[Path]:
        return [self.root / path for path in self.filter(**metadata)["relative_path"].to_list()]


def _row(dataset_dir: Path, root: Path) -> dict[str, Any]:
    metadata = _read_yaml(dataset_dir / "metadata.yaml")
    splits = metadata.get("splits") or {}
    return {
        "relative_path": str(dataset_dir.relative_to(root)),
        "n_events": metadata.get("n_events"),
        "feature_names": metadata.get("feature_names"),
        "truth_objects": (metadata.get("required_fields") or {}).get("truth.objects"),
        "train_frac": splits.get("train_frac"),
        "val_frac": splits.get("val_frac"),
        **_flatten(metadata.get("padding") or {}, "padding"),
        **_flatten(metadata.get("preprocessing") or {}, "preprocessing"),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data
