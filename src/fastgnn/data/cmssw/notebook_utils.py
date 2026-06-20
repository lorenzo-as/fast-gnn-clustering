"""Shared notebook helpers for CMSSW/ROOT inspection."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import polars as pl

from fastgnn.utils import get_project_root


def pdgid_to_name(pdgid: int) -> str:
    try:
        from particle import Particle

        return str(Particle.from_pdgid(int(pdgid)).name)
    except Exception:
        return str(int(pdgid))


def compact_path(path: str | Path) -> str:
    path = str(path)
    project_root = get_project_root()
    try:
        return str(Path(path).relative_to(project_root))
    except Exception:
        return path


def to_numpy(value) -> np.ndarray:
    try:
        return np.asarray(ak.to_numpy(value))
    except Exception:
        return np.asarray(ak.to_list(value))


def object_collection_frame(
    arrays: ak.Array,
    event_index: int,
    prefix: str,
    field_map: list[tuple[str, str]],
) -> pl.DataFrame:
    columns = {}
    lengths = []
    for source_name, target_name in field_map:
        branch_name = f"{prefix}_{source_name}"
        if branch_name not in arrays.fields:
            continue
        values = to_numpy(arrays[branch_name][event_index])
        columns[target_name] = values
        lengths.append(len(values))
    if not columns:
        return pl.DataFrame()
    n_rows = max(lengths, default=0)
    if n_rows == 0:
        return pl.DataFrame({name: [] for name in columns})
    normalized = {}
    for name, values in columns.items():
        if len(values) == n_rows:
            normalized[name] = values
            continue
        padded = np.full(n_rows, None, dtype=object)
        padded[: len(values)] = values.tolist()
        normalized[name] = padded.tolist()
    return pl.DataFrame(normalized)


def normalize_numeric_frame(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for column, dtype in zip(frame.columns, frame.dtypes, strict=True):
        if dtype == pl.Float64:
            out = out.with_columns(pl.col(column).round(4))
    return out


def collection_summary_row(arrays: ak.Array, prefix: str) -> dict[str, float | int]:
    base = f"{prefix}_pt"
    if base not in arrays.fields:
        return {
            "collection": prefix,
            "events_with_entries": 0,
            "total_entries": 0,
            "min_entries": 0,
            "median_entries": 0.0,
            "max_entries": 0,
        }
    counts = ak.to_numpy(ak.num(arrays[base], axis=1))
    return {
        "collection": prefix,
        "events_with_entries": int(np.sum(counts > 0)),
        "total_entries": int(np.sum(counts)),
        "min_entries": int(np.min(counts)) if len(counts) else 0,
        "median_entries": float(np.median(counts)) if len(counts) else 0.0,
        "max_entries": int(np.max(counts)) if len(counts) else 0,
    }
