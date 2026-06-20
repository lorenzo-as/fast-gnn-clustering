"""Awkward preprocessing and validation for classical CMSSW trigger-cell ntuples."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

TREE_PATH = "l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple"
TC_FIELDS = (
    "tc_id",
    "tc_subdet",
    "tc_zside",
    "tc_layer",
    "tc_waferu",
    "tc_waferv",
    "tc_wafertype",
    "tc_cellu",
    "tc_cellv",
    "tc_pt",
    "tc_mipPt",
    "tc_energy",
    "tc_eta",
    "tc_phi",
    "tc_x",
    "tc_y",
    "tc_z",
)
OPTIONAL_FIELDS = ("tc_n", "event")
VALIDATION_FIELDS = ("tc_x", "tc_y", "tc_z", "tc_eta", "tc_phi", "tc_energy", "tc_pt", "tc_layer")


def require_zside(cfg: dict | None) -> int:
    cfg = cfg or {}
    if "zside" not in cfg:
        raise ValueError("CMSSW classical conversion requires explicit zside=1 or zside=-1")
    zside = cfg["zside"]
    if isinstance(zside, str):
        if zside.lower() == "auto":
            raise ValueError("CMSSW classical conversion does not support zside='auto'")
        zside = int(zside)
    zside = int(zside)
    if zside not in {-1, 1}:
        raise ValueError(f"CMSSW classical conversion requires zside=1 or zside=-1, got {zside}")
    return zside


def load_cmssw_classical_arrays(
    file_path: str | Path,
    tree_path: str,
    *,
    max_events: int | None = None,
) -> ak.Array:
    """Load required trigger-cell branches into one awkward array."""
    fields = validate_branches(file_path, tree_path)
    with uproot.open(file_path) as root_file:
        tree = root_file[tree_path]
        return tree.arrays(fields, library="ak", entry_stop=max_events)


def validate_branches(file_path: str | Path, tree_path: str) -> list[str]:
    """Return available required+optional fields, failing if a required branch is absent."""
    with uproot.open(f"{file_path}:{tree_path}") as tree:
        keys = set(tree.keys())
        missing = [field for field in TC_FIELDS if field not in tree]
    if missing:
        raise KeyError(
            f"{file_path}:{tree_path} is missing trigger-cell branch(es): "
            f"{', '.join(missing)}"
        )
    return [field for field in OPTIONAL_FIELDS if field in keys] + list(TC_FIELDS)


def filter_trigger_cells(arrays: ak.Array, zside: int) -> ak.Array:
    """Filter trigger-cell jagged fields to one endcap, preserving event rows."""
    mask = arrays["tc_zside"] == zside
    filtered = {field: arrays[field][mask] for field in TC_FIELDS}
    for field in arrays.fields:
        if field not in filtered:
            filtered[field] = arrays[field]
    return ak.zip(filtered, depth_limit=1)


def nonempty_events(arrays: ak.Array) -> ak.Array:
    """Return only events with at least one kept trigger cell."""
    return arrays[ak.num(arrays["tc_x"]) > 0]


def event_to_numpy(arrays: ak.Array, index: int) -> dict[str, np.ndarray]:
    """Materialize one awkward event row as numpy arrays/scalars."""
    event: dict[str, np.ndarray] = {}
    for field in arrays.fields:
        event[field] = _to_numpy(arrays[field][index])
    return event


def validation_row(
    input_file: str | Path,
    raw_arrays: ak.Array,
    kept_arrays: ak.Array,
    *,
    zside: int,
) -> dict[str, Any]:
    """Summarize one file and z-side using vectorized awkward reductions."""
    raw_zside = raw_arrays["tc_zside"]
    kept_energy = _flat_numpy(kept_arrays["tc_energy"])
    kept_pt = _flat_numpy(kept_arrays["tc_pt"])

    row: dict[str, Any] = {
        "input_file": str(input_file),
        "zside": int(zside),
        "total_events": len(raw_arrays),
        "raw_tc_count": int(ak.sum(ak.num(raw_zside))),
        "kept_tc_count": int(ak.sum(ak.num(kept_arrays["tc_x"]))),
        "tc_zside_pos_count": int(ak.sum(raw_zside == 1)),
        "tc_zside_neg_count": int(ak.sum(raw_zside == -1)),
        "tc_zside_other_count": int(ak.sum((raw_zside != 1) & (raw_zside != -1))),
        "tc_n_mismatch_events": _tc_n_mismatches(raw_arrays),
        "negative_energy_count": int(np.sum(kept_energy < 0)),
        "negative_pt_count": int(np.sum(kept_pt < 0)),
    }

    nonfinite_count = 0
    for field in VALIDATION_FIELDS:
        values = _flat_numpy(kept_arrays[field])
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        nonfinite_count += int(np.sum(~finite_mask))
        row[f"finite_fraction_{field}"] = float(np.mean(finite_mask)) if values.size else 1.0
        row[f"min_{field}"] = float(np.min(finite)) if finite.size else None
        row[f"max_{field}"] = float(np.max(finite)) if finite.size else None
    row["nonfinite_count"] = nonfinite_count
    return row


def write_validation(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> None:
    """Write CSV and human-readable validation summaries."""
    csv_path = output_dir / "validation.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    total_events = sum(int(row["total_events"]) for row in rows)
    raw_tc = sum(int(row["raw_tc_count"]) for row in rows)
    kept_tc = sum(int(row["kept_tc_count"]) for row in rows)
    mismatches = sum(int(row["tc_n_mismatch_events"]) for row in rows)
    negative_energy = sum(int(row["negative_energy_count"]) for row in rows)

    lines = [
        "CMSSW classical trigger-cell validation",
        f"files: {len(rows)}",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"total_events: {total_events}",
        f"raw_trigger_cells: {raw_tc}",
        f"kept_trigger_cells: {kept_tc}",
        f"tc_n_mismatch_events: {mismatches}",
        f"negative_energy_count: {negative_energy}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"file: {row['input_file']}",
                f"  zside: {row['zside']}",
                f"  events: {row['total_events']}",
                f"  raw_tc_count: {row['raw_tc_count']}",
                f"  kept_tc_count: {row['kept_tc_count']}",
                f"  tc_zside counts: +1={row['tc_zside_pos_count']}, -1={row['tc_zside_neg_count']}, other={row['tc_zside_other_count']}",
                f"  finite tc_x/tc_y/tc_z: {row['finite_fraction_tc_x']:.6g}, {row['finite_fraction_tc_y']:.6g}, {row['finite_fraction_tc_z']:.6g}",
                f"  energy range: {row['min_tc_energy']} .. {row['max_tc_energy']}",
                f"  pt range: {row['min_tc_pt']} .. {row['max_tc_pt']}",
            ]
        )
    (output_dir / "validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_numpy(value: Any) -> np.ndarray:
    try:
        return ak.to_numpy(value)
    except Exception:
        return np.asarray(ak.to_list(value))


def _tc_n_mismatches(raw_arrays: ak.Array) -> int:
    if "tc_n" not in raw_arrays.fields:
        return 0
    return int(ak.sum(raw_arrays["tc_n"] != ak.num(raw_arrays["tc_zside"])))


def _flat_numpy(values: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(values, axis=None))
