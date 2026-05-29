"""
CMSSW ROOT -> canonical ragged Parquet conversion.
Once converted, data is loaded through fastgnn.data.CaloDataset like every other source.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path
import time
from typing import Any

import awkward as ak
import numpy as np
from tqdm import tqdm
import uproot
import yaml

from fastgnn.data.base import compute_normalization, make_splits
from fastgnn.data.cmssw.preprocessing import (
    DERIVED_OBJECT_FIELDS,
    HIT_CLUSTER0,
    HIT_ENERGY,
    HIT_FRAC0,
    HIT_LAYER,
    HIT_NCLUSTERS,
    HIT_PREFIX,
    HIT_TIME,
    HIT_X,
    HIT_Y,
    HIT_Z,
    HIT_ZSIDE,
    OBJECT_FIELDS,
    build_truth,
    get_feature_names,
)
from fastgnn.data.cmssw.transforms import preprocess_vertices
from fastgnn.geometry import xyz_to_eta_phi

logger = logging.getLogger(__name__)

_BRANCHES_TO_LOAD = [
    HIT_X,
    HIT_Y,
    HIT_Z,
    HIT_ENERGY,
    HIT_LAYER,
    HIT_ZSIDE,
    HIT_TIME,
    HIT_NCLUSTERS,
    HIT_CLUSTER0,
    HIT_FRAC0,
    *OBJECT_FIELDS.values(),
]


def iter_cmssw_raw_events(
    file_paths: list[str | Path],
    cfg: dict[str, Any] | None = None,
    max_events: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield filtered CMSSW events as flat NumPy field dictionaries."""
    cfg = cfg or {}
    paths = [Path(path) for path in file_paths]
    zside_req = cfg.get("zside", 1)
    events_loaded = 0

    iterator = uproot.iterate(
        [f"{path}:Events" for path in paths],
        expressions=_BRANCHES_TO_LOAD,
        library="ak",
        step_size=cfg.get("step_size") or max_events or "100 MB",
    )

    for chunk in iterator:
        if max_events is not None and events_loaded >= max_events:
            break

        zside_mask = chunk[HIT_ZSIDE] == zside_req
        for field in chunk.fields:
            if field.startswith(HIT_PREFIX):
                chunk[field] = chunk[field][zside_mask]

        valid_events_mask = ak.num(chunk[HIT_X]) > 0
        chunk = chunk[valid_events_mask]

        for i in range(len(chunk)):
            if max_events is not None and events_loaded >= max_events:
                break
            raw_event = {field: ak.to_numpy(chunk[field][i]) for field in chunk.fields}
            processed_event = preprocess_vertices(raw_event, cfg)
            if len(processed_event[HIT_X]) == 0:
                continue
            yield processed_event
            events_loaded += 1


def cmssw_event_to_record(
    raw_event: dict[str, np.ndarray],
    event_id: int,
    cfg: dict[str, Any] | None = None,
) -> ak.Record:
    """Build one canonical event record from one raw CMSSW event."""
    cfg = cfg or {}
    truth = build_truth(raw_event, cfg)

    return ak.Record(
        {
            "event_id": int(event_id),
            "hits": _build_hits(raw_event, cfg),
            "truth": {
                "hit_object_id": truth["hit_object_id"].astype(np.int32),
                "objects": truth["objects"],
            },
            "metadata": {
                "source": "cmssw",
                "zside": int(cfg.get("zside", 1)),
            },
        }
    )


def convert_cmssw_root(
    input_files: list[str | Path],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    max_events: int | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Convert CMSSW ROOT files into canonical ragged Parquet plus sidecar metadata.

    Returns the output directory path.
    """
    convert_timings = []
    convert_timings.append([time.time(), "start"])
    cfg = dict(config or {})
    if max_events is None:
        max_events = cfg.get("max_events")

    output_dir = Path(output_dir)
    events_path = output_dir / "events.parquet"
    if events_path.exists() and not overwrite:
        raise FileExistsError(f"{events_path} already exists. Pass overwrite=True to replace it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    convert_timings.append([time.time(), "start_record_loop"])
    records = [
        cmssw_event_to_record(raw_event, event_id=i, cfg=cfg)
        for i, raw_event in tqdm(enumerate(iter_cmssw_raw_events(input_files, cfg, max_events)))
    ]
    convert_timings.append([time.time(), "end_record_loop"])
    if not records:
        raise RuntimeError("No CMSSW events passed filtering. Check input files and zside.")

    events = ak.Array(records)
    convert_timings.append([time.time(), "end_ak_array"])
    ak.to_parquet(events, events_path, compression="snappy")
    convert_timings.append([time.time(), "end_parquet_write"])

    feature_names = get_feature_names(cfg)
    convert_timings.append([time.time(), "end_feature_names"])
    splits = make_splits(len(records), cfg)
    convert_timings.append([time.time(), "end_make_splits"])
    normalization = compute_normalization(records, feature_names, splits["train"])
    convert_timings.append([time.time(), "end_compute_normalization"])

    metadata = {
        "format": "fastgnn-canonical-ragged-parquet",
        "format_version": 2,
        "source": "cmssw",
        "n_events": len(records),
        "source_files": [str(Path(path)) for path in input_files],
        "feature_names": feature_names,
        "required_fields": {
            "hits": ["x", "y", "z", "energy", "cluster0", "frac0"],
            "truth": ["hit_object_id", "objects"],
            "truth.objects": list(OBJECT_FIELDS.keys()) + list(DERIVED_OBJECT_FIELDS),
        },
        "padding": {
            "hit_object_id": -1,
            "features": 0.0,
            "mask": False,
        },
        "preprocessing": {
            "mode": cfg.get(
                "preprocessing", cfg.get("preprocessing_mode", "rechits_energy_threshold")
            ),
            "hit_min_energy": cfg.get("hit_min_energy", cfg.get("min_hit_energy")),
            "truth_min_object_energy": cfg.get(
                "truth_min_object_energy",
                cfg.get("truth_object_energy_threshold"),
            ),
            "truth_min_visible_energy": cfg.get("truth_min_visible_energy"),
            "truth_object_energy_field": cfg.get("truth_object_energy_field", "impact_energy"),
            "filter_truth_by_zside": cfg.get("filter_truth_by_zside", True),
            "thresholded_truth_objects": "hits become hit_object_id=0 OC background/noise",
            "visible_truth_objects": (
                "truth.objects.n_hits > 0 after hit preprocessing selections"
            ),
        },
        "splits": splits["metadata"],
        "config": cfg,
    }
    _write_yaml(output_dir / "metadata.yaml", metadata)
    convert_timings.append([time.time(), "end_metadata_write"])
    _write_yaml(output_dir / "normalization.yaml", normalization)
    np.savez_compressed(
        output_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )
    convert_timings.append([time.time(), "end_splits_write"])
    logger.info("CMSSW conversion timings:")
    for timing in convert_timings:
        logger.info("  %s: %.2f s", timing[1], timing[0] - convert_timings[0][0])

    return output_dir


def _build_hits(raw_event: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    hits = {
        "x": raw_event[HIT_X].astype(np.float32),
        "y": raw_event[HIT_Y].astype(np.float32),
        "z": raw_event[HIT_Z].astype(np.float32),
        "energy": raw_event[HIT_ENERGY].astype(np.float32),
        "layer": raw_event[HIT_LAYER].astype(np.int16),
        "time": raw_event[HIT_TIME].astype(np.float32),
        "zside": raw_event[HIT_ZSIDE].astype(np.int8),
        "n_clusters": raw_event[HIT_NCLUSTERS].astype(np.int16),
        "cluster0": raw_event[HIT_CLUSTER0].astype(np.int32),
        "frac0": raw_event[HIT_FRAC0].astype(np.float32),
    }

    coord_sys = cfg.get("coordinate_system", "cartesian")
    if coord_sys in {"cylindrical", "both"}:
        eta, phi = xyz_to_eta_phi(hits["x"], hits["y"], hits["z"])
        hits["eta"] = eta.astype(np.float32)
        hits["phi"] = phi.astype(np.float32)
    return hits


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
