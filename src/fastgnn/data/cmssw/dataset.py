"""
CMSSW ROOT -> canonical ragged Parquet conversion.
Once converted, data is loaded through fastgnn.data.CaloDataset like every other source.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time
from typing import Any

import awkward as ak
import numpy as np
from tqdm import tqdm
import uproot

from fastgnn.data.base import compute_normalization, make_splits, record_field_names, write_yaml
from fastgnn.data.cmssw.hit_features import (
    DEFAULT_LOG_FLOOR,
    HitFeatures,
)
from fastgnn.data.cmssw.preprocessing import (
    HIT_BASE_PREFIX,
    CmsswBranches,
    build_truth,
    cmssw_branches_from_cfg,
    object_aggregation_spec,
    processed_object_properties_supported,
)
from fastgnn.data.cmssw.transforms import preprocess_vertices

logger = logging.getLogger(__name__)

_PERSISTED_HIT_FEATURES = ("x", "y", "z", "energy", "layer")


def _branches_to_load(cfg: dict[str, Any], branches: CmsswBranches) -> list[str]:
    expressions = [
        branches.x,
        branches.y,
        branches.z,
        branches.energy,
        branches.layer,
        branches.zside,
        branches.time,
        branches.nclusters,
        branches.cluster0,
        branches.frac0,
        *branches.object_fields.values(),
    ]
    slots, _ = object_aggregation_spec(cfg)
    if len(slots) > 1:
        expressions.extend([*branches.clusters[1:], *branches.fracs[1:]])
    return expressions


def _available_chains(file_path: str | Path) -> list[str]:
    """Discover the merging-chain suffixes present in a CMSSW ROOT file."""
    prefix = HIT_BASE_PREFIX
    suffix = "_x"
    with uproot.open(f"{file_path}:Events") as tree:
        keys = tree.keys(filter_name=f"{prefix}*{suffix}")
    chains = sorted(key[len(prefix) : -len(suffix)] for key in keys if key.endswith(suffix))
    return chains


def _validate_chain(file_paths: list[Path], branches: CmsswBranches) -> None:
    """Fail fast if the requested merging chain is absent from the first ROOT file."""
    if not file_paths:
        return
    first = file_paths[0]
    with uproot.open(f"{first}:Events") as tree:
        present = branches.x in tree
    if present:
        return
    available = _available_chains(first)
    pretty = ", ".join("'' (baseline)" if chain == "" else chain for chain in available)
    requested = branches.merging or "'' (baseline)"
    raise ValueError(
        f"merging={requested} not found in {first} (missing branch {branches.x!r}). "
        f"Available chains: {pretty or 'none found'}"
    )


def iter_cmssw_events(
    file_paths: list[str | Path],
    cfg: dict[str, Any] | None = None,
    max_events: int | None = None,
) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Yield raw and preprocessed CMSSW event views after endcap filtering."""
    cfg = cfg or {}
    paths = [Path(path) for path in file_paths]
    branches = cmssw_branches_from_cfg(cfg)
    _validate_chain(paths, branches)
    zside_req = cfg.get("zside", 1)
    events_loaded = 0

    iterator = uproot.iterate(
        [f"{path}:Events" for path in paths],
        expressions=_branches_to_load(cfg, branches),
        library="ak",
        step_size=cfg.get("step_size") or max_events or "100 MB",
    )

    for chunk in iterator:
        if max_events is not None and events_loaded >= max_events:
            break

        zside_mask = chunk[branches.zside] == zside_req
        for field in chunk.fields:
            if field.startswith(branches.hit_prefix):
                chunk[field] = chunk[field][zside_mask]

        valid_events_mask = ak.num(chunk[branches.x]) > 0
        chunk = chunk[valid_events_mask]

        for i in range(len(chunk)):
            if max_events is not None and events_loaded >= max_events:
                break
            raw_event = {field: ak.to_numpy(chunk[field][i]) for field in chunk.fields}
            processed_event = preprocess_vertices(raw_event, cfg)
            if len(processed_event[branches.x]) == 0:
                continue
            yield raw_event, processed_event
            events_loaded += 1


def cmssw_event_to_record(
    raw_event: dict[str, np.ndarray],
    event_id: int,
    cfg: dict[str, Any] | None = None,
    *,
    processed_event: dict[str, np.ndarray] | None = None,
    hit_features: list[str] | None = None,
) -> ak.Record:
    """Build one canonical event record from one raw CMSSW event."""
    cfg = cfg or {}
    branches = cmssw_branches_from_cfg(cfg)
    processed_event = preprocess_vertices(raw_event, cfg) if processed_event is None else processed_event #iterate_cmssw_events should have already preprocessed, keep fallback to make this function standalone
    hit_features = (
        list(cfg.get("hit_features", HitFeatures.available_features()))
        if hit_features is None
        else hit_features
    )
    truth = build_truth(raw_event, cfg, processed_event=processed_event)

    return ak.Record(
        {
            "event_id": int(event_id),
            "hits": _build_hits(processed_event, cfg, hit_features, branches),
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
    num_workers: int = 1,
) -> Path:
    """
    Convert CMSSW ROOT files into canonical ragged Parquet plus sidecar metadata.

    Returns the output directory path.
    """
    _t = time.time()
    cfg = dict(config or {})
    hit_features = list(cfg.get("hit_features", HitFeatures.available_features()))
    stored_hit_features = _persisted_hit_features(hit_features)
    if max_events is None:
        max_events = cfg.get("max_events")

    output_dir = Path(output_dir)
    events_path = output_dir / "events.parquet"
    if events_path.exists() and not overwrite:
        raise FileExistsError(f"{events_path} already exists. Pass overwrite=True to replace it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = _convert_records(input_files, cfg, hit_features, max_events, num_workers)
    logger.info("Finished converting %d events in %.2f seconds.", len(records), time.time() - _t)
    _t = time.time()
    if not records:
        raise RuntimeError("No CMSSW events passed filtering. Check input files and zside.")

    events = ak.Array(records)
    logger.info("Finished creating AkArray in %.2f seconds.", time.time() - _t)
    _t = time.time()
    ak.to_parquet(events, events_path, compression="snappy")
    logger.info("Finished writing Parquet file in %.2f seconds.", time.time() - _t)
    _t = time.time()

    splits = make_splits(len(records), cfg)
    normalization = compute_normalization(records, stored_hit_features, splits["train"])

    metadata = {
        "format": "fastgnn-canonical-ragged-parquet",
        "format_version": 4,
        "source": "cmssw",
        "n_events": len(records),
        "source_files": [str(Path(path)) for path in input_files],
        "hit_features": stored_hit_features,
        "log_floor": float(cfg.get("log_floor", DEFAULT_LOG_FLOOR)),
        "required_fields": {
            "hits": record_field_names(records[0]["hits"]),
            "truth": ["hit_object_id", "objects"],
            "truth.objects": record_field_names(records[0]["truth"]["objects"]),
        },
        "padding": {
            "hit_object_id": -1,
            "features": 0.0,
            "mask": False,
        },
        "preprocessing": {
            "mode": cfg.get("preprocessing", "rechits_energy_threshold"),
            "merging": cfg.get("merging") or "",
            "hit_min_energy": cfg.get("hit_min_energy"),
            "truth_min_object_energy": cfg.get("truth_min_object_energy"),
            "truth_min_sum_energy": cfg.get("truth_min_sum_energy"),
            "truth_object_energy_field": cfg.get("truth_object_energy_field", "impact_energy"),
            "object_aggregation_mode": cfg.get("object_aggregation_mode", "hard_assigned_full"),
            "processed_object_properties": processed_object_properties_supported(cfg),
            "filter_truth_by_zside": cfg.get("filter_truth_by_zside", True),
            "thresholded_truth_objects": "hits become hit_object_id=0 OC background/noise",
        },
        "splits": splits["metadata"],
        "config": cfg,
    }
    write_yaml(output_dir / "metadata.yaml", metadata)
    write_yaml(output_dir / "normalization.yaml", normalization)
    np.savez_compressed(
        output_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )
    return output_dir


def _convert_records(
    input_files: list[str | Path],
    cfg: dict[str, Any],
    hit_features: list[str],
    max_events: int | None,
    num_workers: int,
) -> list[Any]:
    num_workers = max(1, int(num_workers or 1))
    if num_workers == 1 or len(input_files) <= 1:
        return [
            cmssw_event_to_record(
                raw_event,
                event_id=i,
                cfg=cfg,
                processed_event=processed_event,
                hit_features=hit_features,
            )
            for i, (raw_event, processed_event) in tqdm(
                enumerate(iter_cmssw_events(input_files, cfg, max_events))
            )
        ]

    records: list[dict[str, Any]] = []
    worker_args = [(str(path), cfg, hit_features, max_events) for path in input_files]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_records in tqdm(
            executor.map(_convert_file_records, worker_args),
            total=len(worker_args),
        ):
            records.extend(file_records)
            if max_events is not None and len(records) >= max_events:
                records = records[:max_events]
                break

    for event_id, record in enumerate(records):
        record["event_id"] = event_id
    return records


def _convert_file_records(
    args: tuple[str, dict[str, Any], list[str], int | None],
) -> list[dict[str, Any]]:
    path, cfg, hit_features, max_events = args
    return [
        ak.to_list(
            cmssw_event_to_record(
                raw_event,
                event_id=i,
                cfg=cfg,
                processed_event=processed_event,
                hit_features=hit_features,
            )
        )
        for i, (raw_event, processed_event) in enumerate(
            iter_cmssw_events([path], cfg, max_events)
        )
    ]


def _build_hits(
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
    hit_features: list[str],
    branches: CmsswBranches,
) -> dict[str, np.ndarray]:
    hits = HitFeatures(
        {
            "x": raw_event[branches.x],
            "y": raw_event[branches.y],
            "z": raw_event[branches.z],
            "energy": raw_event[branches.energy],
            "layer": raw_event[branches.layer],
        },
        log_floor=float(cfg.get("log_floor", DEFAULT_LOG_FLOOR)),
    ).build(_persisted_hit_features(hit_features))
    hits.update(
        {
            "time": raw_event[branches.time].astype(np.float32),
            "zside": raw_event[branches.zside].astype(np.int8),
            "n_clusters": raw_event[branches.nclusters].astype(np.int16),
            "cluster0": raw_event[branches.cluster0].astype(np.int32),
            "frac0": raw_event[branches.frac0].astype(np.float32),
        }
    )
    slots, _ = object_aggregation_spec(cfg)
    if len(slots) > 1:
        for slot in slots[1:]:
            hits[f"cluster{slot}"] = raw_event[branches.clusters[slot]].astype(np.int32)
            hits[f"frac{slot}"] = raw_event[branches.fracs[slot]].astype(np.float32)
    return hits


def _persisted_hit_features(hit_features: list[str]) -> list[str]:
    """Retain physical hit fields needed by diagnostics and object properties."""
    persisted = {*_PERSISTED_HIT_FEATURES, *hit_features}
    return [name for name in HitFeatures.available_features() if name in persisted]
