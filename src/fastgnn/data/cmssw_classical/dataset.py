"""Classical CMSSW trigger-cell ROOT -> canonical inference parquet conversion."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import time
from typing import Any

import awkward as ak
import numpy as np
from tqdm import tqdm

from fastgnn.data.base import compute_normalization, make_splits, record_field_names, write_yaml
from fastgnn.data.cmssw.hit_features import DEFAULT_LOG_FLOOR, HitFeatures
from fastgnn.data.cmssw_classical.hit_features import (
    CANONICAL_ALIASES,
    CURRENT_CMSSW_HIT_FEATURES,
    alias_dtype,
    build_current_cmssw_features,
    hit_dtype,
)
from fastgnn.data.cmssw_classical.preprocessing import (
    TC_FIELDS,
    TREE_PATH,
    event_to_numpy,
    filter_trigger_cells,
    load_cmssw_classical_arrays,
    nonempty_events,
    require_zside,
    validation_row,
    write_validation,
)

UNLABELED_OBJECT_ID = -10
EMPTY_OBJECTS_SENTINEL_FIELD = "_empty"


def iter_cmssw_classical_events(
    file_paths: list[str | Path],
    cfg: dict[str, Any] | None = None,
    max_events: int | None = None,
) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Yield raw and z-side-filtered trigger-cell event views."""
    cfg = dict(cfg or {})
    zside = require_zside(cfg)
    tree_path = str(cfg.get("tree_path", TREE_PATH))
    events_loaded = 0

    for path in [Path(path) for path in file_paths]:
        raw_arrays = load_cmssw_classical_arrays(path, tree_path, max_events=max_events)
        kept_arrays = filter_trigger_cells(raw_arrays, zside)
        for index in range(len(kept_arrays)):
            if max_events is not None and events_loaded >= max_events:
                return
            if len(kept_arrays["tc_x"][index]) == 0:
                continue
            yield event_to_numpy(raw_arrays, index), event_to_numpy(kept_arrays, index)
            events_loaded += 1


def convert_cmssw_classical_root(
    input_files: list[str | Path],
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    max_events: int | None = None,
    overwrite: bool = False,
    num_workers: int = 1,
) -> Path:
    """Convert classical CMSSW trigger-cell ROOT files into canonical parquet."""
    del num_workers  # Kept for API symmetry with the training CMSSW converter.
    started = time.time()
    cfg = dict(config or {})
    zside = require_zside(cfg)
    tree_path = str(cfg.get("tree_path", TREE_PATH))
    store_derived = bool(cfg.get("store_derived_features", True))
    hit_features = _metadata_hit_features(cfg, store_derived)
    if max_events is None:
        max_events = cfg.get("max_events")

    output_dir = Path(output_dir)
    events_path = output_dir / "events.parquet"
    if events_path.exists() and not overwrite:
        raise FileExistsError(f"{events_path} already exists. Pass overwrite=True to replace it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for input_file in tqdm([Path(path) for path in input_files]):
        raw_arrays = load_cmssw_classical_arrays(input_file, tree_path)
        kept_arrays = filter_trigger_cells(raw_arrays, zside)
        validation_rows.append(
            validation_row(input_file, raw_arrays, kept_arrays, zside=zside)
        )

        for kept_event in _iter_kept_events(kept_arrays):
            if max_events is not None and len(records) >= max_events:
                break
            records.append(
                _event_to_record(
                    kept_event,
                    event_id=len(records),
                    cfg=cfg,
                    hit_features=hit_features,
                    store_derived_features=store_derived,
                )
            )
        if max_events is not None and len(records) >= max_events:
            break

    write_validation(output_dir, validation_rows, elapsed_seconds=time.time() - started)
    if not records:
        raise RuntimeError("No CMSSW classical trigger-cell events passed the zside filter.")

    ak.to_parquet(ak.Array(records), events_path, compression="snappy")

    splits = _test_only_splits(len(records), cfg)
    normalization = compute_normalization(records, hit_features, splits["train"])
    write_yaml(
        output_dir / "metadata.yaml",
        _metadata(records, input_files, tree_path, hit_features, cfg, zside, store_derived, splits),
    )
    write_yaml(output_dir / "normalization.yaml", normalization)
    np.savez_compressed(
        output_dir / "splits.npz",
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )
    return output_dir


def _iter_kept_events(kept_arrays: ak.Array) -> Iterator[dict[str, np.ndarray]]:
    kept_arrays = nonempty_events(kept_arrays)
    for index in range(len(kept_arrays)):
        yield event_to_numpy(kept_arrays, index)


def _event_to_record(
    kept_event: Mapping[str, np.ndarray],
    *,
    event_id: int,
    cfg: Mapping[str, Any],
    hit_features: list[str],
    store_derived_features: bool,
) -> dict[str, Any]:
    hits = _build_hits(kept_event, cfg, hit_features, store_derived_features)
    n_hits = len(hits["tc_x"])
    return {
        "event_id": int(event_id),
        "hits": hits,
        "truth": {
            "hit_object_id": np.full(n_hits, UNLABELED_OBJECT_ID, dtype=np.int32),
            "objects": {EMPTY_OBJECTS_SENTINEL_FIELD: np.asarray([], dtype=np.float32)},
        },
        "metadata": {
            "source": "cmssw_classical",
            "zside": int(cfg["zside"]),
            "inference_only": True,
        },
    }


def _build_hits(
    kept_event: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    hit_features: list[str],
    store_derived_features: bool,
) -> dict[str, np.ndarray]:
    hits = {
        field: np.asarray(kept_event[field]).astype(hit_dtype(field), copy=False)
        for field in TC_FIELDS
    }
    for alias, field in CANONICAL_ALIASES.items():
        hits[alias] = np.asarray(kept_event[field]).astype(alias_dtype(alias), copy=False)

    if not store_derived_features:
        return hits

    derived_feature_names = [name for name in hit_features if name != "pt"]
    derived = build_current_cmssw_features(
        {name: hits[name] for name in ("x", "y", "z", "energy", "layer")},
        feature_names=derived_feature_names,
        log_floor=float(cfg.get("log_floor", DEFAULT_LOG_FLOOR)),
    )
    hits.update(
        {
            name: np.asarray(value).astype(alias_dtype(name), copy=False)
            for name, value in derived.items()
        }
    )
    # Prefer ntuple eta/phi/pt aliases where available instead of recomputing eta/phi.
    hits["eta"] = np.asarray(kept_event["tc_eta"], dtype=np.float32)
    hits["phi"] = np.asarray(kept_event["tc_phi"], dtype=np.float32)
    hits["pt"] = np.asarray(kept_event["tc_pt"], dtype=np.float32)
    return hits


def _metadata_hit_features(cfg: Mapping[str, Any], store_derived_features: bool) -> list[str]:
    requested = list(cfg.get("hit_features") or CURRENT_CMSSW_HIT_FEATURES)
    if store_derived_features:
        unknown = sorted(set(requested) - set(HitFeatures.available_features()) - {"pt"})
        if unknown:
            raise ValueError(
                f"Unknown CMSSW classical hit feature(s): {', '.join(unknown)}. "
                f"Available: {', '.join((*HitFeatures.available_features(), 'pt'))}"
            )
        return requested

    missing = sorted(set(requested) - set(CANONICAL_ALIASES))
    if missing:
        raise ValueError(
            "store_derived_features=false can only use canonical alias features; "
            f"unsupported requested feature(s): {', '.join(missing)}"
        )
    return requested


def _metadata(
    records: list[dict[str, Any]],
    input_files: list[str | Path],
    tree_path: str,
    hit_features: list[str],
    cfg: dict[str, Any],
    zside: int,
    store_derived: bool,
    splits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "fastgnn-canonical-ragged-parquet",
        "format_version": 4,
        "source": "cmssw_classical",
        "task": "inference_only",
        "n_events": len(records),
        "source_files": [str(Path(path)) for path in input_files],
        "tree_path": tree_path,
        "hit_features": hit_features,
        "log_floor": float(cfg.get("log_floor", DEFAULT_LOG_FLOOR)),
        "required_fields": {
            "hits": record_field_names(records[0]["hits"]),
            "truth": ["hit_object_id", "objects"],
            "truth.objects": [],
        },
        "labels": {
            "unlabeled_object_id": UNLABELED_OBJECT_ID,
            "noise_object_id": 0,
            "padding_object_id": -1,
            "semantics": "real trigger cells are unlabeled; no truth objects are available",
        },
        "padding": {
            "hit_object_id": -1,
            "features": 0.0,
            "mask": False,
        },
        "preprocessing": {
            "zside": zside,
            "store_derived_features": store_derived,
            "train_frac": 0.0,
            "val_frac": 0.0,
        },
        "splits": splits["metadata"],
        "config": cfg,
    }


def _test_only_splits(n_events: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    return make_splits(n_events, {**dict(cfg), "train_frac": 0.0, "val_frac": 0.0})
