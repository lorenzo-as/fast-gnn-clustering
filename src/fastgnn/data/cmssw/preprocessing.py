"""
CMSSW HGCAL L1 trigger preprocessing.

This module translates the CMSSW NanoAOD branch layout into the canonical event
schema used by ``fastgnn.data``. It is source-specific: the inputs
are L1 RecHits from ``RecHitsL1THGCALTruthL1THGCAL`` and truth objects from
``MergedSimClusterL1THGCAL``.

For object condensation training, every selected RecHit gets one canonical label:

    ``hit_object_id == 0``
        real detector hit treated as OC background/noise;
    ``hit_object_id > 0``
        1-based index into ``truth.objects`` for a kept SimCluster;
    ``hit_object_id == -1``
        padding only, added later by the batch collator, never stored here.

CMSSW provides hit-to-SimCluster associations through ``cluster0`` and ``frac0``.
``cluster0`` is the original 0-based SimCluster index with the largest
association fraction for that hit; ``frac0`` is stored for diagnostics but is not
used when assigning the OC label. SimClusters can be removed from the training
truth by z-side or energy thresholds. Hits whose best SimCluster is removed are
kept as real hits and relabelled to 0, so the model sees them as background for
this configured task.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from fastgnn.data.object_properties import compute_object_properties_from_links

# Branch name prefixes
HIT_PREFIX = "RecHitsL1THGCALTruthL1THGCAL"
CLUSTER_PREFIX = "MergedSimClusterL1THGCAL"

# Available hit features from NanoAOD
HIT_X = f"{HIT_PREFIX}_x"
HIT_Y = f"{HIT_PREFIX}_y"
HIT_Z = f"{HIT_PREFIX}_z"
HIT_ENERGY = f"{HIT_PREFIX}_energy"
HIT_LAYER = f"{HIT_PREFIX}_layer"
HIT_ZSIDE = f"{HIT_PREFIX}_zside"
HIT_TIME = f"{HIT_PREFIX}_time"

# Truth association.
HIT_NCLUSTERS = f"{HIT_PREFIX}_nClusters"
HIT_CLUSTER0 = f"{HIT_PREFIX}_cluster0"
HIT_FRAC0 = f"{HIT_PREFIX}_frac0"
HIT_CLUSTERS = tuple(f"{HIT_PREFIX}_cluster{i}" for i in range(4))
HIT_FRACS = tuple(f"{HIT_PREFIX}_frac{i}" for i in range(4))

OBJECT_FIELDS = {
    "impact_eta": f"{CLUSTER_PREFIX}_impact_eta",
    "impact_phi": f"{CLUSTER_PREFIX}_impact_phi",
    "impact_energy": f"{CLUSTER_PREFIX}_impact_energy",
    "impact_pt": f"{CLUSTER_PREFIX}_impact_pt",
    "sim_energy": f"{CLUSTER_PREFIX}_simEnergy",
    "track_pdg_id": f"{CLUSTER_PREFIX}_track_pdgId",
}

# Raw properties are computed before vertex preprocessing. Processed properties
# are available only when the preprocessing mode preserves meaningful hit energy.
PROCESSED_OBJECT_PROPERTY_MODES = {"rechits_energy_threshold"}
OBJECT_AGGREGATION_MODES = {
    "hard_assigned_full": ((0,), False),
    "hard_assigned_fractional": ((0,), True),
    "all_fractional": (tuple(range(4)), True),
}


def build_truth(
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
    *,
    processed_event: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Build per-hit OC labels and object-level truth arrays from SimClusters.

    Truth objects are filtered first according to the configured task, then hits
    are assigned to the surviving objects. Hits assigned to filtered-away truth
    objects are mapped to hit_object_id=0, so they remain real detector hits but
    become OC background/noise for this task.
    By default, truth objects are filtered to the configured endcap using
    impact_eta sign so ``truth.objects`` matches the z-side-selected hits.
    Kept truth objects are not required to be visible in the selected hits; the
    derived object fields record selected-hit multiplicity and energy.

    Returns:
        dict with:
            "hit_object_id": (N_hits,) int32, 0 = noise, >0 = 1-based truth object
            "objects":      dict of object-level arrays indexed by hit_object_id - 1,
                            plus raw and supported processed object properties
    """
    processed_event = raw_event if processed_event is None else processed_event
    objects = _extract_objects(raw_event)
    n_objects = _object_count(objects)
    raw_properties = _cmssw_object_properties(raw_event, n_objects, cfg)
    objects.update({f"{name}_raw": values for name, values in raw_properties.items()})
    if processed_object_properties_supported(cfg):
        objects.update(_cmssw_object_properties(processed_event, n_objects, cfg))

    cluster0 = processed_event[HIT_CLUSTER0].astype(np.int32)
    valid = _valid_cluster0(processed_event, n_objects)
    object_keep = _filter_truth_objects(objects, cfg)

    # Assign OC labels by indexing the lookup table with each hit's best cluster.
    # Unassociated hits and hits whose best cluster was filtered away stay at 0,
    # ie. are assigned to the OC noise/background.
    hit_object_id = np.zeros(len(cluster0), dtype=np.int32)
    hit_object_id[valid] = cluster0[valid] + 1
    hit_object_id, _ = compact_hit_object_ids(hit_object_id, object_keep)

    filtered_objects = {
        name: values[object_keep].astype(_object_dtype(name)) for name, values in objects.items()
    }
    return {
        "hit_object_id": hit_object_id,
        "objects": filtered_objects,
    }


def compact_hit_object_ids(
    hit_object_id: np.ndarray,
    object_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter one-based OC labels and compact the IDs of retained truth objects."""
    object_keep = np.asarray(object_keep, dtype=bool)
    kept_indices = np.flatnonzero(object_keep)
    old_to_new = np.zeros(len(object_keep) + 1, dtype=np.int32)
    old_to_new[kept_indices + 1] = np.arange(1, len(kept_indices) + 1, dtype=np.int32)
    valid = (hit_object_id >= 0) & (hit_object_id < len(old_to_new))
    compact = np.zeros_like(hit_object_id, dtype=np.int32)
    compact[valid] = old_to_new[hit_object_id[valid]]
    return compact, kept_indices


def _extract_objects(raw_event: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Read CMSSW SimCluster branches into canonical object-field names."""
    return {
        name: raw_event[branch].astype(_object_dtype(name))
        for name, branch in OBJECT_FIELDS.items()
    }


def _object_count(objects: dict[str, np.ndarray]) -> int:
    """Return the number of SimClusters in object arrays."""
    if not objects:
        return 0
    first = next(iter(objects.values()))
    return len(first)


def _filter_truth_objects(objects: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    """
    Select SimClusters that should count as truth objects for this task.

    Filters can use z-side via impact_eta sign and an object energy threshold.
    """
    n_objects = _object_count(objects)
    keep = np.ones(n_objects, dtype=bool)

    if cfg.get("filter_truth_by_zside", True):
        zside = cfg.get("zside")
        if zside is not None:
            keep &= _truth_zside_mask(objects, int(zside))

    min_energy = cfg.get("truth_min_object_energy")
    if min_energy is not None:
        energy_field = str(cfg.get("truth_object_energy_field", "impact_energy"))
        if energy_field not in objects:
            available = ", ".join(objects)
            raise KeyError(
                f"Unknown truth_object_energy_field={energy_field!r}. "
                f"Available object fields: {available}"
            )
        keep &= objects[energy_field] >= float(min_energy)

    min_sum_energy = cfg.get("truth_min_sum_energy")
    if min_sum_energy is not None:
        if "sum_energy" not in objects:
            mode = _preprocessing_mode(cfg)
            raise ValueError(
                f"truth_min_sum_energy requires processed object properties, but "
                f"preprocessing mode {mode!r} does not provide them"
            )
        keep &= objects["sum_energy"] >= float(min_sum_energy)
    return keep


def _truth_zside_mask(objects: dict[str, np.ndarray], zside: int) -> np.ndarray:
    """Match SimCluster impact eta sign to the selected HGCAL endcap."""
    if zside not in {-1, 1}:
        raise ValueError(f"zside must be -1 or 1, got {zside}")
    if "impact_eta" not in objects:
        raise KeyError("Cannot filter truth objects by zside without impact_eta")
    return np.sign(objects["impact_eta"]) == zside


def cmssw_object_links(
    raw_event: dict[str, np.ndarray],
    n_objects: int,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build configured RecHit-to-SimCluster aggregation links."""
    slots, fractional = object_aggregation_spec(cfg)
    return _links_for_slots(raw_event, n_objects, slots=slots, fractional=fractional)


def object_aggregation_spec(cfg: dict[str, Any]) -> tuple[tuple[int, ...], bool]:
    """Return association slots and fractional-weight policy for one CMSSW mode."""
    mode = str(cfg.get("object_aggregation_mode", "hard_assigned_full"))
    try:
        return OBJECT_AGGREGATION_MODES[mode]
    except KeyError as exc:
        available = ", ".join(OBJECT_AGGREGATION_MODES)
        raise ValueError(
            f"Unknown object_aggregation_mode {mode!r}. Available: {available}"
        ) from exc


def _cmssw_object_properties(
    raw_event: dict[str, np.ndarray],
    n_objects: int,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    hit_indices, linked_object_ids, weights = cmssw_object_links(raw_event, n_objects, cfg)
    return compute_object_properties_from_links(
        {
            "x": raw_event[HIT_X],
            "y": raw_event[HIT_Y],
            "z": raw_event[HIT_Z],
            "energy": raw_event[HIT_ENERGY],
            "layer": raw_event[HIT_LAYER],
        },
        hit_indices,
        linked_object_ids,
        weights,
        object_ids=np.arange(n_objects, dtype=np.int32),
    )


def _links_for_slots(
    raw_event: dict[str, np.ndarray],
    n_objects: int,
    *,
    slots: Iterable[int],
    fractional: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit_indices = []
    linked_object_ids = []
    weights = []
    nclusters = raw_event[HIT_NCLUSTERS]
    for slot in slots:
        clusters = raw_event[HIT_CLUSTERS[slot]].astype(np.int32)
        valid = (nclusters > slot) & (clusters >= 0) & (clusters < n_objects)
        selected_hits = np.flatnonzero(valid)
        hit_indices.append(selected_hits)
        linked_object_ids.append(clusters[valid])
        weights.append(
            np.abs(raw_event[HIT_FRACS[slot]][valid]).astype(np.float32)
            if fractional
            else np.ones(len(selected_hits), dtype=np.float32)
        )
    return tuple(
        np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
        for parts, dtype in (
            (hit_indices, np.int64),
            (linked_object_ids, np.int32),
            (weights, np.float32),
        )
    )


def _valid_cluster0(raw_event: dict[str, np.ndarray], n_objects: int) -> np.ndarray:
    cluster0 = raw_event[HIT_CLUSTER0]
    return (raw_event[HIT_NCLUSTERS] > 0) & (cluster0 >= 0) & (cluster0 < n_objects)


def _preprocessing_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("preprocessing", "rechits_energy_threshold"))


def processed_object_properties_supported(cfg: dict[str, Any]) -> bool:
    return _preprocessing_mode(cfg) in PROCESSED_OBJECT_PROPERTY_MODES


def _object_dtype(name: str) -> Any:
    if name == "track_pdg_id" or name.startswith(("n_hits", "core_shower_length")):
        return np.int32
    return np.float32
