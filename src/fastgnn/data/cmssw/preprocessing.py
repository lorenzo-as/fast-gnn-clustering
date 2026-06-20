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
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fastgnn.data.object_properties import compute_object_properties_from_links

# Base branch-name prefixes (baseline L1THGCAL truth-merging chain). A merging
# variant is selected by appending a suffix (e.g. "d0p5f0p4") to both prefixes; see
# ``cmssw_branches``.
HIT_BASE_PREFIX = "RecHitsL1THGCALTruthL1THGCAL"
CLUSTER_BASE_PREFIX = "MergedSimClusterL1THGCAL"


@dataclass(frozen=True)
class CmsswBranches:
    """Resolved NanoAOD branch names for one HGCAL truth-merging sim chain.

    All hit/object branch names derive from a single ``merging`` suffix appended
    identically to both prefixes, so the hit->SimCluster association indices stay
    internally consistent. Use ``cmssw_branches`` to build one.
    """

    merging: str
    hit_prefix: str
    cluster_prefix: str
    x: str
    y: str
    z: str
    energy: str
    layer: str
    zside: str
    time: str
    nclusters: str
    cluster0: str
    frac0: str
    clusters: tuple[str, ...]
    fracs: tuple[str, ...]
    object_fields: dict[str, str] = field(default_factory=dict)


def cmssw_branches(merging: str | None = None) -> CmsswBranches:
    """Build the resolved branch names for a merging chain (``None``/``""`` = baseline)."""
    suffix = "" if merging in (None, "") else str(merging)
    hit_prefix = f"{HIT_BASE_PREFIX}{suffix}"
    cluster_prefix = f"{CLUSTER_BASE_PREFIX}{suffix}"
    return CmsswBranches(
        merging=suffix,
        hit_prefix=hit_prefix,
        cluster_prefix=cluster_prefix,
        x=f"{hit_prefix}_x",
        y=f"{hit_prefix}_y",
        z=f"{hit_prefix}_z",
        energy=f"{hit_prefix}_energy",
        layer=f"{hit_prefix}_layer",
        zside=f"{hit_prefix}_zside",
        time=f"{hit_prefix}_time",
        nclusters=f"{hit_prefix}_nClusters",
        cluster0=f"{hit_prefix}_cluster0",
        frac0=f"{hit_prefix}_frac0",
        clusters=tuple(f"{hit_prefix}_cluster{i}" for i in range(4)),
        fracs=tuple(f"{hit_prefix}_frac{i}" for i in range(4)),
        object_fields={
            "impact_eta": f"{cluster_prefix}_impact_eta",
            "impact_phi": f"{cluster_prefix}_impact_phi",
            "impact_energy": f"{cluster_prefix}_impact_energy",
            "impact_pt": f"{cluster_prefix}_impact_pt",
            "sim_energy": f"{cluster_prefix}_simEnergy",
            "track_pdg_id": f"{cluster_prefix}_track_pdgId",
        },
    )


def cmssw_branches_from_cfg(cfg: dict[str, Any]) -> CmsswBranches:
    """Build branch names for the chain selected by ``cfg["merging"]``."""
    return cmssw_branches(cfg.get("merging"))


# Backward-compatible module-level constants for the baseline chain. These are kept
# so existing imports (tests, diagnostics) that use them as dict keys keep working.
_BASELINE = cmssw_branches("")
HIT_PREFIX = _BASELINE.hit_prefix
CLUSTER_PREFIX = _BASELINE.cluster_prefix
HIT_X = _BASELINE.x
HIT_Y = _BASELINE.y
HIT_Z = _BASELINE.z
HIT_ENERGY = _BASELINE.energy
HIT_LAYER = _BASELINE.layer
HIT_ZSIDE = _BASELINE.zside
HIT_TIME = _BASELINE.time
HIT_NCLUSTERS = _BASELINE.nclusters
HIT_CLUSTER0 = _BASELINE.cluster0
HIT_FRAC0 = _BASELINE.frac0
HIT_CLUSTERS = _BASELINE.clusters
HIT_FRACS = _BASELINE.fracs
OBJECT_FIELDS = _BASELINE.object_fields

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
    branches = cmssw_branches_from_cfg(cfg)
    objects = _extract_objects(raw_event, branches)
    n_objects = _object_count(objects)
    raw_properties = _cmssw_object_properties(raw_event, n_objects, cfg, branches)
    objects.update({f"{name}_raw": values for name, values in raw_properties.items()})
    if processed_object_properties_supported(cfg):
        objects.update(_cmssw_object_properties(processed_event, n_objects, cfg, branches))

    cluster0 = processed_event[branches.cluster0].astype(np.int32)
    valid = _valid_cluster0(processed_event, n_objects, branches)
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


def _extract_objects(
    raw_event: dict[str, np.ndarray],
    branches: CmsswBranches,
) -> dict[str, np.ndarray]:
    """Read CMSSW SimCluster branches into canonical object-field names."""
    return {
        name: raw_event[branch].astype(_object_dtype(name))
        for name, branch in branches.object_fields.items()
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
    branches: CmsswBranches | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build configured RecHit-to-SimCluster aggregation links."""
    branches = cmssw_branches_from_cfg(cfg) if branches is None else branches
    slots, fractional = object_aggregation_spec(cfg)
    return _links_for_slots(
        raw_event, n_objects, slots=slots, fractional=fractional, branches=branches
    )


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
    branches: CmsswBranches | None = None,
) -> dict[str, np.ndarray]:
    branches = cmssw_branches_from_cfg(cfg) if branches is None else branches
    hit_indices, linked_object_ids, weights = cmssw_object_links(
        raw_event, n_objects, cfg, branches
    )
    return compute_object_properties_from_links(
        {
            "x": raw_event[branches.x],
            "y": raw_event[branches.y],
            "z": raw_event[branches.z],
            "energy": raw_event[branches.energy],
            "layer": raw_event[branches.layer],
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
    branches: CmsswBranches = _BASELINE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit_indices = []
    linked_object_ids = []
    weights = []
    nclusters = raw_event[branches.nclusters]
    for slot in slots:
        clusters = raw_event[branches.clusters[slot]].astype(np.int32)
        valid = (nclusters > slot) & (clusters >= 0) & (clusters < n_objects)
        selected_hits = np.flatnonzero(valid)
        hit_indices.append(selected_hits)
        linked_object_ids.append(clusters[valid])
        weights.append(
            np.abs(raw_event[branches.fracs[slot]][valid]).astype(np.float32)
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


def _valid_cluster0(
    raw_event: dict[str, np.ndarray],
    n_objects: int,
    branches: CmsswBranches = _BASELINE,
) -> np.ndarray:
    cluster0 = raw_event[branches.cluster0]
    return (raw_event[branches.nclusters] > 0) & (cluster0 >= 0) & (cluster0 < n_objects)


def _preprocessing_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("preprocessing", "rechits_energy_threshold"))


def processed_object_properties_supported(cfg: dict[str, Any]) -> bool:
    return _preprocessing_mode(cfg) in PROCESSED_OBJECT_PROPERTY_MODES


def _object_dtype(name: str) -> Any:
    if name == "track_pdg_id" or name.startswith(("n_hits", "core_shower_length")):
        return np.int32
    return np.float32
