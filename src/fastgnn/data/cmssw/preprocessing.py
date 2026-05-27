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

from typing import Any

import numpy as np

from .utils import xyz_to_eta_phi

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

OBJECT_FIELDS = {
    "impact_eta": f"{CLUSTER_PREFIX}_impact_eta",
    "impact_phi": f"{CLUSTER_PREFIX}_impact_phi",
    "impact_energy": f"{CLUSTER_PREFIX}_impact_energy",
    "impact_pt": f"{CLUSTER_PREFIX}_impact_pt",
    "sim_energy": f"{CLUSTER_PREFIX}_simEnergy",
    "track_pdg_id": f"{CLUSTER_PREFIX}_track_pdgId",
}

# Derived after hit preprocessing. A kept SimCluster may have zero selected hits.
DERIVED_OBJECT_FIELDS = ("n_hits", "visible_energy", "is_visible")


def select_features(raw_event: dict, cfg: dict) -> np.ndarray:
    """Select and stack model input features for one already-filtered event."""
    x = raw_event[HIT_X]
    y = raw_event[HIT_Y]
    z = raw_event[HIT_Z]
    energy = raw_event[HIT_ENERGY]
    layer = raw_event[HIT_LAYER].astype(np.float32)

    coord_sys = cfg.get("coordinate_system", "cartesian")

    if coord_sys == "cartesian":
        return np.stack([x, y, z, energy], axis=1).astype(np.float32)
    if coord_sys == "cylindrical":
        eta, phi = xyz_to_eta_phi(x, y, z)
        return np.stack([eta, phi, layer, energy], axis=1).astype(np.float32)
    if coord_sys == "both":
        eta, phi = xyz_to_eta_phi(x, y, z)
        return np.stack([x, y, z, eta, phi, layer, energy], axis=1).astype(np.float32)
    raise ValueError(
        f"Unknown coordinate_system: '{coord_sys}'. "
        "Choose from: 'cartesian', 'cylindrical', 'both'."
    )


def get_feature_names(cfg: dict) -> list[str]:
    """Return feature names matching the output of select_features()."""
    coord_sys = cfg.get("coordinate_system", "cartesian")
    if coord_sys == "cartesian":
        return ["x", "y", "z", "energy"]
    if coord_sys == "cylindrical":
        return ["eta", "phi", "layer", "energy"]
    if coord_sys == "both":
        return ["x", "y", "z", "eta", "phi", "layer", "energy"]
    raise ValueError(f"Unknown coordinate_system: '{coord_sys}'")


def build_truth(raw_event: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, Any]:
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
                            plus derived n_hits, visible_energy, is_visible fields
    """
    objects = _extract_objects(raw_event)
    n_objects = _object_count(objects)

    # CMSSW association fields are hit-shaped. Only use cluster0 values that
    # actually point into the source SimCluster arrays.
    nclusters_per_hit = raw_event[HIT_NCLUSTERS]
    cluster0 = raw_event[HIT_CLUSTER0].astype(np.int32)
    valid = (nclusters_per_hit > 0) & (cluster0 >= 0) & (cluster0 < n_objects)

    # Filter SimClusters, then remap original
    # CMSSW indices to OC labels: old index -> compact 1-based hit_object_id.
    object_keep = _filter_truth_objects(objects, cfg)
    kept_cluster_indices = np.flatnonzero(object_keep)
    old_to_new = np.zeros(n_objects, dtype=np.int32)
    old_to_new[kept_cluster_indices] = np.arange(
        1,
        len(kept_cluster_indices) + 1,
        dtype=np.int32,
    )

    # Assign OC labels by indexing the lookup table with each hit's best cluster.
    # Unassociated hits and hits whose best cluster was filtered away stay at 0,
    # ie. are assigned to the OC noise/background.
    hit_object_id = np.zeros(len(cluster0), dtype=np.int32)
    hit_object_id[valid] = old_to_new[cluster0[valid]]

    object_n_hits, object_visible_energy = _object_visibility(
        hit_object_id,
        raw_event[HIT_ENERGY],
        n_kept_objects=len(kept_cluster_indices),
    )

    filtered_objects = {
        name: values[object_keep].astype(_object_dtype(name))
        for name, values in objects.items()
    }

    visible_energy_keep = _visible_energy_mask(object_visible_energy, cfg)
    if visible_energy_keep is not None:
        hit_object_id, kept_visible_indices = _apply_kept_object_mask_to_labels(
            hit_object_id,
            visible_energy_keep,
        )
        filtered_objects = {
            name: values[kept_visible_indices]
            for name, values in filtered_objects.items()
        }
        object_n_hits, object_visible_energy = _object_visibility(
            hit_object_id,
            raw_event[HIT_ENERGY],
            n_kept_objects=len(kept_visible_indices),
        )

    filtered_objects["n_hits"] = object_n_hits
    filtered_objects["visible_energy"] = object_visible_energy
    filtered_objects["is_visible"] = object_n_hits > 0
    return {
        "hit_object_id": hit_object_id,
        "objects": filtered_objects,
    }


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
    return int(len(first))


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

    min_energy = cfg.get("truth_min_object_energy", cfg.get("truth_object_energy_threshold"))
    if min_energy is None:
        return keep

    energy_field = str(cfg.get("truth_object_energy_field", "impact_energy"))
    if energy_field not in objects:
        available = ", ".join(objects)
        raise KeyError(
            f"Unknown truth_object_energy_field={energy_field!r}. "
            f"Available object fields: {available}"
        )
    return keep & (objects[energy_field] >= float(min_energy))


def _truth_zside_mask(objects: dict[str, np.ndarray], zside: int) -> np.ndarray:
    """Match SimCluster impact eta sign to the selected HGCAL endcap."""
    if zside not in {-1, 1}:
        raise ValueError(f"zside must be -1 or 1, got {zside}")
    if "impact_eta" not in objects:
        raise KeyError("Cannot filter truth objects by zside without impact_eta")
    return np.sign(objects["impact_eta"]) == zside


def _visible_energy_mask(
    visible_energy: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """
    Select already-kept truth objects by selected RecHit energy.

    This threshold is disabled by default because visible energy is well-defined
    for RecHit inputs, but may be ill-defined for ECON-T AutoEncoder compression
    and possibly Best Choice, where individual RecHits are no longer input
    vertices.
    """
    min_visible_energy = cfg.get("truth_min_visible_energy")
    if min_visible_energy is None:
        return None
    return visible_energy >= float(min_visible_energy)


def _apply_kept_object_mask_to_labels(
    hit_object_id: np.ndarray,
    object_keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Relabel compact OC labels after a second object-level filter.

    ``object_keep`` is indexed over the current filtered object arrays. Hits
    assigned to removed objects become 0; kept objects are compacted back to
    1-based labels.
    """
    kept_object_indices = np.flatnonzero(object_keep)
    old_to_new = np.zeros(len(object_keep) + 1, dtype=np.int32)
    old_to_new[kept_object_indices + 1] = np.arange(
        1,
        len(kept_object_indices) + 1,
        dtype=np.int32,
    )
    return old_to_new[hit_object_id], kept_object_indices


def _object_visibility(
    hit_object_id: np.ndarray,
    hit_energy: np.ndarray,
    n_kept_objects: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count RecHits and visible energy per kept truth object.

    The function ignores noise hits, converts positive labels back to 0-based
    object indices, then accumulates one hit count and that hit's energy into
    the matching object slot. Repeated object indices are expected because many
    RecHits can belong to the same SimCluster;

    A kept SimCluster can have zero selected hits after hit preprocessing, for
    example when all associated RecHits failed the hit-energy threshold.

    This RecHit-based definition will need revisiting for compressed inputs
    where individual RecHits are no longer vertices.
    """
    n_hits = np.zeros(n_kept_objects, dtype=np.int32)
    visible_energy = np.zeros(n_kept_objects, dtype=np.float32)
    if n_kept_objects == 0:
        return n_hits, visible_energy

    signal = hit_object_id > 0
    object_indices = hit_object_id[signal] - 1
    np.add.at(n_hits, object_indices, 1)
    np.add.at(visible_energy, object_indices, hit_energy[signal].astype(np.float32))
    return n_hits, visible_energy


def _object_dtype(name: str) -> Any:
    if name in {"track_pdg_id", "n_hits"}:
        return np.int32
    if name == "is_visible":
        return np.bool_
    return np.float32
