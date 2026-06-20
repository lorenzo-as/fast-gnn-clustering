"""Feature construction for classical CMSSW trigger-cell ntuples."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fastgnn.data.cmssw.hit_features import DEFAULT_LOG_FLOOR, HitFeatures

CANONICAL_ALIASES = {
    "x": "tc_x",
    "y": "tc_y",
    "z": "tc_z",
    "energy": "tc_energy",
    "layer": "tc_layer",
    "zside": "tc_zside",
    "eta": "tc_eta",
    "phi": "tc_phi",
    "pt": "tc_pt",
}

CURRENT_CMSSW_HIT_FEATURES = (
    "x",
    "y",
    "z",
    "r",
    "eta",
    "phi",
    "energy",
    "et",
    "layer",
    "x_over_z",
    "y_over_z",
    "log_energy",
    "log_et",
    "cos_phi",
    "sin_phi",
)


def hit_dtype(field: str):
    if field in {"tc_pt", "tc_mipPt", "tc_energy", "tc_eta", "tc_phi", "tc_x", "tc_y", "tc_z"}:
        return np.float32
    if field == "tc_id":
        return np.int64
    if field in {"tc_zside", "tc_subdet", "tc_wafertype"}:
        return np.int8
    return np.int16


def alias_dtype(alias: str):
    if alias == "layer":
        return np.int16
    if alias == "zside":
        return np.int8
    return np.float32


def build_current_cmssw_features(
    base_hits: Mapping[str, np.ndarray],
    feature_names: list[str] | tuple[str, ...] | None = None,
    *,
    log_floor: float = DEFAULT_LOG_FLOOR,
) -> dict[str, np.ndarray]:
    """Build the current GravNet input features from canonical hit fields."""
    return HitFeatures(base_hits, log_floor=log_floor).build(
        CURRENT_CMSSW_HIT_FEATURES if feature_names is None else feature_names
    )
