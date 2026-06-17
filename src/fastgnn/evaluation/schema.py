"""Canonical schemas for OC evaluation result frames.

Centralizing the column names here keeps empty (zero-row) frames carrying their
full schema -- so a degenerate model that produces no clusters yields frames the
notebook can still index by column instead of raising ``ColumnNotFoundError`` --
and gives the notebook a single source for the aggregated-vs-payload column names.
"""

from __future__ import annotations

import polars as pl

_INT = pl.Int64
_FLOAT = pl.Float64
_BOOL = pl.Boolean
_STR = pl.String

ATTRIBUTES = ("events", "truth", "predicted", "matches", "seeds")

_EVENTS: dict[str, pl.DataType] = {
    "event_idx": _INT,
    "n_hits": _INT,
    "n_truth": _INT,
    "n_pred": _INT,
    "n_matched": _INT,
    "signal_fraction": _FLOAT,
    "tbeta": _FLOAT,
    "td": _FLOAT,
    "max_match_distance": _FLOAT,
    "min_energy_ratio": _FLOAT,
    "max_energy_ratio": _FLOAT,
    "matching_algorithm": _STR,
    "hungarian_energy_ratio_log_weight": _FLOAT,
}

_TRUTH: dict[str, pl.DataType] = {
    "event_idx": _INT,
    "object_id": _INT,
    "truth_energy": _FLOAT,
    "truth_sum_energy": _FLOAT,
    "truth_et": _FLOAT,
    "truth_pt": _FLOAT,
    "truth_eta": _FLOAT,
    "truth_phi": _FLOAT,
    "truth_centroid_x": _FLOAT,
    "truth_centroid_y": _FLOAT,
    "truth_centroid_z": _FLOAT,
    "truth_centroid_eta": _FLOAT,
    "truth_centroid_phi": _FLOAT,
    "n_hits_truth": _INT,
    "truth_object_n_hits": _INT,
    "event_truth_multiplicity": _INT,
    "beta_max": _FLOAT,
    "seed_found": _BOOL,
    "nearest_truth_dist": _FLOAT,
    "matched": _BOOL,
}

_PAYLOAD_PRED: dict[str, pl.DataType] = {
    f"payload_{key}_pred": _FLOAT for key in ("energy", "et", "x", "y", "z", "eta", "phi")
}

_PREDICTED: dict[str, pl.DataType] = {
    "event_idx": _INT,
    "cluster_id_pred": _INT,
    "seed_hit_idx": _INT,
    "beta_seed": _FLOAT,
    "seed_truth_id": _INT,
    "n_hits_pred": _INT,
    "energy_pred": _FLOAT,
    "sum_et_reco": _FLOAT,
    "centroid_x_reco": _FLOAT,
    "centroid_y_reco": _FLOAT,
    "centroid_z_reco": _FLOAT,
    "centroid_eta_reco": _FLOAT,
    "centroid_phi_reco": _FLOAT,
    "dominant_truth_id": _INT,
    "dominant_truth_hit_fraction": _FLOAT,
    "matched": _BOOL,
    "fake": _BOOL,
    **_PAYLOAD_PRED,
}

_MATCHES_BASE: dict[str, pl.DataType] = {
    "event_idx": _INT,
    "object_id": _INT,
    "cluster_id_pred": _INT,
    "seed_hit_idx": _INT,
    "matching_reference": _STR,
    "beta_seed": _FLOAT,
    "centroid_distance": _FLOAT,
    "energy_ratio": _FLOAT,
    "purity": _FLOAT,
    "completeness": _FLOAT,
    "energy_assigned_fraction": _FLOAT,
    "energy_recovered_fraction": _FLOAT,
    "truth_energy": _FLOAT,
    "truth_sum_energy": _FLOAT,
    "truth_et": _FLOAT,
    "energy_pred": _FLOAT,
    "sum_et_reco": _FLOAT,
    "sum_energy_response": _FLOAT,
    "et_response": _FLOAT,
    "energy_response": _FLOAT,
    "relative_energy_residual": _FLOAT,
    "relative_sum_energy_residual": _FLOAT,
    "relative_et_residual": _FLOAT,
    "truth_centroid_x": _FLOAT,
    "truth_centroid_y": _FLOAT,
    "truth_centroid_z": _FLOAT,
    "truth_centroid_eta": _FLOAT,
    "truth_centroid_phi": _FLOAT,
    "centroid_x_reco": _FLOAT,
    "centroid_y_reco": _FLOAT,
    "centroid_z_reco": _FLOAT,
    "centroid_eta_reco": _FLOAT,
    "centroid_phi_reco": _FLOAT,
    "delta_x": _FLOAT,
    "delta_y": _FLOAT,
    "delta_z": _FLOAT,
    "delta_eta": _FLOAT,
    "delta_phi": _FLOAT,
    "eta_residual": _FLOAT,
    "phi_residual": _FLOAT,
    "z_residual": _FLOAT,
    "relative_eta_residual": _FLOAT,
    "relative_phi_residual": _FLOAT,
    "relative_z_residual": _FLOAT,
    "delta_r": _FLOAT,
}

_PAYLOAD_MATCH: dict[str, pl.DataType] = {
    **_PAYLOAD_PRED,
    "payload_energy_response": _FLOAT,
    "payload_et_response": _FLOAT,
    "payload_relative_energy_residual": _FLOAT,
    "payload_relative_et_residual": _FLOAT,
    "payload_eta_residual": _FLOAT,
    "payload_phi_residual": _FLOAT,
    "payload_z_residual": _FLOAT,
    "payload_relative_eta_residual": _FLOAT,
    "payload_relative_phi_residual": _FLOAT,
    "payload_relative_z_residual": _FLOAT,
    "payload_delta_r": _FLOAT,
}

_SEEDS: dict[str, pl.DataType] = {
    "event_idx": _INT,
    "seed_hit_idx": _INT,
    "seed_rank": _INT,
    "beta_seed": _FLOAT,
    "seed_truth_id": _INT,
    "seed_class": _STR,
}


def frame_schema(attribute: str, *, payload: bool) -> dict[str, pl.DataType]:
    """Column -> dtype schema for one result frame.

    ``payload`` controls whether the payload-derived match columns are present;
    ``predicted`` always carries the ``payload_*_pred`` columns.
    """
    if attribute == "events":
        return dict(_EVENTS)
    if attribute == "truth":
        return dict(_TRUTH)
    if attribute == "predicted":
        return dict(_PREDICTED)
    if attribute == "matches":
        return {**_MATCHES_BASE, **(_PAYLOAD_MATCH if payload else {})}
    if attribute == "seeds":
        return dict(_SEEDS)
    raise ValueError(f"Unknown evaluation frame {attribute!r}")


def empty_frame(attribute: str, *, payload: bool) -> pl.DataFrame:
    """Zero-row frame with the full declared schema for ``attribute``."""
    return pl.DataFrame(schema=frame_schema(attribute, payload=payload))


def match_response_columns(*, payload: bool) -> dict[str, object]:
    """Column-name mapping for an aggregated (``payload=False``) or payload source."""
    prefix = "payload_" if payload else ""
    return {
        "response_col": f"{prefix}et_response",
        "residual_col": f"{prefix}relative_et_residual",
        "pred_et_col": "payload_et_pred" if payload else "sum_et_reco",
        "position_cols": {
            "eta": f"{prefix}eta_residual",
            "phi": f"{prefix}phi_residual",
            "z": f"{prefix}z_residual",
        },
        "relative_position_cols": {
            "eta": f"{prefix}relative_eta_residual",
            "phi": f"{prefix}relative_phi_residual",
            "z": f"{prefix}relative_z_residual",
        },
    }
