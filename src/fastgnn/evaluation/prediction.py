"""Prediction-side helpers shared by analysis notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from fastgnn.data.base import EventRecord
from fastgnn.evaluation.clustering import clustering_from_beta
from fastgnn.evaluation.payload import decode_payload, payload_pred_columns
from fastgnn.evaluation.reco import reco_properties_by_assignment
from fastgnn.training.oc_outputs import OCOutputLayout

AGGREGATED_PREDICTION_FIELDS = (
    "beta_seed",
    "n_hits_pred",
    "sum_energy_reco",
    "sum_et_reco",
    "centroid_x_reco",
    "centroid_y_reco",
    "centroid_z_reco",
    "centroid_eta_reco",
    "centroid_phi_reco",
)

PAYLOAD_PREDICTION_FIELDS = tuple(
    f"payload_{name}_pred" for name in ("energy", "et", "x", "y", "z", "eta", "phi")
)


def payload_quantities_from_config(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Configured payload quantities for a run config."""
    return list(
        (((config or {}).get("training") or {}).get("payload") or {}).get("quantities") or []
    )


def prediction_eval_feature_names(
    payload_quantities: list[dict[str, Any]] | None,
    *,
    base_feature_names: list[str] | None = None,
) -> list[str]:
    """Unnormalized feature names needed for reco and payload decoding."""
    feature_names = list(base_feature_names or ["x", "y", "z", "energy"])
    for quantity in payload_quantities or []:
        seed = quantity.get("seed")
        if seed is not None and str(seed) not in feature_names:
            feature_names.append(str(seed))
    return feature_names


def oc_layout_from_model_config(model_cfg: dict[str, Any]) -> OCOutputLayout:
    """Backward-compatible layout construction from run config."""
    try:
        return OCOutputLayout.from_config(model_cfg)
    except ValueError as exc:
        if "output_layout.regressions is no longer supported" not in str(exc):
            raise

    layout_cfg = dict(model_cfg.get("output_layout") or {})
    regressions = list(layout_cfg.pop("regressions", []) or [])
    cluster_cfg = dict(layout_cfg.get("cluster_space") or {})
    cluster_start = int(cluster_cfg.get("start", 1))
    cluster_dim = int(cluster_cfg["dim"])
    payload_start = cluster_start + cluster_dim
    payload_dim = int(model_cfg.get("payload_output_dim") or 0)

    if regressions:
        starts = [int(reg["start"]) for reg in regressions]
        stops = [int(reg["start"]) + int(reg["dim"]) for reg in regressions]
        payload_start = min(starts)
        payload_dim = max(stops) - payload_start
    elif payload_dim == 0:
        payload_dim = int(model_cfg["output_dim"]) - 1 - cluster_dim

    if payload_dim > 0:
        layout_cfg["payload"] = {"start": payload_start, "dim": payload_dim}

    compat_cfg = dict(model_cfg)
    compat_cfg["output_layout"] = layout_cfg
    return OCOutputLayout.from_config(compat_cfg)


def decode_payload_outputs(
    payload: np.ndarray | None,
    payload_quantities: list[dict[str, Any]] | None,
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray] | None:
    """Decode per-hit payload outputs into physical quantities."""
    return decode_payload(payload, payload_quantities, features, feature_names)


def available_prediction_quantities(
    decoded_payload_features: dict[str, np.ndarray] | None,
) -> tuple[list[str], list[str]]:
    """Available aggregated and payload prediction field names."""
    aggregated = list(AGGREGATED_PREDICTION_FIELDS)
    payload = [
        field
        for field in PAYLOAD_PREDICTION_FIELDS
        if decoded_payload_features is not None
        and field.removeprefix("payload_").removesuffix("_pred") in decoded_payload_features
    ]
    return aggregated, payload


def build_predicted_events(
    dataset: list[EventRecord] | Any,
    eval_inputs: dict[str, np.ndarray],
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    tbeta: float,
    td: float,
    *,
    eval_feature_names: list[str],
    decoded_payload_features: dict[str, np.ndarray] | None = None,
    selected_prediction_quantities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Materialize one row per event with nested predicted cluster properties."""
    selected_prediction_quantities = list(
        selected_prediction_quantities or AGGREGATED_PREDICTION_FIELDS[:4]
    )

    predicted_event_rows: list[dict[str, Any]] = []
    for event_idx in range(len(dataset)):
        event = dataset[event_idx]
        valid = np.asarray(eval_inputs["mask"][event_idx], dtype=bool)
        event_features = np.asarray(eval_inputs["features"][event_idx, valid], dtype=np.float64)
        event_beta = np.asarray(beta[event_idx])[valid]
        event_coords = np.asarray(cluster_coords[event_idx])[valid]
        payload_event_features = (
            None
            if decoded_payload_features is None
            else {
                key: np.asarray(values[event_idx], dtype=np.float64)[valid]
                for key, values in decoded_payload_features.items()
            }
        )

        cluster_rows, event_summary, clustering = _build_event_prediction(
            beta=event_beta,
            cluster_coords=event_coords,
            event_features=event_features,
            eval_feature_names=eval_feature_names,
            tbeta=float(tbeta),
            td=float(td),
            payload_event_features=payload_event_features,
            selected_prediction_quantities=selected_prediction_quantities,
        )
        source_event_id = (
            int(event.metadata["source_event_id"])
            if "source_event_id" in getattr(event.metadata, "fields", [])
            else None
        )
        predicted_event_rows.append(
            {
                "dataset_idx": int(event_idx),
                "event_id": int(event.event_id),
                "source_event_id": source_event_id,
                "selected_prediction_quantities": list(selected_prediction_quantities),
                **event_summary,
                "pred_cluster_id_per_tc": clustering.tolist(),
                "clusters": cluster_rows,
            }
        )
    return predicted_event_rows


def _build_event_prediction(
    *,
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    event_features: np.ndarray,
    eval_feature_names: list[str],
    tbeta: float,
    td: float,
    payload_event_features: dict[str, np.ndarray] | None,
    selected_prediction_quantities: list[str],
) -> tuple[list[dict[str, Any]], dict[str, float | int | None], np.ndarray]:
    if len(beta) and np.any(beta > tbeta):
        clustering = clustering_from_beta(beta, cdist(cluster_coords, cluster_coords), tbeta, td)
    else:
        clustering = -1 * np.ones(len(beta), dtype=np.int32)

    assigned_mask = clustering >= 0
    seed_indices = (
        np.unique(clustering[assigned_mask])
        if np.any(assigned_mask)
        else np.asarray([], dtype=np.int32)
    )
    seed_indices = seed_indices[np.argsort(-beta[seed_indices])]
    pred_assignments = -1 * np.ones(len(clustering), dtype=np.int32)
    for cluster_id_pred, seed_idx in enumerate(seed_indices, start=1):
        pred_assignments[clustering == seed_idx] = cluster_id_pred

    physical_hits = {
        name: np.asarray(event_features[:, eval_feature_names.index(name)], dtype=np.float64)
        for name in ("x", "y", "z", "energy")
    }
    pred_reco = reco_properties_by_assignment(
        pred_assignments,
        np.arange(1, len(seed_indices) + 1, dtype=np.int32),
        physical_hits,
    )

    cluster_rows = []
    payload_energy_sum = 0.0
    payload_et_sum = 0.0
    for cluster_id_pred, seed_idx in enumerate(seed_indices):
        cluster_mask = clustering == seed_idx
        reco = pred_reco[cluster_id_pred + 1]
        cluster_values: dict[str, float | int | None] = {
            "beta_seed": float(beta[seed_idx]),
            "n_hits_pred": int(np.sum(cluster_mask)),
            "sum_energy_reco": reco["energy"],
            "sum_et_reco": reco["et"],
            "centroid_x_reco": reco["x"],
            "centroid_y_reco": reco["y"],
            "centroid_z_reco": reco["z"],
            "centroid_eta_reco": reco["eta"],
            "centroid_phi_reco": reco["phi"],
        }
        if payload_event_features is not None:
            cluster_values.update(payload_pred_columns(payload_event_features, int(seed_idx)))
            if cluster_values.get("payload_energy_pred") is not None:
                payload_energy_sum += float(cluster_values["payload_energy_pred"])
            if cluster_values.get("payload_et_pred") is not None:
                payload_et_sum += float(cluster_values["payload_et_pred"])

        row = {"cluster_id_pred": int(cluster_id_pred), "seed_hit_idx": int(seed_idx)}
        for quantity in selected_prediction_quantities:
            row[quantity] = cluster_values.get(quantity)
        cluster_rows.append(row)

    event_summary: dict[str, float | int | None] = {
        "n_pred_clusters": len(seed_indices),
        "event_sum_energy_reco": float(
            sum(pred_reco[idx + 1]["energy"] for idx in range(len(seed_indices)))
        ),
        "event_sum_et_reco": float(
            sum(pred_reco[idx + 1]["et"] for idx in range(len(seed_indices)))
        ),
        "event_payload_energy_sum": payload_energy_sum
        if payload_event_features is not None
        else None,
        "event_payload_et_sum": payload_et_sum if payload_event_features is not None else None,
        "unassigned_tc_energy": float(np.sum(physical_hits["energy"][~assigned_mask])),
        "valid_tc_energy": float(np.sum(physical_hits["energy"])),
    }
    return cluster_rows, event_summary, clustering
