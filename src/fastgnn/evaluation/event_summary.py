"""Single-event accounting helpers for the analysis notebook.

These build per-event tables (energy accounting, per-truth-object match summaries,
fake-cluster rows) from an :class:`~fastgnn.evaluation.tables.OCEvaluation`. They are
pure data transforms -- the notebook owns the figure assembly that consumes them.

``prediction_source`` selects which predicted columns to read: the notebook labels
its payload regression source ``"payl."``; any other value uses the aggregated
(energy-weighted) reco columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastgnn.evaluation._common import delta_phi
from fastgnn.geometry import xyz_to_eta_phi

PAYLOAD_SOURCE = "payl."


def event_table_rows(frame: Any, event_idx: int) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return [row for row in frame.to_dicts() if int(row.get("event_idx", -1)) == int(event_idx)]


def event_seed_truth_matches(oc_eval: Any, event_idx: int) -> dict[int, int]:
    if oc_eval is None or getattr(oc_eval, "matches", None) is None:
        raise ValueError("plot_true_vs_pred_oc requires oc_eval.matches for cluster matching.")
    rows = event_table_rows(oc_eval.matches, event_idx)
    return {
        int(row["seed_hit_idx"]): int(row["object_id"])
        for row in rows
        if row.get("seed_hit_idx") is not None and row.get("object_id") is not None
    }


def event_hit_observables(
    event: Any,
    event_idx: int,
    valid_mask: np.ndarray,
    *,
    eval_features: np.ndarray | None = None,
    eval_feature_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    if eval_features is not None and eval_feature_names is not None:
        feature_names = list(eval_feature_names)
        energy = np.asarray(
            eval_features[event_idx, :, feature_names.index("energy")], dtype=np.float64
        )[valid_mask]
        x = np.asarray(eval_features[event_idx, :, feature_names.index("x")], dtype=np.float64)[
            valid_mask
        ]
        y = np.asarray(eval_features[event_idx, :, feature_names.index("y")], dtype=np.float64)[
            valid_mask
        ]
        z = np.asarray(eval_features[event_idx, :, feature_names.index("z")], dtype=np.float64)[
            valid_mask
        ]
    else:
        energy = np.asarray(event.hits.energy, dtype=np.float64)
        x = np.asarray(event.hits.x, dtype=np.float64)
        y = np.asarray(event.hits.y, dtype=np.float64)
        z = np.asarray(event.hits.z, dtype=np.float64)
        if len(energy) == len(valid_mask):
            energy, x, y, z = energy[valid_mask], x[valid_mask], y[valid_mask], z[valid_mask]
        elif len(energy) >= int(valid_mask.sum()):
            n_valid = int(valid_mask.sum())
            energy, x, y, z = energy[:n_valid], x[:n_valid], y[:n_valid], z[:n_valid]
        else:
            raise ValueError(
                "Cannot align event hit observables to the padded mask. Pass eval_features "
                "with unnormalized 'x', 'y', 'z', and 'energy' features."
            )

    eta, _ = xyz_to_eta_phi(x, y, z)
    return {"energy": energy, "et": energy / np.cosh(eta)}


def pred_et_from_row(row: dict[str, Any], prediction_source: str) -> float | None:
    key = "payload_et_pred" if prediction_source == PAYLOAD_SOURCE else "sum_et_reco"
    return row.get(key)


def pred_energy_from_row(row: dict[str, Any], prediction_source: str) -> float | None:
    key = "payload_energy_pred" if prediction_source == PAYLOAD_SOURCE else "energy_pred"
    return row.get(key)


def _nearest_truth_dist_for_pred(
    pred_row: dict[str, Any], truth_rows: list[dict[str, Any]]
) -> float | None:
    pred_eta = pred_row.get("centroid_eta_reco")
    pred_phi = pred_row.get("centroid_phi_reco")
    if pred_eta is None or pred_phi is None:
        return None

    distances = []
    for truth_row in truth_rows:
        truth_eta = truth_row.get("truth_centroid_eta")
        truth_phi = truth_row.get("truth_centroid_phi")
        dphi = delta_phi(pred_phi, truth_phi)
        if truth_eta is not None and truth_phi is not None and dphi is not None:
            distances.append(float(np.hypot(float(pred_eta) - float(truth_eta), dphi)))
    return min(distances) if distances else None


def event_energy_accounting(
    *,
    hit_et: np.ndarray,
    hit_object_id: np.ndarray,
    clustering: np.ndarray,
    truth_match_rows: list[dict[str, Any]],
    oc_eval: Any,
    event_idx: int,
    prediction_source: str,
) -> dict[str, float]:
    pred_rows = event_table_rows(None if oc_eval is None else oc_eval.predicted, event_idx)
    hit_et = np.asarray(hit_et, dtype=np.float64)
    total = float(hit_et.sum())
    signal = float(hit_et[hit_object_id > 0].sum())
    noise = float(hit_et[hit_object_id == 0].sum())
    assigned = float(hit_et[clustering >= 0].sum())
    unassigned = float(hit_et[clustering < 0].sum())
    matched_pred = sum(
        float(pred_et_from_row(row, prediction_source))
        for row in pred_rows
        if row.get("matched") and pred_et_from_row(row, prediction_source) is not None
    )
    fake_pred = sum(
        float(pred_et_from_row(row, prediction_source))
        for row in pred_rows
        if row.get("fake") and pred_et_from_row(row, prediction_source) is not None
    )
    recovered = sum(
        float(row["recovered_et"])
        for row in truth_match_rows
        if row.get("recovered_et") is not None
    )
    return {
        "valid_hit_energy": total,
        "signal_hit_energy": signal,
        "noise_hit_energy": noise,
        "assigned_hit_energy": assigned,
        "unassigned_hit_energy": unassigned,
        "matched_pred_energy": matched_pred,
        "fake_pred_energy": fake_pred,
        "recovered_matched_energy": recovered,
        "signal_energy_fraction": 100.0 * signal / total if total > 0 else 0.0,
        "assigned_energy_fraction": 100.0 * assigned / total if total > 0 else 0.0,
        "unassigned_energy_fraction": 100.0 * unassigned / total if total > 0 else 0.0,
        "recovered_energy_fraction": 100.0 * recovered / total if total > 0 else 0.0,
    }


def event_truth_match_summary(
    oc_eval: Any,
    event_idx: int,
    object_order: list[int],
    *,
    hit_object_id: np.ndarray,
    clustering: np.ndarray,
    hit_et: np.ndarray,
    prediction_source: str,
) -> list[dict[str, Any]]:
    if oc_eval is None:
        return []
    truth_rows = event_table_rows(oc_eval.truth, event_idx)
    match_rows = event_table_rows(oc_eval.matches, event_idx)
    pred_rows = event_table_rows(oc_eval.predicted, event_idx)
    match_by_object = {int(row["object_id"]): row for row in match_rows}
    pred_by_cluster = {int(row["cluster_id_pred"]): row for row in pred_rows}

    ordered_object_ids = list(object_order)
    remaining_object_ids = sorted(
        int(row["object_id"])
        for row in truth_rows
        if int(row["object_id"]) not in ordered_object_ids
    )
    ordered_object_ids.extend(remaining_object_ids)
    truth_by_object = {int(row["object_id"]): row for row in truth_rows}

    rows = []
    for object_id in ordered_object_ids:
        truth_row = truth_by_object.get(object_id)
        if truth_row is None:
            continue
        match_row = match_by_object.get(object_id)
        pred_row = (
            None if match_row is None else pred_by_cluster.get(int(match_row["cluster_id_pred"]))
        )
        truth_ref_energy = truth_row.get("truth_energy")
        truth_sum_energy = (
            match_row.get("truth_sum_energy")
            if match_row is not None
            else truth_row.get("truth_sum_energy", truth_ref_energy)
        )
        truth_energy = truth_sum_energy
        pred_energy = (
            None if pred_row is None else pred_energy_from_row(pred_row, prediction_source)
        )
        truth_et = truth_row.get("truth_et")
        pred_et = None if pred_row is None else pred_et_from_row(pred_row, prediction_source)
        seed_hit_idx = None if match_row is None else int(match_row["seed_hit_idx"])
        recovered_et = None
        if seed_hit_idx is not None:
            overlap_mask = (hit_object_id == object_id) & (clustering == seed_hit_idx)
            recovered_et = float(np.asarray(hit_et, dtype=np.float64)[overlap_mask].sum())
        pred_over_truth_et = (
            None if pred_et is None or truth_et in (None, 0) else float(pred_et) / float(truth_et)
        )
        energy_recovered_fraction = (
            None if match_row is None else match_row.get("energy_recovered_fraction")
        )
        et_recovered_fraction = (
            None
            if recovered_et is None or truth_et in (None, 0)
            else float(recovered_et) / float(truth_et)
        )
        et_assigned_fraction = (
            None
            if recovered_et is None or pred_et in (None, 0)
            else float(recovered_et) / float(pred_et)
        )
        recovered_energy = (
            None
            if energy_recovered_fraction is None or truth_sum_energy is None
            else float(energy_recovered_fraction) * float(truth_sum_energy)
        )
        rows.append(
            {
                "object_id": object_id,
                "truth_energy": truth_energy,
                "truth_ref_energy": truth_ref_energy,
                "truth_sum_energy": truth_sum_energy,
                "truth_et": truth_et,
                "impact_pt": truth_row.get("truth_pt"),
                "n_hits_truth": truth_row.get("n_hits_truth"),
                "nearest_truth_dist": truth_row.get("nearest_truth_dist"),
                "beta_max": truth_row.get("beta_max"),
                "matched": bool(truth_row.get("matched", False)),
                "n_hits_pred": None if pred_row is None else pred_row.get("n_hits_pred"),
                "energy_pred": pred_energy,
                "et_pred": pred_et,
                "pred_over_truth_et": pred_over_truth_et,
                "recovered_energy": recovered_energy,
                "recovered_et": recovered_et,
                "purity": None if match_row is None else match_row.get("purity"),
                "completeness": None if match_row is None else match_row.get("completeness"),
                "energy_recovered_fraction": energy_recovered_fraction,
                "et_recovered_fraction": et_recovered_fraction,
                "et_assigned_fraction": et_assigned_fraction,
                "energy_assigned_fraction": None
                if match_row is None
                else match_row.get("energy_assigned_fraction"),
            }
        )
    return rows


def event_fake_cluster_rows(
    oc_eval: Any, event_idx: int, *, prediction_source: str
) -> list[dict[str, Any]]:
    if oc_eval is None:
        return []
    truth_rows = event_table_rows(oc_eval.truth, event_idx)
    pred_rows = event_table_rows(oc_eval.predicted, event_idx)
    fake_rows = [
        row
        for row in pred_rows
        if bool(row.get("fake", False)) and not bool(row.get("matched", False))
    ]
    fake_rows = sorted(
        fake_rows,
        key=lambda row: (
            int(row.get("seed_hit_idx", -1))
            if row.get("seed_hit_idx") is not None
            else int(row.get("cluster_id_pred", -1))
        ),
    )
    return [
        {
            "object_id": f"fakecluster seed {int(row['seed_hit_idx'])}",
            "matched": False,
            "truth_energy": None,
            "energy_pred": pred_energy_from_row(row, prediction_source),
            "impact_pt": None,
            "truth_et": None,
            "et_pred": pred_et_from_row(row, prediction_source),
            "recovered_et": None,
            "n_hits_truth": None,
            "n_hits_pred": row.get("n_hits_pred"),
            "purity": None,
            "completeness": None,
            "et_recovered_fraction": None,
            "et_assigned_fraction": None,
            "beta_max": row.get("beta_seed"),
            "nearest_truth_dist": _nearest_truth_dist_for_pred(row, truth_rows),
        }
        for row in fake_rows
    ]
