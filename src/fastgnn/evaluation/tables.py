"""OC evaluation tables: clustering, truth/pred matching, and result frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from fastgnn.data.base import EventRecord
from fastgnn.evaluation._common import (
    delta_phi,
    none_subtract,
    none_to_nan,
    safe_ratio,
)
from fastgnn.evaluation.matching import (
    energy_ratio_matrix,
    match_candidates,
    matched_positions,
    payload_position_distance,
    validate_matching_algorithm,
    validate_matching_reference,
)
from fastgnn.evaluation.payload import (
    decode_payload,
    payload_match_columns,
    payload_pred_columns,
    payload_seed_reco,
)
from fastgnn.evaluation.reco import (
    delta_r,
    nearest_truth_distance,
    prefixed_centroid,
    reco_properties_by_assignment,
    residual_block,
    truth_object_props,
)
from fastgnn.evaluation.schema import empty_frame
from fastgnn.training.objectcondensation_loss import get_clustering_np
from fastgnn.training.oc_outputs import OCOutputLayout, split_oc_outputs


@dataclass(frozen=True)
class OCEvaluation:
    """Tabular OC evaluation result."""

    events: pl.DataFrame
    truth: pl.DataFrame
    predicted: pl.DataFrame
    matches: pl.DataFrame
    seeds: pl.DataFrame

    def count_summary(self) -> dict[str, float]:
        if self.events.is_empty():
            return {
                "median_abs_count_diff": 0.0,
                "mean_abs_count_diff": 0.0,
                "frac_exact": 0.0,
                "frac_within_1": 0.0,
            }
        diff = self.events["n_truth"].to_numpy() - self.events["n_pred"].to_numpy()
        return {
            "median_abs_count_diff": float(np.median(np.abs(diff))),
            "mean_abs_count_diff": float(np.mean(np.abs(diff))),
            "frac_exact": float(np.mean(diff == 0)),
            "frac_within_1": float(np.mean(np.abs(diff) <= 1)),
        }

    def seed_summary(self) -> dict[str, float]:
        n_truth = int(self.events["n_truth"].sum()) if "n_truth" in self.events.columns else 0
        n_seeds = len(self.seeds)
        n_matched = _count_value(self.seeds, "seed_class", "matched_seed")
        n_duplicate = _count_value(self.seeds, "seed_class", "duplicate_fake")
        n_noise = _count_value(self.seeds, "seed_class", "noise_fake")
        return {
            "seed_efficiency": safe_ratio(n_matched, n_truth),
            "duplicate_rate": safe_ratio(n_duplicate, n_seeds),
            "noise_seed_fake_rate": safe_ratio(n_noise, n_seeds),
            "missed_truth": float(max(n_truth - n_matched, 0)),
        }

    def matching_summary(self) -> dict[str, float]:
        n_truth = int(self.events["n_truth"].sum()) if "n_truth" in self.events.columns else 0
        n_pred = int(self.events["n_pred"].sum()) if "n_pred" in self.events.columns else 0
        n_matches = len(self.matches)
        return {
            "efficiency": safe_ratio(n_matches, n_truth),
            "fake_rate": safe_ratio(n_pred - n_matches, n_pred),
            "purity": safe_ratio(n_matches, n_pred),
            "miss_rate": safe_ratio(n_truth - n_matches, n_truth),
        }


@dataclass(frozen=True)
class _EventEval:
    event_idx: int
    event: EventRecord | None
    labels: np.ndarray
    beta: np.ndarray
    coords: np.ndarray
    payload_features: dict[str, np.ndarray] | None
    features: dict[str, np.ndarray]
    tbeta: float
    td: float
    max_match_distance: float
    min_energy_ratio: float
    max_energy_ratio: float
    matching_algorithm: str
    hungarian_energy_ratio_log_weight: float
    matching_reference: str
    n_valid_hits: int
    clustering: np.ndarray
    truth_ids: np.ndarray
    seed_indices: np.ndarray
    pred_ids: dict[int, int]
    truth_reco: dict[int, dict[str, float | None]]
    pred_reco: dict[int, dict[str, float | None]]

    @property
    def energy(self) -> np.ndarray:
        return self.features["energy"]

    @property
    def has_payload(self) -> bool:
        return self.payload_features is not None

    @classmethod
    def build(
        cls,
        *,
        beta: np.ndarray,
        cluster_coords: np.ndarray,
        payload_features: dict[str, np.ndarray] | None,
        hit_object_id: np.ndarray,
        mask: np.ndarray | None,
        features: np.ndarray,
        feature_names: list[str],
        tbeta: float,
        td: float,
        event_idx: int,
        event: EventRecord | None,
        max_match_distance: float,
        min_energy_ratio: float,
        max_energy_ratio: float,
        matching_algorithm: str,
        hungarian_energy_ratio_log_weight: float,
        matching_reference: str,
    ) -> _EventEval:
        valid = np.ones(len(hit_object_id), dtype=bool) if mask is None else mask.astype(bool)
        labels = np.asarray(hit_object_id[valid], dtype=np.int32)
        beta_v = np.asarray(beta[valid], dtype=np.float64)
        coords_v = np.asarray(cluster_coords[valid], dtype=np.float64)
        payload_features_v = (
            None
            if payload_features is None
            else {
                key: np.asarray(values[valid], dtype=np.float64)
                for key, values in payload_features.items()
            }
        )
        features_v = np.asarray(features[valid], dtype=np.float64)
        physical_hits = {
            name: _required_feature(features_v, feature_names, name)
            for name in ("x", "y", "z", "energy")
        }
        matching_reference = validate_matching_reference(matching_reference, payload_features_v)

        clustering = get_clustering_np(beta_v, coords_v, tbeta=tbeta, td=td)
        truth_ids = np.unique(labels[labels > 0]).astype(np.int32)
        seed_indices = np.unique(clustering[clustering >= 0]).astype(np.int32)
        seed_indices = seed_indices[np.argsort(-beta_v[seed_indices])]
        pred_ids = {int(seed_index): pred_id for pred_id, seed_index in enumerate(seed_indices)}

        pred_assignments = np.asarray(
            [pred_ids.get(int(seed_index), -1) + 1 for seed_index in clustering],
            dtype=np.int32,
        )
        return cls(
            event_idx=event_idx,
            event=event,
            labels=labels,
            beta=beta_v,
            coords=coords_v,
            payload_features=payload_features_v,
            features=physical_hits,
            tbeta=tbeta,
            td=td,
            max_match_distance=max_match_distance,
            min_energy_ratio=min_energy_ratio,
            max_energy_ratio=max_energy_ratio,
            matching_algorithm=validate_matching_algorithm(matching_algorithm),
            hungarian_energy_ratio_log_weight=float(hungarian_energy_ratio_log_weight),
            matching_reference=matching_reference,
            n_valid_hits=int(valid.sum()),
            clustering=clustering,
            truth_ids=truth_ids,
            seed_indices=seed_indices,
            pred_ids=pred_ids,
            truth_reco=reco_properties_by_assignment(labels, truth_ids, physical_hits),
            pred_reco=reco_properties_by_assignment(
                pred_assignments,
                np.arange(1, len(seed_indices) + 1, dtype=np.int32),
                physical_hits,
            ),
        )


def evaluate_oc_padded(
    *,
    preds: np.ndarray,
    hit_object_id: np.ndarray,
    mask: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    tbeta: float,
    td: float,
    layout: OCOutputLayout | None = None,
    model_cfg: dict[str, Any] | None = None,
    payload_quantities: list[dict[str, Any]] | None = None,
    events: list[EventRecord] | None = None,
    max_match_distance: float = 10.0,
    min_energy_ratio: float = 0.5,
    max_energy_ratio: float = 2.0,
    matching_algorithm: str = "hungarian",
    hungarian_energy_ratio_log_weight: float = 0.0,
    matching_reference: str = "aggregated",
) -> OCEvaluation:
    """
    Evaluate OC predictions on a padded batch.

    ``features`` must contain unnormalized ``x``, ``y``, ``z``, and ``energy``
    fields. They are an evaluation payload, independent of the model inputs.
    """
    layout = layout or (OCOutputLayout.from_config(model_cfg) if model_cfg is not None else None)
    layout = layout or OCOutputLayout.from_output_dim(int(preds.shape[-1]))
    outputs = split_oc_outputs(preds, layout)
    payload_features = decode_payload(outputs.payload, payload_quantities, features, feature_names)

    return _concat_evaluations(
        [
            evaluate_oc_event(
                beta=np.asarray(outputs.beta[event_idx]),
                cluster_coords=np.asarray(outputs.cluster_coords[event_idx]),
                payload_features=None
                if payload_features is None
                else {key: np.asarray(value[event_idx]) for key, value in payload_features.items()},
                hit_object_id=np.asarray(hit_object_id[event_idx]),
                mask=np.asarray(mask[event_idx], dtype=bool),
                features=np.asarray(features[event_idx]),
                feature_names=feature_names,
                tbeta=tbeta,
                td=td,
                event_idx=event_idx,
                event=None if events is None else events[event_idx],
                max_match_distance=max_match_distance,
                min_energy_ratio=min_energy_ratio,
                max_energy_ratio=max_energy_ratio,
                matching_algorithm=matching_algorithm,
                hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
                matching_reference=matching_reference,
            )
            for event_idx in range(preds.shape[0])
        ],
        payload=payload_features is not None,
    )


def evaluate_oc_event(
    *,
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    payload_features: dict[str, np.ndarray] | None = None,
    hit_object_id: np.ndarray,
    mask: np.ndarray | None,
    features: np.ndarray,
    feature_names: list[str],
    tbeta: float,
    td: float,
    event_idx: int = 0,
    event: EventRecord | None = None,
    max_match_distance: float = 10.0,
    min_energy_ratio: float = 0.5,
    max_energy_ratio: float = 2.0,
    matching_algorithm: str = "hungarian",
    hungarian_energy_ratio_log_weight: float = 0.0,
    matching_reference: str = "aggregated",
) -> OCEvaluation:
    """Evaluate one padded or unpadded event."""
    ctx = _EventEval.build(
        beta=beta,
        cluster_coords=cluster_coords,
        payload_features=payload_features,
        hit_object_id=hit_object_id,
        mask=mask,
        features=features,
        feature_names=feature_names,
        tbeta=tbeta,
        td=td,
        event_idx=event_idx,
        event=event,
        max_match_distance=max_match_distance,
        min_energy_ratio=min_energy_ratio,
        max_energy_ratio=max_energy_ratio,
        matching_algorithm=matching_algorithm,
        hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
        matching_reference=matching_reference,
    )
    truth_rows = _truth_rows(ctx)
    pred_rows = _pred_rows(ctx)
    seed_rows = _seed_rows(ctx)
    match_rows = _match_rows(ctx)
    _mark_matches(truth_rows, pred_rows, match_rows)

    payload = ctx.has_payload
    return OCEvaluation(
        events=pl.DataFrame([_event_row(ctx, len(match_rows))]),
        truth=_as_frame(truth_rows, "truth", payload=payload),
        predicted=_as_frame(pred_rows, "predicted", payload=payload),
        matches=_as_frame(match_rows, "matches", payload=payload),
        seeds=_as_frame(seed_rows, "seeds", payload=payload),
    )


def _truth_rows(ctx: _EventEval) -> list[dict[str, Any]]:
    rows = []
    nearest = nearest_truth_distance(ctx.truth_reco)
    for truth_id_np in ctx.truth_ids:
        truth_id = int(truth_id_np)
        truth_mask = ctx.labels == truth_id
        props = truth_object_props(ctx.event, truth_id)
        centroid = ctx.truth_reco[truth_id]
        rows.append(
            {
                "event_idx": ctx.event_idx,
                "object_id": truth_id,
                "truth_energy": _first(
                    props.get("impact_energy"), float(ctx.energy[truth_mask].sum())
                ),
                "truth_sum_energy": _first(
                    props.get("sum_energy"), float(ctx.energy[truth_mask].sum())
                ),
                "truth_et": centroid["et"],
                "truth_pt": props.get("impact_pt"),
                "truth_eta": _first(props.get("impact_eta"), centroid["eta"]),
                "truth_phi": _first(props.get("impact_phi"), centroid["phi"]),
                **prefixed_centroid("truth_centroid", centroid),
                "n_hits_truth": int(truth_mask.sum()),
                "truth_object_n_hits": props.get("n_hits"),
                "event_truth_multiplicity": len(ctx.truth_ids),
                "beta_max": float(ctx.beta[truth_mask].max()),
                "seed_found": bool(np.any(ctx.beta[truth_mask] > ctx.tbeta)),
                "nearest_truth_dist": nearest.get(truth_id),
                "matched": False,
            }
        )
    return rows


def _pred_rows(ctx: _EventEval) -> list[dict[str, Any]]:
    rows = []
    for seed_index_np in ctx.seed_indices:
        seed_index = int(seed_index_np)
        pred_id = ctx.pred_ids[seed_index]
        cluster_mask = ctx.clustering == seed_index
        dominant_truth_id, dominant_truth_hits = _dominant_positive_label(ctx.labels[cluster_mask])
        reco = ctx.pred_reco[pred_id + 1]
        row = {
            "event_idx": ctx.event_idx,
            "cluster_id_pred": pred_id,
            "seed_hit_idx": seed_index,
            "beta_seed": float(ctx.beta[seed_index]),
            "seed_truth_id": int(ctx.labels[seed_index]),
            "n_hits_pred": int(cluster_mask.sum()),
            "energy_pred": reco["energy"],
            "sum_et_reco": reco["et"],
            **prefixed_centroid("centroid", reco, suffix="reco"),
            "dominant_truth_id": dominant_truth_id,
            "dominant_truth_hit_fraction": safe_ratio(dominant_truth_hits, int(cluster_mask.sum())),
            "matched": False,
            "fake": True,
            **payload_pred_columns(ctx.payload_features, seed_index),
        }
        rows.append(row)
    return rows


def _seed_rows(ctx: _EventEval) -> list[dict[str, Any]]:
    rows = []
    matched_truth_ids: set[int] = set()
    for rank, seed_index_np in enumerate(ctx.seed_indices, start=1):
        seed_index = int(seed_index_np)
        truth_id = int(ctx.labels[seed_index])
        seed_class = (
            "noise_fake"
            if truth_id == 0
            else "duplicate_fake"
            if truth_id in matched_truth_ids
            else "matched_seed"
        )
        if seed_class == "matched_seed":
            matched_truth_ids.add(truth_id)
        rows.append(
            {
                "event_idx": ctx.event_idx,
                "seed_hit_idx": seed_index,
                "seed_rank": rank,
                "beta_seed": float(ctx.beta[seed_index]),
                "seed_truth_id": truth_id,
                "seed_class": seed_class,
            }
        )
    return rows


def _match_rows(ctx: _EventEval) -> list[dict[str, Any]]:
    if len(ctx.truth_ids) == 0 or len(ctx.seed_indices) == 0:
        return []

    truth_energy = np.asarray(
        [ctx.energy[ctx.labels == truth_id].sum() for truth_id in ctx.truth_ids]
    )
    pred_energy = np.asarray(
        [ctx.energy[ctx.clustering == seed_index].sum() for seed_index in ctx.seed_indices]
    )
    centroid_distance, energy_ratio = _matching_matrices(ctx, truth_energy, pred_energy)
    valid_match = match_candidates(
        centroid_distance,
        energy_ratio,
        max_match_distance=ctx.max_match_distance,
        min_energy_ratio=ctx.min_energy_ratio,
        max_energy_ratio=ctx.max_energy_ratio,
    )
    if not np.any(valid_match):
        return []

    intersection, intersection_hits = _overlap_matrices(ctx)
    rows = []
    for truth_pos, pred_pos in matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=ctx.max_match_distance,
        algorithm=ctx.matching_algorithm,
        hungarian_energy_ratio_log_weight=ctx.hungarian_energy_ratio_log_weight,
    ):
        truth_id = int(ctx.truth_ids[truth_pos])
        seed_index = int(ctx.seed_indices[pred_pos])
        pred_id = ctx.pred_ids[seed_index]
        truth_mask = ctx.labels == truth_id
        cluster_mask = ctx.clustering == seed_index
        truth_centroid = ctx.truth_reco[truth_id]
        pred_centroid = ctx.pred_reco[pred_id + 1]
        props = truth_object_props(ctx.event, truth_id)
        truth_ref_energy = _first(props.get("impact_energy"), float(truth_energy[truth_pos]))
        truth_eta = _first(props.get("impact_eta"), truth_centroid["eta"])
        truth_phi = _first(props.get("impact_phi"), truth_centroid["phi"])
        dphi = delta_phi(pred_centroid["phi"], truth_phi)

        row = {
            "event_idx": ctx.event_idx,
            "object_id": truth_id,
            "cluster_id_pred": pred_id,
            "seed_hit_idx": seed_index,
            "matching_reference": ctx.matching_reference,
            "beta_seed": float(ctx.beta[seed_index]),
            "centroid_distance": float(centroid_distance[truth_pos, pred_pos]),
            "energy_ratio": float(energy_ratio[truth_pos, pred_pos]),
            "purity": safe_ratio(intersection_hits[truth_pos, pred_pos], cluster_mask.sum()),
            "completeness": safe_ratio(intersection_hits[truth_pos, pred_pos], truth_mask.sum()),
            "energy_assigned_fraction": safe_ratio(
                intersection[truth_pos, pred_pos], pred_energy[pred_pos]
            ),
            "energy_recovered_fraction": safe_ratio(
                intersection[truth_pos, pred_pos], truth_energy[truth_pos]
            ),
            "truth_energy": truth_ref_energy,
            "truth_sum_energy": float(truth_energy[truth_pos]),
            "truth_et": truth_centroid["et"],
            "energy_pred": pred_centroid["energy"],
            "sum_et_reco": pred_centroid["et"],
            "sum_energy_response": safe_ratio(pred_centroid["energy"], truth_energy[truth_pos]),
            "relative_sum_energy_residual": safe_ratio(
                pred_centroid["energy"] - truth_energy[truth_pos], truth_energy[truth_pos]
            ),
            **residual_block(pred_centroid, truth_centroid, truth_ref_energy, prefix=""),
            **prefixed_centroid("truth_centroid", truth_centroid),
            **prefixed_centroid("centroid", pred_centroid, suffix="reco"),
            "delta_x": none_subtract(pred_centroid["x"], truth_centroid["x"]),
            "delta_y": none_subtract(pred_centroid["y"], truth_centroid["y"]),
            "delta_z": none_subtract(pred_centroid["z"], truth_centroid["z"]),
            "delta_eta": none_subtract(pred_centroid["eta"], truth_eta),
            "delta_phi": dphi,
            "delta_r": delta_r(pred_centroid["eta"], truth_eta, dphi),
        }
        row.update(
            payload_match_columns(
                ctx.payload_features, seed_index, truth_ref_energy, truth_centroid
            )
        )
        rows.append(row)
    return rows


def _event_row(ctx: _EventEval, n_matches: int) -> dict[str, Any]:
    return {
        "event_idx": ctx.event_idx,
        "n_hits": ctx.n_valid_hits,
        "n_truth": len(ctx.truth_ids),
        "n_pred": len(ctx.seed_indices),
        "n_matched": n_matches,
        "signal_fraction": float(np.mean(ctx.labels > 0)) if len(ctx.labels) else 0.0,
        "tbeta": float(ctx.tbeta),
        "td": float(ctx.td),
        "max_match_distance": float(ctx.max_match_distance),
        "min_energy_ratio": float(ctx.min_energy_ratio),
        "max_energy_ratio": float(ctx.max_energy_ratio),
        "matching_algorithm": ctx.matching_algorithm,
        "hungarian_energy_ratio_log_weight": float(ctx.hungarian_energy_ratio_log_weight),
    }


def _matching_matrices(
    ctx: _EventEval,
    truth_energy: np.ndarray,
    pred_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if ctx.matching_reference == "aggregated":
        centroid_distance = cdist(_truth_xyz(ctx), _pred_xyz(ctx))
        centroid_distance[~np.isfinite(centroid_distance)] = np.inf
        return centroid_distance, energy_ratio_matrix(pred_energy, truth_energy)

    pred_payloads = [payload_seed_reco(ctx.payload_features, int(s)) for s in ctx.seed_indices]
    centroid_distance = np.asarray(
        [
            [payload_position_distance(ctx.truth_reco[int(t)], pred) for pred in pred_payloads]
            for t in ctx.truth_ids
        ],
        dtype=np.float64,
    ).reshape(len(ctx.truth_ids), len(pred_payloads))
    truth_et = np.asarray(
        [none_to_nan(ctx.truth_reco[int(t)]["et"]) for t in ctx.truth_ids], dtype=np.float64
    )
    pred_et = np.asarray([none_to_nan(p.get("et")) for p in pred_payloads], dtype=np.float64)
    return centroid_distance, energy_ratio_matrix(pred_et, truth_et)


def _truth_xyz(ctx: _EventEval) -> np.ndarray:
    return np.asarray(
        [
            [none_to_nan(ctx.truth_reco[int(obj_id)][axis]) for axis in ("x", "y", "z")]
            for obj_id in ctx.truth_ids
        ],
        dtype=np.float64,
    )


def _pred_xyz(ctx: _EventEval) -> np.ndarray:
    return np.asarray(
        [
            [
                none_to_nan(ctx.pred_reco[ctx.pred_ids[int(seed_index)] + 1][axis])
                for axis in ("x", "y", "z")
            ]
            for seed_index in ctx.seed_indices
        ],
        dtype=np.float64,
    )


def _overlap_matrices(ctx: _EventEval) -> tuple[np.ndarray, np.ndarray]:
    intersection = np.zeros((len(ctx.truth_ids), len(ctx.seed_indices)), dtype=np.float64)
    intersection_hits = np.zeros_like(intersection)
    for truth_pos, truth_id in enumerate(ctx.truth_ids):
        truth_mask = ctx.labels == truth_id
        for pred_pos, seed_index in enumerate(ctx.seed_indices):
            overlap = truth_mask & (ctx.clustering == seed_index)
            intersection[truth_pos, pred_pos] = ctx.energy[overlap].sum()
            intersection_hits[truth_pos, pred_pos] = overlap.sum()
    return intersection, intersection_hits


def _mark_matches(
    truth_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
) -> None:
    matched_truth = {row["object_id"] for row in match_rows}
    matched_pred = {row["cluster_id_pred"] for row in match_rows}
    for row in truth_rows:
        row["matched"] = row["object_id"] in matched_truth
    for row in pred_rows:
        row["matched"] = row["cluster_id_pred"] in matched_pred
        row["fake"] = not row["matched"]


def _dominant_positive_label(labels: np.ndarray) -> tuple[int | None, int]:
    positive = labels[labels > 0]
    if len(positive) == 0:
        return None, 0
    ids, counts = np.unique(positive, return_counts=True)
    best = int(np.argmax(counts))
    return int(ids[best]), int(counts[best])


def _as_frame(rows: list[dict[str, Any]], attribute: str, *, payload: bool) -> pl.DataFrame:
    return pl.DataFrame(rows) if rows else empty_frame(attribute, payload=payload)


def _concat_evaluations(evaluations: list[OCEvaluation], *, payload: bool) -> OCEvaluation:
    def concat_attr(name: str) -> pl.DataFrame:
        frames = [getattr(evaluation, name) for evaluation in evaluations]
        non_empty = [frame for frame in frames if not frame.is_empty()]
        if non_empty:
            return pl.concat(non_empty, how="diagonal_relaxed")
        if name == "events":
            return frames[0] if frames else pl.DataFrame()
        return empty_frame(name, payload=payload)

    return OCEvaluation(
        events=concat_attr("events"),
        truth=concat_attr("truth"),
        predicted=concat_attr("predicted"),
        matches=concat_attr("matches"),
        seeds=concat_attr("seeds"),
    )


def _required_feature(features: np.ndarray, feature_names: list[str], name: str) -> np.ndarray:
    if name not in feature_names:
        raise ValueError(
            f"OC evaluation requires unnormalized hit feature {name!r}. "
            f"Available evaluation features: {', '.join(feature_names)}"
        )
    return np.asarray(features[:, feature_names.index(name)], dtype=np.float64)


def _count_value(frame: pl.DataFrame, column: str, value: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int((frame[column] == value).sum())


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)
