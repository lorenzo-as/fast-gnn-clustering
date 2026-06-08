"""Object Condensation clustering metrics for padded calorimeter batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from statsmodels.stats.proportion import proportion_confint

from fastgnn.data.base import EventRecord
from fastgnn.data.object_properties import compute_object_properties
from fastgnn.geometry import xyz_to_eta_phi
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
            "seed_efficiency": _safe_ratio(n_matched, n_truth),
            "duplicate_rate": _safe_ratio(n_duplicate, n_seeds),
            "noise_seed_fake_rate": _safe_ratio(n_noise, n_seeds),
            "missed_truth": float(max(n_truth - n_matched, 0)),
        }

    def matching_summary(self) -> dict[str, float]:
        n_truth = int(self.events["n_truth"].sum()) if "n_truth" in self.events.columns else 0
        n_pred = int(self.events["n_pred"].sum()) if "n_pred" in self.events.columns else 0
        n_matches = len(self.matches)
        return {
            "efficiency": _safe_ratio(n_matches, n_truth),
            "fake_rate": _safe_ratio(n_pred - n_matches, n_pred),
            "purity": _safe_ratio(n_matches, n_pred),
            "miss_rate": _safe_ratio(n_truth - n_matches, n_truth),
        }


@dataclass(frozen=True)
class _EventEval:
    event_idx: int
    event: EventRecord | None
    labels: np.ndarray
    beta: np.ndarray
    coords: np.ndarray
    features: dict[str, np.ndarray]
    regressions: dict[str, np.ndarray]
    tbeta: float
    td: float
    max_match_distance: float
    min_energy_ratio: float
    max_energy_ratio: float
    matching_algorithm: str
    hungarian_energy_ratio_log_weight: float
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

    @classmethod
    def build(
        cls,
        *,
        beta: np.ndarray,
        cluster_coords: np.ndarray,
        hit_object_id: np.ndarray,
        mask: np.ndarray | None,
        features: np.ndarray,
        feature_names: list[str],
        tbeta: float,
        td: float,
        regressions: dict[str, np.ndarray] | None,
        event_idx: int,
        event: EventRecord | None,
        max_match_distance: float,
        min_energy_ratio: float,
        max_energy_ratio: float,
        matching_algorithm: str,
        hungarian_energy_ratio_log_weight: float,
    ) -> _EventEval:
        valid = np.ones(len(hit_object_id), dtype=bool) if mask is None else mask.astype(bool)
        labels = np.asarray(hit_object_id[valid], dtype=np.int32)
        beta_v = np.asarray(beta[valid], dtype=np.float64)
        coords_v = np.asarray(cluster_coords[valid], dtype=np.float64)
        features_v = np.asarray(features[valid], dtype=np.float64)
        regressions_v = {
            name: np.asarray(values[valid], dtype=np.float64)
            for name, values in (regressions or {}).items()
        }
        physical_hits = {
            name: _required_feature(features_v, feature_names, name)
            for name in ("x", "y", "z", "energy")
        }

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
            features=physical_hits,
            regressions=regressions_v,
            tbeta=tbeta,
            td=td,
            max_match_distance=max_match_distance,
            min_energy_ratio=min_energy_ratio,
            max_energy_ratio=max_energy_ratio,
            matching_algorithm=_validate_matching_algorithm(matching_algorithm),
            hungarian_energy_ratio_log_weight=float(hungarian_energy_ratio_log_weight),
            n_valid_hits=int(valid.sum()),
            clustering=clustering,
            truth_ids=truth_ids,
            seed_indices=seed_indices,
            pred_ids=pred_ids,
            truth_reco=_reco_properties_by_assignment(labels, truth_ids, physical_hits),
            pred_reco=_reco_properties_by_assignment(
                pred_assignments,
                np.arange(1, len(seed_indices) + 1, dtype=np.int32),
                physical_hits,
            ),
        )


@dataclass(frozen=True)
class _ThresholdScanEvent:
    labels: np.ndarray
    beta: np.ndarray
    distances: np.ndarray
    energy: np.ndarray
    et: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    truth_ids: np.ndarray
    truth_energy: np.ndarray
    truth_et: np.ndarray
    truth_xyz: np.ndarray

    @property
    def n_truth(self) -> int:
        return len(self.truth_ids)

    @property
    def total_truth_et(self) -> float:
        return float(self.truth_et.sum())


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
    events: list[EventRecord] | None = None,
    max_match_distance: float = 10.0,
    min_energy_ratio: float = 0.5,
    max_energy_ratio: float = 2.0,
    matching_algorithm: str = "hungarian",
    hungarian_energy_ratio_log_weight: float = 0.0,
) -> OCEvaluation:
    """
    Evaluate OC predictions on a padded batch.

    ``features`` must contain unnormalized ``x``, ``y``, ``z``, and ``energy``
    fields. They are an evaluation payload, independent of the model inputs.
    """
    layout = layout or (OCOutputLayout.from_config(model_cfg) if model_cfg is not None else None)
    layout = layout or OCOutputLayout.from_output_dim(int(preds.shape[-1]))
    outputs = split_oc_outputs(preds, layout)

    return _concat_evaluations(
        [
            evaluate_oc_event(
                beta=np.asarray(outputs.beta[event_idx]),
                cluster_coords=np.asarray(outputs.cluster_coords[event_idx]),
                regressions={
                    name: np.asarray(values[event_idx])
                    for name, values in outputs.regressions.items()
                },
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
            )
            for event_idx in range(preds.shape[0])
        ]
    )


def evaluate_oc_event(
    *,
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    hit_object_id: np.ndarray,
    mask: np.ndarray | None,
    features: np.ndarray,
    feature_names: list[str],
    tbeta: float,
    td: float,
    regressions: dict[str, np.ndarray] | None = None,
    event_idx: int = 0,
    event: EventRecord | None = None,
    max_match_distance: float = 10.0,
    min_energy_ratio: float = 0.5,
    max_energy_ratio: float = 2.0,
    matching_algorithm: str = "hungarian",
    hungarian_energy_ratio_log_weight: float = 0.0,
) -> OCEvaluation:
    """Evaluate one padded or unpadded event."""
    ctx = _EventEval.build(
        beta=beta,
        cluster_coords=cluster_coords,
        hit_object_id=hit_object_id,
        mask=mask,
        features=features,
        feature_names=feature_names,
        tbeta=tbeta,
        td=td,
        regressions=regressions,
        event_idx=event_idx,
        event=event,
        max_match_distance=max_match_distance,
        min_energy_ratio=min_energy_ratio,
        max_energy_ratio=max_energy_ratio,
        matching_algorithm=matching_algorithm,
        hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
    )
    truth_rows = _truth_rows(ctx)
    pred_rows = _pred_rows(ctx)
    seed_rows = _seed_rows(ctx)
    match_rows = _match_rows(ctx)
    _mark_matches(truth_rows, pred_rows, match_rows)

    return OCEvaluation(
        events=pl.DataFrame([_event_row(ctx, len(match_rows))]),
        truth=pl.DataFrame(truth_rows),
        predicted=pl.DataFrame(pred_rows),
        matches=pl.DataFrame(match_rows),
        seeds=pl.DataFrame(seed_rows),
    )


def count_truth_objects(hit_object_id: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(
        [
            len(
                np.unique(
                    _event_labels(hit_object_id, mask, i)[_event_labels(hit_object_id, mask, i) > 0]
                )
            )
            for i in range(len(hit_object_id))
        ],
        dtype=np.int32,
    )


def count_clusters_from_labels(clustering: np.ndarray) -> int:
    return int(np.sum(np.unique(clustering) >= 0))


def count_pred_objects(
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    tbeta: float,
    td: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    counts = []
    for event_beta, event_distances in zip(
        *_masked_beta_distances(beta, cluster_coords, mask), strict=True
    ):
        counts.append(
            0
            if not np.any(event_beta > tbeta)
            else count_clusters_from_labels(
                _get_clustering_precomputed(event_beta, event_distances, tbeta, td)
            )
        )
    return np.asarray(counts, dtype=np.int32)


def grid_search_thresholds(
    *,
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    hit_object_id: np.ndarray,
    mask: np.ndarray | None = None,
    tbeta_values: np.ndarray | None = None,
    td_values: np.ndarray | None = None,
    objective: str = "count_median",
    features: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    max_match_distance: float = 50.0,
    min_energy_ratio: float = 0.01,
    max_energy_ratio: float = 10.0,
    matching_algorithm: str = "hungarian",
    hungarian_energy_ratio_log_weight: float = 0.0,
    f1_epsilon: float = 1e-12,
    progress: bool | Any = False,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Grid search OC thresholds for notebook operating-point scans.

    ``objective="count_median"`` keeps the legacy median count-based selector.
    ``objective="count_mean"`` selects thresholds by mean absolute count error.
    ``objective="matched_f1"`` scores thresholds with matched-cluster object and
    energy-weighted F1 terms, using compact per-threshold summaries rather than
    materializing full evaluation tables.
    Pass ``progress=True`` for a tqdm progress bar, or pass a callable that wraps
    the tbeta iterator, e.g. ``lambda values: mo.status.progress_bar(values)``.
    """
    tbeta_values = np.linspace(0.05, 0.9, 20) if tbeta_values is None else tbeta_values
    td_values = np.linspace(0.05, 1.0, 20) if td_values is None else td_values
    objective = str(objective).lower()
    if objective not in {"count_median", "count_mean", "matched_f1"}:
        raise ValueError("objective must be 'count_median', 'count_mean', or 'matched_f1'")

    n_truth = count_truth_objects(hit_object_id, mask=mask)
    masked_beta, masked_distances = _masked_beta_distances(beta, cluster_coords, mask)
    if objective == "matched_f1":
        if features is None or feature_names is None:
            raise ValueError(
                "grid_search_thresholds with objective='matched_f1' requires "
                "unnormalized evaluation features and feature_names"
            )
        scan_events = _threshold_scan_events(
            beta=beta,
            cluster_coords=cluster_coords,
            hit_object_id=hit_object_id,
            mask=mask,
            features=features,
            feature_names=feature_names,
        )
        rows = []
        for tbeta in _progress_iterator(tbeta_values, progress, title="Scanning OC thresholds"):
            seed_orders = [
                _seed_order_for_threshold(event.beta, float(tbeta)) for event in scan_events
            ]
            rows.extend(
                _matched_f1_scan_row(
                    float(tbeta),
                    float(td),
                    scan_events,
                    seed_orders,
                    n_truth,
                    max_match_distance=float(max_match_distance),
                    min_energy_ratio=float(min_energy_ratio),
                    max_energy_ratio=float(max_energy_ratio),
                    matching_algorithm=matching_algorithm,
                    hungarian_energy_ratio_log_weight=float(hungarian_energy_ratio_log_weight),
                    f1_epsilon=float(f1_epsilon),
                )
                for td in td_values
            )

        table = pl.DataFrame(rows)
        best = table.sort(
            by=["score", "f1_E", "f1_obj", "median_abs_diff"],
            descending=[True, True, True, False],
        ).row(0, named=True)
        return best, table

    rows = []
    for tbeta in _progress_iterator(tbeta_values, progress, title="Scanning OC thresholds"):
        has_seed = [np.any(event_beta > tbeta) for event_beta in masked_beta]
        for td in td_values:
            counts = np.asarray(
                [
                    count_clusters_from_labels(
                        _get_clustering_precomputed(event_beta, event_distances, tbeta, td)
                    )
                    if has_seed[idx]
                    else 0
                    for idx, (event_beta, event_distances) in enumerate(
                        zip(masked_beta, masked_distances, strict=True)
                    )
                ],
                dtype=np.int32,
            )
            rows.append(_count_scan_row(tbeta, td, counts, n_truth))

    table = pl.DataFrame(rows)
    if objective == "count_mean":
        sort_by = ["mean_abs_diff", "median_abs_diff", "frac_exact"]
    else:
        sort_by = ["median_abs_diff", "mean_abs_diff", "frac_exact"]
    best = table.sort(
        by=sort_by,
        descending=[False, False, True],
    ).row(0, named=True)
    return best, table


def binned_efficiency(
    truth: pl.DataFrame,
    value_col: str,
    bins: np.ndarray,
    *,
    matched_col: str = "matched",
) -> pl.DataFrame:
    return _binned_rate(truth, value_col, bins, numerator_col=matched_col, rate_name="efficiency")


def binned_fake_rate(
    predicted: pl.DataFrame,
    value_col: str,
    bins: np.ndarray,
    *,
    fake_col: str = "fake",
) -> pl.DataFrame:
    return _binned_rate(predicted, value_col, bins, numerator_col=fake_col, rate_name="fake_rate")


def _truth_rows(ctx: _EventEval) -> list[dict[str, Any]]:
    rows = []
    nearest = _nearest_truth_distance(ctx.truth_reco)
    for truth_id_np in ctx.truth_ids:
        truth_id = int(truth_id_np)
        truth_mask = ctx.labels == truth_id
        props = _truth_object_props(ctx.event, truth_id)
        centroid = ctx.truth_reco[truth_id]
        rows.append(
            {
                "event_idx": ctx.event_idx,
                "object_id": truth_id,
                "truth_energy": _first_not_none(
                    props.get("impact_energy"), float(ctx.energy[truth_mask].sum())
                ),
                "truth_sum_energy": _first_not_none(
                    props.get("sum_energy"), float(ctx.energy[truth_mask].sum())
                ),
                "truth_et": centroid["et"],
                "truth_pt": props.get("impact_pt"),
                "truth_eta": _first_not_none(props.get("impact_eta"), centroid["eta"]),
                "truth_phi": _first_not_none(props.get("impact_phi"), centroid["phi"]),
                **_prefixed_centroid("truth_centroid", centroid),
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
            **_prefixed_centroid("centroid", reco, suffix="reco"),
            "dominant_truth_id": dominant_truth_id,
            "dominant_truth_hit_fraction": _safe_ratio(
                dominant_truth_hits, int(cluster_mask.sum())
            ),
            "matched": False,
            "fake": True,
        }
        row.update(_model_reco_columns(ctx.regressions, seed_index))
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
        matched_truth_ids.add(truth_id) if seed_class == "matched_seed" else None
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
    centroid_distance = cdist(_xyz_array(ctx.truth_reco, ctx.truth_ids), _pred_xyz_array(ctx))
    centroid_distance[~np.isfinite(centroid_distance)] = np.inf
    energy_ratio = np.divide(
        pred_energy[None, :],
        truth_energy[:, None],
        out=np.full((len(ctx.truth_ids), len(ctx.seed_indices)), np.inf),
        where=truth_energy[:, None] > 0,
    )
    valid_match = (
        (centroid_distance <= ctx.max_match_distance)
        & (energy_ratio >= ctx.min_energy_ratio)
        & (energy_ratio <= ctx.max_energy_ratio)
    )
    if not np.any(valid_match):
        return []

    intersection, intersection_hits = _overlap_matrices(ctx)
    rows = []
    for truth_pos, pred_pos in _matched_positions(
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
        props = _truth_object_props(ctx.event, truth_id)
        truth_ref_energy = _first_not_none(
            props.get("impact_energy"), float(truth_energy[truth_pos])
        )
        truth_eta = _first_not_none(props.get("impact_eta"), truth_centroid["eta"])
        truth_phi = _first_not_none(props.get("impact_phi"), truth_centroid["phi"])
        dphi = _delta_phi(pred_centroid["phi"], truth_phi)
        centroid_dphi = _delta_phi(truth_centroid["phi"], pred_centroid["phi"])

        row = {
            "event_idx": ctx.event_idx,
            "object_id": truth_id,
            "cluster_id_pred": pred_id,
            "seed_hit_idx": seed_index,
            "beta_seed": float(ctx.beta[seed_index]),
            "centroid_distance": float(centroid_distance[truth_pos, pred_pos]),
            "energy_ratio": float(energy_ratio[truth_pos, pred_pos]),
            "purity": _safe_ratio(intersection_hits[truth_pos, pred_pos], cluster_mask.sum()),
            "completeness": _safe_ratio(intersection_hits[truth_pos, pred_pos], truth_mask.sum()),
            "energy_assigned_fraction": _safe_ratio(
                intersection[truth_pos, pred_pos], pred_energy[pred_pos]
            ),
            "energy_recovered_fraction": _safe_ratio(
                intersection[truth_pos, pred_pos], truth_energy[truth_pos]
            ),
            "truth_energy": truth_ref_energy,
            "truth_sum_energy": float(truth_energy[truth_pos]),
            "truth_et": truth_centroid["et"],
            "energy_pred": pred_centroid["energy"],
            "sum_et_reco": pred_centroid["et"],
            "sum_energy_response": _safe_ratio(pred_centroid["energy"], truth_energy[truth_pos]),
            "et_response": _safe_ratio(pred_centroid["et"], truth_centroid["et"]),
            "energy_response": _safe_ratio(pred_centroid["energy"], truth_ref_energy),
            "relative_energy_residual": _safe_ratio(
                pred_centroid["energy"] - truth_ref_energy, truth_ref_energy
            ),
            "relative_sum_energy_residual": _safe_ratio(
                pred_centroid["energy"] - truth_energy[truth_pos], truth_energy[truth_pos]
            ),
            "relative_et_residual": _safe_ratio(
                pred_centroid["et"] - truth_centroid["et"], truth_centroid["et"]
            ),
            **_prefixed_centroid("truth_centroid", truth_centroid),
            **_prefixed_centroid("centroid", pred_centroid, suffix="reco"),
            "delta_x": _none_subtract(pred_centroid["x"], truth_centroid["x"]),
            "delta_y": _none_subtract(pred_centroid["y"], truth_centroid["y"]),
            "delta_z": _none_subtract(pred_centroid["z"], truth_centroid["z"]),
            "delta_eta": _none_subtract(pred_centroid["eta"], truth_eta),
            "delta_phi": dphi,
            "eta_residual": _none_subtract(truth_centroid["eta"], pred_centroid["eta"]),
            "phi_residual": centroid_dphi,
            "z_residual": _none_subtract(truth_centroid["z"], pred_centroid["z"]),
            "relative_eta_residual": _safe_none_ratio(
                _none_subtract(truth_centroid["eta"], pred_centroid["eta"]), truth_centroid["eta"]
            ),
            "relative_phi_residual": _safe_none_ratio(centroid_dphi, truth_centroid["phi"]),
            "relative_z_residual": _safe_none_ratio(
                _none_subtract(truth_centroid["z"], pred_centroid["z"]), truth_centroid["z"]
            ),
            "delta_r": None
            if pred_centroid["eta"] is None or truth_eta is None or dphi is None
            else float(np.hypot(pred_centroid["eta"] - truth_eta, dphi)),
        }
        row.update(_model_reco_columns(ctx.regressions, seed_index))
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


def _prefixed_centroid(
    prefix: str, centroid: dict[str, float | None], *, suffix: str | None = None
) -> dict[str, float | None]:
    keys = ("x", "y", "z", "eta", "phi")
    return {f"{prefix}_{key}" + (f"_{suffix}" if suffix else ""): centroid[key] for key in keys}


def _xyz_array(centroids: dict[int, dict[str, float | None]], ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [_none_to_nan(centroids[int(obj_id)][axis]) for axis in ("x", "y", "z")]
            for obj_id in ids
        ],
        dtype=np.float64,
    )


def _pred_xyz_array(ctx: _EventEval) -> np.ndarray:
    return np.asarray(
        [
            [
                _none_to_nan(ctx.pred_reco[ctx.pred_ids[int(seed_index)] + 1][axis])
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


def _matched_positions(
    centroid_distance: np.ndarray,
    energy_ratio: np.ndarray,
    valid_match: np.ndarray,
    *,
    max_match_distance: float,
    algorithm: str,
    hungarian_energy_ratio_log_weight: float,
) -> list[tuple[int, int]]:
    algorithm = _validate_matching_algorithm(algorithm)
    if algorithm == "greedy":
        return _greedy_matched_positions(centroid_distance, energy_ratio, valid_match)
    return _hungarian_matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=max_match_distance,
        energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
    )


def _hungarian_matched_positions(
    centroid_distance: np.ndarray,
    energy_ratio: np.ndarray,
    valid_match: np.ndarray,
    *,
    max_match_distance: float,
    energy_ratio_log_weight: float,
) -> list[tuple[int, int]]:
    finite_costs = centroid_distance[np.isfinite(centroid_distance)]
    large_cost = (
        1.0 if len(finite_costs) == 0 else float(finite_costs.max() + max_match_distance + 1.0)
    )
    cost = centroid_distance.copy()
    if energy_ratio_log_weight != 0:
        cost = cost + float(energy_ratio_log_weight) * np.abs(np.log(energy_ratio))
    row_ind, col_ind = linear_sum_assignment(np.where(valid_match, cost, large_cost))
    return [(int(t), int(p)) for t, p in zip(row_ind, col_ind, strict=True) if valid_match[t, p]]


def _greedy_matched_positions(
    centroid_distance: np.ndarray,
    energy_ratio: np.ndarray,
    valid_match: np.ndarray,
) -> list[tuple[int, int]]:
    pred_best: list[tuple[int, int]] = []
    for pred_pos in range(valid_match.shape[1]):
        truth_candidates = np.flatnonzero(valid_match[:, pred_pos])
        if len(truth_candidates) == 0:
            continue
        distances = centroid_distance[truth_candidates, pred_pos]
        truth_pos = int(truth_candidates[np.argmin(distances)])
        pred_best.append((truth_pos, pred_pos))

    matches: list[tuple[int, int]] = []
    for truth_pos in sorted({truth_pos for truth_pos, _ in pred_best}):
        pred_candidates = [
            pred_pos for candidate_truth, pred_pos in pred_best if candidate_truth == truth_pos
        ]
        ratios = energy_ratio[truth_pos, pred_candidates]
        ratio_distance = np.abs(np.log(ratios))
        pred_pos = int(pred_candidates[int(np.argmin(ratio_distance))])
        matches.append((truth_pos, pred_pos))
    return matches


def _validate_matching_algorithm(algorithm: str) -> str:
    algorithm = str(algorithm).lower()
    if algorithm not in {"hungarian", "greedy"}:
        raise ValueError("matching_algorithm must be 'hungarian' or 'greedy'")
    return algorithm


def _event_labels(hit_object_id: np.ndarray, mask: np.ndarray | None, event_idx: int) -> np.ndarray:
    labels = np.asarray(hit_object_id[event_idx])
    return labels if mask is None else labels[np.asarray(mask[event_idx], dtype=bool)]


def _count_scan_row(
    tbeta: float, td: float, counts: np.ndarray, n_truth: np.ndarray
) -> dict[str, float]:
    diff = counts - n_truth
    return {
        "tbeta": float(tbeta),
        "td": float(td),
        "median_abs_diff": float(np.median(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "frac_exact": float(np.mean(diff == 0)),
        "frac_within_1": float(np.mean(np.abs(diff) <= 1)),
        "mean_pred": float(np.mean(counts)),
        "mean_truth": float(np.mean(n_truth)),
    }


def _threshold_scan_events(
    *,
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    hit_object_id: np.ndarray,
    mask: np.ndarray | None,
    features: np.ndarray,
    feature_names: list[str],
) -> list[_ThresholdScanEvent]:
    events = []
    for event_idx in range(len(beta)):
        valid = (
            np.ones(len(hit_object_id[event_idx]), dtype=bool)
            if mask is None
            else np.asarray(mask[event_idx], dtype=bool)
        )
        labels = np.asarray(hit_object_id[event_idx][valid], dtype=np.int32)
        beta_v = np.asarray(beta[event_idx][valid], dtype=np.float64)
        coords_v = np.asarray(cluster_coords[event_idx][valid], dtype=np.float64)
        features_v = np.asarray(features[event_idx][valid], dtype=np.float64)
        physical_hits = {
            name: _required_feature(features_v, feature_names, name)
            for name in ("x", "y", "z", "energy")
        }
        eta, _ = xyz_to_eta_phi(
            physical_hits["x"],
            physical_hits["y"],
            physical_hits["z"],
        )
        et = np.divide(
            physical_hits["energy"],
            np.cosh(eta),
            out=np.zeros_like(physical_hits["energy"], dtype=np.float64),
            where=np.isfinite(eta),
        )
        truth_ids = np.unique(labels[labels > 0]).astype(np.int32)
        truth_reco = _reco_properties_by_assignment(labels, truth_ids, physical_hits)
        events.append(
            _ThresholdScanEvent(
                labels=labels,
                beta=beta_v,
                distances=cdist(coords_v, coords_v),
                energy=physical_hits["energy"],
                et=et,
                x=physical_hits["x"],
                y=physical_hits["y"],
                z=physical_hits["z"],
                truth_ids=truth_ids,
                truth_energy=np.asarray(
                    [truth_reco[int(truth_id)]["energy"] for truth_id in truth_ids],
                    dtype=np.float64,
                ),
                truth_et=np.asarray(
                    [truth_reco[int(truth_id)]["et"] for truth_id in truth_ids],
                    dtype=np.float64,
                ),
                truth_xyz=np.asarray(
                    [
                        [_none_to_nan(truth_reco[int(truth_id)][axis]) for axis in ("x", "y", "z")]
                        for truth_id in truth_ids
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return events


def _matched_f1_scan_row(
    tbeta: float,
    td: float,
    events: list[_ThresholdScanEvent],
    seed_orders: list[np.ndarray],
    n_truth: np.ndarray,
    *,
    max_match_distance: float,
    min_energy_ratio: float,
    max_energy_ratio: float,
    matching_algorithm: str,
    hungarian_energy_ratio_log_weight: float,
    f1_epsilon: float,
) -> dict[str, float]:
    counts = np.zeros(len(events), dtype=np.int32)
    n_matched = 0
    n_pred_total = 0
    matched_truth_et = 0.0
    matched_pred_et = 0.0
    total_truth_et = 0.0
    total_pred_et = 0.0

    for event_idx, (event, seed_order) in enumerate(zip(events, seed_orders, strict=True)):
        total_truth_et += event.total_truth_et
        if len(seed_order) == 0:
            continue

        clustering = _get_clustering_from_seed_order(event.distances, seed_order, td)
        seed_indices = np.unique(clustering[clustering >= 0]).astype(np.int32)
        seed_indices = seed_indices[np.argsort(-event.beta[seed_indices])]
        counts[event_idx] = len(seed_indices)
        n_pred_total += len(seed_indices)
        pred = _pred_summary_for_clustering(event, clustering, seed_indices)
        total_pred_et += float(pred["et"].sum())
        if event.n_truth == 0:
            continue
        match_positions = _scan_matched_positions(
            event,
            pred,
            max_match_distance=max_match_distance,
            min_energy_ratio=min_energy_ratio,
            max_energy_ratio=max_energy_ratio,
            matching_algorithm=matching_algorithm,
            hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
        )
        n_matched += len(match_positions)
        for truth_pos, pred_pos in match_positions:
            matched_truth_et += float(event.truth_et[truth_pos])
            matched_pred_et += float(pred["et"][pred_pos])

    n_truth_total = int(n_truth.sum())
    epsilon_obj = _safe_ratio(n_matched, n_truth_total)
    p_obj = _safe_ratio(n_matched, n_pred_total)
    epsilon_e = _safe_ratio(matched_truth_et, total_truth_et)
    p_e = _safe_ratio(matched_pred_et, total_pred_et)
    f1_obj = _f1(epsilon_obj, p_obj, f1_epsilon)
    f1_e = _f1(epsilon_e, p_e, f1_epsilon)
    row = _count_scan_row(tbeta, td, counts, n_truth)
    row.update(
        {
            "objective": "matched_f1",
            "n_truth": float(n_truth_total),
            "n_pred": float(n_pred_total),
            "n_matched": float(n_matched),
            "total_truth_et": float(total_truth_et),
            "total_pred_et": float(total_pred_et),
            "matched_truth_et": float(matched_truth_et),
            "matched_pred_et": float(matched_pred_et),
            "epsilon_obj": float(epsilon_obj),
            "p_obj": float(p_obj),
            "epsilon_E": float(epsilon_e),
            "p_E": float(p_e),
            "f1_obj": float(f1_obj),
            "f1_E": float(f1_e),
            "score": float(f1_e + 0.5 * f1_obj),
        }
    )
    return row


def _seed_order_for_threshold(beta: np.ndarray, tbeta: float) -> np.ndarray:
    seed_order = np.nonzero(beta > tbeta)[0]
    return seed_order[np.argsort(-beta[seed_order])].astype(np.int32)


def _get_clustering_from_seed_order(
    distances: np.ndarray, seed_order: np.ndarray, td: float
) -> np.ndarray:
    clustering = -1 * np.ones(distances.shape[0], dtype=np.int32)
    unassigned = np.ones(distances.shape[0], dtype=bool)
    for seed_index in seed_order:
        if unassigned[seed_index]:
            assign_mask = unassigned & (distances[seed_index] < td)
            clustering[assign_mask] = int(seed_index)
            unassigned[assign_mask] = False
    return clustering


def _pred_summary_for_clustering(
    event: _ThresholdScanEvent, clustering: np.ndarray, seed_indices: np.ndarray
) -> dict[str, np.ndarray]:
    n_pred = len(seed_indices)
    pred_energy = np.zeros(n_pred, dtype=np.float64)
    pred_et = np.zeros(n_pred, dtype=np.float64)
    weighted_xyz = np.zeros((n_pred, 3), dtype=np.float64)
    seed_to_pos = {int(seed_index): pos for pos, seed_index in enumerate(seed_indices)}
    for seed_index, pos in seed_to_pos.items():
        cluster_mask = clustering == seed_index
        energy = event.energy[cluster_mask]
        pred_energy[pos] = float(energy.sum())
        pred_et[pos] = float(event.et[cluster_mask].sum())
        if pred_energy[pos] > 0:
            weighted_xyz[pos, 0] = float(np.sum(energy * event.x[cluster_mask]) / pred_energy[pos])
            weighted_xyz[pos, 1] = float(np.sum(energy * event.y[cluster_mask]) / pred_energy[pos])
            weighted_xyz[pos, 2] = float(np.sum(energy * event.z[cluster_mask]) / pred_energy[pos])
        else:
            weighted_xyz[pos] = np.nan
    return {"energy": pred_energy, "et": pred_et, "xyz": weighted_xyz}


def _scan_matched_positions(
    event: _ThresholdScanEvent,
    pred: dict[str, np.ndarray],
    *,
    max_match_distance: float,
    min_energy_ratio: float,
    max_energy_ratio: float,
    matching_algorithm: str,
    hungarian_energy_ratio_log_weight: float,
) -> list[tuple[int, int]]:
    centroid_distance = cdist(event.truth_xyz, pred["xyz"])
    centroid_distance[~np.isfinite(centroid_distance)] = np.inf
    energy_ratio = np.divide(
        pred["energy"][None, :],
        event.truth_energy[:, None],
        out=np.full((event.n_truth, len(pred["energy"])), np.inf),
        where=event.truth_energy[:, None] > 0,
    )
    valid_match = (
        (centroid_distance <= max_match_distance)
        & (energy_ratio >= min_energy_ratio)
        & (energy_ratio <= max_energy_ratio)
    )
    if not np.any(valid_match):
        return []
    return _matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=max_match_distance,
        algorithm=matching_algorithm,
        hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
    )


def _f1(efficiency: float, purity: float, epsilon: float) -> float:
    return float(2.0 * efficiency * purity / (efficiency + purity + epsilon))


def _progress_iterator(values: Any, progress: bool | Any, *, title: str) -> Any:
    if not progress:
        return values
    if callable(progress):
        return progress(values)

    from tqdm.auto import tqdm

    return tqdm(values, total=len(values), desc=title)


def _masked_beta_distances(
    beta: np.ndarray, cluster_coords: np.ndarray, mask: np.ndarray | None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    masked_beta = []
    masked_distances = []
    for event_idx in range(len(beta)):
        valid = (
            np.ones(len(beta[event_idx]), dtype=bool)
            if mask is None
            else np.asarray(mask[event_idx], dtype=bool)
        )
        masked_beta.append(np.asarray(beta[event_idx][valid], dtype=np.float64))
        masked_distances.append(
            cdist(cluster_coords[event_idx][valid], cluster_coords[event_idx][valid])
        )
    return masked_beta, masked_distances


def _get_clustering_precomputed(
    beta: np.ndarray, distances: np.ndarray, tbeta: float, td: float
) -> np.ndarray:
    cond_indices = np.nonzero(beta > tbeta)[0]
    cond_indices = cond_indices[np.argsort(-beta[cond_indices])]
    clustering = -1 * np.ones(len(beta), dtype=np.int32)
    unassigned = np.ones(len(beta), dtype=bool)
    for seed_index in cond_indices:
        if unassigned[seed_index]:
            assign_mask = unassigned & (distances[seed_index] < td)
            clustering[assign_mask] = seed_index
            unassigned[assign_mask] = False
    return clustering


def _concat_evaluations(evaluations: list[OCEvaluation]) -> OCEvaluation:
    def concat_attr(name: str) -> pl.DataFrame:
        frames = [
            getattr(evaluation, name)
            for evaluation in evaluations
            if not getattr(evaluation, name).is_empty()
        ]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

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


def _truth_object_props(event: EventRecord | None, truth_id: int) -> dict[str, Any]:
    if event is None or "objects" not in event.truth:
        return {}
    index = truth_id - 1
    props = {}
    for name in event.truth.objects.fields:
        values = event.truth.objects[name]
        if 0 <= index < len(values):
            value = values[index]
            props[name] = value.item() if hasattr(value, "item") else value
    return props


def _nearest_truth_distance(
    centroids: dict[int, dict[str, float | None]],
) -> dict[int, float | None]:
    out = dict.fromkeys(centroids)
    for truth_id, centroid in centroids.items():
        distances = []
        for other_id, other in centroids.items():
            if other_id == truth_id:
                continue
            dphi = _delta_phi(centroid["phi"], other["phi"])
            if centroid["eta"] is not None and other["eta"] is not None and dphi is not None:
                distances.append(float(np.hypot(centroid["eta"] - other["eta"], dphi)))
        if distances:
            out[truth_id] = min(distances)
    return out


def _reco_properties_by_assignment(
    assignments: np.ndarray, object_ids: np.ndarray, hits: dict[str, np.ndarray]
) -> dict[int, dict[str, float | None]]:
    props = compute_object_properties(
        hits,
        assignments,
        object_ids=object_ids,
        properties=(
            "sum_energy",
            "sum_et",
            "x_energy_weighted",
            "y_energy_weighted",
            "z_energy_weighted",
            "eta_energy_weighted",
            "phi_energy_weighted",
        ),
    )
    return {
        int(object_id): {
            "energy": float(props["sum_energy"][pos]),
            "et": float(props["sum_et"][pos]),
            "x": _none_if_nan(props["x_energy_weighted"][pos]),
            "y": _none_if_nan(props["y_energy_weighted"][pos]),
            "z": _none_if_nan(props["z_energy_weighted"][pos]),
            "eta": _none_if_nan(props["eta_energy_weighted"][pos]),
            "phi": _none_if_nan(props["phi_energy_weighted"][pos]),
        }
        for pos, object_id in enumerate(object_ids)
    }


def _none_if_nan(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _none_to_nan(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _model_reco_columns(
    regressions: dict[str, np.ndarray], seed_index: int
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "model_energy_reco": None,
        "model_x_reco": None,
        "model_y_reco": None,
        "model_z_reco": None,
    }
    if "energy" in regressions and regressions["energy"].shape[-1] == 1:
        out["model_energy_reco"] = float(regressions["energy"][seed_index, 0])
    if "position" in regressions and regressions["position"].shape[-1] >= 3:
        out.update(
            {
                "model_x_reco": float(regressions["position"][seed_index, 0]),
                "model_y_reco": float(regressions["position"][seed_index, 1]),
                "model_z_reco": float(regressions["position"][seed_index, 2]),
            }
        )
    return out


def _binned_rate(
    frame: pl.DataFrame, value_col: str, bins: np.ndarray, *, numerator_col: str, rate_name: str
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    values = frame[value_col].to_numpy()
    numerator_values = frame[numerator_col].to_numpy()
    indices = np.digitize(values, bins) - 1
    rows = []
    for bin_idx in range(len(bins) - 1):
        in_bin = indices == bin_idx
        denom = int(np.sum(in_bin))
        numerator = int(np.sum(numerator_values[in_bin])) if denom else 0
        rate = _safe_ratio(numerator, denom)
        low, high = (
            proportion_confint(numerator, denom, alpha=0.32, method="wilson")
            if denom
            else (0.0, 0.0)
        )
        rows.append(
            {
                "bin_low": float(bins[bin_idx]),
                "bin_high": float(bins[bin_idx + 1]),
                "n": denom,
                "numerator": numerator,
                rate_name: rate,
                f"{rate_name}_binomial_uncertainty": np.sqrt(rate * (1 - rate) / denom)
                if denom
                else 0.0,
                f"{rate_name}_confidence_low": float(low),
                f"{rate_name}_confidence_high": float(high),
            }
        )
    return pl.DataFrame(rows)


def _delta_phi(phi0: float | None, phi1: float | None) -> float | None:
    return (
        None if phi0 is None or phi1 is None else float((phi0 - phi1 + np.pi) % (2 * np.pi) - np.pi)
    )


def _none_subtract(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _safe_none_ratio(numerator: float | None, denominator: float | None) -> float | None:
    return (
        None
        if numerator is None or denominator is None or denominator == 0
        else float(numerator) / float(denominator)
    )


def _count_value(frame: pl.DataFrame, column: str, value: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int((frame[column] == value).sum())
