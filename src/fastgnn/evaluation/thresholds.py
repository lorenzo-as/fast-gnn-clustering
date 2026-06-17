"""Threshold (tbeta, td) grid search for OC operating-point selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from fastgnn.data.object_properties import compute_object_properties
from fastgnn.evaluation._common import none_to_nan, safe_ratio
from fastgnn.evaluation.clustering import (
    clustering_from_beta,
    count_clusters_from_labels,
    count_truth_objects,
    greedy_clustering,
    masked_beta_distances,
    seed_order_for_threshold,
)
from fastgnn.evaluation.matching import (
    energy_ratio_matrix,
    match_candidates,
    matched_positions,
)
from fastgnn.evaluation.reco import reco_properties_by_assignment
from fastgnn.evaluation.tables import _required_feature

_PRED_PROPERTIES = (
    "sum_energy",
    "sum_et",
    "x_energy_weighted",
    "y_energy_weighted",
    "z_energy_weighted",
)


@dataclass(frozen=True)
class _ScanEvent:
    labels: np.ndarray
    beta: np.ndarray
    distances: np.ndarray
    hits: dict[str, np.ndarray]
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
    masked_beta, masked_distances = masked_beta_distances(beta, cluster_coords, mask)
    if objective == "matched_f1":
        if features is None or feature_names is None:
            raise ValueError(
                "grid_search_thresholds with objective='matched_f1' requires "
                "unnormalized evaluation features and feature_names"
            )
        scan_events = _build_scan_events(
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
                seed_order_for_threshold(event.beta, float(tbeta)) for event in scan_events
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
                        clustering_from_beta(event_beta, event_distances, tbeta, td)
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


def _build_scan_events(
    *,
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    hit_object_id: np.ndarray,
    mask: np.ndarray | None,
    features: np.ndarray,
    feature_names: list[str],
) -> list[_ScanEvent]:
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
        hits = {
            name: _required_feature(features_v, feature_names, name)
            for name in ("x", "y", "z", "energy")
        }
        truth_ids = np.unique(labels[labels > 0]).astype(np.int32)
        truth_reco = reco_properties_by_assignment(labels, truth_ids, hits)
        events.append(
            _ScanEvent(
                labels=labels,
                beta=beta_v,
                distances=cdist(coords_v, coords_v),
                hits=hits,
                truth_ids=truth_ids,
                truth_energy=np.asarray(
                    [truth_reco[int(t)]["energy"] for t in truth_ids], dtype=np.float64
                ),
                truth_et=np.asarray(
                    [truth_reco[int(t)]["et"] for t in truth_ids], dtype=np.float64
                ),
                truth_xyz=np.asarray(
                    [
                        [none_to_nan(truth_reco[int(t)][axis]) for axis in ("x", "y", "z")]
                        for t in truth_ids
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return events


def _pred_summary_for_clustering(
    event: _ScanEvent, clustering: np.ndarray, seed_indices: np.ndarray
) -> dict[str, np.ndarray]:
    seed_to_pos = {int(seed_index): pos for pos, seed_index in enumerate(seed_indices)}
    assignments = np.asarray(
        [seed_to_pos.get(int(seed_index), -1) + 1 for seed_index in clustering], dtype=np.int32
    )
    props = compute_object_properties(
        event.hits,
        assignments,
        object_ids=np.arange(1, len(seed_indices) + 1, dtype=np.int32),
        properties=_PRED_PROPERTIES,
    )
    return {
        "energy": props["sum_energy"].astype(np.float64),
        "et": props["sum_et"].astype(np.float64),
        "xyz": np.column_stack(
            [
                props["x_energy_weighted"],
                props["y_energy_weighted"],
                props["z_energy_weighted"],
            ]
        ).astype(np.float64),
    }


def _matched_f1_scan_row(
    tbeta: float,
    td: float,
    events: list[_ScanEvent],
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

        clustering = greedy_clustering(event.distances, seed_order, td)
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
    epsilon_obj = safe_ratio(n_matched, n_truth_total)
    p_obj = safe_ratio(n_matched, n_pred_total)
    epsilon_e = safe_ratio(matched_truth_et, total_truth_et)
    p_e = safe_ratio(matched_pred_et, total_pred_et)
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


def _scan_matched_positions(
    event: _ScanEvent,
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
    energy_ratio = energy_ratio_matrix(pred["energy"], event.truth_energy)
    valid_match = match_candidates(
        centroid_distance,
        energy_ratio,
        max_match_distance=max_match_distance,
        min_energy_ratio=min_energy_ratio,
        max_energy_ratio=max_energy_ratio,
    )
    if not np.any(valid_match):
        return []
    return matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=max_match_distance,
        algorithm=matching_algorithm,
        hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight,
    )


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


def _f1(efficiency: float, purity: float, epsilon: float) -> float:
    return float(2.0 * efficiency * purity / (efficiency + purity + epsilon))


def _progress_iterator(values: Any, progress: bool | Any, *, title: str) -> Any:
    if not progress:
        return values
    if callable(progress):
        return progress(values)

    from tqdm.auto import tqdm

    return tqdm(values, total=len(values), desc=title)
