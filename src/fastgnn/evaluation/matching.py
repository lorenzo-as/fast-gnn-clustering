"""Truth/prediction matching: candidate gating, distance/ratio matrices, assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from fastgnn.evaluation._common import delta_phi


def validate_matching_algorithm(algorithm: str) -> str:
    algorithm = str(algorithm).lower()
    if algorithm not in {"hungarian", "greedy"}:
        raise ValueError("matching_algorithm must be 'hungarian' or 'greedy'")
    return algorithm


def validate_matching_reference(
    matching_reference: str,
    payload_features: dict[str, np.ndarray] | None,
) -> str:
    matching_reference = str(matching_reference).lower()
    if matching_reference not in {"aggregated", "payload"}:
        raise ValueError("matching_reference must be 'aggregated' or 'payload'")
    if matching_reference == "payload":
        if payload_features is None:
            raise ValueError("matching_reference='payload' requires decoded payload predictions")
        if "et" not in payload_features:
            raise ValueError("matching_reference='payload' requires an ET payload quantity")
        if not (
            {"x", "y", "z"}.issubset(payload_features) or {"eta", "phi"}.issubset(payload_features)
        ):
            raise ValueError(
                "matching_reference='payload' requires either x/y/z or eta/phi payload quantities"
            )
    return matching_reference


def energy_ratio_matrix(pred_values: np.ndarray, truth_values: np.ndarray) -> np.ndarray:
    """``pred / truth`` matrix of shape ``(n_truth, n_pred)``; inf where truth <= 0."""
    return np.divide(
        pred_values[None, :],
        truth_values[:, None],
        out=np.full((len(truth_values), len(pred_values)), np.inf),
        where=truth_values[:, None] > 0,
    )


def match_candidates(
    centroid_distance: np.ndarray,
    energy_ratio: np.ndarray,
    *,
    max_match_distance: float,
    min_energy_ratio: float,
    max_energy_ratio: float,
) -> np.ndarray:
    return (
        (centroid_distance <= max_match_distance)
        & (energy_ratio >= min_energy_ratio)
        & (energy_ratio <= max_energy_ratio)
    )


def payload_position_distance(
    truth: dict[str, float | None],
    pred: dict[str, float | None],
) -> float:
    if all(pred.get(axis) is not None for axis in ("x", "y", "z")):
        return float(
            np.linalg.norm(
                [
                    float(pred[axis]) - float(truth[axis])
                    for axis in ("x", "y", "z")
                    if truth[axis] is not None
                ]
            )
        )
    if pred.get("eta") is None or pred.get("phi") is None:
        return np.inf
    if truth["eta"] is None or truth["phi"] is None:
        return np.inf
    components = [float(pred["eta"]) - float(truth["eta"])]
    dphi = delta_phi(pred["phi"], truth["phi"])
    if dphi is None:
        return np.inf
    components.append(dphi)
    if pred.get("z") is not None and truth["z"] is not None:
        components.append((float(pred["z"]) - float(truth["z"])) / 100.0)
    return float(np.linalg.norm(components))


def matched_positions(
    centroid_distance: np.ndarray,
    energy_ratio: np.ndarray,
    valid_match: np.ndarray,
    *,
    max_match_distance: float,
    algorithm: str,
    hungarian_energy_ratio_log_weight: float,
) -> list[tuple[int, int]]:
    algorithm = validate_matching_algorithm(algorithm)
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
