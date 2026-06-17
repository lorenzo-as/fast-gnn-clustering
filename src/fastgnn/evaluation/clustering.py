"""Greedy condensation clustering and object-count helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def greedy_clustering(distances: np.ndarray, seed_indices: np.ndarray, td: float) -> np.ndarray:
    """Assign each hit to the highest-beta seed within ``td``.

    ``seed_indices`` must already be ordered by descending beta. Returns a label
    per hit: the seed index it is assigned to, or ``-1`` if unassigned.
    """
    n = distances.shape[0]
    clustering = -1 * np.ones(n, dtype=np.int32)
    unassigned = np.ones(n, dtype=bool)
    for seed_index in seed_indices:
        if unassigned[seed_index]:
            assign_mask = unassigned & (distances[seed_index] < td)
            clustering[assign_mask] = int(seed_index)
            unassigned[assign_mask] = False
    return clustering


def seed_order_for_threshold(beta: np.ndarray, tbeta: float) -> np.ndarray:
    """Indices of hits above ``tbeta``, ordered by descending beta."""
    seed_order = np.nonzero(beta > tbeta)[0]
    return seed_order[np.argsort(-beta[seed_order])].astype(np.int32)


def clustering_from_beta(
    beta: np.ndarray, distances: np.ndarray, tbeta: float, td: float
) -> np.ndarray:
    """Greedy clustering seeded by all hits above ``tbeta`` (precomputed distances)."""
    return greedy_clustering(distances, seed_order_for_threshold(beta, tbeta), td)


def count_clusters_from_labels(clustering: np.ndarray) -> int:
    return int(np.sum(np.unique(clustering) >= 0))


def count_truth_objects(hit_object_id: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(
        [
            len(np.unique(labels[labels > 0]))
            for labels in (_event_labels(hit_object_id, mask, i) for i in range(len(hit_object_id)))
        ],
        dtype=np.int32,
    )


def count_pred_objects(
    beta: np.ndarray,
    cluster_coords: np.ndarray,
    tbeta: float,
    td: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    counts = []
    for event_beta, event_distances in zip(
        *masked_beta_distances(beta, cluster_coords, mask), strict=True
    ):
        counts.append(
            0
            if not np.any(event_beta > tbeta)
            else count_clusters_from_labels(
                clustering_from_beta(event_beta, event_distances, tbeta, td)
            )
        )
    return np.asarray(counts, dtype=np.int32)


def masked_beta_distances(
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


def _event_labels(hit_object_id: np.ndarray, mask: np.ndarray | None, event_idx: int) -> np.ndarray:
    labels = np.asarray(hit_object_id[event_idx])
    return labels if mask is None else labels[np.asarray(mask[event_idx], dtype=bool)]
