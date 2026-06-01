"""
Object Condensation loss for calorimeter GNN training.

TensorFlow port of:
    https://github.com/mmarchegiani/hgcal-gravnet/blob/master/src/loss/objectcondensation.py
    (based on https://arxiv.org/abs/2002.03605, Jan Kieseler)
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def batch_and_mask_to_flat(
    batch_dict: dict,
) -> tuple[dict, tf.Tensor]:
    """
    Convert padded (B, V, ...) batch to flat (total_real_hits, ...) tensors.

    Args:
        batch_dict: output of PadCollator, contains "mask" (B, V) bool tensor
                    and padded arrays such as features and hit_object_id

    Returns:
        flat_dict:  same keys, values are (total_real_hits, ...) — padding stripped
        batch_idx:  (total_real_hits,) int32 — which event each hit belongs to
                    analogous to torch_geometric `batch` vector
    """
    mask = tf.cast(batch_dict["mask"], tf.bool)  # (B, V)
    B = tf.shape(mask)[0]

    # Build batch index: for each hit position, which event index
    event_idx = tf.repeat(
        tf.range(B, dtype=tf.int32),
        tf.reduce_sum(tf.cast(mask, tf.int32), axis=1),
    )

    flat = {}
    for key, val in batch_dict.items():
        if key == "mask":
            continue
        t = tf.cast(val, tf.float32) if val.dtype != tf.bool else val
        # Flatten (B, V, ...) -> (B*V, ...) then select real hits
        flat_val = tf.reshape(t, [-1, *t.shape[2:].as_list()])
        flat_mask = tf.reshape(mask, [-1])
        flat[key] = tf.boolean_mask(flat_val, flat_mask)

    return flat, event_idx


# Scatter helpers


def scatter_sum(data: tf.Tensor, indices: tf.Tensor, n_segments: int) -> tf.Tensor:
    return tf.math.unsorted_segment_sum(data, indices, n_segments)


def scatter_max_vals(data: tf.Tensor, indices: tf.Tensor, n_segments: int) -> tf.Tensor:
    return tf.math.unsorted_segment_max(data, indices, n_segments)


def scatter_count(indices: tf.Tensor, n_segments: int) -> tf.Tensor:
    """Count occurrences of each segment index."""
    ones = tf.ones_like(indices, dtype=tf.int32)
    return tf.math.unsorted_segment_sum(ones, indices, n_segments)


def safe_divide(numerator: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    """Divide with zero denominators mapped to zero contribution."""
    denominator = tf.cast(denominator, numerator.dtype)
    return tf.math.divide_no_nan(numerator, denominator)


def batch_cluster_indices(
    cluster_id_per_event: tf.Tensor,
    batch: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Convert per-event cluster indices to global batch-level indices.

    Example:
        cluster_id_per_event = [0, 0, 1, 1, 2, 0, 0, 1]
        batch                = [0, 0, 0, 0, 0, 1, 1, 1]
        -> cluster_index     = [0, 0, 1, 1, 2, 3, 3, 4]

    Returns:
        cluster_index:        (n_hits,) global cluster index
        n_clusters_per_event: (batch_size,)
    """
    batch_size = tf.reduce_max(batch) + 1

    # Max cluster id per event + 1 = n_clusters per event
    n_clusters_per_event = tf.cast(
        tf.maximum(
            scatter_max_vals(
                tf.cast(cluster_id_per_event, tf.int32) + 1,
                batch,
                batch_size,
            ),
            0,
        ),
        tf.int32,
    )

    # Cumulative offset per event, aligned to hits
    offsets_per_event = tf.concat([[0], tf.cumsum(n_clusters_per_event)[:-1]], axis=0)
    hit_offsets = tf.gather(offsets_per_event, batch)

    cluster_index = tf.cast(cluster_id_per_event, tf.int32) + hit_offsets
    return cluster_index, n_clusters_per_event


def get_inter_event_norms_mask(
    batch: tf.Tensor,
    n_clusters_per_event: tf.Tensor,
) -> tf.Tensor:
    """
    Build (n_hits, n_clusters) mask that is 1 only if hit i and cluster j
    are in the same event.

    Used to exclude cross-event repulsion terms.
    """
    batch_size = tf.shape(n_clusters_per_event)[0]

    # (batch_size, n_hits): one-hot event membership per hit
    hit_event_onehot = tf.cast(
        tf.equal(
            tf.expand_dims(batch, 0),
            tf.expand_dims(tf.range(batch_size, dtype=batch.dtype), 1),
        ),
        tf.int32,
    )  # (batch_size, n_hits)

    # Expand to (n_clusters, n_hits) by repeating each event row n_clusters_per_event times
    mask = tf.repeat(hit_event_onehot, n_clusters_per_event, axis=0)  # (n_clusters, n_hits)
    return tf.transpose(mask)  # (n_hits, n_clusters)


def huber(d: tf.Tensor, delta: float) -> tf.Tensor:
    """
    Huber loss, multiplied by 2
    """
    return tf.where(
        tf.abs(d) <= delta,
        d**2,
        2.0 * delta * (tf.abs(d) - delta),
    )


# ---------------------------------------------------------------------------
# Main OC loss: L_V + L_beta
# ---------------------------------------------------------------------------


def calc_LV_Lbeta(
    beta: tf.Tensor,  # (n_hits,)   predicted condensation probability
    cluster_space_coords: tf.Tensor,  # (n_hits, D) predicted cluster space coords
    cluster_index_per_event: tf.Tensor,  # (n_hits,)   truth hit->cluster index (0=noise)
    batch: tf.Tensor,  # (n_hits,)   event index per hit
    qmin: float = 1.0,
    s_B: float = 0.1,
    noise_cluster_index: int = 0,
    beta_stabilizing: str = "soft_q_scaling",
    huberize_norm_for_V_attractive: bool = True,
    beta_term_option: str = "paper",
    return_components: bool = False,
) -> tuple[tf.Tensor, tf.Tensor] | dict:
    """
    Compute L_V (potential) and L_beta object condensation losses.

    Expects flat tensors over all hits in the batch.
    Use batch_and_mask_to_flat() to convert from padded (B, V, F) format.

    Args:
        beta:                      predicted beta per hit, in (0, 1)
        cluster_space_coords:      predicted coordinates in clustering space
        cluster_index_per_event:   truth cluster index, 0 = noise
        batch:                     event index for each hit
        qmin:                      minimum charge (stability)
        s_B:                       noise beta penalty weight
        noise_cluster_index:       cluster index value meaning noise (must be 0)
        beta_stabilizing:          "paper" | "clip" | "soft_q_scaling"
        huberize_norm_for_V_attractive: use Huber norm in attractive potential
        beta_term_option:          "paper" | "short-range-potential"
        return_components:         if True, return dict of all loss components

    Returns:
        (L_V, L_beta) normalised by batch size, or dict of components

    Concepts:
        - Each hit is assigned to exactly one cluster (cluster_index_per_event has shape (n_hits,))
        and exactly one event (batch has shape (n_hits,))
        - A cluster index equal to `noise_cluster_index` indicates that the cluster is noise.
        Typically, there is one noise cluster per event. Any hit belonging to a noise cluster
        is considered a 'noise hit'. Hits that belong to an object are referred to as
        'signal hits' for lack of a better term.
        - An 'object' refers to any cluster that is *not* a noise cluster.

    Notes:
        - beta_stabilizing: Available options are ['paper', 'clip', 'soft_q_scaling']:
            paper: beta = sigmoid(model_output), q = beta.arctanh()**2 + qmin
            clip:  beta is clipped at 1-1e-4, q = beta.arctanh()**2 + qmin
            soft_q_scaling: beta = sigmoid(model_output), q = (clip(beta)/1.002).arctanh()**2 + qmin

        - huberize_norm_for_V_attractive: Applies Huberization to norms used in the attractive potential

        - beta_term_option: Available options are ['paper', 'short-range-potential']:
            Selecting 'short-range-potential' introduces a localized potential around high-beta
            points, behaving similarly to V_attractive.

    Note that this function differs slightly from the implementation in 2002.03605:
    - The norms used for V_repulsive are Gaussian rather than a linear hinge formulation
    """
    if noise_cluster_index != 0:
        raise NotImplementedError("noise_cluster_index != 0 not supported")

    batch = tf.cast(batch, tf.int32)
    cluster_index_per_event = tf.cast(cluster_index_per_event, tf.int32)

    batch_size = tf.reduce_max(batch) + 1
    n_hits_per_event = tf.cast(scatter_count(batch, batch_size), tf.float32)

    # Global cluster indices
    cluster_index, n_clusters_per_event = batch_cluster_indices(cluster_index_per_event, batch)
    n_clusters = tf.reduce_sum(n_clusters_per_event)

    # Cluster -> event mapping
    batch_cluster = tf.repeat(tf.range(batch_size, dtype=tf.int32), n_clusters_per_event)

    # Signal / noise masks
    is_noise = tf.equal(cluster_index_per_event, noise_cluster_index)
    is_sig = ~is_noise

    # Per-cluster: is it an object (signal) or noise cluster?
    is_object = tf.cast(
        tf.maximum(
            scatter_max_vals(
                tf.cast(is_sig, tf.int32),
                cluster_index,
                n_clusters,
            ),
            0,
        ),
        tf.bool,
    )

    object_index_per_event = cluster_index_per_event[is_sig] - 1
    object_index, n_objects_per_event = batch_cluster_indices(object_index_per_event, batch[is_sig])
    n_objects = tf.reduce_sum(n_objects_per_event)
    n_hits_per_object = tf.cast(scatter_count(object_index, n_objects), tf.float32)
    batch_object = batch_cluster[is_object]
    n_objects_per_event = scatter_count(batch_object, batch_size)

    # ------------------------------------------------------------------
    # Compute q (charge) from beta
    # ------------------------------------------------------------------

    if beta_stabilizing == "paper":
        q = tf.math.atanh(beta) ** 2 + qmin
    elif beta_stabilizing == "clip":
        beta = tf.clip_by_value(beta, 0.0, 1.0 - 1e-4)
        q = tf.math.atanh(beta) ** 2 + qmin
    elif beta_stabilizing == "soft_q_scaling":
        q = tf.math.atanh(tf.clip_by_value(beta, 0.0, 1.0 - 1e-4) / 1.002) ** 2 + qmin
    else:
        raise ValueError(f"Unknown beta_stabilizing: {beta_stabilizing}")

    # q_alpha: max q per object, at the condensation point (alpha hit)
    q_sig = tf.boolean_mask(q, is_sig)
    q_alpha, index_alpha = (
        tf.math.unsorted_segment_max_with_argmax(  # type: ignore
            q_sig, object_index, n_objects
        )
        if hasattr(tf.math, "unsorted_segment_max_with_argmax")
        else _q_alpha_fallback(q_sig, object_index, n_objects)
    )

    x_sig = tf.boolean_mask(cluster_space_coords, is_sig)
    x_alpha = tf.gather(x_sig, index_alpha)  # (n_objects, D)
    beta_alpha = tf.gather(tf.boolean_mask(beta, is_sig), index_alpha)  # (n_objects,)

    # ------------------------------------------------------------------
    # Connectivity matrices M and M_inv  (n_hits, n_objects)
    # ------------------------------------------------------------------

    M = tf.cast(
        tf.boolean_mask(
            tf.one_hot(cluster_index, n_clusters, dtype=tf.int32),
            is_object,
            axis=1,
        ),
        tf.float32,
    )
    inter_event_mask = tf.cast(
        tf.boolean_mask(
            get_inter_event_norms_mask(batch, n_clusters_per_event),
            is_object,
            axis=1,
        ),
        tf.float32,
    )
    M_inv = inter_event_mask - M

    # ------------------------------------------------------------------
    # All pairwise norms: (n_hits, n_objects)
    # ------------------------------------------------------------------

    # (n_hits, 1, D) - (1, n_objects, D) -> (n_hits, n_objects, D)
    diff = tf.expand_dims(cluster_space_coords, 1) - tf.expand_dims(x_alpha, 0)
    norms = tf.sqrt(tf.reduce_sum(diff**2, axis=-1) + 1e-9)  # (n_hits, n_objects)

    # ------------------------------------------------------------------
    # L_V attractive
    # ------------------------------------------------------------------

    norms_att = tf.boolean_mask(norms, is_sig)  # (n_hits_sig, n_objects)

    norms_att = huber(norms_att + 1e-5, 4.0) if huberize_norm_for_V_attractive else norms_att**2

    M_sig = tf.boolean_mask(M, is_sig)
    norms_att = norms_att * M_sig

    q_sig_col = tf.expand_dims(q_sig, -1)  # (n_hits_sig, 1)
    q_alpha_row = tf.expand_dims(q_alpha, 0)  # (1, n_objects)
    V_attractive = q_sig_col * q_alpha_row * norms_att  # (n_hits_sig, n_objects)

    # Sum over hits -> (n_objects,), then sum per event / n_hits_per_event
    V_att_per_object = tf.reduce_sum(V_attractive, axis=0)
    V_att_per_event = scatter_sum(V_att_per_object, batch_object, batch_size)
    L_V_attractive = tf.reduce_sum(safe_divide(V_att_per_event, n_hits_per_event))

    # ------------------------------------------------------------------
    # L_V repulsive
    # ------------------------------------------------------------------

    norms_rep = tf.exp(-4.0 * norms**2) * M_inv  # (n_hits, n_objects)

    q_col = tf.expand_dims(q, -1)
    V_repulsive = q_col * q_alpha_row * norms_rep

    V_rep_per_object = tf.reduce_sum(V_repulsive, axis=0)
    V_rep_per_event = scatter_sum(V_rep_per_object, batch_object, batch_size)
    L_V_repulsive = tf.reduce_sum(safe_divide(V_rep_per_event, n_hits_per_event))

    L_V = L_V_attractive + L_V_repulsive

    # ------------------------------------------------------------------
    # L_beta noise term
    # ------------------------------------------------------------------

    beta_noise = tf.boolean_mask(beta, is_noise)
    batch_noise = tf.boolean_mask(batch, is_noise)
    n_noise_per_event = tf.cast(scatter_count(batch_noise, batch_size), tf.float32)
    beta_noise_sum = scatter_sum(beta_noise, batch_noise, batch_size)
    L_beta_noise = s_B * tf.reduce_sum(safe_divide(beta_noise_sum, n_noise_per_event))

    # ------------------------------------------------------------------
    # L_beta signal term
    # ------------------------------------------------------------------

    if beta_term_option == "paper":
        L_beta_sig_per_object = 1.0 - beta_alpha
        L_beta_sig = tf.reduce_sum(
            safe_divide(
                scatter_sum(L_beta_sig_per_object, batch_object, batch_size),
                tf.cast(n_objects_per_event, tf.float32),
            )
        )

    elif beta_term_option == "short-range-potential":
        norms_beta = 1.0 / (20.0 * tf.boolean_mask(norms, is_sig) ** 2 + 1.0)
        norms_beta_masked = tf.reduce_sum(norms_beta * M_sig, axis=0)
        norms_beta_masked = (1.0 - norms_beta_masked) / n_hits_per_object
        norms_beta_masked = norms_beta_masked * beta_alpha

        L_beta_norms_term = tf.reduce_sum(
            safe_divide(
                scatter_sum(norms_beta_masked, batch_object, batch_size),
                tf.cast(n_objects_per_event, tf.float32),
            )
        )
        L_beta_logbeta_term = tf.reduce_sum(
            safe_divide(
                scatter_sum(
                    -0.2 * tf.math.log(beta_alpha + 1e-9),
                    batch_object,
                    batch_size,
                ),
                tf.cast(n_objects_per_event, tf.float32),
            )
        )
        L_beta_sig = L_beta_norms_term + L_beta_logbeta_term

    else:
        raise ValueError(f"Unknown beta_term_option: {beta_term_option}")

    L_beta = L_beta_noise + L_beta_sig

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------

    batch_size_f = tf.cast(batch_size, tf.float32)

    if return_components:
        components = {
            "L_V": L_V / batch_size_f,
            "L_V_attractive": L_V_attractive / batch_size_f,
            "L_V_repulsive": L_V_repulsive / batch_size_f,
            "L_beta": L_beta / batch_size_f,
            "L_beta_noise": L_beta_noise / batch_size_f,
            "L_beta_sig": L_beta_sig / batch_size_f,
        }
        if beta_term_option == "short-range-potential":
            components["L_beta_norms_term"] = L_beta_norms_term / batch_size_f
            components["L_beta_logbeta_term"] = L_beta_logbeta_term / batch_size_f
        return components

    return L_V / batch_size_f, L_beta / batch_size_f


def _q_alpha_fallback(
    q_sig: tf.Tensor,
    object_index: tf.Tensor,
    n_objects: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Fallback for tf versions without unsorted_segment_max_with_argmax.
    Returns (q_alpha, index_alpha) — max q and its index per object.
    """
    membership = tf.cast(tf.one_hot(object_index, n_objects, dtype=tf.int32), tf.bool)
    q_per_object = tf.where(
        membership,
        tf.expand_dims(q_sig, axis=1),
        tf.fill(tf.shape(membership), tf.constant(-np.inf, dtype=q_sig.dtype)),
    )
    q_alpha = tf.reduce_max(q_per_object, axis=0)
    index_alpha = tf.argmax(q_per_object, axis=0, output_type=tf.int32)
    return q_alpha, index_alpha


# Post-training inference: clustering from model output


def get_clustering_np(
    betas: np.ndarray,
    X: np.ndarray,
    tbeta: float = 0.1,
    td: float = 1.0,
) -> np.ndarray:
    """
    Greedy clustering of hits based on predicted betas and cluster space coords.

    Args:
        betas:  (n_hits,) predicted condensation probabilities
        X:      (n_hits, D) predicted cluster space coordinates
        tbeta:  beta threshold to identify condensation points
        td:     distance threshold for cluster assignment

    Returns:
        clustering: (n_hits,) integer cluster assignment, -1 = unassigned (noise)
    """
    betas = np.asarray(betas)
    X = np.asarray(X)

    if betas.ndim != 1:
        raise ValueError(f"betas must be one-dimensional, got shape {betas.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
    if X.shape[0] != betas.shape[0]:
        raise ValueError(
            f"betas and X must have the same number of hits, got {betas.shape[0]} and {X.shape[0]}"
        )

    n_points = len(betas)
    select_condpoints = betas > tbeta
    indices_condpoints = np.nonzero(select_condpoints)[0]
    indices_condpoints = indices_condpoints[np.argsort(-betas[indices_condpoints])]

    clustering = -1 * np.ones(n_points, dtype=np.int32)
    unassigned = np.ones(n_points, dtype=bool)

    for idx_cp in indices_condpoints:
        if not unassigned[idx_cp]:
            continue

        d = np.linalg.norm(X - X[idx_cp], axis=-1)
        assign_mask = unassigned & (d < td)

        clustering[assign_mask] = idx_cp
        unassigned[assign_mask] = False

    return clustering


# Logging utility


def formatted_loss_components_string(components: dict) -> str:
    """Pretty-print loss component breakdown for training logs."""
    total = components.get("L_total", components["L_V"] + components["L_beta"])
    total_val = float(total.numpy()) if hasattr(total, "numpy") else float(total)

    def fmt(key):
        val = (
            float(components[key].numpy())
            if hasattr(components[key], "numpy")
            else float(components[key])
        )
        frac = val / total_val if total_val > 0 else 0.0
        return f"{val:+.4f} ({100.0 * frac:.1f}%)"

    lines = [
        f"  L_total        = {fmt('L_total')}"
        if "L_total" in components
        else f"  L_total        = {total_val:+.4f} (100.0%)",
        f"  L_V            = {fmt('L_V')}",
        f"  L_V_attractive = {fmt('L_V_attractive')}",
        f"  L_V_repulsive  = {fmt('L_V_repulsive')}",
        f"  L_beta         = {fmt('L_beta')}",
        f"  L_beta_noise   = {fmt('L_beta_noise')}",
        f"  L_beta_sig     = {fmt('L_beta_sig')}",
    ]
    if "L_beta_norms_term" in components:
        lines += [
            f"  L_beta_norms   = {fmt('L_beta_norms_term')}",
            f"  L_beta_logbeta = {fmt('L_beta_logbeta_term')}",
        ]
    return "\n".join(lines)
