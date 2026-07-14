"""Check whether the exp(-10*d) GravNet aggregation kernel is effectively zero.

Extracts learned coordinate-space representations from each GravNet block,
computes actual pairwise L1 distances to the k nearest neighbors, then
reports the resulting exp(-10*d) weight distribution.  If median weight is
near zero, neighbor information is not propagating meaningfully.

Run::

    python plotting/feature_target_normalization/check_gravnet_kernel_weights.py \
        --config outputs/lxplus-outputs/outputs/2026-06-17/18-23-53/resolved_config.yaml \
        --model  outputs/lxplus-outputs/outputs/2026-06-17/18-23-53/checkpoint_epoch020.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import qgravnet  # noqa: F401
import tensorflow as tf
import yaml

from fastgnn.data import CaloDataset


def l1_knn_distances(coords: np.ndarray, k: int) -> np.ndarray:
    """Return sorted L1 distances to k nearest neighbors, shape (n_hits, k)."""
    n = coords.shape[0]
    # pairwise L1: (n, n)
    diff = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=-1)
    np.fill_diagonal(diff, np.inf)
    idx = np.argpartition(diff, k, axis=1)[:, :k]
    return np.sort(diff[np.arange(n)[:, None], idx], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--max-events",
        type=int,
        default=50,
        help="events to process (small is fine, pairwise is O(V^2))",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir

    model = tf.keras.models.load_model(args.model, compile=False)
    ds = CaloDataset(dataset_dir, split="val", max_events=args.max_events)

    n_blocks = model_cfg["n_blocks"]
    n_neighbours = model_cfg["n_neighbours"]
    distance_scale = 10.0  # hard-coded in core.py

    # Build sub-models: input -> coords for each block
    coord_models = []
    for ib in range(n_blocks):
        layer_name = f"qgnblock_{ib}_input_spatial_transform"
        coord_layer = model.get_layer(layer_name)
        # walk the functional graph to find the right intermediate tensor
        # by building a new model with the same input
        coord_out = coord_layer.output
        coord_models.append(
            tf.keras.Model(inputs=model.input, outputs=coord_out, name=f"coord_block{ib}")
        )

    all_dists = [[] for _ in range(n_blocks)]
    all_weights = [[] for _ in range(n_blocks)]

    for batch in ds.batches(
        max_vertices=model_cfg["max_vertices"],
        batch_size=train_cfg["batch_size"],
        feature_names=data_cfg["feature_names"],
        shuffle=False,
        normalize_features=train_cfg.get("normalize_features", True),
        normalization_method=data_cfg.get("normalization", "zscore"),
    ):
        feats = tf.constant(batch["features"])
        mask = np.asarray(batch["mask"]).astype(bool)  # (B, V)

        for ib, cm in enumerate(coord_models):
            coords_batch = np.asarray(cm(feats, training=False))  # (B, V, D)
            for b in range(coords_batch.shape[0]):
                m = mask[b]
                c = coords_batch[b][m]  # (n_real_hits, D)
                if c.shape[0] < 2:
                    continue
                k = min(n_neighbours, c.shape[0] - 1)
                d = l1_knn_distances(c, k)  # (n_hits, k)
                w = np.exp(-distance_scale * d)
                all_dists[ib].append(d.ravel())
                all_weights[ib].append(w.ravel())

    print(f"\ndistance_metric=l1  exp_scale={distance_scale}  k={n_neighbours}")
    print(f"events processed: {args.max_events}\n")
    print(
        f"{'block':<8} {'d_nn p10':>10} {'d_nn p50':>10} {'d_nn p90':>10} "
        f"{'w p10':>10} {'w p50':>10} {'w p90':>10} {'w>0.01 %':>10} {'w>0.1 %':>10}"
    )
    print("-" * 98)
    for ib in range(n_blocks):
        d_all = np.concatenate(all_dists[ib])
        w_all = np.concatenate(all_weights[ib])
        print(
            f"  {ib:<6} "
            f"{np.percentile(d_all, 10):>10.4f} "
            f"{np.percentile(d_all, 50):>10.4f} "
            f"{np.percentile(d_all, 90):>10.4f} "
            f"{np.percentile(w_all, 10):>10.4e} "
            f"{np.percentile(w_all, 50):>10.4e} "
            f"{np.percentile(w_all, 90):>10.4e} "
            f"{100 * (w_all > 0.01).mean():>10.1f} "
            f"{100 * (w_all > 0.1).mean():>10.1f}"
        )

    print("\nInterpretation:")
    print("  w median >> 0   -> neighbors are contributing (kernel not collapsed)")
    print("  w median ~  0   -> exp(-10*d) is killing all neighbors; only max survives")
    print("  d_nn p50 > 0.5  -> coordinate space spread is large relative to kernel width")
    print(f"  kernel ~0 threshold: d > {np.log(100) / distance_scale:.2f} gives w < 0.01")


if __name__ == "__main__":
    main()
