"""Decisive test: is the payload-position problem sigma miscalibration, or is the
beta**2 weighting failing to concentrate on hits near the cluster centroid?

The payload correction loss is a beta**2-weighted mean over signal hits. The design
intent is that beta concentrates on the OC condensation point (~ the leading-E_T
hit, which sits near the energy-weighted centroid and needs only a tiny position
correction). If that holds, the *beta-weighted* position residual should look like
the leading-hit prior (small). If beta is diffuse, the beta-weighted residual looks
like the uniform all-hits prior (large) and position correction is near-impossible.

This loads the trained model, runs it on val events, and compares -- per quantity --
the prior (r=0) Huber loss under three weightings:
    uniform        : 1/N over all signal hits      (diffuse-beta limit)
    beta**2        : the ACTUAL loss weighting
    leading-hit    : delta on highest-E_T (seed) hit  (perfect-concentration limit)
plus beta summary stats.

Residuals come straight from the collator's payload_seeds / payload_targets, so they
are aligned with beta and the same truncation/ordering the training loss sees.

Run::

    python plotting/feature_target_normalization/check_beta_concentration.py \
        --config outputs/lxplus-outputs/outputs/2026-06-17/17-01-47/resolved_config.yaml \
        --model  outputs/lxplus-outputs/outputs/2026-06-17/17-01-47/checkpoint_epoch020.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import qgravnet  # noqa: F401  (registers custom keras layers for load_model)
import tensorflow as tf
import yaml

from fastgnn.data import CaloDataset
from fastgnn.training.oc_outputs import OCOutputLayout, split_oc_outputs

# Correction-mode payload spec (configs/payload/correction.yaml), col order = seed/target order.
QUANTITIES = [
    {"name": "et", "seed": "et", "target": "sum_et", "correction": "mul_exp_tanh"},
    {"name": "eta", "seed": "eta", "target": "eta_energy_weighted", "correction": "additive"},
    {
        "name": "phi",
        "seed": "phi",
        "target": "phi_energy_weighted",
        "correction": "additive_circular",
    },
    {"name": "z", "seed": "z", "target": "z_energy_weighted", "correction": "additive"},
]
CONFIG_SIGMA = {"eta": 0.02, "phi": 0.02, "z": 5.0}
HUBER_DELTA = 1.0
EPS = 1.0e-6


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _huber(d, delta=HUBER_DELTA):
    a = np.abs(d)
    return np.where(a <= delta, d**2, 2.0 * delta * (a - delta))


def _residuals(seed, target):
    """Per-hit prior (r=0) Huber residuals, columns [et, eta, phi, z]."""
    return {
        "et": _huber(np.log(target[:, 0] + EPS) - np.log(np.maximum(seed[:, 0], 1e-9) + EPS)),
        "eta": _huber((seed[:, 1] - target[:, 1]) / CONFIG_SIGMA["eta"]),
        "phi": _huber(_wrap(seed[:, 2] - target[:, 2]) / CONFIG_SIGMA["phi"]),
        "z": _huber((seed[:, 3] - target[:, 3]) / CONFIG_SIGMA["z"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-events", type=int, default=400)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir

    ds = CaloDataset(dataset_dir, split="val", max_events=args.max_events)
    layout = OCOutputLayout.from_config(model_cfg)
    model = tf.keras.models.load_model(args.model, compile=False)

    wkeys = ("uniform", "beta2", "et_weight", "betaxet", "lead")
    acc = {q: {w: [] for w in wkeys} for q in ("et", "eta", "phi", "z")}
    beta_max, beta_mean, lead_is_argmaxbeta = [], [], []

    for batch in ds.batches(
        max_vertices=model_cfg["max_vertices"],
        batch_size=train_cfg["batch_size"],
        feature_names=data_cfg["feature_names"],
        payload_quantities=QUANTITIES,
        shuffle=False,
        truncate=train_cfg.get("truncate", "energy_desc"),
        normalize_features=train_cfg.get("normalize_features", True),
        normalization_method=data_cfg.get("normalization", "zscore"),
    ):
        outputs = model(tf.constant(batch["features"]), training=False)
        slices = split_oc_outputs(outputs, layout)
        beta = np.asarray(slices.beta)
        mask = np.asarray(batch["mask"]).astype(bool)
        hid = np.asarray(batch["hit_object_id"]).astype(np.int64)
        seeds = np.asarray(batch["payload_seeds"])
        targets = np.asarray(batch["payload_targets"])

        for b in range(beta.shape[0]):
            m = mask[b]
            ev_hid, ev_beta = hid[b][m], beta[b][m]
            ev_seed, ev_tgt = seeds[b][m], targets[b][m]
            for o in np.unique(ev_hid[ev_hid > 0]):
                sel = ev_hid == o
                if sel.sum() == 0:
                    continue
                bq = np.clip(ev_beta[sel], 0.0, 1.0 - 1e-4) ** 2
                if bq.sum() <= 0:
                    continue
                seed_o, tgt_o = ev_seed[sel], ev_tgt[sel]
                et_o = np.maximum(seed_o[:, 0], 0.0)
                klead = int(np.argmax(seed_o[:, 0]))  # highest seed E_T
                n = sel.sum()
                betaxet = bq * et_o
                w = {
                    "uniform": np.full(n, 1.0 / n),
                    "beta2": bq / bq.sum(),
                    "et_weight": et_o / et_o.sum() if et_o.sum() > 0 else np.full(n, 1.0 / n),
                    "betaxet": betaxet / betaxet.sum()
                    if betaxet.sum() > 0
                    else np.full(n, 1.0 / n),
                    "lead": (np.arange(n) == klead).astype(float),
                }
                resid = _residuals(seed_o, tgt_o)
                for q in resid:
                    for wkey in w:
                        acc[q][wkey].append(float(np.sum(w[wkey] * resid[q])))
                beta_max.append(float(ev_beta[sel].max()))
                beta_mean.append(float(ev_beta[sel].mean()))
                lead_is_argmaxbeta.append(int(np.argmax(ev_beta[sel]) == klead))

    n_obj = len(beta_max)
    print(f"\nClusters evaluated: {n_obj:,}\n")
    print(
        f"beta_max  per cluster: mean={np.mean(beta_max):.3f} median={np.median(beta_max):.3f} "
        f"p90={np.percentile(beta_max, 90):.3f}"
    )
    print(f"beta_mean per cluster: mean={np.mean(beta_mean):.3f}")
    print(
        f"argmax(beta) == leading-E_T hit: {100 * np.mean(lead_is_argmaxbeta):.1f}% of clusters\n"
    )
    print("Prior (r=0) per-object Huber loss at CONFIG sigma, by weighting:")
    hdr = f"  {'qty':<5}{'uniform':>10}{'beta2(actual)':>15}{'E_T':>10}{'beta2*E_T':>12}{'leading':>10}"
    print(hdr)
    for q in ("et", "eta", "phi", "z"):
        print(
            f"  {q:<5}{np.mean(acc[q]['uniform']):>10.3f}"
            f"{np.mean(acc[q]['beta2']):>15.3f}{np.mean(acc[q]['et_weight']):>10.3f}"
            f"{np.mean(acc[q]['betaxet']):>12.3f}{np.mean(acc[q]['lead']):>10.3f}"
        )


if __name__ == "__main__":
    main()
