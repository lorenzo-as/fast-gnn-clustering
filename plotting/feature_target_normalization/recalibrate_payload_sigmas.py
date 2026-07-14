"""Recalibrate payload-correction sigmas so each quantity's Huber loss is on the
same scale as the E_T term.

Motivation
----------
``configs/payload/correction.yaml`` normalizes position residuals by a fixed
``sigma`` (eta/phi: 0.02, z: 5.0). Those were read off the *robust* (IQR/1.349)
spread of the leading-hit seed-centroid residual. In a real training run
(outputs/.../2026-06-17/17-01-47) the per-quantity losses are NOT balanced:
eta/phi/z carry 3-5x the E_T loss despite identical config weights. The cause is
that the position residual distributions are sharply peaked with heavy tails: the
robust IQR scale is set by the core, but the Huber loss is driven by the tails.

This script computes, from the data, the sigma that makes each position
quantity's mean prior (r=0) Huber loss equal to the E_T term's prior Huber loss.
That is the actual scale-balancing condition for the optimizer.

The loss (see calc_payload_correction_loss) runs over *all* signal hits, each
correcting from its own seed feature toward the cluster aggregate, beta**2
weighted. We therefore report the balancing sigma for two limits that bracket the
beta-weighted reality:
  - all-hits (uniform over signal hits)        -> early training, diffuse beta
  - leading-hit (highest-E_T seed proxy)       -> converged, beta concentrated on OC point

Run::

    python plotting/feature_target_normalization/recalibrate_payload_sigmas.py \
        --dataset data/processed/cmssw/v1/NANO_10_000samples-new_pt50-100_merging-0.5cm-0.8_tau/ThresholdRecHits-ClusterTransverseEnergyThreshold_1GeV-HitMinEnergy_0.2GeV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fastgnn.data import CaloDataset
from fastgnn.data.base import robust_center_scale
from fastgnn.data.object_properties import compute_object_properties

HERE = Path(__file__).resolve().parent

# Config currently in use (configs/payload/correction.yaml).
CONFIG_SIGMA = {"eta": 0.02, "phi": 0.02, "z": 5.0}
A_ET = 3.0  # max_log_corr / a for the E_T mul_exp_tanh correction
HUBER_DELTA = 1.0
EPS = 1.0e-6


def _wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def _huber(d: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    """Huber loss x2 (matches objectcondensation_loss.huber)."""
    a = np.abs(d)
    return np.where(a <= delta, d**2, 2.0 * delta * (a - delta))


def accumulate(ds: CaloDataset, max_events: int) -> dict[str, np.ndarray]:
    """Per-hit prior residuals (r=0) for every signal hit, plus a leading-hit flag.

    Returns flat arrays over all signal hits in all clusters:
      et_logr   = log(target_sumet) - log(hit_et)      -> E_T residual at r=0
      deta/dphi/dz = hit_coord - centroid               -> position residual at r=0
      is_lead   = bool, hit is the highest-E_T hit in its cluster (seed proxy)
    """
    fields = ["x", "y", "z", "energy", "et", "eta", "phi"]
    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("et_logr", "deta", "dphi", "dz", "is_lead")
    }
    n_events = min(max_events, len(ds)) if max_events else len(ds)
    for idx in range(n_events):
        e = ds[idx]
        h = {k: np.asarray(e.hits[k], dtype=np.float64) for k in fields}
        hid = np.asarray(e.truth.hit_object_id, dtype=np.int64)
        pos = hid[hid > 0]
        if len(pos) == 0:
            continue
        nobj = int(pos.max())
        p = compute_object_properties(
            h,
            hid,
            object_ids=np.arange(1, nobj + 1),
            properties=(
                "sum_et",
                "eta_energy_weighted",
                "phi_energy_weighted",
                "z_energy_weighted",
            ),
        )
        for o in range(1, nobj + 1):
            m = hid == o
            sum_et = p["sum_et"][o - 1]
            if not m.any() or sum_et <= 0:
                continue
            et_m = h["et"][m]
            ce, cp, cz = (
                p["eta_energy_weighted"][o - 1],
                p["phi_energy_weighted"][o - 1],
                p["z_energy_weighted"][o - 1],
            )
            lead = np.zeros(et_m.shape, dtype=bool)
            lead[int(np.argmax(et_m))] = True
            cols["et_logr"].append(np.log(sum_et + EPS) - np.log(np.maximum(et_m, 1e-9) + EPS))
            cols["deta"].append(h["eta"][m] - ce)
            cols["dphi"].append(_wrap(h["phi"][m] - cp))
            cols["dz"].append(h["z"][m] - cz)
            cols["is_lead"].append(lead)
    return {k: np.concatenate(v) if v else np.asarray([]) for k, v in cols.items()}


def _et_prior_huber(et_logr: np.ndarray) -> np.ndarray:
    """E_T Huber residual at r=0: pred=seed*exp(a*tanh 0)=seed, residual=log(seed/target).

    log(seed)-log(target) = -et_logr  (et_logr = log(target)-log(seed)).
    """
    return _huber(-et_logr)


def _sigma_for_target_loss(diff: np.ndarray, target_loss: float) -> float:
    """Find sigma such that mean Huber(diff/sigma) == target_loss (monotone in sigma)."""
    lo, hi = 1e-6, 1e6
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if np.mean(_huber(diff / mid)) > target_loss:
            lo = mid  # loss too big -> need larger sigma
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def report(data: dict[str, np.ndarray]) -> None:
    is_lead = data["is_lead"].astype(bool)
    masks = {"all-hits": np.ones_like(is_lead), "leading-hit": is_lead}

    print(f"\nSignal hits accumulated: {len(is_lead):,}  (leading-hit: {is_lead.sum():,})\n")

    for view, mask in masks.items():
        et_loss = float(np.mean(_et_prior_huber(data["et_logr"][mask])))
        print(f"================  view = {view}  ================")
        print(f"  E_T prior Huber loss (target to match) = {et_loss:.4f}")
        print(
            f"  {'qty':<5} {'robust_sig':>11} {'std':>9} {'cfg_sig':>9} "
            f"{'cfg_loss':>9} {'-> new_sig':>11} {'loss@new':>9}"
        )
        for qty, key in [("eta", "deta"), ("phi", "dphi"), ("z", "dz")]:
            diff = data[key][mask]
            rob = robust_center_scale(diff)[1]
            std = float(np.std(diff))
            cfg_sig = CONFIG_SIGMA[qty]
            cfg_loss = float(np.mean(_huber(diff / cfg_sig)))
            new_sig = _sigma_for_target_loss(diff, et_loss)
            new_loss = float(np.mean(_huber(diff / new_sig)))
            print(
                f"  {qty:<5} {rob:>11.4g} {std:>9.4g} {cfg_sig:>9.4g} "
                f"{cfg_loss:>9.4f} {new_sig:>11.4g} {new_loss:>9.4f}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=3000)
    args = parser.parse_args()

    ds = CaloDataset(args.dataset)
    print(f"Loaded {len(ds)} events from {args.dataset}")
    data = accumulate(ds, args.max_events)
    report(data)


if __name__ == "__main__":
    main()
