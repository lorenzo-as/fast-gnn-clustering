"""Feature/target normalization and seed-correction diagnostics for a processed dataset.

Run::

    python plotting/feature_target_normalization/make_feature_target_normalization.py \
        --dataset data/processed/cmssw/v1/.../CartesianRecHits-ClusterEnergyThreshold_1GeV-HitMinEnergy_0.2GeV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from scipy import stats as sstats
import yaml

from fastgnn.data import CaloDataset
from fastgnn.data.base import robust_center_scale
from fastgnn.data.object_properties import compute_object_properties
from fastgnn.utils import PLOTTING_CONFIG

mplhep.style.use("CMS")
HERE = Path(__file__).resolve().parent
PANEL_W, PANEL_H = PLOTTING_CONFIG["figsize"]["A4"]["panel"]

# LaTeX display names for hit features.
FEATURE_LABEL = {
    "x": r"$x\ \mathrm{[cm]}$",
    "y": r"$y\ \mathrm{[cm]}$",
    "z": r"$z\ \mathrm{[cm]}$",
    "r": r"$r\ \mathrm{[cm]}$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "energy": r"$E\ \mathrm{[GeV]}$",
    "et": r"$E_\mathrm{T}\ \mathrm{[GeV]}$",
    "layer": r"$\mathrm{layer}$",
    "x_over_z": r"$x/z$",
    "y_over_z": r"$y/z$",
    "log_energy": r"$\ln E$",
    "log_et": r"$\ln E_\mathrm{T}$",
    "sin_phi": r"$\sin\phi$",
    "cos_phi": r"$\cos\phi$",
}


def _wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    """Median and IQR/1.349 (= sigma for a Gaussian); shared with the data pipeline."""
    median, scale = robust_center_scale(values)
    return float(median), float(scale)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Figure 1: hit-feature normalization
# --------------------------------------------------------------------------- #
def _train_hit_values(ds: CaloDataset, feature_names: list[str]) -> dict[str, np.ndarray]:
    train_ids = {int(i) for i in ds.splits.get("train", [])}
    out: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
    for idx in range(len(ds)):
        event = ds[idx]
        if train_ids and int(event.event_id) not in train_ids:
            continue
        for name in feature_names:
            out[name].append(np.asarray(event.hits[name], dtype=np.float64))
    return {name: (np.concatenate(v) if v else np.asarray([])) for name, v in out.items()}


def plot_features(ds: CaloDataset, normalization: dict, out_dir: Path) -> None:
    # Show stored features plus derived sin/cos phi so the oc_v1 choices are visible.
    feature_names = list(ds.hit_features)
    values = _train_hit_values(ds, [*feature_names, "phi"])
    phi = _finite(values["phi"])
    derived = {"sin_phi": np.sin(phi), "cos_phi": np.cos(phi)}
    rows = [*feature_names, "sin_phi", "cos_phi"]

    n = len(rows)
    col_title = ["raw", "z-score", "robust (median/IQR)"]
    col_xlabel = [
        r"raw value",
        r"$z\text{-}$score $(x-\mu)/\sigma$",
        r"robust $(x-\mathrm{med})/(\mathrm{IQR}/1.35)$",
    ]
    fig, axs = plt.subplots(n, 3, figsize=(3 * PANEL_W, PANEL_H * n), squeeze=False)
    print(f"\n{'feature':12s} {'skew':>7s} {'kurt':>8s} {'%|z|>5':>8s}  verdict")
    for row, name in enumerate(rows):
        x = derived[name] if name in derived else _finite(values[name])
        if name in normalization:
            mu, sd = normalization[name]["mean"], normalization[name]["std"]
        else:
            mu, sd = float(x.mean()), float(x.std() or 1.0)
        med, robust_scale = _robust_center_scale(x)
        z = (x - mu) / sd
        rz = (x - med) / robust_scale
        skew, kurt = float(sstats.skew(x)), float(sstats.kurtosis(x))
        frac = 100.0 * np.mean(np.abs(z) > 5)
        heavy = frac > 0.1 or abs(skew) > 3.0
        verdict = "HEAVY TAIL -> robust/log" if heavy else "z-score ok"
        if name in derived:
            verdict = "derived (replaces raw phi)"
        print(f"{name:12s} {skew:7.2f} {kurt:8.1f} {frac:8.3f}  {verdict}")

        label = FEATURE_LABEL.get(name, name)
        if heavy:
            label = label + r"$\;\bf{(heavy\ tail)}$"
        for col, data in enumerate([x, z, np.clip(rz, -8, 8)]):
            ax = axs[row, col]
            ax.hist(data, bins=60, histtype="step", density=True, lw=1.6, color="C0")
            ax.margins(x=0)
            if row == 0:  # diagnostic grid (not a report figure): label columns
                ax.set_title(col_title[col])
            if row == n - 1:
                ax.set_xlabel(col_xlabel[col])
        axs[row, 0].set_ylabel(label)
    fig.tight_layout()
    _save(fig, out_dir / "hit_feature_normalization.png")


# --------------------------------------------------------------------------- #
# Accumulate object-level seed / target / residual data
# --------------------------------------------------------------------------- #
def _accumulate(ds: CaloDataset, max_events: int) -> dict[str, np.ndarray]:
    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "sum_et",
            "log_sum_et",
            "eta_c",
            "phi_c",
            "z_c",
            "logr_seed",
            "logr_top3",
            "logr_all",
            "seed_frac",
            "deta_seed",
            "dphi_seed",
            "dz_seed",
            "deta_all",
            "dphi_all",
            "dz_all",
        )
    }
    fields = ["x", "y", "z", "energy", "et", "eta", "phi"]
    for idx in range(min(max_events, len(ds))):
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
            ce, cp, cz = (
                p["eta_energy_weighted"][o - 1],
                p["phi_energy_weighted"][o - 1],
                p["z_energy_weighted"][o - 1],
            )
            et_m, eta_m, phi_m, z_m = h["et"][m], h["eta"][m], h["phi"][m], h["z"][m]
            k = int(np.argmax(et_m))  # OC-point proxy: highest-E_T hit
            # In-between case: the OC point may be any of the 3 hardest hits, not
            # necessarily the hardest. Pool each of the top-3 hits as a candidate seed.
            top3_et = np.sort(et_m)[-3:]
            cols["sum_et"].append(sum_et)
            cols["log_sum_et"].append(np.log(sum_et + 1e-6))
            cols["eta_c"].append(ce)
            cols["phi_c"].append(cp)
            cols["z_c"].append(cz)
            cols["seed_frac"].append(et_m[k] / sum_et)
            cols["logr_seed"].append(np.log(sum_et / et_m[k]))
            cols["logr_top3"].append(np.log(sum_et / top3_et))  # up to 3 values/cluster
            cols["logr_all"].append(np.log(sum_et / np.maximum(et_m, 1e-9)))
            cols["deta_seed"].append(eta_m[k] - ce)
            cols["dphi_seed"].append(_wrap(phi_m[k] - cp))
            cols["dz_seed"].append(z_m[k] - cz)
            cols["deta_all"].append(eta_m - ce)
            cols["dphi_all"].append(_wrap(phi_m - cp))
            cols["dz_all"].append(z_m - cz)
    out = {}
    for k, v in cols.items():
        if not v:
            out[k] = np.asarray([])
        elif np.ndim(v[0]) == 0:
            out[k] = np.asarray(v, dtype=np.float64)
        else:
            out[k] = np.concatenate(v)
    return out


# --------------------------------------------------------------------------- #
# Figure 2: payload targets
# --------------------------------------------------------------------------- #
def plot_targets(data: dict[str, np.ndarray], out_dir: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(2 * PANEL_W, 2 * PANEL_H))
    se = _finite(data["sum_et"])
    axs[0, 0].hist(
        se,
        bins=60,
        histtype="step",
        lw=1.6,
        density=True,
        color="C0",
        label=rf"skew $={sstats.skew(se):.1f}$",
    )
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel(r"$\sum_i E_{\mathrm{T},i}\ \mathrm{[GeV]}$")
    lse = _finite(data["log_sum_et"])
    axs[0, 1].hist(
        lse,
        bins=60,
        histtype="step",
        lw=1.6,
        density=True,
        color="C1",
        label=rf"skew $={sstats.skew(lse):.2f}$, $\sigma={lse.std():.2f}$",
    )
    axs[0, 1].set_xlabel(r"$\ln \sum_i E_{\mathrm{T},i}$")
    z = _finite(data["z_c"])
    axs[1, 0].hist(
        z,
        bins=60,
        histtype="step",
        lw=1.6,
        density=True,
        color="C0",
        label=rf"$\mu={z.mean():.0f}$, $\sigma={z.std():.0f}\,\mathrm{{cm}}$",
    )
    axs[1, 0].set_xlabel(r"$z_{\mathrm{centroid}}^{E\text{-}\mathrm{wt}}\ \mathrm{[cm]}$")
    eta = _finite(data["eta_c"])
    axs[1, 1].hist(
        eta,
        bins=60,
        histtype="step",
        lw=1.6,
        density=True,
        color="C0",
        label=rf"$\mu={eta.mean():.2f}$, $\sigma={eta.std():.2f}$",
    )
    axs[1, 1].set_xlabel(r"$\eta_{\mathrm{centroid}}^{E\text{-}\mathrm{wt}}$")
    for ax in axs.flat:
        ax.set_ylabel(r"$\mathrm{a.u.}$")
        ax.margins(x=0)
        ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_dir / "payload_target_distributions.png")


# --------------------------------------------------------------------------- #
# Figure 3: seed-correction hyper-parameter determination
# --------------------------------------------------------------------------- #
def plot_seed_corrections(data: dict[str, np.ndarray], out_dir: Path) -> dict:
    logr_seed = _finite(data["logr_seed"])
    logr_top3 = _finite(data["logr_top3"])
    logr_all = _finite(data["logr_all"])
    fig, axs = plt.subplots(1, 2, figsize=(3.5 * PANEL_W, PANEL_H * 1.4))

    # Left: E_T multiplicative range -> choose a (coverage = fraction of clusters
    # reachable by the leading-hit seed within e^a).
    ax = axs[0]
    edges = np.linspace(0, max(8.5, logr_all.max()), 80)
    ax.hist(
        logr_seed,
        bins=edges,
        histtype="step",
        lw=2.0,
        density=True,
        color="C3",
        label=r"seed $=$ leading hit",
    )
    ax.hist(
        logr_top3,
        bins=edges,
        histtype="step",
        lw=2.0,
        density=True,
        color="C0",
        label=r"seed $\in$ 3 leading hits",
    )
    ax.hist(
        logr_all,
        bins=edges,
        histtype="step",
        lw=1.6,
        ls="--",
        density=True,
        color="0.5",
        label=r"seed $=$ each hit",
    )
    coverage = {}
    for a, c in [(2, "C2"), (3, "C1"), (4, "C4")]:
        cov = 100.0 * np.mean(logr_seed <= a)
        coverage[a] = cov
        ax.axvline(a, color=c, lw=1.6, ls=":", label=rf"$a={a}$: {cov:.0f}% covered")
    ax.set_xlabel(r"$\ln\!\left(\sum_i E_{\mathrm{T},i}\,/\,E_{\mathrm{T}}^{\mathrm{seed}}\right)$")
    ax.set_ylabel(r"$\mathrm{a.u.}$")
    ax.margins(x=0)
    ax.legend(loc="upper right", ncol=1)
    mplhep.add_text(
        r"$E_\mathrm{T}^\mathrm{pred}=E_\mathrm{T}^\mathrm{seed}\,e^{a\tanh r}$",
        loc="upper left",
        ax=ax,
    )

    # Right: position residuals -> choose sigma (seed-hit case).
    ax = axs[1]
    sigmas = {}
    for key, lbl, col, scale in [
        ("deta_seed", r"$\eta$", "C0", 1.0),
        ("dphi_seed", r"$\phi$", "C2", 1.0),
        ("dz_seed", r"$z/100$", "C3", 1 / 100.0),
    ]:
        sig = _robust_center_scale(_finite(data[key]))[1]
        sigmas[key] = sig
        r = _finite(data[key]) * scale
        rng = np.percentile(r, [1, 99])
        ax.hist(
            np.clip(r, *rng),
            bins=70,
            histtype="step",
            lw=1.8,
            density=True,
            color=col,
            label=rf"{lbl}: $\sigma_{{\mathrm{{rob}}}}={sig:.3g}$",
        )
    ax.set_xlabel(r"$\xi_{\mathrm{seed}}-\xi_{\mathrm{centroid}}$ (per coordinate)")
    ax.set_ylabel(r"$\mathrm{a.u.}$")
    ax.margins(x=0)
    ax.legend(loc="upper right", title=r"$\xi^\mathrm{pred}=\xi^\mathrm{seed}+s\,r$")

    fig.tight_layout(w_pad=3.0)
    _save(fig, out_dir / "seed_correction_determination.png")

    # Console summary used to set config defaults.
    print("\n=== a determination (seed = highest-E_T hit) ===")
    print(
        f"  log-ratio: median={np.median(logr_seed):.2f} p90={np.percentile(logr_seed, 90):.2f} "
        f"p99={np.percentile(logr_seed, 99):.2f} max={logr_seed.max():.2f}"
    )
    print(
        f"  3-leading-hits seed: median={np.median(logr_top3):.2f} p99={np.percentile(logr_top3, 99):.2f}"
    )
    for a, cov in coverage.items():
        print(f"  a={a}: covers {cov:.1f}% of clusters (ceiling e^{a}={np.exp(a):.1f}x)")
    print(
        f"  -> recommend a=3 (covers {coverage[3]:.1f}%); a=2 saturates {100 - coverage[2]:.0f}% of clusters"
    )
    print(
        "  WARNING: all-hits log-ratio median "
        f"{np.median(logr_all):.1f} (p99 {np.percentile(logr_all, 99):.1f}) "
        "-> E_T loss must be beta/q-weighted toward OC-candidate hits, not uniform."
    )
    print("=== position sigma (robust, seed-hit) ===")
    print(
        f"  sigma_eta={sigmas['deta_seed']:.3g}  sigma_phi={sigmas['dphi_seed']:.3g}  "
        f"sigma_z={sigmas['dz_seed']:.3g} cm"
    )
    return {"a_coverage": coverage, "sigmas": sigmas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=3000)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()

    ds = CaloDataset(args.dataset)
    normalization = yaml.safe_load((args.dataset / "normalization.yaml").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(ds)} events from {args.dataset}")

    plot_features(ds, normalization, args.out_dir)
    data = _accumulate(ds, args.max_events)
    plot_targets(data, args.out_dir)
    plot_seed_corrections(data, args.out_dir)


if __name__ == "__main__":
    main()
