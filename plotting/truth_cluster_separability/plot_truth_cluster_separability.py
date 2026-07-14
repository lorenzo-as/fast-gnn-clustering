"""Truth-cluster separability diagnostics for a processed dataset.

Answers: are the truth objects (hit_object_id) actually separable in physical
space, or do they overlap?  This sets the *ceiling* on what any object-condensation
coordinate space can achieve -- if truth showers overlap in (eta, phi), no model
can cleanly separate them.

HGCAL note: a single shower spans the full calorimeter depth, so the intra-cluster
*z* spread is large by construction and z is not a discriminating variable between
showers.  The real separation lives in the transverse (eta, phi) plane, so we treat
z separately from (eta, phi) throughout.

Metrics (energy-weighted, per truth object):
  - intra-cluster RMS in eta, phi, z, and dR = sqrt(d_eta^2 + d_phi^2)
  - inter-cluster nearest-neighbour distance between object centroids (dR and dz)
  - per-object separation ratio = (NN inter dR) / (own intra dR); >~3 = well separated

Run::

    python plotting/truth_cluster_separability/plot_truth_cluster_separability.py \
        --dataset data/processed/cmssw/v1/.../ThresholdRecHits-...

"""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.figure import Axes, Figure
import matplotlib.pyplot as plt
import mplhep
import numpy as np

from fastgnn.data import CaloDataset
from fastgnn.utils import PLOTTING_CONFIG

mplhep.style.use("CMS")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)
HERE = Path(__file__).resolve().parent
PANEL_W, PANEL_H = PLOTTING_CONFIG["figsize"]["A4"]["panel"]

NOISE_ID = 0
MAX_EVENT_DISPLAY_EXAMPLES = 6
ETA_DISPLAY_RANGE = (1.3, 3.2)
PHI_WRAP_PADDING = 0.25
PHI_DISPLAY_RANGE = (-np.pi - PHI_WRAP_PADDING, np.pi + PHI_WRAP_PADDING)
PHI_TICKS = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
PHI_TICK_LABELS = [
    r"$-\pi$",
    r"$-\pi/2$",
    r"$0$",
    r"$\pi/2$",
    r"$\pi$",
]
CMS_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
CMS_GREY = "#9c9ca1"
NOISE_COLOR = CMS_GREY
WRAP_LINE_COLOR = "0.55"
OBJECT_COLORS = [color for color in CMS_COLORS if color.lower() != CMS_GREY]


def _display_wrap_phi(
    eta: np.ndarray, phi: np.ndarray, en: np.ndarray, oid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate hits across the phi boundary for continuous wrap-around displays."""
    eta_wrapped = np.tile(eta, 3)
    phi_wrapped = np.concatenate([phi, phi + 2.0 * np.pi, phi - 2.0 * np.pi])
    en_wrapped = np.tile(en, 3)
    oid_wrapped = np.tile(oid, 3)
    keep = (
        (ETA_DISPLAY_RANGE[0] <= eta_wrapped)
        & (eta_wrapped <= ETA_DISPLAY_RANGE[1])
        & (PHI_DISPLAY_RANGE[0] <= phi_wrapped)
        & (phi_wrapped <= PHI_DISPLAY_RANGE[1])
    )
    return eta_wrapped[keep], phi_wrapped[keep], en_wrapped[keep], oid_wrapped[keep]


def _wrap(delta: np.ndarray) -> np.ndarray:
    """Wrap an angular difference to (-pi, pi]."""
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _label(ax: Axes) -> None:
    mplhep.add_text("Simulation", ax=ax, loc="upper right", fontsize=10)


def _metric_box(ax: Axes, text: str, *, y: float = 0.76) -> None:
    ax.text(
        0.98,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.85",
            "alpha": 0.92,
        },
    )


def _with_title_label(title: str, title_label: str | None) -> str:
    if not title_label:
        return title
    return f"{title}\n{title_label}"


class TruthGeometry:
    """Accumulated per-object and per-pair truth-cluster geometry."""

    def __init__(self) -> None:
        # per object
        self.intra_eta: list[float] = []
        self.intra_phi: list[float] = []
        self.intra_z: list[float] = []
        self.intra_dR: list[float] = []
        self.n_hits: list[int] = []
        self.sep_ratio: list[float] = []  # NN inter dR / own intra dR
        # per object (nearest neighbour)
        self.inter_dR: list[float] = []
        self.inter_dz: list[float] = []
        # per event
        self.n_clusters: list[int] = []
        # example events for displays: (eta, phi, energy, ids)
        self.examples: list[tuple] = []

    def add_event(self, hits, oid: np.ndarray, keep_example: bool) -> None:
        eta = np.asarray(hits["eta"], dtype=np.float64)
        phi = np.asarray(hits["phi"], dtype=np.float64)
        z = np.asarray(hits["z"], dtype=np.float64)
        en = np.asarray(hits["energy"], dtype=np.float64)
        oid = np.asarray(oid)

        uids = [u for u in np.unique(oid) if u > NOISE_ID]
        self.n_clusters.append(len(uids))
        if not uids:
            return

        cen_eta = np.empty(len(uids))
        cen_phi = np.empty(len(uids))
        cen_z = np.empty(len(uids))
        intra_dR_per_obj = np.empty(len(uids))
        for i, u in enumerate(uids):
            sel = oid == u
            w = en[sel]
            W = max(float(w.sum()), 1e-9)
            ce = float((w * eta[sel]).sum() / W)
            cp = float(np.arctan2((w * np.sin(phi[sel])).sum(), (w * np.cos(phi[sel])).sum()))
            cz = float((w * z[sel]).sum() / W)
            cen_eta[i], cen_phi[i], cen_z[i] = ce, cp, cz

            d_eta = eta[sel] - ce
            d_phi = _wrap(phi[sel] - cp)
            d_z = z[sel] - cz
            r_eta = float(np.sqrt((w * d_eta**2).sum() / W))
            r_phi = float(np.sqrt((w * d_phi**2).sum() / W))
            r_z = float(np.sqrt((w * d_z**2).sum() / W))
            r_dR = float(np.sqrt((w * (d_eta**2 + d_phi**2)).sum() / W))
            self.intra_eta.append(r_eta)
            self.intra_phi.append(r_phi)
            self.intra_z.append(r_z)
            self.intra_dR.append(r_dR)
            self.n_hits.append(int(sel.sum()))
            intra_dR_per_obj[i] = r_dR

        if len(uids) >= 2:
            dR = np.sqrt(
                (cen_eta[:, None] - cen_eta[None, :]) ** 2
                + _wrap(cen_phi[:, None] - cen_phi[None, :]) ** 2
            )
            dz = np.abs(cen_z[:, None] - cen_z[None, :])
            np.fill_diagonal(dR, np.inf)
            np.fill_diagonal(dz, np.inf)
            nn_dR = dR.min(axis=1)
            nn_dz = dz[np.arange(len(uids)), dR.argmin(axis=1)]
            self.inter_dR.extend(nn_dR.tolist())
            self.inter_dz.extend(nn_dz.tolist())
            self.sep_ratio.extend((nn_dR / np.maximum(intra_dR_per_obj, 1e-9)).tolist())

        if keep_example and len(uids) >= 3 and len(self.examples) < MAX_EVENT_DISPLAY_EXAMPLES:
            self.examples.append((eta.copy(), phi.copy(), en.copy(), oid.copy()))


def _pct(a: list, qs=(10, 50, 90)) -> np.ndarray:
    return np.percentile(np.asarray(a), qs)


def plot_intra_inter_dR(g: TruthGeometry, out_dir: Path, *, title_label: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.65, PANEL_H * 1.38), constrained_layout=True)
    bins = np.linspace(0, 0.25, 60)
    ax.hist(
        g.intra_dR,
        bins=bins,
        histtype="step",
        lw=2,
        density=True,
        label=r"intra-cluster RMS $\Delta R$",
    )
    ax.hist(
        g.inter_dR,
        bins=bins,
        histtype="step",
        lw=2,
        density=True,
        label=r"nearest-neighbour inter-cluster $\Delta R$",
    )
    ax.axvline(np.median(g.intra_dR), color="C0", ls="--", lw=1)
    ax.axvline(np.median(g.inter_dR), color="C1", ls="--", lw=1)
    ax.set_xlabel(r"$\Delta R$ in $(\eta,\phi)$")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", frameon=False)
    _label(ax)
    _metric_box(
        ax,
        "Per truth object:\n"
        r"$\mathrm{intra}\ \Delta R$ = energy-weighted hit RMS"
        "\n"
        r"$\mathrm{inter}\ \Delta R$ = nearest centroid distance",
    )
    sep = np.median(g.inter_dR) / max(np.median(g.intra_dR), 1e-9)
    ax.set_title(_with_title_label(f"median inter/intra separation = {sep:.2f}", title_label))
    _save(fig, out_dir / "transverse_dR_intra_vs_inter.png")


def plot_sep_ratio(g: TruthGeometry, out_dir: Path, *, title_label: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.65, PANEL_H * 1.38), constrained_layout=True)
    r = np.clip(np.asarray(g.sep_ratio), 0, 10)
    ax.hist(r, bins=np.linspace(0, 10, 60), histtype="step", lw=2)
    ax.axvline(1.0, color="grey", ls=":", lw=1.5, label="ratio = 1 (overlap)")
    ax.axvline(3.0, color="green", ls="--", lw=1.5, label="ratio = 3 (well separated)")
    frac_overlap = float((np.asarray(g.sep_ratio) < 1.0).mean())
    ax.set_xlabel(r"NN inter $\Delta R$ / own intra $\Delta R$")
    ax.set_ylabel("objects")
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.82), frameon=False)
    _label(ax)
    _metric_box(
        ax,
        r"$S = \Delta R_{\mathrm{NN}} / \sigma_{\Delta R}$"
        "\n"
        r"$S < 1$: nearest object lies within own spread"
        "\n"
        r"$S \gtrsim 3$: well separated",
        y=0.58,
    )
    ax.set_title(
        _with_title_label(
            f"{100 * frac_overlap:.0f}% of objects have nearest neighbour within own extent",
            title_label,
        )
    )
    _save(fig, out_dir / "separation_ratio.png")


def plot_z(g: TruthGeometry, out_dir: Path, *, title_label: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.65, PANEL_H * 1.38), constrained_layout=True)
    bins = np.linspace(0, 80, 60)
    ax.hist(
        g.intra_z,
        bins=bins,
        histtype="step",
        lw=2,
        density=True,
        label=r"intra-cluster RMS $z$ [cm]",
    )
    ax.hist(
        g.inter_dz,
        bins=bins,
        histtype="step",
        lw=2,
        density=True,
        label=r"nearest-neighbour inter-cluster $\Delta z$ [cm]",
    )
    ax.set_xlabel(r"$z$ spread / $\Delta z$ [cm]")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.82), frameon=False)
    _label(ax)
    _metric_box(
        ax,
        "Longitudinal check:\n"
        r"$\sigma_z$ = energy-weighted hit RMS in $z$"
        "\n"
        r"$\Delta z_{\mathrm{NN}}$ uses same nearest object as $\Delta R_{\mathrm{NN}}$",
        y=0.60,
    )
    ax.set_title(_with_title_label("longitudinal spread", title_label))
    _save(fig, out_dir / "longitudinal_z.png")


def plot_event_displays(g: TruthGeometry, out_dir: Path, *, title_label: str | None = None) -> None:
    n = len(g.examples)
    if n == 0:
        return
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(PANEL_W * ncol * 1.08, PANEL_H * nrow * 1.08),
        squeeze=False,
        constrained_layout=True,
    )
    for k, (eta, phi, en, oid) in enumerate(g.examples):
        ax = axes[k // ncol][k % ncol]
        eta_plot, phi_plot, en_plot, oid_plot = _display_wrap_phi(eta, phi, en, oid)
        uids = [u for u in np.unique(oid_plot) if u > NOISE_ID]
        noise = oid_plot <= NOISE_ID
        if noise.any():
            ax.scatter(
                eta_plot[noise],
                phi_plot[noise],
                s=2,
                c=NOISE_COLOR,
                alpha=0.5,
                label="noise",
            )
        for color_idx, u in enumerate(uids):
            sel = oid_plot == u
            size = 4 + 60 * en_plot[sel] / max(en_plot[sel].max(), 1e-9)
            ax.scatter(
                eta_plot[sel],
                phi_plot[sel],
                s=size,
                alpha=0.7,
                color=OBJECT_COLORS[color_idx % len(OBJECT_COLORS)],
            )
        ax.set_xlim(*ETA_DISPLAY_RANGE)
        ax.set_ylim(*PHI_DISPLAY_RANGE)
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
        ax.set_yticks(PHI_TICKS)
        ax.set_yticklabels(PHI_TICK_LABELS)
        ax.axhline(np.pi, color=WRAP_LINE_COLOR, ls="--", lw=0.7, alpha=0.8, zorder=0)
        ax.axhline(-np.pi, color=WRAP_LINE_COLOR, ls="--", lw=0.7, alpha=0.8, zorder=0)
        if k // ncol == nrow - 1:
            ax.set_xlabel(r"$\eta$")
        if k % ncol == 0:
            ax.set_ylabel(r"$\phi$")
        ax.set_title(f"{len(uids)} truth objects")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    mplhep.add_text("Simulation", ax=axes[0][ncol - 1], loc="upper right", fontsize=10)
    fig.suptitle(
        _with_title_label(
            r"Truth objects in $(\eta,\phi)$; marker size $\propto$ hit energy",
            title_label,
        )
    )
    _save(fig, out_dir / "event_displays_eta_phi.png")


def print_summary(g: TruthGeometry) -> None:
    print("\n========== TRUTH CLUSTER SEPARABILITY ==========")
    print(f"events: {len(g.n_clusters)}   objects: {len(g.intra_dR)}")
    print(f"clusters/event  p10/50/90: {_pct(g.n_clusters)}")
    print(f"hits/object     p10/50/90: {_pct(g.n_hits)}")
    print("\nINTRA-cluster (energy-weighted RMS)   p10/50/90")
    print(f"  eta        : {np.round(_pct(g.intra_eta), 4)}")
    print(f"  phi        : {np.round(_pct(g.intra_phi), 4)}")
    print(f"  z [cm]     : {np.round(_pct(g.intra_z), 2)}")
    print(f"  dR(eta,phi): {np.round(_pct(g.intra_dR), 4)}")
    print("\nINTER-cluster nearest-neighbour centroid distance   p10/50/90")
    print(f"  dR(eta,phi): {np.round(_pct(g.inter_dR), 4)}")
    print(f"  dz [cm]    : {np.round(_pct(g.inter_dz), 2)}")
    sep = np.median(g.inter_dR) / max(np.median(g.intra_dR), 1e-9)
    frac = float((np.asarray(g.sep_ratio) < 1.0).mean())
    print(f"\nSEPARATION (median inter dR / median intra dR): {sep:.2f}")
    print(f"objects with NN closer than own extent (ratio<1): {100 * frac:.0f}%")
    print("  ratio >~3 => well separated (model can succeed)")
    print("  ratio ~1  => showers overlap in eta-phi (clustering ceiling is low)")
    print("================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to read. Omit for the full dataset.",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional event cap. Omit for all selected events.",
    )
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument(
        "--title-label",
        type=str,
        default=None,
        help="Optional label appended as a second title line on every plot.",
    )
    args = ap.parse_args()

    ds = CaloDataset(args.dataset, split=args.split, max_events=args.max_events)
    split_label = args.split or "full dataset"
    print(f"Loaded {len(ds)} events ({split_label}) from {args.dataset}")

    g = TruthGeometry()
    for _i, ev in enumerate(ds):
        g.add_event(
            ev.hits,
            ev.truth.hit_object_id,
            keep_example=len(g.examples) < MAX_EVENT_DISPLAY_EXAMPLES,
        )

    print_summary(g)
    plot_intra_inter_dR(g, args.out_dir, title_label=args.title_label)
    plot_sep_ratio(g, args.out_dir, title_label=args.title_label)
    plot_z(g, args.out_dir, title_label=args.title_label)
    plot_event_displays(g, args.out_dir, title_label=args.title_label)


if __name__ == "__main__":
    main()
