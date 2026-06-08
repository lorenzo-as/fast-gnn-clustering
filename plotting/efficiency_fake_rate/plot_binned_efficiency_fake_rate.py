from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import mplhep
import numpy as np

from fastgnn.utils import PLOTTING_CONFIG

mplhep.style.use("CMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare binned efficiency and fake rate from pickled OC evaluations."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Pickled OCEvaluation files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels, one per input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output plot path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=12,
        help="Number of logarithmic bins per panel.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Energy threshold [GeV] drawn on the energy panels. When supplied, "
            "predicted-energy fake-rate bins match the notebook split-bin construction."
        ),
    )
    args = parser.parse_args()
    if len(args.inputs) != len(args.labels):
        parser.error("--inputs and --labels must have the same length")
    return args


def load_oc_eval(path: Path):
    with path.open("rb") as f:
        oc_eval = pickle.load(f)

    for attr in ("truth", "predicted"):
        if not hasattr(oc_eval, attr):
            raise TypeError(f"{path} does not look like a pickled OCEvaluation: missing {attr}")
    return oc_eval


def log_edges(values, *, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.logspace(
        np.log10(values.min()),
        np.log10(values.max()),
        bins + 1,
    )


def notebook_bins(oc_eval, column: str, *, bins: int, threshold: float | None) -> np.ndarray:
    if column == "truth_energy":
        return log_edges(oc_eval.truth["truth_energy"], bins=bins)
    if column == "assigned_cluster_energy":
        pred_energy = oc_eval.predicted["assigned_cluster_energy"]
        if threshold is None:
            return log_edges(pred_energy, bins=bins)
        truth_bins = log_edges(oc_eval.truth["truth_energy"], bins=bins)
        pred_low_bins = log_edges(pred_energy.filter(pred_energy < threshold), bins=3)
        return np.concatenate([pred_low_bins, truth_bins[1:]])
    if column == "truth_n_hits":
        return log_edges(oc_eval.truth["truth_n_hits"], bins=bins)
    if column == "cluster_size":
        return log_edges(oc_eval.predicted["cluster_size"], bins=bins)
    raise ValueError(f"no notebook binning rule for column {column}")


def main() -> None:
    args = parse_args()
    oc_evals = [load_oc_eval(path) for path in args.inputs]

    from fastgnn.evaluation import binned_efficiency, binned_fake_rate
    from fastgnn.plotting import plot_binned_efficiency, plot_binned_fake_rate

    panels = [
        (
            "truth",
            "truth_energy",
            binned_efficiency,
            plot_binned_efficiency,
            "Cluster truth energy [GeV]",
            "Efficiency",
        ),
        (
            "predicted",
            "assigned_cluster_energy",
            binned_fake_rate,
            plot_binned_fake_rate,
            "Cluster predicted energy [GeV]",
            "Fake Rate",
        ),
        (
            "truth",
            "truth_n_hits",
            binned_efficiency,
            plot_binned_efficiency,
            "True cluster size",
            "Efficiency",
        ),
        (
            "predicted",
            "cluster_size",
            binned_fake_rate,
            plot_binned_fake_rate,
            "Predicted cluster size",
            "Fake Rate",
        ),
    ]

    fullwidth_2pane = PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"]
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(fullwidth_2pane[0], 2 * fullwidth_2pane[1]),
        constrained_layout=True,
    )

    for ax, (frame_name, column, table_func, plot_func, xlabel, ylabel) in zip(
        axs.flat, panels, strict=True
    ):
        for oc_eval, label in zip(oc_evals, args.labels, strict=True):
            frame = getattr(oc_eval, frame_name)
            bins = notebook_bins(oc_eval, column, bins=args.bins, threshold=args.threshold)
            table = table_func(frame, column, bins)
            plot_func(ax, table, label=label)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if args.threshold is not None and column in {"truth_energy", "assigned_cluster_energy"}:
            ax.axvline(args.threshold, color="grey", linestyle="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(alpha=0.25)

    axs[0, 0].legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
