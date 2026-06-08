import argparse
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Wedge
import matplotlib.pyplot as plt
import numpy as np

clusters = {
    "A": {"center": (0.0, 0.0), "color": "#d62728", "label": "target cluster A"},
    "B": {"center": (1.15, 0.15), "color": "#2ca02c", "label": "cluster B"},
    "C": {"center": (0.55, 1.05), "color": "#1f77b4", "label": "cluster C"},
}
bg_color = "#bdbdbd"
residual_color = "#eeeeee"

hits = [
    {"x": -0.42, "y": -0.10, "E": 1.2, "frac": {"A": 0.92}},
    {"x": -0.05, "y": 0.18, "E": 1.8, "frac": {"A": 0.72, "B": 0.18}},
    {"x": 0.34, "y": -0.22, "E": 1.0, "frac": {"A": 0.63, "C": 0.10}},
    {"x": 0.55, "y": 0.20, "E": 2.2, "frac": {"A": 0.46, "B": 0.42}},
    {"x": 0.82, "y": -0.02, "E": 1.6, "frac": {"B": 0.62, "A": 0.28}},
    {"x": 1.10, "y": 0.25, "E": 1.5, "frac": {"B": 0.83}},
    {"x": 1.42, "y": 0.03, "E": 1.1, "frac": {"B": 0.78}},
    {"x": 0.42, "y": 0.72, "E": 1.7, "frac": {"C": 0.58, "A": 0.30}},
    {"x": 0.70, "y": 0.98, "E": 2.1, "frac": {"C": 0.78, "B": 0.10}},
    {"x": 0.23, "y": 1.10, "E": 1.3, "frac": {"C": 0.68, "A": 0.16}},
    {"x": 1.03, "y": 0.72, "E": 0.9, "frac": {"B": 0.45, "C": 0.39}},
    {"x": -0.20, "y": 0.55, "E": 0.8, "frac": {"A": 0.52, "C": 0.25}},
]
noise_hits = [
    {"x": -0.72, "y": 0.76, "E": 0.35},
    {"x": 1.55, "y": 0.85, "E": 0.30},
    {"x": 1.58, "y": -0.36, "E": 0.25},
    {"x": -0.55, "y": -0.45, "E": 0.28},
    {"x": 0.22, "y": -0.66, "E": 0.22},
    {"x": 1.85, "y": 0.38, "E": 0.18},
    {"x": -0.85, "y": 0.20, "E": 0.20},
]

target = "A"
order = ["A", "B", "C"]


def leading_cluster(hit):
    return max(hit["frac"], key=hit["frac"].get)


def radius_from_energy(E):
    return 0.105 * np.sqrt(E)


def draw_background(ax):
    ax.set_xlim(-1.05, 2.05)
    ax.set_ylim(-0.85, 1.45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(
        Rectangle((-1.05, -0.85), 3.10, 2.30, facecolor="#fafafa", edgecolor="none", zorder=-10)
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#bbbbbb")


def draw_pie_hit(
    ax, x, y, E, fracs, *, alpha=1.0, edge="#222222", linewidth=0.55, only_target=None
):
    r = radius_from_energy(E)
    start = 90.0
    if only_target is None:
        for k in order:
            f = fracs.get(k, 0.0)
            if f <= 0:
                continue
            ax.add_patch(
                Wedge(
                    (x, y),
                    r,
                    start,
                    start + 360 * f,
                    facecolor=clusters[k]["color"],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=alpha,
                )
            )
            start += 360 * f
        residual = max(0.0, 1.0 - sum(fracs.values()))
        if residual > 1e-6:
            ax.add_patch(
                Wedge(
                    (x, y),
                    r,
                    start,
                    start + 360 * residual,
                    facecolor=residual_color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=alpha,
                )
            )
    else:
        f = fracs.get(only_target, 0.0)
        if f > 0:
            ax.add_patch(
                Wedge(
                    (x, y),
                    r,
                    start,
                    start + 360 * f,
                    facecolor=clusters[only_target]["color"],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=alpha,
                )
            )
            start += 360 * f
        ax.add_patch(
            Wedge(
                (x, y),
                r,
                start,
                450.0,
                facecolor=residual_color,
                edgecolor="white",
                linewidth=0.35,
                alpha=alpha,
            )
        )
    ax.add_patch(
        Circle(
            (x, y), r, facecolor="none", edgecolor=edge, linewidth=linewidth, alpha=alpha, zorder=5
        )
    )


def draw_full_hit(ax, x, y, E, color, *, alpha=1.0, edge="#222222", linewidth=0.55):
    r = radius_from_energy(E)
    ax.add_patch(
        Circle((x, y), r, facecolor=color, edgecolor=edge, linewidth=linewidth, alpha=alpha)
    )


def draw_noise(ax, alpha=0.55):
    for h in noise_hits:
        draw_full_hit(
            ax, h["x"], h["y"], h["E"], bg_color, alpha=alpha, edge="#999999", linewidth=0.35
        )


def panel_raw(ax):
    draw_background(ax)
    draw_noise(ax, alpha=0.65)
    for h in hits:
        draw_pie_hit(ax, h["x"], h["y"], h["E"], h["frac"], alpha=0.97)
    ax.set_title("Raw RecHits: pie = truth-particle fractions\n", fontsize=9.5, pad=8)


def panel_assigned_full(ax):
    draw_background(ax)
    draw_noise(ax, alpha=0.18)
    energy = 0.0
    for h in hits:
        lead = leading_cluster(h)
        if lead == target:
            energy += h["E"]
            draw_full_hit(
                ax,
                h["x"],
                h["y"],
                h["E"],
                clusters[target]["color"],
                alpha=0.96,
                edge=clusters[lead]["color"],
                linewidth=1.1,
            )
        else:
            draw_full_hit(
                ax,
                h["x"],
                h["y"],
                h["E"],
                "#d8d8d8",
                alpha=0.22,
                edge=clusters[lead]["color"],
                linewidth=0.7,
            )
    ax.set_title(
        r"(1) assigned-full: $\sum_{\mathrm{lead}=A} E_{\rm hit}$"
        f"\nvisible = {energy:.2f} a.u.",
        fontsize=9.5,
        pad=8,
    )


def panel_assigned_fractional(ax):
    draw_background(ax)
    draw_noise(ax, alpha=0.18)
    energy = 0.0
    for h in hits:
        lead = leading_cluster(h)
        if lead == target:
            energy += h["E"] * h["frac"].get(target, 0.0)
            draw_pie_hit(
                ax,
                h["x"],
                h["y"],
                h["E"],
                h["frac"],
                alpha=0.96,
                edge=clusters[lead]["color"],
                linewidth=1.1,
                only_target=target,
            )
        else:
            draw_full_hit(
                ax,
                h["x"],
                h["y"],
                h["E"],
                "#d8d8d8",
                alpha=0.22,
                edge=clusters[lead]["color"],
                linewidth=0.7,
            )
    ax.set_title(
        r"(2) assigned-fractional: $\sum_{\mathrm{lead}=A} f_A E_{\rm hit}$"
        f"\nvisible = {energy:.2f} a.u.",
        fontsize=9.5,
        pad=8,
    )


def panel_all_fractional(ax):
    draw_background(ax)
    draw_noise(ax, alpha=0.18)
    energy = 0.0
    for h in hits:
        lead = leading_cluster(h)
        fA = h["frac"].get(target, 0.0)
        if fA > 0:
            energy += h["E"] * fA
            draw_pie_hit(
                ax,
                h["x"],
                h["y"],
                h["E"],
                h["frac"],
                alpha=0.96,
                edge=clusters[lead]["color"],
                linewidth=1.0,
                only_target=target,
            )
        else:
            draw_full_hit(
                ax,
                h["x"],
                h["y"],
                h["E"],
                "#d8d8d8",
                alpha=0.20,
                edge=clusters[lead]["color"],
                linewidth=0.7,
            )
    ax.set_title(
        r"(3) all-fractional: $\sum_{\mathrm{all\ hits}} f_A E_{\rm hit}$"
        f"\nvisible = {energy:.2f} a.u.",
        fontsize=9.5,
        pad=8,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make figure illustrating different visible-energy definitions for HGCal clustering"
    )
    parser.add_argument(
        "-o", "--out", help="Output directory or file for the generated figure", default="."
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(7.4, 5.9),
        help="Figure size in inches",
    )
    parser.add_argument(
        "--transparent", action="store_true", help="Make figure background transparent"
    )
    args = parser.parse_args()

    outpath = Path(args.out)
    outdir = outpath if outpath.is_dir() else outpath.parent
    filename = outpath.name if not outpath.is_dir() else "hgcal_visible_energy_definitions.png"

    assert outdir.exists(), f"Output directory {outdir} does not exist"
    assert outpath.is_dir() or outpath.suffix.lower() in {".png", ".pdf"}, (
        "Output file must be a PNG or PDF file"
    )
    assert int(args.dpi) > 0, "DPI must be a positive integer"

    fig = plt.figure(
        figsize=tuple(args.figsize),
        dpi=180,
        layout="constrained",
    )

    gs = fig.add_gridspec(2, 2)

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panel_raw(axes[0])
    panel_assigned_full(axes[1])
    panel_assigned_fractional(axes[2])
    panel_all_fractional(axes[3])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=clusters["A"]["color"],
            markeredgecolor="none",
            markersize=7,
            label="truth A",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=clusters["B"]["color"],
            markeredgecolor="none",
            markersize=7,
            label="truth B",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=clusters["C"]["color"],
            markeredgecolor="none",
            markersize=7,
            label="truth C",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=bg_color,
            markeredgecolor="#999999",
            markersize=7,
            label="noise / unused",
        ),
    ]

    fig.suptitle(
        "Visible-energy choices for one truth object with overlapping RecHits",
        fontsize=12.5,
    )

    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    fig.savefig(
        outdir / filename,
        dpi=args.dpi,
        bbox_inches="tight",
        transparent=args.transparent,
    )

    plt.show()
