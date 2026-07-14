"""One-off 2x2 event-display comparison for the v3 multimerge cocktail dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from plot_truth_cluster_separability import (
    ETA_DISPLAY_RANGE,
    NOISE_COLOR,
    OBJECT_COLORS,
    PHI_DISPLAY_RANGE,
    PHI_TICK_LABELS,
    PHI_TICKS,
    WRAP_LINE_COLOR,
    _display_wrap_phi,
    _label,
)

from fastgnn.data import CaloDataset

mplhep.style.use("CMS")

DATASET_ROOT = Path("data/processed/cmssw/v3/NANO_trial_cocktail32_pt5-50-multimerge-fast.root")
OUT_DIR = Path(
    "plotting/truth_cluster_separability/v3/"
    "NANO_trial_cocktail32_pt5-50-multimerge-fast/selected_2x2_comparison"
)
TITLE_PREFIX = "NANO_trial_cocktail32_pt5-50-multimerge-fast"


@dataclass(frozen=True)
class Panel:
    variant: str
    threshold_key: str
    threshold_dir: str
    label: str


@dataclass(frozen=True)
class EventDisplay:
    eta: np.ndarray
    phi: np.ndarray
    energy: np.ndarray
    object_id: np.ndarray


PANELS = (
    Panel(
        variant="default",
        threshold_key="0_hit0",
        threshold_dir="ThresholdRecHits-ClusterTransverseEnergyThreshold_0GeV-HitMinEnergy_0GeV",
        label="default, 0_hit0",
    ),
    Panel(
        variant="default",
        threshold_key="1_hit0p2",
        threshold_dir="ThresholdRecHits-ClusterTransverseEnergyThreshold_1GeV-HitMinEnergy_0.2GeV",
        label="default, 1_hit0p2",
    ),
    Panel(
        variant="d0p5f0p4",
        threshold_key="1_hit0p2",
        threshold_dir="ThresholdRecHits-ClusterTransverseEnergyThreshold_1GeV-HitMinEnergy_0.2GeV",
        label="d0p5f0p4, 1_hit0p2",
    ),
    Panel(
        variant="d0p5f0p6",
        threshold_key="1_hit0p2",
        threshold_dir="ThresholdRecHits-ClusterTransverseEnergyThreshold_1GeV-HitMinEnergy_0.2GeV",
        label="d0p5f0p6, 1_hit0p2",
    ),
)


def _load(panel: Panel) -> list[EventDisplay]:
    ds = CaloDataset(DATASET_ROOT / panel.variant / panel.threshold_dir, split=None)
    return [
        EventDisplay(
            eta=np.asarray(event.hits["eta"], dtype=np.float64),
            phi=np.asarray(event.hits["phi"], dtype=np.float64),
            energy=np.asarray(event.hits["energy"], dtype=np.float64),
            object_id=np.asarray(event.truth.hit_object_id),
        )
        for event in ds
    ]


def _save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _axes() -> tuple[plt.Figure, np.ndarray]:
    return plt.subplots(2, 2, figsize=(11.5, 9.0), squeeze=False)


def _panel_title(panel: Panel, event: EventDisplay) -> str:
    n_objects = len(np.unique(event.object_id[event.object_id > 0]))
    return f"{panel.label}  |  {n_objects} truth objects"


def _plot_event_panel(ax: plt.Axes, panel: Panel, event: EventDisplay) -> None:
    eta_plot, phi_plot, energy_plot, object_id_plot = _display_wrap_phi(
        event.eta, event.phi, event.energy, event.object_id
    )
    noise = object_id_plot <= 0
    if noise.any():
        ax.scatter(
            eta_plot[noise],
            phi_plot[noise],
            s=2,
            c=NOISE_COLOR,
            alpha=0.4,
            label="noise",
        )
    for color_index, object_id_value in enumerate(
        uid for uid in np.unique(object_id_plot) if uid > 0
    ):
        selected = object_id_plot == object_id_value
        size = 4 + 50 * energy_plot[selected] / max(energy_plot[selected].max(), 1e-9)
        ax.scatter(
            eta_plot[selected],
            phi_plot[selected],
            s=size,
            alpha=0.7,
            color=OBJECT_COLORS[color_index % len(OBJECT_COLORS)],
        )
    ax.set_xlim(*ETA_DISPLAY_RANGE)
    ax.set_ylim(*PHI_DISPLAY_RANGE)
    ax.set_xticks([1.5, 2.0, 2.5, 3.0])
    ax.set_yticks(PHI_TICKS)
    ax.set_yticklabels(PHI_TICK_LABELS)
    ax.axhline(np.pi, color=WRAP_LINE_COLOR, ls="--", lw=0.7, alpha=0.8, zorder=0)
    ax.axhline(-np.pi, color=WRAP_LINE_COLOR, ls="--", lw=0.7, alpha=0.8, zorder=0)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(_panel_title(panel, event), fontsize=10, pad=5)


def plot_event_display(event_index: int, events_by_panel: dict[Panel, list[EventDisplay]]) -> None:
    fig, axes = _axes()
    for ax, panel in zip(axes.ravel(), PANELS, strict=True):
        _plot_event_panel(ax, panel, events_by_panel[panel][event_index])
    _label(axes[0, 1])
    fig.suptitle(f"{TITLE_PREFIX}: event {event_index}", fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.91, wspace=0.22, hspace=0.34)
    _save(fig, f"event_display_event{event_index:03d}_2x2.png")


def _clean_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*.png"):
        path.unlink()


def main() -> None:
    _clean_output_dir()
    events_by_panel = {panel: _load(panel) for panel in PANELS}
    n_events = min(len(events) for events in events_by_panel.values())
    for event_index in range(n_events):
        plot_event_display(event_index, events_by_panel)


if __name__ == "__main__":
    main()
