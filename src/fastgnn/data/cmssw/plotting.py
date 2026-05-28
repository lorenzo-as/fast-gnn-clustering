from __future__ import annotations

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fastgnn.data.base import EventRecord

from .utils import HGCAL_Z

_GREEK = [
    ("nu_ebar", "ν̄ₑ"),
    ("nu_mubar", "ν̄μ"),
    ("nu_taubar", "ν̄τ"),
    ("nu_e", "νₑ"),
    ("nu_mu", "νμ"),
    ("nu_tau", "ντ"),
    ("gamma", "γ"),
    ("omega", "ω"),
    ("phi", "φ"),
    ("Upsilon", "Υ"),
    ("Lambda", "Λ"),
    ("Sigma", "Σ"),
    ("Delta", "Δ"),
    ("Omega", "Ω"),
    ("Xi", "Ξ"),
    ("alpha", "α"),
    ("beta", "β"),
    ("rho", "ρ"),
    ("psi", "ψ"),
    ("chi", "χ"),
    ("eta", "η"),
    ("tau", "τ"),
    ("mu", "μ"),
    ("pi", "π"),
]
_SUP = str.maketrans("0+-*", "⁰⁺⁻*")


def _to_pretty(name: str) -> str:
    """'pi-' → 'π⁻', 'gamma' → 'γ', etc."""
    s = name
    for ascii_, uni in _GREEK:
        s = s.replace(ascii_, uni)
    return s.translate(_SUP)


def _pdgid_to_name(pdgid: int) -> str:
    try:
        from particle import Particle

        return _to_pretty(Particle.from_pdgid(int(pdgid)).name)
    except Exception:
        return str(pdgid)


_PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#ffffff",
]


def _cluster_color(hit_object_id: int) -> str:
    if hit_object_id == 0:
        return "#888888"  # noise: solid grey
    return _PALETTE[(hit_object_id - 1) % len(_PALETTE)]


def plot_event(
    event: EventRecord,
    color_by: Literal["energy", "hit_object_id"] = "energy",
    view: Literal["2d", "3d", "both"] = "both",
    show_cluster_markers: bool = True,
    energy_threshold: float = 0.0,
    show_clusters_below_threshold: bool = True,
    width: int = 1000,
    height: int = 580,
) -> tuple[go.Figure, str]:
    """
    Plot one canonical CMSSW event.

    This is the notebook-facing API: pass an EventRecord, not storage-specific
    HDF5/Parquet internals.
    """
    event.hits.require("x", "y", "z", "energy")
    event.truth.require("hit_object_id", "objects")
    event.truth.objects.require("impact_eta", "impact_phi", "impact_energy", "track_pdg_id")

    mode = "truth" if color_by == "hit_object_id" else "energy"
    return plot_event_display(
        h_x=event.hits.x,
        h_y=event.hits.y,
        h_z=event.hits.z,
        h_e=event.hits.energy,
        hit_object_id=event.truth.hit_object_id,
        c_eta=event.truth.objects.impact_eta,
        c_phi=event.truth.objects.impact_phi,
        c_e=event.truth.objects.impact_energy,
        c_pdg=event.truth.objects.track_pdg_id,
        mode=mode,
        view=view,
        show_cluster_markers=show_cluster_markers,
        energy_threshold=energy_threshold,
        show_clusters_below_threshold=show_clusters_below_threshold,
        event_idx=event.event_id,
        width=width,
        height=height,
    )


def plot_event_display(
    h_x: np.ndarray,
    h_y: np.ndarray,
    h_z: np.ndarray,
    h_e: np.ndarray,
    hit_object_id: np.ndarray,
    c_eta: np.ndarray,
    c_phi: np.ndarray,
    c_e: np.ndarray,
    c_pdg: np.ndarray,
    mode: str = "energy",
    view: Literal["2d", "3d", "both"] = "both",
    show_cluster_markers: bool = True,
    energy_threshold: float = 0.0,
    show_clusters_below_threshold: bool = True,
    event_idx: int | None = None,
    width: int = 1000,
    height: int = 580,
) -> tuple[go.Figure, str]:
    """
    CMSSW HGCAL L1 event display and dataset visualisation.

    Primary function: plot_event_display()
        - Colors hits by log(energy)  [mode="energy"]
        - Colors hits by SimCluster assignment  [mode="truth"]
        - Both modes can show XY, 3D, or both views

    Usage:
        from fastgnn.data.cmssw.plotting import plot_event_display

        fig, summary = plot_event_display(
            h_x=event["features"][:, 0],       # Hit X coordinates
            h_y=event["features"][:, 1],       # Hit Y coordinates
            h_z=event["features"][:, 2],       # Hit Z coordinates
            h_e=event["features"][:, 3],       # Hit Energy
            hit_object_id=event["hit_object_id"],  # Truth assignment (0=noise)
            c_eta=event["cluster_impact_eta"], # Cluster impact Eta
            c_phi=event["cluster_impact_phi"], # Cluster impact Phi
            c_e=event["cluster_impact_energy"],# Cluster energy
            c_pdg=event["cluster_track_pdgId"],# Cluster particle ID
            mode="truth",                      # "energy" | "truth"
            energy_threshold=2.0,              # Optional: filter summary by energy
        )

        print(summary)
        fig.show()
    """
    if mode not in ("energy", "truth"):
        raise ValueError(f"mode must be 'energy' or 'truth', got '{mode}'")
    if view not in ("2d", "3d", "both"):
        raise ValueError(f"view must be '2d', '3d', or 'both', got '{view}'")

    # Project impact (eta, phi) to x, y at z=318.5 cm (HGCAL face)
    from .utils import etaphi_to_xy_at_z

    c_x, c_y, c_z = etaphi_to_xy_at_z(c_eta, c_phi, z=HGCAL_Z)
    c_names = [_pdgid_to_name(p) for p in c_pdg]

    mask_above = c_e > energy_threshold
    mask_below = ~mask_above

    # Build figure
    if view == "both":
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.5, 0.5],
            specs=[[{"type": "xy"}, {"type": "scene"}]],
            subplot_titles=[f"XY Plane (z≈{HGCAL_Z:.1f} cm)", "3D View"],
        )
    elif view == "2d":
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "xy"}]],
            subplot_titles=[f"XY Plane (z≈{HGCAL_Z:.1f} cm)"],
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=["3D View"],
        )

    if mode == "energy":
        _add_energy_traces(fig, h_x, h_y, h_z, h_e, view=view)
    else:
        # hit_object_id is 1-indexed into c_names/c_e (0 = noise)
        object_labels = {0: "Noise"}
        for i, (name, energy) in enumerate(zip(c_names, c_e)):
            object_labels[i + 1] = f"{name} {energy:.2f} GeV"
        _add_truth_traces(fig, h_x, h_y, h_z, h_e, hit_object_id, object_labels, view=view)

    # SimCluster impact points
    if show_cluster_markers:
        _add_cluster_markers(
            fig,
            c_x[mask_above],
            c_y[mask_above],
            c_z[mask_above],
            c_e[mask_above],
            [n for n, m in zip(c_names, mask_above) if m],
            color="red",
            label="Clusters",
            which=view,
        )
    if show_clusters_below_threshold and mask_below.any():
        _add_cluster_markers(
            fig,
            c_x[mask_below],
            c_y[mask_below],
            c_z[mask_below],
            c_e[mask_below],
            [n for n, m in zip(c_names, mask_below) if m],
            color="grey",
            label="Clusters (below threshold)",
            which=view,
        )

    # Annotations on XY view
    if view in ("2d", "both") and show_cluster_markers and mask_above.any():
        offsets = [(12, 12), (-12, 12), (12, -12), (-12, -12)]
        cx_above = c_x[mask_above]
        cy_above = c_y[mask_above]
        names_above = [n for n, m in zip(c_names, mask_above) if m]
        for i, (x_, y_, name) in enumerate(zip(cx_above, cy_above, names_above)):
            ax, ay = offsets[i % len(offsets)]
            fig.add_annotation(
                x=x_,
                y=y_,
                ax=ax,
                ay=ay,
                xref="x",
                yref="y",
                axref="pixel",
                ayref="pixel",
                text=name,
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
            )

    title = f"Event {event_idx} | " if event_idx is not None else ""
    title += f"Mode: {mode} | {len(h_x)} hits ({int((hit_object_id > 0).sum())} signal, {int((hit_object_id == 0).sum())} noise) | {np.sum(mask_above)} clusters above {energy_threshold} GeV"
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(
            itemsizing="constant",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
    )
    if view in ("2d", "both"):
        fig.update_xaxes(title_text="x [cm]", row=1, col=1)
        fig.update_yaxes(
            title_text="y [cm]",
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1,
        )
    if view in ("3d", "both"):
        fig.update_layout(
            scene=dict(
                xaxis_title="x [cm]",
                yaxis_title="y [cm]",
                zaxis_title="z [cm]",
                aspectmode="data",
                zaxis=dict(autorange="reversed"),
            ),
            scene_camera=dict(
                eye=dict(x=0, y=0, z=2.5),
                up=dict(x=0, y=1, z=0),
            ),
        )

    # Particle summary markdown
    above = [(n, e) for n, e, m in zip(c_names, c_e, mask_above) if m]
    below = [(n, e) for n, e, m in zip(c_names, c_e, mask_below) if m]
    above_sorted = sorted(above, key=lambda x: x[1], reverse=True)
    below_sorted = sorted(below, key=lambda x: x[1], reverse=True)

    summary = f"**Particles above {energy_threshold} GeV ({len(above_sorted)}):** "
    summary += ", ".join(f"{n} ({e:.2f} GeV)" for n, e in above_sorted) or "—"
    if below_sorted:
        summary += f"\n\n**Particles below threshold ({len(below_sorted)}):** "
        summary += ", ".join(f"{n} ({e:.2f} GeV)" for n, e in below_sorted)

    return fig, summary


def _add_energy_traces(
    fig: go.Figure,
    h_x,
    h_y,
    h_z,
    h_e,
    view: Literal["2d", "3d", "both"] = "both",
) -> None:
    """Add hit traces coloured by log10(energy)."""
    log_e = np.log10(np.clip(h_e, 1e-6, None))
    e_min = log_e.min() if len(log_e) > 0 else 0
    e_max = log_e.max() if len(log_e) > 0 else 1
    tick_vals = np.arange(np.floor(e_min), np.ceil(e_max) + 1)
    tick_text = [f"10<sup>{int(v)}</sup>" for v in tick_vals]

    marker_2d = dict(
        size=3,
        color=log_e,
        cmin=e_min,
        cmax=e_max,
        colorscale="Viridis",
        showscale=False,
        opacity=0.85,
    )
    marker_3d = dict(
        size=2,
        color=log_e,
        cmin=e_min,
        cmax=e_max,
        colorscale="Viridis",
        showscale=True,
        opacity=0.85,
        colorbar=dict(
            title="Hit E [GeV]",
            thickness=12,
            len=0.6,
            x=1.01,
            y=0.35,
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
    )

    hover = "x=%{x:.2f}<br>y=%{y:.2f}<br>E=%{marker.color:.4f} GeV<extra>RecHit</extra>"
    hover3d = (
        "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>E=%{marker.color:.4f} GeV<extra>RecHit</extra>"
    )

    if view in ("2d", "both"):
        fig.add_trace(
            go.Scatter(
                x=h_x,
                y=h_y,
                mode="markers",
                marker=marker_2d,
                name="RecHits",
                legendgroup="rechits",
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )

    if view in ("3d", "both"):
        fig.add_trace(
            go.Scatter3d(
                x=h_x,
                y=h_y,
                z=h_z,
                mode="markers",
                marker=marker_3d,
                name="RecHits",
                legendgroup="rechits",
                showlegend=(view == "3d"),
                hovertemplate=hover3d,
            ),
            row=1,
            col=2 if view == "both" else 1,
        )


def _add_truth_traces(
    fig: go.Figure,
    h_x,
    h_y,
    h_z,
    h_e,
    hit_object_id: np.ndarray,
    object_labels: dict[int, str],
    view: Literal["2d", "3d", "both"] = "both",
) -> None:
    """
    Add hit traces coloured by SimCluster assignment (hit_object_id).
    Noise hits shown in solid grey. Signal hits labelled by particle type + energy.
    """
    unique_objects = np.unique(hit_object_id)
    cluster_energies = {obj: h_e[hit_object_id == obj].sum() for obj in unique_objects if obj != 0}
    sorted_objects = [0] + sorted(
        cluster_energies, key=lambda obj: cluster_energies[obj], reverse=True
    )

    for obj in sorted_objects:
        mask = hit_object_id == obj
        color = _cluster_color(obj)
        label = object_labels.get(obj, f"Cluster {obj}")
        size_2d = 2 if obj == 0 else np.clip(3 + 3 * np.sqrt(h_e[mask]), 2, 10).tolist()
        size_3d = 1 if obj == 0 else np.clip(2 + 2 * np.sqrt(h_e[mask]), 1, 6).tolist()
        opacity = 0.3 if obj == 0 else 0.85

        hover = f"x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>E=%{{customdata:.4f}} GeV<extra>{label}</extra>"
        hover3d = (
            f"x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<br>"
            f"E=%{{customdata:.4f}} GeV<extra>{label}</extra>"
        )

        if view in ("2d", "both"):
            fig.add_trace(
                go.Scatter(
                    x=h_x[mask],
                    y=h_y[mask],
                    mode="markers",
                    marker=dict(size=size_2d, color=color, opacity=opacity),
                    name=label,
                    legendgroup=label,
                    showlegend=True,
                    customdata=h_e[mask],
                    hovertemplate=hover,
                ),
                row=1,
                col=1,
            )

        if view in ("3d", "both"):
            fig.add_trace(
                go.Scatter3d(
                    x=h_x[mask],
                    y=h_y[mask],
                    z=h_z[mask],
                    mode="markers",
                    marker=dict(size=size_3d, color=color, opacity=opacity),
                    name=label,
                    legendgroup=label,
                    showlegend=(view == "3d"),
                    customdata=h_e[mask],
                    hovertemplate=hover3d,
                ),
                row=1,
                col=2 if view == "both" else 1,
            )


def _add_cluster_markers(
    fig: go.Figure,
    c_x,
    c_y,
    c_z,
    c_e,
    c_names,
    color: str,
    label: str,
    which: Literal["2d", "3d", "both"] = "both",
) -> None:
    """Add SimCluster impact point markers to XY and 3D views."""
    if len(c_x) == 0:
        return

    hover = (
        "<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<br>"
        "E=%{customdata:.3f} GeV<extra>Cluster</extra>"
    )
    hover3d = (
        "<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>"
        "E=%{customdata:.3f} GeV<extra>Cluster</extra>"
    )

    if which in ("2d", "both"):
        fig.add_trace(
            go.Scatter(
                x=c_x,
                y=c_y,
                mode="markers",
                text=c_names,
                marker=dict(size=14, color=color, line=dict(color="black", width=1)),
                name=label,
                legendgroup=label,
                customdata=c_e,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
    if which in ("3d", "both"):
        fig.add_trace(
            go.Scatter3d(
                x=c_x,
                y=c_y,
                z=c_z,
                mode="markers",
                text=c_names,
                marker=dict(size=6, color=color, line=dict(color="black", width=1)),
                name=label,
                legendgroup=label,
                showlegend=(which == "3d"),
                customdata=c_e,
                hovertemplate=hover3d,
            ),
            row=1,
            col=2 if which == "both" else 1,
        )
