from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fastgnn.data.base import EventRecord

from .constants import HGCAL_Z

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
EventView = Literal["xy", "yz", "3d"]


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


_OBJECT_THRESHOLD_LABELS = {
    "impact_energy": "impact energy",
    "impact_pt": "impact pT",
}


def _threshold_label_from_field(field: str) -> str:
    return _OBJECT_THRESHOLD_LABELS.get(field, field.replace("_", " "))


def plot_event(
    event: EventRecord,
    color_by: Literal["energy", "hit_object_id"] = "energy",
    view: EventView | Sequence[EventView] | None = None,
    views: Sequence[EventView] | None = None,
    show_cluster_markers: bool = True,
    energy_threshold: float = 0.0,
    cluster_threshold_field: str = "impact_energy",
    show_clusters_below_threshold: bool = True,
    width: int | None = None,
    height: int = 580,
) -> tuple[go.Figure, str]:
    """
    Plot one canonical CMSSW event.

    This is the notebook-facing API: pass an EventRecord.
    """
    event.hits.require("x", "y", "z", "energy")
    event.truth.require("hit_object_id", "objects")
    event.truth.objects.require(
        "impact_eta", "impact_phi", "impact_energy", cluster_threshold_field, "track_pdg_id"
    )

    mode = "truth" if color_by == "hit_object_id" else "energy"
    threshold_values = getattr(event.truth.objects, cluster_threshold_field)
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
        views=views,
        show_cluster_markers=show_cluster_markers,
        energy_threshold=energy_threshold,
        cluster_threshold_values=threshold_values,
        cluster_threshold_label=_threshold_label_from_field(cluster_threshold_field),
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
    view: EventView | Sequence[EventView] | None = None,
    views: Sequence[EventView] | None = None,
    show_cluster_markers: bool = True,
    energy_threshold: float = 0.0,
    cluster_threshold_values: np.ndarray | None = None,
    cluster_threshold_label: str = "impact energy",
    cluster_threshold_unit: str = "GeV",
    show_clusters_below_threshold: bool = True,
    event_idx: int | None = None,
    width: int | None = None,
    height: int = 580,
) -> tuple[go.Figure, str]:
    """
    CMSSW HGCAL L1 event display and dataset visualisation.

    Primary function: plot_event_display()
        - Colors hits by log(energy)  [mode="energy"]
        - Colors hits by SimCluster assignment  [mode="truth"]
        - Both modes can show any combination of XY, YZ, and 3D views

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
            cluster_threshold_values=event["cluster_impact_pt"],
            cluster_threshold_label="impact pT",
        )

        print(summary)
        fig.show()
    """
    if mode not in ("energy", "truth"):
        raise ValueError(f"mode must be 'energy' or 'truth', got '{mode}'")
    selected_views = _normalize_views(view=view, views=views)

    # Project impact (eta, phi) to x, y at z=318.5 cm (HGCAL face)
    from fastgnn.geometry import etaphi_to_xy_at_z

    c_x, c_y, c_z = etaphi_to_xy_at_z(c_eta, c_phi, z=HGCAL_Z)
    c_names = [_pdgid_to_name(p) for p in c_pdg]

    threshold_values = c_e if cluster_threshold_values is None else np.asarray(cluster_threshold_values)
    if threshold_values.shape != np.asarray(c_e).shape:
        raise ValueError(
            "cluster_threshold_values must have the same shape as c_e, "
            f"got {threshold_values.shape} and {np.asarray(c_e).shape}"
        )
    threshold_label = cluster_threshold_label.strip() or "threshold"
    threshold_unit = cluster_threshold_unit.strip()
    threshold_suffix = f" {threshold_unit}" if threshold_unit else ""
    threshold_text = f"{threshold_label} {energy_threshold:g}{threshold_suffix}"

    mask_above = threshold_values > energy_threshold
    mask_below = ~mask_above
    below_threshold_object_ids = set((np.flatnonzero(mask_below) + 1).tolist())

    # Build figure
    fig = make_subplots(
        rows=1,
        cols=len(selected_views),
        column_widths=[1] * len(selected_views),
        specs=[[_view_subplot_spec(v) for v in selected_views]],
        subplot_titles=[_view_title(v) for v in selected_views],
        horizontal_spacing=0.04 if len(selected_views) > 1 else 0.0,
    )
    figure_width = width if width is not None else _default_figure_width(selected_views)

    if mode == "energy":
        _add_energy_traces(fig, h_x, h_y, h_z, h_e, views=selected_views)
    else:
        # hit_object_id is 1-indexed into c_names/c_e (0 = noise)
        object_labels = {0: "Noise"}
        for i, (name, energy, threshold_value) in enumerate(zip(c_names, c_e, threshold_values)):
            object_labels[i + 1] = (
                f"{name} E={energy:.2f} GeV, {threshold_label}={threshold_value:.2f}"
                f"{threshold_suffix}"
            )
        _add_truth_traces(
            fig,
            h_x,
            h_y,
            h_z,
            h_e,
            hit_object_id,
            object_labels,
            views=selected_views,
            below_threshold_object_ids=below_threshold_object_ids,
        )

    # SimCluster impact points
    if show_cluster_markers:
        _add_cluster_markers(
            fig,
            c_x[mask_above],
            c_y[mask_above],
            c_z[mask_above],
            c_e[mask_above],
            threshold_values[mask_above],
            [n for n, m in zip(c_names, mask_above) if m],
            color="red",
            label="Clusters",
            threshold_label=threshold_label,
            threshold_unit=threshold_unit,
            views=selected_views,
        )
    if show_clusters_below_threshold and mask_below.any():
        _add_cluster_markers(
            fig,
            c_x[mask_below],
            c_y[mask_below],
            c_z[mask_below],
            c_e[mask_below],
            threshold_values[mask_below],
            [n for n, m in zip(c_names, mask_below) if m],
            color="grey",
            label="Clusters (below threshold)",
            threshold_label=threshold_label,
            threshold_unit=threshold_unit,
            views=selected_views,
        )

    # Annotations on XY view
    if "xy" in selected_views and show_cluster_markers and mask_above.any():
        xy_col = _view_col(selected_views, "xy")
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
                axref="pixel",
                ayref="pixel",
                text=name,
                showarrow=True,
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10),
                row=1,
                col=xy_col,
            )

    title = f"Event {event_idx} | " if event_idx is not None else ""
    title += f"Mode: {mode} | {len(h_x)} hits ({int((hit_object_id > 0).sum())} signal, {int((hit_object_id == 0).sum())} noise) | {np.sum(mask_above)} clusters above {threshold_text}"
    fig.update_layout(
        title=title,
        width=figure_width,
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
    if "xy" in selected_views:
        xy_col = _view_col(selected_views, "xy")
        fig.update_xaxes(title_text="x [cm]", row=1, col=xy_col)
        fig.update_yaxes(
            title_text="y [cm]",
            scaleanchor=_axis_ref(selected_views, "xy", "x"),
            scaleratio=1,
            constrain="domain",
            row=1,
            col=xy_col,
        )
    if "yz" in selected_views:
        yz_col = _view_col(selected_views, "yz")
        fig.update_xaxes(title_text="y [cm]", row=1, col=yz_col)
        fig.update_yaxes(
            title_text="z [cm]",
            scaleanchor=_axis_ref(selected_views, "yz", "x"),
            scaleratio=1,
            constrain="domain",
            row=1,
            col=yz_col,
        )
    if "3d" in selected_views:
        scene_name = _scene_layout_name(selected_views, "3d")
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis_title="x [cm]",
                    yaxis_title="y [cm]",
                    zaxis_title="z [cm]",
                    aspectmode="data",
                    zaxis=dict(autorange="reversed"),
                    camera=dict(
                        eye=dict(x=0, y=0, z=2.5),
                        up=dict(x=0, y=1, z=0),
                    ),
                )
            }
        )

    # Particle summary markdown
    above = [(n, e, t) for n, e, t, m in zip(c_names, c_e, threshold_values, mask_above) if m]
    below = [(n, e, t) for n, e, t, m in zip(c_names, c_e, threshold_values, mask_below) if m]
    above_sorted = sorted(above, key=lambda x: x[2], reverse=True)
    below_sorted = sorted(below, key=lambda x: x[2], reverse=True)

    def _particle_summary(name: str, energy: float, threshold_value: float) -> str:
        return (
            f"{name} ({threshold_label} {threshold_value:.2f}{threshold_suffix}, "
            f"E {energy:.2f} GeV)"
        )

    summary = f"**Particles above {threshold_text} ({len(above_sorted)}):** "
    summary += ", ".join(_particle_summary(n, e, t) for n, e, t in above_sorted) or "—"
    if below_sorted:
        summary += f"\n\n**Particles below threshold ({len(below_sorted)}):** "
        summary += ", ".join(_particle_summary(n, e, t) for n, e, t in below_sorted)

    return fig, summary


def _normalize_views(
    view: EventView | Sequence[EventView] | None = None,
    views: Sequence[EventView] | None = None,
) -> tuple[EventView, ...]:
    if views is not None and view is not None:
        raise ValueError("Pass either 'views' or legacy 'view', not both")

    raw_views = views if views is not None else view
    if raw_views is None:
        raw_views = ("xy", "3d")
    elif isinstance(raw_views, str):
        legacy = {"2d": ("xy",), "both": ("xy", "3d")}
        raw_views = legacy.get(raw_views, tuple(v.strip() for v in raw_views.split(",")))

    selected: list[EventView] = []
    valid = {"xy", "yz", "3d"}
    for raw_view in raw_views:
        if raw_view not in valid:
            raise ValueError(
                "views must contain only 'xy', 'yz', and/or '3d' "
                f"(legacy view accepts '2d', '3d', or 'both'), got {raw_view!r}"
            )
        if raw_view not in selected:
            selected.append(raw_view)

    if not selected:
        raise ValueError("At least one event display view must be selected")
    return tuple(selected)


def _view_subplot_spec(view: EventView) -> dict[str, str]:
    return {"type": "scene"} if view == "3d" else {"type": "xy"}


def _default_figure_width(views: Sequence[EventView]) -> int:
    return max(650, 500 * len(views))


def _view_title(view: EventView) -> str:
    if view == "xy":
        return f"XY Plane (z≈{HGCAL_Z:.1f} cm)"
    if view == "yz":
        return "YZ Plane"
    return "3D View"


def _view_col(views: Sequence[EventView], view: EventView) -> int:
    return views.index(view) + 1


def _scene_layout_name(views: Sequence[EventView], view: EventView) -> str:
    scene_number = sum(v == "3d" for v in views[: views.index(view) + 1])
    return "scene" if scene_number == 1 else f"scene{scene_number}"


def _axis_ref(views: Sequence[EventView], view: EventView, axis: Literal["x", "y"]) -> str:
    axis_number = sum(v in ("xy", "yz") for v in views[: views.index(view) + 1])
    return axis if axis_number == 1 else f"{axis}{axis_number}"


def _add_energy_traces(
    fig: go.Figure,
    h_x,
    h_y,
    h_z,
    h_e,
    views: Sequence[EventView],
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

    h_e_arr = np.asarray(h_e)
    hover = "x=%{x:.2f}<br>y=%{y:.2f}<br>E=%{customdata:.4f} GeV<extra>RecHit</extra>"
    hover3d = (
        "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>E=%{customdata:.4f} GeV<extra>RecHit</extra>"
    )

    if "xy" in views:
        fig.add_trace(
            go.Scatter(
                x=h_x,
                y=h_y,
                mode="markers",
                marker=marker_2d,
                name="RecHits",
                legendgroup="rechits",
                customdata=h_e_arr,
                hovertemplate=hover,
            ),
            row=1,
            col=_view_col(views, "xy"),
        )

    if "yz" in views:
        fig.add_trace(
            go.Scatter(
                x=h_y,
                y=h_z,
                mode="markers",
                marker=marker_2d,
                name="RecHits",
                legendgroup="rechits",
                showlegend="xy" not in views,
                customdata=h_e_arr,
                hovertemplate="y=%{x:.2f}<br>z=%{y:.2f}<br>E=%{customdata:.4f} GeV<extra>RecHit</extra>",
            ),
            row=1,
            col=_view_col(views, "yz"),
        )

    if "3d" in views:
        fig.add_trace(
            go.Scatter3d(
                x=h_x,
                y=h_y,
                z=h_z,
                mode="markers",
                marker=marker_3d,
                name="RecHits",
                legendgroup="rechits",
                showlegend=("xy" not in views and "yz" not in views),
                customdata=h_e_arr,
                hovertemplate=hover3d,
            ),
            row=1,
            col=_view_col(views, "3d"),
        )


def _add_truth_traces(
    fig: go.Figure,
    h_x,
    h_y,
    h_z,
    h_e,
    hit_object_id: np.ndarray,
    object_labels: dict[int, str],
    views: Sequence[EventView],
    below_threshold_object_ids: set[int] | None = None,
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
    below_threshold_object_ids = below_threshold_object_ids or set()

    for obj in sorted_objects:
        mask = hit_object_id == obj
        is_below_threshold = obj in below_threshold_object_ids
        color = "#888888" if is_below_threshold else _cluster_color(obj)
        label = object_labels.get(obj, f"Cluster {obj}")
        size_2d = 2 if obj == 0 else np.clip(3 + 3 * np.sqrt(h_e[mask]), 2, 10).tolist()
        size_3d = 1 if obj == 0 else np.clip(2 + 2 * np.sqrt(h_e[mask]), 1, 6).tolist()
        opacity = 0.3 if obj == 0 else (0.45 if is_below_threshold else 0.85)

        hover = f"x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>E=%{{customdata:.4f}} GeV<extra>{label}</extra>"
        hover3d = (
            f"x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<br>"
            f"E=%{{customdata:.4f}} GeV<extra>{label}</extra>"
        )

        if "xy" in views:
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
                col=_view_col(views, "xy"),
            )

        if "yz" in views:
            fig.add_trace(
                go.Scatter(
                    x=h_y[mask],
                    y=h_z[mask],
                    mode="markers",
                    marker=dict(size=size_2d, color=color, opacity=opacity),
                    name=label,
                    legendgroup=label,
                    showlegend="xy" not in views,
                    customdata=h_e[mask],
                    hovertemplate=f"y=%{{x:.2f}}<br>z=%{{y:.2f}}<br>E=%{{customdata:.4f}} GeV<extra>{label}</extra>",
                ),
                row=1,
                col=_view_col(views, "yz"),
            )

        if "3d" in views:
            fig.add_trace(
                go.Scatter3d(
                    x=h_x[mask],
                    y=h_y[mask],
                    z=h_z[mask],
                    mode="markers",
                    marker=dict(size=size_3d, color=color, opacity=opacity),
                    name=label,
                    legendgroup=label,
                    showlegend=("xy" not in views and "yz" not in views),
                    customdata=h_e[mask],
                    hovertemplate=hover3d,
                ),
                row=1,
                col=_view_col(views, "3d"),
            )


def _add_cluster_markers(
    fig: go.Figure,
    c_x,
    c_y,
    c_z,
    c_e,
    c_threshold,
    c_names,
    color: str,
    label: str,
    threshold_label: str,
    threshold_unit: str,
    views: Sequence[EventView],
) -> None:
    """Add SimCluster impact point markers to XY and 3D views."""
    if len(c_x) == 0:
        return

    threshold_suffix = f" {threshold_unit}" if threshold_unit else ""
    hover = (
        "<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<br>"
        "E=%{customdata[0]:.3f} GeV<br>"
        f"{threshold_label}=%{{customdata[1]:.3f}}{threshold_suffix}<extra>Cluster</extra>"
    )
    hover3d = (
        "<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>"
        "E=%{customdata[0]:.3f} GeV<br>"
        f"{threshold_label}=%{{customdata[1]:.3f}}{threshold_suffix}<extra>Cluster</extra>"
    )
    customdata = np.column_stack([c_e, c_threshold])

    if "xy" in views:
        fig.add_trace(
            go.Scatter(
                x=c_x,
                y=c_y,
                mode="markers",
                text=c_names,
                marker=dict(size=14, color=color, line=dict(color="black", width=1)),
                name=label,
                legendgroup=label,
                customdata=customdata,
                hovertemplate=hover,
            ),
            row=1,
            col=_view_col(views, "xy"),
        )
    if "yz" in views:
        fig.add_trace(
            go.Scatter(
                x=c_y,
                y=c_z,
                mode="markers",
                text=c_names,
                marker=dict(size=14, color=color, line=dict(color="black", width=1)),
                name=label,
                legendgroup=label,
                showlegend="xy" not in views,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{text}</b><br>y=%{x:.2f}<br>z=%{y:.2f}<br>"
                    "E=%{customdata[0]:.3f} GeV<br>"
                    f"{threshold_label}=%{{customdata[1]:.3f}}{threshold_suffix}<extra>Cluster</extra>"
                ),
            ),
            row=1,
            col=_view_col(views, "yz"),
        )
    if "3d" in views:
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
                showlegend=("xy" not in views and "yz" not in views),
                customdata=customdata,
                hovertemplate=hover3d,
            ),
            row=1,
            col=_view_col(views, "3d"),
        )
