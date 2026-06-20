import numpy as np

from fastgnn.data.cmssw.plotting import plot_event_display


def _event_kwargs() -> dict[str, np.ndarray]:
    return {
        "h_x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "h_y": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "h_z": np.array([320.0, 321.0, 322.0, 323.0], dtype=np.float32),
        "h_e": np.array([1.0, 2.0, 2.0, 0.5], dtype=np.float32),
        "hit_object_id": np.array([1, 1, 2, 0], dtype=np.int32),
        "c_eta": np.array([0.01, 0.02], dtype=np.float32),
        "c_phi": np.array([0.0, 0.2], dtype=np.float32),
        "c_e": np.array([10.0, 2.0], dtype=np.float32),
        "c_pdg": np.array([22, 11], dtype=np.int32),
    }


def _trace_by_name(fig, name: str):
    matches = [trace for trace in fig.data if trace.name == name]
    assert len(matches) == 1
    return matches[0]


def test_event_display_threshold_values_select_clusters_independently_of_energy() -> None:
    kwargs = _event_kwargs()

    energy_fig, energy_summary = plot_event_display(
        **kwargs,
        mode="energy",
        views=["xy"],
        energy_threshold=3.0,
    )
    pt_fig, pt_summary = plot_event_display(
        **kwargs,
        mode="energy",
        views=["xy"],
        energy_threshold=3.0,
        cluster_threshold_values=np.array([1.0, 5.0], dtype=np.float32),
        cluster_threshold_label="impact pT",
    )

    energy_clusters = _trace_by_name(energy_fig, "Clusters")
    pt_clusters = _trace_by_name(pt_fig, "Clusters")

    np.testing.assert_allclose(energy_clusters.customdata[:, 0], [10.0])
    np.testing.assert_allclose(energy_clusters.customdata[:, 1], [10.0])
    np.testing.assert_allclose(pt_clusters.customdata[:, 0], [2.0])
    np.testing.assert_allclose(pt_clusters.customdata[:, 1], [5.0])
    assert "impact energy 3 GeV" in energy_summary
    assert "impact pT 3 GeV" in pt_summary


def test_truth_coloring_greys_hits_from_below_threshold_objects() -> None:
    fig, _ = plot_event_display(
        **_event_kwargs(),
        mode="truth",
        views=["xy"],
        energy_threshold=3.0,
        cluster_threshold_values=np.array([5.0, 1.0], dtype=np.float32),
        cluster_threshold_label="impact pT",
    )

    above_trace = next(trace for trace in fig.data if "E=10.00 GeV" in trace.name)
    below_trace = next(trace for trace in fig.data if "E=2.00 GeV" in trace.name)

    assert above_trace.marker.color != "#888888"
    assert below_trace.marker.color == "#888888"
    assert below_trace.marker.opacity == 0.45
