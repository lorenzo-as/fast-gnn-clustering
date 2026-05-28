# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import awkward as ak
    import marimo as mo
    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np

    from fastgnn import get_project_root
    from fastgnn.data import CaloDataset
    from fastgnn.data.cmssw.plotting import plot_event
    from fastgnn.datasets import DatasetRegistry

    mplhep.style.use("CMS")

    PROJECT_ROOT = get_project_root()
    PROCESSED_DIR = PROJECT_ROOT / "data/processed/cmssw"
    return (
        CaloDataset,
        DatasetRegistry,
        PROCESSED_DIR,
        ak,
        mo,
        mplhep,
        np,
        plot_event,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load parquet data

    Convert CMSSW ROOT files with `fastgnn-convert-cmssw`; this notebook only loads and inspects processed datasets.
    """)
    return


@app.cell(hide_code=True)
def _(DatasetRegistry, PROCESSED_DIR, mo):
    dataset_registry = DatasetRegistry(PROCESSED_DIR)
    dataset_table = dataset_registry.table()
    dataset_selector = mo.ui.table(
        dataset_table,
        selection="single",
        pagination=True,
        page_size=10,
        label="Select processed dataset",
    )
    max_load_events = mo.ui.number(
        label="Max events to load (all if 0)",
        value=0,
        start=0,
        step=100,
    )

    mo.vstack([dataset_selector, max_load_events])
    return dataset_selector, max_load_events


@app.cell(hide_code=True)
def _(CaloDataset, PROCESSED_DIR, dataset_selector, max_load_events, mo):
    _rows = dataset_selector.value.to_dicts()
    mo.stop(len(_rows) == 0, mo.md("Select a processed dataset above."))

    dataset_path = PROCESSED_DIR / _rows[0]["relative_path"]
    ds = CaloDataset(dataset_path, max_events=int(max_load_events.value) or None)
    data = ds.events

    mo.md(f"Loaded **{len(ds)} events** from `{dataset_path}`.")
    return data, ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview Plots
    """)
    return


@app.cell
def _():
    NBINS = 40  # Keep all histograms bars same width
    return (NBINS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hit Level Statistics
    """)
    return


@app.cell(hide_code=True)
def _(NBINS, ak, data, mplhep, np, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(15, 6))
    _axs = _axs.flatten()

    # distribution of number of hits per event, separated by signal vs noise
    _n_total = ak.to_numpy(ak.num(data.hits.x))
    _n_sig = ak.to_numpy(ak.sum(data.truth.hit_object_id > 0, axis=1))
    _n_noise = ak.to_numpy(ak.sum(data.truth.hit_object_id == 0, axis=1))
    assert ak.all(_n_total == _n_sig + _n_noise)
    _hist_sig_and_noise, _edges = np.histogram(_n_total, bins=NBINS)

    _ax = _axs[0]
    mplhep.histplot(
        [_hist_sig_and_noise],
        bins=_edges,
        histtype="fill",
        alpha=0.7,
        ax=_ax,
    )

    # mplhep.add_text(
    #     "Mean: {:.1f} hits/event".format(np.mean(_n_total)),
    #     loc="upper right",
    #     ax=_ax,
    # )
    _ax.set_xlabel("Number of hits per event")
    _ax.set_xlim(_edges[0], _edges[-1])
    _ax.set_ylabel("Number of events")
    mplhep.yscale_legend(soft_fail=True)

    # distribution of hit energy, separated by signal vs noise
    _sig_mask = data.truth.hit_object_id > 0
    _energy_sig = ak.flatten(data.hits.energy[_sig_mask])
    _energy_noise = ak.flatten(data.hits.energy[~_sig_mask])

    _exp_range = -3, 3
    _edges = np.logspace(-3, 3, NBINS)
    assert ak.all((_energy_sig >= _edges[0]) & (_energy_sig <= _edges[-1])) and ak.all(
        (_energy_noise >= _edges[0]) & (_energy_noise <= _edges[-1])
    ), (
        "Some hit energies are outside the expected range. Check the data or adjust the histogram edges."
    )
    _hist_sig, _edges = np.histogram(
        _energy_sig, bins=_edges, range=(0, np.percentile(_energy_sig, 99))
    )
    _hist_noise, _ = np.histogram(_energy_noise, bins=_edges)

    _ax = _axs[1]
    mplhep.histplot(
        [_hist_sig, _hist_noise],
        bins=_edges,
        histtype="fill",
        alpha=0.7,
        ax=_ax,
        label=["Signal hits", "Noise hits"],
    )
    _ax.set_xticks([10**i for i in range(_exp_range[0], _exp_range[1] + 1)])
    _ax.set_xlim(10 ** _exp_range[0], 10 ** _exp_range[1])
    _ax.set_xlabel("Hit energy [GeV]")
    _ax.set_ylabel("Number of hits")
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.legend()
    mplhep.yscale_legend(soft_fail=True)

    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cluster-level Statistics
    """)
    return


@app.cell(hide_code=True)
def _(ds, mo):
    _preprocessing_cluster_energy_threshold = ds.metadata["preprocessing"][
        "truth_min_object_energy"
    ]
    mo.stop(
        _preprocessing_cluster_energy_threshold is not None
        and _preprocessing_cluster_energy_threshold > 0,
        mo.md(
            "Dataset was preprocessed with a minimum truth object energy > 0, so cluster energy threshold selection is not applicable."
        ),
    )
    cluster_energy_threshold_selector = mo.ui.text(
        label="Cluster energy thresholds [GeV]",
        value="0, 1,",
        full_width=True,
    )
    cluster_energy_threshold_selector
    return (cluster_energy_threshold_selector,)


@app.cell(hide_code=True)
def _(NBINS, ak, cluster_energy_threshold_selector, data, mplhep, np, plt):
    _thresholds = np.array(
        [
            float(_threshold.strip())
            for _threshold in str(cluster_energy_threshold_selector.value)
            .replace(";", ",")
            .split(",")
            if _threshold.strip()
        ],
        dtype=float,
    )
    _thresholds = _thresholds[np.isfinite(_thresholds)]
    if len(_thresholds) == 0:
        _thresholds = np.array([0.0])

    _mask_visible = data.truth.objects.is_visible
    _mask_has_hits = data.truth.objects.n_hits > 0
    _base_mask = _mask_visible & _mask_has_hits

    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _threshold_label(_threshold, _extra=""):
        _label = rf"$E_\mathrm{{C}} > {_threshold:.1f}$ GeV"
        return f"{_label}{_extra}"

    def _values_for_threshold(_field, _threshold):
        _mask = _base_mask & (data.truth.objects.impact_energy > _threshold)
        return ak.to_numpy(ak.flatten(_field[_mask]))

    def _finite_values(_field):
        _values = ak.to_numpy(ak.flatten(_field[_base_mask]))
        return _values[np.isfinite(_values)]

    def _linear_edges(_values, _lower=None, _upper=None):
        assert len(_values) > 0, (
            "No valid values found to determine histogram edges. Check the data or adjust the filters."
        )
        _min_val = np.min(_values)
        _max_val = np.max(_values)
        _lo = _min_val if _lower is None else _lower
        _hi = _max_val if _upper is None else _upper
        assert _min_val >= _lo and _max_val <= _hi, (
            f"Some values are outside the specified range [{_lo}, {_hi}]. Check the data or adjust the range."
        )
        return np.linspace(_lo, _hi, NBINS)

    def _plot_overlaid_hist(_ax, _field, _edges, _xlabel, _ylabel, _yscale="log"):
        for _idx, _threshold in enumerate(_thresholds):
            _values = _values_for_threshold(_field, _threshold)
            _hist, _ = np.histogram(_values, bins=_edges)
            mplhep.histplot(
                _hist,
                bins=_edges,
                histtype="step",
                linewidth=2,
                color=_colors[_idx % len(_colors)],
                ax=_ax,
                label=_threshold_label(_threshold),
            )
        _ax.set_xlim(_edges[0], _edges[-1])
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel(_ylabel)
        if _yscale:
            _ax.set_yscale(_yscale)
        _ax.legend(loc="upper right")
        mplhep.yscale_legend(soft_fail=True)

    _fig, _axs = plt.subplots(3, 2, figsize=(15, 20))
    _axs = _axs.flatten()

    # distribution of visible objects passing the energy threshold and having at least one associated hit per event
    _overflow_threshold = 80
    _edges = np.linspace(0, _overflow_threshold, NBINS)

    _ax = _axs[0]
    for _idx, _threshold in enumerate(_thresholds):
        _mask = _base_mask & (data.truth.objects.impact_energy > _threshold)
        _n_objects = ak.to_numpy(ak.sum(_mask, axis=1))
        _n_overflow = int(np.sum(_n_objects >= _overflow_threshold))
        _n_objects_plot = np.clip(_n_objects, _edges[0], np.nextafter(_edges[-1], _edges[0]))
        _hist_n_objects, _ = np.histogram(_n_objects_plot, bins=_edges)
        mplhep.histplot(
            _hist_n_objects,
            bins=_edges,
            histtype="step",
            linewidth=2,
            color=_colors[_idx % len(_colors)],
            ax=_ax,
            label=_threshold_label(_threshold, rf" (overflow: {_n_overflow})"),
        )
    _ax.axvline(_overflow_threshold, color="black", linestyle="--", linewidth=1.5)
    _ax.set_xlabel("Number of clusters per event")
    _ax.set_ylabel("Number of events")
    _ax.legend(loc="upper right")
    _xticks = np.arange(0, _overflow_threshold + 1, 10)
    _ax.set_xticks(_xticks)
    _ax.set_xlim(_edges[0], _edges[-1])
    mplhep.yscale_legend(soft_fail=True)

    # distribution of number of hits per object
    _edges = _linear_edges(
        _finite_values(data.truth.objects.n_hits),
    )
    _ax = _axs[1]
    _plot_overlaid_hist(
        _ax,
        data.truth.objects.n_hits,
        _edges,
        "Number of hits per cluster",
        "Number of clusters",
    )

    # distribution of object impact energy, pT, eta, phi (if available in the dataset)
    if all(
        field in data.truth.objects.fields
        for field in ["impact_energy", "impact_pt", "impact_eta", "impact_phi", "is_visible"]
    ):
        _ax = _axs[2]
        _edges = _linear_edges(
            _finite_values(data.truth.objects.impact_energy),
        )
        _plot_overlaid_hist(
            _ax,
            data.truth.objects.impact_energy,
            _edges,
            r"Cluster energy (GeV)",
            "Number of clusters",
        )

        _ax = _axs[3]
        _edges = _linear_edges(
            _finite_values(data.truth.objects.impact_pt),
        )
        _plot_overlaid_hist(
            _ax,
            data.truth.objects.impact_pt,
            _edges,
            "Cluster $p_T$ (GeV)",
            "Number of clusters",
        )

        _ax = _axs[4]
        _edges = _linear_edges(_finite_values(data.truth.objects.impact_eta), _lower=0, _upper=6)
        _plot_overlaid_hist(
            _ax,
            data.truth.objects.impact_eta,
            _edges,
            r"Cluster $\eta$",
            "Number of clusters",
        )

        _ax = _axs[5]
        _edges = np.linspace(-np.pi, np.pi, NBINS)
        _plot_overlaid_hist(
            _ax,
            data.truth.objects.impact_phi,
            _edges,
            r"Cluster $\phi$",
            "Number of clusters",
            _yscale=None,
        )
        _ax.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            labels=[r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )

    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(ds, mo):
    _preprocessing_cluster_energy_threshold = ds.metadata["preprocessing"][
        "truth_min_object_energy"
    ]
    mo.stop(
        _preprocessing_cluster_energy_threshold is not None
        and _preprocessing_cluster_energy_threshold > 0,
        mo.md(
            "Dataset was preprocessed with a minimum truth object energy > 0.\n This plot should be viewed with minimum cluster energy threshold 0.)"
        ),
    )

    energy_threshold_plot_xlog_selector = mo.ui.checkbox(
        label="Use logarithmic scale for cluster energy threshold", value=True
    )
    _aggregation_options = [
        "mean",
        "median",
        "min",
        "max",
    ]
    objects_per_event_aggregation_selector = mo.ui.multiselect(
        options=_aggregation_options,
        value=[
            "mean",
        ],
        label="Clusters/event aggregations",
    )
    hits_per_object_aggregation_selector = mo.ui.multiselect(
        options=_aggregation_options,
        value=[
            "mean",
        ],
        label="Hits/cluster aggregations",
    )

    mo.vstack(
        [
            energy_threshold_plot_xlog_selector,
            mo.hstack(
                [
                    objects_per_event_aggregation_selector,
                    hits_per_object_aggregation_selector,
                ]
            ),
        ]
    )
    return (
        energy_threshold_plot_xlog_selector,
        hits_per_object_aggregation_selector,
        objects_per_event_aggregation_selector,
    )


@app.cell(hide_code=True)
def _(
    ak,
    data,
    energy_threshold_plot_xlog_selector,
    hits_per_object_aggregation_selector,
    np,
    objects_per_event_aggregation_selector,
    plt,
):
    _left_aggregations = list(objects_per_event_aggregation_selector.value) or ["mean"]
    _right_aggregations = list(hits_per_object_aggregation_selector.value) or ["mean"]
    _quantile = None
    _n_thresholds = 50
    _xlog = energy_threshold_plot_xlog_selector.value
    if _xlog:
        _threshold_min, _threshold_max = -3, 2
    else:
        _threshold_min, _threshold_max = 0.0, 20.0

    _mask_visible = data.truth.objects.is_visible
    _mask_has_hits = data.truth.objects.n_hits > 0
    _base_mask = _mask_visible & _mask_has_hits

    _cluster_energy = ak.to_numpy(ak.flatten(data.truth.objects.impact_energy[_base_mask]))
    _cluster_energy = _cluster_energy[np.isfinite(_cluster_energy)]
    assert len(_cluster_energy) > 0, "No visible clusters with hits found."

    if _threshold_max is None:
        _threshold_max = float(np.max(_cluster_energy))
    _get_edges_func = np.logspace if _xlog else np.linspace
    _thresholds = _get_edges_func(_threshold_min, _threshold_max, _n_thresholds)

    def _aggregate(_values, _aggregation):
        _values = np.asarray(_values)
        _values = _values[np.isfinite(_values)]
        if len(_values) == 0:
            return np.nan
        if _aggregation == "mean":
            return np.mean(_values)
        if _aggregation == "median":
            return np.median(_values)
        if _aggregation == "min":
            return np.min(_values)
        if _aggregation == "max":
            return np.max(_values)
        if _aggregation == "quantile":
            return np.quantile(_values, _quantile)
        raise ValueError(f"Unsupported aggregation: {_aggregation}")

    def _aggregation_label(_aggregation):
        if _aggregation == "quantile":
            return f"{_quantile:g} quantile"
        return _aggregation

    def _axis_ylabel(_base_label, _aggregations):
        if len(_aggregations) == 1:
            return f"{_aggregation_label(_aggregations[0]).capitalize()} {_base_label}"
        return _base_label.capitalize()

    def _maybe_add_legend(_ax, _aggregations):
        if len(_aggregations) > 1:
            _ax.legend()

    _objects_per_event = {aggregation: [] for aggregation in _left_aggregations}
    _hits_per_object = {aggregation: [] for aggregation in _right_aggregations}

    for _threshold in _thresholds:
        _mask = _base_mask & (data.truth.objects.impact_energy > _threshold)
        _n_objects = ak.to_numpy(ak.sum(_mask, axis=1))
        _n_hits = ak.to_numpy(ak.flatten(data.truth.objects.n_hits[_mask]))
        for _aggregation in _left_aggregations:
            _objects_per_event[_aggregation].append(_aggregate(_n_objects, _aggregation))
        for _aggregation in _right_aggregations:
            _hits_per_object[_aggregation].append(_aggregate(_n_hits, _aggregation))

    _fig, _axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    _ax = _axs[0]
    for _aggregation, _values in _objects_per_event.items():
        _ax.plot(
            _thresholds, _values, linewidth=2, label=_aggregation_label(_aggregation).capitalize()
        )
    _ax.set_xlabel(r"Cluster energy threshold $E_\mathrm{C}$ [GeV]")
    if _xlog:
        _ax.set_xscale("log")
    _ax.set_xlim(_thresholds[0], _thresholds[-1])
    _max_objects = np.nanmax([_values for _values in _objects_per_event.values()])
    _ax.set_yticks(np.arange(1, _max_objects + 1.5, 1), minor=True)
    _ax.set_ylim(1e-3, None)
    _ax.set_ylabel(_axis_ylabel("clusters per event", _left_aggregations))
    _ax.grid(alpha=0.3)
    _maybe_add_legend(_ax, _left_aggregations)

    _ax = _axs[1]
    for _aggregation, _values in _hits_per_object.items():
        _ax.plot(
            _thresholds, _values, linewidth=2, label=_aggregation_label(_aggregation).capitalize()
        )
    _ax.set_xlabel(r"Cluster energy threshold $E_\mathrm{C}$ [GeV]")
    if _xlog:
        _ax.set_xscale("log")
    _ax.set_xlim(_thresholds[0], _thresholds[-1])
    _ax.set_ylabel(_axis_ylabel("hits per cluster", _right_aggregations))
    _ax.grid(alpha=0.3)
    _maybe_add_legend(_ax, _right_aggregations)

    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Event Display
    """)
    return


@app.cell(hide_code=True)
def _(ds, mo):
    event_slider = mo.ui.number(
        start=0,
        stop=len(ds) - 1,
        step=1,
        value=0,
        label="Event index",
    )
    color_mode = mo.ui.radio(
        options=["energy", "hit_object_id"],
        value="energy",
        label="Hit colour mode",
    )
    event_display_view = mo.ui.radio(
        options={"2D": "2d", "3D": "3d", "Both": "both"},
        value="Both",
        label="Event display view",
    )
    show_cluster_markers = mo.ui.checkbox(value=True, label="Show SimCluster markers")
    threshold_input = mo.ui.number(
        start=0.0,
        step=0.1,
        value=0.0,
        label="SimCluster E threshold [GeV]",
    )
    show_below = mo.ui.checkbox(value=True, label="Show clusters below threshold")

    mo.vstack(
        [
            mo.hstack(
                [event_slider, color_mode, event_display_view, threshold_input], justify="start"
            ),
            show_cluster_markers,
            show_below,
        ]
    )
    return (
        color_mode,
        event_display_view,
        event_slider,
        show_below,
        show_cluster_markers,
        threshold_input,
    )


@app.cell(hide_code=True)
def _(
    color_mode,
    ds,
    event_display_view,
    event_slider,
    mo,
    plot_event,
    show_below,
    show_cluster_markers,
    threshold_input,
):
    event = ds[int(event_slider.value or 0)]
    hit_object_id = event.truth.hit_object_id
    n_sig = int((hit_object_id > 0).sum())
    n_noise = int((hit_object_id == 0).sum())

    fig, summary = plot_event(
        event,
        color_by=color_mode.value,
        view=event_display_view.value,
        show_cluster_markers=show_cluster_markers.value,
        energy_threshold=float(threshold_input.value or 0.0),
        show_clusters_below_threshold=show_below.value,
    )

    mo.vstack(
        [
            mo.md(
                f"**Event {event.event_id}** | "
                f"{event.n_hits} hits | "
                f"{n_sig} signal hits | {n_noise} noise hits | "
                f"{event.n_objects} SimClusters"
            ),
            mo.ui.plotly(fig),
            mo.md(summary),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Summary Statistics
    """)
    return


@app.cell(hide_code=True)
def _(ds, mo):
    stats = ds.summary()

    mo.md(f"""
    **Dataset summary ({stats["n_events"]} events)**

    | Metric | Value |
    |---|---|
    | Features | {", ".join(stats["feature_names"])} |
    | Hit fields | {", ".join(stats["fields"]["hits"])} |
    | Truth fields | {", ".join(stats["fields"]["truth"])} |
    | Object fields | {", ".join(stats["fields"]["truth.objects"])} |
    | Hits/event | {stats["hits_per_event"]["mean"]:.0f} ± {stats["hits_per_event"]["std"]:.0f} ({stats["hits_per_event"]["min"]}–{stats["hits_per_event"]["max"]}) |
    | Objects/event | {stats["objects_per_event"]["mean"]:.1f} ± {stats["objects_per_event"]["std"]:.1f} ({stats["objects_per_event"]["min"]}–{stats["objects_per_event"]["max"]}) |
    | Noise fraction | {stats["noise_fraction"]["mean"]:.2%} ± {stats["noise_fraction"]["std"]:.2%} |
    """)
    return


if __name__ == "__main__":
    app.run()
