import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import awkward as ak
    import marimo as mo
    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np
    import polars as pl

    from fastgnn import get_project_root
    from fastgnn.data import CaloDataset
    from fastgnn.data.base import PadCollator
    from fastgnn.datasets import DatasetRegistry

    mplhep.style.use("CMS")

    PROJECT_ROOT = get_project_root()
    PROCESSED_DIR = PROJECT_ROOT / "data/processed/cmssw"
    return (
        CaloDataset,
        DatasetRegistry,
        PROCESSED_DIR,
        PadCollator,
        ak,
        mo,
        mplhep,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select Baseline Dataset

    Select one zero-threshold processed dataset. The notebook replays the hit,
    cluster, sum-energy, normalization, and padding/truncation steps from
    this single baseline.
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
        label="Zero-threshold processed dataset",
    )
    max_load_events = mo.ui.number(
        label="Max events to load (all if 0)",
        value=0,
        start=0,
        step=100,
    )
    hit_min_energy_input = mo.ui.number(
        label="Hit energy threshold [GeV]",
        value=0.2,
        start=0.0,
        step=0.05,
    )
    truth_min_energy_input = mo.ui.number(
        label="Cluster impact-energy threshold [GeV]",
        value=1.0,
        start=0.0,
        step=0.5,
    )
    use_sum_energy_threshold = mo.ui.checkbox(
        label="Apply sum-energy threshold",
        value=False,
    )
    sum_energy_min_input = mo.ui.number(
        label="Sum-energy threshold [GeV]",
        value=1.0,
        start=0.0,
        step=0.5,
    )
    make_plots_button = mo.ui.run_button(label="Make plots", kind="success")

    mo.vstack(
        [
            max_load_events,
            dataset_selector,
            mo.hstack([hit_min_energy_input, truth_min_energy_input]),
            mo.hstack([use_sum_energy_threshold, sum_energy_min_input]),
            make_plots_button,
        ]
    )
    return (
        dataset_selector,
        hit_min_energy_input,
        make_plots_button,
        max_load_events,
        sum_energy_min_input,
        truth_min_energy_input,
        use_sum_energy_threshold,
    )


@app.cell(hide_code=True)
def _(
    CaloDataset,
    PROCESSED_DIR,
    dataset_selector,
    hit_min_energy_input,
    make_plots_button,
    max_load_events,
    mo,
    sum_energy_min_input,
    truth_min_energy_input,
    use_sum_energy_threshold,
):
    mo.stop(
        not make_plots_button.value,
        mo.md("Choose the baseline dataset and thresholds, then click **Make plots**."),
    )
    _selected_rows = dataset_selector.value.to_dicts()
    mo.stop(len(_selected_rows) == 0, mo.md("Select one zero-threshold processed dataset."))

    selected_record = _selected_rows[0]
    max_events = int(max_load_events.value) or None
    ds = CaloDataset(PROCESSED_DIR / selected_record["relative_path"], max_events=max_events)
    data = ds.events
    _preprocessing = ds.metadata.get("preprocessing") or {}

    _hit_min_metadata = _preprocessing.get("hit_min_energy")
    _truth_min_metadata = _preprocessing.get("truth_min_object_energy")
    _sum_energy_min_metadata = _preprocessing.get("truth_min_sum_energy")
    mo.stop(
        _hit_min_metadata not in (None, 0, 0.0),
        mo.md("Baseline dataset must have no hit-energy threshold."),
    )
    mo.stop(
        _truth_min_metadata not in (None, 0, 0.0),
        mo.md("Baseline dataset must have `truth_min_object_energy` unset or 0."),
    )
    mo.stop(
        _sum_energy_min_metadata not in (None, 0, 0.0),
        mo.md("Baseline dataset must have no sum-energy threshold."),
    )

    _required_hit_fields = {"x", "y", "z", "energy", "cluster0", "n_clusters"}
    _required_object_fields = {"impact_energy", "n_hits", "sum_energy"}
    mo.stop(
        not _required_hit_fields.issubset(ds.fields["hits"]),
        mo.md(
            f"Baseline dataset missing hit fields: `{sorted(_required_hit_fields - set(ds.fields['hits']))}`"
        ),
    )
    mo.stop(
        not _required_object_fields.issubset(ds.fields["truth.objects"]),
        mo.md(
            f"Baseline dataset missing truth object fields: "
            f"`{sorted(_required_object_fields - set(ds.fields['truth.objects']))}`"
        ),
    )

    thresholds = {
        "hit_min_energy": float(hit_min_energy_input.value or 0.0),
        "truth_min_object_energy": float(truth_min_energy_input.value or 0.0),
        "truth_min_sum_energy": (
            float(sum_energy_min_input.value or 0.0) if use_sum_energy_threshold.value else None
        ),
    }

    sum_energy_threshold_text = (
        "disabled"
        if thresholds["truth_min_sum_energy"] is None
        else f">= **{thresholds['truth_min_sum_energy']:.3g} GeV**"
    )
    mo.md(
        f"Loaded **{len(ds)} events** from `{selected_record['relative_path']}`.\n\n"
        f"Replayed thresholds: hit >= **{thresholds['hit_min_energy']:.3g} GeV**, "
        f"cluster impact energy >= **{thresholds['truth_min_object_energy']:.3g} GeV**, "
        f"sum energy {sum_energy_threshold_text}."
    )
    return data, ds, selected_record, thresholds


@app.cell(hide_code=True)
def _(mplhep, np):
    NBINS = 40
    SPLIT_NAMES = ("train", "val", "test")

    def finite(values):
        values = np.asarray(values).reshape(-1)
        return values[np.isfinite(values)]

    def linear_edges(*arrays, bins=NBINS, lower=None, upper=None, integer=False):
        valid_arrays = [finite(array) for array in arrays if len(finite(array))]
        values = finite(np.concatenate(valid_arrays)) if valid_arrays else np.asarray([])
        if len(values) == 0:
            return np.linspace(0.0, 1.0, bins + 1)
        lo = float(np.min(values) if lower is None else lower)
        hi = float(np.max(values) if upper is None else upper)
        if lo == hi:
            hi = lo + 1.0
        if integer:
            lo = np.floor(lo)
            hi = np.ceil(hi)
        return np.linspace(lo, hi, bins + 1)

    def log_edges(*arrays, bins=NBINS):
        valid_arrays = [finite(array) for array in arrays if len(finite(array))]
        values = finite(np.concatenate(valid_arrays)) if valid_arrays else np.asarray([])
        values = values[values > 0]
        if len(values) == 0:
            return np.logspace(-3, 1, bins + 1)
        lo = 10 ** np.floor(np.log10(np.min(values)))
        hi = 10 ** np.ceil(np.log10(np.max(values)))
        if lo == hi:
            hi = lo * 10.0
        return np.logspace(np.log10(lo), np.log10(hi), bins + 1)

    def hist_step(ax, values, edges, label=None, color=None, linewidth=2, density=False):
        values = finite(values)
        if len(values) == 0:
            return
        hist, _ = np.histogram(values, bins=edges, density=density)
        mplhep.histplot(
            hist,
            bins=edges,
            histtype="step",
            linewidth=linewidth,
            label=label,
            color=color,
            ax=ax,
        )

    def hist_fill(ax, values, edges, label=None, alpha=0.7, density=False):
        values = finite(values)
        if len(values) == 0:
            return
        hist, _ = np.histogram(values, bins=edges, density=density)
        mplhep.histplot(hist, bins=edges, histtype="fill", alpha=alpha, label=label, ax=ax)

    def style_hist_axis(ax, legend=False, legend_loc="upper right"):
        if legend:
            ax.legend(loc=legend_loc)
        mplhep.yscale_legend(ax=ax, soft_fail=True)

    def annotate_empty(ax, text):
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    def event_hit_counts(events):
        return np.asarray([len(event["hit_object_id"]) for event in events], dtype=np.int64)

    def event_signal_fraction(events):
        n_hits = event_hit_counts(events)
        n_signal = np.asarray(
            [np.sum(event["hit_object_id"] > 0) for event in events],
            dtype=np.float64,
        )
        out = np.zeros_like(n_signal, dtype=np.float64)
        np.divide(n_signal, n_hits, out=out, where=n_hits > 0)
        return out

    def signal_noise_energy(events):
        signal = []
        noise = []
        for event in events:
            ids = event["hit_object_id"]
            energy = event["hits"]["energy"]
            signal.extend(energy[ids > 0].tolist())
            noise.extend(energy[ids == 0].tolist())
        return finite(signal), finite(noise)

    def positive_objects_per_event(events):
        return np.asarray(
            [int(np.sum(np.unique(event["hit_object_id"]) > 0)) for event in events],
            dtype=np.int64,
        )

    def hits_per_positive_object(events):
        values = []
        for event in events:
            positive = event["hit_object_id"][event["hit_object_id"] > 0]
            if len(positive) == 0:
                continue
            _, counts = np.unique(positive, return_counts=True)
            values.extend(counts.tolist())
        return np.asarray(values, dtype=np.int64)

    return (
        NBINS,
        SPLIT_NAMES,
        event_hit_counts,
        event_signal_fraction,
        finite,
        hist_fill,
        hist_step,
        hits_per_positive_object,
        linear_edges,
        log_edges,
        positive_objects_per_event,
        signal_noise_energy,
        style_hist_axis,
    )


@app.cell(hide_code=True)
def _(PadCollator, ak, data, ds, np, thresholds):
    from fastgnn.data.object_properties import compute_object_properties
    from fastgnn.data.cmssw.preprocessing import compact_hit_object_ids

    def _field_dict(group):
        return {name: np.asarray(ak.to_numpy(group[name])) for name in group.fields}

    def _objects_with_properties(objects, hits, hit_object_id):
        out = {name: np.asarray(values).copy() for name, values in objects.items()}
        n_objects = len(next(iter(out.values()))) if out else 0
        out.update(
            compute_object_properties(
                hits,
                hit_object_id,
                object_ids=np.arange(1, n_objects + 1),
                properties=("sum_energy", "n_hits"),
            )
        )
        return out

    def _copy_event(event):
        return {
            "event_id": int(event["event_id"]),
            "hits": {name: values.copy() for name, values in event["hits"].items()},
            "objects": {name: values.copy() for name, values in event["objects"].items()},
            "hit_object_id": event["hit_object_id"].copy(),
        }

    baseline_events = []
    for row in data:
        hit_object_id = np.asarray(ak.to_numpy(row["truth"]["hit_object_id"]), dtype=np.int32)
        hits = _field_dict(row["hits"])
        objects = _field_dict(row["truth"]["objects"])
        objects = _objects_with_properties(objects, hits, hit_object_id)
        baseline_events.append(
            {
                "event_id": int(row["event_id"]),
                "hits": hits,
                "objects": objects,
                "hit_object_id": hit_object_id,
            }
        )

    hit_filtered_events = []
    for event in baseline_events:
        keep_hits = event["hits"]["energy"] >= thresholds["hit_min_energy"]
        hit_object_id = event["hit_object_id"][keep_hits]
        hits = {
            name: values[keep_hits] if len(values) == len(keep_hits) else values.copy()
            for name, values in event["hits"].items()
        }
        objects = _objects_with_properties(event["objects"], hits, hit_object_id)
        hit_filtered_events.append(
            {
                "event_id": event["event_id"],
                "hits": hits,
                "objects": objects,
                "hit_object_id": hit_object_id.astype(np.int32),
            }
        )

    truth_filtered_events = []
    orphan_counts = []
    orphan_energies = []
    for _event in hit_filtered_events:
        _objects = _event["objects"]
        n_objects = len(_objects["impact_energy"])
        keep_objects = _objects["impact_energy"] >= thresholds["truth_min_object_energy"]
        if thresholds["truth_min_sum_energy"] is not None:
            keep_objects &= _objects["sum_energy"] >= thresholds["truth_min_sum_energy"]

        hit_object_id = _event["hit_object_id"]
        _positive = hit_object_id > 0
        object_indices = hit_object_id[_positive] - 1
        valid = (object_indices >= 0) & (object_indices < n_objects)
        orphan_mask = np.zeros(len(hit_object_id), dtype=bool)
        positive_positions = np.flatnonzero(_positive)
        orphan_mask[positive_positions[valid]] = ~keep_objects[object_indices[valid]]
        orphan_counts.append(int(np.sum(orphan_mask)))
        orphan_energies.extend(_event["hits"]["energy"][orphan_mask].tolist())

        compact, kept_indices = compact_hit_object_ids(hit_object_id, keep_objects)
        kept_objects = {name: values[kept_indices] for name, values in _objects.items()}
        kept_objects = _objects_with_properties(kept_objects, _event["hits"], compact)
        truth_filtered_events.append(
            {
                "event_id": _event["event_id"],
                "hits": {name: values.copy() for name, values in _event["hits"].items()},
                "objects": kept_objects,
                "hit_object_id": compact.astype(np.int32),
            }
        )

    def training_dicts(events, feature_names=None):
        feature_names = list(feature_names or ds.hit_features)
        out = []
        for event in events:
            out.append(
                {
                    "features": np.stack(
                        [event["hits"][name] for name in feature_names],
                        axis=1,
                    ).astype(np.float32),
                    "hit_object_id": event["hit_object_id"].astype(np.int32),
                }
            )
        return out

    def split_events(events, split_name):
        _split_indices = set(map(int, ds.splits.get(split_name, [])))
        return [_event for _event in events if int(_event["event_id"]) in _split_indices]

    def compute_stage_normalization(events):
        train_events = split_events(events, "train") or events
        features = [item["features"] for item in training_dicts(train_events)]
        flat = np.concatenate(features, axis=0)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return {
            name: {"mean": float(mean[index]), "std": float(std[index])}
            for index, name in enumerate(ds.hit_features)
        }

    stage_normalization = compute_stage_normalization(truth_filtered_events)

    def padded_stage(events, max_vertices, truncate, normalize_features=True, seed=0):
        normalization = stage_normalization if normalize_features else None
        collator = PadCollator(
            max_vertices=max_vertices,
            feature_names=ds.hit_features,
            truncate=truncate,
            seed=seed,
            normalization=normalization,
        )
        return collator(training_dicts(events))

    stage = {
        "baseline": baseline_events,
        "hit_filtered": hit_filtered_events,
        "truth_filtered": truth_filtered_events,
        "orphan_counts": np.asarray(orphan_counts, dtype=np.int64),
        "orphan_energies": np.asarray(orphan_energies, dtype=np.float64),
    }
    return (
        padded_stage,
        split_events,
        stage,
        stage_normalization,
        training_dicts,
    )


@app.cell(hide_code=True)
def _(mo, selected_record, thresholds):
    _visible = thresholds["truth_min_sum_energy"]
    mo.md(
        f"""
        ## Replayed Configuration

        | Setting | Value |
        |---|---:|
        | Baseline dataset | `{selected_record["relative_path"]}` |
        | Hit energy threshold | {thresholds["hit_min_energy"]:.3g} GeV |
        | Cluster impact-energy threshold | {thresholds["truth_min_object_energy"]:.3g} GeV |
        | Sum-energy threshold | {"disabled" if _visible is None else f"{_visible:.3g} GeV"} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filter Hits/Vertices
    """)
    return


@app.cell
def _(
    NBINS,
    event_hit_counts,
    event_signal_fraction,
    hist_fill,
    linear_edges,
    log_edges,
    np,
    plt,
    signal_noise_energy,
    stage,
    style_hist_axis,
    thresholds,
):
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    _datasets = [
        ("Pre-filtering", stage["baseline"]),
        ("Filtered", stage["hit_filtered"]),
    ]

    _fig, _axs = plt.subplots(3, 2, figsize=(15, 14))
    for _col, (_title, _events) in enumerate(_datasets):
        for _row in range(3):
            _axs[_row, _col].set_axis_on()
        _axs[0, _col].set_title(_title, fontsize=18, pad=14)

        _n_hits = event_hit_counts(_events)
        _hit_edges = linear_edges(_n_hits, bins=NBINS, lower=0, integer=True)
        hist_fill(_axs[0, _col], _n_hits, _hit_edges, _title)
        _axs[0, _col].set_xlabel("Number of hits per event")
        _axs[0, _col].set_ylabel("Number of events")
        style_hist_axis(_axs[0, _col])

        _signal_fraction = 100.0 * event_signal_fraction(_events)
        _frac_edges = linear_edges(_signal_fraction, bins=NBINS, lower=0, upper=100)
        hist_fill(_axs[1, _col], _signal_fraction, _frac_edges, _title)
        _axs[1, _col].set_xlabel("Signal hits per event [%]")
        _axs[1, _col].set_ylabel("Number of events")
        _axs[1, _col].set_xlim(0, 100)
        style_hist_axis(_axs[1, _col])

        _signal_energy, _noise_energy = signal_noise_energy(_events)
        _energy_edges = log_edges(_signal_energy, _noise_energy, bins=NBINS)
        hist_fill(_axs[2, _col], _signal_energy, _energy_edges, "Signal hits")
        hist_fill(_axs[2, _col], _noise_energy, _energy_edges, "Noise hits")
        _axs[2, _col].set_xlabel("Hit energy [GeV]")
        _axs[2, _col].set_ylabel("Number of hits")
        _axs[2, _col].set_xscale("log")
        _axs[2, _col].set_xlim(1.01e-3, 3e2)
        _axs[2, _col].xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=7))
        _axs[2, _col].xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

        _axs[2, _col].xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
        )
        _axs[2, _col].xaxis.set_minor_formatter(NullFormatter())
        _axs[2, _col].tick_params(axis="x", which="minor", bottom=True)
        _axs[2, _col].set_yscale("log")
        if thresholds["hit_min_energy"] > 0:
            _axs[2, _col].axvline(
                thresholds["hit_min_energy"],
                color="red",
                linestyle="--",
                linewidth=2,
            )
        style_hist_axis(_axs[2, _col], legend=True)

    _fig.suptitle("Hit/vertex filtering validation", y=0.995)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build and Filter Truth Clusters
    """)
    return


@app.cell
def _(
    NBINS,
    hist_fill,
    hist_step,
    hits_per_positive_object,
    linear_edges,
    log_edges,
    np,
    plt,
    positive_objects_per_event,
    stage,
    style_hist_axis,
    thresholds,
):
    _fig, _axs = plt.subplots(3, 3, figsize=(20, 16))
    _datasets = [
        ("Raw", stage["baseline"]),
        ("Post-vertex filtering", stage["hit_filtered"]),
        ("Post-cluster filtering", stage["truth_filtered"]),
    ]

    for _col, (_title, _events) in enumerate(_datasets):
        _clusters = positive_objects_per_event(_events)
        _cluster_edges = linear_edges(_clusters, bins=NBINS, lower=0, integer=True)
        hist_fill(_axs[0, _col], _clusters, _cluster_edges, _title)
        _axs[0, _col].set_title(_title, fontsize=18, pad=14)
        _axs[0, _col].set_xlabel("Clusters per event")
        _axs[0, _col].set_ylabel("Number of events")
        style_hist_axis(_axs[0, _col])

        _hits_per_cluster = hits_per_positive_object(_events)
        _hit_edges = linear_edges(_hits_per_cluster, bins=NBINS, lower=0, integer=True)
        hist_step(_axs[1, _col], _hits_per_cluster, _hit_edges, _title)
        _axs[1, _col].set_xlabel("Hits per cluster")
        _axs[1, _col].set_ylabel("Number of clusters")
        _axs[1, _col].set_yscale("log")
        style_hist_axis(_axs[1, _col])

    _orphan_count_edges = linear_edges(
        stage["orphan_counts"],
        bins=NBINS,
        lower=0.5,
        integer=True,
    )
    hist_fill(_axs[2, 0], stage["orphan_counts"], _orphan_count_edges)
    _axs[2, 0].set_title("Removed cluster hits", fontsize=18, pad=14)
    _axs[2, 0].set_xlabel("Orphan hits per event")
    _axs[2, 0].set_ylabel("Number of events")
    style_hist_axis(_axs[2, 0])

    _orphan_energy_edges = log_edges(stage["orphan_energies"], bins=NBINS)
    hist_step(_axs[2, 1], stage["orphan_energies"], _orphan_energy_edges)
    _axs[2, 1].set_title("Removed cluster-hit energy", fontsize=18, pad=14)
    _axs[2, 1].set_xlabel("Orphan hit energy [GeV]")
    _axs[2, 1].set_ylabel("Number of hits")
    _axs[2, 1].set_xscale("log")
    _axs[2, 1].set_yscale("log")
    style_hist_axis(_axs[2, 1], legend=True)

    _cluster_energy = np.concatenate(
        [
            _event["objects"]["impact_energy"][_event["objects"]["n_hits"] > 0]
            for _event in stage["hit_filtered"]
            if len(_event["objects"]["impact_energy"])
        ]
        or [np.asarray([], dtype=np.float64)]
    )
    _energy_edges = log_edges(_cluster_energy, bins=NBINS)
    hist_step(
        _axs[2, 2],
        _cluster_energy,
        _energy_edges,
    )
    if thresholds["truth_min_object_energy"] > 0:
        _axs[2, 2].axvline(
            thresholds["truth_min_object_energy"],
            color="red",
            linestyle="--",
            linewidth=2,
        )
    _axs[2, 2].set_title("Post-vertex cluster energy", fontsize=18, pad=14)
    _axs[2, 2].set_xlabel("Cluster energy [GeV]")
    _axs[2, 2].set_ylabel("Number of clusters")
    _axs[2, 2].set_xscale("log")
    _axs[2, 2].set_yscale("log")
    style_hist_axis(_axs[2, 2], legend=True)

    _fig.suptitle("Truth-cluster filtering validation", y=0.995)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Truth Cluster Impact-Energy Threshold Scan
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    energy_threshold_plot_xlog_selector = mo.ui.checkbox(
        label="Use logarithmic scale for cluster energy threshold",
        value=True,
    )
    _aggregation_options = ["mean", "median", "min", "max"]
    objects_per_event_aggregation_selector = mo.ui.multiselect(
        options=_aggregation_options,
        value=["mean"],
        label="Clusters/event aggregations",
    )
    hits_per_object_aggregation_selector = mo.ui.multiselect(
        options=_aggregation_options,
        value=["mean"],
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
    compact_hit_labels_for_plot,
    energy_threshold_plot_xlog_selector,
    hits_per_object_aggregation_selector,
    np,
    objects_per_event_aggregation_selector,
    plt,
    stage,
    thresholds,
):
    _left_aggregations = list(objects_per_event_aggregation_selector.value) or ["mean"]
    _right_aggregations = list(hits_per_object_aggregation_selector.value) or ["mean"]
    _n_thresholds = 50
    _xlog = energy_threshold_plot_xlog_selector.value
    if _xlog:
        _threshold_min, _threshold_max = -3, 2
    else:
        _threshold_min, _threshold_max = 0.0, 20.0
    _thresholds = (np.logspace if _xlog else np.linspace)(
        _threshold_min,
        _threshold_max,
        _n_thresholds,
    )

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
        raise ValueError(f"Unsupported aggregation: {_aggregation}")

    def _axis_ylabel(_base_label, _aggregations):
        if len(_aggregations) == 1:
            return f"{_aggregations[0].capitalize()} {_base_label}"
        return _base_label.capitalize()

    def _add_selected_threshold_guides(_ax, _series_by_aggregation, _selected_threshold):
        _xmin, _xmax = _thresholds[0], _thresholds[-1]
        _draw_threshold = _selected_threshold
        if _xlog and _draw_threshold <= 0:
            _draw_threshold = _xmin
        _draw_threshold = float(np.clip(_draw_threshold, _xmin, _xmax))
        _ax.axvline(
            _draw_threshold,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=rf"Selected $E_\mathrm{{C}}={_selected_threshold:.3g}$ GeV",
        )
        for _line, (_aggregation, _values) in zip(_ax.lines[:-1], _series_by_aggregation.items()):
            _values = np.asarray(_values, dtype=np.float64)
            if not np.any(np.isfinite(_values)):
                continue
            _guide_value = float(np.interp(_draw_threshold, _thresholds, _values))
            _ax.hlines(
                _guide_value,
                _xmin,
                _draw_threshold,
                colors=_line.get_color(),
                linestyles="--",
                linewidth=1.4,
                alpha=0.75,
            )
            _ax.plot(
                [_draw_threshold],
                [_guide_value],
                marker="o",
                color=_line.get_color(),
                markersize=5,
            )

    _objects_per_event = {aggregation: [] for aggregation in _left_aggregations}
    _hits_per_object = {aggregation: [] for aggregation in _right_aggregations}
    for _threshold in _thresholds:
        _n_objects = []
        _n_hits = []
        for _event in stage["hit_filtered"]:
            _objects = _event["objects"]
            keep = (_objects["impact_energy"] > _threshold) & (_objects["n_hits"] > 0)
            _n_objects.append(int(np.sum(keep)))
            kept_labels, _ = compact_hit_labels_for_plot(_event["hit_object_id"], keep)
            _positive = kept_labels[kept_labels > 0]
            if len(_positive):
                _, counts = np.unique(_positive, return_counts=True)
                _n_hits.extend(counts.tolist())
        for _aggregation in _left_aggregations:
            _objects_per_event[_aggregation].append(_aggregate(_n_objects, _aggregation))
        for _aggregation in _right_aggregations:
            _hits_per_object[_aggregation].append(_aggregate(_n_hits, _aggregation))

    _fig, _axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    _ax = _axs[0]
    for _aggregation, _values in _objects_per_event.items():
        _ax.plot(_thresholds, _values, linewidth=2, label=_aggregation.capitalize())
    _add_selected_threshold_guides(
        _ax,
        _objects_per_event,
        thresholds["truth_min_object_energy"],
    )
    _ax.set_xlabel(r"Cluster energy threshold $E_\mathrm{C}$ [GeV]")
    if _xlog:
        _ax.set_xscale("log")
    _ax.set_xlim(_thresholds[0], _thresholds[-1])
    _ax.set_ylabel(_axis_ylabel("clusters per event", _left_aggregations))
    _ax.grid(alpha=0.3)
    _ax.legend()

    _ax = _axs[1]
    for _aggregation, _values in _hits_per_object.items():
        _ax.plot(_thresholds, _values, linewidth=2, label=_aggregation.capitalize())
    _add_selected_threshold_guides(
        _ax,
        _hits_per_object,
        thresholds["truth_min_object_energy"],
    )
    _ax.set_xlabel(r"Cluster energy threshold $E_\mathrm{C}$ [GeV]")
    if _xlog:
        _ax.set_xscale("log")
    _ax.set_xlim(_thresholds[0], _thresholds[-1])
    _ax.set_ylabel(_axis_ylabel("hits per cluster", _right_aggregations))
    _ax.grid(alpha=0.3)
    _ax.legend()

    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(np):
    def compact_hit_labels_for_plot(hit_object_id, keep_objects):
        old_to_new = np.zeros(len(keep_objects) + 1, dtype=np.int32)
        kept_indices = np.flatnonzero(keep_objects)
        old_to_new[kept_indices + 1] = np.arange(1, len(kept_indices) + 1, dtype=np.int32)
        valid = (hit_object_id >= 0) & (hit_object_id < len(old_to_new))
        out = np.zeros_like(hit_object_id, dtype=np.int32)
        out[valid] = old_to_new[hit_object_id[valid]]
        return out, kept_indices

    return (compact_hit_labels_for_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Normalization
    """)
    return


@app.cell
def _(
    NBINS,
    SPLIT_NAMES,
    ds,
    finite,
    hist_step,
    linear_edges,
    mo,
    np,
    plt,
    split_events,
    stage,
    stage_normalization,
    style_hist_axis,
    training_dicts,
):
    _feature_names = ds.hit_features
    _n_features = len(_feature_names)
    _fig, _axs = plt.subplots(_n_features, 2, figsize=(11, 3.4 * _n_features), squeeze=False)

    _raw_values_by_feature = {name: [] for name in _feature_names}
    _norm_values_by_feature = {name: [] for name in _feature_names}
    _split_payloads = []
    _mean = np.asarray(
        [stage_normalization[name]["mean"] for name in _feature_names],
        dtype=np.float32,
    )
    _std = np.asarray(
        [stage_normalization[name]["std"] for name in _feature_names],
        dtype=np.float32,
    )
    for _split_name in SPLIT_NAMES:
        _events = split_events(stage["truth_filtered"], _split_name)
        if not _events:
            continue
        _features = np.concatenate(
            [item["features"] for item in training_dicts(_events)],
            axis=0,
        )
        _norm_features = ((_features - _mean) / _std).astype(np.float32)
        _split_payloads.append((_split_name, _features, _norm_features))
        for _row, _feature_name in enumerate(_feature_names):
            _raw_values_by_feature[_feature_name].append(finite(_features[:, _row]))
            _norm_values_by_feature[_feature_name].append(finite(_norm_features[:, _row]))

    mo.stop(len(_split_payloads) == 0, mo.md("No non-empty splits found."))
    for _row, _feature_name in enumerate(_feature_names):
        _raw_ax = _axs[_row, 0]
        _norm_ax = _axs[_row, 1]
        _raw_edges = linear_edges(*_raw_values_by_feature[_feature_name], bins=NBINS)
        _norm_edges = linear_edges(*_norm_values_by_feature[_feature_name], bins=NBINS)
        for _split_name, _features, _norm_features in _split_payloads:
            hist_step(_raw_ax, _features[:, _row], _raw_edges, _split_name, density=True)
            hist_step(_norm_ax, _norm_features[:, _row], _norm_edges, _split_name, density=True)
        _feature_title = (
            rf"{_feature_name} "
            rf"($\mu={_mean[_row]:.3g}$, $\sigma={_std[_row]:.3g}$)"
        )
        _raw_ax.set_title(_feature_title, fontsize=16)
        _norm_ax.set_title(_feature_title, fontsize=16)
        _raw_ax.set_ylabel("Density")
        _norm_ax.set_ylabel("Density")
        _raw_ax.grid(alpha=0.25)
        _norm_ax.grid(alpha=0.25)
        if _row == _n_features - 1:
            _raw_ax.set_xlabel("Feature value")
            _norm_ax.set_xlabel("Feature value")
        style_hist_axis(_raw_ax)
        style_hist_axis(_norm_ax)

    _handles, _labels = _axs[0, 0].get_legend_handles_labels()
    if _handles:
        _fig.legend(
            _handles,
            _labels,
            loc="upper center",
            ncol=len(_labels),
            bbox_to_anchor=(0.5, 0.95),
            frameon=False,
        )
    _fig.text(0.30, 0.96, "Raw", ha="center", va="center", fontsize=28)
    _fig.text(0.75, 0.96, "Normalized", ha="center", va="center", fontsize=28)
    _fig.tight_layout(rect=[0, 0, 1, 0.93])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Truncate or Pad Vertices in Each Event
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    max_vertices_input = mo.ui.number(
        label="Max vertices",
        value=1024,
        start=1,
        step=64,
    )
    truncate_selector = mo.ui.radio(
        options=["energy_desc", "first", "random"],
        value="energy_desc",
        label="Truncation policy",
    )
    normalize_batch_features = mo.ui.checkbox(
        label="Normalize features before padding",
        value=True,
    )
    mo.hstack([max_vertices_input, truncate_selector, normalize_batch_features])
    return max_vertices_input, normalize_batch_features, truncate_selector


@app.cell
def _(
    NBINS,
    event_hit_counts,
    hist_fill,
    hist_step,
    linear_edges,
    log_edges,
    max_vertices_input,
    mo,
    normalize_batch_features,
    np,
    padded_stage,
    plt,
    stage,
    style_hist_axis,
    truncate_selector,
):
    _max_vertices = int(max_vertices_input.value or 1)
    _batch = padded_stage(
        stage["truth_filtered"],
        max_vertices=_max_vertices,
        truncate=truncate_selector.value,
        normalize_features=normalize_batch_features.value,
    )
    _labels = _batch["hit_object_id"]
    _mask = _batch["mask"]

    _fig, _axs = plt.subplots(2, 2, figsize=(15, 10))
    _n_hits = event_hit_counts(stage["truth_filtered"])
    _hit_edges = linear_edges(_n_hits, bins=NBINS, lower=0, integer=True)
    hist_fill(_axs[0, 0], _n_hits, _hit_edges, "Before batching")
    _axs[0, 0].axvline(_max_vertices, color="black", linestyle="--", label="max vertices")
    _axs[0, 0].set_xlabel("Number of hits per event")
    _axs[0, 0].set_ylabel("Number of events")
    style_hist_axis(_axs[0, 0], legend=True)

    _denom = float(_max_vertices)
    _signal_fraction = 100.0 * np.sum((_labels > 0) & _mask, axis=1) / _denom
    _noise_fraction = 100.0 * np.sum((_labels == 0) & _mask, axis=1) / _denom
    _padding_fraction = 100.0 * np.sum(~_mask, axis=1) / _denom
    _fraction_edges = np.linspace(0, 100, NBINS + 1)
    hist_step(_axs[0, 1], _signal_fraction, _fraction_edges, "Signal")
    hist_step(_axs[0, 1], _noise_fraction, _fraction_edges, "Noise")
    hist_step(_axs[0, 1], _padding_fraction, _fraction_edges, "Padding")
    _axs[0, 1].set_xlabel("Vertices per event [% of max_vertices]")
    _axs[0, 1].set_ylabel("Number of events")
    style_hist_axis(_axs[0, 1], legend=True)

    _rng = np.random.default_rng(0)
    _avg_removed_hits_per_event = []
    _removed_energy_per_cluster = []
    _n_disappeared_clusters = 0
    _n_events_with_disappeared_clusters = 0
    for _event in stage["truth_filtered"]:
        labels_raw = _event["hit_object_id"]
        energies_raw = _event["hits"]["energy"]
        n_event_hits = len(labels_raw)
        if n_event_hits <= _max_vertices:
            order = np.arange(n_event_hits)
        elif truncate_selector.value == "random":
            order = _rng.permutation(n_event_hits)
        elif truncate_selector.value == "energy_desc":
            order = np.argsort(-energies_raw)
        else:
            order = np.arange(n_event_hits)
        kept = np.zeros(n_event_hits, dtype=bool)
        kept[order[:_max_vertices]] = True

        positive_ids = np.unique(labels_raw[labels_raw > 0])
        event_removed_hits = []
        event_disappeared = 0
        for object_id in positive_ids:
            object_mask = labels_raw == object_id
            before_count = int(np.sum(object_mask))
            after_mask = object_mask & kept
            after_count = int(np.sum(after_mask))
            removed_hits = before_count - after_count
            removed_energy = float(np.sum(energies_raw[object_mask & ~kept]))
            event_removed_hits.append(removed_hits)
            if removed_energy > 0:
                _removed_energy_per_cluster.append(removed_energy)
            if before_count > 0 and after_count == 0:
                event_disappeared += 1
        _avg_removed_hits_per_event.append(
            float(np.mean(event_removed_hits)) if event_removed_hits else 0.0
        )
        if event_disappeared:
            _n_events_with_disappeared_clusters += 1
            _n_disappeared_clusters += event_disappeared

    _avg_removed_hits_per_event = np.asarray(_avg_removed_hits_per_event, dtype=np.float64)
    _removed_energy_per_cluster = np.asarray(_removed_energy_per_cluster, dtype=np.float64)
    _avg_removed_mean = float(np.mean(_avg_removed_hits_per_event))
    _avg_removed_min = float(np.min(_avg_removed_hits_per_event))
    _avg_removed_max = float(np.max(_avg_removed_hits_per_event))
    _avg_removed_edges = linear_edges(_avg_removed_hits_per_event, bins=NBINS, lower=0)
    hist_fill(_axs[1, 0], _avg_removed_hits_per_event, _avg_removed_edges, "Average removed hits")
    _axs[1, 0].set_xlabel("Average removed hits per cluster per event")
    _axs[1, 0].set_ylabel("Number of events")
    style_hist_axis(_axs[1, 0])

    if len(_removed_energy_per_cluster):
        _removed_energy_edges = log_edges(_removed_energy_per_cluster, bins=NBINS)
        hist_step(_axs[1, 1], _removed_energy_per_cluster, _removed_energy_edges, "Removed energy")
        _axs[1, 1].set_xscale("log")
        _axs[1, 1].set_yscale("log")
    else:
        _removed_energy_edges = linear_edges([0.0, 1.0], bins=NBINS, lower=0)
        hist_fill(_axs[1, 1], [0.0], _removed_energy_edges, "No removed energy")
    _axs[1, 1].set_xlabel("Energy removed from cluster [GeV]")
    _axs[1, 1].set_ylabel("Number of clusters")
    style_hist_axis(_axs[1, 1], legend=True)

    _fig.suptitle(
        f"Padding/truncation validation ({truncate_selector.value}, max_vertices={_max_vertices})",
        y=0.995,
    )
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"Average removed hits per cluster per event: "
                f"**{_avg_removed_mean:.3g}** "
                f"(range **{_avg_removed_min:.3g}** to **{_avg_removed_max:.3g}**)."
            ),
            mo.md(
                f"Clusters disappearing because all hits were truncated: "
                f"**{_n_disappeared_clusters}** clusters in "
                f"**{_n_events_with_disappeared_clusters}** events."
            ),
            _fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
