# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.8",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import awkward as ak
    import marimo as mo
    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np
    import polars as pl
    import uproot
    import yaml

    from fastgnn import get_project_root
    from fastgnn.data import CaloDataset
    from fastgnn.data.cmssw.notebook_utils import (
        collection_summary_row,
        compact_path,
        normalize_numeric_frame,
        object_collection_frame,
        pdgid_to_name,
    )
    from fastgnn.data.cmssw.plotting import plot_event_display
    from fastgnn.evaluation import (
        available_prediction_quantities,
        build_predicted_events,
        decode_payload_outputs,
        oc_layout_from_model_config,
        payload_quantities_from_config,
        prediction_eval_feature_names,
    )
    from fastgnn.training.oc_outputs import split_oc_outputs

    mplhep.style.use("CMS")
    PROJECT_ROOT = get_project_root()

    def flatten_field(field) -> np.ndarray:
        values = ak.to_numpy(ak.flatten(field, axis=None))
        return np.asarray(values)

    def finite_positive(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values[np.isfinite(values) & (values > 0)]

    def finite_values(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values[np.isfinite(values)]

    return (
        CaloDataset,
        PROJECT_ROOT,
        Path,
        ak,
        available_prediction_quantities,
        build_predicted_events,
        collection_summary_row,
        compact_path,
        decode_payload_outputs,
        finite_positive,
        finite_values,
        flatten_field,
        mo,
        mplhep,
        normalize_numeric_frame,
        np,
        object_collection_frame,
        oc_layout_from_model_config,
        payload_quantities_from_config,
        pdgid_to_name,
        pl,
        plot_event_display,
        plt,
        prediction_eval_feature_names,
        split_oc_outputs,
        uproot,
        yaml,
    )


@app.cell(hide_code=True)
def _(PROJECT_ROOT, mo):
    dataset_path = mo.ui.text(
        label="Converted inference dataset",
        value=str(
            PROJECT_ROOT
            / "data/processed/cmssw_classical/emyr_samples/N200_PU0-single_pion/zside_+1"
        ),
        full_width=True,
    )
    output_path = mo.ui.text(
        label="Training outputs directory",
        value=str(PROJECT_ROOT / "outputs"),
        full_width=True,
    )
    max_events = mo.ui.number(label="Max events", value=200, start=1, step=1)
    mo.vstack([dataset_path, output_path, max_events])
    return dataset_path, max_events, output_path


@app.cell(hide_code=True)
def _(CaloDataset, Path, dataset_path, max_events, mo):
    dataset_dir = Path(dataset_path.value).expanduser()
    mo.stop(not dataset_dir.exists(), mo.md(f"Dataset path does not exist: `{dataset_dir}`"))
    dataset = CaloDataset(dataset_dir, split="test", max_events=int(max_events.value))
    mo.md(f"Loaded `{dataset}`")
    return dataset, dataset_dir


@app.cell(hide_code=True)
def _(dataset, mo):
    mo.vstack(
        [
            mo.md("## Dataset Overview"),
            mo.md(
                f"**Hit features**: {', '.join(dataset.hit_features)}\n\n"
                f"**Hit fields**: {', '.join(dataset.fields['hits'])}\n\n"
                f"**Truth fields**: {', '.join(dataset.fields['truth']) or '<none>'}\n\n"
                f"**Metadata fields**: {', '.join(dataset.fields['metadata']) or '<none>'}"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(PROJECT_ROOT, compact_path, dataset, dataset_dir, mo):
    metadata = dataset.metadata
    source_files = metadata.get("source_files") or []
    default_root = PROJECT_ROOT / source_files[0] if source_files else ""
    root_path_override = mo.ui.text(
        label="ROOT file for uproot inspection",
        value=default_root,
        full_width=True,
    )
    tree_path = mo.ui.text(
        label="ROOT tree path",
        value=str(metadata.get("tree_path", "l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple")),
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                f"Dataset directory: `{compact_path(dataset_dir)}`\n\n"
                f"Configured source ROOT files: `{', '.join(compact_path(path) for path in source_files)}`"
            ),
            root_path_override,
            tree_path,
        ]
    )
    return metadata, root_path_override, tree_path


@app.cell(hide_code=True)
def _(
    Path,
    ak,
    dataset,
    metadata,
    mo,
    np,
    root_path_override,
    tree_path,
    uproot,
):
    root_path = Path(root_path_override.value)
    zside = int(
        metadata.get("preprocessing", {}).get("zside", metadata.get("config", {}).get("zside", 1))
    )
    mo.stop(not root_path.exists(), mo.md(f"ROOT file does not exist: `{root_path}`"))

    with uproot.open(f"{root_path}:{tree_path.value}") as tree:
        root_arrays = tree.arrays(library="ak")

    kept_entry_mask = ak.sum(root_arrays["tc_zside"] == zside, axis=1) > 0
    kept_entries = np.flatnonzero(ak.to_numpy(kept_entry_mask))
    dataset_size = len(dataset)
    mo.stop(
        len(kept_entries) < dataset_size,
        mo.md(
            "The ROOT file has fewer z-side-selected events than the converted dataset. "
            "Check that the ROOT file matches the dataset metadata."
        ),
    )
    dataset_to_root_entry = kept_entries[:dataset_size]
    mo.md(
        f"Loaded `{len(root_arrays)}` ROOT entries from `{root_path}`.\n\n"
        f"Selected z-side: `{zside}`. Mapped `{len(dataset_to_root_entry)}` converted events to ROOT entries."
    )
    return dataset_to_root_entry, root_arrays, zside


@app.cell
def _(ak, np, plt, root_arrays):
    _fig, _axes = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

    gen_endcap_mask = root_arrays.gen_eta > 0
    tc_endcap_mask = root_arrays.tc_eta > 0
    _gen_axs = _axes[0]

    def linestyle(sign):
        return "--" if sign == -1 else "-"

    for _mask, _sign in zip([gen_endcap_mask, ~gen_endcap_mask], [1, -1]):
        plt.sca(_gen_axs[0])
        plt.hist(
            root_arrays.gen_pt[_mask],
            bins=np.linspace(0, 200, 21),
            histtype="step",
            label=f"zside={_sign}",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("gen_pt")

        plt.sca(_gen_axs[1])
        plt.hist(
            np.abs(root_arrays.gen_eta[_mask].to_numpy()),
            bins=np.arange(0, 6, 0.25),
            histtype="step",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("|gen_eta|")
        plt.axvline(1.5, color="gray", linestyle=":", linewidth=1)
        plt.axvline(3.0, color="gray", linestyle=":", linewidth=1)

        plt.sca(_gen_axs[2])
        plt.hist(
            root_arrays.gen_phi[_mask],
            bins=np.linspace(-np.pi, np.pi, 21),
            histtype="step",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("gen_phi")

    _tc_axs = _axes[1:].flatten()
    for _mask, _sign in zip([tc_endcap_mask, ~tc_endcap_mask], [1, -1]):
        plt.sca(_tc_axs[0])
        plt.hist(
            ak.flatten(root_arrays.tc_pt[_mask]),
            bins=np.linspace(0, 0.2, 21),
            histtype="step",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("tc_pt")

        plt.sca(_tc_axs[1])
        plt.hist(
            np.abs(ak.flatten(root_arrays.tc_eta[_mask]).to_numpy()),
            bins=np.arange(0, 6, 0.25),
            histtype="step",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("|tc_eta|")
        plt.axvline(1.5, color="gray", linestyle=":", linewidth=1)
        plt.axvline(3.0, color="gray", linestyle=":", linewidth=1)

        plt.sca(_tc_axs[2])
        plt.hist(
            ak.flatten(root_arrays.tc_phi[_mask]),
            bins=np.linspace(-np.pi, np.pi, 21),
            histtype="step",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("tc_phi")

        plt.sca(_tc_axs[3])
        _lvals, _lcts = np.unique(
            ak.flatten(root_arrays.tc_layer[_mask]).to_numpy(), return_counts=True
        )
        plt.bar(
            _lvals,
            _lcts,
            alpha=0.5,
            edgecolor="#f89c20" if _sign == -1 else "#5790fc",
            linestyle=linestyle(_sign),
            fill=False,
            label=None,
        )  # type: ignore
        plt.xlabel("tc_layer")

        plt.sca(_tc_axs[4])
        plt.hist(
            ak.sum(root_arrays.tc_zside == _sign, axis=1),
            bins=np.arange(800, 2500, 50),
            histtype="step",
            linestyle=linestyle(_sign),
        )  # type: ignore
        plt.xlabel("tc_n")

    _fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(collection_summary_row, normalize_numeric_frame, pl, root_arrays):
    _summary = normalize_numeric_frame(
        pl.DataFrame(
            [
                collection_summary_row(root_arrays, "gen"),
                collection_summary_row(root_arrays, "genpart"),
                collection_summary_row(root_arrays, "genjet"),
                collection_summary_row(root_arrays, "gentau"),
            ]
        )
    )
    _summary
    return


@app.cell(hide_code=True)
def _(dataset, mo):
    event_selector = mo.ui.number(
        label="Selected event index",
        start=0,
        stop=max(len(dataset) - 1, 0),
        step=1,
        value=0,
    )
    event_selector
    return (event_selector,)


@app.cell(hide_code=True)
def _(
    dataset,
    event_selector,
    mo,
    np,
    pdgid_to_name,
    pl,
    plot_event_display,
    root_arrays,
):
    _event = dataset[int(event_selector.value or 0)]

    _empty_f = np.asarray([], dtype=np.float32)
    _empty_i = np.asarray([], dtype=np.int32)
    _fig, _summary = plot_event_display(
        h_x=np.asarray(_event.hits.x),
        h_y=np.asarray(_event.hits.y),
        h_z=np.asarray(_event.hits.z),
        h_e=np.asarray(_event.hits.energy),
        hit_object_id=np.asarray(_event.truth.hit_object_id),
        c_eta=_empty_f,
        c_phi=_empty_f,
        c_e=_empty_f,
        c_pdg=_empty_i,
        mode="energy",
        views=("xy", "yz", "3d"),
        show_cluster_markers=False,
        show_clusters_below_threshold=False,
        event_idx=_event.event_id,
    )

    _event_gen_arr = root_arrays[root_arrays.event == _event.metadata["source_event_id"]][
        ["gen_eta", "gen_pt", "gen_pdgid"]
    ]
    _event_gen_arr = _event_gen_arr[np.sign(_event_gen_arr["gen_eta"]) == _event.metadata["zside"]][
        0
    ]
    _gen_df = pl.DataFrame(
        {
            "gen_eta": _event_gen_arr["gen_eta"].to_numpy(),
            "gen_pt": _event_gen_arr["gen_pt"].to_numpy(),
            "gen_pdgid": _event_gen_arr["gen_pdgid"].to_numpy(),
            "gen_particle": map(pdgid_to_name, _event_gen_arr["gen_pdgid"].to_list()),
        }
    )

    mo.vstack(
        [
            mo.md("### Event Display"),
            _gen_df,
            mo.md(_summary),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    dataset_to_root_entry,
    event_selector,
    mo,
    normalize_numeric_frame,
    object_collection_frame,
    pdgid_to_name,
    pl,
    root_arrays,
):
    _event_idx = int(event_selector.value or 0)
    _root_idx = int(dataset_to_root_entry[_event_idx])

    gen_frame = object_collection_frame(
        root_arrays,
        _root_idx,
        "gen",
        [
            ("pdgid", "pdgid"),
            ("status", "status"),
            ("energy", "energy"),
            ("pt", "pt"),
            ("eta", "eta"),
            ("phi", "phi"),
        ],
    )
    if "pdgid" in gen_frame.columns:
        gen_frame = gen_frame.with_columns(
            pl.col("pdgid").map_elements(pdgid_to_name, return_dtype=pl.String).alias("particle")
        ).select(["particle", "pdgid", "status", "energy", "pt", "eta", "phi"])
    gen_frame = normalize_numeric_frame(gen_frame)

    genpart_frame = object_collection_frame(
        root_arrays,
        _root_idx,
        "genpart",
        [
            ("pid", "pid"),
            ("gen", "gen"),
            ("reachedEE", "reachedEE"),
            ("fromBeamPipe", "fromBeamPipe"),
            ("energy", "energy"),
            ("pt", "pt"),
            ("eta", "eta"),
            ("phi", "phi"),
        ],
    )
    if "pid" in genpart_frame.columns:
        genpart_frame = genpart_frame.with_columns(
            pl.col("pid").map_elements(pdgid_to_name, return_dtype=pl.String).alias("particle")
        ).select(
            ["particle", "pid", "gen", "reachedEE", "fromBeamPipe", "energy", "pt", "eta", "phi"]
        )
    genpart_frame = normalize_numeric_frame(genpart_frame)

    genjet_frame = normalize_numeric_frame(
        object_collection_frame(
            root_arrays,
            _root_idx,
            "genjet",
            [("energy", "energy"), ("pt", "pt"), ("eta", "eta"), ("phi", "phi")],
        )
    )
    gentau_frame = normalize_numeric_frame(
        object_collection_frame(
            root_arrays,
            _root_idx,
            "gentau",
            [
                ("decayMode", "decayMode"),
                ("energy", "energy"),
                ("pt", "pt"),
                ("eta", "eta"),
                ("phi", "phi"),
                ("vis_energy", "vis_energy"),
                ("vis_pt", "vis_pt"),
                ("vis_eta", "vis_eta"),
                ("vis_phi", "vis_phi"),
            ],
        )
    )

    mo.vstack(
        [
            mo.md("## Selected Event Truth/Object Tables"),
            mo.ui.table(gen_frame, label="gen collection"),
            mo.ui.table(genpart_frame, label="genpart collection"),
            mo.ui.table(genjet_frame, label="genjet collection"),
            mo.ui.table(gentau_frame, label="gentau collection"),
            mo.md(
                "Interpretation for this sample: `gen` is the compact primary generated-particle "
                "collection, `genpart` is the larger detailed generator-particle collection, "
                "`genjet` mirrors the event-level generated jets, and `gentau` is only useful if "
                "the sample actually contains generated taus."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    NBINS = 50
    return (NBINS,)


@app.cell(hide_code=True)
def _(
    NBINS,
    dataset,
    finite_positive,
    finite_values,
    flatten_field,
    mplhep,
    np,
    plt,
):
    n_hits = np.asarray([dataset[i].n_hits for i in range(len(dataset))], dtype=np.int64)
    hit_energy = finite_positive(flatten_field(dataset.events.hits.energy))

    hit_layer = finite_values(flatten_field(dataset.events.hits.layer))
    event_energy = np.asarray(
        [np.asarray(dataset[i].hits.energy, dtype=np.float64).sum() for i in range(len(dataset))],
        dtype=np.float64,
    )

    _fig, _gen_axs = plt.subplots(2, 2, figsize=(15, 10))
    _gen_axs = _gen_axs.flatten()

    _gen_axs[0].hist(n_hits, bins=min(NBINS, max(10, len(n_hits))))
    _gen_axs[0].set_xlabel("Number of trigger cells per event")
    _gen_axs[0].set_ylabel("Events")

    event_energy_edges = np.logspace(
        np.log10(max(event_energy.min(), 1e-3)),
        np.log10(max(event_energy.max(), 1e-2)),
        NBINS,
    )
    _gen_axs[1].hist(event_energy, bins=event_energy_edges)
    _gen_axs[1].set_xscale("log")
    _gen_axs[1].set_yscale("log")
    _gen_axs[1].set_xlabel("Total trigger-cell energy per event [GeV]")
    _gen_axs[1].set_ylabel("Events")

    hit_energy_edges = np.logspace(
        np.log10(max(hit_energy.min(), 1e-6)),
        np.log10(max(hit_energy.max(), 1e-5)),
        NBINS,
    )
    _gen_axs[2].hist(hit_energy, bins=hit_energy_edges)
    _gen_axs[2].set_xscale("log")
    _gen_axs[2].set_yscale("log")
    _gen_axs[2].set_xlabel("Trigger-cell energy [GeV]")
    _gen_axs[2].set_ylabel("Trigger cells")

    layer_bins = (
        np.arange(hit_layer.min() - 0.5, hit_layer.max() + 1.5, 1.0) if len(hit_layer) else 10
    )
    _gen_axs[3].hist(hit_layer, bins=layer_bins)
    _gen_axs[3].set_xlabel("Trigger-cell layer")
    _gen_axs[3].set_ylabel("Trigger cells")

    mplhep.cms.text(ax=_gen_axs[0])
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(Path, mo, output_path, pl, yaml):
    output_dir = Path(output_path.value).expanduser()
    mo.stop(not output_dir.exists(), mo.md(f"Outputs directory does not exist: `{output_dir}`"))

    run_dirs = sorted(
        {path.parent for path in output_dir.glob("**/*.keras")},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    run_options = []
    for path in run_dirs:
        _config_path = path / "resolved_config.yaml"
        if not _config_path.exists():
            continue
        with _config_path.open() as _handle:
            _config = yaml.safe_load(_handle) or {}
        run_options.append(
            {"run_name": _config.get("run_name"), "path": str(path.relative_to(output_dir))}
        )

    mo.stop(len(run_options) == 0, mo.md(f"No model runs found below `{output_dir}`."))
    runs_table = mo.ui.table(
        pl.DataFrame(run_options).select(["run_name", "path"]),
        selection="single",
        label="Select run",
    )
    runs_table
    return output_dir, runs_table


@app.cell(hide_code=True)
def _(Path, mo, output_dir, runs_table):
    mo.stop(len(runs_table.value) == 0 or runs_table.value["path"][0] is None, mo.md(""))
    run_dir = output_dir / Path(runs_table.value["path"][0])

    model_files = {}
    if (run_dir / "best_model.keras").exists():
        model_files["best_model"] = "best_model"
    for checkpoint in sorted(run_dir.glob("checkpoint_epoch*.keras")):
        epoch = str(int(checkpoint.stem.replace("checkpoint_epoch", "")))
        model_files[f"ep {epoch}"] = checkpoint.name
    if (run_dir / "final_model.keras").exists():
        model_files["final_model"] = "final_model.keras"

    mo.stop(len(model_files) == 0, mo.md(f"No model files found in `{run_dir}`."))
    model_file_selector = mo.ui.dropdown(
        options=model_files,
        label="Model checkpoint",
        value=next(iter(model_files.values())),
    )
    model_file_selector
    return model_file_selector, run_dir


@app.cell(hide_code=True)
def _(mo, model_file_selector, run_dir, yaml):
    model_path = (run_dir / model_file_selector.value).with_suffix(".keras")
    _config_path = run_dir / "resolved_config.yaml"
    mo.stop(not model_path.exists(), mo.md(f"Model file does not exist: `{model_path}`"))
    mo.stop(not _config_path.exists(), mo.md(f"Config file does not exist: `{_config_path}`"))

    import qgravnet as _qgravnet
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    config = yaml.safe_load(_config_path.read_text())
    mo.md(f"Loaded `{model_file_selector.value}` from `{run_dir}`")
    return config, model


@app.cell(hide_code=True)
def _(config, dataset, mo):
    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    feature_names = list(data_cfg.get("feature_names") or dataset.hit_features)
    max_vertices = int(model_cfg["max_vertices"])
    truncate = training_cfg.get("truncate", "first")
    normalize_features = bool(training_cfg.get("normalize_features", True))
    mo.md(
        f"Using `{len(feature_names)}` model features, max vertices `{max_vertices}`, "
        f"truncate `{truncate}`, normalize `{normalize_features}`."
    )
    return feature_names, max_vertices, normalize_features, truncate


@app.cell(hide_code=True)
def _(
    config,
    dataset,
    feature_names,
    max_vertices,
    model,
    normalize_features,
    np,
    oc_layout_from_model_config,
    split_oc_outputs,
    truncate,
):
    model_inputs = dataset.as_padded(
        max_vertices=max_vertices,
        feature_names=feature_names,
        truncate=truncate,
        normalize_features=normalize_features,
        seed=config.get("seed", 0),
    )
    predictions = model.predict(model_inputs["features"], batch_size=1024, verbose=1)
    layout = oc_layout_from_model_config(config["model"])
    output_slices = split_oc_outputs(predictions, layout)
    _beta = np.asarray(output_slices.beta)
    _mask = np.asarray(model_inputs["mask"], dtype=bool)
    stats = {
        "input_shape": tuple(model_inputs["features"].shape),
        "prediction_shape": tuple(predictions.shape),
        "mask_occupancy": float(_mask.mean()),
        "beta_min": float(np.min(_beta[_mask])) if np.any(_mask) else None,
        "beta_max": float(np.max(_beta[_mask])) if np.any(_mask) else None,
        "beta_mean": float(np.mean(_beta[_mask])) if np.any(_mask) else None,
        "beta_std": float(np.std(_beta[_mask])) if np.any(_mask) else None,
        "cluster_dim": layout.cluster_dim,
        "payload_dim": layout.payload_dim,
    }
    return output_slices, stats


@app.cell(hide_code=True)
def _(mo, pl, stats):
    mo.vstack(
        [
            mo.md("## Forward Pass Summary"),
            mo.ui.table(pl.DataFrame([stats]), label="Forward-pass summary"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(config, mo, output_slices, payload_quantities_from_config):
    payload_quantities = payload_quantities_from_config(config)
    if output_slices.payload is not None and payload_quantities:
        _payload_lines = [
            f"- `{quantity.get('name', quantity.get('field', quantity.get('target')))}`"
            for quantity in payload_quantities
        ]
        _payload_summary = mo.md("Decoded payload quantities:\n" + "\n".join(_payload_lines))
    elif output_slices.payload is not None:
        _payload_summary = mo.md(
            "Model has payload outputs, but `training.payload.quantities` is not configured."
        )
    else:
        _payload_summary = mo.md("Model has no payload outputs.")
    _payload_summary
    return (payload_quantities,)


@app.cell(hide_code=True)
def _(
    config,
    dataset,
    max_vertices,
    payload_quantities,
    prediction_eval_feature_names,
    truncate,
):
    payload_eval_feature_names = prediction_eval_feature_names(payload_quantities)
    payload_eval_inputs = dataset.as_padded(
        max_vertices=max_vertices,
        feature_names=payload_eval_feature_names,
        truncate=truncate,
        normalize_features=False,
        seed=config.get("seed", 0),
    )
    return payload_eval_feature_names, payload_eval_inputs


@app.cell(hide_code=True)
def _(
    decode_payload_outputs,
    output_slices,
    payload_eval_feature_names,
    payload_eval_inputs,
    payload_quantities,
):
    decoded_payload_features = decode_payload_outputs(
        output_slices.payload,
        payload_quantities,
        payload_eval_inputs["features"],
        payload_eval_feature_names,
    )
    return (decoded_payload_features,)


@app.cell(hide_code=True)
def _(mo, pl, run_dir, yaml):
    with (run_dir / "oc_threshold_calibrations.json").open() as _handle:
        _cached_oc_thresholds_dict = yaml.safe_load(_handle) or {}

    cached_oc_thresholds = []
    for _r in _cached_oc_thresholds_dict["records"]:
        cached_oc_thresholds.append(
            (
                _r["calibration_options"]["objective"],
                _r["best_thresholds"]["td"],
                _r["best_thresholds"]["tbeta"],
            )
        )
    cached_oc_thresholds = pl.DataFrame(
        cached_oc_thresholds,
        schema=[
            ("objective", pl.Utf8),
            ("td_threshold", pl.Float32),
            ("tbeta_threshold", pl.Float32),
        ],
        strict=False,
        orient="row",
    )

    tbeta = mo.ui.number(
        label="OC beta threshold",
        value=cached_oc_thresholds["tbeta_threshold"][-1] or 0.2,
        start=0.0,
        step=0.001,
    )
    td = mo.ui.number(
        label="OC distance threshold",
        value=cached_oc_thresholds["td_threshold"][-1] or 1.0,
        start=0.01,
        step=0.001,
    )
    mo.vstack([cached_oc_thresholds, mo.hstack([td, tbeta], justify="start")])
    return tbeta, td


@app.cell(hide_code=True)
def _(available_prediction_quantities, decoded_payload_features, mo):
    aggregated_quantity_options, payload_quantity_options = available_prediction_quantities(
        decoded_payload_features
    )
    prediction_quantity_selector = mo.ui.multiselect(
        options=[*aggregated_quantity_options, *payload_quantity_options],
        value=[
            _name
            for _name in ("beta_seed", "n_hits_pred", "sum_energy_reco", "sum_et_reco")
            if _name in [*aggregated_quantity_options, *payload_quantity_options]
        ],
        label="Predicted cluster quantities to keep",
    )
    prediction_quantity_selector
    return (prediction_quantity_selector,)


@app.cell(hide_code=True)
def _(
    ak,
    build_predicted_events,
    dataset,
    decoded_payload_features,
    np,
    output_slices,
    payload_eval_feature_names,
    payload_eval_inputs,
    prediction_quantity_selector,
    tbeta,
    td,
):
    pred_events = ak.Array(
        build_predicted_events(
            dataset,
            payload_eval_inputs,
            np.asarray(output_slices.beta),
            np.asarray(output_slices.cluster_coords),
            float(tbeta.value),
            float(td.value),
            eval_feature_names=payload_eval_feature_names,
            decoded_payload_features=decoded_payload_features,
            selected_prediction_quantities=list(prediction_quantity_selector.value),
        )
    )
    return (pred_events,)


@app.cell
def _(pred_events_df):
    pred_events_df
    return


@app.cell
def _(ak, pdgid_to_name, pl, root_arrays, zside):
    # filter to one endcap
    gen_singleEndcap_df = pl.DataFrame(
        ak.zip(
            {
                "event": root_arrays.event,
                **{
                    f: root_arrays[f][root_arrays.gen_eta * zside > 0]
                    for f in ["gen_pt", "gen_energy", "gen_eta", "gen_phi", "gen_pdgid"]
                },
            },
            depth_limit=1,
        ).to_list()
    ).with_columns(
        pl.col("gen_pdgid")
        .list.eval(pl.element().map_elements(pdgid_to_name, return_dtype=pl.String))
        .alias("gen_particle")
    )
    return (gen_singleEndcap_df,)


@app.cell
def _(pl, pred_events, prediction_quantity_selector):
    pred_events_df = pl.DataFrame(pred_events.tolist())
    _selected_quantities = list(prediction_quantity_selector.value)
    _summary_columns = ["source_event_id", "n_pred_clusters"]
    if (
        "sum_energy_reco" in _selected_quantities
        and "event_sum_energy_reco" in pred_events_df.columns
    ):
        _summary_columns.append("event_sum_energy_reco")
    if "sum_et_reco" in _selected_quantities and "event_sum_et_reco" in pred_events_df.columns:
        _summary_columns.append("event_sum_et_reco")
    if (
        "payload_energy_pred" in _selected_quantities
        and "event_payload_energy_sum" in pred_events_df.columns
    ):
        _summary_columns.append("event_payload_energy_sum")
    if (
        "payload_et_pred" in _selected_quantities
        and "event_payload_et_sum" in pred_events_df.columns
    ):
        _summary_columns.append("event_payload_et_sum")
    _summary_columns.extend(
        [
            "unassigned_tc_energy",
            "valid_tc_energy",
            "selected_prediction_quantities",
            "clusters",
        ]
    )
    pred_events_df = pred_events_df.select(_summary_columns)
    return (pred_events_df,)


@app.cell
def _(gen_singleEndcap_df, pred_events_df, prediction_quantity_selector):
    comparison_df = pred_events_df.join(
        gen_singleEndcap_df,
        left_on="source_event_id",
        right_on="event",
        how="inner",
    ).select(
        [
            "source_event_id",
            "n_pred_clusters",
            *[
                _column
                for _column in (
                    "unassigned_tc_energy",
                    "valid_tc_energy",
                    *[
                        quantity
                        for quantity in prediction_quantity_selector.value
                        if quantity in pred_events_df.columns
                    ],
                    "clusters",
                )
                if _column in pred_events_df.columns
            ],
            "gen_particle",
            "gen_pt",
            "gen_energy",
            "gen_eta",
            "gen_phi",
        ]
    )
    comparison_df
    return


if __name__ == "__main__":
    app.run()
