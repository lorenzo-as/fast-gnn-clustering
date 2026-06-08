# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import itertools
    from datetime import datetime
    from pathlib import Path
    import pickle

    import marimo as mo
    import matplotlib.pyplot as plt
    import mplhep
    import numpy as np
    import polars as pl
    import yaml

    from fastgnn import get_project_root
    from fastgnn.utils import PLOTTING_CONFIG
    from fastgnn.data import CaloDataset

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    mplhep.style.use("CMS")

    COLORS = {
        "train": "#4477AA",
        "val": "#EE6677",
        "signal": "#e42536",
        "noise": "#f89c20",
        "padding": "#5790fc",
    }

    PRJ_ROOT = get_project_root()
    OUTPUT_DIR = PRJ_ROOT / "outputs"

    run_dirs = sorted(
        {path.parent for path in OUTPUT_DIR.glob("**/*.keras")},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    run_options = {}
    for path in run_dirs:
        relative_path = str(path.relative_to(OUTPUT_DIR))
        with (path / "resolved_config.yaml").open() as f:
            _config = yaml.safe_load(f)
        run_name = _config.get("run_name")
        run_options[relative_path] = run_name
    runs_df = pl.DataFrame(
        {"path": list(run_options.keys()), "run_name": list(run_options.values())}
    ).select(["run_name", "path"])
    runs_table = mo.ui.table(
        runs_df,
        selection="single",
        label="Select Run",
    )

    runs_table
    return (
        COLORS,
        CaloDataset,
        OUTPUT_DIR,
        PLOTTING_CONFIG,
        PRJ_ROOT,
        Path,
        datetime,
        itertools,
        json,
        mo,
        mplhep,
        np,
        pickle,
        pl,
        plt,
        runs_table,
        yaml,
    )


@app.cell(hide_code=True)
def _(CaloDataset, OUTPUT_DIR, PRJ_ROOT, Path, mo, np, runs_table, yaml):
    def load_run(dir_path: Path, skip_model: bool = False):
        if skip_model:
            best_model = None
        else:
            import qgravnet
            import tensorflow as tf

            best_model = tf.keras.models.load_model(dir_path / "best_model.keras", compile=False)
        if (dir_path / "history.npy").exists():
            history = np.load(dir_path / "history.npy", allow_pickle=True).item()
        else:
            history = None
        with (dir_path / "resolved_config.yaml").open() as f:
            config = yaml.safe_load(f)

        return best_model, history, config

    mo.stop(
        len(runs_table.value) == 0 or runs_table.value["path"][0] is None,
        mo.md("Please select a run to analyze."),
    )
    selected_run = Path(runs_table.value["path"][0])
    selected_run_dir = OUTPUT_DIR / selected_run
    best_model, history, config = load_run(selected_run_dir)
    test_ds = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split="test")
    messages = [
        f"Loaded {'model, ' if best_model is not None else ''}history, and config for run: `{selected_run}`",
        f"Loaded test dataset: `{test_ds}`",
    ]

    mo.md("\n\n".join(messages))
    return best_model, config, history, selected_run_dir, test_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Loss Evolution
    """)
    return


@app.cell
def _(COLORS, PLOTTING_CONFIG, config, history, mo, mplhep, np, pl, plt):
    mo.stop(history is None, mo.md("No training history found for this run."))
    train_components_df = pl.DataFrame(history["train_components"])
    val_components_df = pl.DataFrame(history["val_components"])
    if "L_total" not in train_components_df.columns:
        train_components_df = train_components_df.with_columns(
            (train_components_df["L_V"] + train_components_df["L_beta"]).alias("L_total")
        )
    if "L_total" not in val_components_df.columns:
        val_components_df = val_components_df.with_columns(
            (val_components_df["L_V"] + val_components_df["L_beta"]).alias("L_total")
        )
    np.testing.assert_array_almost_equal(history["train"], train_components_df["L_total"])
    np.testing.assert_array_almost_equal(history["val"], val_components_df["L_total"])
    _max_loss = max(train_components_df["L_total"].max(), val_components_df["L_total"].max())

    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"],
        constrained_layout=True,
    )

    plt.sca(_axs[0])
    plt.plot(history["train"], label="Training", color=COLORS["train"])
    plt.plot(history["val"], label="Validation", color=COLORS["val"])
    plt.ylabel("Loss")
    plt.xlim(0, config.get("training", {}).get("n_epochs", len(history["train"])))
    plt.legend(loc="upper right")

    loss_labels = {
        "L_total": r"$\mathcal{L}_{total}$",
        "L_V_attractive": r"$\mathcal{L}_V^{attr}$",
        "L_V_repulsive": r"$\mathcal{L}_V^{rep}$",
        "L_beta_sig": r"$\mathcal{L}_\beta^{sig}$",
        "L_beta_noise": r"$\mathcal{L}_\beta^{noise}$",
    }

    for _i, (_df_type, _df) in enumerate(
        zip(["train", "val"], [train_components_df, val_components_df]), start=1
    ):
        plt.sca(_axs[_i])
        _colors = {"V": "#228833", "beta": "#CCBB44"}
        linestyles = {
            "L_V_attractive": "dashed",
            "L_V_repulsive": "dotted",
            "L_beta_sig": "dashed",
            "L_beta_noise": "dotted",
        }
        for _loss_key, _loss_label in loss_labels.items():
            plt.plot(
                _df[_loss_key],
                label=_loss_label,
                color=_colors.get(_loss_key.split("_")[1], "black"),
                linestyle=linestyles.get(_loss_key, "solid"),
            )
        mplhep.add_text("Training" if _df_type == "train" else "Validation", loc="upper right")
        plt.yscale("log")
        plt.ylim(5e-3, _max_loss * 2)
        plt.xlim(0, config.get("training", {}).get("n_epochs", len(history["train"])))

    plt.legend(loc="center left", ncol=1, bbox_to_anchor=(1.0, 0.5))
    plt.xlabel("Epoch")

    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Predict Test Set
    """)
    return


@app.cell(hide_code=True)
def _(best_model, config, np, test_ds):
    from fastgnn.training.oc_outputs import OCOutputLayout, split_oc_outputs

    oc_layout = OCOutputLayout.from_config(config["model"])
    test_data = test_ds.as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=config["data"]["feature_names"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=config["training"].get("normalize_features", True),
        seed=config.get("seed", 0),
    )
    test_preds = best_model.predict(test_data["features"], batch_size=1024, verbose=1)

    test_output_slices = split_oc_outputs(test_preds, oc_layout)
    test_beta = np.asarray(test_output_slices.beta)
    test_cluster_coords = np.asarray(test_output_slices.cluster_coords)
    return (
        oc_layout,
        split_oc_outputs,
        test_beta,
        test_cluster_coords,
        test_data,
        test_preds,
    )


@app.cell(hide_code=True)
def _(config, np, test_data, test_ds):
    test_eval_data = test_ds.as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=["x", "y", "z", "energy"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=False,
        seed=config.get("seed", 0),
    )
    np.testing.assert_array_equal(test_eval_data["mask"], test_data["mask"])
    np.testing.assert_array_equal(test_eval_data["hit_object_id"], test_data["hit_object_id"])
    return (test_eval_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fraction of Signal, Noise, Padded & their $\beta$ distributions
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(CaloDataset, PRJ_ROOT, config, mo, test_beta, test_data):
    full_data = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split=None).as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=config["data"]["feature_names"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=config["training"].get("normalize_features", True),
        seed=config.get("seed", 0),
    )
    full_data_padded_mask = ~full_data["mask"].astype(bool)
    full_data_signal_mask = (full_data["hit_object_id"] > 0) & ~full_data_padded_mask
    full_data_noise_mask = (full_data["hit_object_id"] == 0) & ~full_data_padded_mask

    test_data_beta_min, test_data_beta_max = test_beta.min(), test_beta.max()
    test_data_beta_min = 0 if test_data_beta_min > 0 else test_data_beta_min
    test_data_padded_mask = ~test_data["mask"].astype(bool)
    test_data_signal_mask = (test_data["hit_object_id"] > 0) & ~test_data_padded_mask
    test_data_noise_mask = (test_data["hit_object_id"] == 0) & ~test_data_padded_mask

    yscale_selector_01 = mo.ui.dropdown(
        options=["linear", "log"], label="Select y-axis scale.", value="linear"
    )
    yscale_selector_01
    return (
        full_data_noise_mask,
        full_data_padded_mask,
        full_data_signal_mask,
        test_data_beta_max,
        test_data_beta_min,
        test_data_noise_mask,
        test_data_padded_mask,
        test_data_signal_mask,
        yscale_selector_01,
    )


@app.cell(hide_code=True)
def _(
    COLORS,
    PLOTTING_CONFIG,
    config,
    full_data_noise_mask,
    full_data_padded_mask,
    full_data_signal_mask,
    mplhep,
    np,
    plt,
    test_beta,
    test_data_beta_max,
    test_data_beta_min,
    test_data_noise_mask,
    test_data_padded_mask,
    test_data_signal_mask,
    yscale_selector_01,
):
    _fig1, _axs = plt.subplots(
        nrows=1, ncols=2, figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"]
    )

    plt.sca(_axs[0])

    _bins_truth = np.linspace(0, 100, 50)
    for _mask, _label in reversed(
        list(
            zip(
                [full_data_signal_mask, full_data_noise_mask, full_data_padded_mask],
                ["Signal  ", "Noise   ", "Padding"],
            )
        )
    ):
        _fractions = _mask.sum(axis=1) / config["model"]["max_vertices"] * 100
        plt.hist(
            _fractions,
            bins=_bins_truth,
            alpha=0.5,
            label=_label + f" (mean: {_fractions.mean():.1f}%)",
            color=COLORS[_label.strip().lower()],
        )
    plt.xlabel("Fraction of vertices per event [%]")
    plt.xlim(0, 100)
    plt.ylabel("Count")
    plt.yscale(yscale_selector_01.value)
    if yscale_selector_01.value == "linear":
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    mplhep.add_text(f"Full Dataset (N={len(full_data_signal_mask)})", loc="over right")
    plt.legend()
    mplhep.mpl_magic()
    plt.ylim(1, None)

    # Histogram of beta values for signal hits and noise hits
    plt.sca(_axs[1])
    _bins_truth = np.linspace(test_data_beta_min, test_data_beta_max, 50)
    plt.hist(
        test_beta[test_data_padded_mask],
        bins=_bins_truth,
        alpha=0.5,
        label="Padding",
        color=COLORS["padding"],
    )
    plt.hist(
        test_beta[test_data_noise_mask],
        bins=_bins_truth,
        alpha=0.5,
        label="Noise",
        color=COLORS["noise"],
    )
    plt.hist(
        test_beta[test_data_signal_mask],
        bins=_bins_truth,
        alpha=0.5,
        label="Signal",
        color=COLORS["signal"],
    )
    plt.xlabel(r"$\beta_\mathrm{pred}$ scores ")
    plt.xlim(0, 1)
    plt.ylabel("Count")
    plt.yscale(yscale_selector_01.value)
    if yscale_selector_01.value == "linear":
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    mplhep.add_text(f"Test Set (N={len(test_beta)})", loc="over right")
    plt.legend()
    mplhep.mpl_magic()
    plt.ylim(1, None)

    plt.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### OC Threshold Selection
    """)
    return


@app.cell(hide_code=True)
def _(
    CaloDataset,
    PRJ_ROOT,
    best_model,
    config,
    mo,
    np,
    oc_layout,
    split_oc_outputs,
):
    mo.stop(
        best_model is None,
        mo.md("No model found for this run. Cannot compute predictions on validation set."),
    )
    val_data = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split="val").as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=config["data"]["feature_names"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=config["training"].get("normalize_features", True),
        seed=config.get("seed", 0),
    )
    val_eval_data = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split="val").as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=["x", "y", "z", "energy"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=False,
        seed=config.get("seed", 0),
    )
    np.testing.assert_array_equal(val_eval_data["mask"], val_data["mask"])
    np.testing.assert_array_equal(val_eval_data["hit_object_id"], val_data["hit_object_id"])
    val_preds = best_model.predict(val_data["features"], batch_size=1024, verbose=1)
    val_output_slices = split_oc_outputs(val_preds, oc_layout)
    val_beta = np.asarray(val_output_slices.beta)
    val_cluster_coords = np.asarray(val_output_slices.cluster_coords)
    return val_beta, val_cluster_coords, val_data, val_eval_data


@app.cell(hide_code=True)
def _():
    from fastgnn.evaluation import (
        binned_efficiency,
        binned_fake_rate,
        count_clusters_from_labels,
        count_pred_objects,
        count_truth_objects,
        evaluate_oc_padded,
        grid_search_thresholds,
    )
    from fastgnn.training.objectcondensation_loss import get_clustering_np

    return (
        binned_efficiency,
        binned_fake_rate,
        count_clusters_from_labels,
        count_pred_objects,
        count_truth_objects,
        evaluate_oc_padded,
        get_clustering_np,
        grid_search_thresholds,
    )


@app.cell
def _(mo):
    threshold_objective_selector = mo.ui.dropdown(
        options=["matched_f1", "count_median"],
        label="Threshold Objective",
        value="matched_f1",
    )
    matching_algorithm_selector = mo.ui.dropdown(
        options=["hungarian", "greedy"],
        label="Matching Algorithm",
        value="hungarian",
    )
    matching_distance_threshold_selector = mo.ui.number(
        label="Distance Threshold in cm", value=50, step=0.5
    )
    matching_energy_ratio_min_selector = mo.ui.number(
        label="Minimum Energy Ratio", value=0.01, step=0.01
    )
    matching_energy_ratio_max_selector = mo.ui.number(
        label="Maximum Energy Ratio", value=10.0, step=0.01
    )
    hungarian_energy_ratio_log_weight_selector = mo.ui.number(
        label="Hungarian Energy Ratio Log Weight", value=0.0, step=0.1
    )
    start_calibration_button = mo.ui.run_button(
        label="Start threshold calibration",
        kind="success",
    )
    mo.vstack(
        [
            mo.md(
                "**Select the threshold objective and matching parameters used for calibration.**"
            ),
            threshold_objective_selector,
            mo.hstack(
                [
                    matching_algorithm_selector,
                    matching_distance_threshold_selector,
                    matching_energy_ratio_min_selector,
                    matching_energy_ratio_max_selector,
                    hungarian_energy_ratio_log_weight_selector,
                ]
            ),
            start_calibration_button,
        ]
    )
    return (
        hungarian_energy_ratio_log_weight_selector,
        matching_algorithm_selector,
        matching_distance_threshold_selector,
        matching_energy_ratio_max_selector,
        matching_energy_ratio_min_selector,
        start_calibration_button,
        threshold_objective_selector,
    )


@app.cell(hide_code=True)
def _(
    datetime,
    grid_search_thresholds,
    hungarian_energy_ratio_log_weight_selector,
    json,
    matching_algorithm_selector,
    matching_distance_threshold_selector,
    matching_energy_ratio_max_selector,
    matching_energy_ratio_min_selector,
    mo,
    np,
    pl,
    selected_run_dir,
    start_calibration_button,
    threshold_objective_selector,
    val_beta,
    val_cluster_coords,
    val_data,
    val_eval_data,
):
    mo.stop(
        not start_calibration_button.value,
        mo.md("Select a threshold objective, then click **Start threshold calibration**."),
    )
    _tbeta_values = np.linspace(0.001, 0.9, 50)
    _td_values = np.linspace(0.05, 2.0, 40)
    _progress_title = "Scanning OC thresholds"
    _progress_subtitle = (
        f"{len(_tbeta_values)} beta thresholds x {len(_td_values)} distance thresholds"
    )

    def _float_list(values):
        return [float(value) for value in values]

    def _read_cache(path):
        if not path.exists():
            return {"schema_version": 1, "records": []}
        with path.open() as f:
            cache = json.load(f)
        cache.setdefault("schema_version", 1)
        cache.setdefault("records", [])
        return cache

    def _write_cache(path, cache):
        with path.open("w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
            f.write("\n")

    _calibration_options = {
        "schema_version": 1,
        "objective": threshold_objective_selector.value,
        "score_formula": "F1_E + 0.5 * F1_obj",
        "tbeta_values": _float_list(_tbeta_values),
        "td_values": _float_list(_td_values),
        "feature_names": ["x", "y", "z", "energy"],
    }
    if threshold_objective_selector.value == "matched_f1":
        _calibration_options["matching"] = {
            "algorithm": matching_algorithm_selector.value,
            "max_match_distance": float(matching_distance_threshold_selector.value),
            "min_energy_ratio": float(matching_energy_ratio_min_selector.value),
            "max_energy_ratio": float(matching_energy_ratio_max_selector.value),
            "hungarian_energy_ratio_log_weight": float(
                hungarian_energy_ratio_log_weight_selector.value
            ),
        }
    _cache_path = selected_run_dir / "oc_threshold_calibrations.json"
    _cache = _read_cache(_cache_path)
    _cached_record = next(
        (
            record
            for record in _cache["records"]
            if record.get("calibration_options") == _calibration_options
        ),
        None,
    )

    if _cached_record is None:
        best_oc_thresholds, _grid_results = grid_search_thresholds(
            beta=val_beta,
            cluster_coords=val_cluster_coords,
            hit_object_id=val_data["hit_object_id"],
            mask=val_data.get("mask", None),
            tbeta_values=_tbeta_values,
            td_values=_td_values,
            objective=threshold_objective_selector.value,
            features=val_eval_data["features"],
            feature_names=["x", "y", "z", "energy"],
            max_match_distance=matching_distance_threshold_selector.value,
            min_energy_ratio=matching_energy_ratio_min_selector.value,
            max_energy_ratio=matching_energy_ratio_max_selector.value,
            matching_algorithm=matching_algorithm_selector.value,
            hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight_selector.value,
            progress=lambda values: mo.status.progress_bar(
                values,
                title=_progress_title,
                subtitle=_progress_subtitle,
                completion_title="OC threshold calibration complete",
            ),
        )
        _record = {
            "calibration_options": _calibration_options,
            "best_thresholds": {
                key: value.item() if hasattr(value, "item") else value
                for key, value in best_oc_thresholds.items()
            },
            "created_at": datetime.now().astimezone().isoformat(),
        }
        _cache["records"].append(_record)
        _write_cache(_cache_path, _cache)
        _status = "Computed threshold calibration and wrote cache."
    else:
        best_oc_thresholds = _cached_record["best_thresholds"]
        _status = "Loaded threshold calibration from cache."

    mo.vstack(
        [
            mo.md(f"**{_status}**"),
            mo.md(f"Cache file: `{_cache_path}`"),
            mo.ui.table(pl.DataFrame([best_oc_thresholds])),
        ]
    )
    return (best_oc_thresholds,)


@app.cell(hide_code=True)
def _(
    best_oc_thresholds,
    evaluate_oc_padded,
    hungarian_energy_ratio_log_weight_selector,
    matching_algorithm_selector,
    matching_distance_threshold_selector,
    matching_energy_ratio_max_selector,
    matching_energy_ratio_min_selector,
    oc_layout,
    test_data,
    test_ds,
    test_eval_data,
    test_preds,
):
    _tbeta, _td = best_oc_thresholds["tbeta"], best_oc_thresholds["td"]
    oc_eval = evaluate_oc_padded(
        preds=test_preds,
        hit_object_id=test_data["hit_object_id"],
        mask=test_data["mask"],
        features=test_eval_data["features"],
        feature_names=["x", "y", "z", "energy"],
        tbeta=_tbeta,
        td=_td,
        layout=oc_layout,
        events=[test_ds[i] for i in range(len(test_ds))],
        max_match_distance=matching_distance_threshold_selector.value,
        min_energy_ratio=matching_energy_ratio_min_selector.value,
        max_energy_ratio=matching_energy_ratio_max_selector.value,
        matching_algorithm=matching_algorithm_selector.value,
        hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight_selector.value,
    )
    count_summary = oc_eval.count_summary()
    seed_summary = oc_eval.seed_summary()
    matching_summary = oc_eval.matching_summary()
    print(count_summary, "\n\n", seed_summary, "\n\n", matching_summary)
    return (oc_eval,)


@app.cell
def _(PRJ_ROOT, mo, selected_run_dir):
    default_path = (
        PRJ_ROOT / "plotting" / "efficiency_fake_rate" / f"{selected_run_dir.name}_oc_eval.pkl"
    )
    oc_eval_pickle_path = mo.ui.text(
        label="OC evaluation pickle path",
        value=str(default_path),
        full_width=True,
    )
    save_oc_eval_button = mo.ui.run_button(
        label="Save OC evaluation pickle",
        kind="success",
    )

    _items = [oc_eval_pickle_path, save_oc_eval_button]
    mo.vstack(_items)
    return oc_eval_pickle_path, save_oc_eval_button


@app.cell
def _(
    PRJ_ROOT,
    Path,
    mo,
    oc_eval,
    oc_eval_pickle_path,
    pickle,
    save_oc_eval_button,
):
    _out = None
    if save_oc_eval_button.value:
        _path = Path(oc_eval_pickle_path.value).expanduser()
        if not _path.is_absolute():
            _path = PRJ_ROOT / _path
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _path.open("wb") as _f:
            pickle.dump(oc_eval, _f)
            _out = mo.md(f"Saved OC evaluation pickle to `{_path}`.")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Efficiency and Fake Rate
    """)
    return


@app.cell
def _():
    from fastgnn.plotting import plot_binned_efficiency, plot_binned_fake_rate

    return plot_binned_efficiency, plot_binned_fake_rate


@app.cell
def _(
    PLOTTING_CONFIG,
    binned_efficiency,
    binned_fake_rate,
    log_edges,
    np,
    oc_eval,
    plot_binned_efficiency,
    plot_binned_fake_rate,
    plt,
    test_ds,
):
    _truth_energy = oc_eval.truth["truth_energy"]
    _bins_truth = log_edges(_truth_energy)

    _truth_energy_table = binned_efficiency(
        oc_eval.truth,
        "truth_energy",
        _bins_truth,
    )

    threshold = test_ds.metadata["preprocessing"]["truth_min_object_energy"]

    _pred_energy = oc_eval.predicted["assigned_cluster_energy"]
    _bins_pred = np.concatenate(
        [
            log_edges(
                _pred_energy.filter(_pred_energy < threshold),
                bins=3,
            ),
            _bins_truth[1:],
        ]
    )

    _pred_energy_table = binned_fake_rate(
        oc_eval.predicted,
        "assigned_cluster_energy",
        _bins_pred,
    )

    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"],
    )

    plot_binned_efficiency(_axs[0], _truth_energy_table, color="black")
    _axs[0].set_xscale("log")
    _axs[0].set_xlabel("Cluster truth energy [GeV]")
    _axs[0].set_ylabel("Efficiency")
    _axs[0].axvline(threshold, color="grey", linestyle="--")
    _axs[0].set_xlim(7e-1, 1e3)
    _axs[0].set_ylim(0.0, 1.0)

    plot_binned_fake_rate(_axs[1], _pred_energy_table, color="black")
    _axs[1].set_xscale("log")
    _axs[1].set_xlabel("Cluster predicted energy [GeV]")
    _axs[1].set_ylabel("Fake Rate")
    _axs[1].axvline(threshold, color="grey", linestyle="--")
    _axs[1].set_xlim(1.5e-1, 1e3)
    _axs[1].set_ylim(-0.02, 1.0)

    assert (_pred_energy_table["fake_rate"] <= _axs[1].get_ylim()[1]).all()

    _fig1.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cluster Size and $\beta$
    """)
    return


@app.cell
def _(
    PLOTTING_CONFIG,
    binned_efficiency,
    binned_fake_rate,
    log_edges,
    oc_eval,
    plot_binned_efficiency,
    plot_binned_fake_rate,
    plt,
):
    _fig1, _axs = plt.subplots(1, 3, figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"])

    for _ax, _col, _xlabel, _metric, _frame, _bin_values in zip(
        _axs,
        ["truth_n_hits", "cluster_size", "seed_beta"],
        ["True cluster size", "Predicted cluster size", r"$\beta$ score of OC seed"],
        ["efficiency", "fake_rate", "fake_rate"],
        [oc_eval.truth, oc_eval.predicted, oc_eval.predicted],
        [
            oc_eval.truth["truth_n_hits"],
            oc_eval.predicted["cluster_size"],
            oc_eval.predicted["seed_beta"],
        ],
    ):
        _bins = log_edges(_bin_values)

        if _metric == "efficiency":
            _table = binned_efficiency(_frame, _col, _bins)
        else:
            _table = binned_fake_rate(_frame, _col, _bins)

        if _metric == "efficiency":
            plot_binned_efficiency(_ax, _table, color="black")
        else:
            plot_binned_fake_rate(_ax, _table, color="black")
        _ax.set_xscale("log")
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel("Efficiency" if _metric == "efficiency" else "Fake Rate")
        _ax.set_ylim(-0.02, 1.0)
        _ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        _ax.grid(alpha=0.25)

    plt.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy and Position Regression
    """)
    return


@app.cell
def _(itertools, np, pl):
    REFERENCE_LINE_KWARGS = {"color": "green", "linestyle": "--"}

    def energy_weighted_str(s: str) -> str:
        """Wraps a Latex string (must start and end with $) to mark weighting by energy."""
        assert s[0] == "$" and s[-1] == "$"
        return r"$\langle " + s[1:-1] + r"\rangle_\mathrm{E}$"

    def log_edges(values, bins=12):
        values = np.asarray(values, dtype=float)
        return np.logspace(
            np.log10(values.min()),
            np.log10(values.max()),
            bins + 1,
        )

    def binned_profile(x, y, bins):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        bins = np.asarray(bins, dtype=float)

        assert x.shape == y.shape
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        assert np.all(np.diff(bins) > 0)

        idx = np.digitize(x, bins) - 1

        rows = []
        for i, (lo, hi) in enumerate(itertools.pairwise(bins)):
            values = y[idx == i]
            n = values.size

            if n:
                q16, q84 = np.quantile(values, [0.16, 0.84])
                rows.append(
                    {
                        "bin_low": lo,
                        "bin_high": hi,
                        "n": int(n),
                        "mean": values.mean(),
                        "median": np.median(values),
                        "sigma68": 0.5 * (q84 - q16),
                    }
                )
            else:
                rows.append(
                    {
                        "bin_low": lo,
                        "bin_high": hi,
                        "n": 0,
                        "mean": None,
                        "median": None,
                        "sigma68": None,
                    }
                )

        return pl.DataFrame(rows)

    def plot_profile_points(ax, table, y_col, *, color="black", marker="o", label=None):
        if table.is_empty():
            return
        table = table.filter(pl.col(y_col).is_not_null())
        if table.is_empty():
            return
        bin_low = table["bin_low"].to_numpy()
        bin_high = table["bin_high"].to_numpy()
        x = np.where(
            (bin_low > 0) & (bin_high > 0), np.sqrt(bin_low * bin_high), 0.5 * (bin_low + bin_high)
        )
        xerr = [x - bin_low, bin_high - x]
        ax.errorbar(
            x,
            table[y_col].to_numpy(),
            xerr=xerr,
            fmt=marker,
            color=color,
            capsize=2,
            label=label,
        )

    return (
        REFERENCE_LINE_KWARGS,
        binned_profile,
        energy_weighted_str,
        log_edges,
        plot_profile_points,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Energy and Transverse Energy Response
    """)
    return


@app.cell
def _(mo):
    show_median_response = mo.ui.checkbox(
        label="Show median response in profile plots", value=False
    )
    show_median_response
    return (show_median_response,)


@app.cell
def _(
    PLOTTING_CONFIG,
    REFERENCE_LINE_KWARGS,
    binned_profile,
    energy_weighted_str,
    log_edges,
    mo,
    np,
    oc_eval,
    plot_profile_points,
    plt,
    show_median_response,
):
    ## Response histograms for energy and transverse energy
    _fig1, _axs = plt.subplots(
        nrows=1, ncols=2, figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"]
    )

    _matches = oc_eval.matches
    _energy_response = _matches["energy_response"]
    _et_response = _matches["et_response"]

    for _ax, _values, _title, _xlabel in [
        (_axs[0], _energy_response, "Energy Response", r"$E_\mathrm{pred}/E_\mathrm{true}$"),
        (
            _axs[1],
            _et_response,
            r"Transverse Energy Response",
            r"$E_{T,\mathrm{pred}}/E_{T,\mathrm{true}}$",
        ),
    ]:
        _upper_lim = 15
        _bins = np.linspace(0, _upper_lim, 50)
        _ax.hist(_values, bins=_bins, color="black", histtype="step")
        _ax.axvline(1.0, **REFERENCE_LINE_KWARGS)
        _ax.set_xlim(None, _upper_lim)
        _ax.set_xlabel(_xlabel)
        _ax.set_xlim(0, _upper_lim)
        _ax.set_ylim(1e-3, None)
        _txt1 = "Overflow:\nMean:"
        _txt2 = f"{(_values > _upper_lim).sum() / len(_values):.2%} \n{_values.mean():.2f}"
        _ax.text(0.6, 0.95, _txt1, transform=_ax.transAxes, ha="left", va="top", fontsize="small")
        _ax.text(0.82, 0.95, _txt2, transform=_ax.transAxes, ha="left", va="top", fontsize="small")
    _axs[0].set_ylabel("Matched clusters")
    plt.tight_layout()

    ## Response vs truth for transverse energy and eta
    _fig2, _axs = plt.subplots(
        nrows=1, ncols=2, figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"]
    )
    _aggs = ["mean", "median"] if show_median_response.value else ["mean"]

    _truth_et, _et_response_vs_et = (
        _matches.select(["truth_et", "et_response"]).drop_nulls().to_numpy().T
    )
    _truth_eta, _et_response_vs_eta = (
        _matches.select(["truth_centroid_eta", "et_response"]).drop_nulls().to_numpy().T
    )

    _axs[0].scatter(_truth_et, _et_response_vs_et, alpha=0.25, s=10, color="grey")
    for _agg in _aggs:
        plot_profile_points(
            _axs[0],
            binned_profile(_truth_et, _et_response_vs_et, log_edges(_truth_et)),
            _agg,
            label=_agg.capitalize(),
            color="black" if _agg == "mean" else "purple",
        )
    _axs[0].set_xscale("log")
    _axs[0].set_xlabel(r"$E_{T,\mathrm{true}}$ [GeV]")
    _axs[0].set_ylabel(r"$E_{T,\mathrm{pred}}/E_{T,\mathrm{true}}$")

    _abs_eta = np.abs(_truth_eta)
    _axs[1].scatter(_abs_eta, _et_response_vs_eta, alpha=0.25, s=10, color="grey")
    _eta_bins = np.linspace(np.quantile(_abs_eta, 0.01), np.quantile(_abs_eta, 0.99), 12)
    for _agg in _aggs:
        plot_profile_points(
            _axs[1],
            binned_profile(_abs_eta, _et_response_vs_eta, _eta_bins),
            _agg,
            label=_agg.capitalize(),
            color="black" if _agg == "mean" else "purple",
        )
    _axs[1].set_xlabel(energy_weighted_str(r"$\eta_\mathrm{{true}}$"))

    for _ax in _axs:
        _ax.grid(alpha=0.25)
        _ax.set_yscale("log")
        _ax.axhline(1.0, **REFERENCE_LINE_KWARGS)
        _ax.legend()
    plt.tight_layout()

    ## Predicted vs true energy and transverse energy
    _fig3, _axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"],
    )

    for _ax, _en, _label in [
        (_axs[0], "energy", "E_{"),
        (_axs[1], "et", "E_{T,"),
    ]:
        _truth, _pred = (
            oc_eval.matches.select(f"truth_{_en}", f"sum_{_en}_reco").drop_nulls().to_numpy().T
        )
        _truth_et_all = oc_eval.truth.select(f"truth_{_en}").drop_nulls().to_numpy()
        _et_bins = log_edges(_truth_et_all)

        _ax.scatter(_truth, _pred, alpha=0.25, s=10, color="grey")
        plot_profile_points(
            _ax,
            binned_profile(_truth, _pred, log_edges(_truth)),
            "mean",
        )
        _low = min(float(_truth.min()), float(_pred.min()))
        _high = max(float(_truth.max()), float(_pred.max()))
        _ax.plot([_low, _high], [_low, _high], **REFERENCE_LINE_KWARGS)
        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set(
            xlabel=rf"${_label}" + r"\mathrm{{true}}}$ [GeV]",
            ylabel=rf"${_label}" + r"\mathrm{{pred}}}$ [GeV]",
        )
        _axs[0].grid(alpha=0.25)

    plt.tight_layout()
    _fig3

    mo.vstack([_fig1, _fig2, _fig3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Position Resolution
    """)
    return


@app.cell
def _(PLOTTING_CONFIG, energy_weighted_str, oc_eval, plt):
    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"],
    )
    for _ax, _col, _label in [
        (
            _axs[0],
            "eta_residual",
            energy_weighted_str(r"$\eta_\mathrm{true}$")
            + " - "
            + energy_weighted_str(r"$\eta_\mathrm{pred}$"),
        ),
        (
            _axs[1],
            "phi_residual",
            energy_weighted_str(r"$\phi_\mathrm{true}$")
            + " - "
            + energy_weighted_str(r"$\phi_\mathrm{pred}$"),
        ),
        (
            _axs[2],
            "z_residual",
            energy_weighted_str(r"$z_\mathrm{true}$")
            + " - "
            + energy_weighted_str(r"$z_\mathrm{pred}$")
            + " [cm]",
        ),
    ]:
        _values = oc_eval.matches[_col].drop_nulls()
        if len(_values):
            _ax.hist(_values, bins=50, color="black", histtype="step")
            _ax.axvline(0.0, color="grey", linestyle="--")
        else:
            _ax.text(0.5, 0.5, "No entries", transform=_ax.transAxes, ha="center", va="center")
        _ax.set_xlabel(_label)
        _ax.grid(alpha=0.25)
    _axs[0].set_ylabel("Matched clusters")

    plt.tight_layout()
    _fig1
    return


@app.cell
def _(
    PLOTTING_CONFIG,
    REFERENCE_LINE_KWARGS,
    binned_profile,
    energy_weighted_str,
    log_edges,
    oc_eval,
    plot_profile_points,
    plt,
):
    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"],
    )
    for _ax, _col, _ylabel in [
        (_axs[0], "relative_eta_residual", energy_weighted_str(r"$\Delta_\mathrm{rel}\eta$")),
        (_axs[1], "relative_phi_residual", energy_weighted_str(r"$\Delta_\mathrm{rel}\phi$")),
        (_axs[2], "relative_z_residual", energy_weighted_str(r"$\Delta_\mathrm{rel}z$")),
    ]:
        _truth_et, _values = oc_eval.matches.select("truth_et", _col).drop_nulls().to_numpy().T
        if len(_truth_et):
            _table = binned_profile(_truth_et, _values, log_edges(_truth_et))
            plot_profile_points(_ax, _table, "mean")
            _ax.axhline(0.0, **REFERENCE_LINE_KWARGS)
            _ax.set_xscale("log")
        else:
            _ax.text(0.5, 0.5, "No entries", transform=_ax.transAxes, ha="center", va="center")
        _ax.set(
            xlabel=r"$E_{T,\mathrm{true}}$ [GeV]",
            ylabel=_ylabel,
        )
        _ax.grid(alpha=0.25)
    plt.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering Performance
    """)
    return


@app.cell
def _(
    REFERENCE_LINE_KWARGS,
    best_oc_thresholds,
    count_pred_objects,
    count_truth_objects,
    mplhep,
    np,
    plt,
    test_beta,
    test_cluster_coords,
    test_data,
):
    mplhep.style.use("CMS")

    _tbeta, _td = best_oc_thresholds["tbeta"], best_oc_thresholds["td"]

    _n_truth_test = count_truth_objects(
        test_data["hit_object_id"], mask=test_data.get("mask", None)
    )
    _n_pred_test = count_pred_objects(
        test_beta, test_cluster_coords, _tbeta, _td, mask=test_data.get("mask", None)
    )

    _fig1, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(11, 5))

    _ax0.scatter(_n_truth_test, _n_pred_test, alpha=0.45, s=18)
    _ax0.plot(
        [0, max(_n_truth_test.max(), _n_pred_test.max())],
        [0, max(_n_truth_test.max(), _n_pred_test.max())],
        **REFERENCE_LINE_KWARGS,
    )
    _ax0.set(
        xlabel="# truth clusters",
        ylabel="# predicted clusters",
        xlim=(-0.5, max(_n_truth_test.max(), _n_pred_test.max()) + 0.5),
        ylim=(-0.5, max(_n_truth_test.max(), _n_pred_test.max()) + 0.5),
    )
    _ax0.set_aspect("equal", adjustable="box")
    _ax0.grid(alpha=0.25)

    mplhep.add_text(
        rf"$t_\beta={_tbeta:.3f}$, $t_d={_td:.3f}$",
        ax=_ax0,
        loc="upper left",
        fontsize="small",
    )

    _cts_diff = _n_truth_test - _n_pred_test
    _diffs, _cts = np.unique(_cts_diff, return_counts=True)
    _ax1.bar(_diffs, _cts, width=0.8)
    _ax1.set(xlabel="# truth - # pred clusters")
    _ax1.grid(alpha=0.25)
    _ax1.set_xlim(None, 25)

    plt.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single Event Analysis and Visualization
    """)
    return


@app.cell
def _(
    count_clusters_from_labels,
    count_truth_objects,
    get_clustering_np,
    np,
    oc_layout,
    split_oc_outputs,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    zoom = 2
    _color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
        "#3182bd",
        "#e6550d",
        "#31a354",
        "#756bb1",
        "#636363",
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
    ]

    def color_at(index):
        if index < len(_color_palette):
            return _color_palette[index]
        hue = (index * 137) % 360
        return f"hsl({hue}, 65%, 45%)"

    def truth_object_colors(hit_object_ids):
        _hit_object_ids = np.asarray(hit_object_ids)
        _truth_ids = np.unique(_hit_object_ids[_hit_object_ids > 0])
        return {int(object_id): color_at(i) for i, object_id in enumerate(_truth_ids.astype(int))}

    def apply_truth_colors_to_event_display(fig, event, views=("xy", "yz", "3d")):
        _hit_object_id = np.asarray(event.truth.hit_object_id)
        _hit_energy = np.asarray(event.hits.energy)
        _truth_colors = truth_object_colors(_hit_object_id)
        _unique_objects = np.unique(_hit_object_id)
        _cluster_energies = {
            int(object_id): _hit_energy[_hit_object_id == object_id].sum()
            for object_id in _unique_objects
            if object_id != 0
        }
        _sorted_objects = [
            0,
            *sorted(
                _cluster_energies,
                key=lambda object_id: _cluster_energies[object_id],
                reverse=True,
            ),
        ]
        _trace_index = 0
        for _object_id in _sorted_objects:
            _color = "rgba(150,150,150,0.45)" if _object_id == 0 else _truth_colors[int(_object_id)]
            for _ in views:
                if _trace_index >= len(fig.data):
                    return fig
                fig.data[_trace_index].marker.color = _color
                _trace_index += 1
        return fig

    def project_event_coords(coords, projection_mode, raw_dims, global_pca):
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2:
            raise ValueError(f"Expected 2D coordinates for plotting, got shape {coords.shape}")

        if projection_mode == "Validation PCA":
            centered = coords - global_pca["mean"]
            projected = centered @ global_pca["components"].T
            display = np.zeros((len(coords), 3), dtype=np.float64)
            available = min(3, projected.shape[1])
            if available > 0:
                display[:, :available] = projected[:, :available]
            axis_labels = [f"PC{i + 1}" for i in range(3)]
            projection_label = "Validation PCA"
        else:
            display = np.column_stack([coords[:, dim] for dim in raw_dims])
            axis_labels = [f"c{dim + 1}" for dim in raw_dims]
            projection_label = "Raw dims"

        return display, axis_labels, projection_label

    def plot_true_vs_pred_oc(
        tbeta,
        td,
        event,
        preds,
        hit_object_id,
        projection_mode,
        raw_dims,
        global_pca,
        mask=None,
        oc_eval=None,
        event_idx=0,
    ):
        _outputs = split_oc_outputs(preds[event_idx : event_idx + 1], oc_layout)
        _hit_object_id = hit_object_id[event_idx]

        if mask is None:
            m = np.ones(len(_hit_object_id), dtype=bool)
        else:
            m = mask[event_idx].astype(bool)

        _beta = np.asarray(_outputs.beta[0])[m]
        _coords = np.asarray(_outputs.cluster_coords[0])[m]
        _display_coords, _axis_labels, _projection_label = project_event_coords(
            _coords,
            projection_mode=projection_mode,
            raw_dims=raw_dims,
            global_pca=global_pca,
        )
        pred_c1 = _display_coords[:, 0]
        pred_c2 = _display_coords[:, 1]
        pred_c3 = _display_coords[:, 2]

        _hit_object_id = _hit_object_id[m]
        signal = _hit_object_id > 0
        noise = ~signal
        signal_ids = _hit_object_id[signal]
        _clustering = get_clustering_np(
            _beta,
            _coords,
            tbeta=tbeta,
            td=td,
        )
        _n_pred_clusters = count_clusters_from_labels(_clustering)
        _n_truth_clusters = count_truth_objects(
            hit_object_id[event_idx : event_idx + 1],
            mask=None if mask is None else mask[event_idx : event_idx + 1],
        )[0]

        _seed_indices = np.unique(_clustering[_clustering >= 0])
        _truth_ids = np.unique(signal_ids).astype(int)
        _truth_colors = truth_object_colors(signal_ids)

        _seed_to_truth = event_seed_truth_matches(oc_eval, event_idx)
        _truth_to_seed = {truth_id: seed_index for seed_index, truth_id in _seed_to_truth.items()}
        _n_matched_clusters = len(_seed_to_truth)
        _n_unpadded_hits = int(m.sum())
        _fake_seed_indices = [
            int(seed_index) for seed_index in _seed_indices if int(seed_index) not in _seed_to_truth
        ]
        _fake_seed_colors = {
            seed_index: color_at(len(_truth_ids) + i)
            for i, seed_index in enumerate(_fake_seed_indices)
        }

        _fig = make_subplots(
            rows=2,
            cols=6,
            specs=[
                [
                    {"type": "scene", "colspan": 3},
                    None,
                    None,
                    {"type": "scene", "colspan": 3},
                    None,
                    None,
                ],
                [
                    {"type": "xy", "colspan": 2},
                    None,
                    {"type": "xy", "colspan": 2},
                    None,
                    {"type": "xy", "colspan": 2},
                    None,
                ],
            ],
            subplot_titles=(
                f"Truth assignment in OC space ({_projection_label})",
                f"Predicted clusters in OC space ({_projection_label})",
                "Event beta distribution",
                "Truth cluster sizes",
                "Predicted OC seed sizes",
            ),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        def _marker_size(values):
            return np.clip(5 + np.asarray(values) * 28, 5, 34)

        def _hover(object_label):
            return (
                f"{_axis_labels[0]}=%{{x:.3f}}<br>"
                f"{_axis_labels[1]}=%{{y:.3f}}<br>"
                f"{_axis_labels[2]}=%{{z:.3f}}<br>"
                "β=%{customdata:.3f}<extra>" + object_label + "</extra>"
            )

        if noise.any():
            _fig.add_trace(
                go.Scatter3d(
                    x=pred_c1[noise],
                    y=pred_c2[noise],
                    z=pred_c3[noise],
                    customdata=_beta[noise],
                    mode="markers",
                    marker={"color": "rgba(150,150,150,0.45)", "size": 4},
                    name="truth noise",
                    showlegend=False,
                    hovertemplate=_hover("truth noise"),
                ),
                row=1,
                col=1,
            )

        for _object_id in _truth_ids:
            _object_mask = signal & (_hit_object_id == _object_id)
            _fig.add_trace(
                go.Scatter3d(
                    x=pred_c1[_object_mask],
                    y=pred_c2[_object_mask],
                    z=pred_c3[_object_mask],
                    customdata=_beta[_object_mask],
                    mode="markers",
                    marker={
                        "color": _truth_colors[int(_object_id)],
                        "size": _marker_size(_beta[_object_mask]),
                        "opacity": 0.78,
                    },
                    name=f"truth object {int(_object_id)}",
                    showlegend=False,
                    hovertemplate=_hover(f"truth object {int(_object_id)}"),
                ),
                row=1,
                col=1,
            )

        _unassigned = _clustering < 0
        if _unassigned.any():
            _fig.add_trace(
                go.Scatter3d(
                    x=pred_c1[_unassigned],
                    y=pred_c2[_unassigned],
                    z=pred_c3[_unassigned],
                    customdata=_beta[_unassigned],
                    mode="markers",
                    marker={"color": "rgba(190,170,135,0.65)", "size": 4},
                    name="unassigned hits",
                    showlegend=False,
                    hovertemplate=_hover("unassigned hits"),
                ),
                row=1,
                col=4,
            )

        for _seed_index in _seed_indices:
            _seed_index = int(_seed_index)
            _cluster_mask = _clustering == _seed_index
            _matched_truth = _seed_to_truth.get(_seed_index)
            _color = (
                _truth_colors[_matched_truth]
                if _matched_truth in _truth_colors
                else _fake_seed_colors[_seed_index]
            )
            _label = (
                f"seed {_seed_index} → object {_matched_truth}"
                if _matched_truth is not None
                else f"fake seed hit {_seed_index}"
            )
            _fig.add_trace(
                go.Scatter3d(
                    x=pred_c1[_cluster_mask],
                    y=pred_c2[_cluster_mask],
                    z=pred_c3[_cluster_mask],
                    customdata=_beta[_cluster_mask],
                    mode="markers",
                    marker={
                        "color": _color,
                        "size": _marker_size(_beta[_cluster_mask]),
                        "opacity": 0.78,
                    },
                    name=_label,
                    showlegend=False,
                    hovertemplate=_hover(_label),
                ),
                row=1,
                col=4,
            )

        if len(_seed_indices) > 0:
            _fig.add_trace(
                go.Scatter3d(
                    x=pred_c1[_seed_indices],
                    y=pred_c2[_seed_indices],
                    z=pred_c3[_seed_indices],
                    mode="markers",
                    marker={
                        "symbol": "x",
                        "color": "black",
                        "size": 2,
                        "line": {"color": "black", "width": 1},
                    },
                    name="OC seeds",
                    showlegend=False,
                    hovertemplate="seed hit index=%{text}<extra>OC seed</extra>",
                    text=[str(int(seed_index)) for seed_index in _seed_indices],
                ),
                row=1,
                col=4,
            )

        _bins = np.linspace(0, max(1.0, float(_beta.max())), 40)
        if noise.any():
            _fig.add_trace(
                go.Histogram(
                    x=_beta[noise],
                    xbins={
                        "start": float(_bins[0]),
                        "end": float(_bins[-1]),
                        "size": float(_bins[1] - _bins[0]),
                    },
                    name=f"Noise ({noise.sum()} hits)",
                    marker_color="rgba(150,150,150,0.65)",
                    opacity=0.65,
                ),
                row=2,
                col=1,
            )
        if signal.any():
            _fig.add_trace(
                go.Histogram(
                    x=_beta[signal],
                    xbins={
                        "start": float(_bins[0]),
                        "end": float(_bins[-1]),
                        "size": float(_bins[1] - _bins[0]),
                    },
                    name=f"Signal ({signal.sum()} hits)",
                    marker_color="#1f77b4",
                    opacity=0.65,
                ),
                row=2,
                col=1,
            )
        _fig.add_shape(
            type="line",
            x0=tbeta,
            x1=tbeta,
            y0=0,
            y1=1,
            xref="x",
            yref="y domain",
            line={"color": "black", "dash": "dash", "width": 1},
        )

        _truth_labels, _truth_counts = np.unique(signal_ids.astype(int), return_counts=True)
        if len(_truth_labels) > 0:
            _truth_order = np.argsort(_truth_counts)[::-1]
            _truth_labels = _truth_labels[_truth_order]
            _truth_counts = _truth_counts[_truth_order]
            _fig.add_trace(
                go.Bar(
                    x=[str(int(label)) for label in _truth_labels],
                    y=_truth_counts,
                    marker_color=[_truth_colors[int(label)] for label in _truth_labels],
                    name="Truth object hits",
                    showlegend=False,
                    hovertemplate="truth object=%{x}<br>valid hits=%{y}<extra></extra>",
                ),
                row=2,
                col=3,
            )

        _labels, _counts = np.unique(_clustering[_clustering >= 0], return_counts=True)
        _bar_colors = []
        if len(_labels) > 0:
            for _label in _labels:
                _matched_truth = _seed_to_truth.get(int(_label))
                _bar_colors.append(
                    _truth_colors[_matched_truth]
                    if _matched_truth in _truth_colors
                    else _fake_seed_colors[int(_label)]
                )
            _fig.add_trace(
                go.Bar(
                    x=[str(int(label)) for label in _labels],
                    y=_counts,
                    marker_color=_bar_colors,
                    name="Assigned hits",
                    showlegend=False,
                    hovertemplate="seed hit index=%{x}<br>assigned hits=%{y}<extra></extra>",
                ),
                row=2,
                col=5,
            )

        _truth_entries = truth_object_legend_entries(event, signal_ids, _truth_colors, top_n=10)
        _truth_entries = [
            (
                color,
                label,
                add_seed_arrow_to_truth_description(
                    description, _truth_to_seed.get(int(label.split()[-1]))
                ),
            )
            for color, label, description in _truth_entries
        ]
        _matched_entries = [
            (
                _truth_colors[truth_id],
                f"seed hit {seed_index} → object {truth_id}",
            )
            for seed_index, truth_id in sorted(_seed_to_truth.items())
            if truth_id in _truth_colors
        ]
        _fake_entries = [
            (
                _fake_seed_colors[int(seed_index)],
                f"fake seed hit {int(seed_index)}",
            )
            for seed_index in _fake_seed_indices
        ]
        _truth_text = "<b>Truth clusters:</b><br>" + "<br>".join(
            f"<span style='color:{color}'>●</span> {description}"
            for color, _, description in _truth_entries
        )
        if noise.any():
            _truth_text += (
                ("<br>" if _truth_entries else "")
                + f"<span style='color:rgba(150,150,150,0.75)'>●</span> noise ({int(noise.sum())} hits)"
            )
        _pred_text = "<b>Predicted clusters:</b><br>" + "<br>".join(
            f"<span style='color:{color}'>●</span> {label}"
            for color, label in _matched_entries + _fake_entries
        )
        if _unassigned.any():
            _pred_text += (
                ("<br>" if (_matched_entries or _fake_entries) else "")
                + f"<span style='color:rgb(190,170,135)'>●</span> unassigned ({int(_unassigned.sum())} hits)"
            )
        if not _truth_entries:
            _truth_text += ("<br>" if noise.any() else "") + "No signal truth clusters"
        if not (_matched_entries or _fake_entries or _unassigned.any()):
            _pred_text += "No predicted clusters"

        for _x, _text in [(0.015, _truth_text), (0.515, _pred_text)]:
            _fig.add_annotation(
                x=_x,
                y=0.43,
                xref="paper",
                yref="paper",
                text=_text,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font={"size": 11},
                bordercolor="#d9d9d9",
                borderwidth=1,
                borderpad=8,
                bgcolor="rgba(255,255,255,0.95)",
            )

        _display_event_id = getattr(event, "event_id", event_idx)
        _camera = {"eye": {"x": -1 / zoom * 2.35, "y": -1 / zoom * 2.35, "z": 1 / zoom * 2.0}}
        _scene_layout = {
            "xaxis_title": _axis_labels[0],
            "yaxis_title": _axis_labels[1],
            "zaxis_title": _axis_labels[2],
            "aspectmode": "cube",
            "camera": _camera,
        }
        _fig.update_layout(
            title={
                "text": (
                    f"Event {_display_event_id} | Truth clusters: {_n_truth_clusters} | "
                    f"Predicted clusters: {_n_pred_clusters} | Matched clusters: {_n_matched_clusters} | "
                    f"Unpadded hits: {_n_unpadded_hits}"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            template="plotly_white",
            height=1080,
            autosize=True,
            barmode="overlay",
            legend={"orientation": "h", "y": -0.06, "x": 0},
            margin={"l": 35, "r": 35, "t": 90, "b": 35},
            scene=_scene_layout,
            scene2=_scene_layout,
            uirevision=f"event-{event_idx}",
        )
        _fig.update_xaxes(title_text="beta", row=2, col=1)
        _fig.update_yaxes(title_text="Hits", type="log", row=2, col=1)
        _fig.update_xaxes(title_text="Truth object id", row=2, col=3)
        _fig.update_yaxes(title_text="Truth-assigned valid hits", row=2, col=3)
        _fig.update_xaxes(title_text="Predicted OC seed hit index", row=2, col=5)
        _fig.update_yaxes(title_text="Assigned valid hits", row=2, col=5)
        return _fig

    def event_seed_truth_matches(oc_eval, event_idx):
        if oc_eval is None or getattr(oc_eval, "matches", None) is None:
            raise ValueError("plot_true_vs_pred_oc requires oc_eval.matches for cluster matching.")
        rows = [
            row
            for row in oc_eval.matches.to_dicts()
            if int(row.get("event_idx", -1)) == int(event_idx)
        ]
        return {
            int(row["seed_index"]): int(row["truth_id"])
            for row in rows
            if row.get("seed_index") is not None and row.get("truth_id") is not None
        }

    def add_seed_arrow_to_truth_description(description, seed_index):
        if seed_index is None:
            return description
        return (
            description[:-1] + f", ← seed hit {seed_index})"
            if description.endswith(")")
            else description
        )

    def pdgid_to_name(pdgid):
        try:
            from particle import Particle

            return Particle.from_pdgid(int(pdgid)).name
        except Exception:
            return str(pdgid)

    def truth_object_legend_entries(event, object_ids, truth_colors, top_n=10):
        if object_ids.size == 0:
            return []

        event.truth.require("objects")
        event.truth.objects.require("impact_energy", "track_pdg_id")

        unique_object_ids = np.unique(object_ids)
        entries = []
        for object_id in unique_object_ids:
            object_index = int(object_id) - 1
            energy = float(event.truth.objects.impact_energy[object_index])
            pdgid = int(event.truth.objects.track_pdg_id[object_index])
            entries.append((energy, int(object_id), pdgid_to_name(pdgid)))

        entries = sorted(entries, reverse=True)[:top_n]
        return [
            (
                truth_colors[int(object_id)],
                f"object {object_id}",
                f"{particle} ({energy:.1f} GeV, object {object_id})",
            )
            for energy, object_id, particle in entries
        ]

    return apply_truth_colors_to_event_display, plot_true_vs_pred_oc


@app.cell
def _(mo, np, oc_layout, test_ds, val_cluster_coords, val_data):
    cluster_dim = int(oc_layout.cluster_dim)
    dim_options = {f"c{i + 1}": i for i in range(cluster_dim)}
    valid_mask = np.asarray(
        val_data.get("mask", np.ones(val_cluster_coords.shape[:2], dtype=bool)), dtype=bool
    )
    valid_coords = np.asarray(val_cluster_coords, dtype=np.float64)[valid_mask]
    valid_truth_ids = np.asarray(val_data["hit_object_id"])

    raw_variances = (
        valid_coords.var(axis=0)
        if len(valid_coords) > 0
        else np.zeros(cluster_dim, dtype=np.float64)
    )

    if len(valid_coords) > 0:
        pca_mean = valid_coords.mean(axis=0)
        centered = valid_coords - pca_mean
        if len(valid_coords) > 1:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
            explained_variance = (singular_values**2) / (len(valid_coords) - 1)
            components = vh
        else:
            explained_variance = np.zeros(cluster_dim, dtype=np.float64)
            components = np.eye(cluster_dim, dtype=np.float64)
    else:
        pca_mean = np.zeros(cluster_dim, dtype=np.float64)
        explained_variance = np.zeros(cluster_dim, dtype=np.float64)
        components = np.eye(cluster_dim, dtype=np.float64)

    explained_variance_ratio = (
        explained_variance / explained_variance.sum()
        if explained_variance.sum() > 0
        else np.zeros_like(explained_variance)
    )
    positive_evals = explained_variance[explained_variance > 0]
    if len(positive_evals) > 0:
        eval_probs = positive_evals / positive_evals.sum()
        effective_rank = float(np.exp(-(eval_probs * np.log(eval_probs)).sum()))
        participation_rank = float((positive_evals.sum() ** 2) / np.square(positive_evals).sum())
    else:
        effective_rank = 0.0
        participation_rank = 0.0

    between = np.zeros(cluster_dim, dtype=np.float64)
    within = np.zeros(cluster_dim, dtype=np.float64)
    for event_coords, event_truth_ids, event_mask in zip(
        np.asarray(val_cluster_coords, dtype=np.float64), valid_truth_ids, valid_mask, strict=True
    ):
        event_signal_mask = event_mask & (event_truth_ids > 0)
        if event_signal_mask.sum() < 2:
            continue
        event_values = event_coords[event_signal_mask]
        event_groups = event_truth_ids[event_signal_mask]
        unique_groups = np.unique(event_groups)
        if len(unique_groups) < 2:
            continue
        global_mean = event_values.mean(axis=0)
        for group in unique_groups:
            group_values = event_values[event_groups == group]
            if len(group_values) == 0:
                continue
            group_mean = group_values.mean(axis=0)
            between += len(group_values) * np.square(group_mean - global_mean)
            within += np.square(group_values - group_mean).sum(axis=0)
    separation_scores = np.divide(
        between,
        np.maximum(within, 1e-12),
        out=np.zeros_like(between),
        where=within > 0,
    )

    def _top_dim_table(values, title, precision=4, top_n=8):
        order = np.argsort(values)[::-1][: min(top_n, len(values))]
        rows = "\n".join(
            f"| c{int(index) + 1} | {values[index]:.{precision}f} |" for index in order
        )
        return f"**{title}**\n\n| Dim | Value |\n| --- | ---: |\n{rows}"

    dims_md = ", ".join(f"`c{i + 1}`" for i in range(cluster_dim))
    pca_md = ", ".join(
        f"`PC{i + 1}={explained_variance_ratio[i]:.1%}`"
        for i in range(min(3, len(explained_variance_ratio)))
    )
    diagnostics_md = [
        mo.md(f"**Available raw OC dims:** {dims_md}"),
        mo.md(
            f"**Raw dim:** `{cluster_dim}`  |  **Effective rank:** `{effective_rank:.2f}`  |  **Participation rank:** `{participation_rank:.2f}`"
        ),
        mo.md(f"**Validation PCA explained variance:** {pca_md}"),
    ]

    event_idx_selector = mo.ui.number(label="Event Index", start=0, stop=len(test_ds) - 1, step=1)
    projection_mode_selector = mo.ui.dropdown(
        options=["Raw dims", "Validation PCA"],
        value="Raw dims",
        label="Display Projection",
    )
    x_dim_selector = mo.ui.dropdown(options=dim_options, value="c1", label="X dim")
    y_dim_selector = mo.ui.dropdown(
        options=dim_options,
        value=f"c{min(2, cluster_dim)}",
        label="Y dim",
    )
    z_dim_selector = mo.ui.dropdown(
        options=dim_options,
        value=f"c{min(3, cluster_dim)}",
        label="Z dim",
    )
    oc_display_projection = {
        "mean": pca_mean,
        "components": components,
        "explained_variance_ratio": explained_variance_ratio,
        "effective_rank": effective_rank,
        "participation_rank": participation_rank,
        "raw_variances": raw_variances,
        "separation_scores": separation_scores,
    }

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(diagnostics_md),
                    mo.md(_top_dim_table(raw_variances, "Top raw variances")),
                    mo.md(_top_dim_table(separation_scores, "Top truth-separation scores")),
                ],
                widths=[0.5, 0.25, 0.25],
            ),
            mo.hstack(
                [
                    event_idx_selector,
                    projection_mode_selector,
                    x_dim_selector,
                    y_dim_selector,
                    z_dim_selector,
                ]
            ),
        ]
    )
    return (
        event_idx_selector,
        oc_display_projection,
        projection_mode_selector,
        x_dim_selector,
        y_dim_selector,
        z_dim_selector,
    )


@app.cell(hide_code=True)
def _(
    best_oc_thresholds,
    event_idx_selector,
    mo,
    mplhep,
    oc_display_projection,
    oc_eval,
    plot_true_vs_pred_oc,
    plt,
    projection_mode_selector,
    test_data,
    test_ds,
    test_preds,
    x_dim_selector,
    y_dim_selector,
    z_dim_selector,
):
    mo.stop(
        best_oc_thresholds is None,
        mo.md("Run OC threshold calibration above before using the single-event OC analysis."),
    )
    plt.style.use("default")
    _tbeta, _td = best_oc_thresholds["tbeta"], best_oc_thresholds["td"]
    _fig = plot_true_vs_pred_oc(
        tbeta=_tbeta,
        td=_td,
        event=test_ds[event_idx_selector.value],
        preds=test_preds,
        hit_object_id=test_data["hit_object_id"],
        projection_mode=projection_mode_selector.value,
        raw_dims=(x_dim_selector.value, y_dim_selector.value, z_dim_selector.value),
        global_pca=oc_display_projection,
        mask=test_data["mask"],
        oc_eval=oc_eval,
        event_idx=event_idx_selector.value,
    )
    mplhep.style.use("CMS")
    mo.ui.plotly(_fig, config={"responsive": True})
    return


@app.cell
def _(apply_truth_colors_to_event_display, event_idx_selector, mo, test_ds):
    from fastgnn.data.cmssw.plotting import plot_event

    _views = ("xy", "yz", "3d")
    _fig1, _desc = plot_event(
        event=test_ds[event_idx_selector.value],
        color_by="hit_object_id",
        views=_views,
        show_cluster_markers=False,
        energy_threshold=test_ds.metadata["preprocessing"]["truth_min_object_energy"],
    )
    _fig1 = apply_truth_colors_to_event_display(
        _fig1,
        test_ds[event_idx_selector.value],
        views=_views,
    )
    mo.vstack(
        [
            mo.md(
                f"**Event {test_ds[event_idx_selector.value].event_id}** | "
                f"{test_ds[event_idx_selector.value].n_hits} hits | "
                f"{test_ds[event_idx_selector.value].n_objects} SimClusters"
            ),
            mo.ui.plotly(_fig1),
            mo.md(_desc),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
