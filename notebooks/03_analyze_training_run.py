# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    import os
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


@app.cell(hide_code=True)
def _(history, mo):
    _has_payload_breakdown = (
        history is not None
        and len(history.get("train_components", [])) > 0
        and any(str(key).startswith("L_payload_") for key in history["train_components"][0])
    )
    show_payload_breakdown = mo.ui.checkbox(
        label="Show payload loss breakdown by quantity", value=False
    )
    show_payload_breakdown if _has_payload_breakdown else mo.md("")
    return (show_payload_breakdown,)


@app.cell(hide_code=True)
def _(
    COLORS,
    PLOTTING_CONFIG,
    config,
    history,
    mo,
    mplhep,
    np,
    pl,
    plt,
    show_payload_breakdown,
):
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
    if "L_payload" in train_components_df.columns:
        loss_labels["L_payload"] = r"$\mathcal{L}_{payload}$"
    payload_component_cols = [
        _col for _col in train_components_df.columns if _col.startswith("L_payload_")
    ]

    for _i, (_df_type, _df) in enumerate(
        zip(["train", "val"], [train_components_df, val_components_df]), start=1
    ):
        plt.sca(_axs[_i])
        _colors = {"V": "#228833", "beta": "#CCBB44", "payload": "#AA3377"}
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
        if show_payload_breakdown.value and payload_component_cols:
            _payload_styles = ["dashed", "dotted", "dashdot", (0, (3, 1, 1, 1))]
            for _j, _payload_col in enumerate(payload_component_cols):
                _name = _payload_col.replace("L_payload_", "")
                plt.plot(
                    _df[_payload_col],
                    label=r"$\mathcal{L}_{payload}^{\mathrm{" + _name + r"}}$",
                    color=_colors["payload"],
                    linestyle=_payload_styles[_j % len(_payload_styles)],
                    alpha=0.85,
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

    def _oc_layout_from_model_config(model_cfg):
        try:
            return OCOutputLayout.from_config(model_cfg)
        except ValueError as exc:
            if "output_layout.regressions is no longer supported" not in str(exc):
                raise

        layout_cfg = dict(model_cfg.get("output_layout") or {})
        regressions = list(layout_cfg.pop("regressions", []) or [])
        cluster_cfg = dict(layout_cfg.get("cluster_space") or {})
        cluster_start = int(cluster_cfg.get("start", 1))
        cluster_dim = int(cluster_cfg["dim"])
        payload_start = cluster_start + cluster_dim
        payload_dim = int(model_cfg.get("payload_output_dim") or 0)

        if regressions:
            regression_starts = [int(reg["start"]) for reg in regressions]
            regression_stops = [int(reg["start"]) + int(reg["dim"]) for reg in regressions]
            payload_start = min(regression_starts)
            payload_dim = max(regression_stops) - payload_start
        elif payload_dim == 0:
            payload_dim = int(model_cfg["output_dim"]) - 1 - cluster_dim

        if payload_dim > 0:
            layout_cfg["payload"] = {"start": payload_start, "dim": payload_dim}

        compat_model_cfg = dict(model_cfg)
        compat_model_cfg["output_layout"] = layout_cfg
        return OCOutputLayout.from_config(compat_model_cfg)

    oc_layout = _oc_layout_from_model_config(config["model"])
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
        test_output_slices,
        test_preds,
    )


@app.cell(hide_code=True)
def _(config, mo, oc_layout):
    payload_quantities = list(
        ((config.get("training") or {}).get("payload") or {}).get("quantities") or []
    )
    has_payload_model = oc_layout.payload_dim > 0 and len(payload_quantities) > 0
    if has_payload_model:
        _payload_lines = [
            f"- `{quantity.get('name', quantity.get('field'))}`: "
            f"field `{quantity.get('field')}`, "
            f"transform `{quantity.get('transform', 'identity')}`"
            for quantity in payload_quantities
        ]
        _message = (
            f"Model payload output dim: `{oc_layout.payload_dim}`. "
            "Configured payloads:\n" + "\n".join(_payload_lines)
        )
    elif oc_layout.payload_dim > 0:
        _message = (
            f"Model payload output dim: `{oc_layout.payload_dim}`, "
            "but `training.payload.quantities` is not configured."
        )
    else:
        _message = "Model has no configured payload output."
    mo.md(_message)
    return has_payload_model, payload_quantities


@app.cell(hide_code=True)
def _(has_payload_model, mo):
    _source_options = ["aggr.", "payl."] if has_payload_model else ["aggr."]
    regression_primary_selector = mo.ui.dropdown(
        options=_source_options,
        label="Primary regression source",
        value="aggr.",
    )
    regression_secondary_selector = mo.ui.dropdown(
        options=["None", *_source_options],
        label="Secondary regression source",
        value="payl." if has_payload_model else "None",
    )
    mo.hstack([regression_primary_selector, regression_secondary_selector])
    return regression_primary_selector, regression_secondary_selector


@app.cell(hide_code=True)
def _(regression_primary_selector, regression_secondary_selector):
    primary_regression_source_key = regression_primary_selector.value
    selected_regression_source_keys = [primary_regression_source_key]
    if (
        regression_secondary_selector.value != "None"
        and regression_secondary_selector.value != primary_regression_source_key
    ):
        selected_regression_source_keys.append(regression_secondary_selector.value)
    return primary_regression_source_key, selected_regression_source_keys


@app.cell(hide_code=True)
def _(config, np, payload_quantities, test_data, test_ds):
    # Payload *corrections* are decoded from each hit's own raw (unnormalized) seed
    # feature (e.g. ``et``, ``eta``, ``phi``), so the eval payload must carry those
    # columns in addition to the physical x/y/z/energy used for matching and reco.
    eval_feature_names = ["x", "y", "z", "energy"]
    for _quantity in payload_quantities:
        _seed = _quantity.get("seed")
        if _seed is not None and str(_seed) not in eval_feature_names:
            eval_feature_names.append(str(_seed))
    test_eval_data = test_ds.as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=eval_feature_names,
        truncate=config["training"].get("truncate", "first"),
        normalize_features=False,
        seed=config.get("seed", 0),
    )
    np.testing.assert_array_equal(test_eval_data["mask"], test_data["mask"])
    np.testing.assert_array_equal(test_eval_data["hit_object_id"], test_data["hit_object_id"])
    return eval_feature_names, test_eval_data


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
    from fastgnn.evaluation.schema import match_response_columns
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
        match_response_columns,
    )


@app.cell(hide_code=True)
def _(has_payload_model, mo):
    threshold_objective_selector = mo.ui.dropdown(
        options=["matched_f1", "count_median", "count_mean"],
        label="Threshold Objective",
        value="count_mean",
    )
    matching_algorithm_selector = mo.ui.dropdown(
        options=["hungarian", "greedy"],
        label="Matching Algorithm",
        value="hungarian",
    )
    matching_distance_threshold_selector = mo.ui.number(
        label="Distance Threshold in cm", value=50, step=0.5
    )
    payload_matching_distance_threshold_selector = mo.ui.number(
        label="Payload matching distance threshold", value=0.3, step=0.05
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
    _matching_controls = [
        matching_algorithm_selector,
        matching_distance_threshold_selector,
    ]
    if has_payload_model:
        _matching_controls.append(payload_matching_distance_threshold_selector)
    _matching_controls.extend(
        [
            matching_energy_ratio_min_selector,
            matching_energy_ratio_max_selector,
            hungarian_energy_ratio_log_weight_selector,
        ]
    )
    mo.vstack(
        [
            mo.md(
                "**Select the threshold objective and matching parameters used for calibration.**"
            ),
            threshold_objective_selector,
            mo.hstack(_matching_controls),
            start_calibration_button,
        ]
    )
    return (
        hungarian_energy_ratio_log_weight_selector,
        matching_algorithm_selector,
        matching_distance_threshold_selector,
        matching_energy_ratio_max_selector,
        matching_energy_ratio_min_selector,
        payload_matching_distance_threshold_selector,
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
    _td_values = np.linspace(0.05, 2.0, 50)

    _REFINE_N = 50
    _REFINE_LOW = 0.75
    _REFINE_HIGH = 1.25

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
        "refinement": {
            "n_values": _REFINE_N,
            "low_factor": _REFINE_LOW,
            "high_factor": _REFINE_HIGH,
        },
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

    def _run_grid_search(tbeta_values, td_values, title, subtitle):
        return grid_search_thresholds(
            beta=val_beta,
            cluster_coords=val_cluster_coords,
            hit_object_id=val_data["hit_object_id"],
            mask=val_data.get("mask", None),
            tbeta_values=tbeta_values,
            td_values=td_values,
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
                title=title,
                subtitle=subtitle,
                completion_title="OC threshold calibration complete",
            ),
        )

    if _cached_record is None:
        # Pass 1: coarse scan
        _coarse_best, _ = _run_grid_search(
            _tbeta_values,
            _td_values,
            title=_progress_title,
            subtitle=_progress_subtitle,
        )

        # Pass 2: refined scan — 50 values in [0.75x, 1.25x] around each coarse best
        _best_tbeta = _coarse_best["tbeta"]
        _best_td = _coarse_best["td"]

        _fine_tbeta_values = np.linspace(
            max(_REFINE_LOW * _best_tbeta, 1e-4),
            min(_REFINE_HIGH * _best_tbeta, 1.0),
            _REFINE_N,
        )
        _fine_td_values = np.linspace(
            max(_REFINE_LOW * _best_td, 1e-4),
            _REFINE_HIGH * _best_td,
            _REFINE_N,
        )
        _fine_subtitle = (
            f"{_REFINE_N} beta thresholds x {_REFINE_N} distance thresholds "
            f"(refined around tbeta={_best_tbeta:.4f}, td={_best_td:.4f})"
        )

        best_oc_thresholds, _grid_results = _run_grid_search(
            _fine_tbeta_values,
            _fine_td_values,
            title="Refining OC thresholds",
            subtitle=_fine_subtitle,
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
        _status = "Computed threshold calibration (coarse + refined) and wrote cache."
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
    eval_feature_names,
    evaluate_oc_padded,
    has_payload_model,
    hungarian_energy_ratio_log_weight_selector,
    match_response_columns,
    matching_algorithm_selector,
    matching_distance_threshold_selector,
    matching_energy_ratio_max_selector,
    matching_energy_ratio_min_selector,
    mo,
    oc_layout,
    payload_matching_distance_threshold_selector,
    payload_quantities,
    pl,
    primary_regression_source_key,
    selected_regression_source_keys,
    test_data,
    test_ds,
    test_eval_data,
    test_preds,
):
    _tbeta, _td = best_oc_thresholds["tbeta"], best_oc_thresholds["td"]
    _events = [test_ds[i] for i in range(len(test_ds))]

    def _evaluate_with_reference(reference, max_match_distance):
        return evaluate_oc_padded(
            preds=test_preds,
            hit_object_id=test_data["hit_object_id"],
            mask=test_data["mask"],
            features=test_eval_data["features"],
            feature_names=eval_feature_names,
            tbeta=_tbeta,
            td=_td,
            layout=oc_layout,
            payload_quantities=payload_quantities,
            events=_events,
            max_match_distance=max_match_distance,
            min_energy_ratio=matching_energy_ratio_min_selector.value,
            max_energy_ratio=matching_energy_ratio_max_selector.value,
            matching_algorithm=matching_algorithm_selector.value,
            hungarian_energy_ratio_log_weight=hungarian_energy_ratio_log_weight_selector.value,
            matching_reference=reference,
        )

    oc_evals_by_matching = {
        "aggr.": _evaluate_with_reference(
            "aggregated",
            matching_distance_threshold_selector.value,
        )
    }
    if has_payload_model:
        oc_evals_by_matching["payl."] = _evaluate_with_reference(
            "payload",
            payload_matching_distance_threshold_selector.value,
        )

    oc_eval = oc_evals_by_matching[primary_regression_source_key]
    regression_sources = []
    for _source_key in selected_regression_source_keys:
        if _source_key not in oc_evals_by_matching:
            continue
        _source_eval = oc_evals_by_matching[_source_key]
        _is_payload = _source_key == "payl."
        if _is_payload and "payload_et_pred" not in _source_eval.predicted.columns:
            continue
        regression_sources.append(
            {
                "key": _source_key,
                "label": _source_key,
                "eval": _source_eval,
                "matches": _source_eval.matches,
                "predicted": _source_eval.predicted,
                **match_response_columns(payload=_is_payload),
            }
        )
    _source_colors = (
        ["black"]
        if len(regression_sources) == 1
        else ["black", "#4477AA"] + ["#66CCEE"] * max(0, len(regression_sources) - 2)
    )
    for _source, _color in zip(regression_sources, _source_colors):
        _source["color"] = _color
    count_summary = oc_eval.count_summary()
    seed_summary = oc_eval.seed_summary()
    matching_summary = oc_eval.matching_summary()
    _comparison_rows = []
    for _reference, _eval in oc_evals_by_matching.items():
        _summary = {
            "matching_reference": _reference,
            **_eval.count_summary(),
            **_eval.seed_summary(),
            **_eval.matching_summary(),
        }
        _comparison_rows.append(_summary)
    print(count_summary, "\n\n", seed_summary, "\n\n", matching_summary)
    mo.vstack(
        [
            mo.md(f"Primary source for single-source views: `{primary_regression_source_key}`."),
            mo.ui.table(pl.DataFrame(_comparison_rows)),
        ]
    )
    return oc_eval, regression_sources


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    from fastgnn.plotting import plot_binned_efficiency, plot_binned_fake_rate

    return plot_binned_efficiency, plot_binned_fake_rate


@app.cell(hide_code=True)
def _(
    PLOTTING_CONFIG,
    binned_efficiency,
    binned_fake_rate,
    log_edges,
    mo,
    np,
    plot_binned_efficiency,
    plot_binned_fake_rate,
    plt,
    regression_sources,
):
    mo.stop(len(regression_sources) == 0, mo.md("Select at least one regression source."))

    _truth_et_values = np.concatenate(
        [
            _source["eval"].truth["truth_et"].drop_nulls().to_numpy()
            for _source in regression_sources
        ]
    )
    _truth_et_values = _truth_et_values[np.isfinite(_truth_et_values) & (_truth_et_values > 0)]
    _pred_et_values = np.concatenate(
        [
            _source["predicted"][_source["pred_et_col"]].drop_nulls().to_numpy()
            for _source in regression_sources
        ]
    )
    _pred_et_values = _pred_et_values[np.isfinite(_pred_et_values) & (_pred_et_values > 0)]

    mo.stop(
        _truth_et_values.size == 0 or _pred_et_values.size == 0,
        mo.md(
            "No truth/predicted clusters at the selected thresholds (the model may have collapsed)."
        ),
    )
    _bins_truth = log_edges(_truth_et_values)
    _bins_pred = log_edges(_pred_et_values)

    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"],
    )

    for _source in regression_sources:
        _truth_et_table = binned_efficiency(
            _source["eval"].truth,
            "truth_et",
            _bins_truth,
        )
        _pred_et_table = binned_fake_rate(
            _source["predicted"],
            _source["pred_et_col"],
            _bins_pred,
        )
        plot_binned_efficiency(
            _axs[0],
            _truth_et_table,
            label=_source["label"],
            color=_source["color"],
        )
        plot_binned_fake_rate(
            _axs[1],
            _pred_et_table,
            label=_source["label"],
            color=_source["color"],
        )

    _axs[0].set_xscale("log")
    _axs[0].set_xlabel(r"Cluster truth $E_T$ [GeV]")
    _axs[0].set_ylabel("Efficiency")
    _axs[0].set_ylim(0.0, 1.0)
    _axs[0].legend(loc="lower right")

    _axs[1].set_xscale("log")
    _axs[1].set_xlabel(r"Cluster predicted $E_T$ [GeV]")
    _axs[1].set_ylabel("Fake Rate")
    _axs[1].set_ylim(-0.02, 1.0)
    _axs[1].legend(loc="upper left")

    _fig1.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cluster Size and $\beta$
    """)
    return


@app.cell(hide_code=True)
def _(
    PLOTTING_CONFIG,
    binned_efficiency,
    binned_fake_rate,
    log_edges,
    mo,
    np,
    plot_binned_efficiency,
    plot_binned_fake_rate,
    plt,
    regression_sources,
):
    mo.stop(len(regression_sources) == 0, mo.md("Select at least one regression source."))
    _fig1, _axs = plt.subplots(1, 3, figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"])

    _panels = [
        (_axs[0], "n_hits_truth", "True cluster size", "efficiency", "truth"),
        (_axs[1], "n_hits_pred", "Predicted cluster size", "fake_rate", "predicted"),
        (_axs[2], "beta_seed", r"$\beta$ score of OC seed", "fake_rate", "predicted"),
    ]
    for _ax, _col, _xlabel, _metric, _frame_name in _panels:
        _bin_values = np.concatenate(
            [
                _source["eval"].truth[_col].drop_nulls().to_numpy()
                if _frame_name == "truth"
                else _source["predicted"][_col].drop_nulls().to_numpy()
                for _source in regression_sources
            ]
        )
        _bin_values = _bin_values[np.isfinite(_bin_values) & (_bin_values > 0)]
        if _bin_values.size == 0:
            _ax.text(0.5, 0.5, "No entries", transform=_ax.transAxes, ha="center", va="center")
            _ax.set_xlabel(_xlabel)
            continue
        _bins = log_edges(_bin_values)
        for _source in regression_sources:
            _frame = _source["eval"].truth if _frame_name == "truth" else _source["predicted"]
            if _metric == "efficiency":
                _table = binned_efficiency(_frame, _col, _bins)
                plot_binned_efficiency(
                    _ax,
                    _table,
                    label=_source["label"],
                    color=_source["color"],
                )
            else:
                _table = binned_fake_rate(_frame, _col, _bins)
                plot_binned_fake_rate(
                    _ax,
                    _table,
                    label=_source["label"],
                    color=_source["color"],
                )
        _ax.set_xscale("log")
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel("Efficiency" if _metric == "efficiency" else "Fake Rate")
        _ax.set_ylim(-0.02, 1.0)
        _ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        _ax.legend(loc="lower right" if _metric == "efficiency" else "upper left")
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


@app.cell(hide_code=True)
def _():
    # Binned-profile statistics live in fastgnn.evaluation.binned; the matplotlib
    # profile/response helpers live in fastgnn.plotting.
    from fastgnn.evaluation.binned import binned_profile, fwhm_and_mu, log_edges
    from fastgnn.plotting import (
        REFERENCE_LINE_KWARGS,
        energy_weighted_str,
        plot_profile_points,
    )

    return (
        REFERENCE_LINE_KWARGS,
        binned_profile,
        energy_weighted_str,
        fwhm_and_mu,
        log_edges,
        plot_profile_points,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Energy and Transverse Energy Response
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    show_median_response = mo.ui.checkbox(
        label="Show median response in profile plots", value=False
    )
    show_relative_residual = mo.ui.checkbox(
        label="Show relative residual instead of ratio", value=False
    )
    mo.vstack(
        [
            show_median_response,
            show_relative_residual,
        ]
    )
    return show_median_response, show_relative_residual


@app.cell(hide_code=True)
def _(
    PLOTTING_CONFIG,
    REFERENCE_LINE_KWARGS,
    binned_profile,
    energy_weighted_str,
    log_edges,
    mo,
    np,
    plot_profile_points,
    plt,
    regression_sources,
    show_median_response,
    show_relative_residual,
):
    mo.stop(len(regression_sources) == 0, mo.md("Select at least one regression source."))

    _value_col = "residual_col" if show_relative_residual.value else "response_col"
    _ylabel = (
        r"$(E_{T,\mathrm{pred}} - E_{T,\mathrm{true}}) / E_{T,\mathrm{true}}$"
        if show_relative_residual.value
        else r"$E_{T,\mathrm{pred}}/E_{T,\mathrm{true}}$"
    )
    _reference_value = 0.0 if show_relative_residual.value else 1.0

    ## Response histogram for transverse energy
    _fig1, _ax = plt.subplots(
        nrows=1, ncols=1, figsize=PLOTTING_CONFIG["figsize"]["A4"]["halfwidth"]
    )
    for _i, _source in enumerate(regression_sources):
        _matches = _source["matches"]
        _values = _matches[_source[_value_col]].drop_nulls().to_numpy()
        _values = _values[np.isfinite(_values)]
        if len(_values) == 0:
            continue
        if show_relative_residual.value:
            _lim = max(0.5, float(np.quantile(np.abs(_values), 0.98)))
            _bins = np.linspace(-_lim, _lim, 50)
        else:
            _upper_lim = 15
            _bins = np.linspace(0, _upper_lim, 50)
        _ax.hist(
            _values,
            bins=_bins,
            color=_source["color"],
            histtype="step",
        )
        _lower_lim = float(_bins[0])
        _upper_lim = float(_bins[-1])
        _overflow = ((_values < _lower_lim) | (_values > _upper_lim)).sum() / len(_values)
        _txt = f"{_source['label']}\n$\\mu$={_values.mean():.2f}\noverflow: {_overflow:.1%}"
        _ax.text(
            0.05,
            0.90 - 0.16 * _i,
            _txt,
            transform=_ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color=_source["color"],
        )
    _ax.axvline(_reference_value, **REFERENCE_LINE_KWARGS)
    _ax.set_xlabel(_ylabel)
    _ax.set_ylabel("Matched clusters")
    _ax.set_ylim(1e-3, None)
    _ax.grid(alpha=0.25)
    plt.tight_layout()

    ## Response vs truth for transverse energy and eta
    _fig2, _axs = plt.subplots(
        nrows=1, ncols=2, figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_2pane"]
    )
    _aggs = ["mean", "median"] if show_median_response.value else ["mean"]

    for _source in regression_sources:
        _matches = _source["matches"]
        _col = _source[_value_col]
        _truth_et, _values_vs_et = _matches.select(["truth_et", _col]).drop_nulls().to_numpy().T
        _truth_eta, _values_vs_eta = (
            _matches.select(["truth_centroid_eta", _col]).drop_nulls().to_numpy().T
        )
        if len(_truth_et) == 0:
            continue
        _axs[0].scatter(
            _truth_et,
            _values_vs_et,
            alpha=0.18,
            s=10,
            color=_source["color"],
        )
        for _agg in _aggs:
            plot_profile_points(
                _axs[0],
                binned_profile(_truth_et, _values_vs_et, log_edges(_truth_et)),
                _agg,
                label=_source["label"] if _agg == "mean" else None,
                color=_source["color"] if _agg == "mean" else "purple",
            )

        _abs_eta = np.abs(_truth_eta)
        _axs[1].scatter(
            _abs_eta,
            _values_vs_eta,
            alpha=0.18,
            s=10,
            color=_source["color"],
        )
        _eta_bins = np.linspace(np.quantile(_abs_eta, 0.01), np.quantile(_abs_eta, 0.99), 12)
        for _agg in _aggs:
            plot_profile_points(
                _axs[1],
                binned_profile(_abs_eta, _values_vs_eta, _eta_bins),
                _agg,
                label=_source["label"] if _agg == "mean" else None,
                color=_source["color"] if _agg == "mean" else "purple",
            )
    _axs[0].set_xscale("log")
    _axs[0].set_xlabel(r"$E_{T,\mathrm{true}}$ [GeV]")
    _axs[0].set_ylabel(_ylabel)
    _axs[1].set_xlabel(energy_weighted_str(r"$\eta_\mathrm{{true}}$"))

    for _ax in _axs:
        _ax.grid(alpha=0.25)
        if not show_relative_residual.value:
            _ax.set_yscale("log")
        _ax.axhline(_reference_value, **REFERENCE_LINE_KWARGS)
        _ax.legend()
    plt.tight_layout()

    ## Predicted vs true transverse energy
    _fig3, _ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["halfwidth"],
    )

    for _source in regression_sources:
        _matches = _source["matches"]
        _truth, _pred = (
            _matches.select("truth_et", _source["pred_et_col"]).drop_nulls().to_numpy().T
        )
        if len(_truth) == 0:
            continue
        _ax.scatter(_truth, _pred, alpha=0.25, s=10, color=_source["color"])
        plot_profile_points(
            _ax,
            binned_profile(_truth, _pred, log_edges(_truth)),
            "mean",
            color=_source["color"],
            label=_source["label"],
        )
    _truth_all = np.concatenate(
        [_source["matches"]["truth_et"].drop_nulls().to_numpy() for _source in regression_sources]
    )
    _pred_cols = [
        _source["matches"][_source["pred_et_col"]].drop_nulls().to_numpy()
        for _source in regression_sources
        if _source["pred_et_col"] in _source["matches"].columns
    ]
    _pred_all = np.concatenate(_pred_cols) if _pred_cols else _truth_all
    if _truth_all.size and _pred_all.size:
        _low = min(float(_truth_all.min()), float(_pred_all.min()))
        _high = max(float(_truth_all.max()), float(_pred_all.max()))
        _ax.plot([_low, _high], [_low, _high], **REFERENCE_LINE_KWARGS)
    _ax.set_xscale("log")
    _ax.set_yscale("log")
    _ax.set(
        xlabel=r"$E_{T,\mathrm{true}}$ [GeV]",
        ylabel=r"$E_{T,\mathrm{pred}}$ [GeV]",
    )
    _ax.grid(alpha=0.25)
    _ax.legend()

    plt.tight_layout()
    mo.vstack([_fig1, _fig2, _fig3])
    return


@app.cell(hide_code=True)
def _(
    has_payload_model,
    mo,
    np,
    payload_quantities,
    pl,
    test_data,
    test_output_slices,
):
    _rows = []
    if has_payload_model and test_output_slices.payload is not None:
        _cursor = 0
        _valid = test_data["mask"].astype(bool)
        for _quantity in payload_quantities:
            _transform = str(_quantity.get("transform", "identity"))
            _name = str(_quantity.get("name", _quantity.get("field")))
            if _transform == "sin_cos":
                _sin = test_output_slices.payload[..., _cursor][_valid]
                _cos = test_output_slices.payload[..., _cursor + 1][_valid]
                _norm = np.hypot(_sin, _cos)
                _rows.append(
                    {
                        "payload": _name,
                        "mean_norm": float(np.mean(_norm)),
                        "median_norm": float(np.median(_norm)),
                        "sigma68_norm": float(
                            0.5 * (np.quantile(_norm, 0.84) - np.quantile(_norm, 0.16))
                        ),
                        "min_norm": float(np.min(_norm)),
                        "max_norm": float(np.max(_norm)),
                    }
                )
                _cursor += 2
            else:
                _cursor += 1
    _out = (
        mo.vstack(
            [
                mo.md("#### Payload sin/cos calibration cross-check"),
                mo.ui.table(pl.DataFrame(_rows)),
            ]
        )
        if _rows
        else mo.md(
            "#### Payload sin/cos calibration cross-check\n\n"
            "No payload `sin_cos` quantities configured for this run."
        )
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Position Resolution
    """)
    return


@app.cell(hide_code=True)
def _(
    PLOTTING_CONFIG,
    energy_weighted_str,
    fwhm_and_mu,
    mo,
    plt,
    regression_sources,
):
    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"],
    )
    mo.stop(len(regression_sources) == 0, mo.md("Select at least one regression source."))

    for _ax, _axis, _label in [
        (
            _axs[0],
            "eta",
            energy_weighted_str(r"$\eta_\mathrm{true}$")
            + " - "
            + energy_weighted_str(r"$\eta_\mathrm{pred}$"),
        ),
        (
            _axs[1],
            "phi",
            energy_weighted_str(r"$\phi_\mathrm{true}$")
            + " - "
            + energy_weighted_str(r"$\phi_\mathrm{pred}$"),
        ),
        (
            _axs[2],
            "z",
            energy_weighted_str(r"$z_\mathrm{true}$")
            + " - "
            + energy_weighted_str(r"$z_\mathrm{pred}$")
            + " [cm]",
        ),
    ]:
        _stats_lines = []
        for _source in regression_sources:
            _values = _source["matches"][_source["position_cols"][_axis]].drop_nulls().to_numpy()
            if len(_values):
                _ax.hist(
                    _values,
                    bins=50,
                    color=_source["color"],
                    histtype="step",
                )
                _fwhm, _mu = fwhm_and_mu(_values)
                _stats_lines.append(
                    (
                        _source["color"],
                        f"{_source['label']}\nFWHM={_fwhm:.3g}\n$\\mu$={_mu:.3g}"
                        if _fwhm is not None and _mu is not None
                        else f"{_source['label']}: no entries",
                    )
                )
        if _stats_lines:
            _ax.axvline(0.0, color="grey", linestyle="--")
            for _i, (_color, _text) in enumerate(_stats_lines):
                _ax.text(
                    0.97,
                    0.92 - 0.16 * _i,
                    _text,
                    transform=_ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize="small",
                    color=_color,
                )
        else:
            _ax.text(0.5, 0.5, "No entries", transform=_ax.transAxes, ha="center", va="center")
        _ax.set_xlabel(_label)
        _ax.grid(alpha=0.25)
    _axs[0].set_ylabel("Matched clusters")

    plt.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(
    PLOTTING_CONFIG,
    REFERENCE_LINE_KWARGS,
    binned_profile,
    energy_weighted_str,
    log_edges,
    mo,
    plot_profile_points,
    plt,
    regression_sources,
):
    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"],
    )
    mo.stop(len(regression_sources) == 0, mo.md("Select at least one regression source."))

    for _ax, _axis, _ylabel in [
        (_axs[0], "eta", energy_weighted_str(r"$\Delta_\mathrm{rel}\eta$")),
        (_axs[1], "phi", energy_weighted_str(r"$\Delta_\mathrm{rel}\phi$")),
        (_axs[2], "z", energy_weighted_str(r"$\Delta_\mathrm{rel}z$")),
    ]:
        _had_values = False
        for _source in regression_sources:
            _truth_et, _values = (
                _source["matches"]
                .select("truth_et", _source["relative_position_cols"][_axis])
                .drop_nulls()
                .to_numpy()
                .T
            )
            if len(_truth_et):
                _had_values = True
                _table = binned_profile(_truth_et, _values, log_edges(_truth_et))
                plot_profile_points(
                    _ax,
                    _table,
                    "mean",
                    color=_source["color"],
                    label=_source["label"],
                )
        if _had_values:
            _ax.axhline(0.0, **REFERENCE_LINE_KWARGS)
            _ax.set_xscale("log")
            _ax.legend()
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _():
    # Per-event accounting tables (energy balance, per-truth match summary, fake
    # clusters) are computed by fastgnn.evaluation.event_summary; this cell only
    # assembles the Plotly figure from them.
    from fastgnn.evaluation.event_summary import (
        event_energy_accounting,
        event_fake_cluster_rows,
        event_hit_observables,
        event_seed_truth_matches,
        event_truth_match_summary,
    )

    return (
        event_energy_accounting,
        event_fake_cluster_rows,
        event_hit_observables,
        event_seed_truth_matches,
        event_truth_match_summary,
    )


@app.cell(hide_code=True)
def _(
    count_clusters_from_labels,
    count_truth_objects,
    event_energy_accounting,
    event_fake_cluster_rows,
    event_hit_observables,
    event_seed_truth_matches,
    event_truth_match_summary,
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
        eval_features=None,
        eval_feature_names=None,
        oc_eval=None,
        event_idx=0,
        prediction_source="aggr.",
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
        _hit_observables = event_hit_observables(
            event,
            event_idx,
            m,
            eval_features=eval_features,
            eval_feature_names=eval_feature_names,
        )
        _hit_energy = _hit_observables["energy"]
        _hit_et = _hit_observables["et"]
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
            rows=5,
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
                [
                    {"type": "xy", "colspan": 2},
                    None,
                    {"type": "xy", "colspan": 2},
                    None,
                    {"type": "xy", "colspan": 2},
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
                [
                    {"type": "table", "colspan": 6},
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            ],
            subplot_titles=(
                f"Truth assignment in OC space ({_projection_label})",
                f"Predicted clusters in OC space ({_projection_label})",
                "Event beta distribution",
                "Truth cluster sizes",
                "Predicted OC seed sizes",
                "Pred / truth ET",
                "Truth ET recovered",
                "Pred ET purity",
                "Event ET accounting",
                "Purity",
                "Completeness",
                "Truth object match summary",
            ),
            row_heights=[0.44, 0.16, 0.13, 0.13, 0.14],
            vertical_spacing=0.065,
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
        _summary_order = [int(label) for label in _truth_labels]
        _truth_match_rows = event_truth_match_summary(
            oc_eval,
            event_idx,
            _summary_order,
            hit_object_id=_hit_object_id,
            clustering=_clustering,
            hit_et=_hit_et,
            prediction_source=prediction_source,
        )
        _match_table_rows = _truth_match_rows + event_fake_cluster_rows(
            oc_eval,
            event_idx,
            prediction_source=prediction_source,
        )
        _truth_match_by_object = {int(row["object_id"]): row for row in _truth_match_rows}
        _energy_accounting = event_energy_accounting(
            hit_et=_hit_et,
            hit_object_id=_hit_object_id,
            clustering=_clustering,
            truth_match_rows=_truth_match_rows,
            oc_eval=oc_eval,
            event_idx=event_idx,
            prediction_source=prediction_source,
        )
        _pred_energy_closure = (
            _energy_accounting["matched_pred_energy"]
            + _energy_accounting["fake_pred_energy"]
            + _energy_accounting["unassigned_hit_energy"]
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

        def _metric_values(name):
            return [
                100.0 * float(_truth_match_by_object.get(object_id, {}).get(name) or 0.0)
                for object_id in _summary_order
            ]

        def _ratio_values(name):
            return [
                float(_truth_match_by_object.get(object_id, {}).get(name) or 0.0)
                for object_id in _summary_order
            ]

        def _add_truth_metric_bar(name, title, row, col):
            if len(_summary_order) == 0:
                return
            _fig.add_trace(
                go.Bar(
                    x=[str(object_id) for object_id in _summary_order],
                    y=_metric_values(name),
                    marker_color=[_truth_colors[object_id] for object_id in _summary_order],
                    name=title,
                    showlegend=False,
                    hovertemplate=("truth object=%{x}<br>" + title + "=%{y:.1f}%<extra></extra>"),
                ),
                row=row,
                col=col,
            )

        if len(_summary_order) > 0:
            _fig.add_trace(
                go.Bar(
                    x=[str(object_id) for object_id in _summary_order],
                    y=_ratio_values("pred_over_truth_et"),
                    marker_color=[_truth_colors[object_id] for object_id in _summary_order],
                    name="Pred / truth ET",
                    showlegend=False,
                    hovertemplate=(
                        "truth object=%{x}<br>ET_pred / ET_true=%{y:.3g}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )

        _add_truth_metric_bar("et_recovered_fraction", "Truth ET recovered", 3, 3)
        _add_truth_metric_bar("et_assigned_fraction", "Pred ET purity", 3, 5)
        _add_truth_metric_bar("purity", "Purity", 4, 3)
        _add_truth_metric_bar("completeness", "Completeness", 4, 5)

        _fig.add_trace(
            go.Bar(
                x=[
                    "Valid hits ET",
                    "Signal truth ET",
                    "Noise",
                    "Matched pred ET",
                    "Fake pred ET",
                    "Unassigned",
                ],
                y=[
                    _energy_accounting["valid_hit_energy"],
                    _energy_accounting["signal_hit_energy"],
                    _energy_accounting["noise_hit_energy"],
                    _energy_accounting["matched_pred_energy"],
                    _energy_accounting["fake_pred_energy"],
                    _energy_accounting["unassigned_hit_energy"],
                ],
                marker_color=[
                    "#6b6b6b",
                    "#1f77b4",
                    "rgba(150,150,150,0.75)",
                    "#2ca02c",
                    "#d62728",
                    "#bca875",
                ],
                text=[
                    f"{_energy_accounting['valid_hit_energy']:.3g} GeV",
                    f"{_energy_accounting['signal_hit_energy']:.3g} GeV<br>{_energy_accounting['signal_energy_fraction']:.1f}%",
                    f"{_energy_accounting['noise_hit_energy']:.3g} GeV",
                    f"{_energy_accounting['matched_pred_energy']:.3g} GeV",
                    f"{_energy_accounting['fake_pred_energy']:.3g} GeV",
                    f"{_energy_accounting['unassigned_hit_energy']:.3g} GeV<br>{_energy_accounting['unassigned_energy_fraction']:.1f}%",
                ],
                textposition="auto",
                name="Event ET accounting",
                showlegend=False,
                hovertemplate=(
                    "%{x}<br>energy=%{y:.3g} GeV<br>"
                    f"pred+unassigned closure={_pred_energy_closure:.3g} GeV"
                    "<extra></extra>"
                ),
            ),
            row=4,
            col=1,
        )

        _table_columns = [
            ("object_id", "object / fake cluster"),
            ("matched", "matched"),
            ("truth_energy", "E_true hits [GeV]"),
            ("energy_pred", "E_pred [GeV]"),
            ("truth_et", "ET_true hits [GeV]"),
            ("et_pred", "ET_pred [GeV]"),
            ("recovered_energy", "E_recovered [GeV]"),
            ("recovered_et", "ET_recovered [GeV]"),
            ("n_hits_truth", "n_hits true"),
            ("n_hits_pred", "n_hits pred"),
            ("purity", "hit purity"),
            ("completeness", "hit completeness"),
            ("et_recovered_fraction", "ET recovered"),
            ("et_assigned_fraction", "pred ET purity"),
            ("beta_max", "beta max"),
            ("nearest_truth_dist", "nearest truth dist"),
        ]
        _fig.add_trace(
            go.Table(
                header={
                    "values": [label for _, label in _table_columns],
                    "align": "left",
                    "fill_color": "#f2f2f2",
                    "font": {"size": 11},
                },
                cells={
                    "values": [
                        [format_match_table_value(row.get(key)) for row in _match_table_rows]
                        for key, _ in _table_columns
                    ],
                    "align": "left",
                    "height": 24,
                    "font": {"size": 10},
                },
            ),
            row=5,
            col=1,
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

        for _x, _text in [(0.49, _truth_text), (0.985, _pred_text)]:
            _fig.add_annotation(
                x=_x,
                y=0.92,
                xref="paper",
                yref="paper",
                text=_text,
                showarrow=False,
                xanchor="right",
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
            height=1550,
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
        _fig.update_xaxes(title_text="Truth object id", row=3, col=1)
        _fig.update_yaxes(title_text=r"ET_pred / ET_true", rangemode="tozero", row=3, col=1)
        _fig.update_xaxes(title_text="Truth object id", row=3, col=3)
        _fig.update_yaxes(title_text="ET recovered [%]", range=[0, 105], row=3, col=3)
        _fig.update_xaxes(title_text="Truth object id", row=3, col=5)
        _fig.update_yaxes(title_text="ET purity [%]", range=[0, 105], row=3, col=5)
        _fig.update_xaxes(title_text="", row=4, col=1)
        _fig.update_yaxes(title_text="ET [GeV]", rangemode="tozero", row=4, col=1)
        _fig.update_xaxes(title_text="Truth object id", row=4, col=3)
        _fig.update_yaxes(title_text="Purity [%]", range=[0, 105], row=4, col=3)
        _fig.update_xaxes(title_text="Truth object id", row=4, col=5)
        _fig.update_yaxes(title_text="Completeness [%]", range=[0, 105], row=4, col=5)
        return _fig

    def format_match_table_value(value):
        if value is None:
            return "None"
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "None"
            return f"{float(value):.3g}"
        return str(value)

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


@app.cell(hide_code=True)
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
    eval_feature_names,
    event_idx_selector,
    mo,
    mplhep,
    oc_display_projection,
    oc_eval,
    plot_true_vs_pred_oc,
    plt,
    primary_regression_source_key,
    projection_mode_selector,
    test_data,
    test_ds,
    test_eval_data,
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
        eval_features=test_eval_data["features"],
        eval_feature_names=eval_feature_names,
        oc_eval=oc_eval,
        event_idx=event_idx_selector.value,
        prediction_source=primary_regression_source_key,
    )
    mplhep.style.use("CMS")
    mo.ui.plotly(_fig, config={"responsive": True})
    return


@app.cell(hide_code=True)
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
