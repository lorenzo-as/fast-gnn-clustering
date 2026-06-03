# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import itertools

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import os
    from pathlib import Path

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
        mo,
        mplhep,
        np,
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
    best_model, history, config = load_run(OUTPUT_DIR / selected_run)
    test_ds = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split="test")
    messages = [
        f"Loaded {'model, ' if best_model is not None else ''}history, and config for run: `{selected_run}`",
        f"Loaded test dataset: `{test_ds}`",
    ]

    mo.md("\n\n".join(messages))
    return best_model, config, history, test_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Loss Evolution
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    We find the $\beta$ and distance thresholds ($t_\beta$, $t_d$) that **minimize the median difference between truth number of clusters and predicted number of clusters on the validation data**. Then we evaluate on the test dataset.

    *This is probably reasonable but needs a more refined way to do this*
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
    val_preds = best_model.predict(val_data["features"], batch_size=1024, verbose=1)
    val_output_slices = split_oc_outputs(val_preds, oc_layout)
    val_beta = np.asarray(val_output_slices.beta)
    val_cluster_coords = np.asarray(val_output_slices.cluster_coords)
    return val_beta, val_cluster_coords, val_data


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


@app.cell(hide_code=True)
def _(grid_search_thresholds, np, val_beta, val_cluster_coords, val_data):
    _tbeta_values = np.linspace(0.001, 0.9, 50)
    _td_values = np.linspace(0.05, 2.0, 40)

    best_oc_thresholds, _grid_results = grid_search_thresholds(
        beta=val_beta,
        cluster_coords=val_cluster_coords,
        hit_object_id=val_data["hit_object_id"],
        mask=val_data.get("mask", None),
        tbeta_values=_tbeta_values,
        td_values=_td_values,
    )

    best_oc_thresholds
    return (best_oc_thresholds,)


@app.cell
def _(mo):
    matching_distance_threshold_selector = mo.ui.number(
        label="Distance Threshold in cm", value=50, step=0.5
    )
    matching_energy_ratio_min_selector = mo.ui.number(
        label="Minimum Energy Ratio", value=0.01, step=0.01
    )
    matching_energy_ratio_max_selector = mo.ui.number(
        label="Maximum Energy Ratio", value=10.0, step=0.01
    )

    # vstack
    mo.vstack(
        [
            mo.md(
                "**Select the maximum euclidean distance (x,y,z) in cm and the minimum and maximum energy ratio between predicted clusters and true objects for the matching.**"
            ),
            mo.hstack(
                [
                    matching_distance_threshold_selector,
                    matching_energy_ratio_min_selector,
                    matching_energy_ratio_max_selector,
                ]
            ),
        ]
    )
    return (
        matching_distance_threshold_selector,
        matching_energy_ratio_max_selector,
        matching_energy_ratio_min_selector,
    )


@app.cell(hide_code=True)
def _(
    best_oc_thresholds,
    evaluate_oc_padded,
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
    )
    count_summary = oc_eval.count_summary()
    seed_summary = oc_eval.seed_summary()
    matching_summary = oc_eval.matching_summary()
    print(count_summary, "\n\n", seed_summary, "\n\n", matching_summary)
    return (oc_eval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy and Position Regression
    """)
    return


@app.cell
def _(np, pl):
    REFERENCE_LINE_KWARGS = {"color": "green", "linestyle": "--"}

    def et_weighted_str(s: str) -> str:
        return s
        """Wraps a Latex string (must start and end with $) to mark weighting by transverse energy."""
        assert s[0] == "$" and s[-1] == "$"
        return r"$\langle " + s[1:-1] + r"\rangle_\mathrm{E_T}$"

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
        for i, (lo, hi) in enumerate(itertools.pairwise(bins[:-1], bins[1:])):
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
        et_weighted_str,
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
    et_weighted_str,
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
    _axs[1].set_xlabel(et_weighted_str(r"$\eta_\mathrm{{true}}$"))

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
def _(PLOTTING_CONFIG, et_weighted_str, oc_eval, plt):
    _fig1, _axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=PLOTTING_CONFIG["figsize"]["A4"]["fullwidth_3pane"],
    )
    for _ax, _col, _label in [
        (
            _axs[0],
            "eta_residual",
            et_weighted_str(r"$\eta_\mathrm{true}$")
            + " - "
            + et_weighted_str(r"$\eta_\mathrm{pred}$"),
        ),
        (
            _axs[1],
            "phi_residual",
            et_weighted_str(r"$\phi_\mathrm{true}$")
            + " - "
            + et_weighted_str(r"$\phi_\mathrm{pred}$"),
        ),
        (
            _axs[2],
            "z_residual",
            et_weighted_str(r"$z_\mathrm{true}$")
            + " - "
            + et_weighted_str(r"$z_\mathrm{pred}$")
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
    et_weighted_str,
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
        (_axs[0], "relative_eta_residual", et_weighted_str(r"$\Delta_\mathrm{rel}\eta$")),
        (_axs[1], "relative_phi_residual", et_weighted_str(r"$\Delta_\mathrm{rel}\phi$")),
        (_axs[2], "relative_z_residual", et_weighted_str(r"$\Delta_\mathrm{rel}z$")),
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
    #### Efficiency and Fake Rate over Cluster Energy
    """)
    return


@app.cell
def _(np):
    def log_bin_centers_and_xerr(bin_edges):
        """Return geometric bin centers and asymmetric x errors for log-scaled bins."""
        centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        xerr = [
            centers - bin_edges[:-1],
            bin_edges[1:] - centers,
        ]
        return centers, xerr

    def plot_binned_metric(
        ax,
        table,
        *,
        metric,
        xlabel,
        ylabel,
        threshold=None,
        xlim=None,
        ylim=None,
        color="black",
    ):
        """
        Plot a binned metric with asymmetric confidence intervals.

        Expects columns:
          - bin_low
          - bin_high
          - {metric}
          - {metric}_confidence_low
          - {metric}_confidence_high
        """
        bin_edges = np.append(
            table["bin_low"].to_numpy(),
            table["bin_high"].to_numpy()[-1],
        )

        centers, xerr = log_bin_centers_and_xerr(bin_edges)

        values = table[metric].to_numpy()
        yerr = [
            values - table[f"{metric}_confidence_low"].to_numpy(),
            table[f"{metric}_confidence_high"].to_numpy() - values,
        ]

        ax.errorbar(
            centers,
            values,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=color,
        )

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if threshold is not None:
            ax.axvline(
                threshold,
                color="grey",
                linestyle="--",
            )

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        return ax

    return (plot_binned_metric,)


@app.cell
def _(
    PLOTTING_CONFIG,
    binned_efficiency,
    binned_fake_rate,
    log_edges,
    np,
    oc_eval,
    plot_binned_metric,
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

    plot_binned_metric(
        _axs[0],
        _truth_energy_table,
        metric="efficiency",
        xlabel="Cluster truth energy [GeV]",
        ylabel="Efficiency",
        threshold=threshold,
        xlim=(7e-1, 1e3),
        ylim=(0.0, 1.0),
    )

    plot_binned_metric(
        _axs[1],
        _pred_energy_table,
        metric="fake_rate",
        xlabel="Cluster predicted energy [GeV]",
        ylabel="Fake Rate",
        threshold=threshold,
        xlim=(1.5e-1, 1e3),
        ylim=(-0.02, 1.0),
    )

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
    plot_binned_metric,
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

        plot_binned_metric(
            _ax,
            _table,
            metric=_metric,
            xlabel=_xlabel,
            ylabel="Efficiency" if _metric == "efficiency" else "Fake Rate",
            ylim=(None, 1.0),
        )
        _ax.set_ylim(-0.02, 1.0)
        _ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        _ax.grid(alpha=0.25)

    plt.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering Performance
    """)
    return


@app.cell(disabled=True)
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
def _(
    count_clusters_from_labels,
    count_truth_objects,
    get_clustering_np,
    mo,
    np,
    oc_layout,
    split_oc_outputs,
    test_ds,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    def plot_true_vs_pred_oc(
        tbeta,
        td,
        event,
        preds,
        hit_object_id,
        mask=None,
        oc_eval=None,
        event_idx=0,
        show_noise=False,
    ):
        _outputs = split_oc_outputs(preds[event_idx : event_idx + 1], oc_layout)
        _hit_object_id = hit_object_id[event_idx]

        if mask is None:
            m = np.ones(len(_hit_object_id), dtype=bool)
        else:
            m = mask[event_idx].astype(bool)

        _beta = np.asarray(_outputs.beta[0])[m]
        _coords = np.asarray(_outputs.cluster_coords[0])[m]
        pred_c1 = _coords[:, 0]
        pred_c2 = _coords[:, 1] if _coords.shape[1] > 1 else np.zeros(len(_coords))
        pred_c3 = _coords[:, 2] if _coords.shape[1] > 2 else np.zeros(len(_coords))

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
        _fake_seed_indices = [
            int(seed_index) for seed_index in _seed_indices if int(seed_index) not in _seed_to_truth
        ]
        _fake_seed_colors = {
            seed_index: color_at(len(_truth_ids) + i)
            for i, seed_index in enumerate(_fake_seed_indices)
        }

        _fig = make_subplots(
            rows=3,
            cols=6,
            specs=[
                [
                    {"type": "scene", "colspan": 2},
                    None,
                    {"type": "scene", "colspan": 2},
                    None,
                    {"type": "scene", "colspan": 2},
                    None,
                ],
                [{"type": "xy", "colspan": 6}, None, None, None, None, None],
                [
                    {"type": "xy", "colspan": 3},
                    None,
                    None,
                    {"type": "xy", "colspan": 3},
                    None,
                    None,
                ],
            ],
            subplot_titles=(
                "Truth assignment in OC space",
                "Predicted clusters in OC space",
                "β in OC space",
                "",
                "Event β distribution",
                "Predicted OC seed sizes",
            ),
            row_heights=[0.58, 0.19, 0.23],
            vertical_spacing=0.06,
            horizontal_spacing=0.035,
        )

        def _marker_size(values):
            return np.clip(5 + np.asarray(values) * 28, 5, 34)

        def _hover(object_label):
            return (
                "c1=%{x:.3f}<br>c2=%{y:.3f}<br>c3=%{z:.3f}<br>"
                "β=%{customdata:.3f}<extra>" + object_label + "</extra>"
            )

        if show_noise and noise.any():
            for _col in (1, 3):
                _fig.add_trace(
                    go.Scatter3d(
                        x=pred_c1[noise],
                        y=pred_c2[noise],
                        z=pred_c3[noise],
                        customdata=_beta[noise],
                        mode="markers",
                        marker={"color": "rgba(150,150,150,0.45)", "size": 4},
                        name="noise",
                        showlegend=False,
                        hovertemplate=_hover("noise"),
                    ),
                    row=1,
                    col=_col,
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
                col=3,
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
                col=3,
            )

        _fig.add_trace(
            go.Scatter3d(
                x=pred_c1,
                y=pred_c2,
                z=pred_c3,
                customdata=_hit_object_id,
                mode="markers",
                marker={
                    "color": _beta,
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": max(1.0, float(_beta.max())) if _beta.size else 1.0,
                    "size": 5,
                    "opacity": 0.82,
                    "colorbar": {"title": "β", "x": 1.0, "len": 0.48, "y": 0.79},
                },
                name="β",
                showlegend=False,
                hovertemplate=(
                    "c1=%{x:.3f}<br>c2=%{y:.3f}<br>c3=%{z:.3f}<br>"
                    "β=%{marker.color:.3f}<br>object_id=%{customdata}<extra></extra>"
                ),
            ),
            row=1,
            col=5,
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
                row=3,
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
                row=3,
                col=1,
            )
        _fig.add_shape(
            type="line",
            x0=tbeta,
            x1=tbeta,
            y0=0,
            y1=1,
            xref="x2",
            yref="y2 domain",
            line={"color": "black", "dash": "dash", "width": 1},
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
                row=3,
                col=4,
            )

        _truth_entries = truth_object_legend_entries(event, signal_ids, _truth_colors, top_n=10)
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
        _pred_text = "<b>Predicted clusters:</b><br>" + "<br>".join(
            f"<span style='color:{color}'>●</span> {label}"
            for color, label in _matched_entries + _fake_entries
        )
        if not _truth_entries:
            _truth_text += "No signal truth clusters"
        if not (_matched_entries or _fake_entries):
            _pred_text += "No predicted clusters"

        for _x, _text in [(0.015, _truth_text), (0.44, _pred_text)]:
            _fig.add_annotation(
                x=_x,
                y=0.49,
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

        _fig.update_xaxes(visible=False, row=2, col=1)
        _fig.update_yaxes(visible=False, row=2, col=1)

        _display_event_id = getattr(event, "event_id", event_idx)
        _camera = {"eye": {"x": 1.35, "y": 1.35, "z": 1.0}}
        _scene_layout = {
            "xaxis_title": "c1",
            "yaxis_title": "c2",
            "zaxis_title": "c3",
            "aspectmode": "cube",
            "camera": _camera,
        }
        _fig.update_layout(
            title={
                "text": (
                    f"Event {_display_event_id} | Truth clusters: {_n_truth_clusters} | "
                    f"Predicted clusters: {_n_pred_clusters}"
                ),
                "x": 0.02,
                "xanchor": "left",
            },
            template="plotly_white",
            height=1080,
            autosize=False,
            barmode="overlay",
            legend={"orientation": "h", "y": -0.06, "x": 0},
            margin={"l": 35, "r": 35, "t": 90, "b": 35},
            scene=_scene_layout,
            scene2=_scene_layout,
            scene3=_scene_layout,
            uirevision=f"event-{event_idx}",
        )
        _fig.update_xaxes(title_text="β", row=3, col=1)
        _fig.update_yaxes(title_text="Hits", type="log", row=3, col=1)
        _fig.update_xaxes(title_text="Predicted OC seed hit index", row=3, col=4)
        _fig.update_yaxes(title_text="Assigned valid hits", row=3, col=4)
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

    event_idx_selector = mo.ui.number(label="Event Index", start=0, stop=len(test_ds) - 1, step=1)
    event_idx_selector
    return apply_truth_colors_to_event_display, event_idx_selector, plot_true_vs_pred_oc


@app.cell(hide_code=True)
def _(
    best_oc_thresholds,
    event_idx_selector,
    mo,
    mplhep,
    oc_eval,
    plot_true_vs_pred_oc,
    plt,
    test_data,
    test_ds,
    test_preds,
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
        mask=test_data["mask"],
        oc_eval=oc_eval,
        event_idx=event_idx_selector.value,
        show_noise=True,
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
