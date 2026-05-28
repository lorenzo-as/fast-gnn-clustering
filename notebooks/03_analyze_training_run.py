import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from fastgnn import get_project_root
    from fastgnn.data import CaloDataset

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    PRJ_ROOT = get_project_root()
    OUTPUT_DIR = PRJ_ROOT / "outputs"

    run_dirs = sorted(
        (p.parent for p in OUTPUT_DIR.glob("**/*.keras")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    selector_run = mo.ui.dropdown(
        options=[str(path.relative_to(OUTPUT_DIR)) for path in run_dirs],
        label="Select Run",
        # searchable=True
    )
    selector_run
    return (
        CaloDataset,
        OUTPUT_DIR,
        PRJ_ROOT,
        Path,
        mo,
        np,
        plt,
        selector_run,
        yaml,
    )


@app.cell(hide_code=True)
def _(OUTPUT_DIR, Path, mo, np, selector_run, yaml):
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

    mo.stop(selector_run.value is None, mo.md("Please select a run to analyze."))
    best_model, history, config = load_run(OUTPUT_DIR / selector_run.value)
    mo.md(
        "Loaded "
        + ("model, " if best_model is not None else "")
        + "history, and config for run: "
        + selector_run.value
    )
    return best_model, config, history


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview Plots
    """)
    return


@app.cell(hide_code=True)
def _(CaloDataset, PRJ_ROOT, config):
    test_ds = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split="test")
    test_ds
    return (test_ds,)


@app.cell(hide_code=True)
def _(history, plt):
    if history is not None:
        _fig = plt.figure(figsize=(5, 4))
        plt.plot(history["train"], label="Train Loss")
        plt.plot(history["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(best_model, config, np, test_ds):
    test_data = test_ds.as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=config["model"]["feature_names"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=config["training"].get("normalize_features", True),
        seed=config.get("seed", 0),
    )
    test_preds = best_model.predict(test_data["features"], batch_size=1024, verbose=1)

    def sigmoid(x):
        x = np.asarray(x)
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    test_beta = sigmoid(test_preds[..., 0])
    return sigmoid, test_beta, test_data, test_preds


@app.cell
def _(CaloDataset, PRJ_ROOT, config, plt):
    _data = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split=None).as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=config["model"]["feature_names"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=config["training"].get("normalize_features", True),
        seed=config.get("seed", 0),
    )
    _valid = _data["mask"].astype(bool)
    _signal = (_data["hit_object_id"] > 0) & _valid
    _signal_fraction = _signal.sum(axis=1) / _valid.sum(axis=1)
    # Distribution of percentage of signal vs real noise per event; padding is masked.
    _fig = plt.figure(figsize=(5, 4))
    plt.hist(_signal_fraction * 100, bins=50, alpha=0.5)
    plt.xlabel("Percentage of Signal Hits")
    plt.ylabel("Count")
    # plt.yscale("log")
    plt.title("Distribution of Signal Percentage (vs Noise) per Event")
    plt.text(
        0.95,
        0.95,
        f"Mean: {_signal_fraction.mean() * 100:.2f}%",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
    )
    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(np, plt, test_beta, test_data):
    # Histogram of beta values for signal hits and noise hits
    _beta_min, _beta_max = test_beta.min(), test_beta.max()
    _beta_min = 0 if _beta_min > 0 else _beta_min
    _valid_mask = test_data["mask"].astype(bool)
    _signal_mask = (test_data["hit_object_id"] > 0) & _valid_mask
    _noise_mask = (test_data["hit_object_id"] == 0) & _valid_mask
    _padding_mask = ~_valid_mask
    _fig = plt.figure(figsize=(5, 4))
    bins = np.linspace(_beta_min, _beta_max, 50)
    plt.hist(test_beta[_noise_mask], bins=bins, alpha=0.5, label="Noise")
    plt.hist(test_beta[_padding_mask], bins=bins, alpha=0.5, label="Padding")
    plt.hist(test_beta[_signal_mask], bins=bins, alpha=0.5, label="Signal")
    plt.xlabel(r"$\beta_\mathrm{pred}$")
    plt.ylabel("Count")
    # plt.yscale("log")
    plt.legend()
    plt.title(
        r"Distribution of $\beta_\mathrm{pred}$"
        + f"\n across {test_beta.shape[0]} test events with {test_beta.shape[1]} hits each."
    )
    plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OC Threshold Calibration and Validation

    We find the $\beta$ and distance thresholds ($t_\beta$, $t_d$) that **minimize the median difference between truth number of clusters and predicted number of clusters on the validation data**. Then we evaluate on the test dataset.

    *This is probably reasonable but needs a more refined way to do this*
    """)
    return


@app.cell
def _(CaloDataset, PRJ_ROOT, best_model, config, mo, sigmoid):
    mo.stop(
        best_model is None,
        mo.md("No model found for this run. Cannot compute predictions on validation set."),
    )
    val_data = CaloDataset(PRJ_ROOT / config["data"]["dataset_dir"], split="val").as_padded(
        max_vertices=config["model"]["max_vertices"],
        feature_names=config["model"]["feature_names"],
        truncate=config["training"].get("truncate", "first"),
        normalize_features=config["training"].get("normalize_features", True),
        seed=config.get("seed", 0),
    )
    val_preds = best_model.predict(val_data["features"], batch_size=1024, verbose=1)
    val_beta = sigmoid(val_preds[..., 0])
    return val_beta, val_data, val_preds


@app.cell
def _(np):
    import polars as pl
    from scipy.spatial.distance import cdist
    from tqdm import tqdm

    from fastgnn.training.objectcondensation import get_clustering_np

    def count_truth_objects(hit_object_id, mask=None):
        counts = []

        for i in range(len(hit_object_id)):
            ids = hit_object_id[i]

            if mask is not None:
                ids = ids[mask[i].astype(bool)]

            ids = ids[ids > 0]  # remove truth noise
            counts.append(len(np.unique(ids)))

        return np.asarray(counts)

    # ----------------------------------------------------------------------
    # Faster clustering using precomputed distances
    # ----------------------------------------------------------------------

    def get_clustering_precomputed(
        betas,
        D,
        tbeta,
        td,
    ):
        """
        Parameters
        ----------
        betas : (N,)
        D     : (N, N) pairwise distances
        """

        n_points = len(betas)

        # candidate condensation points
        cond_mask = betas > tbeta
        cond_indices = np.nonzero(cond_mask)[0]

        # highest beta first
        cond_indices = cond_indices[np.argsort(-betas[cond_indices])]

        clustering = -1 * np.ones(n_points, dtype=np.int32)

        unassigned = np.ones(n_points, dtype=bool)

        for idx_cp in cond_indices:
            if not unassigned[idx_cp]:
                continue

            assign_mask = unassigned & (D[idx_cp] < td)

            clustering[assign_mask] = idx_cp
            unassigned[assign_mask] = False

        return clustering

    def count_clusters_from_labels(clustering):
        pred_ids = np.unique(clustering)
        pred_ids = pred_ids[pred_ids >= 0]
        return len(pred_ids)

    # ----------------------------------------------------------------------
    # Precompute distances once
    # ----------------------------------------------------------------------

    def precompute_event_distances(preds):
        """
        preds shape:
            (n_events, n_hits, output_dim)

        Uses OC coordinates only:
            preds[..., 1:]
        """

        distances = []

        for i in range(len(preds)):
            X = preds[i, ..., 1:]

            # (N, N)
            D = cdist(X, X)

            distances.append(D)

        return distances

    def prepare_masked_oc_inputs(beta, distance_matrices, mask=None):
        masked_beta = []
        masked_distances = []

        for i in range(len(beta)):
            valid = np.ones(len(beta[i]), dtype=bool) if mask is None else mask[i].astype(bool)
            masked_beta.append(beta[i, valid])
            masked_distances.append(distance_matrices[i][np.ix_(valid, valid)])

        return masked_beta, masked_distances

    # ----------------------------------------------------------------------
    # Efficient threshold counting
    # ----------------------------------------------------------------------

    def count_pred_objects(
        beta,
        distance_matrices,
        tbeta,
        td,
        mask=None,
    ):
        masked_beta, masked_distances = prepare_masked_oc_inputs(beta, distance_matrices, mask)
        counts = []

        for b, D in zip(masked_beta, masked_distances, strict=True):
            if not np.any(b > tbeta):
                counts.append(0)
                continue

            clustering = get_clustering_precomputed(
                b,
                D,
                tbeta=tbeta,
                td=td,
            )

            counts.append(count_clusters_from_labels(clustering))

        return np.asarray(counts)

    # ----------------------------------------------------------------------
    # Efficient grid search
    # ----------------------------------------------------------------------

    def grid_search_thresholds(
        beta,
        preds,
        hit_object_id,
        mask=None,
        tbeta_values=None,
        td_values=None,
    ):
        if tbeta_values is None:
            tbeta_values = np.linspace(0.05, 0.9, 20)

        if td_values is None:
            td_values = np.linspace(0.05, 1.0, 20)

        # --------------------------------------------------------------
        # truth counts
        # --------------------------------------------------------------

        n_truth = count_truth_objects(
            hit_object_id,
            mask=mask,
        )

        # --------------------------------------------------------------
        # precompute pairwise distances once
        # --------------------------------------------------------------

        distance_matrices = precompute_event_distances(preds)
        masked_beta, masked_distances = prepare_masked_oc_inputs(beta, distance_matrices, mask)

        # --------------------------------------------------------------
        # grid search
        # --------------------------------------------------------------

        results = []

        for tbeta in tqdm(tbeta_values, desc="tbeta scan"):
            has_seed = [np.any(b > tbeta) for b in masked_beta]

            for td in td_values:
                counts = np.zeros(len(masked_beta), dtype=np.int32)

                for i, (b, D) in enumerate(zip(masked_beta, masked_distances, strict=True)):
                    if not has_seed[i]:
                        continue

                    clustering = get_clustering_precomputed(
                        b,
                        D,
                        tbeta=tbeta,
                        td=td,
                    )

                    counts[i] = count_clusters_from_labels(clustering)

                diff = counts - n_truth

                results.append(
                    {
                        "tbeta": tbeta,
                        "td": td,
                        "median_abs_diff": np.median(np.abs(diff)),
                        "mean_abs_diff": np.mean(np.abs(diff)),
                        "frac_exact": np.mean(diff == 0),
                        "frac_within_1": np.mean(np.abs(diff) <= 1),
                        "mean_pred": np.mean(counts),
                        "mean_truth": np.mean(n_truth),
                    }
                )

        results = pl.DataFrame(results)

        best = results.sort(
            by=[
                "median_abs_diff",
                "mean_abs_diff",
                "frac_exact",
            ],
            descending=[
                False,
                False,
                True,
            ],
        ).row(0, named=True)

        return best, results

    return (
        cdist,
        count_clusters_from_labels,
        count_pred_objects,
        count_truth_objects,
        get_clustering_np,
        grid_search_thresholds,
    )


@app.cell
def _(grid_search_thresholds, np, val_beta, val_data, val_preds):
    _tbeta_values = np.linspace(0.001, 0.9, 50)
    _td_values = np.linspace(0.05, 2.0, 40)

    best_oc_thresholds, _grid_results = grid_search_thresholds(
        beta=val_beta,
        preds=val_preds,
        hit_object_id=val_data["hit_object_id"],
        mask=val_data.get("mask", None),
        tbeta_values=_tbeta_values,
        td_values=_td_values,
    )

    best_oc_thresholds
    return (best_oc_thresholds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering Performance
    """)
    return


@app.cell
def _(
    best_oc_thresholds,
    cdist,
    count_pred_objects,
    count_truth_objects,
    np,
    plt,
    test_beta,
    test_data,
    test_preds,
    val_beta,
    val_data,
    val_preds,
):
    _tbeta, _td = best_oc_thresholds["tbeta"], best_oc_thresholds["td"]

    _n_truth_val = count_truth_objects(val_data["hit_object_id"], mask=val_data.get("mask", None))
    _n_pred_val = count_pred_objects(
        val_beta,
        [cdist(p[..., 1:], p[..., 1:]) for p in val_preds],
        _tbeta,
        _td,
        mask=val_data.get("mask", None),
    )

    _n_truth_test = count_truth_objects(
        test_data["hit_object_id"], mask=test_data.get("mask", None)
    )
    _n_pred_test = count_pred_objects(
        test_beta,
        [cdist(p[..., 1:], p[..., 1:]) for p in test_preds],
        _tbeta,
        _td,
        mask=test_data.get("mask", None),
    )

    _diff_val = _n_truth_val - _n_pred_val
    _diff_test = _n_truth_test - _n_pred_test

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(11, 5))

    _ax0.scatter(_n_truth_val, _n_pred_val, alpha=0.45, s=18, label="Validation")
    _ax0.scatter(_n_truth_test, _n_pred_test, alpha=0.45, s=18, label="Test")

    _max = max(_n_truth_val.max(), _n_pred_val.max(), _n_truth_test.max(), _n_pred_test.max())

    _ax0.plot([0, _max], [0, _max], "k--", linewidth=1)
    _ax0.set(
        xlabel="Truth signal object count",
        ylabel="Predicted cluster count",
        title=rf"Predicted vs truth cluster count" "\n" rf"$t_\beta={_tbeta:.3f}$, $t_d={_td:.3f}$",
        xlim=(-0.5, _max + 0.5),
        ylim=(-0.5, _max + 0.5),
    )
    _ax0.set_aspect("equal", adjustable="box")
    _ax0.grid(alpha=0.25)
    _ax0.legend()

    _ax1.hist(
        [_diff_val, _diff_test],
        bins=30,
        alpha=0.6,
        label=["Validation", "Test"],
    )

    _ax1.axvline(0, color="k", linestyle="--", linewidth=1)
    _ax1.set(
        xlabel="Truth count - predicted count", ylabel="Events", title="Cluster count residuals"
    )
    _ax1.grid(alpha=0.25)
    _ax1.legend()
    _ax1.text(
        0.95,
        0.95,
        f"Mean residual: {np.mean(_diff_val):.2f} (val), {np.mean(_diff_test):.2f} (test)",
        transform=_ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.tight_layout()
    _fig
    return


@app.cell
def _(np, plt, test_beta, test_data, test_ds):
    def aggregate_truth_object_beta_summary(events, hit_object_id, beta, mask):
        energies = []
        n_hits = []
        max_beta = []

        for event_idx in range(len(events)):
            event = events[event_idx]
            valid = mask[event_idx].astype(bool)
            event_hit_object_id = hit_object_id[event_idx][valid]
            event_beta = beta[event_idx][valid]

            object_ids = np.unique(event_hit_object_id[event_hit_object_id > 0])
            for object_id in object_ids:
                object_index = int(object_id) - 1
                object_mask = event_hit_object_id == object_id
                energies.append(float(event.truth.objects.impact_energy[object_index]))
                n_hits.append(int(object_mask.sum()))
                max_beta.append(float(event_beta[object_mask].max()))

        return np.asarray(energies), np.asarray(n_hits), np.asarray(max_beta)

    _object_energy, _object_n_hits, _object_max_beta = aggregate_truth_object_beta_summary(
        test_ds,
        test_data["hit_object_id"],
        test_beta,
        test_data["mask"],
    )

    _xscale = "log"
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    _ax0.scatter(_object_energy, _object_max_beta, alpha=0.35, s=14)
    _ax0.set(
        xlabel="Truth object energy [GeV]",
        ylabel=r"Max $\beta_\mathrm{pred}$ in object",
        title="Truth object energy vs max beta",
    )
    _ax0.grid(alpha=0.25)
    _ax0.set_xscale(_xscale)

    _ax1.scatter(_object_n_hits, _object_max_beta, alpha=0.35, s=14)
    _ax1.set(
        xlabel="Hits per truth object in model input",
        ylabel=r"Max $\beta_\mathrm{pred}$ in object",
        title="Truth object hit count vs max beta",
    )
    _ax1.grid(alpha=0.25)
    _ax1.set_xscale(_xscale)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Single Event Analysis and Visualization
    """)
    return


@app.cell
def _(
    count_clusters_from_labels,
    count_truth_objects,
    get_clustering_np,
    mo,
    np,
    plt,
    sigmoid,
    test_ds,
):
    def plot_true_vs_pred_oc(
        tbeta,
        td,
        event,
        preds,
        hit_object_id,
        mask=None,
        event_idx=0,
        show_noise=False,
    ):
        _pred = preds[event_idx]  # (hits, 4): beta, c1, c2, c3
        _hit_object_id = hit_object_id[event_idx]

        if mask is None:
            m = np.ones(len(_hit_object_id), dtype=bool)
        else:
            m = mask[event_idx].astype(bool)

        _beta, pred_c1, pred_c2, pred_c3 = _pred[m].T
        _beta = sigmoid(_beta)

        _hit_object_id = _hit_object_id[m]
        signal = _hit_object_id > 0
        noise = ~signal
        signal_ids = _hit_object_id[signal]
        truth_cmap = plt.get_cmap("tab20")
        truth_norm = (
            plt.Normalize(vmin=float(signal_ids.min()), vmax=float(signal_ids.max()))
            if signal_ids.size > 0
            else None
        )
        _coords = np.column_stack([pred_c1, pred_c2, pred_c3])
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

        _fig_3d = plt.figure(figsize=(14, 6), constrained_layout=True)

        ax1 = _fig_3d.add_subplot(1, 2, 1, projection="3d")

        if show_noise:
            ax1.scatter(
                pred_c1[noise],
                pred_c2[noise],
                pred_c3[noise],
                c="grey",
                s=8,
                alpha=0.5,
                label="noise",
            )

        ax1.scatter(
            pred_c1[signal],
            pred_c2[signal],
            pred_c3[signal],
            c=signal_ids,
            s=np.clip(_beta[signal] * 100, 5, 100),
            cmap=truth_cmap,
            norm=truth_norm,
            alpha=0.7,
        )
        _seed_indices = np.unique(_clustering[_clustering >= 0])
        if len(_seed_indices) > 0:
            ax1.scatter(
                pred_c1[_seed_indices],
                pred_c2[_seed_indices],
                pred_c3[_seed_indices],
                marker="x",
                c="black",
                s=90,
                linewidths=2,
            )

        ax1.set_title(f"Event {event_idx}: truth ID / beta size", pad=12)
        ax1.set_xlabel("c1", labelpad=8)
        ax1.set_ylabel("c2", labelpad=8)
        ax1.set_zlabel("c3", labelpad=8)
        _legend_handles = truth_object_legend_handles(
            event=event,
            object_ids=signal_ids,
            cmap=truth_cmap,
            norm=truth_norm,
            top_n=10,
        )
        if show_noise:
            _legend_handles.insert(
                0,
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="grey",
                    markersize=6,
                    alpha=0.5,
                    label="noise",
                ),
            )
        if len(_seed_indices) > 0:
            _legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="black",
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=2,
                    label="OC seeds",
                )
            )
        if _legend_handles:
            ax1.legend(
                handles=_legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                title="Top 10 highest-energy\ntruth objects",
            )

        ax2 = _fig_3d.add_subplot(1, 2, 2, projection="3d")

        if show_noise:
            ax2.scatter(
                pred_c1[noise],
                pred_c2[noise],
                pred_c3[noise],
                c="grey",
                s=8,
                alpha=0.5,
                label="noise",
            )

        sc2 = ax2.scatter(
            pred_c1[signal],
            pred_c2[signal],
            pred_c3[signal],
            c=_beta[signal],
            s=12,
            alpha=0.8,
        )

        ax2.set_title(f"Event {event_idx}: beta", pad=12)
        ax2.set_xlabel("c1", labelpad=8)
        ax2.set_ylabel("c2", labelpad=8)
        ax2.set_zlabel("c3", labelpad=8)
        if show_noise:
            ax2.legend(loc="upper left")

        _fig_3d.colorbar(sc2, ax=ax2, shrink=0.65, pad=0.08, label="beta")

        _fig_diag, (ax3, ax4) = plt.subplots(
            1,
            2,
            figsize=(12, 4),
            constrained_layout=True,
        )
        _bins = np.linspace(0, max(1.0, float(_beta.max())), 40)
        if noise.any():
            ax3.hist(_beta[noise], bins=_bins, alpha=0.55, label=f"Noise ({noise.sum()} hits)")
        if signal.any():
            ax3.hist(_beta[signal], bins=_bins, alpha=0.55, label=f"Signal ({signal.sum()} hits)")
        ax3.axvline(tbeta, color="k", linestyle="--", linewidth=1, label=rf"$t_\beta={tbeta:.3f}$")
        ax3.set(
            xlabel=r"$\beta_\mathrm{pred}$",
            ylabel="Hits",
            title="Event beta distribution",
        )
        ax3.grid(alpha=0.25)
        ax3.set_yscale("log")
        ax3.legend()

        _labels, _counts = np.unique(_clustering[_clustering >= 0], return_counts=True)
        if len(_labels) > 0:
            ax4.bar(np.arange(len(_labels)), _counts)
            ax4.set_xticks(np.arange(len(_labels)))
            ax4.set_xticklabels([str(label) for label in _labels])
        ax4.set(
            xlabel="Predicted OC seed hit index",
            ylabel="Assigned valid hits",
            title=(
                f"Predicted clusters after OC: {_n_pred_clusters}\n"
                f"Truth clusters: {_n_truth_clusters}; "
                rf"$t_\beta={tbeta:.3f}$, $t_d={td:.3f}$"
            ),
        )
        ax4.grid(axis="y", alpha=0.25)

        return _fig_3d, _fig_diag

    def pdgid_to_name(pdgid):
        try:
            from particle import Particle

            return Particle.from_pdgid(int(pdgid)).name
        except Exception:
            return str(pdgid)

    def truth_object_legend_handles(event, object_ids, cmap, norm, top_n=10):
        if norm is None or object_ids.size == 0:
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
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(norm(object_id)),
                markersize=6,
                label=f"{particle} ({energy:.1f} GeV)",
            )
            for energy, object_id, particle in entries
        ]

    event_idx_selector = mo.ui.number(label="Event Index", start=0, stop=len(test_ds) - 1, step=1)
    event_idx_selector

    return event_idx_selector, plot_true_vs_pred_oc


@app.cell(hide_code=True)
def _(
    best_oc_thresholds,
    event_idx_selector,
    mo,
    plot_true_vs_pred_oc,
    test_data,
    test_ds,
    test_preds,
):
    mo.stop(
        best_oc_thresholds is None,
        mo.md("Run OC threshold calibration above before using the single-event OC analysis."),
    )
    _tbeta, _td = best_oc_thresholds["tbeta"], best_oc_thresholds["td"]
    _fig_3d, _fig_diag = plot_true_vs_pred_oc(
        tbeta=_tbeta,
        td=_td,
        event=test_ds[event_idx_selector.value],
        preds=test_preds,
        hit_object_id=test_data["hit_object_id"],
        mask=test_data["mask"],
        event_idx=event_idx_selector.value,
        show_noise=True,
    )
    mo.vstack([_fig_3d, _fig_diag])
    return


@app.cell
def _(event_idx_selector, mo, test_ds):
    from fastgnn.data.cmssw.plotting import plot_event

    _fig, _desc = plot_event(
        event=test_ds[event_idx_selector.value],
        color_by="hit_object_id",
        show_cluster_markers=False,
        energy_threshold=test_ds.metadata["preprocessing"]["truth_min_object_energy"],
    )
    mo.vstack(
        [
            mo.md(
                f"**Event {test_ds[event_idx_selector.value].event_id}** | "
                f"{test_ds[event_idx_selector.value].n_hits} hits | "
                f"{test_ds[event_idx_selector.value].n_objects} SimClusters"
            ),
            mo.ui.plotly(_fig),
            mo.md(_desc),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
