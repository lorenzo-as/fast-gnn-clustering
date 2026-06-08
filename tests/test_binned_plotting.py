from __future__ import annotations

from importlib import util
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from fastgnn.evaluation import OCEvaluation, binned_efficiency
from fastgnn.plotting import (
    bin_centers_and_xerr,
    plot_binned_efficiency,
    plot_binned_fake_rate,
    rate_yerr,
)


def test_log_bin_centers_and_xerr() -> None:
    centers, xerr = bin_centers_and_xerr(np.array([1.0, 4.0, 16.0]))

    np.testing.assert_allclose(centers, [2.0, 8.0])
    np.testing.assert_allclose(xerr, [[1.0, 4.0], [2.0, 8.0]])


def test_rate_yerr_clips_to_non_negative() -> None:
    yerr = rate_yerr(
        np.array([0.2, 0.8]),
        np.array([0.3, 0.5]),
        np.array([0.4, 0.7]),
    )

    np.testing.assert_allclose(yerr, [[0.0, 0.3], [0.2, 0.0]])


def test_binned_rate_wrappers_plot_expected_centers() -> None:
    truth = pl.DataFrame({"truth_energy": [1.0, 2.0, 8.0], "matched": [True, False, True]})
    table = binned_efficiency(truth, "truth_energy", np.array([1.0, 4.0, 16.0]))
    fig, ax = plt.subplots()

    plot_binned_efficiency(ax, table, label="run-a", color="black")

    np.testing.assert_allclose(ax.lines[0].get_xdata(), [2.0, 8.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.5, 1.0])
    assert ax.get_legend_handles_labels()[1] == ["run-a"]
    plt.close(fig)


def test_fake_rate_wrapper_uses_fake_rate_columns() -> None:
    table = pl.DataFrame(
        {
            "bin_low": [1.0],
            "bin_high": [10.0],
            "fake_rate": [0.25],
            "fake_rate_confidence_low": [0.1],
            "fake_rate_confidence_high": [0.4],
        }
    )
    fig, ax = plt.subplots()

    plot_binned_fake_rate(ax, table)

    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.25])
    plt.close(fig)


def test_efficiency_fake_rate_script_smoke(tmp_path: Path, monkeypatch) -> None:
    script_path = (
        Path(__file__).parents[1]
        / "plotting"
        / "efficiency_fake_rate"
        / "plot_binned_efficiency_fake_rate.py"
    )
    spec = util.spec_from_file_location("plot_binned_efficiency_fake_rate", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = []
    for idx in range(2):
        oc_eval = OCEvaluation(
            events=pl.DataFrame(),
            truth=pl.DataFrame(
                {
                    "truth_energy": [1.0, 2.0, 8.0 + idx],
                    "truth_n_hits": [2, 4, 8 + idx],
                    "matched": [True, False, True],
                }
            ),
            predicted=pl.DataFrame(
                {
                    "assigned_cluster_energy": [1.5, 3.0, 12.0 + idx],
                    "cluster_size": [2, 5, 10 + idx],
                    "fake": [False, True, False],
                }
            ),
            matches=pl.DataFrame(),
            seeds=pl.DataFrame(),
        )
        path = tmp_path / f"eval_{idx}.pkl"
        with path.open("wb") as f:
            pickle.dump(oc_eval, f)
        paths.append(path)

    output = tmp_path / "comparison.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_binned_efficiency_fake_rate.py",
            "--inputs",
            str(paths[0]),
            str(paths[1]),
            "--labels",
            "a",
            "b",
            "--output",
            str(output),
        ],
    )

    module.main()

    assert output.exists()
    assert output.stat().st_size > 0
    plt.close("all")
