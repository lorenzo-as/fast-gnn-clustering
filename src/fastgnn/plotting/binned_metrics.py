"""Small helpers for plotting binned metrics."""

from __future__ import annotations

import numpy as np


def bin_centers_and_xerr(bins, *, scale: str = "log"):
    """Return bin centers and asymmetric x-errors."""
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or len(bins) < 2:
        raise ValueError("bins must be a one-dimensional array with at least two edges")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be strictly increasing")

    if scale == "log":
        if np.any(bins <= 0):
            raise ValueError("log-scale bin centers require positive bin edges")
        centers = np.sqrt(bins[:-1] * bins[1:])
    elif scale == "linear":
        centers = 0.5 * (bins[:-1] + bins[1:])
    else:
        raise ValueError("scale must be 'log' or 'linear'")

    xerr = np.vstack([centers - bins[:-1], bins[1:] - centers])
    return centers, xerr


def rate_yerr(rate, confidence_low, confidence_high):
    """Return non-negative asymmetric errors from a rate confidence interval."""
    rate = np.asarray(rate, dtype=float)
    confidence_low = np.asarray(confidence_low, dtype=float)
    confidence_high = np.asarray(confidence_high, dtype=float)
    return np.clip(
        np.vstack([rate - confidence_low, confidence_high - rate]),
        a_min=0.0,
        a_max=None,
    )


def plot_binned_metric(
    ax,
    centers,
    values,
    *,
    xerr=None,
    yerr=None,
    label: str | None = None,
    color: str | None = None,
):
    """Plot one binned metric series as error bars."""
    ax.errorbar(
        centers,
        values,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        label=label,
        color=color,
    )
    return ax


def plot_binned_efficiency(ax, table, *, label: str | None = None, color: str | None = None):
    """Plot an efficiency table returned by ``binned_efficiency``."""
    return _plot_binned_rate(ax, table, "efficiency", label=label, color=color)


def plot_binned_fake_rate(ax, table, *, label: str | None = None, color: str | None = None):
    """Plot a fake-rate table returned by ``binned_fake_rate``."""
    return _plot_binned_rate(ax, table, "fake_rate", label=label, color=color)


def _plot_binned_rate(ax, table, rate_name: str, *, label: str | None, color: str | None):
    bins = np.append(table["bin_low"].to_numpy(), table["bin_high"].to_numpy()[-1])
    centers, xerr = bin_centers_and_xerr(bins)
    values = table[rate_name].to_numpy()
    yerr = rate_yerr(
        values,
        table[f"{rate_name}_confidence_low"].to_numpy(),
        table[f"{rate_name}_confidence_high"].to_numpy(),
    )
    return plot_binned_metric(ax, centers, values, xerr=xerr, yerr=yerr, label=label, color=color)
