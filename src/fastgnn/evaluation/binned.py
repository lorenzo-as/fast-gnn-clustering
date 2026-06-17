"""Binned efficiency / fake-rate tables and binned-profile statistics."""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from statsmodels.stats.proportion import proportion_confint


def binned_efficiency(
    truth: pl.DataFrame,
    value_col: str,
    bins: np.ndarray,
    *,
    matched_col: str = "matched",
) -> pl.DataFrame:
    return _binned_rate(truth, value_col, bins, numerator_col=matched_col, rate_name="efficiency")


def binned_fake_rate(
    predicted: pl.DataFrame,
    value_col: str,
    bins: np.ndarray,
    *,
    fake_col: str = "fake",
) -> pl.DataFrame:
    return _binned_rate(predicted, value_col, bins, numerator_col=fake_col, rate_name="fake_rate")


def log_edges(values: np.ndarray, bins: int = 12) -> np.ndarray:
    """Logarithmically spaced bin edges spanning ``values``."""
    values = np.asarray(values, dtype=float)
    return np.logspace(np.log10(values.min()), np.log10(values.max()), bins + 1)


def binned_profile(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> pl.DataFrame:
    """Per-bin mean, median, and 68% half-width of ``y`` grouped by ``x``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.asarray(bins, dtype=float)

    assert x.shape == y.shape
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
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


def fwhm_and_mu(values: np.ndarray) -> tuple[float | None, float | None]:
    """Full width at half maximum (from a 50-bin histogram) and the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    counts, edges = np.histogram(values, bins=50)
    above_half_max = np.flatnonzero(counts >= 0.5 * counts.max())
    fwhm = (
        None
        if len(above_half_max) == 0
        else float(edges[above_half_max[-1] + 1] - edges[above_half_max[0]])
    )
    return fwhm, float(values.mean())


def _binned_rate(
    frame: pl.DataFrame, value_col: str, bins: np.ndarray, *, numerator_col: str, rate_name: str
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    values = frame[value_col].to_numpy()
    numerator_values = frame[numerator_col].to_numpy()
    indices = np.digitize(values, bins) - 1
    rows = []
    for bin_idx in range(len(bins) - 1):
        in_bin = indices == bin_idx
        denom = int(np.sum(in_bin))
        numerator = int(np.sum(numerator_values[in_bin])) if denom else 0
        rate = _safe_ratio(numerator, denom)
        low, high = (
            proportion_confint(numerator, denom, alpha=0.32, method="wilson")
            if denom
            else (0.0, 0.0)
        )
        rows.append(
            {
                "bin_low": float(bins[bin_idx]),
                "bin_high": float(bins[bin_idx + 1]),
                "n": denom,
                "numerator": numerator,
                rate_name: rate,
                f"{rate_name}_binomial_uncertainty": np.sqrt(rate * (1 - rate) / denom)
                if denom
                else 0.0,
                f"{rate_name}_confidence_low": float(low),
                f"{rate_name}_confidence_high": float(high),
            }
        )
    return pl.DataFrame(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)
