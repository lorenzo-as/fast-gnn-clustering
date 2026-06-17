"""Matplotlib helpers for binned-profile and response plots."""

from __future__ import annotations

import numpy as np
import polars as pl

REFERENCE_LINE_KWARGS = {"color": "green", "linestyle": "--"}


def energy_weighted_str(s: str) -> str:
    """Wrap a LaTeX string (delimited by ``$``) to mark energy weighting."""
    if not (s.startswith("$") and s.endswith("$")):
        raise ValueError("energy_weighted_str expects a $-delimited LaTeX string")
    return r"$\langle " + s[1:-1] + r"\rangle_\mathrm{E}$"


def plot_profile_points(
    ax, table: pl.DataFrame, y_col: str, *, color="black", marker="o", label=None
) -> None:
    """Overlay a binned-profile column as error bars on ``ax``.

    ``table`` is a frame from :func:`fastgnn.evaluation.binned.binned_profile`
    with ``bin_low``/``bin_high`` edges and the ``y_col`` statistic per bin.
    """
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
