"""Small scalar and angular helpers shared across the evaluation package."""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def safe_none_ratio(numerator: float | None, denominator: float | None) -> float | None:
    return (
        None
        if numerator is None or denominator is None or denominator == 0
        else float(numerator) / float(denominator)
    )


def delta_phi(phi0: float | None, phi1: float | None) -> float | None:
    return (
        None if phi0 is None or phi1 is None else float((phi0 - phi1 + np.pi) % (2 * np.pi) - np.pi)
    )


def none_subtract(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left - right)


def first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def none_if_nan(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def none_to_nan(value: float | None) -> float:
    return np.nan if value is None else float(value)
