"""Tests for configurable z-score / robust feature normalization."""

from __future__ import annotations

import numpy as np
import pytest

from fastgnn.data.base import apply_normalization, compute_normalization


def test_compute_normalization_emits_zscore_and_robust_stats() -> None:
    records = [{"hits": {"a": np.array([0.0, 1.0, 2.0, 3.0, 100.0])}}]
    norm = compute_normalization(records, ["a"], np.array([0]))
    stats = norm["a"]
    assert set(stats) == {"mean", "std", "median", "iqr"}
    assert stats["median"] == 2.0  # robust to the 100.0 outlier; mean is not
    assert stats["mean"] != stats["median"]


def test_apply_normalization_zscore_is_default_and_backcompat() -> None:
    features = np.array([[0.0], [10.0]], dtype=np.float32)
    # A legacy normalization dict with only mean/std must still work (zscore).
    norm = {"a": {"mean": 5.0, "std": 2.0}}
    out = apply_normalization(features, ["a"], norm)
    np.testing.assert_allclose(out.ravel(), [(0 - 5) / 2, (10 - 5) / 2])


def test_apply_normalization_robust_uses_median_iqr() -> None:
    features = np.array([[2.0]], dtype=np.float32)
    norm = {"a": {"mean": 50.0, "std": 40.0, "median": 2.0, "iqr": 1.0}}
    out = apply_normalization(features, ["a"], norm, method="robust")
    np.testing.assert_allclose(out.ravel(), [0.0])


def test_apply_normalization_per_feature_overrides() -> None:
    features = np.array([[5.0, 2.0]], dtype=np.float32)
    norm = {
        "a": {"mean": 5.0, "std": 2.0, "median": 0.0, "iqr": 1.0},
        "b": {"mean": 0.0, "std": 1.0, "median": 2.0, "iqr": 4.0},
    }
    method = {"method": "zscore", "overrides": {"b": "robust"}}
    out = apply_normalization(features, ["a", "b"], norm, method=method)
    # a -> zscore: (5-5)/2 = 0 ; b -> robust: (2-2)/4 = 0
    np.testing.assert_allclose(out.ravel(), [0.0, 0.0])


def test_apply_normalization_robust_requires_robust_stats() -> None:
    norm = {"a": {"mean": 5.0, "std": 2.0}}  # no median/iqr
    with pytest.raises(ValueError, match="robust"):
        apply_normalization(np.array([[1.0]], dtype=np.float32), ["a"], norm, method="robust")
