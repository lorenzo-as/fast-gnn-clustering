import numpy as np
import polars as pl
import pytest

from fastgnn.evaluation.oc_metrics import (
    _matched_positions,
    binned_efficiency,
    binned_fake_rate,
    evaluate_oc_padded,
    grid_search_thresholds,
)
from fastgnn.training.oc_outputs import OCOutputLayout, split_oc_outputs


def test_legacy_output_layout_splits_beta_and_cluster_coords() -> None:
    outputs = np.array([[[0.0, 1.0, 2.0, 3.0]]], dtype=np.float32)

    layout = OCOutputLayout.from_config({"output_dim": 4})
    slices = split_oc_outputs(outputs, layout)

    assert layout.beta_index == 0
    assert layout.cluster_start == 1
    assert layout.cluster_dim == 3
    np.testing.assert_allclose(slices.beta, [[0.5]])
    np.testing.assert_allclose(slices.cluster_coords, [[[1.0, 2.0, 3.0]]])
    assert slices.regressions == {}


def test_explicit_output_layout_with_regressions() -> None:
    cfg = {
        "output_dim": 8,
        "output_layout": {
            "beta": {"index": 0, "activation": "sigmoid"},
            "cluster_space": {"start": 1, "dim": 3},
            "regressions": [
                {"name": "energy", "start": 4, "dim": 1, "source": "seed"},
                {
                    "name": "position",
                    "start": 5,
                    "dim": 3,
                    "source": "seed",
                    "components": ["x", "y", "z"],
                },
            ],
        },
    }
    outputs = np.arange(16, dtype=np.float32).reshape(1, 2, 8)

    layout = OCOutputLayout.from_config(cfg)
    slices = split_oc_outputs(outputs, layout)

    np.testing.assert_array_equal(slices.cluster_coords, outputs[..., 1:4])
    np.testing.assert_array_equal(slices.regressions["energy"], outputs[..., 4:5])
    np.testing.assert_array_equal(slices.regressions["position"], outputs[..., 5:8])


def test_output_layout_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="must sum to output_dim"):
        OCOutputLayout.from_config(
            {
                "output_dim": 5,
                "output_layout": {
                    "beta": {"index": 0, "activation": "sigmoid"},
                    "cluster_space": {"start": 1, "dim": 3},
                    "regressions": [],
                },
            }
        )


def test_oc_evaluation_seed_matching_by_xyz_distance_and_binned_rates() -> None:
    beta = np.array([0.90, 0.60, 0.85, 0.70, 0.80, 0.75], dtype=np.float64)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    energy_regression = np.array([6.1, 0.0, 9.5, 0.0, 1.0, 1.2], dtype=np.float64)
    position_regression = np.column_stack([np.arange(6), np.arange(6) + 1, np.arange(6) + 2])
    preds = np.zeros((1, 6, 8), dtype=np.float64)
    preds[0, :, 0] = _logit(beta)
    preds[0, :, 1:4] = coords
    preds[0, :, 4] = energy_regression
    preds[0, :, 5:8] = position_regression

    features = np.array(
        [
            [
                [0.0, 0.0, 100.0, 2.0],
                [1.0, 0.0, 100.0, 3.0],
                [10.0, 0.0, 100.0, 5.0],
                [11.0, 0.0, 100.0, 5.0],
                [20.0, 0.0, 100.0, 4.0],
                [2.0, 0.0, 100.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    labels = np.array([[1, 1, 2, 2, 0, 1]], dtype=np.int32)
    mask = np.ones((1, 6), dtype=bool)
    layout = OCOutputLayout.from_config(
        {
            "output_dim": 8,
            "output_layout": {
                "beta": {"index": 0, "activation": "sigmoid"},
                "cluster_space": {"start": 1, "dim": 3},
                "regressions": [
                    {"name": "energy", "start": 4, "dim": 1, "source": "seed"},
                    {
                        "name": "position",
                        "start": 5,
                        "dim": 3,
                        "source": "seed",
                        "components": ["x", "y", "z"],
                    },
                ],
            },
        }
    )

    evaluation = evaluate_oc_padded(
        preds=preds,
        hit_object_id=labels,
        mask=mask,
        features=features,
        feature_names=["x", "y", "z", "energy"],
        tbeta=0.5,
        td=0.5,
        layout=layout,
        max_match_distance=10.0,
        min_energy_ratio=0.5,
        max_energy_ratio=2.0,
    )

    assert evaluation.count_summary()["frac_within_1"] == 0.0
    assert evaluation.seed_summary() == {
        "seed_efficiency": 1.0,
        "duplicate_rate": 0.25,
        "noise_seed_fake_rate": 0.25,
        "missed_truth": 0.0,
    }
    assert evaluation.matching_summary() == {
        "efficiency": 1.0,
        "fake_rate": 0.5,
        "purity": 0.5,
        "miss_rate": 0.0,
    }

    seed_classes = evaluation.seeds.sort("seed_rank")["seed_class"].to_list()
    assert seed_classes == ["matched_seed", "matched_seed", "noise_fake", "duplicate_fake"]

    event_features = features[0]
    expected_truth1_xyz = _energy_weighted_xyz(event_features, np.array([0, 1, 5]))
    expected_pred1_xyz = _energy_weighted_xyz(event_features, np.array([0, 1]))
    expected_truth2_xyz = _energy_weighted_xyz(event_features, np.array([2, 3]))
    expected_pred2_xyz = _energy_weighted_xyz(event_features, np.array([2, 3]))
    expected_distances = [
        np.linalg.norm(expected_pred1_xyz - expected_truth1_xyz),
        np.linalg.norm(expected_pred2_xyz - expected_truth2_xyz),
    ]

    matches = evaluation.matches.sort("truth_id")
    np.testing.assert_allclose(
        matches["centroid_distance"].to_numpy(),
        expected_distances,
        rtol=1e-6,
    )
    np.testing.assert_allclose(matches["energy_ratio"].to_numpy(), [5.0 / 6.0, 1.0])
    np.testing.assert_allclose(matches["energy_response"].to_numpy(), [5.0 / 6.0, 1.0])
    np.testing.assert_allclose(matches["model_energy_reco"].to_numpy(), [6.1, 9.5])

    predicted = evaluation.predicted.sort("pred_id")
    assert predicted["fake"].to_list() == [False, False, True, True]
    assert predicted["model_x_reco"].to_list()[0] == 0.0
    np.testing.assert_allclose(predicted["centroid_x_reco"].to_list()[0], expected_pred1_xyz[0])
    assert "centroid_x_reco" in predicted.columns
    assert "sum_x_reco" not in predicted.columns

    efficiency = binned_efficiency(evaluation.truth, "truth_energy", np.array([0.0, 5.0, 20.0]))
    fake_rate = binned_fake_rate(
        evaluation.predicted, "assigned_cluster_energy", np.array([0.0, 2.0, 20.0])
    )
    assert efficiency["efficiency"].to_list() == [0.0, 1.0]
    assert fake_rate["fake_rate"].to_list() == [1.0, 1.0 / 3.0]
    assert efficiency["efficiency_confidence_low"].to_list()[0] == 0.0
    assert efficiency["efficiency_confidence_high"].to_list()[0] == 0.0
    assert efficiency["efficiency_confidence_low"].to_list()[1] < 1.0
    assert fake_rate["fake_rate_confidence_low"].to_list()[1] < 1.0 / 3.0
    assert fake_rate["fake_rate_confidence_high"].to_list()[1] > 1.0 / 3.0


def test_binned_efficiency_uses_zero_interval_for_empty_bins() -> None:
    truth = pl.DataFrame({"truth_energy": [1.0], "matched": [True]})

    efficiency = binned_efficiency(truth, "truth_energy", np.array([0.0, 0.5, 2.0]))

    assert efficiency["n"].to_list() == [0, 1]
    assert efficiency["efficiency_confidence_low"].to_list()[0] == 0.0
    assert efficiency["efficiency_confidence_high"].to_list()[0] == 0.0


def test_grid_search_count_median_preserves_legacy_selector() -> None:
    beta = np.array([[0.9, 0.8, 0.7]], dtype=np.float64)
    coords = np.array([[[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]]], dtype=np.float64)
    labels = np.array([[1, 1, 2]], dtype=np.int32)
    mask = np.ones_like(labels, dtype=bool)
    progress_calls = []

    def record_progress(values):
        progress_calls.append(len(values))
        return values

    best, table = grid_search_thresholds(
        beta=beta,
        cluster_coords=coords,
        hit_object_id=labels,
        mask=mask,
        tbeta_values=np.array([0.5]),
        td_values=np.array([0.2, 20.0]),
        objective="count_median",
        progress=record_progress,
    )

    assert best["tbeta"] == 0.5
    assert best["td"] == 0.2
    assert best["median_abs_diff"] == 0.0
    assert "score" not in table.columns
    assert progress_calls == [1]


def test_grid_search_matched_f1_scores_perfect_threshold() -> None:
    beta = np.array([[0.9, 0.8]], dtype=np.float64)
    coords = np.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=np.float64)
    labels = np.array([[1, 2]], dtype=np.int32)
    mask = np.ones_like(labels, dtype=bool)
    features = np.array([[[10.0, 0.0, 100.0, 10.0], [30.0, 0.0, 100.0, 30.0]]])

    best, table = grid_search_thresholds(
        beta=beta,
        cluster_coords=coords,
        hit_object_id=labels,
        mask=mask,
        tbeta_values=np.array([0.5]),
        td_values=np.array([0.5]),
        objective="matched_f1",
        features=features,
        feature_names=["x", "y", "z", "energy"],
        max_match_distance=1.0,
        min_energy_ratio=0.5,
        max_energy_ratio=2.0,
        f1_epsilon=0.0,
    )

    row = table.row(0, named=True)
    assert best["score"] == 1.5
    assert row["epsilon_obj"] == 1.0
    assert row["p_obj"] == 1.0
    assert row["epsilon_E"] == 1.0
    assert row["p_E"] == 1.0
    assert row["f1_obj"] == 1.0
    assert row["f1_E"] == 1.0


def test_grid_search_matched_f1_weights_high_energy_matches() -> None:
    beta = np.array([[0.9, 0.4]], dtype=np.float64)
    coords = np.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=np.float64)
    labels = np.array([[1, 2]], dtype=np.int32)
    mask = np.ones_like(labels, dtype=bool)
    features = np.array([[[10.0, 0.0, 100.0, 1000.0], [30.0, 0.0, 100.0, 1.0]]])

    _, table = grid_search_thresholds(
        beta=beta,
        cluster_coords=coords,
        hit_object_id=labels,
        mask=mask,
        tbeta_values=np.array([0.5]),
        td_values=np.array([0.5]),
        objective="matched_f1",
        features=features,
        feature_names=["x", "y", "z", "energy"],
        max_match_distance=1.0,
        min_energy_ratio=0.5,
        max_energy_ratio=2.0,
        f1_epsilon=0.0,
    )

    row = table.row(0, named=True)
    assert row["epsilon_obj"] == 0.5
    assert row["epsilon_E"] > 0.98
    assert row["f1_E"] > row["f1_obj"]
    assert row["score"] > row["f1_obj"]


def test_grid_search_matched_f1_requires_evaluation_features() -> None:
    with pytest.raises(ValueError, match="requires unnormalized evaluation features"):
        grid_search_thresholds(
            beta=np.array([[0.9]], dtype=np.float64),
            cluster_coords=np.array([[[0.0, 0.0]]], dtype=np.float64),
            hit_object_id=np.array([[1]], dtype=np.int32),
            mask=np.array([[True]]),
            tbeta_values=np.array([0.5]),
            td_values=np.array([0.5]),
            objective="matched_f1",
        )


def test_hungarian_matching_can_penalize_energy_ratio_log_distance() -> None:
    centroid_distance = np.array(
        [
            [0.00, 0.10],
            [0.11, 0.12],
        ],
        dtype=np.float64,
    )
    energy_ratio = np.array(
        [
            [10.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    valid_match = np.ones_like(centroid_distance, dtype=bool)

    distance_only = _matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=1.0,
        algorithm="hungarian",
        hungarian_energy_ratio_log_weight=0.0,
    )
    energy_penalized = _matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=1.0,
        algorithm="hungarian",
        hungarian_energy_ratio_log_weight=1.0,
    )

    assert distance_only == [(0, 0), (1, 1)]
    assert energy_penalized == [(0, 1), (1, 0)]


def test_greedy_matching_resolves_pred_then_truth_conflicts() -> None:
    centroid_distance = np.array(
        [
            [0.20, 0.10, 0.40],
            [0.30, 0.20, 0.05],
        ],
        dtype=np.float64,
    )
    energy_ratio = np.array(
        [
            [1.20, 1.05, 1.00],
            [1.00, 1.00, 2.00],
        ],
        dtype=np.float64,
    )
    valid_match = np.ones_like(centroid_distance, dtype=bool)

    matches = _matched_positions(
        centroid_distance,
        energy_ratio,
        valid_match,
        max_match_distance=1.0,
        algorithm="greedy",
        hungarian_energy_ratio_log_weight=0.0,
    )

    assert matches == [(0, 1), (1, 2)]


def test_oc_evaluation_without_regressions_keeps_nullable_model_columns() -> None:
    beta = np.array([0.9, 0.8], dtype=np.float64)
    preds = np.zeros((1, 2, 3), dtype=np.float64)
    preds[0, :, 0] = _logit(beta)
    preds[0, :, 1:] = [[0.0, 0.0], [0.1, 0.0]]

    evaluation = evaluate_oc_padded(
        preds=preds,
        hit_object_id=np.array([[1, 1]], dtype=np.int32),
        mask=np.array([[True, True]]),
        features=np.array([[[0.0, 0.0, 100.0, 1.0], [1.0, 0.0, 100.0, 1.0]]]),
        feature_names=["x", "y", "z", "energy"],
        tbeta=0.5,
        td=0.5,
        max_match_distance=1.0,
    )

    assert evaluation.predicted["model_energy_reco"].to_list() == [None]
    assert evaluation.matches["model_x_reco"].to_list() == [None]


def test_oc_evaluation_requires_physical_hit_features() -> None:
    with pytest.raises(ValueError, match="requires unnormalized hit feature 'energy'"):
        evaluate_oc_padded(
            preds=np.array([[[3.0, 0.0, 0.0]]]),
            hit_object_id=np.array([[1]], dtype=np.int32),
            mask=np.array([[True]]),
            features=np.array([[[0.0, 0.0, 100.0]]]),
            feature_names=["x", "y", "z"],
            tbeta=0.5,
            td=0.5,
        )


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))


def _energy_weighted_xyz(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    xyz = features[indices, :3]
    energy = features[indices, 3]
    return np.average(xyz, weights=energy, axis=0)
