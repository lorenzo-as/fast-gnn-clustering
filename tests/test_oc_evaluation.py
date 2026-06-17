import numpy as np
import polars as pl
import pytest

from fastgnn.evaluation import (
    binned_efficiency,
    binned_fake_rate,
    decode_payload_predictions,
    evaluate_oc_padded,
    grid_search_thresholds,
)
from fastgnn.evaluation.matching import matched_positions as _matched_positions
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
    assert slices.payload is None


def test_output_layout_derives_generic_payload_regression_from_output_dim() -> None:
    cfg = {
        "output_dim": 9,
        "output_layout": {
            "beta": {"index": 0, "activation": "sigmoid"},
            "cluster_space": {"start": 1, "dim": 3},
            "payload": {"start": 4, "dim": 5},
        },
    }
    outputs = np.arange(18, dtype=np.float32).reshape(1, 2, 9)

    layout = OCOutputLayout.from_config(cfg)
    slices = split_oc_outputs(outputs, layout)

    assert layout.payload_dim == 5
    assert layout.payload is not None
    assert (layout.payload.name, layout.payload.start, layout.payload.dim) == ("payload", 4, 5)
    np.testing.assert_array_equal(slices.payload, outputs[..., 4:9])


def test_output_layout_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="must sum to output_dim"):
        OCOutputLayout.from_config(
            {
                "output_dim": 5,
                "output_layout": {
                    "beta": {"index": 0, "activation": "sigmoid"},
                    "cluster_space": {"start": 1, "dim": 3},
                },
            }
        )


def test_decode_payload_predictions_inverts_training_transforms() -> None:
    payload = np.array([[[np.log(5.0), 1.25, np.sin(0.4), np.cos(0.4), 0.8]]])
    decoded = decode_payload_predictions(
        payload,
        [
            {"name": "log_sum_et", "field": "sum_et", "transform": "log"},
            {"name": "eta", "field": "eta_energy_weighted", "transform": "identity"},
            {"name": "phi", "field": "phi_energy_weighted", "transform": "sin_cos"},
            {"name": "z", "field": "z_energy_weighted", "transform": "scale", "scale": 100.0},
        ],
    )

    assert decoded is not None
    np.testing.assert_allclose(decoded["et"], [[5.0]])
    np.testing.assert_allclose(decoded["eta"], [[1.25]])
    np.testing.assert_allclose(decoded["phi"], [[0.4]])
    np.testing.assert_allclose(decoded["z"], [[80.0]])
    np.testing.assert_allclose(decoded["energy"], [[5.0 * np.cosh(1.25)]])
    np.testing.assert_allclose(decoded["phi_sin_cos_norm"], [[1.0]])


def test_oc_evaluation_adds_seed_hit_payload_regression_columns() -> None:
    beta = np.array([0.90, 0.60, 0.85, 0.70], dtype=np.float64)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    payload_quantities = [
        {"name": "log_sum_et", "field": "sum_et", "transform": "log"},
        {"name": "eta", "field": "eta_energy_weighted", "transform": "identity"},
        {"name": "phi", "field": "phi_energy_weighted", "transform": "sin_cos"},
        {"name": "z", "field": "z_energy_weighted", "transform": "scale", "scale": 100.0},
    ]
    preds = np.zeros((1, 4, 9), dtype=np.float64)
    preds[0, :, 0] = _logit(beta)
    preds[0, :, 1:4] = coords
    preds[0, :, 4:9] = [
        [np.log(5.0), 0.0, 0.0, 1.0, 0.0],
        [np.log(99.0), 9.0, 0.0, 1.0, 9.0],
        [np.log(8.0), 0.0, 0.0, 1.0, 0.0],
        [np.log(99.0), 9.0, 0.0, 1.0, 9.0],
    ]
    features = np.array(
        [
            [
                [100.0, 0.0, 0.0, 2.0],
                [100.0, 0.0, 0.0, 3.0],
                [200.0, 0.0, 0.0, 4.0],
                [200.0, 0.0, 0.0, 4.0],
            ]
        ],
        dtype=np.float64,
    )
    labels = np.array([[1, 1, 2, 2]], dtype=np.int32)
    mask = np.ones((1, 4), dtype=bool)
    layout = OCOutputLayout.from_config(
        {
            "output_dim": 9,
            "output_layout": {
                "beta": {"index": 0, "activation": "sigmoid"},
                "cluster_space": {"start": 1, "dim": 3},
                "payload": {"start": 4, "dim": 5},
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
        payload_quantities=payload_quantities,
        max_match_distance=0.1,
        min_energy_ratio=0.5,
        max_energy_ratio=2.0,
        matching_reference="payload",
    )

    predicted = evaluation.predicted.sort("cluster_id_pred")
    np.testing.assert_allclose(predicted["payload_et_pred"].to_numpy(), [5.0, 8.0])
    np.testing.assert_allclose(predicted["payload_z_pred"].to_numpy(), [0.0, 0.0])

    matches = evaluation.matches.sort("object_id")
    assert matches["matching_reference"].to_list() == ["payload", "payload"]
    np.testing.assert_allclose(matches["payload_et_response"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(
        matches["payload_relative_et_residual"].to_numpy(),
        [0.0, 0.0],
        atol=1e-12,
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
    preds = np.zeros((1, 6, 4), dtype=np.float64)
    preds[0, :, 0] = _logit(beta)
    preds[0, :, 1:4] = coords

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
            "output_dim": 4,
            "output_layout": {
                "beta": {"index": 0, "activation": "sigmoid"},
                "cluster_space": {"start": 1, "dim": 3},
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

    matches = evaluation.matches.sort("object_id")
    np.testing.assert_allclose(
        matches["centroid_distance"].to_numpy(),
        expected_distances,
        rtol=1e-6,
    )
    np.testing.assert_allclose(matches["energy_ratio"].to_numpy(), [5.0 / 6.0, 1.0])
    np.testing.assert_allclose(matches["energy_response"].to_numpy(), [5.0 / 6.0, 1.0])

    predicted = evaluation.predicted.sort("cluster_id_pred")
    assert predicted["fake"].to_list() == [False, False, True, True]
    np.testing.assert_allclose(predicted["centroid_x_reco"].to_list()[0], expected_pred1_xyz[0])
    assert "centroid_x_reco" in predicted.columns
    assert "sum_x_reco" not in predicted.columns

    efficiency = binned_efficiency(evaluation.truth, "truth_energy", np.array([0.0, 5.0, 20.0]))
    fake_rate = binned_fake_rate(evaluation.predicted, "energy_pred", np.array([0.0, 2.0, 20.0]))
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


def test_grid_search_count_mean_selects_by_mean_abs_count_diff() -> None:
    beta = np.zeros((3, 10), dtype=np.float64)
    coords = np.zeros((3, 10, 2), dtype=np.float64)
    labels = np.zeros((3, 10), dtype=np.int32)
    mask = np.zeros((3, 10), dtype=bool)

    for event_idx in (0, 1):
        beta[event_idx, :2] = [0.9, 0.8]
        coords[event_idx, :2] = [[0.0, 0.0], [1.0, 0.0]]
        labels[event_idx, :2] = [1, 1]
        mask[event_idx, :2] = True

    beta[2, :10] = [0.9] * 9 + [0.4]
    coords[2, :10] = [[float(idx), 0.0] for idx in range(10)]
    labels[2, :10] = np.arange(1, 11)
    mask[2, :10] = True

    best, table = grid_search_thresholds(
        beta=beta,
        cluster_coords=coords,
        hit_object_id=labels,
        mask=mask,
        tbeta_values=np.array([0.5]),
        td_values=np.array([0.2, 2.0]),
        objective="count_mean",
    )

    assert best["td"] == 0.2
    assert best["mean_abs_diff"] == 1.0
    median_best = table.sort(
        by=["median_abs_diff", "mean_abs_diff", "frac_exact"],
        descending=[False, False, True],
    ).row(0, named=True)
    assert median_best["td"] == 2.0


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


def test_oc_evaluation_has_no_model_reco_columns() -> None:
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

    assert "model_energy_reco" not in evaluation.predicted.columns
    assert "model_x_reco" not in evaluation.matches.columns


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
