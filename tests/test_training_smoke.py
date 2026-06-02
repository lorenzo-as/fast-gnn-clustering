from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import tensorflow as tf
import yaml

from fastgnn.data import CaloDataset
from fastgnn.data.base import PadCollator, compute_normalization
from fastgnn.data.cmssw.preprocessing import (
    CLUSTER_PREFIX,
    HIT_CLUSTER0,
    HIT_ENERGY,
    HIT_FRAC0,
    HIT_LAYER,
    HIT_NCLUSTERS,
    HIT_TIME,
    HIT_X,
    HIT_Y,
    HIT_Z,
    HIT_ZSIDE,
    build_truth,
)
from fastgnn.data.cmssw.transforms import (
    preprocess_rechits_energy_threshold,
    preprocess_vertices,
)
from fastgnn.training.objectcondensation_loss import calc_LV_Lbeta, get_clustering_np
from fastgnn.training.trainer import _build_optimizer, _train_step, _weight_oc_components


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    records = [
        _event_record(
            0,
            x=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            energy=np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32),
            hit_object_id=np.array([1, 1, 0, 2], dtype=np.int32),
            object_energy=np.array([10.0, 20.0], dtype=np.float32),
        ),
        _event_record(
            1,
            x=np.array([4.0, 5.0, 6.0], dtype=np.float32),
            energy=np.array([0.2, 4.0, 5.0], dtype=np.float32),
            hit_object_id=np.array([0, 1, 1], dtype=np.int32),
            object_energy=np.array([30.0], dtype=np.float32),
        ),
    ]
    ak.to_parquet(ak.Array(records), tmp_path / "events.parquet")
    (tmp_path / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "fastgnn-canonical-ragged-parquet",
                "format_version": 3,
                "hit_features": ["x", "y", "z", "energy"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "normalization.yaml").write_text(
        yaml.safe_dump(
            {
                "x": {"mean": 0.0, "std": 1.0},
                "y": {"mean": 0.0, "std": 1.0},
                "z": {"mean": 100.0, "std": 1.0},
                "energy": {"mean": 0.0, "std": 1.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        tmp_path / "splits.npz",
        train=np.array([0], dtype=np.int64),
        val=np.array([1], dtype=np.int64),
        test=np.array([], dtype=np.int64),
    )
    return tmp_path


def _event_record(
    event_id: int,
    x: np.ndarray,
    energy: np.ndarray,
    hit_object_id: np.ndarray,
    object_energy: np.ndarray,
) -> dict:
    n_hits = len(x)
    n_objects = len(object_energy)
    return {
        "event_id": event_id,
        "hits": {
            "x": x,
            "y": np.zeros(n_hits, dtype=np.float32),
            "z": np.full(n_hits, 100.0, dtype=np.float32),
            "energy": energy,
            "layer": np.arange(n_hits, dtype=np.int16),
            "time": np.zeros(n_hits, dtype=np.float32),
            "zside": np.ones(n_hits, dtype=np.int8),
            "cluster0": np.asarray(hit_object_id - 1, dtype=np.int32),
            "frac0": np.ones(n_hits, dtype=np.float32),
        },
        "truth": {
            "hit_object_id": hit_object_id,
            "objects": {
                "impact_eta": np.arange(n_objects, dtype=np.float32),
                "impact_phi": np.arange(n_objects, dtype=np.float32) + 0.5,
                "impact_energy": object_energy,
                "impact_pt": object_energy / 2.0,
                "sim_energy": object_energy + 1.0,
                "track_pdg_id": np.full(n_objects, 22, dtype=np.int32),
            },
        },
        "metadata": {"source": "test", "zside": 1},
    }


def test_cmssw_processed_dataset_loads(dataset_dir: Path) -> None:
    train_ds = CaloDataset(dataset_dir, split="train", max_events=2)
    val_ds = CaloDataset(dataset_dir, split="val", max_events=2)

    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert train_ds.hit_features == ["x", "y", "z", "energy"]
    assert len(train_ds.events) == 1
    event = train_ds[0]
    assert event.truth.hit_object_id.tolist() == [1, 1, 0, 2]
    assert event.object_index_for_hit(0) == 0
    assert event.object_index_for_hit(2) is None
    assert float(event.truth_object_for_hit(3).impact_energy) == 20.0


def test_compute_normalization_accepts_awkward_records() -> None:
    records = ak.Array(
        [
            _event_record(
                0,
                x=np.array([0.0, 1.0], dtype=np.float32),
                energy=np.array([2.0, 4.0], dtype=np.float32),
                hit_object_id=np.array([1, 0], dtype=np.int32),
                object_energy=np.array([10.0], dtype=np.float32),
            )
        ]
    )

    normalization = compute_normalization(
        [records[0]],
        ["x", "y", "z", "energy"],
        np.array([0], dtype=np.int64),
    )

    assert normalization == {
        "x": {"mean": 0.5, "std": 0.5},
        "y": {"mean": 0.0, "std": 1.0},
        "z": {"mean": 100.0, "std": 1.0},
        "energy": {"mean": 3.0, "std": 1.0},
    }


def test_get_clustering_np_skips_already_assigned_seeds() -> None:
    betas = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    coords = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.75, 0.0],
        ],
        dtype=np.float32,
    )

    clustering = get_clustering_np(betas, coords, tbeta=0.5, td=0.5)

    assert clustering.tolist() == [0, 0, 2]


def test_hit_features_are_required(tmp_path: Path) -> None:
    records = [
        _event_record(
            0,
            x=np.array([0.0], dtype=np.float32),
            energy=np.array([1.0], dtype=np.float32),
            hit_object_id=np.array([0], dtype=np.int32),
            object_energy=np.array([], dtype=np.float32),
        )
    ]
    ak.to_parquet(ak.Array(records), tmp_path / "events.parquet")
    (tmp_path / "metadata.yaml").write_text("format_version: 3\n", encoding="utf-8")

    with pytest.raises(KeyError, match="hit_features"):
        _ = CaloDataset(tmp_path).hit_features


def test_random_truncation_preserves_alignment() -> None:
    event = {
        "features": np.arange(20, dtype=np.float32).reshape(10, 2),
        "hit_object_id": np.array([0, 2, 2, 5, 0, 5, 8, 8, 8, 0], dtype=np.int32),
    }
    collator = PadCollator(
        max_vertices=5,
        feature_names=["x", "energy"],
        truncate="random",
        seed=123,
    )

    batch = collator([event])
    selected = (batch["features"][0, :, 0] // 2).astype(np.int32)
    expected = _compact(event["hit_object_id"][selected])

    assert batch["features"].shape == (1, 5, 2)
    assert batch["mask"].shape == (1, 5)
    np.testing.assert_array_equal(batch["hit_object_id"][0], expected)


def test_padding_uses_distinct_label_sentinel() -> None:
    event = {
        "features": np.arange(6, dtype=np.float32).reshape(3, 2),
        "hit_object_id": np.array([1, 0, 1], dtype=np.int32),
    }
    batch = PadCollator(max_vertices=5, feature_names=["x", "energy"])([event])

    np.testing.assert_array_equal(batch["mask"][0], [True, True, True, False, False])
    np.testing.assert_array_equal(batch["hit_object_id"][0], [1, 0, 1, -1, -1])


def test_dataset_as_padded_and_batches(dataset_dir: Path) -> None:
    dataset = CaloDataset(dataset_dir, split="train")

    padded = dataset.as_padded(max_vertices=6, normalize_features=False)
    np.testing.assert_array_equal(padded["mask"], [[True, True, True, True, False, False]])
    np.testing.assert_array_equal(padded["hit_object_id"], [[1, 1, 0, 2, -1, -1]])

    batch = next(
        dataset.batches(
            max_vertices=6,
            batch_size=1,
            shuffle=False,
            normalize_features=False,
        )
    )
    np.testing.assert_array_equal(batch["hit_object_id"], padded["hit_object_id"])


def test_oc_loss_handles_trailing_events_with_no_signal_hits() -> None:
    beta = tf.fill((9,), 0.1)
    coords = tf.zeros((9, 2), dtype=tf.float32)
    hit_object_id = tf.constant([1, 1, 2, 1, 2, 2, 0, 0, 0], dtype=tf.int32)
    batch = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=tf.int32)

    components = calc_LV_Lbeta(
        beta=beta,
        cluster_space_coords=coords,
        cluster_index_per_event=hit_object_id,
        batch=batch,
        return_components=True,
    )

    loss = components["L_V"] + components["L_beta"]
    assert np.isfinite(float(loss))


def test_qgravnet_oc_model_output_shape() -> None:
    from qgravnet import QGravNetFactory

    model = QGravNetFactory(
        n_blocks=1,
        n_neighbours=4,
        n_dimensions=2,
        n_filters=8,
        n_propagate=4,
        n_postgn_dense_blocks=1,
        output_dim=3,
        output_head="oc",
    ).create_keras_model(n_vertices=16, n_features=4)

    y = model(tf.zeros((2, 16, 4), dtype=tf.float32), training=False)

    assert tuple(y.shape) == (2, 16, 3)


def test_one_batch_training_step(dataset_dir: Path) -> None:
    from qgravnet import QGravNetFactory

    dataset = CaloDataset(dataset_dir, split="train", max_events=2)
    batch = next(
        dataset.batches(
            max_vertices=16,
            feature_names=["x", "y", "z", "energy"],
            truncate="random",
            seed=0,
            batch_size=1,
            shuffle=False,
            normalize_features=True,
        )
    )
    model = QGravNetFactory(
        n_blocks=1,
        n_neighbours=4,
        n_dimensions=2,
        n_filters=8,
        n_propagate=4,
        n_postgn_dense_blocks=1,
        output_dim=3,
        output_head="oc",
    ).create_keras_model(n_vertices=16, n_features=4)
    train_cfg = {
        "optimizer": "adam",
        "lr": 1.0e-3,
        "global_clipnorm": 1.0,
        "qmin": 1.0,
        "s_B": 0.1,
        "beta_stabilizing": "soft_q_scaling",
        "beta_term_option": "paper",
    }
    optimizer = _build_optimizer(train_cfg)

    loss, components = _train_step(model, batch, optimizer, train_cfg)

    assert np.isfinite(float(loss))
    assert "L_V" in components
    assert "L_beta" in components
    assert "L_total" in components
    assert float(components["L_total"]) == pytest.approx(
        float(components["L_V"] + components["L_beta"])
    )
    assert float(loss) == pytest.approx(float(components["L_total"]))


def test_oc_loss_weights_scale_saved_components() -> None:
    components = {
        "L_V": tf.constant(2.0, dtype=tf.float32),
        "L_V_attractive": tf.constant(0.5, dtype=tf.float32),
        "L_V_repulsive": tf.constant(1.5, dtype=tf.float32),
        "L_beta": tf.constant(3.0, dtype=tf.float32),
        "L_beta_noise": tf.constant(1.0, dtype=tf.float32),
        "L_beta_sig": tf.constant(2.0, dtype=tf.float32),
    }

    _weight_oc_components(
        components,
        {"loss_weights": {"L_V": 0.5, "L_beta": 2.0}},
    )

    assert float(components["L_V"]) == pytest.approx(1.0)
    assert float(components["L_V_attractive"]) == pytest.approx(0.25)
    assert float(components["L_V_repulsive"]) == pytest.approx(0.75)
    assert float(components["L_beta"]) == pytest.approx(6.0)
    assert float(components["L_beta_noise"]) == pytest.approx(2.0)
    assert float(components["L_beta_sig"]) == pytest.approx(4.0)
    assert float(components["L_total"]) == pytest.approx(7.0)


def test_truth_object_energy_threshold_removes_and_remaps_objects() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        cluster0=np.array([0, 1, 2, -1], dtype=np.int32),
        nclusters=np.array([1, 1, 1, 0], dtype=np.int16),
        object_energy=np.array([0.5, 2.0, 3.0], dtype=np.float32),
    )

    truth = build_truth(raw_event, {"truth_min_object_energy": 2.5})

    np.testing.assert_array_equal(truth["hit_object_id"], [0, 0, 1, 0])
    np.testing.assert_array_equal(truth["objects"]["impact_energy"], [3.0])
    np.testing.assert_array_equal(truth["objects"]["n_hits"], [1])
    np.testing.assert_allclose(truth["objects"]["visible_energy"], [1.0])
    np.testing.assert_array_equal(truth["objects"]["is_visible"], [True])


def test_truth_objects_record_visibility_without_removing_invisible_objects() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([1.5, 2.0, 0.5], dtype=np.float32),
        cluster0=np.array([0, 2, -1], dtype=np.int32),
        nclusters=np.array([1, 1, 0], dtype=np.int16),
        object_energy=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    truth = build_truth(raw_event, {})

    np.testing.assert_array_equal(truth["hit_object_id"], [1, 3, 0])
    np.testing.assert_array_equal(truth["objects"]["n_hits"], [1, 0, 1])
    np.testing.assert_allclose(truth["objects"]["visible_energy"], [1.5, 0.0, 2.0])
    np.testing.assert_array_equal(truth["objects"]["is_visible"], [True, False, True])


def test_truth_visible_energy_threshold_removes_and_remaps_objects() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([0.4, 1.0, 1.2, 0.7], dtype=np.float32),
        cluster0=np.array([0, 1, 1, 2], dtype=np.int32),
        nclusters=np.array([1, 1, 1, 1], dtype=np.int16),
        object_energy=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    truth = build_truth(raw_event, {"truth_min_visible_energy": 1.5})

    np.testing.assert_array_equal(truth["hit_object_id"], [0, 1, 1, 0])
    np.testing.assert_array_equal(truth["objects"]["impact_energy"], [20.0])
    np.testing.assert_array_equal(truth["objects"]["n_hits"], [2])
    np.testing.assert_allclose(truth["objects"]["visible_energy"], [2.2])
    np.testing.assert_array_equal(truth["objects"]["is_visible"], [True])


def test_truth_visible_energy_threshold_default_preserves_visibility_only() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([0.4, 1.0, 1.2, 0.7], dtype=np.float32),
        cluster0=np.array([0, 1, 1, 2], dtype=np.int32),
        nclusters=np.array([1, 1, 1, 1], dtype=np.int16),
        object_energy=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    truth = build_truth(raw_event, {"truth_min_visible_energy": None})

    np.testing.assert_array_equal(truth["hit_object_id"], [1, 2, 2, 3])
    np.testing.assert_array_equal(truth["objects"]["impact_energy"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(truth["objects"]["n_hits"], [1, 2, 1])
    np.testing.assert_allclose(truth["objects"]["visible_energy"], [0.4, 2.2, 0.7])
    np.testing.assert_array_equal(truth["objects"]["is_visible"], [True, True, True])


def test_truth_objects_default_to_configured_zside() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([1.5, 2.0], dtype=np.float32),
        cluster0=np.array([0, 2], dtype=np.int32),
        nclusters=np.array([1, 1], dtype=np.int16),
        object_energy=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )
    raw_event[f"{CLUSTER_PREFIX}_impact_eta"] = np.array([1.0, -2.0, 3.0], dtype=np.float32)

    truth = build_truth(raw_event, {"zside": 1})

    np.testing.assert_array_equal(truth["hit_object_id"], [1, 2])
    np.testing.assert_array_equal(truth["objects"]["impact_energy"], [10.0, 30.0])
    np.testing.assert_array_equal(truth["objects"]["n_hits"], [1, 1])


def test_truth_zside_filter_can_be_disabled_for_diagnostics() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([1.5, 2.0], dtype=np.float32),
        cluster0=np.array([0, 2], dtype=np.int32),
        nclusters=np.array([1, 1], dtype=np.int16),
        object_energy=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )
    raw_event[f"{CLUSTER_PREFIX}_impact_eta"] = np.array([1.0, -2.0, 3.0], dtype=np.float32)

    truth = build_truth(raw_event, {"zside": 1, "filter_truth_by_zside": False})

    np.testing.assert_array_equal(truth["hit_object_id"], [1, 3])
    np.testing.assert_array_equal(truth["objects"]["impact_energy"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(truth["objects"]["n_hits"], [1, 0, 1])


def test_hit_energy_threshold_keeps_hit_shaped_arrays_aligned() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([0.1, 1.5, 2.0], dtype=np.float32),
        cluster0=np.array([0, 1, 1], dtype=np.int32),
        nclusters=np.array([1, 1, 1], dtype=np.int16),
        object_energy=np.array([1.0, 2.0], dtype=np.float32),
    )

    filtered = preprocess_rechits_energy_threshold(raw_event, {"hit_min_energy": 1.0})

    np.testing.assert_array_equal(filtered[HIT_ENERGY], [1.5, 2.0])
    np.testing.assert_array_equal(filtered[HIT_CLUSTER0], [1, 1])
    np.testing.assert_array_equal(
        filtered[f"{CLUSTER_PREFIX}_impact_energy"],
        [1.0, 2.0],
    )


def test_preprocessing_selector_dispatches_rechit_threshold() -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([0.1, 2.0], dtype=np.float32),
        cluster0=np.array([0, 0], dtype=np.int32),
        nclusters=np.array([1, 1], dtype=np.int16),
        object_energy=np.array([1.0], dtype=np.float32),
    )

    processed = preprocess_vertices(
        raw_event,
        {"preprocessing": "rechits_energy_threshold", "hit_min_energy": 1.0},
    )

    np.testing.assert_array_equal(processed[HIT_ENERGY], [2.0])


@pytest.mark.parametrize(
    "mode",
    [
        "econ_t_threshold",
        "econ_t_best_choice",
        "econ_t_super_trigger_cell",
        "econ_t_autoencoder",
    ],
)
def test_econ_t_preprocessing_stubs_are_explicit(mode: str) -> None:
    raw_event = _raw_cmssw_event(
        hit_energy=np.array([1.0], dtype=np.float32),
        cluster0=np.array([0], dtype=np.int32),
        nclusters=np.array([1], dtype=np.int16),
        object_energy=np.array([1.0], dtype=np.float32),
    )

    with pytest.raises(NotImplementedError, match="ECON-T"):
        preprocess_vertices(raw_event, {"preprocessing": mode})


def _raw_cmssw_event(
    hit_energy: np.ndarray,
    cluster0: np.ndarray,
    nclusters: np.ndarray,
    object_energy: np.ndarray,
) -> dict[str, np.ndarray]:
    n_hits = len(hit_energy)
    n_objects = len(object_energy)
    return {
        HIT_X: np.arange(n_hits, dtype=np.float32),
        HIT_Y: np.zeros(n_hits, dtype=np.float32),
        HIT_Z: np.full(n_hits, 100.0, dtype=np.float32),
        HIT_ENERGY: hit_energy,
        HIT_LAYER: np.arange(n_hits, dtype=np.int16),
        HIT_ZSIDE: np.ones(n_hits, dtype=np.int8),
        HIT_TIME: np.zeros(n_hits, dtype=np.float32),
        HIT_NCLUSTERS: nclusters,
        HIT_CLUSTER0: cluster0,
        HIT_FRAC0: np.ones(n_hits, dtype=np.float32),
        f"{CLUSTER_PREFIX}_impact_eta": np.arange(n_objects, dtype=np.float32),
        f"{CLUSTER_PREFIX}_impact_phi": np.arange(n_objects, dtype=np.float32),
        f"{CLUSTER_PREFIX}_impact_energy": object_energy,
        f"{CLUSTER_PREFIX}_impact_pt": object_energy / 2.0,
        f"{CLUSTER_PREFIX}_simEnergy": object_energy + 1.0,
        f"{CLUSTER_PREFIX}_track_pdgId": np.full(n_objects, 22, dtype=np.int32),
    }


def _compact(hit_object_id: np.ndarray) -> np.ndarray:
    compact = np.full_like(hit_object_id, -1, dtype=np.int32)
    compact[hit_object_id == 0] = 0
    for new_id, old_id in enumerate(np.unique(hit_object_id[hit_object_id > 0]), start=1):
        compact[hit_object_id == old_id] = new_id
    return compact
