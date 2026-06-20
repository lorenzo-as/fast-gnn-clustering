from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
import yaml

from fastgnn.data import CaloDataset
from fastgnn.data.cmssw_classical import (
    UNLABELED_OBJECT_ID,
    build_current_cmssw_features,
    convert_cmssw_classical_root,
    iter_cmssw_classical_events,
)


def test_iter_cmssw_classical_events_requires_explicit_zside(tmp_path: Path) -> None:
    root_path = _write_classical_root(tmp_path / "input.root")

    with pytest.raises(ValueError, match="requires explicit zside"):
        list(iter_cmssw_classical_events([root_path], {}))

    with pytest.raises(ValueError, match="does not support zside='auto'"):
        list(iter_cmssw_classical_events([root_path], {"zside": "auto"}))


def test_cmssw_classical_conversion_zside_positive(tmp_path: Path) -> None:
    root_path = _write_classical_root(tmp_path / "input.root")
    output_dir = tmp_path / "processed_pos"

    convert_cmssw_classical_root(
        [root_path],
        output_dir,
        config={
            "zside": 1,
            "hit_features": ["x", "y", "z", "energy", "eta", "phi", "layer", "log_energy"],
        },
    )

    ds = CaloDataset(output_dir, split="test")
    assert len(ds) == 1
    assert CaloDataset(output_dir, split="train").events.tolist() == []
    assert CaloDataset(output_dir, split="val").events.tolist() == []

    event = ds[0]
    np.testing.assert_array_equal(event.hits.tc_id, np.array([101, 103]))
    np.testing.assert_array_equal(event.hits.tc_zside, np.array([1, 1], dtype=np.int8))
    np.testing.assert_array_equal(event.hits.zside, np.array([1, 1], dtype=np.int8))
    np.testing.assert_allclose(event.hits.x, event.hits.tc_x)
    np.testing.assert_allclose(event.hits.energy, event.hits.tc_energy)
    np.testing.assert_allclose(event.hits.eta, np.array([1.10, 1.30], dtype=np.float32))
    assert "log_energy" in event.hits.fields
    assert event.truth.objects.fields == []
    np.testing.assert_array_equal(
        event.truth.hit_object_id,
        np.full(2, UNLABELED_OBJECT_ID, dtype=np.int32),
    )

    padded = ds.as_padded(max_vertices=4, normalize_features=False)
    assert padded["features"].shape == (1, 4, 8)
    np.testing.assert_array_equal(
        padded["hit_object_id"][0],
        np.array([UNLABELED_OBJECT_ID, UNLABELED_OBJECT_ID, -1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(padded["mask"][0], np.array([True, True, False, False]))

    metadata = yaml.safe_load((output_dir / "metadata.yaml").read_text())
    assert metadata["source"] == "cmssw_classical"
    assert metadata["task"] == "inference_only"
    assert metadata["labels"]["unlabeled_object_id"] == UNLABELED_OBJECT_ID
    assert metadata["required_fields"]["truth.objects"] == []
    assert set(metadata["required_fields"]["hits"]) >= {
        "tc_id",
        "tc_energy",
        "tc_eta",
        "x",
        "energy",
        "zside",
        "log_energy",
    }
    splits = np.load(output_dir / "splits.npz")
    assert splits["train"].tolist() == []
    assert splits["val"].tolist() == []
    assert splits["test"].tolist() == [0]

    validation = (output_dir / "validation.txt").read_text()
    assert "raw_trigger_cells: 5" in validation
    assert "kept_trigger_cells: 2" in validation
    assert (output_dir / "validation.csv").exists()


def test_cmssw_classical_conversion_zside_negative(tmp_path: Path) -> None:
    root_path = _write_classical_root(tmp_path / "input.root")
    output_dir = tmp_path / "processed_neg"

    convert_cmssw_classical_root(
        [root_path],
        output_dir,
        config={
            "zside": -1,
            "hit_features": ["x", "y", "z", "energy", "pt", "layer"],
        },
    )

    ds = CaloDataset(output_dir, split="test")
    assert len(ds) == 2
    np.testing.assert_array_equal(ds[0].hits.tc_id, np.array([102]))
    np.testing.assert_array_equal(ds[1].hits.tc_id, np.array([201, 202]))
    assert ds.hit_features == ["x", "y", "z", "energy", "pt", "layer"]


def test_cmssw_classical_store_aliases_only(tmp_path: Path) -> None:
    root_path = _write_classical_root(tmp_path / "input.root")
    output_dir = tmp_path / "processed_alias_only"

    convert_cmssw_classical_root(
        [root_path],
        output_dir,
        config={
            "zside": 1,
            "store_derived_features": False,
            "hit_features": ["x", "y", "z", "energy", "pt", "layer"],
        },
    )

    ds = CaloDataset(output_dir)
    assert "log_energy" not in ds.fields["hits"]
    assert {"tc_id", "tc_energy", "x", "pt"} <= set(ds.fields["hits"])

    with pytest.raises(ValueError, match="store_derived_features=false"):
        convert_cmssw_classical_root(
            [root_path],
            tmp_path / "bad",
            config={
                "zside": 1,
                "store_derived_features": False,
                "hit_features": ["x", "log_energy"],
            },
        )


def test_build_current_cmssw_features_matches_current_feature_names() -> None:
    features = build_current_cmssw_features(
        {
            "x": np.array([3.0], dtype=np.float32),
            "y": np.array([4.0], dtype=np.float32),
            "z": np.array([5.0], dtype=np.float32),
            "energy": np.array([10.0], dtype=np.float32),
            "layer": np.array([7], dtype=np.int16),
        }
    )

    assert "r" in features
    assert "log_et" in features
    np.testing.assert_allclose(features["r"], np.array([5.0], dtype=np.float32))


def _write_classical_root(path: Path) -> Path:
    arrays = {
        "tc_n": np.array([3, 2], dtype=np.int32),
        "tc_id": ak.Array([[101, 102, 103], [201, 202]]),
        "tc_subdet": ak.Array([[3, 3, 3], [3, 3]]),
        "tc_zside": ak.Array([[1, -1, 1], [-1, -1]]),
        "tc_layer": ak.Array([[1, 2, 3], [4, 5]]),
        "tc_waferu": ak.Array([[1, 2, 3], [4, 5]]),
        "tc_waferv": ak.Array([[2, 3, 4], [5, 6]]),
        "tc_wafertype": ak.Array([[0, 1, 0], [1, 0]]),
        "tc_cellu": ak.Array([[10, 11, 12], [13, 14]]),
        "tc_cellv": ak.Array([[20, 21, 22], [23, 24]]),
        "tc_pt": ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]]),
        "tc_mipPt": ak.Array([[0.1, 0.2, 0.3], [0.4, 0.5]]),
        "tc_energy": ak.Array([[10.0, 20.0, 30.0], [40.0, 50.0]]),
        "tc_eta": ak.Array([[1.10, -1.20, 1.30], [-1.40, -1.50]]),
        "tc_phi": ak.Array([[0.10, 0.20, 0.30], [0.40, 0.50]]),
        "tc_x": ak.Array([[10.0, 20.0, 30.0], [40.0, 50.0]]),
        "tc_y": ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]]),
        "tc_z": ak.Array([[320.0, -320.0, 320.0], [-320.0, -320.0]]),
    }
    with uproot.recreate(path) as root_file:
        root_file["l1tHGCalTriggerNtuplizer/HGCalTriggerNtuple"] = arrays
    return path
