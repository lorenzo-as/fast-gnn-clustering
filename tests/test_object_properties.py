import awkward as ak
import numpy as np
import pytest
import yaml

from fastgnn.data.cmssw.dataset import (
    _branches_to_load,
    _convert_records,
    cmssw_event_to_record,
    convert_cmssw_root,
    iter_cmssw_events,
)
from fastgnn.data.cmssw.hit_features import HitFeatures
from fastgnn.data.cmssw.preprocessing import (
    CLUSTER_PREFIX,
    HIT_CLUSTERS,
    HIT_ENERGY,
    HIT_FRACS,
    HIT_LAYER,
    HIT_NCLUSTERS,
    HIT_TIME,
    HIT_X,
    HIT_Y,
    HIT_Z,
    HIT_ZSIDE,
    build_truth,
    cmssw_branches,
    cmssw_object_links,
)
from fastgnn.data.object_properties import (
    compute_object_properties,
    compute_object_properties_from_links,
)


def test_build_hit_features_computes_formulas() -> None:
    base_hits = {
        "x": np.array([3.0, 0.0], dtype=np.float32),
        "y": np.array([4.0, -1.0], dtype=np.float32),
        "z": np.array([10.0, -10.0], dtype=np.float32),
        "energy": np.array([5.0, 0.0], dtype=np.float32),
        "layer": np.array([1, 2], dtype=np.int16),
    }

    hits = HitFeatures(base_hits, log_floor=1.0e-4).build()

    np.testing.assert_allclose(hits["r"], [5.0, 1.0])
    np.testing.assert_allclose(hits["phi"], [np.arctan2(4.0, 3.0), -np.pi / 2])
    assert hits["eta"][0] > 0
    assert hits["eta"][1] < 0
    np.testing.assert_allclose(hits["et"], hits["energy"] / np.cosh(hits["eta"]))
    np.testing.assert_allclose(hits["x_over_z"], [0.3, 0.0])
    np.testing.assert_allclose(hits["y_over_z"], [0.4, 0.1])
    np.testing.assert_allclose(hits["sin_phi"], np.sin(hits["phi"]))
    np.testing.assert_allclose(hits["cos_phi"], np.cos(hits["phi"]))
    np.testing.assert_allclose(hits["log_energy"], [np.log(5.0), np.log(1.0e-4)])
    np.testing.assert_allclose(hits["log_et"][1], np.log(1.0e-4))


def test_build_hit_features_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown CMSSW hit feature"):
        HitFeatures(_base_hits()).build(["energy", "unknown"])


def test_build_hit_features_does_not_compute_unrequested_derived_fields(monkeypatch) -> None:
    def fail_if_called(*args):
        raise AssertionError("derived feature construction should be lazy")

    monkeypatch.setattr("fastgnn.data.cmssw.hit_features.np.hypot", fail_if_called)

    hits = HitFeatures(_base_hits()).build(["energy"])

    np.testing.assert_array_equal(hits["energy"], [1.0])


def test_compute_object_properties_uses_energy_weights_and_dense_object_ids() -> None:
    hits = {
        "energy": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "et": np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
        "x": np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        "y": np.array([2.0, 4.0, 8.0, 16.0, 32.0]),
        "eta": np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
        "phi": np.array([np.pi - 0.1, -np.pi + 0.1, np.pi, 1.0, 2.0]),
        "z": np.array([10.0, 20.0, 40.0, 80.0, 160.0]),
        "layer": np.array([1, 2, 4, 7, 8]),
    }

    props = compute_object_properties(
        hits,
        np.array([1, 1, 1, 2, 0]),
        object_ids=np.array([1, 2, 3]),
    )

    np.testing.assert_allclose(props["sum_energy"], [6.0, 4.0, 0.0])
    np.testing.assert_allclose(props["sum_et"], [30.0, 10.0, 0.0])
    np.testing.assert_allclose(props["x_energy_weighted"][:2], [17.0 / 6.0, 8.0])
    np.testing.assert_allclose(props["y_energy_weighted"][:2], [34.0 / 6.0, 16.0])
    np.testing.assert_allclose(props["eta_energy_weighted"][:2], [17.0 / 6.0, 8.0])
    np.testing.assert_allclose(props["z_energy_weighted"][:2], [170.0 / 6.0, 80.0])
    assert abs(abs(float(props["phi_energy_weighted"][0])) - np.pi) < 0.1
    np.testing.assert_array_equal(props["n_hits"], [3, 1, 0])
    np.testing.assert_array_equal(props["core_shower_length"], [2, 1, 0])
    assert np.isnan(props["eta_energy_weighted"][2])


def test_link_aggregation_counts_duplicate_hit_object_pair_once() -> None:
    hits = {
        "energy": np.array([4.0]),
        "et": np.array([2.0]),
        "eta": np.array([1.0]),
        "phi": np.array([0.5]),
        "z": np.array([10.0]),
        "layer": np.array([3]),
    }

    props = compute_object_properties_from_links(
        hits,
        hit_indices=np.array([0, 0]),
        linked_object_ids=np.array([7, 7]),
        weights=np.array([0.25, 0.75]),
        object_ids=np.array([7]),
        properties=("sum_energy", "n_hits"),
    )

    np.testing.assert_allclose(props["sum_energy"], [4.0])
    np.testing.assert_array_equal(props["n_hits"], [1])


def test_object_properties_require_layer_for_shower_length() -> None:
    hits = {
        "energy": np.array([1.0]),
        "x": np.array([1.0]),
        "y": np.array([0.0]),
        "z": np.array([10.0]),
    }

    with pytest.raises(KeyError, match="layer"):
        compute_object_properties(hits, np.array([1]))

    props = compute_object_properties(
        hits,
        np.array([1]),
        properties=("sum_energy",),
    )
    np.testing.assert_allclose(props["sum_energy"], [1.0])


@pytest.mark.parametrize(
    ("mode", "expected_hits", "expected_objects", "expected_weights"),
    [
        ("hard_assigned_full", [0, 1], [0, 1], [1.0, 1.0]),
        ("hard_assigned_fractional", [0, 1], [0, 1], [0.5, 0.75]),
        ("all_fractional", [0, 1, 1], [0, 1, 1], [0.5, 0.75, 0.25]),
    ],
)
def test_cmssw_object_links_isolates_aggregation_policy(
    mode: str,
    expected_hits: list[int],
    expected_objects: list[int],
    expected_weights: list[float],
) -> None:
    raw_event = _raw_links()

    hit_indices, linked_object_ids, weights = cmssw_object_links(
        raw_event,
        n_objects=2,
        cfg={"object_aggregation_mode": mode},
    )

    np.testing.assert_array_equal(hit_indices, expected_hits)
    np.testing.assert_array_equal(linked_object_ids, expected_objects)
    np.testing.assert_allclose(weights, expected_weights)


def test_cmssw_record_stores_selected_hits_and_raw_and_processed_properties() -> None:
    raw_event = _raw_event()

    record = cmssw_event_to_record(
        raw_event,
        event_id=4,
        cfg={
            "hit_features": ["energy", "eta"],
            "hit_min_energy": 1.0,
            "object_aggregation_mode": "hard_assigned_full",
        },
    )

    assert record["event_id"] == 4
    assert set(record["hits"].fields) == {
        "x",
        "y",
        "z",
        "energy",
        "eta",
        "layer",
        "time",
        "zside",
        "n_clusters",
        "cluster0",
        "frac0",
    }
    np.testing.assert_allclose(record["hits"]["energy"], [2.0])
    objects = record["truth"]["objects"]
    np.testing.assert_allclose(objects["sum_energy_raw"], [2.2, 0.0])
    np.testing.assert_allclose(objects["sum_energy"], [2.0, 0.0])
    np.testing.assert_array_equal(objects["n_hits_raw"], [2, 0])
    np.testing.assert_array_equal(objects["n_hits"], [1, 0])


def test_all_fractional_properties_keep_hard_cluster0_oc_labels() -> None:
    raw_event = _raw_event()
    raw_event[HIT_NCLUSTERS] = np.array([1, 2])
    raw_event[HIT_CLUSTERS[0]] = np.array([0, 1])
    raw_event[HIT_FRACS[0]] = np.array([0.5, 0.75])
    raw_event[HIT_CLUSTERS[1]] = np.array([-1, 0])
    raw_event[HIT_FRACS[1]] = np.array([0.0, 0.25])

    truth = build_truth(raw_event, {"object_aggregation_mode": "all_fractional"})

    np.testing.assert_array_equal(truth["hit_object_id"], [1, 2])
    np.testing.assert_allclose(truth["objects"]["sum_energy"], [0.6, 1.5])


def test_all_fractional_mode_loads_additional_root_association_branches() -> None:
    branches = cmssw_branches("")
    default = _branches_to_load({}, branches)
    all_fractional = _branches_to_load({"object_aggregation_mode": "all_fractional"}, branches)

    assert HIT_CLUSTERS[1] not in default
    assert HIT_FRACS[1] not in default
    assert set(HIT_CLUSTERS[1:]).issubset(all_fractional)
    assert set(HIT_FRACS[1:]).issubset(all_fractional)


def test_iter_cmssw_events_preprocesses_each_accepted_event_once(monkeypatch) -> None:
    chunk = ak.Array([_raw_event()])
    calls = []

    def fake_preprocess(raw_event, cfg):
        calls.append(raw_event)
        return raw_event

    monkeypatch.setattr(
        "fastgnn.data.cmssw.dataset.uproot.iterate", lambda *args, **kwargs: [chunk]
    )
    monkeypatch.setattr("fastgnn.data.cmssw.dataset._validate_chain", lambda *a, **k: None)
    monkeypatch.setattr("fastgnn.data.cmssw.dataset.preprocess_vertices", fake_preprocess)

    events = list(iter_cmssw_events(["input.root"], max_events=1))

    assert len(events) == 1
    assert len(calls) == 1


def test_conversion_metadata_uses_hit_features_and_retains_physical_fields(
    monkeypatch,
    tmp_path,
) -> None:
    raw_event = _raw_event()
    monkeypatch.setattr(
        "fastgnn.data.cmssw.dataset.iter_cmssw_events",
        lambda *args, **kwargs: [(raw_event, raw_event)],
    )

    output_dir = convert_cmssw_root(
        ["input.root"],
        tmp_path,
        config={"hit_features": ["eta"]},
    )

    metadata = yaml.safe_load((output_dir / "metadata.yaml").read_text())
    assert metadata["format_version"] == 4
    assert metadata["hit_features"] == ["x", "y", "z", "eta", "energy", "layer"]
    assert "x_energy_weighted" in metadata["required_fields"]["truth.objects"]
    assert "eta_energy_weighted" in metadata["required_fields"]["truth.objects"]
    assert "feature_names" not in metadata
    assert "materialized_hit_features" not in metadata


def test_parallel_conversion_preserves_file_order_and_reassigns_event_ids(monkeypatch) -> None:
    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, fn, args):
            assert self.max_workers == 2
            return [fn(arg) for arg in args]

    def fake_convert_file_records(args):
        path, *_ = args
        return [
            {"event_id": -1, "hits": {"x": [path]}, "truth": {"objects": {}}},
            {"event_id": -1, "hits": {"x": [path]}, "truth": {"objects": {}}},
        ]

    monkeypatch.setattr("fastgnn.data.cmssw.dataset.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        "fastgnn.data.cmssw.dataset._convert_file_records", fake_convert_file_records
    )

    records = _convert_records(["b.root", "a.root"], {}, ["x"], max_events=3, num_workers=2)

    assert [record["event_id"] for record in records] == [0, 1, 2]
    assert [record["hits"]["x"][0] for record in records] == ["b.root", "b.root", "a.root"]


def _base_hits() -> dict[str, np.ndarray]:
    return {
        "x": np.array([1.0]),
        "y": np.array([0.0]),
        "z": np.array([10.0]),
        "energy": np.array([1.0]),
        "layer": np.array([1]),
    }


def _raw_links() -> dict[str, np.ndarray]:
    raw_event = {
        HIT_X: np.array([1.0, 2.0]),
        HIT_Y: np.array([0.0, 0.0]),
        HIT_Z: np.array([10.0, 10.0]),
        HIT_ENERGY: np.array([1.0, 2.0]),
        HIT_LAYER: np.array([1, 2]),
        HIT_NCLUSTERS: np.array([1, 2]),
    }
    for slot in range(4):
        raw_event[HIT_CLUSTERS[slot]] = np.array([-1, -1])
        raw_event[HIT_FRACS[slot]] = np.array([0.0, 0.0])
    raw_event[HIT_CLUSTERS[0]] = np.array([0, 1])
    raw_event[HIT_FRACS[0]] = np.array([-0.5, 0.75])
    raw_event[HIT_CLUSTERS[1]] = np.array([-1, 1])
    raw_event[HIT_FRACS[1]] = np.array([0.0, -0.25])
    return raw_event


def _raw_event() -> dict[str, np.ndarray]:
    raw_event = {
        HIT_X: np.array([1.0, 2.0]),
        HIT_Y: np.array([0.0, 0.0]),
        HIT_Z: np.array([10.0, 10.0]),
        HIT_ENERGY: np.array([0.2, 2.0]),
        HIT_LAYER: np.array([1, 3]),
        HIT_ZSIDE: np.array([1, 1]),
        HIT_TIME: np.array([0.0, 0.0]),
        HIT_NCLUSTERS: np.array([1, 1]),
    }
    for slot in range(4):
        raw_event[HIT_CLUSTERS[slot]] = np.array([-1, -1])
        raw_event[HIT_FRACS[slot]] = np.array([0.0, 0.0])
    raw_event[HIT_CLUSTERS[0]] = np.array([0, 0])
    raw_event[HIT_FRACS[0]] = np.array([1.0, 1.0])
    n_objects = 2
    raw_event.update(
        {
            f"{CLUSTER_PREFIX}_impact_eta": np.array([1.0, 1.5]),
            f"{CLUSTER_PREFIX}_impact_phi": np.array([0.1, 0.2]),
            f"{CLUSTER_PREFIX}_impact_energy": np.array([10.0, 20.0]),
            f"{CLUSTER_PREFIX}_impact_pt": np.array([5.0, 10.0]),
            f"{CLUSTER_PREFIX}_simEnergy": np.array([11.0, 21.0]),
            f"{CLUSTER_PREFIX}_track_pdgId": np.full(n_objects, 22),
        }
    )
    return raw_event
