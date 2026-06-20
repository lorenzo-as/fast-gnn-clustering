import awkward as ak
import numpy as np
import polars as pl

from fastgnn.data.cmssw.notebook_utils import collection_summary_row, object_collection_frame
from fastgnn.evaluation.prediction import (
    available_prediction_quantities,
    build_predicted_events,
    payload_quantities_from_config,
    prediction_eval_feature_names,
)


class _Metadata(dict):
    @property
    def fields(self):
        return list(self.keys())


class _Event:
    def __init__(self, event_id: int, source_event_id: int | None = None):
        self.event_id = event_id
        self.metadata = _Metadata()
        if source_event_id is not None:
            self.metadata["source_event_id"] = source_event_id


def test_payload_quantities_and_eval_feature_names() -> None:
    config = {
        "training": {
            "payload": {
                "quantities": [
                    {"seed": "et", "target": "et", "correction": "mul_exp_tanh"},
                    {"seed": "phi", "target": "phi", "correction": "additive_circular"},
                ]
            }
        }
    }
    payload_quantities = payload_quantities_from_config(config)
    assert len(payload_quantities) == 2
    assert prediction_eval_feature_names(payload_quantities) == [
        "x",
        "y",
        "z",
        "energy",
        "et",
        "phi",
    ]


def test_available_prediction_quantities_and_build_predicted_events() -> None:
    decoded_payload_features = {
        "et": np.array([[5.0, 2.0, 8.0]], dtype=np.float64),
        "energy": np.array([[9.0, 4.0, 12.0]], dtype=np.float64),
        "eta": np.array([[1.0, 1.1, 1.5]], dtype=np.float64),
    }
    aggregated, payload = available_prediction_quantities(decoded_payload_features)
    assert "sum_energy_reco" in aggregated
    assert payload == ["payload_energy_pred", "payload_et_pred", "payload_eta_pred"]

    dataset = [_Event(0, source_event_id=9001)]
    eval_inputs = {
        "features": np.array(
            [
                [
                    [10.0, 0.0, 0.0, 2.0, 4.0],
                    [10.2, 0.0, 0.0, 3.0, 5.0],
                    [20.0, 0.0, 0.0, 4.0, 8.0],
                ]
            ],
            dtype=np.float64,
        ),
        "mask": np.array([[True, True, True]], dtype=bool),
    }
    beta = np.array([[0.9, 0.3, 0.85]], dtype=np.float64)
    cluster_coords = np.array(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        dtype=np.float64,
    )

    rows = build_predicted_events(
        dataset,
        eval_inputs,
        beta,
        cluster_coords,
        0.5,
        0.5,
        eval_feature_names=["x", "y", "z", "energy", "et"],
        decoded_payload_features=decoded_payload_features,
        selected_prediction_quantities=["sum_energy_reco", "sum_et_reco", "payload_et_pred"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["source_event_id"] == 9001
    assert row["n_pred_clusters"] == 2
    assert np.isclose(row["event_sum_energy_reco"], 9.0)
    assert np.isclose(row["event_sum_et_reco"], 9.0)
    assert np.isclose(row["event_payload_et_sum"], 13.0)
    assert row["clusters"][0]["payload_et_pred"] == 5.0
    assert "payload_energy_pred" not in row["clusters"][0]


def test_cmssw_notebook_utils_collection_helpers() -> None:
    arrays = ak.Array(
        {
            "gen_pt": [[1.0, 2.0], []],
            "gen_eta": [[0.1], []],
            "gen_phi": [[0.2, 0.3], []],
        }
    )
    frame = object_collection_frame(
        arrays,
        0,
        "gen",
        [("pt", "pt"), ("eta", "eta"), ("phi", "phi")],
    )
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 2
    assert frame["eta"].to_list() == [0.1, None]

    summary = collection_summary_row(arrays, "gen")
    assert summary["events_with_entries"] == 1
    assert summary["total_entries"] == 2
