"""Evaluation utilities for trained clustering models."""

from fastgnn.evaluation.binned import binned_efficiency, binned_fake_rate
from fastgnn.evaluation.clustering import (
    count_clusters_from_labels,
    count_pred_objects,
    count_truth_objects,
)
from fastgnn.evaluation.payload import decode_payload_corrections, decode_payload_predictions
from fastgnn.evaluation.prediction import (
    available_prediction_quantities,
    build_predicted_events,
    decode_payload_outputs,
    oc_layout_from_model_config,
    payload_quantities_from_config,
    prediction_eval_feature_names,
)
from fastgnn.evaluation.tables import OCEvaluation, evaluate_oc_event, evaluate_oc_padded
from fastgnn.evaluation.thresholds import grid_search_thresholds

__all__ = [
    "OCEvaluation",
    "available_prediction_quantities",
    "binned_efficiency",
    "binned_fake_rate",
    "build_predicted_events",
    "count_clusters_from_labels",
    "count_pred_objects",
    "count_truth_objects",
    "decode_payload_corrections",
    "decode_payload_outputs",
    "decode_payload_predictions",
    "evaluate_oc_event",
    "evaluate_oc_padded",
    "grid_search_thresholds",
    "oc_layout_from_model_config",
    "payload_quantities_from_config",
    "prediction_eval_feature_names",
]
