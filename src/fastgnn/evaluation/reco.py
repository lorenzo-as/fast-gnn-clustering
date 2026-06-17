"""Per-object reconstruction and truth/pred residual helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from fastgnn.data.base import EventRecord
from fastgnn.data.object_properties import compute_object_properties
from fastgnn.evaluation._common import (
    delta_phi,
    none_if_nan,
    none_subtract,
    safe_none_ratio,
)

CENTROID_KEYS = ("x", "y", "z", "eta", "phi")

# Column names produced by ``residual_block`` for a given prefix, in output order.
RESIDUAL_BLOCK_FIELDS = (
    "energy_response",
    "et_response",
    "relative_energy_residual",
    "relative_et_residual",
    "eta_residual",
    "phi_residual",
    "z_residual",
    "relative_eta_residual",
    "relative_phi_residual",
    "relative_z_residual",
)


def reco_properties_by_assignment(
    assignments: np.ndarray, object_ids: np.ndarray, hits: dict[str, np.ndarray]
) -> dict[int, dict[str, float | None]]:
    props = compute_object_properties(
        hits,
        assignments,
        object_ids=object_ids,
        properties=(
            "sum_energy",
            "sum_et",
            "x_energy_weighted",
            "y_energy_weighted",
            "z_energy_weighted",
            "eta_energy_weighted",
            "phi_energy_weighted",
        ),
    )
    return {
        int(object_id): {
            "energy": float(props["sum_energy"][pos]),
            "et": float(props["sum_et"][pos]),
            "x": none_if_nan(props["x_energy_weighted"][pos]),
            "y": none_if_nan(props["y_energy_weighted"][pos]),
            "z": none_if_nan(props["z_energy_weighted"][pos]),
            "eta": none_if_nan(props["eta_energy_weighted"][pos]),
            "phi": none_if_nan(props["phi_energy_weighted"][pos]),
        }
        for pos, object_id in enumerate(object_ids)
    }


def residual_block(
    pred: dict[str, float | None],
    truth_centroid: dict[str, float | None],
    truth_ref_energy: float | None,
    *,
    prefix: str = "",
) -> dict[str, float | None]:
    """Response and residual columns shared by centroid and payload match rows.

    ``pred`` carries ``energy``/``et``/``eta``/``phi``/``z`` for the predicted
    cluster (or its payload-seed reco). Energy responses use ``truth_ref_energy``
    (impact energy when available); ET and position residuals use the
    energy-weighted truth centroid.
    """
    dphi = delta_phi(truth_centroid["phi"], pred.get("phi"))
    eta_residual = none_subtract(truth_centroid["eta"], pred.get("eta"))
    z_residual = none_subtract(truth_centroid["z"], pred.get("z"))
    return {
        f"{prefix}energy_response": safe_none_ratio(pred.get("energy"), truth_ref_energy),
        f"{prefix}et_response": safe_none_ratio(pred.get("et"), truth_centroid["et"]),
        f"{prefix}relative_energy_residual": safe_none_ratio(
            none_subtract(pred.get("energy"), truth_ref_energy), truth_ref_energy
        ),
        f"{prefix}relative_et_residual": safe_none_ratio(
            none_subtract(pred.get("et"), truth_centroid["et"]), truth_centroid["et"]
        ),
        f"{prefix}eta_residual": eta_residual,
        f"{prefix}phi_residual": dphi,
        f"{prefix}z_residual": z_residual,
        f"{prefix}relative_eta_residual": safe_none_ratio(eta_residual, truth_centroid["eta"]),
        f"{prefix}relative_phi_residual": safe_none_ratio(dphi, truth_centroid["phi"]),
        f"{prefix}relative_z_residual": safe_none_ratio(z_residual, truth_centroid["z"]),
    }


def delta_r(
    pred_eta: float | None,
    truth_eta: float | None,
    dphi: float | None,
) -> float | None:
    if pred_eta is None or truth_eta is None or dphi is None:
        return None
    return float(np.hypot(pred_eta - truth_eta, dphi))


def prefixed_centroid(
    prefix: str, centroid: dict[str, float | None], *, suffix: str | None = None
) -> dict[str, float | None]:
    return {
        f"{prefix}_{key}" + (f"_{suffix}" if suffix else ""): centroid[key] for key in CENTROID_KEYS
    }


def truth_object_props(event: EventRecord | None, truth_id: int) -> dict[str, Any]:
    if event is None or "objects" not in event.truth:
        return {}
    index = truth_id - 1
    props = {}
    for name in event.truth.objects.fields:
        values = event.truth.objects[name]
        if 0 <= index < len(values):
            value = values[index]
            props[name] = value.item() if hasattr(value, "item") else value
    return props


def nearest_truth_distance(
    centroids: dict[int, dict[str, float | None]],
) -> dict[int, float | None]:
    out = dict.fromkeys(centroids)
    for truth_id, centroid in centroids.items():
        distances = []
        for other_id, other in centroids.items():
            if other_id == truth_id:
                continue
            dphi = delta_phi(centroid["phi"], other["phi"])
            if centroid["eta"] is not None and other["eta"] is not None and dphi is not None:
                distances.append(float(np.hypot(centroid["eta"] - other["eta"], dphi)))
        if distances:
            out[truth_id] = min(distances)
    return out
