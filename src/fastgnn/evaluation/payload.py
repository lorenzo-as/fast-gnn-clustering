"""Payload regression-mode decoding and payload-derived match columns.

Three regression modes map model payload outputs to per-object physical
quantities (``et``/``eta``/``phi``/``z``/``energy``/``x``/``y``):

* ``aggregation`` -- no payload head; object properties come from energy-weighted
  aggregation of clustered hits (handled at match time, not here).
* ``direct`` -- each hit predicts a transformed target inverted by
  :func:`decode_payload_predictions`.
* ``correction`` -- each hit predicts a correction applied to its own raw seed
  feature, inverted by :func:`decode_payload_corrections`.

The mode is inferred from the configured quantities by :func:`payload_mode`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastgnn.evaluation._common import delta_phi
from fastgnn.evaluation.reco import delta_r, residual_block

PAYLOAD_RECO_KEYS = ("energy", "et", "x", "y", "z", "eta", "phi")

_STANDARD_ALIASES = {
    "sum_et": "et",
    "et": "et",
    "sum_energy": "energy",
    "energy": "energy",
    "eta_energy_weighted": "eta",
    "eta": "eta",
    "phi_energy_weighted": "phi",
    "phi": "phi",
    "z_energy_weighted": "z",
    "z": "z",
    "x_energy_weighted": "x",
    "x": "x",
    "y_energy_weighted": "y",
    "y": "y",
}


def payload_mode(payload_quantities: list[dict[str, Any]] | None) -> str:
    """Return ``"aggregation"``, ``"correction"``, or ``"direct"`` for a config."""
    if not payload_quantities:
        return "aggregation"
    if any("correction" in quantity for quantity in payload_quantities):
        return "correction"
    return "direct"


def decode_payload(
    payload: np.ndarray | None,
    payload_quantities: list[dict[str, Any]] | None,
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray] | None:
    """Decode model payload outputs into per-hit physical quantities by mode."""
    mode = payload_mode(payload_quantities)
    if mode == "correction":
        return decode_payload_corrections(payload, payload_quantities, features, feature_names)
    if mode == "direct":
        return decode_payload_predictions(payload, payload_quantities)
    return None


def decode_payload_predictions(
    payload: np.ndarray | None,
    payload_quantities: list[dict[str, Any]] | None,
) -> dict[str, np.ndarray] | None:
    """Invert configured payload target transforms into physical quantities."""
    if payload is None or not payload_quantities:
        return None
    payload = np.asarray(payload, dtype=np.float64)
    quantities = list(payload_quantities)
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for quantity in quantities:
        field = str(quantity["field"])
        name = str(quantity.get("name", field))
        transform = str(quantity.get("transform", "identity"))
        if transform == "identity":
            values = payload[..., cursor]
            cursor += 1
        elif transform == "log":
            epsilon = float(quantity.get("epsilon", 0.0))
            values = np.exp(payload[..., cursor]) - epsilon
            cursor += 1
        elif transform == "scale":
            values = payload[..., cursor] * float(quantity.get("scale", 1.0))
            cursor += 1
        elif transform == "sin_cos":
            sin_values = payload[..., cursor]
            cos_values = payload[..., cursor + 1]
            values = np.arctan2(sin_values, cos_values)
            out[f"{name}_sin"] = sin_values
            out[f"{name}_cos"] = cos_values
            out[f"{name}_sin_cos_norm"] = np.hypot(sin_values, cos_values)
            cursor += 2
        else:
            raise ValueError(f"Unknown payload transform {transform!r}")
        out[field] = values
        out[name] = values

    if cursor != payload.shape[-1]:
        raise ValueError(
            "Configured payload quantities do not consume model payload dimension: "
            f"consumed {cursor}, payload dim is {payload.shape[-1]}"
        )
    _add_standard_payload_aliases(out)
    return out


def decode_payload_corrections(
    payload: np.ndarray | None,
    payload_quantities: list[dict[str, Any]] | None,
    features: np.ndarray,
    feature_names: list[str],
    *,
    max_log_corr: float = 3.0,
) -> dict[str, np.ndarray] | None:
    """Apply per-hit corrections to each hit's own seed feature -> physical payloads.

    Mirrors the training-time correction: ``E_T = seed * exp(a*tanh r)`` and
    ``x = seed + s*r`` (wrapped for angles). The seed is the hit's own input
    feature named by ``seed``; ``features`` must contain those columns.
    """
    if payload is None or not payload_quantities:
        return None
    payload = np.asarray(payload, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for index, quantity in enumerate(payload_quantities):
        seed_name = str(quantity["seed"])
        target = str(quantity["target"])
        name = str(quantity.get("name", target))
        correction = str(quantity["correction"])
        if seed_name not in feature_names:
            raise ValueError(
                f"Correction seed feature {seed_name!r} is not in eval features {feature_names}"
            )
        seed = features[..., feature_names.index(seed_name)]
        r = payload[..., index]
        if correction == "mul_exp_tanh":
            values = seed * np.exp(float(quantity.get("a", max_log_corr)) * np.tanh(r))
        elif correction in ("additive", "additive_circular"):
            values = seed + float(quantity.get("s", 1.0)) * r
            if correction == "additive_circular":
                values = (values + np.pi) % (2 * np.pi) - np.pi
        else:
            raise ValueError(f"Unknown payload correction {correction!r}")
        out[target] = values
        out[name] = values
    _add_standard_payload_aliases(out)
    return out


def payload_seed_reco(
    payload_features: dict[str, np.ndarray] | None, seed_index: int
) -> dict[str, float | None]:
    """Decoded payload of one seed (condensation point) hit."""
    if payload_features is None:
        return {}
    return {
        key: _none_if_nan(float(payload_features[key][seed_index]))
        for key in PAYLOAD_RECO_KEYS
        if key in payload_features
    }


def payload_pred_columns(
    payload_features: dict[str, np.ndarray] | None, seed_index: int
) -> dict[str, float | None]:
    payload = payload_seed_reco(payload_features, seed_index)
    return {f"payload_{key}_pred": payload.get(key) for key in PAYLOAD_RECO_KEYS}


def payload_match_columns(
    payload_features: dict[str, np.ndarray] | None,
    seed_index: int,
    truth_ref_energy: float | None,
    truth_centroid: dict[str, float | None],
) -> dict[str, float | None]:
    payload = payload_seed_reco(payload_features, seed_index)
    if not payload:
        return {}
    out = payload_pred_columns(payload_features, seed_index)
    out.update(residual_block(payload, truth_centroid, truth_ref_energy, prefix="payload_"))
    dphi = delta_phi(truth_centroid["phi"], payload.get("phi"))
    out["payload_delta_r"] = delta_r(payload.get("eta"), truth_centroid["eta"], dphi)
    return out


def _none_if_nan(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _add_standard_payload_aliases(out: dict[str, np.ndarray]) -> None:
    for source, target in _STANDARD_ALIASES.items():
        if source in out and target not in out:
            out[target] = out[source]
    if "energy" not in out and "et" in out and "eta" in out:
        out["energy"] = out["et"] * np.cosh(out["eta"])
