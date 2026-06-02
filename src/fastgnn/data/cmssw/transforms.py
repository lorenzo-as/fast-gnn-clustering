"""CMSSW event-level vertex transforms before truth construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from fastgnn.data.cmssw.preprocessing import HIT_ENERGY, HIT_PREFIX


def preprocess_vertices(raw_event: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Dispatch configured CMSSW vertex preprocessing for one raw event."""
    mode = str(cfg.get("preprocessing", "rechits_energy_threshold"))
    try:
        preprocess = PREPROCESSING_REGISTRY[mode]
    except KeyError as exc:
        available = ", ".join(sorted(PREPROCESSING_REGISTRY))
        raise ValueError(f"Unknown CMSSW preprocessing mode {mode!r}. Available: {available}") from exc
    return preprocess(raw_event, cfg)


def preprocess_rechits_energy_threshold(
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Threshold raw RecHits by energy while keeping hit-shaped branches aligned."""
    min_energy = cfg.get("hit_min_energy")
    if min_energy is None:
        return raw_event

    keep = raw_event[HIT_ENERGY] >= float(min_energy)
    filtered = {}
    n_hits = len(raw_event[HIT_ENERGY])
    for field, values in raw_event.items():
        if len(values) == n_hits and field.startswith(HIT_PREFIX):
            filtered[field] = values[keep]
        else:
            filtered[field] = values
    return filtered


def preprocess_econ_t_threshold(
        # * This should use E_T and correctly quantized inputs once available
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Placeholder for future ECON-T threshold emulation."""
    raise NotImplementedError("ECON-T threshold preprocessing is not implemented yet")


def preprocess_econ_t_best_choice(
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Placeholder for future ECON-T Best Choice preprocessing."""
    raise NotImplementedError("ECON-T Best Choice preprocessing is not implemented yet")


def preprocess_econ_t_super_trigger_cell(
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Placeholder for future ECON-T Super-Trigger Cell preprocessing."""
    raise NotImplementedError("ECON-T Super-Trigger Cell preprocessing is not implemented yet")


def preprocess_econ_t_autoencoder(
    raw_event: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Placeholder for future ECON-T AutoEncoder preprocessing."""
    raise NotImplementedError("ECON-T AutoEncoder preprocessing is not implemented yet")


PREPROCESSING_REGISTRY = {
    "rechits_energy_threshold": preprocess_rechits_energy_threshold,
    "econ_t_threshold": preprocess_econ_t_threshold,
    "econ_t_best_choice": preprocess_econ_t_best_choice,
    "econ_t_super_trigger_cell": preprocess_econ_t_super_trigger_cell,
    "econ_t_autoencoder": preprocess_econ_t_autoencoder,
}
