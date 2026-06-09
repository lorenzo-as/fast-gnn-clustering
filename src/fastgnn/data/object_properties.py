"""Object-level calorimeter hit aggregations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from fastgnn.geometry import xyz_to_eta_phi

PROPERTY_NAMES = (
    "sum_energy",
    "sum_et",
    "x_energy_weighted",
    "y_energy_weighted",
    "z_energy_weighted",
    "eta_energy_weighted",
    "phi_energy_weighted",
    "n_hits",
    "core_shower_length",
)


def compute_object_properties(
    hits: Mapping[str, Any],
    hit_object_id: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    object_ids: np.ndarray | None = None,
    properties: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute object properties from one object ID per hit.

    Noise hits have ID ``0`` and are ignored; positive values identify objects. Use :func:`compute_object_properties_from_links` when one hit may contribute to multiple objects.

    Args:
        hits: Mapping of hit-field names to arrays with shape ``(n_hits,)``.
            Requested properties determine the required fields. ``energy`` is
            always required. Energy-weighted position properties use ``x``,
            ``y``, ``z``, and either stored or derived ``eta``/``phi``.
        hit_object_id: Integer array with shape ``(n_hits,)``. Each entry is the
            object ID assigned to that hit. ID ``0`` is noise.
        weights: Optional per-hit weights with shape ``(n_hits,)``. Defaults to
            one for every hit.
        object_ids: Optional integer array with shape ``(n_objects,)`` defining
            output row order. Use this to include objects with no assigned hits.
            Defaults to the sorted positive IDs present in ``hit_object_id``.
        properties: Optional ordered subset of ``PROPERTY_NAMES``. Defaults to
            every supported property.

    Returns:
        Mapping from property name to an array with shape ``(n_objects,)``.
    """
    hit_object_id = np.asarray(hit_object_id, dtype=np.int64)
    weights = (
        np.ones_like(hit_object_id, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    if hit_object_id.ndim != 1:
        raise ValueError(f"hit_object_id must be 1D, got shape {hit_object_id.shape}")
    if weights.shape != hit_object_id.shape:
        raise ValueError(f"weights shape {weights.shape} does not match {hit_object_id.shape}")

    hit_indices = np.flatnonzero(hit_object_id > 0)
    return compute_object_properties_from_links(
        hits,
        hit_indices,
        hit_object_id[hit_indices],
        weights[hit_indices],
        object_ids=object_ids,
        properties=properties,
    )


def compute_object_properties_from_links(
    hits: Mapping[str, Any],
    hit_indices: np.ndarray,
    linked_object_ids: np.ndarray,
    weights: np.ndarray,
    *,
    object_ids: np.ndarray | None = None,
    properties: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute object properties from weighted hit-to-object links.

    Each link contributes independently to energy sums and energy-weighted
    positions. Duplicate ``(object ID, hit index)`` links count only once for
    ``n_hits`` and ``core_shower_length``.

    Args:
        hits: Mapping of hit-field names to arrays with shape ``(n_hits,)``.
            Requested properties determine the required fields as documented by
            :func:`compute_object_properties`.
        hit_indices: Integer array with shape ``(n_links,)``. Each entry selects
            the hit participating in one link.
        linked_object_ids: Integer array with shape ``(n_links,)``. Entry ``i``
            identifies the object linked to ``hit_indices[i]``. Unlike
            canonical ``hit_object_id``, ID ``0`` is allowed.
        weights: Link weights with shape ``(n_links,)``.
        object_ids: Optional integer array with shape ``(n_objects,)`` defining
            output row order. Every value in ``linked_object_ids`` must appear
            here. Use this to include objects without linked hits. Defaults to
            sorted unique values from ``linked_object_ids``.
        properties: Optional ordered subset of ``PROPERTY_NAMES``. Defaults to
            every supported property.

    Returns:
        Mapping from property name to an array with shape ``(n_objects,)``.
    """
    hit_indices = np.asarray(hit_indices, dtype=np.int64)
    linked_object_ids = np.asarray(linked_object_ids, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    property_names = _validate_properties(properties)

    if not (hit_indices.ndim == linked_object_ids.ndim == weights.ndim == 1):
        raise ValueError("hit_indices, linked_object_ids, and weights must be 1D")
    if not (len(hit_indices) == len(linked_object_ids) == len(weights)):
        raise ValueError("hit_indices, linked_object_ids, and weights must have equal lengths")

    object_ids = (
        np.unique(linked_object_ids)
        if object_ids is None
        else np.asarray(object_ids, dtype=np.int64)
    )
    if object_ids.ndim != 1 or len(np.unique(object_ids)) != len(object_ids):
        raise ValueError("object_ids must be one-dimensional and unique")

    obj_to_row = {int(obj): row for row, obj in enumerate(object_ids)}
    try:
        rows = np.asarray([obj_to_row[int(obj)] for obj in linked_object_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"linked_object_ids contains unknown object ID {exc.args[0]}") from exc

    energy = _field(hits, "energy")
    if np.any((hit_indices < 0) | (hit_indices >= len(energy))):
        raise IndexError("hit_indices contains an index outside the hit arrays")

    n_objects = len(object_ids)

    def sum_links(values: np.ndarray) -> np.ndarray:
        out = np.zeros(n_objects, dtype=np.float64)
        np.add.at(out, rows, values)
        return out.astype(np.float32)

    result = {}
    derived_eta_phi = None

    def geometry() -> tuple[np.ndarray, np.ndarray]:
        nonlocal derived_eta_phi
        if derived_eta_phi is None:
            derived_eta_phi = xyz_to_eta_phi(
                _field(hits, "x"),
                _field(hits, "y"),
                _field(hits, "z"),
            )
        return derived_eta_phi

    if "sum_energy" in property_names:
        result["sum_energy"] = sum_links(weights * energy[hit_indices])

    if "sum_et" in property_names:
        eta = None
        if "et" not in hits:
            eta = _field(hits, "eta") if "eta" in hits else geometry()[0]
        et = _field(hits, "et") if "et" in hits else energy / np.cosh(eta)
        linked_et = weights * et[hit_indices]
        result["sum_et"] = sum_links(linked_et)

    position_properties = {
        "x_energy_weighted",
        "y_energy_weighted",
        "z_energy_weighted",
        "eta_energy_weighted",
        "phi_energy_weighted",
    }
    if set(property_names) & position_properties:
        linked_energy = weights * energy[hit_indices]
        linked_sum_energy = sum_links(linked_energy)

        def energy_mean(values: np.ndarray, cos_values: np.ndarray | None = None) -> np.ndarray:
            out = np.full(n_objects, np.nan, dtype=np.float32)
            ok = linked_sum_energy != 0
            numerator = sum_links(linked_energy * values[hit_indices])
            if cos_values is None:
                out[ok] = numerator[ok] / linked_sum_energy[ok]
            else:
                denominator = sum_links(linked_energy * cos_values[hit_indices])
                out[ok] = np.arctan2(numerator[ok], denominator[ok])
            return out

        if "x_energy_weighted" in property_names:
            result["x_energy_weighted"] = energy_mean(_field(hits, "x"))
        if "y_energy_weighted" in property_names:
            result["y_energy_weighted"] = energy_mean(_field(hits, "y"))
        if "z_energy_weighted" in property_names:
            result["z_energy_weighted"] = energy_mean(_field(hits, "z"))
        if "eta_energy_weighted" in property_names:
            eta = _field(hits, "eta") if "eta" in hits else geometry()[0]
            result["eta_energy_weighted"] = energy_mean(eta)
        if "phi_energy_weighted" in property_names:
            phi = _field(hits, "phi") if "phi" in hits else geometry()[1]
            result["phi_energy_weighted"] = energy_mean(np.sin(phi), np.cos(phi))

    count_properties = {"n_hits", "core_shower_length"}
    if set(property_names) & count_properties:
        unique_pairs = np.unique(np.column_stack([rows, hit_indices]), axis=0)
    if "n_hits" in property_names:
        result["n_hits"] = np.zeros(n_objects, dtype=np.int32)
        np.add.at(result["n_hits"], unique_pairs[:, 0], 1)
    if "core_shower_length" in property_names:
        layer = _field(hits, "layer")
        result["core_shower_length"] = np.asarray(
            [
                _longest_consecutive_run(layer[unique_pairs[unique_pairs[:, 0] == row, 1]])
                for row in range(n_objects)
            ],
            dtype=np.int32,
        )
    return {name: result[name] for name in property_names}


def _validate_properties(properties: Iterable[str] | None) -> list[str]:
    properties = list(PROPERTY_NAMES if properties is None else properties)
    unknown = sorted(set(properties) - set(PROPERTY_NAMES))
    if unknown:
        raise ValueError(f"Unknown object properties: {', '.join(unknown)}")
    if len(properties) != len(set(properties)):
        raise ValueError("Object properties must not contain duplicates")
    return properties


def _field(hits: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        return np.asarray(hits[name], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing required hit field {name!r}") from exc


def _longest_consecutive_run(layers: np.ndarray) -> int:
    layers = np.unique(np.asarray(layers, dtype=np.int64))
    if len(layers) == 0:
        return 0
    gaps = np.where(np.diff(layers) != 1)[0] + 1
    return int(max(len(run) for run in np.split(layers, gaps)))
