"""
Canonical ragged event dataset and training batch helpers.

The canonical storage format is an Awkward/Parquet table with one row per event
and nested record fields:

    event_id
    hits.{...}
    truth.hit_object_id
    truth.objects.{...}  # optional dataset-specific object/regression truth
    metadata.{...}

Stored events are ragged and contain only real detector hits. Padding is applied
only when building fixed-shape training batches:

    features:      padded with 0.0
    hit_object_id: 0 = real noise, >0 = real object, -1 = padded vertex
    mask:          true for real hits, false for padding
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import awkward as ak
import numpy as np
import yaml

from fastgnn.data.object_properties import compute_object_properties

Split = Literal["train", "val", "test"]

BATCH_KEYS = ["features", "hit_object_id"]
PADDING_OBJECT_ID = -1
NOISE_OBJECT_ID = 0
EMPTY_OBJECTS_SENTINEL_FIELD = "_empty"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a small YAML sidecar with stable key order."""
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def record_field_names(record: Any) -> list[str]:
    """Return field names for either an Awkward record or a plain mapping."""
    if hasattr(record, "fields"):
        return list(record.fields)
    return list(record.keys())


def _read_splits(dataset_dir: Path) -> dict[str, np.ndarray]:
    splits_path = dataset_dir / "splits.npz"
    if not splits_path.exists():
        return {}
    with np.load(splits_path) as data:
        return {name: np.asarray(data[name], dtype=np.int64) for name in data.files}


def _to_numpy(value: Any) -> Any:
    """Convert an Awkward scalar/list field to a NumPy-friendly value."""
    if isinstance(value, np.ndarray):
        return value
    try:
        python_value = ak.to_list(value)
        if isinstance(python_value, str | int | float | bool) or python_value is None:
            return python_value
    except Exception:
        pass
    try:
        return ak.to_numpy(value)
    except Exception:
        return np.asarray(ak.to_list(value))


def _field_group_from_record(record: Any) -> FieldGroup:
    """Recursively convert an Awkward record into nested FieldGroup objects."""
    data = {}
    for name in record.fields:
        if name == EMPTY_OBJECTS_SENTINEL_FIELD:
            continue
        value = record[name]
        if hasattr(value, "fields") and value.fields:
            data[name] = _field_group_from_record(value)
        else:
            data[name] = _to_numpy(value)
    return FieldGroup(data)


@dataclass(frozen=True)
class FieldGroup:
    """
    Lightweight immutable view over named event fields.

    Supports dot access (``event.hits.energy``), dict-style
    access (``event.hits["energy"]``), field listing, required-field checks, and
    conversion back to a shallow dictionary.
    """

    _data: dict[str, Any]

    @property
    def fields(self) -> list[str]:
        return list(self._data.keys())

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self._data]
        if missing:
            available = ", ".join(self.fields) or "<none>"
            raise KeyError(
                f"Missing required field(s): {', '.join(missing)}. Available fields: {available}"
            )

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)


@dataclass(frozen=True)
class EventRecord:
    """One canonical event with grouped hit, truth, and metadata fields."""

    event_id: int
    hits: FieldGroup
    truth: FieldGroup
    metadata: FieldGroup

    @property
    def n_hits(self) -> int:
        if "x" in self.hits:
            return len(self.hits.x)
        if self.hits.fields:
            return len(self.hits[self.hits.fields[0]])
        return 0

    @property
    def n_objects(self) -> int:
        self.truth.require("objects")
        objects = self.truth.objects
        if objects.fields:
            return len(objects[objects.fields[0]])
        return 0

    def features(self, names: Iterable[str]) -> np.ndarray:
        names = list(names)
        self.hits.require(*names)
        return np.stack([self.hits[name] for name in names], axis=1).astype(np.float32)

    def training_dict(
        self,
        feature_names: Iterable[str],
        payload_quantities: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, np.ndarray]:
        self.truth.require("hit_object_id")
        out = {
            "features": self.features(feature_names),
            "hit_object_id": self.truth.hit_object_id.astype(np.int32),
        }
        if payload_quantities:
            quantities = list(payload_quantities)
            if _payload_is_correction(quantities):
                out["payload_targets"] = _build_payload_correction_targets(
                    self.truth.hit_object_id, self.hits, quantities
                )
                out["payload_seeds"] = _build_payload_seeds(self.hits, quantities)
            else:
                out["payload_targets"] = _build_payload_targets(
                    self.truth.hit_object_id,
                    self.truth.objects,
                    quantities,
                )
        if "energy" in self.hits:
            out["_hit_energy"] = self.hits.energy.astype(np.float32)
        return out

    def object_index_for_hit(self, hit_index: int) -> int | None:
        """Return the 0-based truth-object index for a hit, or None for noise."""
        self.truth.require("hit_object_id")
        hit_object_id = int(self.truth.hit_object_id[hit_index])
        if hit_object_id <= NOISE_OBJECT_ID:
            return None
        return hit_object_id - 1

    def truth_object_for_hit(self, hit_index: int) -> FieldGroup | None:
        """Inspect dataset-specific object-level truth for one hit."""
        object_index = self.object_index_for_hit(hit_index)
        if object_index is None:
            return None
        self.truth.require("objects")
        return FieldGroup(
            {name: self.truth.objects[name][object_index] for name in self.truth.objects.fields}
        )


class CaloDataset:
    """Generic reader for canonical ragged calorimeter event datasets."""

    def __init__(
        self,
        path: str | Path,
        split: Split | None = None,
        max_events: int | None = None,
    ):
        self.path = Path(path)
        self.dataset_dir = self.path if self.path.is_dir() else self.path.parent
        self.events_path = self._resolve_events_path(self.path)
        self.metadata = _read_yaml(self.dataset_dir / "metadata.yaml")
        self.normalization = _read_yaml(self.dataset_dir / "normalization.yaml")
        self.splits = _read_splits(self.dataset_dir)

        events = ak.from_parquet(self.events_path)
        if split is not None:
            split_indices = self.splits.get(split)
            if split_indices is None:
                raise ValueError(
                    f"Split '{split}' is not defined in {self.dataset_dir / 'splits.npz'}"
                )
            events = events[split_indices]
        if max_events is not None:
            events = events[:max_events]

        self._events = events
        self.split_name = split

    @staticmethod
    def _resolve_events_path(path: Path) -> Path:
        if path.is_file():
            return path
        events_path = path / "events.parquet"
        if events_path.exists():
            return events_path
        shards = sorted(path.glob("events-*.parquet"))
        if shards:
            raise NotImplementedError(
                "Parquet shards are not loaded yet. Use events.parquet for now."
            )
        raise FileNotFoundError(f"Could not find events.parquet in {path}")

    @property
    def events(self) -> ak.Array:
        """Direct Awkward access for inspection, plotting, and custom analysis."""
        return self._events

    @property
    def fields(self) -> dict[str, list[str]]:
        if len(self._events) == 0:
            return {"hits": [], "truth": [], "truth.objects": [], "metadata": []}
        first = self._events[0]
        truth = first["truth"]
        return {
            "hits": list(first["hits"].fields),
            "truth": list(truth.fields),
            "truth.objects": [
                field
                for field in list(truth["objects"].fields)
                if field != EMPTY_OBJECTS_SENTINEL_FIELD
            ]
            if "objects" in truth.fields
            else [],
            "metadata": list(first["metadata"].fields) if "metadata" in first.fields else [],
        }

    @property
    def hit_features(self) -> list[str]:
        try:
            hit_features = self.metadata["hit_features"]
        except KeyError as exc:
            raise KeyError(
                f"{self.dataset_dir / 'metadata.yaml'} must define hit_features"
            ) from exc
        return list(hit_features)

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        split = self.split_name if self.split_name is not None else "all"
        return (
            f"{self.__class__.__name__}("
            f"events={len(self)}, "
            f"split={split!r}, "
            f"path={str(self.events_path)!r}, "
            f"hit_features={self.hit_features!r}"
            ")"
        )

    def __getitem__(self, idx: int) -> EventRecord:
        row = self._events[idx]
        metadata = row["metadata"] if "metadata" in row.fields else {}
        return EventRecord(
            event_id=int(row["event_id"]),
            hits=_field_group_from_record(row["hits"]),
            truth=_field_group_from_record(row["truth"]),
            metadata=_field_group_from_record(metadata)
            if hasattr(metadata, "fields")
            else FieldGroup({}),
        )

    def split(self, split: Split, max_events: int | None = None) -> CaloDataset:
        return CaloDataset(self.dataset_dir, split=split, max_events=max_events)

    def batches(
        self,
        max_vertices: int,
        batch_size: int,
        feature_names: Iterable[str] | None = None,
        payload_quantities: Iterable[Mapping[str, Any]] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        truncate: Literal["first", "energy_desc", "random"] = "first",
        normalize_features: bool = True,
        normalization_method: str | Mapping[str, Any] = "zscore",
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield padded mini-batches for training or evaluation loops."""
        feature_names = list(feature_names) if feature_names is not None else self.hit_features
        self.validate_feature_names(feature_names, normalize=normalize_features)
        collator = PadCollator(
            max_vertices=max_vertices,
            feature_names=feature_names,
            payload_quantities=payload_quantities,
            truncate=truncate,
            seed=seed,
            normalization=self.normalization if normalize_features and self.normalization else None,
            normalization_method=normalization_method,
        )
        return _batch_generator(
            dataset=self,
            collator=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def as_padded(
        self,
        max_vertices: int,
        feature_names: Iterable[str] | None = None,
        payload_quantities: Iterable[Mapping[str, Any]] | None = None,
        truncate: Literal["first", "energy_desc", "random"] = "first",
        normalize_features: bool = True,
        seed: int = 0,
        normalization_method: str | Mapping[str, Any] = "zscore",
    ) -> dict[str, np.ndarray]:
        """Return this whole dataset padded for model.predict or notebook analysis."""
        if len(self) == 0:
            raise ValueError("Cannot build padded data from an empty dataset")
        feature_names = list(feature_names) if feature_names is not None else self.hit_features
        self.validate_feature_names(feature_names, normalize=normalize_features)
        collator = PadCollator(
            max_vertices=max_vertices,
            feature_names=feature_names,
            payload_quantities=payload_quantities,
            truncate=truncate,
            seed=seed,
            normalization=self.normalization if normalize_features and self.normalization else None,
            normalization_method=normalization_method,
        )
        return collator([self[i] for i in range(len(self))])

    def summary(self) -> dict[str, Any]:
        n_hits = np.asarray(ak.num(self._events.truth.hit_object_id), dtype=np.int64)
        hit_object_ids = self._events.truth.hit_object_id
        n_objects = np.asarray(
            [
                len(set(np.asarray(ids, dtype=np.int64)) - {NOISE_OBJECT_ID})
                for ids in ak.to_list(hit_object_ids)
            ],
            dtype=np.int64,
        )
        noise_fraction = np.asarray(
            [
                float(np.mean(np.asarray(ids) == NOISE_OBJECT_ID)) if len(ids) else 0.0
                for ids in ak.to_list(hit_object_ids)
            ],
            dtype=np.float64,
        )

        def stats(values: np.ndarray) -> dict[str, float | int]:
            return {
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "std": float(np.std(values)) if len(values) else 0.0,
                "min": int(np.min(values)) if len(values) else 0,
                "max": int(np.max(values)) if len(values) else 0,
            }

        return {
            "n_events": len(self),
            "split": self.split_name,
            "hit_features": self.hit_features,
            "fields": self.fields,
            "hits_per_event": stats(n_hits),
            "objects_per_event": stats(n_objects),
            "noise_fraction": {
                "mean": float(np.mean(noise_fraction)) if len(noise_fraction) else 0.0,
                "std": float(np.std(noise_fraction)) if len(noise_fraction) else 0.0,
            },
        }

    def validate_feature_names(self, feature_names: Iterable[str], *, normalize: bool) -> None:
        """Validate requested model inputs against stored hit and normalization fields."""
        feature_names = list(feature_names)
        missing = sorted(set(feature_names) - set(self.fields["hits"]))
        if missing:
            raise KeyError(
                f"Requested feature(s) missing from {self.events_path}: {', '.join(missing)}. "
                f"Available hit fields: {', '.join(self.fields['hits'])}"
            )
        if normalize:
            _normalization_arrays(feature_names, self.normalization)


class PadCollator:
    """
    Collate ragged events into fixed-shape batches for hardware-aware training.

    Events can be EventRecord instances or materialized dicts with ``features``
    and ``hit_object_id``. Real noise hits keep label 0; padded vertices use -1.
    """

    def __init__(
        self,
        max_vertices: int,
        feature_names: Iterable[str] = ("x", "y", "z", "energy"),
        payload_quantities: Iterable[Mapping[str, Any]] | None = None,
        pad_value: float = 0.0,
        label_pad_value: int = PADDING_OBJECT_ID,
        truncate: Literal["first", "energy_desc", "random"] = "first",
        seed: int = 0,
        normalization: dict[str, Any] | None = None,
        normalization_method: str | Mapping[str, Any] = "zscore",
    ):
        self.max_vertices = max_vertices
        self.feature_names = list(feature_names)
        self.payload_quantities = list(payload_quantities or [])
        self.pad_value = pad_value
        self.label_pad_value = int(label_pad_value)
        self.truncate = truncate
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.normalization = normalization
        self.normalization_method = normalization_method

    def _as_training_dict(self, event: EventRecord | dict) -> dict[str, np.ndarray]:
        if isinstance(event, EventRecord):
            return event.training_dict(self.feature_names, self.payload_quantities)
        return event

    def __call__(self, events: list[EventRecord | dict]) -> dict[str, np.ndarray]:
        """
        Apply one padding policy to events: order hits, normalize features, pad arrays, build mask.
        """
        out = {}
        masks = []
        training_events = [self._as_training_dict(event) for event in events]
        orders = [self._select_order(event) for event in training_events]

        for key in BATCH_KEYS:
            arrays = []
            for event, order in zip(training_events, orders, strict=True):
                arr = event[key][order]
                if key == "features" and self.normalization is not None:
                    arr = apply_normalization(
                        arr,
                        self.feature_names,
                        self.normalization,
                        self.normalization_method,
                    )

                n_hits = arr.shape[0]
                if n_hits >= self.max_vertices:
                    arr = arr[: self.max_vertices]
                    if key == "hit_object_id":
                        arr = _compact_hit_object_ids(arr)
                    arrays.append(arr)
                    if key == "features":
                        masks.append(np.ones(self.max_vertices, dtype=bool))
                    continue

                if key == "hit_object_id":
                    arr = _compact_hit_object_ids(arr)
                    pad_value = self.label_pad_value
                else:
                    pad_value = self.pad_value
                pad_shape = (self.max_vertices - n_hits, *arr.shape[1:])
                pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
                arrays.append(np.concatenate([arr, pad], axis=0))
                if key == "features":
                    mask = np.zeros(self.max_vertices, dtype=bool)
                    mask[:n_hits] = True
                    masks.append(mask)
            out[key] = np.stack(arrays)

        if self.payload_quantities:
            # payload_targets is always present; payload_seeds only in correction mode.
            payload_keys = [
                k for k in ("payload_targets", "payload_seeds") if k in training_events[0]
            ]
            for payload_key in payload_keys:
                arrays = []
                for event, order in zip(training_events, orders, strict=True):
                    arr = event[payload_key][order]
                    n_hits = arr.shape[0]
                    if n_hits >= self.max_vertices:
                        arrays.append(arr[: self.max_vertices])
                        continue
                    pad_shape = (self.max_vertices - n_hits, *arr.shape[1:])
                    pad = np.full(pad_shape, self.pad_value, dtype=arr.dtype)
                    arrays.append(np.concatenate([arr, pad], axis=0))
                out[payload_key] = np.stack(arrays)

        out["mask"] = np.stack(masks)
        return out

    def _select_order(self, event: dict[str, np.ndarray]) -> np.ndarray:
        """Return hit ordering for truncation according to the configured policy."""
        n_hits = event["features"].shape[0]
        if n_hits <= self.max_vertices:
            return np.arange(n_hits)
        if self.truncate == "random":
            return self._rng.permutation(n_hits)
        if self.truncate != "energy_desc":
            return np.arange(n_hits)
        if "_hit_energy" in event:
            energy = event["_hit_energy"]
        elif "energy" in self.feature_names:
            energy = event["features"][:, self.feature_names.index("energy")]
        else:
            raise ValueError("truncate='energy_desc' requires stored hits.energy")
        return np.argsort(-energy)


def make_splits(n_events: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Create deterministic train/val/test event-index splits."""
    seed = int(cfg.get("seed", 0))
    train_frac = float(cfg.get("train_frac", 0.8))
    val_frac = float(cfg.get("val_frac", 0.1))
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError("train_frac and val_frac must be non-negative and sum to <= 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_events, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(train_frac * n_events)
    n_val = int(val_frac * n_events)
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
        "metadata": {
            "file": "splits.npz",
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_events - n_train - n_val,
        },
    }


def _compact_hit_object_ids(hit_object_id: np.ndarray) -> np.ndarray:
    """Compact positive IDs while preserving -1 padding and 0 real-noise."""
    hit_object_id = np.asarray(hit_object_id, dtype=np.int32)
    positive = np.unique(hit_object_id[hit_object_id > NOISE_OBJECT_ID])
    if len(positive) == 0:
        return hit_object_id

    compact = np.full_like(hit_object_id, PADDING_OBJECT_ID, dtype=np.int32)
    compact[hit_object_id == NOISE_OBJECT_ID] = NOISE_OBJECT_ID
    for new_id, old_id in enumerate(positive, start=1):
        compact[hit_object_id == old_id] = new_id
    return compact


def _build_payload_targets(
    hit_object_id: np.ndarray,
    truth_objects: FieldGroup,
    payload_quantities: Iterable[Mapping[str, Any]],
) -> np.ndarray:
    """Gather configured truth-object payloads into per-hit transformed targets."""
    hit_object_id = np.asarray(hit_object_id, dtype=np.int32)
    quantities = list(payload_quantities)
    target_dim = sum(_payload_quantity_dim(quantity) for quantity in quantities)
    targets = np.zeros((len(hit_object_id), target_dim), dtype=np.float32)
    signal = hit_object_id > NOISE_OBJECT_ID
    if not np.any(signal):
        return targets

    object_rows = hit_object_id[signal] - 1
    cursor = 0
    for quantity in quantities:
        field = str(quantity["field"])
        if field not in truth_objects:
            available = ", ".join(truth_objects.fields) or "<none>"
            raise KeyError(
                f"Missing payload truth.objects field {field!r}. Available fields: {available}"
            )
        raw = np.asarray(truth_objects[field], dtype=np.float32)[object_rows]
        transformed = _transform_payload_target(raw, quantity)
        dim = transformed.shape[-1]
        targets[signal, cursor : cursor + dim] = transformed
        cursor += dim
    return targets


def _payload_is_correction(payload_quantities: Iterable[Mapping[str, Any]]) -> bool:
    """Correction-mode quantities carry a per-hit ``seed`` field and a ``correction`` rule."""
    return any("correction" in quantity for quantity in payload_quantities)


def _build_payload_seeds(
    hits: Mapping[str, Any],
    payload_quantities: Iterable[Mapping[str, Any]],
) -> np.ndarray:
    """Per-hit seed payloads: each hit's own input feature named by ``seed``."""
    return np.stack(
        [np.asarray(hits[str(q["seed"])], dtype=np.float32) for q in payload_quantities],
        axis=1,
    )


def _build_payload_correction_targets(
    hit_object_id: np.ndarray,
    hits: Mapping[str, Any],
    payload_quantities: Iterable[Mapping[str, Any]],
) -> np.ndarray:
    """Broadcast each hit's truth-cluster aggregate (``target`` object property) to that hit."""
    hit_object_id = np.asarray(hit_object_id, dtype=np.int32)
    quantities = list(payload_quantities)
    targets = np.zeros((len(hit_object_id), len(quantities)), dtype=np.float32)
    signal = hit_object_id > NOISE_OBJECT_ID
    if not np.any(signal):
        return targets
    n_objects = int(hit_object_id[signal].max())
    properties = compute_object_properties(
        hits,
        hit_object_id,
        object_ids=np.arange(1, n_objects + 1),
        properties=tuple(dict.fromkeys(str(q["target"]) for q in quantities)),
    )
    object_rows = hit_object_id[signal] - 1
    for index, quantity in enumerate(quantities):
        targets[signal, index] = properties[str(quantity["target"])][object_rows]
    return targets


def _payload_quantity_dim(quantity: Mapping[str, Any]) -> int:
    transform = str(quantity.get("transform", "identity"))
    if transform in {"identity", "log", "scale"}:
        return 1
    if transform == "sin_cos":
        return 2
    raise ValueError(f"Unknown payload transform {transform!r}")


def _transform_payload_target(
    values: np.ndarray,
    quantity: Mapping[str, Any],
) -> np.ndarray:
    transform = str(quantity.get("transform", "identity"))
    values = np.asarray(values, dtype=np.float32)
    if transform == "identity":
        out = values[:, None]
    elif transform == "log":
        epsilon = float(quantity.get("epsilon", 0.0))
        out = np.log(values + epsilon)[:, None]
    elif transform == "scale":
        scale = float(quantity.get("scale", 1.0))
        if scale == 0:
            raise ValueError(f"Payload quantity {quantity.get('name')!r} scale must be non-zero")
        out = (values / scale)[:, None]
    elif transform == "sin_cos":
        out = np.stack([np.sin(values), np.cos(values)], axis=1)
    else:
        raise ValueError(f"Unknown payload transform {transform!r}")
    return out.astype(np.float32)


def robust_center_scale(values: np.ndarray, axis: int | None = None) -> tuple[Any, Any]:
    """Return median and IQR/1.349 (== sigma for a Gaussian), clamped away from zero."""
    median = np.median(values, axis=axis)
    q1, q3 = np.percentile(values, [25, 75], axis=axis)
    scale = (q3 - q1) / 1.349
    scale = np.where(scale < 1e-8, 1.0, scale)
    return median, scale


def compute_normalization(
    records: list[dict[str, Any]],
    feature_names: list[str],
    train_indices: np.ndarray,
) -> dict[str, Any]:
    """Compute per-feature mean and standard deviation over selected training events."""
    if len(train_indices) == 0:
        train_indices = np.arange(len(records), dtype=np.int64)
    features = []
    for idx in train_indices:
        hits = records[int(idx)]["hits"]
        features.append(
            np.stack([_to_numpy(hits[name]) for name in feature_names], axis=1).astype(np.float32)
        )
    flat = np.concatenate(features, axis=0)
    mean = flat.mean(axis=0)
    std = np.where(flat.std(axis=0) < 1e-8, 1.0, flat.std(axis=0))
    median, iqr = robust_center_scale(flat, axis=0)
    return {
        name: {
            "mean": float(mean[index]),
            "std": float(std[index]),
            "median": float(median[index]),
            "iqr": float(iqr[index]),
        }
        for index, name in enumerate(feature_names)
    }


def apply_normalization(
    features: np.ndarray,
    feature_names: list[str],
    normalization: dict[str, Any],
    method: str | Mapping[str, Any] = "zscore",
) -> np.ndarray:
    """Validate feature names and apply per-feature z-score or robust normalization.

    ``method`` is ``"zscore"`` (mean/std), ``"robust"`` (median/IQR), or a mapping
    ``{"method": <default>, "overrides": {feature: method}}`` for per-feature control.
    """
    center, scale = _normalization_arrays(feature_names, normalization, method)
    return ((features.astype(np.float32) - center) / scale).astype(np.float32)


def _feature_method(method: str | Mapping[str, Any], name: str) -> str:
    """Resolve the normalization method for a single feature."""
    if isinstance(method, str):
        resolved = method
    else:  # mapping-like (dict or OmegaConf DictConfig)
        overrides = method.get("overrides") or {}
        resolved = overrides.get(name, method.get("method", "zscore"))
    if resolved not in ("zscore", "robust"):
        raise ValueError(f"Unknown normalization method {resolved!r} for feature {name!r}")
    return resolved


_METHOD_KEYS = {"zscore": ("mean", "std"), "robust": ("median", "iqr")}


def _normalization_arrays(
    feature_names: list[str],
    normalization: dict[str, Any],
    method: str | Mapping[str, Any] = "zscore",
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature (center, scale) arrays for the requested method(s)."""
    missing = [name for name in feature_names if name not in normalization]
    if missing:
        raise ValueError(
            "Normalization is missing requested feature(s): "
            f"{', '.join(missing)}. Available: {', '.join(normalization)}"
        )
    centers, scales = [], []
    for name in feature_names:
        stats = normalization[name]
        center_key, scale_key = _METHOD_KEYS[_feature_method(method, name)]
        if center_key not in stats or scale_key not in stats:
            raise ValueError(
                f"Normalization for {name!r} lacks {center_key!r}/{scale_key!r}; "
                "regenerate normalization.yaml to use robust scaling."
            )
        centers.append(stats[center_key])
        scales.append(stats[scale_key])
    return np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def _batch_generator(
    dataset: CaloDataset,
    collator: PadCollator,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    """Iterate dataset indices and apply a collator to each mini-batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    collator._rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield collator([dataset[int(i)] for i in batch_indices])
