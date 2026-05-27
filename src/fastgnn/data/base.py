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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import awkward as ak
import numpy as np
import yaml

Split = Literal["train", "val", "test"]

BATCH_KEYS = ["features", "hit_object_id"]
PADDING_OBJECT_ID = -1
NOISE_OBJECT_ID = 0


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _read_splits(dataset_dir: Path) -> dict[str, np.ndarray]:
    splits_path = dataset_dir / "splits.npz"
    if not splits_path.exists():
        return {}
    with np.load(splits_path) as data:
        return {name: np.asarray(data[name], dtype=np.int64) for name in data.files}


def _to_numpy(value: Any) -> np.ndarray:
    """Convert an Awkward scalar/list field to a NumPy-friendly value."""
    if isinstance(value, np.ndarray):
        return value
    try:
        return ak.to_numpy(value)
    except Exception:
        return np.asarray(ak.to_list(value))


def _field_group_from_record(record: Any) -> "FieldGroup":
    """Recursively convert an Awkward record into nested FieldGroup objects."""
    data = {}
    for name in record.fields:
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
            return int(len(self.hits.x))
        if self.hits.fields:
            return int(len(self.hits[self.hits.fields[0]]))
        return 0

    @property
    def n_objects(self) -> int:
        self.truth.require("objects")
        objects = self.truth.objects
        if objects.fields:
            return int(len(objects[objects.fields[0]]))
        return 0

    def features(self, names: Iterable[str]) -> np.ndarray:
        names = list(names)
        self.hits.require(*names)
        return np.stack([self.hits[name] for name in names], axis=1).astype(np.float32)

    def training_dict(self, feature_names: Iterable[str]) -> dict[str, np.ndarray]:
        self.truth.require("hit_object_id")
        return {
            "features": self.features(feature_names),
            "hit_object_id": self.truth.hit_object_id.astype(np.int32),
        }

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
            "truth.objects": list(truth["objects"].fields) if "objects" in truth.fields else [],
            "metadata": list(first["metadata"].fields) if "metadata" in first.fields else [],
        }

    @property
    def feature_names(self) -> list[str]:
        try:
            feature_names = self.metadata["feature_names"]
        except KeyError as exc:
            raise KeyError(
                f"{self.dataset_dir / 'metadata.yaml'} must define feature_names"
            ) from exc
        return list(feature_names)

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        split = self.split_name if self.split_name is not None else "all"
        return (
            f"{self.__class__.__name__}("
            f"events={len(self)}, "
            f"split={split!r}, "
            f"path={str(self.events_path)!r}, "
            f"features={self.feature_names!r}"
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

    def split(self, split: Split, max_events: int | None = None) -> "CaloDataset":
        return CaloDataset(self.dataset_dir, split=split, max_events=max_events)

    def batches(
        self,
        max_vertices: int,
        batch_size: int,
        feature_names: Iterable[str] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        truncate: Literal["first", "energy_desc", "random"] = "first",
        normalize_features: bool = True,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield padded mini-batches for training or evaluation loops."""
        feature_names = list(feature_names) if feature_names is not None else self.feature_names
        collator = PadCollator(
            max_vertices=max_vertices,
            feature_names=feature_names,
            truncate=truncate,
            seed=seed,
            normalization=self.normalization if normalize_features and self.normalization else None,
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
        truncate: Literal["first", "energy_desc", "random"] = "first",
        normalize_features: bool = True,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """Return this whole dataset padded for model.predict or notebook analysis."""
        if len(self) == 0:
            raise ValueError("Cannot build padded data from an empty dataset")
        feature_names = list(feature_names) if feature_names is not None else self.feature_names
        collator = PadCollator(
            max_vertices=max_vertices,
            feature_names=feature_names,
            truncate=truncate,
            seed=seed,
            normalization=self.normalization if normalize_features and self.normalization else None,
        )
        return collator([self[i] for i in range(len(self))])

    def summary(self) -> dict[str, Any]:
        n_hits = np.asarray(ak.num(self._events.hits.x), dtype=np.int64)
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
            "feature_names": self.feature_names,
            "fields": self.fields,
            "hits_per_event": stats(n_hits),
            "objects_per_event": stats(n_objects),
            "noise_fraction": {
                "mean": float(np.mean(noise_fraction)) if len(noise_fraction) else 0.0,
                "std": float(np.std(noise_fraction)) if len(noise_fraction) else 0.0,
            },
        }


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
        pad_value: float = 0.0,
        label_pad_value: int = PADDING_OBJECT_ID,
        truncate: Literal["first", "energy_desc", "random"] = "first",
        seed: int = 0,
        normalization: dict[str, Any] | None = None,
    ):
        self.max_vertices = max_vertices
        self.feature_names = list(feature_names)
        self.pad_value = pad_value
        self.label_pad_value = int(label_pad_value)
        self.truncate = truncate
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.normalization = normalization

    def _as_training_dict(self, event: EventRecord | dict) -> dict[str, np.ndarray]:
        if isinstance(event, EventRecord):
            return event.training_dict(self.feature_names)
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
            for event, order in zip(training_events, orders):
                arr = event[key][order]
                if key == "features" and self.normalization is not None:
                    arr = _normalize_features(
                        arr,
                        self.feature_names,
                        self.normalization,
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
                pad_shape = (self.max_vertices - n_hits,) + arr.shape[1:]
                pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
                arrays.append(np.concatenate([arr, pad], axis=0))
                if key == "features":
                    mask = np.zeros(self.max_vertices, dtype=bool)
                    mask[:n_hits] = True
                    masks.append(mask)
            out[key] = np.stack(arrays)

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
        try:
            energy_col = self.feature_names.index("energy")
        except ValueError:
            raise ValueError(
                "truncate='energy_desc' requires 'energy' to be in feature_names. "
                f"Current feature_names: {self.feature_names}. "
                "Otherwise select truncate='first' or truncate='random'."
            )
        return np.argsort(-event["features"][:, energy_col])


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
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "feature_names": feature_names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
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


def _normalize_features(
    features: np.ndarray,
    feature_names: list[str],
    normalization: dict[str, Any],
) -> np.ndarray:
    """Validate feature names and apply z-score normalization."""
    norm_feature_names = list(normalization.get("feature_names", []))
    if norm_feature_names != feature_names:
        raise ValueError(
            "Normalization feature names do not match collator feature names: "
            f"{norm_feature_names} != {feature_names}"
        )
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    return ((features.astype(np.float32) - mean) / std).astype(np.float32)


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
