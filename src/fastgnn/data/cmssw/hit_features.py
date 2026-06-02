"""CMSSW RecHit feature construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property

import numpy as np

from fastgnn.geometry import xyz_to_eta_phi

DEFAULT_LOG_FLOOR = 1.0e-8
BASE_FEATURES = ("x", "y", "z", "eta", "phi", "energy", "layer")
REQUIRED_FIELDS = ("x", "y", "z", "energy", "layer")


class HitFeatures:
    """Build CMSSW RecHit features.

    Base features are loaded in ``__init__``. To add a derived feature, add a
    new ``@cached_property`` method; it will automatically appear in
    ``available_features()`` and can be requested by ``build()``.
    """

    def __init__(
        self,
        base_hits: Mapping[str, np.ndarray],
        *,
        log_floor: float = DEFAULT_LOG_FLOOR,
    ) -> None:
        if log_floor <= 0:
            raise ValueError(f"log_floor must be positive, got {log_floor}")

        missing = set(REQUIRED_FIELDS) - set(base_hits)
        if missing:
            raise ValueError(f"Missing base hit field(s): {', '.join(sorted(missing))}")

        self.log_floor = log_floor
        self.x = np.asarray(base_hits["x"], dtype=np.float32)
        self.y = np.asarray(base_hits["y"], dtype=np.float32)
        self.z = np.asarray(base_hits["z"], dtype=np.float32)
        self.energy = np.asarray(base_hits["energy"], dtype=np.float32)
        self.layer = np.asarray(base_hits["layer"], dtype=np.int16)

        self._check_lengths()

        self.eta, self.phi = (
            arr.astype(np.float32) for arr in xyz_to_eta_phi(self.x, self.y, self.z)
        )

    def _check_lengths(self) -> None:
        lengths = {name: len(getattr(self, name)) for name in REQUIRED_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Hit fields have inconsistent lengths: {lengths}")

    @classmethod
    def available_features(cls) -> tuple[str, ...]:
        derived = tuple(
            name
            for name, value in vars(cls).items()
            if isinstance(value, cached_property)
        )
        return BASE_FEATURES + derived

    @cached_property
    def r(self) -> np.ndarray:
        return np.hypot(self.x, self.y)

    @cached_property
    def et(self) -> np.ndarray:
        return self.energy / np.cosh(self.eta)

    @cached_property
    def x_over_z(self) -> np.ndarray:
        return np.divide(self.x, self.z, out=np.zeros_like(self.x), where=self.z != 0)

    @cached_property
    def y_over_z(self) -> np.ndarray:
        return np.divide(self.y, self.z, out=np.zeros_like(self.y), where=self.z != 0)

    @cached_property
    def log_energy(self) -> np.ndarray:
        return np.log(np.maximum(self.energy, self.log_floor))

    @cached_property
    def log_et(self) -> np.ndarray:
        return np.log(np.maximum(self.et, self.log_floor))

    def build(self, feature_names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """Return selected features in the correct order, or all available features if ``names`` is None."""
        available = self.available_features()
        feature_names = list(available if feature_names is None else feature_names)

        unknown = sorted(set(feature_names) - set(available))
        if unknown:
            raise ValueError(
                f"Unknown CMSSW hit feature(s): {', '.join(unknown)}. "
                f"Available: {', '.join(available)}"
            )

        if len(feature_names) != len(set(feature_names)):
            raise ValueError("hit_features must not contain duplicates")

        return {name: getattr(self, name) for name in feature_names}
