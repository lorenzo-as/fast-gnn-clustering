"""Utilities for interpreting flat Object Condensation model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OCRegressionSpec:
    """One optional per-hit regression slice in the flat OC output tensor."""

    name: str
    start: int
    dim: int
    source: str = "seed"
    target: str | None = None
    components: tuple[str, ...] = ()

    @property
    def stop(self) -> int:
        return self.start + self.dim


@dataclass(frozen=True)
class OCOutputLayout:
    """
    Flat tensor layout for ``output_head: oc`` models.

    The legacy layout is:
        output[..., 0]  beta logit
        output[..., 1:] cluster-space coordinates
    """

    output_dim: int
    beta_index: int = 0
    beta_activation: str = "sigmoid"
    cluster_start: int = 1
    cluster_dim: int = 0
    regressions: tuple[OCRegressionSpec, ...] = ()

    @classmethod
    def from_config(cls, model_cfg: Mapping[str, Any]) -> OCOutputLayout:
        output_dim = int(model_cfg["output_dim"])
        layout_cfg = model_cfg.get("output_layout")
        if layout_cfg is None:
            return cls.from_output_dim(output_dim)

        layout = _plain_mapping(layout_cfg)
        beta_cfg = _plain_mapping(layout.get("beta", {}))
        cluster_cfg = _plain_mapping(layout.get("cluster_space", {}))

        regressions = tuple(
            OCRegressionSpec(
                name=str(reg_cfg["name"]),
                start=int(reg_cfg["start"]),
                dim=int(reg_cfg["dim"]),
                source=str(reg_cfg.get("source", "seed")),
                target=None if reg_cfg.get("target") is None else str(reg_cfg["target"]),
                components=tuple(str(component) for component in reg_cfg.get("components", ())),
            )
            for reg_cfg in (_plain_mapping(item) for item in layout.get("regressions", ()))
        )

        out = cls(
            output_dim=output_dim,
            beta_index=int(beta_cfg.get("index", 0)),
            beta_activation=str(beta_cfg.get("activation", "sigmoid")),
            cluster_start=int(cluster_cfg.get("start", 1)),
            cluster_dim=int(cluster_cfg["dim"]),
            regressions=regressions,
        )
        out.validate()
        return out

    @classmethod
    def from_output_dim(cls, output_dim: int) -> OCOutputLayout:
        out = cls(output_dim=int(output_dim), cluster_dim=int(output_dim) - 1)
        out.validate()
        return out

    @property
    def cluster_stop(self) -> int:
        return self.cluster_start + self.cluster_dim

    @property
    def regression_dim(self) -> int:
        return sum(reg.dim for reg in self.regressions)

    def validate(self) -> None:
        if self.output_dim < 2:
            raise ValueError(f"OC output_dim must be at least 2, got {self.output_dim}")
        if self.beta_activation != "sigmoid":
            raise ValueError(
                f"Only beta activation 'sigmoid' is supported, got {self.beta_activation!r}"
            )
        if self.cluster_dim < 1:
            raise ValueError(f"cluster_space.dim must be at least 1, got {self.cluster_dim}")

        slices = [("beta", self.beta_index, self.beta_index + 1)]
        slices.append(("cluster_space", self.cluster_start, self.cluster_stop))
        slices.extend((reg.name, reg.start, reg.stop) for reg in self.regressions)

        for name, start, stop in slices:
            if start < 0 or stop > self.output_dim or start >= stop:
                raise ValueError(
                    f"Invalid OC output slice {name!r}: [{start}:{stop}] for "
                    f"output_dim={self.output_dim}"
                )

        covered: list[int] = []
        for name, start, stop in slices:
            values = list(range(start, stop))
            overlap = sorted(set(covered).intersection(values))
            if overlap:
                raise ValueError(f"OC output slice {name!r} overlaps indices {overlap}")
            covered.extend(values)

        expected = 1 + self.cluster_dim + self.regression_dim
        if expected != self.output_dim:
            raise ValueError(
                "OC output layout dimensions must sum to output_dim: "
                f"1 + {self.cluster_dim} + {self.regression_dim} = {expected}, "
                f"output_dim={self.output_dim}"
            )


@dataclass(frozen=True)
class OCOutputSlices:
    """Named views of the flat OC output tensor."""

    beta_logits: Any
    beta: Any
    cluster_coords: Any
    regressions: Mapping[str, Any]
    layout: OCOutputLayout


def split_oc_outputs(outputs: Any, layout: OCOutputLayout) -> OCOutputSlices:
    """Split NumPy or TensorFlow OC outputs according to ``layout``."""
    beta_logits = outputs[..., layout.beta_index]
    beta = _sigmoid(beta_logits)
    cluster_coords = outputs[..., layout.cluster_start : layout.cluster_stop]
    regressions = {reg.name: outputs[..., reg.start : reg.stop] for reg in layout.regressions}
    return OCOutputSlices(
        beta_logits=beta_logits,
        beta=beta,
        cluster_coords=cluster_coords,
        regressions=regressions,
        layout=layout,
    )


def _sigmoid(values: Any) -> Any:
    try:
        import tensorflow as tf

        if tf.is_tensor(values):
            return tf.sigmoid(values)
    except Exception:
        pass

    values = np.asarray(values)
    return np.where(values >= 0, 1 / (1 + np.exp(-values)), np.exp(values) / (1 + np.exp(values)))


def _plain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(value, resolve=True)
    except Exception:
        container = value
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    return container
