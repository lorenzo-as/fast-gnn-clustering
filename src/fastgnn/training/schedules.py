"""Small config-driven schedules for non-optimizer training parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any


def scheduled_scalar_value(
    default: float,
    schedule_cfg: Mapping[str, Any] | None,
    epoch: int,
) -> float:
    """Resolve an epoch-based scalar schedule while preserving scalar config defaults."""
    if schedule_cfg is None:
        return float(default)
    if epoch < 0:
        raise ValueError(f"Schedule epoch must be non-negative, got {epoch}")
    if schedule_cfg.get("type", "piecewise") != "piecewise":
        raise ValueError(f"Unknown scalar schedule type: {schedule_cfg.get('type')!r}")

    points = _schedule_points(schedule_cfg.get("points"))
    interpolation = schedule_cfg.get("interpolation", "constant")
    if interpolation == "constant":
        return _piecewise_constant_value(points, epoch)
    if interpolation == "linear":
        return _piecewise_linear_value(points, epoch)
    raise ValueError(f"Unknown scalar schedule interpolation: {interpolation!r}")


def _schedule_points(raw_points: Any) -> tuple[tuple[int, float], ...]:
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise ValueError("Scalar schedule points must be a sequence of [epoch, value] pairs")

    points: list[tuple[int, float]] = []
    for raw_point in raw_points:
        if (
            not isinstance(raw_point, Sequence)
            or isinstance(raw_point, (str, bytes))
            or len(raw_point) != 2
        ):
            raise ValueError("Each scalar schedule point must be an [epoch, value] pair")
        point_epoch = int(raw_point[0])
        if point_epoch != raw_point[0] or point_epoch < 0:
            raise ValueError(f"Schedule point epochs must be non-negative integers: {raw_point!r}")
        points.append((point_epoch, float(raw_point[1])))

    if not points:
        raise ValueError("Scalar schedule must contain at least one point")
    if points[0][0] != 0:
        raise ValueError("Scalar schedule must start at epoch 0")
    if any(left[0] >= right[0] for left, right in pairwise(points)):
        raise ValueError("Scalar schedule point epochs must be strictly increasing")
    return tuple(points)


def _piecewise_constant_value(points: tuple[tuple[int, float], ...], epoch: int) -> float:
    value = points[0][1]
    for point_epoch, point_value in points[1:]:
        if epoch < point_epoch:
            break
        value = point_value
    return value


def _piecewise_linear_value(points: tuple[tuple[int, float], ...], epoch: int) -> float:
    for left, right in pairwise(points):
        if epoch <= right[0]:
            fraction = (epoch - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    return points[-1][1]
