import pytest
import tensorflow as tf

from fastgnn.training.schedules import scheduled_scalar_value
from fastgnn.training.trainer import _build_optimizer


def test_scheduled_scalar_value_defaults_to_scalar() -> None:
    assert scheduled_scalar_value(0.25, None, epoch=12) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 0.01),
        (9, 0.01),
        (10, 0.1),
        (29, 0.1),
        (30, 1.0),
        (100, 1.0),
    ],
)
def test_piecewise_constant_scalar_schedule(epoch: int, expected: float) -> None:
    schedule = {
        "type": "piecewise",
        "interpolation": "constant",
        "points": [[0, 0.01], [10, 0.1], [30, 1.0]],
    }

    assert scheduled_scalar_value(99.0, schedule, epoch) == pytest.approx(expected)


def test_piecewise_linear_scalar_schedule() -> None:
    schedule = {
        "type": "piecewise",
        "interpolation": "linear",
        "points": [[0, 0.1], [5, 0.01], [10, 1e-4]],
    }

    assert scheduled_scalar_value(99.0, schedule, epoch=5) == pytest.approx(0.01)
    assert scheduled_scalar_value(99.0, schedule, epoch=15) == pytest.approx(1e-4)
    assert scheduled_scalar_value(99.0, schedule, epoch=30) == pytest.approx(1e-4)


@pytest.mark.parametrize(
    "schedule",
    [
        {"type": "unknown", "points": [[0, 1.0]]},
        {"type": "piecewise", "interpolation": "unknown", "points": [[0, 1.0]]},
        {"type": "piecewise", "points": []},
        {"type": "piecewise", "points": [[1, 1.0]]},
        {"type": "piecewise", "points": [[0, 1.0], [0, 2.0]]},
    ],
)
def test_invalid_scalar_schedules_raise(schedule: dict) -> None:
    with pytest.raises(ValueError, match=r"schedule|Schedule"):
        scheduled_scalar_value(1.0, schedule, epoch=0)


def test_optimizer_accepts_tensorflow_learning_rate_schedule() -> None:
    optimizer = _build_optimizer(
        {
            "optimizer": "adam",
            "lr_schedule": {
                "class_name": "ExponentialDecay",
                "config": {
                    "initial_learning_rate": 1.0e-3,
                    "decay_steps": 10,
                    "decay_rate": 0.5,
                    "staircase": True,
                },
            },
        }
    )

    assert isinstance(
        optimizer._learning_rate,
        tf.keras.optimizers.schedules.ExponentialDecay,
    )
    assert float(optimizer.learning_rate) == pytest.approx(1.0e-3)
