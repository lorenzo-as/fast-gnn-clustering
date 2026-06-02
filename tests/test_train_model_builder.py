import sys
from types import ModuleType
from typing import ClassVar

from omegaconf import OmegaConf
import pytest

from fastgnn.scripts.train import _build_gravnet_model, _validate_training_contract


class _FakeFactory:
    calls: ClassVar[list[dict]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    def create_keras_model(self, **kwargs):
        return {
            "factory": self.__class__.__name__,
            "factory_kwargs": self.kwargs,
            "create_kwargs": kwargs,
        }


class _FakeGravNetFactory(_FakeFactory):
    calls: ClassVar[list[dict]] = []


class _FakeQGravNetFactory(_FakeFactory):
    calls: ClassVar[list[dict]] = []


@pytest.fixture
def fake_qgravnet(monkeypatch):
    module = ModuleType("qgravnet")
    module.GravNetFactory = _FakeGravNetFactory
    module.QGravNetFactory = _FakeQGravNetFactory
    monkeypatch.setitem(sys.modules, "qgravnet", module)
    _FakeGravNetFactory.calls = []
    _FakeQGravNetFactory.calls = []


def _cfg(quantized: bool | None):
    model = {
        "name": "test_gravnet",
        "max_vertices": 16,
        "n_blocks": 1,
        "n_neighbours": 4,
        "n_dimensions": 2,
        "n_filters": 8,
        "n_propagate": 4,
        "n_postgn_dense_blocks": 1,
        "output_dim": 3,
        "output_head": "oc",
        "distance_metric": "l1",
        "neighbour_selector": "full",
        "dense_layer_dims": {"input_dense": 8},
        "gravnet_cfg": {"fix_coordinate_space": False},
        "selector_cfg": {},
    }
    if quantized is not None:
        model["quantized"] = quantized
    if quantized:
        model["quantization"] = {
            "dense_kernel_quantizer": "quantized_bits(8,0,1,alpha=1.0)",
            "dense_bias_quantizer": "quantized_bits(8,0,1,alpha=1.0)",
        }
    return OmegaConf.create({"model": model})


def test_build_gravnet_model_selects_plain_factory(fake_qgravnet) -> None:
    model = _build_gravnet_model(_cfg(quantized=False), n_features=4)

    assert model["factory"] == "_FakeGravNetFactory"
    assert _FakeQGravNetFactory.calls == []
    assert model["create_kwargs"] == {"n_vertices": 16, "n_features": 4}
    assert "dense_kernel_quantizer" not in model["factory_kwargs"]
    assert "dense_bias_quantizer" not in model["factory_kwargs"]


def test_build_gravnet_model_selects_quantized_factory(fake_qgravnet) -> None:
    model = _build_gravnet_model(_cfg(quantized=True), n_features=4)

    assert model["factory"] == "_FakeQGravNetFactory"
    assert _FakeGravNetFactory.calls == []
    assert model["factory_kwargs"]["dense_kernel_quantizer"].startswith("quantized_bits")
    assert model["factory_kwargs"]["dense_bias_quantizer"].startswith("quantized_bits")


def test_build_gravnet_model_requires_quantized_boolean(fake_qgravnet) -> None:
    with pytest.raises(ValueError, match=r"cfg\.model\.quantized"):
        _build_gravnet_model(_cfg(quantized=None), n_features=4)


class _FakeDataset:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def validate_feature_names(self, feature_names, *, normalize):
        self.calls.append((list(feature_names), normalize))
        if self.error is not None:
            raise self.error


class _FakeModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape


def _training_cfg():
    return OmegaConf.create(
        {
            "data": {"feature_names": ["x", "eta"]},
            "training": {"normalize_features": True},
        }
    )


def test_training_contract_validates_data_features_and_model_shape() -> None:
    train = _FakeDataset()
    val = _FakeDataset()

    _validate_training_contract(_FakeModel((None, 16, 2)), train, val, _training_cfg())

    assert train.calls == [(["x", "eta"], True)]
    assert val.calls == [(["x", "eta"], True)]


def test_training_contract_reports_dataset_feature_failure() -> None:
    train = _FakeDataset(KeyError("missing eta"))

    with pytest.raises(ValueError, match="train dataset feature validation failed"):
        _validate_training_contract(
            _FakeModel((None, 16, 2)), train, _FakeDataset(), _training_cfg()
        )


def test_training_contract_rejects_model_input_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="Model input feature dimension"):
        _validate_training_contract(
            _FakeModel((None, 16, 4)),
            _FakeDataset(),
            _FakeDataset(),
            _training_cfg(),
        )
