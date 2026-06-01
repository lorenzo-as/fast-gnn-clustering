from pathlib import Path

from hydra import compose, initialize_config_dir

from fastgnn.scripts.convert_cmssw import convert_from_config
from fastgnn.scripts.train import (
    _load_datasets,
    _resolve_output_dir,
)


def _compose_test_config(config_name: str, overrides: list[str] | None = None):
    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides or [])


def test_train_config_composes() -> None:
    cfg = _compose_test_config("train")

    assert cfg.data.name == "cmssw_processed"
    assert cfg.data.dataset_dir.startswith("data/processed/cmssw/")
    assert cfg.model.name == "gravnet_v0"
    assert cfg.model.quantized is False
    assert cfg.training.truncate == "energy_desc"
    assert cfg.training.loss_weights.L_V == 1.0
    assert cfg.training.loss_weights.L_beta == 1.0
    assert cfg.model.max_vertices == 1024


def test_quantized_model_config_composes() -> None:
    cfg = _compose_test_config("train", ["model=qgravnet_v0"])

    assert cfg.model.name == "qgravnet_v0"
    assert cfg.model.quantized is True
    assert cfg.model.quantization.dense_kernel_quantizer.startswith("quantized_bits")


def test_cmssw_conversion_config_composes() -> None:
    cfg = _compose_test_config(
        "convert/cmssw",
        [
            "input_files=[data/raw/cmssw/v0/test.root]",
            "output_dir=data/processed/cmssw/v0/test",
            "truth_min_object_energy=null",
        ],
    )

    assert cfg.input_files == ["data/raw/cmssw/v0/test.root"]
    assert cfg.output_dir == "data/processed/cmssw/v0/test"
    assert cfg.max_events is None
    assert cfg.overwrite is False
    assert cfg.zside == 1
    assert cfg.coordinate_system == "cartesian"
    assert cfg.preprocessing == "rechits_energy_threshold"
    assert cfg.hit_min_energy == 0.2
    assert cfg.truth_min_object_energy is None
    assert cfg.truth_min_visible_energy == 1
    assert cfg.truth_object_energy_field == "impact_energy"
    assert cfg.filter_truth_by_zside is True
    assert cfg.step_size is None
    assert cfg.train_frac == 0.9
    assert cfg.val_frac == 0.05
    assert cfg.seed == 0


def test_cmssw_conversion_cli_passes_config(monkeypatch) -> None:
    calls = []

    def fake_convert_cmssw_root(**kwargs):
        calls.append(kwargs)
        return kwargs["output_dir"]

    monkeypatch.setattr(
        "fastgnn.data.cmssw.convert_cmssw_root",
        fake_convert_cmssw_root,
    )

    cfg = _compose_test_config(
        "convert/cmssw",
        [
            "input_files=[data/raw/cmssw/v0/example.root]",
            "output_dir=data/processed/cmssw/v0/example",
            "max_events=100",
            "overwrite=true",
            "zside=-1",
            "coordinate_system=both",
            "hit_min_energy=0.5",
            "truth_min_object_energy=2.0",
            "truth_min_visible_energy=1.5",
            "filter_truth_by_zside=false",
            "step_size=10 MB",
            "train_frac=0.7",
            "val_frac=0.2",
            "seed=123",
        ],
    )

    output_dir = convert_from_config(cfg)

    project_root = Path(__file__).resolve().parents[1]
    assert output_dir == project_root / "data/processed/cmssw/v0/example"
    assert calls == [
        {
            "input_files": [project_root / "data/raw/cmssw/v0/example.root"],
            "output_dir": project_root / "data/processed/cmssw/v0/example",
            "config": {
                "zside": -1,
                "coordinate_system": "both",
                "preprocessing": "rechits_energy_threshold",
                "hit_min_energy": 0.5,
                "truth_min_object_energy": 2.0,
                "truth_min_visible_energy": 1.5,
                "truth_object_energy_field": "impact_energy",
                "filter_truth_by_zside": False,
                "step_size": "10 MB",
                "train_frac": 0.7,
                "val_frac": 0.2,
                "seed": 123,
            },
            "max_events": 100,
            "overwrite": True,
        }
    ]


def test_train_paths_resolve_from_project_root(monkeypatch, tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    cfg = _compose_test_config(
        "train",
        [
            "output_dir=outputs/test-run",
            "data.dataset_dir=data/processed/cmssw/v0/example",
        ],
    )

    assert _resolve_output_dir(cfg, ["output_dir=outputs/test-run"]) == (
        project_root / "outputs/test-run"
    )

    calls = []

    class FakeCaloDataset:
        def __init__(self, path, split=None, max_events=None):
            calls.append({"path": path, "split": split, "max_events": max_events})

    monkeypatch.setattr("fastgnn.data.CaloDataset", FakeCaloDataset)

    _load_datasets(cfg)

    assert calls == [
        {
            "path": project_root / "data/processed/cmssw/v0/example",
            "split": "train",
            "max_events": None,
        },
        {
            "path": project_root / "data/processed/cmssw/v0/example",
            "split": "val",
            "max_events": None,
        },
    ]
