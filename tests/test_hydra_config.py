from pathlib import Path

from fastgnn.scripts.convert_cmssw import (
    _compose_config as _compose_cmssw_config,
    convert_from_config,
)


def test_cmssw_conversion_config_composes() -> None:
    cfg = _compose_cmssw_config(
        [
            "input_files=[data/raw/cmssw/v0/test.root]",
            "output_dir=data/processed/cmssw/v0/test",
            "truth_min_object_energy=null",
        ]
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

    cfg = _compose_cmssw_config(
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
        ]
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
