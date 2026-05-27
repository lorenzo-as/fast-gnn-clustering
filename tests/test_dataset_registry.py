from pathlib import Path

import polars as pl
import yaml

from fastgnn.datasets import DatasetRegistry


def test_registry_scans_yaml_into_table(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        "v1/ds_a",
        {
            "source": "cmssw",
            "n_events": 10000,
            "source_files": ["data/raw/cmssw/v1/NANO_10k_pt50-100.root"],
            "preprocessing": {
                "hit_min_energy": 0.2,
                "truth_min_object_energy": 1,
                "truth_min_visible_energy": 1,
            },
        },
    )

    table = DatasetRegistry(tmp_path).table()
    row = table.row(0, named=True)

    assert table.height == 1
    assert row["relative_path"] == "v1/ds_a"
    assert row["n_events"] == 10000
    assert row["feature_names"] is None
    assert row["train_frac"] is None
    assert row["val_frac"] is None
    assert row["preprocessing.hit_min_energy"] == 0.2
    assert row["preprocessing.truth_min_object_energy"] == 1
    assert row["preprocessing.truth_min_visible_energy"] == 1
    assert "path" not in table.columns
    assert "label" not in table.columns


def test_registry_table_supports_polars_filtering(tmp_path: Path) -> None:
    _write_dataset(
        tmp_path,
        "v1/a",
        {
            "source": "cmssw",
            "n_events": 100,
            "preprocessing": {"hit_min_energy": 0.2},
        },
    )
    _write_dataset(
        tmp_path,
        "v1/b",
        {
            "source": "cmssw",
            "n_events": 200,
            "preprocessing": {"hit_min_energy": 0.5},
        },
    )

    table = DatasetRegistry(tmp_path).table()
    selected = table.filter(
        (pl.col("preprocessing.hit_min_energy") >= 0.5) & (pl.col("n_events") >= 200)
    )

    assert selected["relative_path"].to_list() == ["v1/b"]


def test_registry_paths_convenience_filter(tmp_path: Path) -> None:
    low = _write_dataset(tmp_path, "v1/a", {"source": "cmssw", "n_events": 100})
    high = _write_dataset(tmp_path, "v1/b", {"source": "cmssw", "n_events": 200})

    registry = DatasetRegistry(tmp_path)

    assert registry.paths(n_events=100) == [low]
    assert registry.paths(n_events=200) == [high]


def _write_dataset(root: Path, relative: str, metadata: dict) -> Path:
    dataset_dir = root / relative
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "events.parquet").write_bytes(b"")
    (dataset_dir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_dir
