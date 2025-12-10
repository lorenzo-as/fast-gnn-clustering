"""
Dataset:
Iiyama, Y., & Kieseler, J. (2020). Simulation of an imaging calorimeter to demonstrate
GarNet on FPGA (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3888910
"""

import urllib.request
from pathlib import Path
from typing import Union, Dict, Any

from matplotlib import pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def fetch_dataset(n_files: int, target_dir: Union[str, Path], overwrite: bool = False):
    """Download the Garnet dataset from Zenodo (https://doi.org/10.5281/zenodo.3888910).

    Parameters
    ----------
    n_files : int
        Number of files to download (events_0.h5, events_1.h5, ...).
    target_dir : str or Path
        Destination directory.
    overwrite : bool
        Whether to redownload existing files.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_files), desc="Downloading Garnet dataset"):
        url = f"https://zenodo.org/records/3888910/files/events_{i}.h5?download=1"
        out = target_dir / f"events_{i}.h5"

        if overwrite or not out.exists():
            urllib.request.urlretrieve(url, out)


def load_raw_dataset(n_files: int, data_dir: Union[str, Path]):
    """Load and merge dataset arrays from multiple HDF5 files.

    Returns
    -------
    X_hits, X_size, y_energy, y_pid : np.ndarray
    """
    data_dir = Path(data_dir)

    hits, size, energy, pid = [], [], [], []

    for i in range(n_files):
        fname = data_dir / f"events_{i}.h5"
        if not fname.exists():
            print(f"Warning: missing {fname}, skipping.")
            continue

        with h5py.File(fname, "r") as f:
            hits.append(f["cluster"][:])  # type: ignore[index]
            size.append(f["size"][:])  # type: ignore[index]
            energy.append(f["truth_energy"][:])  # type: ignore[index]
            pid.append(f["truth_pid"][:])  # type: ignore[index]

    return (
        np.concatenate(hits),
        np.concatenate(size),
        np.concatenate(energy),
        np.concatenate(pid),
    )


def compute_hits_normalization(X_hits: np.ndarray):
    """
    Compute per-coordinate mean/std using only non-padded hits.
    A hit row is considered padded if all features are zero.
    """
    flat = X_hits.reshape(-1, X_hits.shape[-1])

    mask = np.any(flat != 0, axis=1)
    flat = flat[mask]

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    return mean, std


def apply_hits_normalization(X_hits: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Apply normalization only to non-padded hit rows.
    Padded rows (all zeros) stay zeros.
    """
    mask = np.any(X_hits != 0, axis=-1, keepdims=True)  # shape (N_events, N_hits, 1)
    X_norm = (X_hits - mean) / std

    return np.where(mask, X_norm, 0.0)


def split_dataset(X_hits, X_size, y_energy, y_pid, test_size=0.25):
    """Wrapper for sklearn train_test_split with fixed seed."""
    return train_test_split(
        X_hits, X_size, y_energy, y_pid, test_size=test_size, random_state=0
    )


def save_processed(data: Dict[str, Any], filename: Union[str, Path]):
    """Save processed data to a single HDF5 file."""
    filename = Path(filename)
    with h5py.File(filename, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression="gzip")


def load_processed(filename: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load processed data from HDF5 file."""
    filename = Path(filename)
    with h5py.File(filename, "r") as f:
        return {key: f[key][:] for key in f.keys()}  # type: ignore[index]


def plot_data_distributions(
    X_hits, mask_zero=True, raw=True, ax=None, title=None, label=None, **hist_kwargs
):
    """
    Plot distributions of hit coordinates (x, y, z, e).

    Parameters
    ----------
    X_hits : ndarray, shape (N_events, N_hits, 4)
        Hit array (raw or normalized).
    mask_zero : bool
        Remove padded zeros. Only applied when raw=True.
    raw : bool
        If True: treat hits as raw detector coordinates (cm, GeV).
        If False: treat hits as processed/normalized (unitless).
    ax : list of Matplotlib axes or None
        If None, new 1x4 figure is created.
    title : str
        Figure title.
    label : str
        Label for the histograms.
    hist_kwargs : dict
        Additional keyword arguments for plt.hist().
    """

    flat = X_hits.reshape(-1, X_hits.shape[-1])

    if raw and mask_zero:
        mask = np.any(flat != 0, axis=1)
        flat = flat[mask]

    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    else:
        assert len(ax) == 4, "Expected 4 axes."

    hist_kwargs = dict(hist_kwargs)
    bins = hist_kwargs.pop("bins", 25)
    alpha = hist_kwargs.pop("alpha", 0.7)

    if raw:
        label_prefix = "Non-Zero " if mask_zero else ""
        labels = [
            f"{label_prefix}X Coordinates [cm]",
            f"{label_prefix}Y Coordinates [cm]",
            f"{label_prefix}Z Coordinates [cm]",
            f"{label_prefix}Hit Energy [GeV]",
        ]
    else:
        labels = [
            "Normalized X",
            "Normalized Y",
            "Normalized Z",
            "Normalized Energy",
        ]

    for i in range(4):
        ax[i].hist(flat[:, i], bins=bins, alpha=alpha, **hist_kwargs)
        ax[i].set_xlabel(labels[i])
        ax[i].set_yscale("log")

    if title is not None:
        plt.suptitle(title, y=0.97)

    return ax
