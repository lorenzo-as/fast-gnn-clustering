import os
import matplotlib.pyplot as plt
from data import (
    fetch_dataset,
    load_raw_dataset,
    compute_hits_normalization,
    apply_hits_normalization,
    split_dataset,
    save_processed,
    plot_data_distributions,
)

FETCH = True  # set True to download the Zenodo files
N_FILES = 1

DATASET_DIR = "data/toy_calo"
RAW_DIR = os.path.join(DATASET_DIR, "raw")

OUTFILE = os.path.join(DATASET_DIR, "toy_calo_processed.h5")
RAW_PLOT = os.path.join(DATASET_DIR, "raw_data.png")
NORM_PLOT = os.path.join(DATASET_DIR, "normalized_data.png")

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    if FETCH:
        print("Fetching dataset from Zenodo…")
        fetch_dataset(N_FILES, RAW_DIR)
        print("Download complete.")

    X_hits, X_size, y_energy, y_pid = load_raw_dataset(N_FILES, RAW_DIR)

    # Plot raw distributions
    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    plot_data_distributions(
        X_hits,
        raw=True,
        mask_zero=False,
        ax=ax,
        title="Raw Data Distributions (Including Padding)",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(RAW_PLOT)
    plt.close(fig)
    print(f"Saved raw plot to {RAW_PLOT}")

    # Train/test split
    (
        X_hits_train,
        X_hits_test,
        X_size_train,
        X_size_test,
        y_energy_train,
        y_energy_test,
        y_pid_train,
        y_pid_test,
    ) = split_dataset(X_hits, X_size, y_energy, y_pid)

    # Normalize hits and plot normalized data
    mean, std = compute_hits_normalization(X_hits_train)

    X_hits_train = apply_hits_normalization(X_hits_train, mean, std)
    X_hits_test = apply_hits_normalization(X_hits_test, mean, std)

    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    plot_data_distributions(
        X_hits_train,
        raw=False,
        mask_zero=False,
        ax=ax,
        alpha=0.5,
        density=True,
        title="Normalized Hits Distributions",
    )
    plot_data_distributions(
        X_hits_test, raw=False, mask_zero=False, ax=ax, alpha=0.5, density=True
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.legend(
        labels=["Train", "Test"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
        fancybox=False,
        fontsize=12,
    )
    fig.savefig(NORM_PLOT)
    plt.close(fig)
    print(f"Saved normalized plot to {NORM_PLOT}")

    # Save
    save_processed(
        {
            "X_hits_train": X_hits_train,
            "X_size_train": X_size_train,
            "y_energy_train": y_energy_train,
            "y_pid_train": y_pid_train,
            "X_hits_test": X_hits_test,
            "X_size_test": X_size_test,
            "y_energy_test": y_energy_test,
            "y_pid_test": y_pid_test,
            "hits_mean": mean,
            "hits_std": std,
        },
        OUTFILE,
    )

    print("Done.")
    print("Saved processed dataset to:", OUTFILE)
