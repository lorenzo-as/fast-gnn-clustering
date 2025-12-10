import os
import json
import pickle
import numpy as np
from pathlib import Path
from qgravnet import QGravNetFactory
from utils import response_rmse
from data import load_processed 

try:
    import keras 
except ImportError:
    from tensorflow import keras # type: ignore

DATA_FILE = "data/toy_calo/toy_calo_processed.h5"
TRAIN_DIR = "results/train1"

model_cfg = {
    "n_blocks": 1,
    "n_neighbours": 16,
    "n_filters": 64,
    "dense_kernel_quantizer": "quantized_bits(8, 0, 1, alpha=1.0)",
    "dense_bias_quantizer": "quantized_bits(8, 0, 1, alpha=1.0)",
}

optimizer_cfg = {
    "optimizer": "adam",
    "loss": {"regression": response_rmse, "classification": "binary_crossentropy"},
    "loss_weights": {"regression": 0.9, "classification": 0.1},
    "metrics": {"classification": ["accuracy"]},
}

callbacks = [
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1),
    keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
]

n_epochs = 100
batch_size = 32

if __name__ == "__main__":
    assert Path(TRAIN_DIR).parent.exists(), f"Parent directory {Path(TRAIN_DIR).parent} does not exist."
    os.makedirs(TRAIN_DIR, exist_ok=False)

    D = load_processed(DATA_FILE)

    model = QGravNetFactory(**model_cfg).create_keras_model(
        n_vertices=128, n_features=4
    )
    model.compile(**optimizer_cfg)

    history = model.fit(
        x=D["X_hits_train"],
        y={
            "regression": D["y_energy_train"],
            "classification": D["y_pid_train"],
        },
        epochs=n_epochs,
        validation_split=0.25,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    model.save_weights(os.path.join(TRAIN_DIR, "model.weights.h5"))

    with open(os.path.join(TRAIN_DIR, "model_cfg.pkl"), "wb") as f:
        pickle.dump(model_cfg, f)

    with open(os.path.join(TRAIN_DIR, "history.json"), "w") as f:
        json.dump(history.history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    with open(os.path.join(TRAIN_DIR, "info.json"), "w") as f:
        info = {
            "datapath": DATA_FILE,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
        }
        json.dump(info, f, indent=2)

    print("Training complete")
