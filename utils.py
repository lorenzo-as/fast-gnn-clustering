import os
import pickle
import json
from keras import ops


def response_rmse(y_true, y_pred):
    y_true = ops.reshape(y_true, (-1,))
    y_pred = ops.reshape(y_pred, (-1,))
    response = ops.divide_no_nan(y_pred, y_true)
    return ops.sqrt(ops.mean(ops.square(response - 1.0)))


def load_run(train_dir):
    """
    Load a training run directory. Returns:
      model_cfg
      weights_path
      history
      datapath
    """
    train_dir = os.path.abspath(train_dir)

    cfg_path = os.path.join(train_dir, "model_cfg.pkl")
    weights_path = os.path.join(train_dir, "model.weights.h5")
    history_path = os.path.join(train_dir, "history.json")
    info_path = os.path.join(train_dir, "info.json")

    # --- existence checks ---
    for path, name in [
        (cfg_path, "model_cfg.pkl"),
        (weights_path, "model.weights.h5"),
        (history_path, "history.json"),
        (info_path, "info.json"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} in {train_dir}")

    with open(cfg_path, "rb") as f:
        model_cfg = pickle.load(f)

    with open(history_path, "r") as f:
        history = json.load(f)

    with open(info_path, "r") as f:
        info = json.load(f)
        datapath = info.get("datapath")

    return model_cfg, weights_path, history, datapath
