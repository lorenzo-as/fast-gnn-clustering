"""Why does beta_alpha stall at ~0.57? Decompose the force on the condensation-point
beta into the contributions of each (weighted) OC loss term.

At a trained equilibrium the NET gradient on beta is ~0, but that hides which terms
set the balance. We compute, at each object's condensation point (max-q hit), the
gradient of each weighted loss component w.r.t. that hit's beta-logit:
    g_LV   = w_att*dL_V_att/dz + w_rep*dL_V_rep/dz   (pushes beta DOWN, >0)
    g_beta = w_bsig*dL_beta_sig/dz                    (pushes beta UP, <0)
If |g_LV| ~ |g_beta| at beta~0.57, the L_V weighting is what clamps beta, and the
ratio tells us how far the weights are from paper-OC (where w_att=w_rep=w_bsig=1).

Run::

    python plotting/feature_target_normalization/beta_force_balance.py \
        --config <run>/resolved_config.yaml --model <run>/final_model.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import qgravnet  # noqa: F401  (registers custom keras layers)
import tensorflow as tf
import yaml

from fastgnn.data import CaloDataset
from fastgnn.training.objectcondensation_loss import batch_and_mask_to_flat, calc_LV_Lbeta
from fastgnn.training.oc_outputs import OCOutputLayout, split_oc_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--qmin", type=float, default=None, help="override qmin (default: config final)"
    )
    ap.add_argument("--max-events", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    ds = CaloDataset(dataset_dir, split="val", max_events=args.max_events)
    try:
        layout = OCOutputLayout.from_config(model_cfg)
    except ValueError:  # legacy output_layout.regressions: build from output_dim (beta + cluster)
        layout = OCOutputLayout.from_output_dim(int(model_cfg["output_dim"]))
    model = tf.keras.models.load_model(args.model, compile=False)

    lw = train_cfg.get("loss_weights") or {}

    def w(key, default):
        v = lw.get(key, default)
        if isinstance(v, dict):  # take the final scheduled value
            v = v["points"][-1][1]
        elif isinstance(v, list):
            v = v[-1][1]
        return float(v)

    w_att, w_rep, w_bsig = w("L_V_attractive", 1.0), w("L_V_repulsive", 1.0), w("L_beta_sig", 1.0)
    qmin = args.qmin if args.qmin is not None else float(train_cfg.get("qmin", 1.0))
    if isinstance(train_cfg.get("qmin_schedule"), dict) and args.qmin is None:
        qmin = float(train_cfg["qmin_schedule"]["points"][-1][1])
    print(f"weights: V_att={w_att} V_rep={w_rep} L_beta_sig={w_bsig}   qmin={qmin:g}\n")

    beta_alpha_all, g_lv_all, g_beta_all = [], [], []
    for batch in ds.batches(
        max_vertices=model_cfg["max_vertices"],
        batch_size=train_cfg["batch_size"],
        feature_names=data_cfg["feature_names"],
        shuffle=False,
        normalize_features=train_cfg.get("normalize_features", True),
        normalization_method=data_cfg.get("normalization", "zscore"),
    ):
        feats = tf.constant(batch["features"])
        hid = tf.cast(tf.constant(batch["hit_object_id"]), tf.int32)
        with tf.GradientTape(persistent=True) as tape:
            outputs = model(feats, training=False)
            slices = split_oc_outputs(outputs, layout)
            flat, batch_idx = batch_and_mask_to_flat(
                {
                    "beta": slices.beta,
                    "beta_logits": slices.beta_logits,
                    "cluster_coords": slices.cluster_coords,
                    "hit_object_id": hid,
                    "mask": tf.constant(batch["mask"]),
                }
            )
            # re-derive beta from logits inside the tape so grads flow to the logit
            beta_logit = flat["beta_logits"]
            tape.watch(beta_logit)
            beta = tf.math.sigmoid(beta_logit)
            comps = calc_LV_Lbeta(
                beta=beta,
                cluster_space_coords=flat["cluster_coords"],
                cluster_index_per_event=flat["hit_object_id"],
                batch=batch_idx,
                qmin=qmin,
                beta_stabilizing=train_cfg.get("beta_stabilizing", "soft_q_scaling"),
                beta_term_option=train_cfg.get("beta_term_option", "paper"),
                return_components=True,
            )
        g_att = tape.gradient(comps["L_V_attractive"], beta_logit)
        g_rep = tape.gradient(comps["L_V_repulsive"], beta_logit)
        g_bsig = tape.gradient(comps["L_beta_sig"], beta_logit)
        del tape
        g_att = np.asarray(g_att) if g_att is not None else np.zeros(beta_logit.shape)
        g_rep = np.asarray(g_rep) if g_rep is not None else np.zeros(beta_logit.shape)
        g_bsig = np.asarray(g_bsig) if g_bsig is not None else np.zeros(beta_logit.shape)
        beta_np = np.asarray(tf.math.sigmoid(beta_logit))

        # restrict to condensation points (max beta per object)
        hid_np = np.asarray(flat["hit_object_id"])
        bidx = np.asarray(batch_idx)
        key = bidx.astype(np.int64) * (hid_np.max() + 1) + hid_np
        for k in np.unique(key[hid_np > 0]):
            m = key == k
            a = np.argmax(np.where(m, beta_np, -1))
            beta_alpha_all.append(beta_np[a])
            g_lv_all.append((g_att[a], g_rep[a]))  # raw, unweighted
            g_beta_all.append(g_bsig[a])

    beta_alpha = np.array(beta_alpha_all)
    g_att_raw = np.array([x[0] for x in g_lv_all])
    g_rep_raw = np.array([x[1] for x in g_lv_all])
    g_lv = w_att * g_att_raw + w_rep * g_rep_raw
    g_beta = w_bsig * np.array(g_beta_all)
    net = g_lv + g_beta
    # paper-faithful counterfactual: V_att = V_rep = L_beta_sig = 1
    g_lv_paper = 1.0 * g_att_raw + 1.0 * g_rep_raw
    net_paper = g_lv_paper + 1.0 * np.array(g_beta_all)
    print(f"condensation points: {len(beta_alpha):,}")
    print(
        f"  beta_alpha            mean={beta_alpha.mean():.3f} median={np.median(beta_alpha):.3f}"
    )
    print("  gradient of loss w.r.t. beta-logit at the condensation point")
    print(f"    g_LV   (pushes beta DOWN, >0): mean={g_lv.mean():+.4e}")
    print(f"    g_beta (pushes beta UP,   <0): mean={g_beta.mean():+.4e}")
    print(f"    net                          : mean={net.mean():+.4e}")
    print(f"  |g_LV| / |g_beta| = {np.abs(g_lv.mean()) / (np.abs(g_beta.mean()) + 1e-30):.2f}")
    print("  (ratio ~1 => the two forces balance here; >1 => L_V dominates and clamps beta low)")
    print("\n  COUNTERFACTUAL with paper-faithful weights V_att=V_rep=L_beta_sig=1:")
    print(f"    g_LV(paper)={g_lv_paper.mean():+.4e}  net(paper)={net_paper.mean():+.4e}")
    print(
        f"    |g_LV| / |g_beta| (paper) = "
        f"{np.abs(g_lv_paper.mean()) / (np.abs(np.array(g_beta_all).mean()) + 1e-30):.3f}"
    )
    print("    (net(paper) strongly negative => beta would be pushed UP toward 1)")


if __name__ == "__main__":
    main()
