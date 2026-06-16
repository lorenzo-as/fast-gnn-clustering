"""Tests for the beta-weighted OC payload correction loss and its data builders."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from fastgnn.data.base import _build_payload_correction_targets, _build_payload_seeds
from fastgnn.evaluation.oc_metrics import decode_payload_corrections
from fastgnn.training.objectcondensation_loss import calc_payload_correction_loss

SPECS = [
    {
        "name": "et",
        "seed": "et",
        "target": "sum_et",
        "correction": "mul_exp_tanh",
        "a": 3.0,
        "epsilon": 1.0e-6,
        "weight": 1.0,
    },
    {
        "name": "eta",
        "seed": "eta",
        "target": "eta_energy_weighted",
        "correction": "additive",
        "s": 1.0,
        "sigma": 0.02,
        "weight": 1.0,
    },
    {
        "name": "phi",
        "seed": "phi",
        "target": "phi_energy_weighted",
        "correction": "additive_circular",
        "s": 1.0,
        "sigma": 0.02,
        "weight": 1.0,
    },
    {
        "name": "z",
        "seed": "z",
        "target": "z_energy_weighted",
        "correction": "additive",
        "s": 1.0,
        "sigma": 5.0,
        "weight": 1.0,
    },
]


def _loss(predictions, seeds, targets, beta, cluster_index, batch, **kw):
    return calc_payload_correction_loss(
        payload_predictions=tf.constant(predictions, tf.float32),
        payload_seeds=tf.constant(seeds, tf.float32),
        payload_targets=tf.constant(targets, tf.float32),
        beta=tf.constant(beta, tf.float32),
        cluster_index_per_event=tf.constant(cluster_index, tf.int32),
        batch=tf.constant(batch, tf.int32),
        payload_specs=SPECS,
        **kw,
    )


def test_zero_loss_when_no_correction_needed() -> None:
    # r=0 and seed==target -> exp(0)=1 and additive diff 0 -> residual 0 for every quantity.
    seeds = np.array([[2.0, 2.3, 0.1, 340.0], [5.0, 2.5, -0.2, 350.0]], np.float32)
    loss = _loss(
        predictions=np.zeros((2, 4), np.float32),
        seeds=seeds,
        targets=seeds.copy(),
        beta=[0.9, 0.8],
        cluster_index=[1, 1],
        batch=[0, 0],
    )
    assert float(loss) == 0.0


def test_mul_exp_tanh_log_residual_value() -> None:
    # single hit, only the E_T quantity active: residual = a*tanh(r) when seed==target.
    # a=1, r=0.5 -> |residual|=tanh(0.5)<1 so huber is the squared regime.
    specs = [{**SPECS[0], "a": 1.0}]
    r = 0.5
    loss = calc_payload_correction_loss(
        payload_predictions=tf.constant([[r]], tf.float32),
        payload_seeds=tf.constant([[2.0]], tf.float32),
        payload_targets=tf.constant([[2.0]], tf.float32),
        beta=tf.constant([0.9], tf.float32),
        cluster_index_per_event=tf.constant([1], tf.int32),
        batch=tf.constant([0], tf.int32),
        payload_specs=specs,
    )
    residual = 1.0 * np.tanh(r)  # log(seed*exp(a*tanh r)) - log(seed)
    assert np.isclose(float(loss), residual**2, atol=1e-5)


def test_phi_wrap_keeps_residual_small_across_pi() -> None:
    # seed near +pi, target near -pi: wrapped diff is small, so loss stays small.
    specs = [
        {
            "name": "phi",
            "seed": "phi",
            "target": "phi",
            "correction": "additive_circular",
            "s": 1.0,
            "sigma": 1.0,
            "weight": 1.0,
        }
    ]
    loss = calc_payload_correction_loss(
        payload_predictions=tf.constant([[0.0]], tf.float32),
        payload_seeds=tf.constant([[np.pi - 0.05]], tf.float32),
        payload_targets=tf.constant([[-np.pi + 0.05]], tf.float32),
        beta=tf.constant([0.9], tf.float32),
        cluster_index_per_event=tf.constant([1], tf.int32),
        batch=tf.constant([0], tf.int32),
        payload_specs=specs,
    )
    assert float(loss) < 0.02  # wrapped diff ~0.1, not ~2*pi


def test_beta_weighting_downweights_low_beta_hits() -> None:
    # Two hits in one object: a large residual on the low-beta hit barely matters.
    seeds = np.array([[2.0], [2.0]], np.float32)
    targets = np.array([[2.0], [2.0]], np.float32)
    specs = [SPECS[1]]  # additive eta-like, sigma in spec
    common = {"cluster_index": [1, 1], "batch": [0, 0]}

    def loss_with(pred, beta):
        return float(
            calc_payload_correction_loss(
                payload_predictions=tf.constant(pred, tf.float32),
                payload_seeds=tf.constant(seeds, tf.float32),
                payload_targets=tf.constant(targets, tf.float32),
                beta=tf.constant(beta, tf.float32),
                cluster_index_per_event=tf.constant(common["cluster_index"], tf.int32),
                batch=tf.constant(common["batch"], tf.int32),
                payload_specs=specs,
            )
        )

    # big residual on the low-beta hit vs the same residual on the high-beta hit
    low = loss_with([[0.0], [1.0]], [0.99, 0.01])  # residual on low-beta hit
    high = loss_with([[1.0], [0.0]], [0.99, 0.01])  # residual on high-beta hit
    assert high > 10 * low  # high-beta hit dominates the object loss


def test_beta_detach_controls_gradient_into_beta() -> None:
    seeds = np.array([[2.0]], np.float32)
    targets = np.array([[2.0]], np.float32)
    specs = [SPECS[1]]

    def grad(beta_detach):
        beta = tf.Variable([0.7], dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = calc_payload_correction_loss(
                payload_predictions=tf.constant([[1.0]], tf.float32),
                payload_seeds=tf.constant(seeds, tf.float32),
                payload_targets=tf.constant(targets, tf.float32),
                beta=beta,
                cluster_index_per_event=tf.constant([1], tf.int32),
                batch=tf.constant([0], tf.int32),
                payload_specs=specs,
                beta_detach=beta_detach,
            )
        return tape.gradient(loss, beta)

    g_detached = grad(beta_detach=True)
    if g_detached is not None:
        assert float(tf.reduce_sum(tf.abs(g_detached))) == 0.0
    g = grad(beta_detach=False)
    assert g is not None
    assert float(tf.reduce_sum(tf.abs(g))) > 0.0


def test_build_seeds_and_correction_targets() -> None:
    # Two hits in object 1, one in object 2; one noise hit.
    hits = {
        "energy": np.array([10.0, 10.0, 4.0, 1.0], np.float32),
        "et": np.array([5.0, 5.0, 4.0, 1.0], np.float32),
        "eta": np.array([2.0, 2.2, 1.5, 0.0], np.float32),
        "phi": np.array([0.1, 0.3, 1.0, 0.0], np.float32),
        "z": np.array([330.0, 350.0, 400.0, 0.0], np.float32),
    }
    hit_object_id = np.array([1, 1, 2, 0], np.int32)
    seeds = _build_payload_seeds(hits, SPECS)
    np.testing.assert_allclose(seeds[:, 0], hits["et"])  # seed = own feature
    np.testing.assert_allclose(seeds[:, 1], hits["eta"])

    targets = _build_payload_correction_targets(hit_object_id, hits, SPECS)
    # object 1 sum_et = 5+5 = 10 broadcast to both its hits; noise row stays 0.
    np.testing.assert_allclose(targets[:2, 0], [10.0, 10.0])
    np.testing.assert_allclose(targets[2, 0], 4.0)
    assert targets[3, 0] == 0.0
    # energy-weighted eta of object 1 (equal energies) = mean(2.0, 2.2)
    assert np.isclose(targets[0, 1], 2.1, atol=1e-5)


def test_decode_payload_corrections_reconstructs_from_own_features() -> None:
    # eval features carry the seed columns (et, eta, phi, z); one event, two hits.
    feature_names = ["et", "eta", "phi", "z"]
    features = np.array([[[4.0, 2.0, 3.10, 330.0], [1.0, 1.5, 0.0, 400.0]]], np.float64)
    # r=0 on the first hit -> corrected == own feature; nonzero r on the second.
    payload = np.array([[[0.0, 0.0, 0.0, 0.0], [0.5, 0.1, 0.0, 2.0]]], np.float64)
    out = decode_payload_corrections(payload, SPECS, features, feature_names)

    # hit 0: r=0 -> exp(0)=1 and additive offsets 0 -> recover the seed features.
    assert np.isclose(out["et"][0, 0], 4.0)
    assert np.isclose(out["eta"][0, 0], 2.0)
    assert np.isclose(out["z"][0, 0], 330.0)
    # energy alias derived from et and eta.
    assert np.isclose(out["energy"][0, 0], 4.0 * np.cosh(2.0))
    # hit 1: mul_exp_tanh and additive corrections applied to its own seed.
    assert np.isclose(out["et"][0, 1], 1.0 * np.exp(3.0 * np.tanh(0.5)))
    assert np.isclose(out["eta"][0, 1], 1.5 + 0.1)
    assert np.isclose(out["z"][0, 1], 400.0 + 2.0)


def test_decode_payload_corrections_requires_seed_features() -> None:
    with pytest.raises(ValueError, match="seed feature"):
        decode_payload_corrections(np.zeros((1, 1, 4)), SPECS, np.zeros((1, 1, 1)), ["energy"])
