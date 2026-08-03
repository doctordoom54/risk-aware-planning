import numpy as np
import pytest
from scipy.stats import norm

import jax
import jax.numpy as jnp

from map.sdf import distance_transform_edt
from map.risk_sdf import (
    build_risk_sdf, query_risk_sdf, collision_margin, ensemble_worst_clearance,
    RiskSDF,
)

RES = 0.1
H, W = 40, 40


def _blocks(*boxes):
    """boxes: list of (r0, r1, c0, c1) -> (K, H, W) bool masks, one per box."""
    masks = np.zeros((len(boxes), H, W), dtype=bool)
    for k, (r0, r1, c0, c1) in enumerate(boxes):
        masks[k, r0:r1, c0:c1] = True
    return masks


# --------------------------------------------------------------------------- #
# (a) reduction to nominal: s_k = 0 -> plain global min-SDF from a union EDT
# --------------------------------------------------------------------------- #
def test_reduction_to_nominal():
    masks = _blocks((5, 9, 5, 9), (20, 25, 28, 33), (30, 34, 10, 14))
    s_k = np.zeros(masks.shape[0])

    risk_sdf, argmin_idx, kappa = build_risk_sdf(masks, s_k, RES, alpha=0.05, d_max=10.0)

    union = masks.any(axis=0)
    nominal = (distance_transform_edt(~union) - distance_transform_edt(union)) * RES
    nominal = np.clip(nominal, -10.0, 10.0)

    np.testing.assert_allclose(risk_sdf, nominal, atol=1e-9)


# --------------------------------------------------------------------------- #
# (b) argmin switching under inflation
# --------------------------------------------------------------------------- #
def test_argmin_switching():
    # obstacle 0: single pixel, 4 px (0.4 m) from the query cell
    # obstacle 1: single pixel, 7 px (0.7 m) from the query cell
    query_r, query_c = 20, 20
    masks = np.zeros((2, H, W), dtype=bool)
    masks[0, query_r, query_c - 4] = True
    masks[1, query_r, query_c + 7] = True
    s_k = np.array([0.02, 0.25])
    alpha = 0.05

    risk_sdf, argmin_idx, kappa = build_risk_sdf(masks, s_k, RES, alpha=alpha, d_max=10.0)

    # sanity: kappa is the CVaR multiplier we expect, and F_1 < F_0 at the query cell
    expected_kappa = norm.pdf(norm.ppf(1 - alpha)) / alpha
    assert kappa == pytest.approx(expected_kappa)

    sdf0 = 0.4 - kappa * s_k[0]
    sdf1 = 0.7 - kappa * s_k[1]
    assert sdf1 < sdf0   # obstacle 1's inflated constraint is tighter

    assert argmin_idx[query_r, query_c] == 1
    assert risk_sdf[query_r, query_c] == pytest.approx(sdf1, abs=1e-6)


# --------------------------------------------------------------------------- #
# (c) monotonicity in s_k
# --------------------------------------------------------------------------- #
def test_monotonic_in_sigma():
    rng = np.random.default_rng(0)
    masks = _blocks((3, 6, 3, 6), (15, 22, 25, 30), (30, 36, 6, 12))
    s_lo = np.array([0.01, 0.02, 0.03])
    s_hi = s_lo.copy()
    s_hi[1] += 0.5   # bump only one obstacle's sigma

    risk_lo, _, _ = build_risk_sdf(masks, s_lo, RES, alpha=0.05, d_max=10.0)
    risk_hi, _, _ = build_risk_sdf(masks, s_hi, RES, alpha=0.05, d_max=10.0)

    assert np.all(risk_hi <= risk_lo + 1e-12)


# --------------------------------------------------------------------------- #
# (d) interior sign: strictly negative inside an obstacle
# --------------------------------------------------------------------------- #
def test_interior_strictly_negative():
    masks = _blocks((10, 20, 10, 20))   # 10x10 block, deep interior exists
    s_k = np.zeros(1)

    risk_sdf, argmin_idx, kappa = build_risk_sdf(masks, s_k, RES, alpha=0.05, d_max=10.0)

    center = risk_sdf[15, 15]
    assert center < 0.0
    assert argmin_idx[15, 15] == 0


# --------------------------------------------------------------------------- #
# (e) JAX vs NumPy parity + jit without retracing across repeated batch sizes
# --------------------------------------------------------------------------- #
def test_jax_query_matches_numpy_at_grid_centers():
    masks = _blocks((5, 9, 5, 9), (20, 25, 28, 33))
    s_k = np.array([0.02, 0.1])
    risk_sdf, argmin_idx, kappa = build_risk_sdf(masks, s_k, RES, alpha=0.05, d_max=10.0)

    origin = (0.0, 0.0)
    rows = np.array([2, 15, 20, 39])
    cols = np.array([2, 15, 28, 39])
    xs = cols * RES + origin[0]
    ys = rows * RES + origin[1]
    points = np.stack([xs, ys], axis=-1)

    field_jnp = jnp.asarray(risk_sdf)
    clearances = query_risk_sdf(field_jnp, jnp.asarray(points), origin, RES)
    expected = risk_sdf[rows, cols]
    np.testing.assert_allclose(np.asarray(clearances), expected, atol=1e-5)


def test_jax_query_jit_no_retrace_same_shape():
    masks = _blocks((5, 9, 5, 9), (20, 25, 28, 33))
    s_k = np.array([0.02, 0.1])
    risk_sdf, argmin_idx, kappa = build_risk_sdf(masks, s_k, RES, alpha=0.05, d_max=10.0)
    field_jnp = jnp.asarray(risk_sdf)
    origin = (0.0, 0.0)

    trace_count = [0]

    def _wrapped(points):
        trace_count[0] += 1
        return query_risk_sdf(field_jnp, points, origin, RES)

    jitted = jax.jit(_wrapped)

    pts_5 = jnp.asarray(np.random.default_rng(1).uniform(0, 3.9, size=(5, 2)))
    pts_5b = jnp.asarray(np.random.default_rng(2).uniform(0, 3.9, size=(5, 2)))
    pts_7 = jnp.asarray(np.random.default_rng(3).uniform(0, 3.9, size=(7, 2)))
    pts_ensemble = jnp.asarray(np.random.default_rng(4).uniform(0, 3.9, size=(3, 4, 2)))

    jitted(pts_5)
    jitted(pts_5b)             # same shape -> cache hit, no retrace
    assert trace_count[0] == 1

    jitted(pts_7)              # new shape -> retrace
    assert trace_count[0] == 2

    jitted(pts_5)              # shape seen before -> cache hit again
    assert trace_count[0] == 2

    # (B, T, 2) ensemble batch shape works through the same code path
    out = jitted(pts_ensemble)
    assert out.shape == (3, 4)
    assert trace_count[0] == 3


# --------------------------------------------------------------------------- #
# Step 3: collision predicate + ensemble reduction
# --------------------------------------------------------------------------- #
def test_collision_margin():
    clearances = jnp.array([0.05, 0.3, 0.5])
    robot_radius = 0.2
    sample_spacing = 0.1
    safe = collision_margin(clearances, robot_radius, sample_spacing)
    # threshold = 0.2 + 0.05 = 0.25
    np.testing.assert_array_equal(np.asarray(safe), [False, True, True])


def test_ensemble_worst_clearance():
    clearances = jnp.array([[0.5, 0.2, 0.9], [0.1, 0.4, 0.3]])
    L = ensemble_worst_clearance(clearances)
    np.testing.assert_allclose(np.asarray(L), [-0.2, -0.1])
    assert L.shape == (2,)


# --------------------------------------------------------------------------- #
# Step 5: RiskSDF wrapper
# --------------------------------------------------------------------------- #
def test_risk_sdf_wrapper_query_and_rebuild():
    masks = _blocks((5, 9, 5, 9), (20, 25, 28, 33))
    s_k = np.array([0.02, 0.1])
    d_max = 10.0
    risk_sdf, argmin_idx, kappa = build_risk_sdf(masks, s_k, RES, alpha=0.05, d_max=d_max)

    field = RiskSDF(risk_sdf, origin=(0.0, 0.0), res=RES, kappa=kappa,
                     argmin_idx=argmin_idx, obstacle_masks=masks, s_k=s_k, d_max=d_max)

    points = jnp.asarray(np.stack([np.array([1.0, 2.0]), np.array([1.0, 2.5])], axis=-1))
    out = field.query(points)
    assert out.shape == (2,)

    old_kappa = field.kappa
    field.rebuild(alpha=0.5)
    assert field.kappa != pytest.approx(old_kappa)
