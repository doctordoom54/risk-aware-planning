#!/usr/bin/env python3
"""
Standalone JAX residual predictor -- load_params/phi_net_forward need only
jax/jax.numpy/numpy, no torch and no learning.py. To use it in another
project, copy BOTH this file and phi_params_scaleonly.npz (produced by
jax_inference.py's export_params()); nothing else is required.

Inputs are RAW (unnormalized) real units:
    x_raw = [vx_body, yaw_rate]     (state)
    u_raw = [cmd_vx, cmd_wz]        (command)
Output is RAW (real units):
    y_hat = [ax_body, yaw_accel]    (predicted residual)
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS_PATH = os.path.join(SCRIPT_DIR, "phi_params_scaleonly.npz")


def load_params(path=DEFAULT_PARAMS_PATH):
    """Load the exported weights into a JAX params pytree."""
    npz = np.load(path)
    return {k: jnp.asarray(npz[k]) for k in npz.files}


def phi_matrix(params, x_raw):
    """Standalone Phi(x) in (2,2), state-dependent only -- before the @u_raw
    product / de-normalization. Exposed separately so callers that need to
    invert the control-input map (e.g. closed-loop tracking) can get the raw
    matrix instead of the already-multiplied-by-u residual."""
    h1 = jax.nn.relu(params["W1"] @ x_raw + params["b1"])
    h2 = jax.nn.relu(params["W2"] @ h1 + params["b2"])
    out = params["W3"] @ h2 + params["b3"]            # (4,)
    return out.reshape(2, 2)                          # row-major, matches torch .view(2,2)


def phi_net_forward(params, x_raw, u_raw):
    """Single-sample forward pass. x_raw, u_raw: shape (2,), real units."""
    phi = phi_matrix(params, x_raw)
    y_hat_n = phi @ u_raw                             # (2,), normalized target space
    y_hat = y_hat_n * params["y_std"]   # de-normalize -> real units
    return y_hat
