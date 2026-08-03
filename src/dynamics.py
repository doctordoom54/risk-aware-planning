import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

# 5 States: [px, py, theta, w_r, w_l]
# 2 Controls: [pwm_r, pwm_l]
NX, NU = 5, 2

# ── core pure functions (params packed as p = [R_WHEEL, TRACK_WIDTH, ALPHA, BETA]) ───────────────────
@jit
def _deriv(x, u, p):
    r, B, alpha, beta = p[0], p[1], p[2], p[3]

    th = x[2]
    w_r = x[3]
    w_l = x[4]

    pwm_r = u[0]
    pwm_l = u[1]

    # Kinematics
    v_body = (r / 2.0) * (w_r + w_l)
    dot_px = 0.98*v_body * jnp.cos(th)
    dot_py = 0.98*v_body * jnp.sin(th)
    dot_theta = 0.50 * (r / B) * (w_r - w_l)

    # Motor Dynamics (PWM model)
    dot_wr = alpha * pwm_r - beta * w_r
    dot_wl = alpha * pwm_l - beta * w_l

    return jnp.array([
        dot_px,
        dot_py,
        dot_theta,
        dot_wr,
        dot_wl
    ])

@jit
def _step(x, u, dt, p):
    """One RK4 step."""
    k1 = _deriv(x, u, p)
    k2 = _deriv(x + 0.5 * dt * k1, u, p)
    k3 = _deriv(x + 0.5 * dt * k2, u, p)
    k4 = _deriv(x + dt * k3, u, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# ── noisy dynamics ────────────────────────────────────────────────────────────
@jit
def _noisy_deriv(x, u, p, noise):
    r, B, alpha, beta = p[0], p[1], p[2], p[3]

    th = x[2]
    w_r = x[3]
    w_l = x[4]

    pwm_r = u[0]
    pwm_l = u[1]

    # Noise modifies the nominal 0.98 and 0.50 slip multipliers
    # noise[0] affects linear slip, noise[1] affects angular slip
    c_lin = 0.98 * (1.0 + noise[0])
    c_ang = 0.50 * (1.0 + noise[1])

    # Kinematics
    v_body = (r / 2.0) * (w_r + w_l)
    dot_px = c_lin * v_body * jnp.cos(th)
    dot_py = c_lin * v_body * jnp.sin(th)
    dot_theta = c_ang * (r / B) * (w_r - w_l)

    # Motor Dynamics (PWM model)
    dot_wr = alpha * pwm_r - beta * w_r
    dot_wl = alpha * pwm_l - beta * w_l

    return jnp.array([
        dot_px,
        dot_py,
        dot_theta,
        dot_wr,
        dot_wl
    ])

@jit
def _noisy_step(x, u, dt, p, noise):
    """One RK4 step with noise in the kinematics parameters."""
    k1 = _noisy_deriv(x, u, p, noise)
    k2 = _noisy_deriv(x + 0.5 * dt * k1, u, p, noise)
    k3 = _noisy_deriv(x + 0.5 * dt * k2, u, p, noise)
    k4 = _noisy_deriv(x + dt * k3, u, p, noise)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
