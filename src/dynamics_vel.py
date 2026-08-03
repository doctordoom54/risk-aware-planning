import numpy as np
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

from .residual_phi_predictor import (load_params as _load_phi_params,
                                      phi_net_forward as _phi_net_forward,
                                      phi_matrix as _phi_matrix)

#state = [x,y,theta,v_b,omega], statedot = [vb*cos(theta), vb*sin(theta), omega, NN_1, NN_2]
#control = [v_cmd, omega_cmd]
#
# The velocity block [dot_vb, dot_omega] is the learned model stacked on the exact
# kinematics: A_n @ v + B_n @ u + Phi(v,u), where Phi(v,u) is the residual predictor's
# output (residual_phi_predictor.py already returns the full Phi(v,u) @ u product, see
# its module docstring). Cartesian pose plays no role in the residual, so the predictor
# only ever sees v = [vb, omega] and u -- pose is recovered by exact integration.
#
# NOTE (applies to the whole file, incl. the closed-loop tracking section below): the
# residual net is SCALE-ONLY -- de-normalization is `y_hat_n * y_std`, there is no
# y_mean term and none is used anywhere here. So the nominal+NN model is exactly
# LINEAR in u, not affine: v_dot = A_n @ v + D_eff(v) @ u, D_eff(v) = B_n + diag(y_std) @ Phi(v).
#
# 5 States: [x, y, theta, v_b, omega]
# 2 Controls: [v_cmd, omega_cmd]
NX, NU = 5, 2

# Nominal linear velocity dynamics (first-order actuator model):
#   A_n = diag(-1/tau_v, -1/tau_w),  B_n = diag(k1/tau_v, k2/tau_w)
# packed as p = [a1, a2, b1, b2] = [diag(A_n), diag(B_n)]. Values (system ID on rover
# response on rigid surfaces) and the closed-loop-tracking clip bounds live in
# config.py (DynParams.a_diag/b_diag, RiskParams.ax_clip_*/yaw_clip_*) -- this module
# takes them as explicit constructor/call arguments, no hardcoded defaults, so it
# stays usable standalone without importing config.py.


# ── core pure functions (nominal params packed as p = [a1, a2, b1, b2]) ─────────────────
@jit
def _deriv(z, u, p, phi_params):
    a1, a2, b1, b2 = p[0], p[1], p[2], p[3]

    th = z[2]
    vb = z[3]
    om = z[4]

    v = jnp.array([vb, om])
    nominal = jnp.array([a1 * vb + b1 * u[0], a2 * om + b2 * u[1]])
    residual = _phi_net_forward(phi_params, v, u)      # Phi(v,u) @ u, raw units
    vdot = nominal + residual

    return jnp.array([
        vb * jnp.cos(th),
        vb * jnp.sin(th),
        om,
        vdot[0],
        vdot[1],
    ])


@jit
def _step(z, u, dt, p, phi_params):
    """One RK4 step."""
    k1 = _deriv(z, u, p, phi_params)
    k2 = _deriv(z + 0.5 * dt * k1, u, p, phi_params)
    k3 = _deriv(z + 0.5 * dt * k2, u, p, phi_params)
    k4 = _deriv(z + dt * k3, u, p, phi_params)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@partial(jit, static_argnums=(2,))
def _rollout(z0, u, nsteps, dt, p, phi_params):
    """Constant control for `nsteps` RK4 steps."""
    def body(z, _):
        zn = _step(z, u, dt, p, phi_params)
        return zn, zn
    _, zs = jax.lax.scan(body, z0, None, length=nsteps)
    return jnp.concatenate([z0[None, :], zs], axis=0)


# batch a constant-control rollout over many controls sharing z0 / nsteps / dt / phi_params
_batch_rollout = jit(
    vmap(_rollout, in_axes=(None, 0, None, None, None, None)),
    static_argnums=(2,),
)


# ── SCP edge Jacobians (A_k = df_edge/ds_k, B_k = df_edge/du_k) ─────────────────────
# f_edge(s_k, u_k) is the multiple-shooting edge propagation map: the constant-control
# `_rollout` above run for `nsteps` RK4 steps of length `dt`, keeping only the terminal
# state. A_k, B_k are its EXACT autodiff Jacobians (jax.jacobian), matching the SCP
# formulation's linearized defect constraint s_{k+1} = f_edge(s_bar_k,u_bar_k)
# + A_k(s_k - s_bar_k) + B_k(u_k - u_bar_k) + nu_k.
@partial(jit, static_argnums=(2,))
def _edge(z0, u, nsteps, dt, p, phi_params):
    """Edge propagation map f_edge: terminal state of an nsteps-step, constant-u
    RK4 rollout starting at z0."""
    return _rollout(z0, u, nsteps, dt, p, phi_params)[-1]


_edge_jacz = jax.jacobian(_edge, argnums=0)   # A_k = df_edge/dz0  (NX, NX)
_edge_jacu = jax.jacobian(_edge, argnums=1)   # B_k = df_edge/du   (NX, NU)


@partial(jit, static_argnums=(2,))
def _edge_jac_single(z0, u, nsteps, dt, p, phi_params):
    return (_edge_jacz(z0, u, nsteps, dt, p, phi_params),
            _edge_jacu(z0, u, nsteps, dt, p, phi_params),
            _edge(z0, u, nsteps, dt, p, phi_params))


# vmap the edge linearisation over a whole knot sequence (shared nsteps/dt/p/phi_params)
_batch_edge_jac = jit(
    vmap(_edge_jac_single, in_axes=(0, 0, None, None, None, None)),
    static_argnums=(2,),
)


# ── closed-loop tracking ensemble (feedback-linearization control law) ──────────────
# Tracks a nominal reference trajectory under an injected constant per-member
# disturbance on (dot(v_b), dot(omega)), using a controller that inverts the known
# nominal+NN model. Reused for the obstacle-risk CVaR term (risk_planner.py, next).
# NO y_mean anywhere below -- the residual net is scale-only (see module note above).

@jit
def _D_eff(v, p, phi_params):
    """Effective input map D_eff(v) = B_n + diag(y_std) @ Phi(v), shape (2,2). The
    nominal+NN model is exactly LINEAR in u: v_dot = A_n @ v + D_eff(v) @ u."""
    b1, b2 = p[2], p[3]
    B_n = jnp.diag(jnp.array([b1, b2]))
    phi = _phi_matrix(phi_params, v)
    return B_n + jnp.diag(phi_params["y_std"]) @ phi


@jit
def _closed_loop_u(v, v_r, vdot_r, p, phi_params, K):
    """Feedback-linearization tracking control (no y_mean term): inverts D_eff at the
    CURRENT (possibly disturbed) velocity `v` so the tracking error v - v_r decays at
    rate K. Solved as a 2x2 linear system rather than an explicit matrix inverse."""
    a1, a2 = p[0], p[1]
    A_n = jnp.diag(jnp.array([a1, a2]))
    vtil = v - v_r
    rhs = vdot_r - A_n @ v_r - K @ vtil
    return jnp.linalg.solve(_D_eff(v, p, phi_params), rhs)


@jit
def _deriv_biased(z, u, p, phi_params, d, clip_lo, clip_hi):
    """Disturbed-path derivative: the injected constant disturbance `d` is added to
    the NN's PREDICTED residual (not the full state derivative), and that sum is
    clipped to the physical envelope [clip_lo, clip_hi] (raw CSV-logged extremes)
    before combining with the nominal A_n @ v + B_n @ u term -- so a disturbed member
    can never imply an acceleration the real rover has never exhibited. Only ever
    used on the disturbed path; the reference rollout (_deriv) is untouched."""
    a1, a2, b1, b2 = p[0], p[1], p[2], p[3]

    th = z[2]
    vb = z[3]
    om = z[4]

    v = jnp.array([vb, om])
    nominal = jnp.array([a1 * vb + b1 * u[0], a2 * om + b2 * u[1]])
    d_pred = _phi_net_forward(phi_params, v, u)          # Phi(v,u) @ u, raw units
    d_new = jnp.clip(d_pred + d, clip_lo, clip_hi)
    vdot = nominal + d_new

    return jnp.array([
        vb * jnp.cos(th),
        vb * jnp.sin(th),
        om,
        vdot[0],
        vdot[1],
    ])


@jit
def _step_tracked(z, u, dt, p, phi_params, d, clip_lo, clip_hi):
    """One RK4 step on the disturbed path. `u` and `d` are held FIXED across all four
    stages -- `u` was already solved once for this step by _closed_loop_u."""
    k1 = _deriv_biased(z, u, p, phi_params, d, clip_lo, clip_hi)
    k2 = _deriv_biased(z + 0.5 * dt * k1, u, p, phi_params, d, clip_lo, clip_hi)
    k3 = _deriv_biased(z + 0.5 * dt * k2, u, p, phi_params, d, clip_lo, clip_hi)
    k4 = _deriv_biased(z + dt * k3, u, p, phi_params, d, clip_lo, clip_hi)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@jit
def _ref_vel_deriv(Z_ref, u_ref, p, phi_params):
    """Reference velocity v_r(t) and its derivative v_dot_r(t) for t=0..N-1, read off
    a nominal _rollout output Z_ref (N+1,5) under the constant reference control
    u_ref. Reuses the unmodified _deriv; never touches the disturbed path.
    Standalone inspection utility (see VelPoseDynamics.reference_traj) -- NOT used by
    tracked_ensemble any more, since the outer pose loop below supplies its own
    v_ref/vdot_ref from the actual per-member state, not this static profile."""
    Zr = Z_ref[:-1]                                       # (N,5)
    Vr = Zr[:, 3:5]                                        # (N,2)
    Dr = vmap(lambda z: _deriv(z, u_ref, p, phi_params))(Zr)
    return Vr, Dr[:, 3:5]


@jit
def _wrap_angle(a):
    """Wrap an angle (or angle difference) to (-pi, pi], jit/autodiff-safe."""
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


@jit
def _ref_pose_vel(Z_ref):
    """Desired position/heading/inertial-frame velocity for t=0..N-1, read directly
    off a nominal _rollout output Z_ref (N+1,5) -- no _deriv/NN call needed (unlike
    _ref_vel_deriv above). Feeds the OUTER pose-tracking loop (_outer_ref)."""
    Zr = Z_ref[:-1]                                       # (N,5)
    P_d = Zr[:, 0:2]                                       # (N,2)
    Theta_d = Zr[:, 2]                                     # (N,)
    vb_d = Zr[:, 3]
    V_d_I = jnp.stack([vb_d * jnp.cos(Theta_d), vb_d * jnp.sin(Theta_d)], axis=1)  # (N,2)
    return P_d, Theta_d, V_d_I


@jit
def _outer_ref(z, p_d, theta_d, v_d_I, psi_ref_prev, dt, Kp, k_psi, v_eps_sq):
    """One step of the outer pose-tracking loop (Kanayama-style unicycle trajectory
    tracking): corrects the reference velocity command using the ACTUAL current
    position/heading error against the nominal (open-loop) pose (p_d, theta_d).
    `psi` (heading) is always the rover's OWN actual current heading -- both where it
    projects the corrected inertial velocity down to a body-frame forward speed, and
    where it's compared against psi_ref for the heading-rate correction; angle
    differences are wrapped to (-pi, pi]. Returns (v_ref, psi_ref); psi_ref is carried
    forward so the next step can finite-difference psi_dot_ref."""
    p = z[0:2]
    psi = z[2]
    p_tilde = p - p_d
    v_ref_I = v_d_I - Kp @ p_tilde                                    # eq. 14
    v_ref_x = jnp.array([jnp.cos(psi), jnp.sin(psi)]) @ v_ref_I        # eq. 14 projection
    speed_sq = v_ref_I @ v_ref_I
    # arctan2's VALUE at (0,0) is well-defined (0, by convention), but its GRADIENT
    # there is 0/0 -- NaN. jnp.where selects branches by value but autodiff still
    # computes both branches' gradients before masking, so that NaN leaks through
    # whenever v_ref_I is exactly/nearly zero (e.g. a trajectory's very first knot,
    # starting from rest). Substitute a safe nonzero vector into arctan2's args on
    # the branch that gets discarded anyway -- standard fix, doesn't change the
    # VALUE (this branch is never selected when v_ref_I is degenerate).
    safe_v_ref_I = jnp.where(speed_sq > v_eps_sq, v_ref_I, jnp.array([1.0, 0.0]))
    psi_ref = jnp.where(speed_sq > v_eps_sq,
                         jnp.arctan2(safe_v_ref_I[1], safe_v_ref_I[0]),
                         theta_d)                                      # eq. 16
    psi_dot_ref = _wrap_angle(psi_ref - psi_ref_prev) / dt             # finite-difference
    omega_ref = psi_dot_ref - k_psi * _wrap_angle(psi - psi_ref)       # eq. 15
    return jnp.array([v_ref_x, omega_ref]), psi_ref


@jit
def _tracked_rollout(z0, u_ref, P_d, Theta_d, V_d_I, dt, p, phi_params, Kp, k_psi,
                      v_eps_sq, K, d, clip_lo, clip_hi, u_clip_lo, u_clip_hi):
    """Closed-loop rollout of one ensemble member absorbing disturbance `d`, with an
    OUTER pose-tracking loop (_outer_ref: corrects v_ref from the ACTUAL position/
    heading error against the nominal (P_d, Theta_d, V_d_I) profile) feeding an INNER
    velocity-tracking loop (_closed_loop_u, model-inversion feedback-linearization).
    v_dot_ref is a backward finite difference of v_ref across steps, not an analytic
    derivative of the outer-loop law -- EXCEPT at t=0, seeded below. Number of steps
    = leading dim of P_d/Theta_d/V_d_I; returns (N+1, 5) like _rollout.

    At t=0, z0 = z_ref(0) exactly, so both the true nominal omega_d(0) (= z0[4]) and
    v_dot_d(0) are known exactly -- the latter via ONE extra _deriv(z0, u_ref, ...)
    call (the EXACT instantaneous nominal acceleration under the reference control,
    not a finite-difference approximation of it). The carry is seeded with a
    "virtual t=-1" psi_ref/v_ref (psi_ref_prev = theta_d(0) - omega_d(0)*dt,
    v_ref_prev = z0[3:5] - v_dot_d(0)*dt) so the t=0 finite differences reconstruct
    the TRUE nominal omega_d(0)/v_dot_d(0) exactly. This makes v_ref(0) = z0[3:5] and
    v_dot_ref(0) = v_dot_d(0) exactly (zero tracking error and the exact feedforward
    at t=0), so u(0) == u_ref exactly (up to floating point) PROVIDED u_ref is itself
    within [u_clip_lo, u_clip_hi] -- callers must clip u_ref before calling.

    u_clip_lo/u_clip_hi: the model-inversion correction (_closed_loop_u) is NOT
    naturally bounded -- it solves for whatever u drives the tracking error to zero
    at rate K, which can and does exceed the command range the residual net was
    trained on (measured: u_v up to ~0.64 against a +-0.38 trained envelope, on a
    real 5-segment/225-member/main_test.py-equivalent run). Feeding out-of-envelope
    u into the NN (_phi_net_forward, inside _step_tracked/_deriv_biased) means
    extrapolating outside its training distribution, unreliably. So u is clipped to
    [u_clip_lo, u_clip_hi] BEFORE being applied -- same physical actuator saturation
    a real controller would hit -- not merely reported."""
    def body(carry, ref_t):
        z, psi_ref_prev, v_ref_prev = carry
        p_d, theta_d, v_d_I = ref_t
        v_ref, psi_ref = _outer_ref(z, p_d, theta_d, v_d_I, psi_ref_prev, dt,
                                     Kp, k_psi, v_eps_sq)
        vdot_ref = (v_ref - v_ref_prev) / dt
        u = _closed_loop_u(z[3:5], v_ref, vdot_ref, p, phi_params, K)
        u = jnp.clip(u, u_clip_lo, u_clip_hi)
        zn = _step_tracked(z, u, dt, p, phi_params, d, clip_lo, clip_hi)
        return (zn, psi_ref, v_ref), zn
    vdot_d0 = _deriv(z0, u_ref, p, phi_params)[3:5]
    carry0 = (z0, z0[2] - z0[4] * dt, z0[3:5] - vdot_d0 * dt)
    _, zs = jax.lax.scan(body, carry0, (P_d, Theta_d, V_d_I))
    return jnp.concatenate([z0[None, :], zs], axis=0)


# batch the tracking rollout over an (M,2) grid of disturbances, everything else shared
_batch_tracked_rollout = jit(
    vmap(_tracked_rollout, in_axes=(None, None, None, None, None, None, None,
                                     None, None, None, None, None, 0, None, None,
                                     None, None))
)


class VelPoseDynamics:
    """Thin host-side wrapper over the jitted JAX kernels for the 5-D planner field
    g(z,u): the pose block is exact kinematics, the velocity block stacks the nominal
    linear model A_n, B_n with the learned residual Phi from residual_phi_predictor.py."""

    def __init__(self, a_diag, b_diag, phi_params_path=None):
        self.p = jnp.array([a_diag[0], a_diag[1], b_diag[0], b_diag[1]], dtype=jnp.float64)
        if phi_params_path is None:
            self.phi_params = _load_phi_params()
        else:
            self.phi_params = _load_phi_params(phi_params_path)

    # ---- single-sample API (numpy in / numpy out) --------------------------
    def deriv(self, z, u):
        return np.asarray(_deriv(jnp.asarray(z), jnp.asarray(u), self.p, self.phi_params))

    def step(self, z, u, dt):
        return np.asarray(_step(jnp.asarray(z), jnp.asarray(u), float(dt), self.p, self.phi_params))

    def propagate(self, z0, u, nsteps, dt):
        return np.asarray(_rollout(jnp.asarray(z0, dtype=jnp.float64),
                                    jnp.asarray(u, dtype=jnp.float64),
                                    int(nsteps), float(dt), self.p, self.phi_params))

    # ---- batched API (the JAX magic) --------------------------------------
    def batch_propagate(self, z0, U, nsteps, dt):
        return np.asarray(_batch_rollout(jnp.asarray(z0, dtype=jnp.float64),
                                          jnp.asarray(U, dtype=jnp.float64),
                                          int(nsteps), float(dt), self.p, self.phi_params))

    def batch_jacobians(self, Z, U, nsteps, dt):
        """Linearise the whole SCP horizon at once: for every edge k, A_k = df_edge/ds_k,
        B_k = df_edge/du_k, f_k = f_edge(s_k,u_k), where f_edge is the nsteps-step,
        constant-control RK4 propagation map (LaTeX Sec. II/V, "Linearized Dynamics").
        Z (K,5) anchor states, U (K,2) anchor controls -> A (K,5,5), B (K,5,2), f (K,5).
        Replaces SCP's per-edge Python loop with one compiled, vmapped, exact-autodiff call."""
        A, B, f = _batch_edge_jac(jnp.asarray(Z, dtype=jnp.float64),
                                   jnp.asarray(U, dtype=jnp.float64),
                                   int(nsteps), float(dt), self.p, self.phi_params)
        return np.asarray(A), np.asarray(B), np.asarray(f)

    # ---- closed-loop tracking ensemble (obstacle-risk CVaR, numpy in/out) -------
    def reference_traj(self, z0, u_ref, nsteps, dt):
        """Nominal reference rollout Z_ref (N+1,5) plus its per-step velocity/
        derivative V_ref, Vdot_ref (N,2) each, all numpy."""
        z0j = jnp.asarray(z0, dtype=jnp.float64)
        u_refj = jnp.asarray(u_ref, dtype=jnp.float64)
        Zref = _rollout(z0j, u_refj, int(nsteps), float(dt), self.p, self.phi_params)
        Vr, Vdotr = _ref_vel_deriv(Zref, u_refj, self.p, self.phi_params)
        return np.asarray(Zref), np.asarray(Vr), np.asarray(Vdotr)

    @staticmethod
    def _gain_matrix(G):
        """Normalize a scalar, (2,) diagonal, or (2,2) gain into a (2,2) matrix."""
        Gj = jnp.asarray(G, dtype=jnp.float64)
        if Gj.ndim == 0:
            return Gj * jnp.eye(2, dtype=jnp.float64)
        if Gj.ndim == 1:
            return jnp.diag(Gj)
        return Gj

    def tracked_ensemble(self, z0, u_ref, nsteps, dt, D, K, clip_lo, clip_hi,
                          Kp, k_psi, v_eps_speed, u_max):
        """Closed-loop tracking ensemble: track the nominal reference trajectory from
        z0 under constant control u_ref, while each member independently absorbs a
        fixed disturbance d in D (M,2) on (dot(v_b), dot(omega)). Two nested loops:
        an OUTER pose-feedback loop (Kp, k_psi, v_eps_speed) corrects the reference
        velocity from each member's ACTUAL position/heading error against the nominal
        trajectory, feeding an INNER model-inversion velocity tracker (K). K/Kp:
        scalar, (2,) diagonal, or full (2,2) feedback gain. u_max = (v_max, w_max):
        the tracking controller's OWN commanded u is clipped to [-u_max, u_max]
        every step -- the model-inversion correction is not naturally bounded and
        can (measured: routinely does) exceed the command range the residual net
        was trained on; caller must pass u_ref already within this same range.
        Returns numpy (M, nsteps+1, 5)."""
        z0j = jnp.asarray(z0, dtype=jnp.float64)
        u_refj = jnp.asarray(u_ref, dtype=jnp.float64)
        Zref = _rollout(z0j, u_refj, int(nsteps), float(dt), self.p, self.phi_params)
        P_d, Theta_d, V_d_I = _ref_pose_vel(Zref)
        Kmat = self._gain_matrix(K)
        Kpmat = self._gain_matrix(Kp)
        v_eps_sq = float(v_eps_speed) #** 2
        u_maxj = jnp.asarray(u_max, dtype=jnp.float64)
        Zens = _batch_tracked_rollout(z0j, u_refj, P_d, Theta_d, V_d_I, float(dt), self.p,
                                       self.phi_params, Kpmat, float(k_psi), v_eps_sq,
                                       Kmat, jnp.asarray(D, dtype=jnp.float64),
                                       jnp.asarray(clip_lo, dtype=jnp.float64),
                                       jnp.asarray(clip_hi, dtype=jnp.float64),
                                       -u_maxj, u_maxj)
        return np.asarray(Zens)
