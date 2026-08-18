"""
RiskSensitiveAORRT (src/risk_planner.py) + RiskAwareSCP (src/scp_vel.py) on
riskmap1, both driven by the SAME per-obstacle sigma array.

Hard collision (shared by both planners) now checks the RISK-INFLATED SDF:
AO-RRT's path_free/collision_free (inherited from PCDEnvironment, but reading
self.sdf which this script's PCDRiskSDFEnv now points at risk_sdf, not the
plain field) and SCP's linearized SDF keep-out (sdf_and_grad, also
overridden to read the same inflated field) both require disc_radius/d_safe
clearance PLUS each obstacle's own kappa*sigma_k push-out. This is a
DELIBERATE reversion of an earlier decoupled design (risk as soft cost only,
hard collision against the plain SDF) -- see git/.backups history if that
behavior is ever needed again. Because the margin now stacks disc_radius/
d_safe PLUS kappa*sigma_k, START/GOAL or corridor points that were valid
under the plain SDF can become infeasible here; this script still asserts
env.valid(*START)/env.valid(*GOAL) up front against the INFLATED field, so
an infeasible start/goal fails fast with a clear message rather than wasting
a full AO-RRT run.

Two SEPARATE risk-cost fields, built from ONE shared sigma array `s_k`
(rasterize_obstacle_id / assign_sigma / MANUAL_SIGMA / RANDOM_SIGMA_RANGE,
imported UNMODIFIED from riskmap1_boundary_risk.py) -- each planner consumes
its own native shape, same underlying obstacle-boundary uncertainty:

    - RiskSensitiveAORRT._edge_tracking_cvar (NN-dynamics tracked-rollout
      CVaR-of-penetration, ENABLED via use_edge_cvar=True; the OTHER, older
      risk_weight/_edge_risk slip-CVaR term is DISABLED -- risk_weight=0 AND
      env.risk=None): reads env.sdf_device, the per-obstacle CVaR-inflated
      SIGNED-DISTANCE field (map.risk_sdf.build_risk_sdf, same construction
      as riskmap1_risk_sdf.py) built from s_k.
    - RiskAwareSCP's EdgeRiskEvaluator (src/scp_vel.py -- "the SCP CVaR does
      the same thing"): reads env._risk_dev, the Gaussian per-obstacle
      PROBABILITY field (riskmap1_boundary_risk.fused_obstacle_risk, same
      construction as riskmap1_boundary_risk.py) built from the SAME s_k.

cfg.risk.alpha is used for both build_risk_sdf's alpha (-> kappa) and
RiskSensitiveAORRT's/RiskAwareSCP's own CVaR alpha, so every CVaR evaluation
in this script shares one risk level.

Nothing in src/ or map/ is edited -- everything under src/, map/, test_exp.py,
riskmap1_boundary_risk.py, and test_scp_vel_plot.py is imported and used
read-only (test_scp_vel_plot.py lives at the repo root, same read-only-import
pattern hardwarexps/riskmap1_scp_cvar.py already uses for its cost/risk
comparison helpers).

    python hardwarexps/pcd_risk_aware_planner.py [seed] [iters]
"""
import os
import sys
import time
import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in hardwarexps/)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(HERE)

from src.config import PlannerConfig                                    # noqa: E402
from src.risk_planner import RiskSensitiveAORRT                        # noqa: E402  (read-only reuse, not modified)
from src.scp_vel import RiskAwareSCP, EdgeRiskEvaluator, edges_from_chain  # noqa: E402  (read-only reuse, not modified)
from src.environment import _sdf_value_and_grad                        # noqa: E402  (read-only reuse, not modified)
from map.risk_sdf import build_risk_sdf                                 # noqa: E402  (read-only reuse, not modified)

from test_exp import PCDEnvironment, load_pcd_env                       # noqa: E402  (read-only reuse, not modified)
from riskmap1_boundary_risk import (                                    # noqa: E402  (read-only reuse, not modified)
    rasterize_obstacle_id, assign_sigma, fused_obstacle_risk,
    MANUAL_SIGMA, RANDOM_SIGMA_RANGE, DEFAULT_SEED,
)
from test_scp_vel_plot import control_cost, total_risk, joint_fail_prob  # noqa: E402  (read-only reuse, not modified)

PCD_DIR = r"C:\Users\skelkar37\GT\Research F26\pcd\data\processed_pcd"
SDF_NAME = "riskmap6_polygons_cleaned_sdf.npz"
POLY_NAME = "riskmap6_polygons_cleaned.npz"   # explicit, pinned here -- test_exp.py's own SDF_NAME/POLY_NAME
                                       # defaults have since drifted to riskmap2; this script never
                                       # relies on load_pcd_env's defaults, always passes these explicitly

START = (0.75, 0.0)
GOAL = (2.8, 3.4)

# True  -> hard collision (AO-RRT path_free/collision_free AND SCP's sdf_and_grad)
#          checks the risk-inflated SDF (disc_radius/d_safe PLUS kappa*sigma_k
#          push-out per obstacle).
# False -> hard collision reverts to the PLAIN, un-inflated SDF (the original
#          decoupled behavior this module's docstring used to describe) --
#          risk still shapes soft edge/knot COST wherever it already did
#          (env.sdf_device / RiskSensitiveAORRT's CVaR kernel, env._risk_dev /
#          RiskAwareSCP's EdgeRiskEvaluator are untouched by this flag either
#          way), only the hard keep-out margin changes.
USE_RISK_INFLATED_SDF = False

# Set True (or pass "savecsv" on the command line) to write the SCP-refined
# trajectory to results/. Off by default so exploratory runs don't pile up files.
SAVE_TRAJECTORY_CSV = False
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Set True (or pass "plotvel" on the command line) to open a SECOND window
# plotting the SCP-refined trajectory's v_b and omega vs. time (two stacked
# subplots). Off by default -- when False, that window/subplot is never built.
PLOT_VEL_OMEGA = True


class PCDRiskSDFEnv(PCDEnvironment):
    """PCDEnvironment (test_exp.py's read-only AO-RRT collision surface).
    self.sdf is swapped to the risk-inflated field when use_risk_inflated_sdf
    is True (the default, gated by the module-level USE_RISK_INFLATED_SDF
    flag) -- hard collision (path_free/collision_free, inherited unchanged,
    but now reading the swapped self.sdf) and SCP's own sdf_and_grad keep-out
    both then check against the INFLATED field, so every knot must clear
    disc_radius/d_safe PLUS that obstacle's own kappa*sigma_k push-out, not
    just the plain geometric margin. With the flag False, self.sdf/_sdf_dev
    stay the ORIGINAL plain field instead, and self._plain_sdf_dev is always
    kept on hand either way (unused unless the flag is False, or for
    reference/debugging).
        .sdf_device -> the CVaR-inflated risk_sdf, ALWAYS (regardless of this
                       flag), read by RiskSensitiveAORRT._edge_tracking_cvar's
                       kernel -- that's a separate SOFT cost term, not the
                       hard collision check this flag controls.
        ._risk_dev  -> Gaussian per-obstacle p_fail field, read by
                       RiskAwareSCP's EdgeRiskEvaluator (also a SEPARATE
                       soft-cost field, unrelated to this flag).
    env.risk is fixed at None so RiskSensitiveAORRT._edge_risk (the disabled
    slip-CVaR term) short-circuits to 0.0 unconditionally, not just via
    risk_weight=0."""

    def __init__(self, base_env, risk_sdf, risk_dev, use_risk_inflated_sdf=True):
        self.__dict__.update(base_env.__dict__)   # copy origin/resolution/disc_radius/sdf/... already loaded once
        self._plain_sdf_dev = jnp.asarray(self.sdf)  # original un-inflated field, always kept on hand
        if use_risk_inflated_sdf:
            self.sdf = risk_sdf                       # HARD collision (inherited path_free/collision_free/sdf_at,
                                                        # all read self.sdf) now checks the INFLATED field directly
            self._sdf_dev = jnp.asarray(self.sdf)     # on-device copy of the SAME inflated field, for sdf_and_grad
        else:
            self._sdf_dev = self._plain_sdf_dev       # hard collision stays on the PLAIN field (self.sdf untouched)
        self._risk_sdf_dev = jnp.asarray(risk_sdf)    # env.sdf_device always reads the inflated field (soft cost)
        self._risk_dev = jnp.asarray(risk_dev)        # Gaussian p_fail grid, for RiskAwareSCP's EdgeRiskEvaluator
        self.risk = None                               # -> RiskSensitiveAORRT._edge_risk always returns 0.0

    @property
    def sdf_device(self):
        return self._risk_sdf_dev   # RiskSensitiveAORRT's CVaR-of-penetration kernel reads the INFLATED field

    def sdf_and_grad(self, pts):
        """(N,2) m -> (vals (N,), grads (N,2)), both metres -- RiskAwareSCP's
        linearized SDF keep-out constraint. Reads self._sdf_dev, now the
        RISK-INFLATED field (same as self.sdf/AO-RRT's own collision check),
        so SCP's hard constraint and AO-RRT's hard constraint agree on what
        counts as a collision."""
        pts = np.atleast_2d(np.asarray(pts, float))
        cells = (jnp.asarray(pts) - jnp.asarray(self.origin)) * self.resolution
        v, g = _sdf_value_and_grad(self._sdf_dev, cells)
        return np.asarray(v), np.asarray(g) * self.resolution

    def risk_vals(self, pts):
        """(N,2) m -> (N,) terrain risk in [0,1] -- used by joint_fail_prob
        (test_scp_vel_plot.py). Reads self._risk_dev, the SAME Gaussian
        per-obstacle field RiskAwareSCP's EdgeRiskEvaluator reads."""
        pts = np.atleast_2d(np.asarray(pts, float))
        cells = (jnp.asarray(pts) - jnp.asarray(self.origin)) * self.resolution
        v, _ = _sdf_value_and_grad(self._risk_dev, cells)
        return np.asarray(v)


def build_risk_env(cfg, seed):
    """Load riskmap1's true SDF/polygons (AO-RRT's own collision geometry),
    build ONE shared per-obstacle sigma array, then both risk-cost fields
    from it (risk_sdf for AO-RRT's kernel, risk_dev for SCP's evaluator)."""
    base_env, kinds, polygons = load_pcd_env(cfg, pcd_dir=PCD_DIR, sdf_name=SDF_NAME, poly_name=POLY_NAME)
    obstacle_id = rasterize_obstacle_id(base_env.sdf.shape, base_env.res_m, base_env.origin, kinds, polygons)
    s_k = assign_sigma(obstacle_id, MANUAL_SIGMA, RANDOM_SIGMA_RANGE, seed)
    K = int(obstacle_id.max())
    obstacle_masks = np.stack([obstacle_id == (k + 1) for k in range(K)], axis=0)
    risk_sdf, argmin_idx, kappa = build_risk_sdf(
        obstacle_masks, s_k, res=base_env.res_m, alpha=cfg.risk.alpha, d_max=cfg.risk_sdf.d_max)
    risk_dev = fused_obstacle_risk(obstacle_id, s_k, base_env.res_m)
    env = PCDRiskSDFEnv(base_env, risk_sdf, risk_dev, use_risk_inflated_sdf=USE_RISK_INFLATED_SDF)
    return env, kinds, polygons, s_k, kappa


def plot_result(env, pl, kinds, polygons, paths):
    # env.origin is the map's true world-frame lower-left corner (not necessarily
    # (0,0) -- riskmap5's arena origin is (-0.225, -0.535)). env._risk_dev's cell
    # [0,0] corresponds to env.origin (see riskmap1_boundary_risk.rasterize_obstacle_id),
    # so imshow's extent must start there too, or the risk heatmap renders shifted
    # away from the (true-world-coordinate) path/start/goal plotted on top of it.
    ext = [env.origin[0], env.origin[0] + env.width, env.origin[1], env.origin[1] + env.height]

    fig_m, axm = plt.subplots(1, 1, figsize=(7, 7))
    for kind, poly in zip(kinds, polygons):
        poly = np.asarray(poly, float)
        closed = np.vstack([poly, poly[0]])
        if kind == "map":
            axm.plot(closed[:, 0], closed[:, 1], "-", color="k", lw=1.2, zorder=2)
        else:
            axm.fill(closed[:, 0], closed[:, 1], color="0.6", alpha=0.85, zorder=1)
    for e in pl.tree_edges():
        axm.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)

    fig_r, axr = plt.subplots(1, 1, figsize=(7, 7))
    risk_im = axr.imshow(np.asarray(env._risk_dev), origin="lower", extent=ext, cmap="YlOrRd",
                          vmin=0, vmax=1, alpha=0.9, interpolation="bilinear")
    # make_axes_locatable ties the colorbar to axr's actual (aspect-shrunk) size
    # instead of the raw figure width -- fig.colorbar(ax=axr) reserves its slice
    # BEFORE set_aspect("equal") shrinks axr, leaving a dead gap and an
    # oversized window otherwise.
    cax = make_axes_locatable(axr).append_axes("right", size="4%", pad=0.08)
    fig_r.colorbar(risk_im, cax=cax, label="terrain risk (p_fail)")

    for ax in (axm, axr):
        for S, color, label in paths:
            ax.plot(S[:, 0], S[:, 1], "-o", color=color, lw=2.0, ms=3, label=label, zorder=3)
        ax.plot(*START, "o", color="lime", ms=12, mec="k", zorder=5, label="start")
        ax.plot(*GOAL, "*", color="magenta", ms=18, mec="k", zorder=5, label="goal")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="upper left", fontsize=8)

    fig_m.tight_layout()
    fig_r.tight_layout()
    print("\nshowing plots -- close both windows to exit")
    plt.show()


def plot_vel_omega_profile(S_bar, nsteps_k, dt):
    """SCP-refined v_b (forward speed) and omega (yaw rate) vs. time, each
    knot's timestamp being the cumulative RK4 duration of edges 0..k-1 (same
    convention as save_trajectory_csv). Opens its own figure/window -- only
    called when PLOT_VEL_OMEGA (or "plotvel") is set, so this second window
    never appears otherwise."""
    t = np.concatenate([[0.0], np.cumsum(np.asarray(nsteps_k) * dt)])
    v_b, omega = S_bar[:, 3], S_bar[:, 4]

    fig, (axv, axw) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axv.plot(t, v_b, "-o", color="tab:blue", lw=1.5, ms=3)
    axv.set_ylabel("v_b (m/s)")
    axw.plot(t, omega, "-o", color="tab:red", lw=1.5, ms=3)
    axw.set_ylabel("omega (rad/s)")
    axw.set_xlabel("t (s)")
    fig.tight_layout()


def save_trajectory_csv(S_bar, nsteps_k, dt, out_path):
    """S_bar: (K+1,5) SCP-refined knot states [x, y, theta, v_b, omega] (same
    state layout as ao_rrt.py's x0 / dynamics_vel.py's convention -- global
    x, y, heading, body-frame forward speed, body-frame yaw rate). Knot k+1's
    timestamp is the cumulative RK4 duration of edges 0..k (nsteps_k * dt each,
    since SCP never changes edge duration -- only the continuous anchor moves).
    Writes x,y,theta,v_b,omega,t to out_path."""
    t = np.concatenate([[0.0], np.cumsum(np.asarray(nsteps_k) * dt)])
    data = np.column_stack([S_bar, t])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, data, delimiter=",",
               header="x,y,theta,v_b,omega,t", comments="")
    print(f"saved SCP trajectory csv -> {out_path}")

#TODO add flag to exit the code when goal is reached but keep going if flag is false
def main():
    args = sys.argv[1:]
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 101
    iters = nums[1] if len(nums) > 1 else 80000
    save_csv = SAVE_TRAJECTORY_CSV or ("savecsv" in args)
    plot_vel = PLOT_VEL_OMEGA or ("plotvel" in args)

    np.random.seed(seed)
    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = iters
    cfg.risk.risk_weight = 0.0     # disable _edge_risk (slip-CVaR, src/dynamics.py path)
    cfg.risk.use_edge_cvar = False  # enable _edge_tracking_cvar (NN-dynamics tracked-rollout CVaR)

    env, kinds, polygons, s_k, kappa = build_risk_env(cfg, seed)
    print(f"pcd map {env.width:.2f}x{env.height:.2f} m, seed {seed}, {iters} AO-RRT iters")
    print(f"alpha={cfg.risk.alpha}  ->  kappa={kappa:.4f}  (shared by risk_sdf's inflation, "
          f"RiskSensitiveAORRT's own CVaR, and RiskAwareSCP's CVaR)")
    for i, sigma in enumerate(s_k, start=1):
        tag = " (manual)" if i in MANUAL_SIGMA else " (random)"
        print(f"  idx {i}: sigma={sigma:.3f}{tag}  ->  push-out={kappa * sigma:.3f} m")
    assert env.valid(*START), f"start {START} not collision-free (sdf={env.sdf_at(*START):.3f})"
    assert env.valid(*GOAL), f"goal {GOAL} not collision-free (sdf={env.sdf_at(*GOAL):.3f})"

    start_heading = math.atan2(GOAL[1] - START[1], GOAL[0] - START[0])  # face straight at GOAL
    pl = RiskSensitiveAORRT(env, cfg, START, GOAL, start_heading=np.pi / 2)   # always face +y
    pl.plan(verbose=True)
    chain = pl._chain()   # best-effort (nearest-to-goal) path even if goal wasn't reached
    print(f"RiskSensitiveAORRT: goal={pl.goal_reached()} cost={pl.best_cost:.3f} "
          f"nodes={len(pl.nodes)} time={pl.timers['total']:.2f}s "
          f"t_first={pl.t_first if pl.t_first else float('nan'):.2f}s")

    if not pl.goal_reached():
        S_best, _, nsteps_best = edges_from_chain(chain)
        print("RiskSensitiveAORRT did not reach the goal -- showing the tree/best-effort path only, no SCP.")
        plot_result(env, pl, kinds, polygons,
                    paths=[(S_best, "tab:cyan", f"RiskSensitiveAORRT best effort (K={len(nsteps_best)})")])
        return

    S_bar0, U_bar0, nsteps_k = edges_from_chain(chain)
    print(f"AO-RRT path: K={len(nsteps_k)} edges, cost={pl.best_cost:.3f}")

    scp = RiskAwareSCP(env, cfg, pl.model)
    t0 = time.perf_counter()
    S_bar, U_bar, info = scp.solve(chain, GOAL, verbose=True)
    t1 = time.perf_counter()
    print(f"\nSCP refine: {info['iters']} iters, {info['n_solves']} solves, "
          f"converged={info['converged']}, {t1 - t0:.2f}s")

    svp = cfg.scp_vel
    dt = cfg.dyn.dt

    if save_csv:
        out_path = os.path.join(RESULTS_DIR, f"riskmap1_risk_aware_scp_trajectory_seed{seed}.csv")
        save_trajectory_csv(S_bar, nsteps_k, dt, out_path)

    for label, dw in [("duration-weighted (r_v*v^2+r_omega*w^2)*nsteps*dt", True),
                       ("raw sum(u^2), no weighting", False)]:
        c0 = control_cost(U_bar0, nsteps_k, dt, svp.r_v, svp.r_omega, dw)
        c1 = control_cost(U_bar, nsteps_k, dt, svp.r_v, svp.r_omega, dw)
        pct = 100.0 * (c1 - c0) / c0 if c0 != 0 else float("nan")
        print(f"\n[{label}]")
        print(f"  AO-RRT seed control cost : {c0:.5f}")
        print(f"  Risk aware SCP refined control cost : {c1:.5f}  ({pct:+.1f}%)")

    risk_eval = EdgeRiskEvaluator(env, cfg, pl.model)
    R0 = risk_eval.edge_risk(S_bar0[:-1], U_bar0, nsteps_k)
    R1 = risk_eval.edge_risk(S_bar[:-1], U_bar, nsteps_k)
    alpha = cfg.risk.alpha
    tr0 = total_risk(R0, alpha)
    tr1 = total_risk(R1, alpha)
    pct_r = 100.0 * (tr1 - tr0) / tr0 if tr0 != 0 else float("nan")
    print(f"\n[total risk / collision probability -- sum_k CVaR_{alpha}(R_k,:)]")
    print(f"  AO-RRT seed total risk   : {tr0:.5f}  (mean R={R0.mean():.4f}, max R={R0.max():.4f})")
    print(f"  Risk Aware SCP total risk   : {tr1:.5f}  (mean R={R1.mean():.4f}, max R={R1.max():.4f})  ({pct_r:+.1f}%)")

    pj0, ls0 = joint_fail_prob(env, S_bar0)
    pj1, ls1 = joint_fail_prob(env, S_bar)
    print(f"\n[joint collision probability -- P_fail = 1 - prod_k(1 - p_fail(knot_k))]")
    print(f"  AO-RRT seed  P_fail (joint, {S_bar0.shape[0]} knots) : {pj0:.6f}  (log P_survive = {ls0:.2f})")
    print(f"  Risk Aware SCP  P_fail (joint, {S_bar.shape[0]} knots)  : {pj1:.6f}  (log P_survive = {ls1:.2f})")

    if plot_vel:
        plot_vel_omega_profile(S_bar, nsteps_k, dt)

    plot_result(env, pl, kinds, polygons,
                paths=[(S_bar0, "tab:cyan", f"AO-RRT"),
                       (S_bar, "tab:red", "Risk Aware SCP")])


if __name__ == "__main__":
    main()
