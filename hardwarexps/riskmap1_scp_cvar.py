"""
AO-RRT + RiskAwareSCP (CVaR risk, closed-loop feedback tracking) on the
pcd-derived riskmap1 arena -- the risk-aware analogue of
hardwarexps/test_exp.py (plain AO-RRT, no risk) and the pcd-map analogue of
test_scp_vel_plot.py (AO-RRT + RiskAwareSCP, synthetic map).

Pipeline:
    1. Load riskmap1_sdf.npz / riskmap1_polygons.npz and run AO-RRT exactly
       like hardwarexps/test_exp.py does (PCDEnvironment/load_pcd_env
       REUSED, imported from test_exp.py, not duplicated).
    2. Build the Gaussian obstacle-boundary risk grid exactly like
       hardwarexps/riskmap1_boundary_risk.py does (rasterize_obstacle_id /
       assign_sigma / fused_obstacle_risk REUSED, imported, not duplicated) --
       always against riskmap1's OWN polygons (kinds/polygons come from the
       SAME load_pcd_env() call used for AO-RRT's collision map, so the risk
       grid's obstacles can never drift out of sync with AO-RRT's collision
       geometry, regardless of what riskmap1_boundary_risk.py's own default
       POLY_NAME currently points at).
    3. Wrap PCDEnvironment in PCDRiskEnv (defined in THIS file only), adding
       the two things RiskAwareSCP/EdgeRiskEvaluator need beyond AO-RRT's own
       read-only surface: a differentiable sdf_and_grad (src.environment's
       own bilinear+autodiff primitive, reused unmodified) and a static
       terrain risk grid (_risk_dev) -- the risk grid from step 2, wired in
       exactly where map/risk_map.py's TerrainRiskMap output goes for the
       synthetic map.
    4. Run RiskAwareSCP.solve() on the AO-RRT seed path. Its CVaR risk term
       (src/scp_vel.py's EdgeRiskEvaluator) already simulates the CLOSED-LOOP
       tracked rollout under a model-inversion feedback controller
       (src/dynamics_vel.py's _tracked_rollout/_closed_loop_u) for every edge
       under every disturbance scenario -- that machinery is not
       reimplemented here, just exercised, by wiring env/cfg/model into the
       existing RiskAwareSCP/EdgeRiskEvaluator classes unmodified.

Nothing in src/, map/, or any other existing file is edited. test_exp.py and
riskmap1_boundary_risk.py are imported read-only for their pure
environment-loading / risk-grid-building functions, same pattern test_exp.py
already uses for src.environment's bilinear helpers.

    python hardwarexps/riskmap1_scp_cvar.py [seed] [iters]
"""
import os
import sys
import time
import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in hardwarexps/)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(HERE)

from src.config import PlannerConfig                                    # noqa: E402
from src.ao_rrt import AORRT                                            # noqa: E402
from src.scp_vel import RiskAwareSCP, edges_from_chain                  # noqa: E402
from src.environment import _sdf_value_and_grad                        # noqa: E402  (read-only reuse, not modified)

from test_exp import PCDEnvironment, load_pcd_env                       # noqa: E402  (read-only reuse, not modified)
from riskmap1_boundary_risk import (                                    # noqa: E402  (read-only reuse, not modified)
    rasterize_obstacle_id, assign_sigma, fused_obstacle_risk,
    MANUAL_SIGMA, RANDOM_SIGMA_RANGE, DEFAULT_SEED,
)
from test_scp_vel_plot import control_cost, total_risk, joint_fail_prob  # noqa: E402  (read-only reuse, not modified)

START = (1.00, 0.5)
GOAL = (3.0, 3.5)


class PCDRiskEnv(PCDEnvironment):
    """PCDEnvironment (test_exp.py's read-only AO-RRT collision surface) plus
    the two extra things RiskAwareSCP/EdgeRiskEvaluator need: a
    differentiable SDF (sdf_and_grad) and a static terrain risk grid
    (_risk_dev), both queried with the SAME cell-coordinate convention
    PCDEnvironment already uses ((pts - origin) * resolution) and the same
    bilinear+autodiff primitive src/environment.py's real Environment class
    uses for both of its own sdf_and_grad / risk_and_grad."""

    def __init__(self, base_env, risk_grid):
        self.__dict__.update(base_env.__dict__)   # copy sdf/origin/resolution/... already loaded once
        self._sdf_dev = jnp.asarray(self.sdf)
        self._risk_dev = jnp.asarray(risk_grid)

    def sdf_and_grad(self, pts):
        """(N,2) m -> (vals (N,), grads (N,2)), both metres -- required by
        RiskAwareSCP's linearized SDF keep-out constraint (src/scp_vel.py)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        cells = (jnp.asarray(pts) - jnp.asarray(self.origin)) * self.resolution
        v, g = _sdf_value_and_grad(self._sdf_dev, cells)
        return np.asarray(v), np.asarray(g) * self.resolution

    def risk_vals(self, pts):
        """(N,2) m -> (N,) terrain risk in [0,1] -- used by joint_fail_prob
        (test_scp_vel_plot.py), reused unmodified below."""
        pts = np.atleast_2d(np.asarray(pts, float))
        cells = (jnp.asarray(pts) - jnp.asarray(self.origin)) * self.resolution
        v, _ = _sdf_value_and_grad(self._risk_dev, cells)
        return np.asarray(v)


def build_risk_env(cfg, seed):
    """Load riskmap1 once (AO-RRT's own collision map), build the Gaussian
    obstacle-boundary risk grid against those SAME polygons, and wrap both
    into one PCDRiskEnv."""
    base_env, kinds, polygons = load_pcd_env(cfg)
    obstacle_id = rasterize_obstacle_id(base_env.sdf.shape, base_env.res_m, base_env.origin, kinds, polygons)
    s_k = assign_sigma(obstacle_id, MANUAL_SIGMA, RANDOM_SIGMA_RANGE, seed)
    risk_grid = fused_obstacle_risk(obstacle_id, s_k, base_env.res_m)
    env = PCDRiskEnv(base_env, risk_grid)
    return env, kinds, polygons


def plot_result(env, pl, kinds, polygons, paths, obst_title, risk_title):
    """Obstacle map (left, rock polygons like test_exp.py) + Gaussian risk
    field (right, riskmap1_boundary_risk.py's own colormap convention --
    light/pale = low risk, dark red = high risk), both with the AO-RRT tree
    (left only) and whatever trajectories are passed in `paths`
    (list of (S, color, label))."""
    fig, (axm, axr) = plt.subplots(1, 2, figsize=(14, 7))
    ext = [0, env.width, 0, env.height]

    for kind, poly in zip(kinds, polygons):
        poly = np.asarray(poly, float)
        closed = np.vstack([poly, poly[0]])
        if kind == "map":
            axm.plot(closed[:, 0], closed[:, 1], "-", color="k", lw=1.2, zorder=2)
        else:
            axm.fill(closed[:, 0], closed[:, 1], color="0.6", alpha=0.85, zorder=1)
    for e in pl.tree_edges():
        axm.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)
    axm.set_title(obst_title)

    risk_im = axr.imshow(np.asarray(env._risk_dev), origin="lower", extent=ext, cmap="YlOrRd",
                          vmin=0, vmax=1, alpha=0.9, interpolation="bilinear")
    fig.colorbar(risk_im, ax=axr, fraction=0.046, pad=0.04, label="terrain risk (p_fail)")
    axr.set_title(risk_title)

    for ax in (axm, axr):
        for S, color, label in paths:
            ax.plot(S[:, 0], S[:, 1], "-o", color=color, lw=2.0, ms=3, label=label, zorder=3)
        ax.plot(*START, "o", color="lime", ms=12, mec="k", zorder=5, label="start")
        ax.plot(*GOAL, "*", color="magenta", ms=18, mec="k", zorder=5, label="goal")
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    print("\nshowing plot -- close the window to exit")
    plt.show()


def main():
    args = sys.argv[1:]
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 42
    iters = nums[1] if len(nums) > 1 else 80000

    np.random.seed(seed)
    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = iters
    cfg.risk.dist_grid_n = 10   # keep the disturbance ensemble small/fast (test_scp_vel_plot.py's default)

    env, kinds, polygons = build_risk_env(cfg, seed)
    print(f"pcd map {env.width:.2f}x{env.height:.2f} m, seed {seed}, {iters} AO-RRT iters")
    assert env.valid(*START), f"start {START} not collision-free (sdf={env.sdf_at(*START):.3f})"
    assert env.valid(*GOAL), f"goal {GOAL} not collision-free (sdf={env.sdf_at(*GOAL):.3f})"

    pl = AORRT(env, cfg, START, GOAL, start_heading=math.pi / 2)   # face +y regardless of goal direction
    pl.plan(verbose=True)
    chain = pl._chain()   # best-effort (nearest-to-goal) path even if goal wasn't reached

    if not pl.goal_reached():
        S_best, _, nsteps_best = edges_from_chain(chain)
        print("AO-RRT did not reach the goal -- try more iterations or a different seed. "
              "Showing the obstacle/risk course and the tree's best-effort path so far.")
        plot_result(env, pl, kinds, polygons,
                    paths=[(S_best, "tab:cyan", f"AO-RRT best effort (K={len(nsteps_best)}, goal NOT reached)")],
                    obst_title=f"Obstacle map (AO-RRT did NOT reach goal, {iters} iters, {len(pl.nodes)} nodes)",
                    risk_title="Gaussian obstacle-boundary risk field")
        return

    S_bar0, U_bar0, nsteps_k = edges_from_chain(chain)
    print(f"AO-RRT path: K={len(nsteps_k)} edges, cost={pl.best_cost:.3f}")

    scp = RiskAwareSCP(env, cfg, pl.model)
    t0 = time.perf_counter()
    S_bar, U_bar, info = scp.solve(chain, GOAL, verbose=True)
    t1 = time.perf_counter()
    print(f"\nSCP refine: {info['iters']} iters, {info['n_solves']} solves, "
          f"converged={info['converged']}, {t1 - t0:.2f}s")

    # ---- control cost comparison --------------------------------------------
    svp = cfg.scp_vel
    dt = cfg.dyn.dt
    for label, dw in [("duration-weighted (r_v*v^2+r_omega*w^2)*nsteps*dt", True),
                       ("raw sum(u^2), no weighting", False)]:
        c0 = control_cost(U_bar0, nsteps_k, dt, svp.r_v, svp.r_omega, dw)
        c1 = control_cost(U_bar, nsteps_k, dt, svp.r_v, svp.r_omega, dw)
        pct = 100.0 * (c1 - c0) / c0 if c0 != 0 else float("nan")
        print(f"\n[{label}]")
        print(f"  AO-RRT seed control cost : {c0:.5f}")
        print(f"  SCP refined control cost : {c1:.5f}  ({pct:+.1f}%)")

    # ---- total risk (CVaR) comparison ---------------------------------------
    from src.scp_vel import EdgeRiskEvaluator
    risk_eval = EdgeRiskEvaluator(env, cfg, pl.model)
    R0 = risk_eval.edge_risk(S_bar0[:-1], U_bar0, nsteps_k)
    R1 = risk_eval.edge_risk(S_bar[:-1], U_bar, nsteps_k)
    alpha = cfg.risk.alpha
    tr0 = total_risk(R0, alpha)
    tr1 = total_risk(R1, alpha)
    pct_r = 100.0 * (tr1 - tr0) / tr0 if tr0 != 0 else float("nan")
    print(f"\n[total risk / collision probability -- sum_k CVaR_{alpha}(R_k,:)]")
    print(f"  AO-RRT seed total risk   : {tr0:.5f}  (mean R={R0.mean():.4f}, max R={R0.max():.4f})")
    print(f"  SCP refined total risk   : {tr1:.5f}  (mean R={R1.mean():.4f}, max R={R1.max():.4f})  ({pct_r:+.1f}%)")

    # ---- joint collision probability across knots ---------------------------
    pj0, ls0 = joint_fail_prob(env, S_bar0)
    pj1, ls1 = joint_fail_prob(env, S_bar)
    print(f"\n[joint collision probability -- P_fail = 1 - prod_k(1 - p_fail(knot_k))]")
    print(f"  AO-RRT seed  P_fail (joint, {S_bar0.shape[0]} knots) : {pj0:.6f}  (log P_survive = {ls0:.2f})")
    print(f"  SCP refined  P_fail (joint, {S_bar.shape[0]} knots)  : {pj1:.6f}  (log P_survive = {ls1:.2f})")
    if pj0 == 1.0 and pj1 == 1.0:
        print(f"  NOTE: both round to 1.0 (float64 underflow). SCP is exp({ls1 - ls0:+.2f}) = "
              f"{np.exp(ls1 - ls0):.3e}x as likely to survive as the AO-RRT seed.")

    # ---- plot -----------------------------------------------------------------
    plot_result(env, pl, kinds, polygons,
                paths=[(S_bar0, "tab:cyan", f"AO-RRT seed (K={len(nsteps_k)})"),
                       (S_bar, "tab:red", "RiskAwareSCP")],
                obst_title=f"Obstacle map ({info['iters']} SCP iters, converged={info['converged']})",
                risk_title=f"Risk field  (total risk {tr0:.2f} -> {tr1:.2f}, {pct_r:+.1f}%)")


if __name__ == "__main__":
    main()
