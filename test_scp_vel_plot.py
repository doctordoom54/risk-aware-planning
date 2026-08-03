"""
Run AO-RRT + RiskAwareSCP on the same small test map used elsewhere in this
session, plot the AO-RRT seed path vs the SCP-refined path, and compare net
control cost between the two trajectories. Opens an interactive plot window
(blocks until closed) instead of saving to disk.

    python test_scp_vel_plot.py [seed] [iters]
"""
import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.ao_rrt import AORRT
from src.scp_vel import RiskAwareSCP, EdgeRiskEvaluator, edges_from_chain
from src.risk_planner import cvar


def control_cost(U, nsteps_k, dt, r_v, r_omega, duration_weighted):
    per_edge = r_v * U[:, 0] ** 2 + r_omega * U[:, 1] ** 2
    if duration_weighted:
        per_edge = per_edge * (nsteps_k * dt)
    return float(np.sum(per_edge))


def joint_fail_prob(env, S):
    """P(fail somewhere along this trajectory) = 1 - prod_k(1 - p_fail(knot_k)),
    treating each knot's terrain failure as an independent event -- the SAME
    probabilistic-OR combination TerrainRiskMap.compute() already uses to fuse
    its 3 hazard LAYERS at one cell, just applied across a trajectory's KNOTS
    instead. Uses the static risk field (env.risk_vals) at each knot's (x,y),
    NOT the tracked-ensemble R_{k,m} -- a literal "does this planned path fail
    anywhere" probability, not a risk-aversion-weighted aggregate like
    total_risk().

    Returns (p_fail, log_survive): p_fail is the raw probability, but with
    ~100+ knots at realistic (non-tiny) per-knot risk, prod_k(1-p_k) underflows
    and p_fail rounds to EXACTLY 1.0 in float64 for any decent trajectory --
    mathematically correct, numerically useless for comparison. log_survive =
    sum_k log(1-p_fail_k) (via log1p, no cancellation) stays informative: it's
    log(P_survive), so a DIFFERENCE of D between two trajectories means one is
    exp(D) times more likely to survive, even when both raw p_fail print as
    1.0."""
    p_fail = env.risk_vals(S[:, 0:2])            # (K+1,)
    log_survive = float(np.sum(np.log1p(-p_fail)))
    return 1.0 - float(np.prod(1.0 - p_fail)), log_survive


def total_risk(R, alpha):
    """sum_k CVaR_alpha(R[k,:]) -- the SAME aggregate the QP's CVaR objective
    term penalizes (Sec. 2's sum_k(tau_k + mean_m(excess)/alpha)), just
    evaluated directly off the risk grid instead of the QP's tau/eta variables
    -- so it's comparable between a trajectory that never went through the QP
    (the AO-RRT seed) and one that did (the SCP-refined result). Reuses
    risk_planner.cvar (already the Rockafellar-Uryasev CVaR estimator the rest
    of this codebase's risk-sensitive planners use), not a new formula."""
    return float(sum(cvar(R[k], alpha) for k in range(R.shape[0])))


def plot_map(env, grid, pl, start, goal, paths, obst_title, risk_title):
    """Obstacle map (left) + risk field (right), both showing the full AO-RRT
    tree (faint grey, context for how the search explored) plus whatever
    trajectories are passed in `paths` (list of (S, color, label)). Used both
    when AO-RRT reaches the goal (paths = seed + SCP-refined) and when it
    doesn't (paths = just the best-effort/nearest-to-goal tree path), so the
    obstacle/risk course is always visible for inspection either way."""
    fig, (axm, axr) = plt.subplots(1, 2, figsize=(14, 7))
    ext = [0, env.width, 0, env.height]

    obst = np.ma.masked_where(grid.features == 0, grid.features)
    axm.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
               alpha=0.85, interpolation="bilinear")
    axm.set_title(obst_title)

    # risk field: env.risk.risk is the fused TerrainRiskMap p_fail raster (map/risk_map.py),
    # the SAME array EdgeRiskEvaluator reads (env._risk_dev) -- shown here for reference,
    # so it's visually apparent where the risk actually is relative to the path(s).
    risk_im = axr.imshow(env.risk.risk, origin="lower", extent=ext, cmap="inferno",
                          vmin=0, vmax=1, alpha=0.9, interpolation="bilinear")
    fig.colorbar(risk_im, ax=axr, fraction=0.046, pad=0.04, label="terrain risk (p_fail)")
    axr.set_title(risk_title)

    for e in pl.tree_edges():          # obstacle-map panel only -- clutters the risk heatmap
        axm.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)

    for ax in (axm, axr):
        for S, color, label in paths:
            ax.plot(S[:, 0], S[:, 1], "-o", color=color, lw=2.0, ms=3, label=label, zorder=3)
        ax.plot(*start, "o", color="lime", ms=12, mec="k", zorder=5, label="start")
        ax.plot(*goal, "*", color="magenta", ms=18, mec="k", zorder=5, label="goal")
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    print("\nshowing plot -- close the window to exit")
    plt.show()


def main():
    args = sys.argv[1:]
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 31269
    iters = nums[1] if len(nums) > 1 else 10000

    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = iters
    cfg.risk.dist_grid_n = 10

    W, H = cfg.env.width, cfg.env.height
    start = (0.15 * W, 0.13 * H)
    goal = (0.75 * W, 0.87 * H)
    grid, env = make_map_env(cfg, seed, num_rocks=6, start=start, goal=goal,
                              with_risk=True,num_craters=0, undulation=8,
                              risk_weights=(0.3, 0.3, 1.0),
                              obstacle_sigma_min=0.1, obstacle_sigma_max=0.5)

    pl = AORRT(env, cfg, start, goal, start_heading=math.pi / 2)   # always face +y
    pl.plan(verbose=False)
    chain = pl._chain()   # best-effort (nearest-to-goal) path even if goal wasn't reached

    if not pl.goal_reached():
        S_best, _, nsteps_best = edges_from_chain(chain)
        print("AO-RRT did not reach the goal -- try more iterations or a different seed. "
              "Showing the obstacle/risk course and the tree's best-effort path so far.")
        plot_map(env, grid, pl, start, goal,
                 paths=[(S_best, "tab:cyan",
                         f"AO-RRT best effort (K={len(nsteps_best)}, goal NOT reached)")],
                 obst_title=f"Obstacle map (AO-RRT did NOT reach goal, {iters} iters, "
                            f"{len(pl.nodes)} nodes)",
                 risk_title="Risk field")
        return
    S_bar0, U_bar0, nsteps_k = edges_from_chain(chain)
    print(f"AO-RRT path: K={len(nsteps_k)} edges, cost={pl.best_cost:.3f}")

    scp = RiskAwareSCP(env, cfg, pl.model)
    t0 = time.perf_counter()
    S_bar, U_bar, info = scp.solve(chain, goal, verbose=True)
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

    # ---- total risk (collision probability) comparison -----------------------
    risk_eval = EdgeRiskEvaluator(env, cfg, pl.model)
    R0 = risk_eval.edge_risk(S_bar0[:-1], U_bar0, nsteps_k)   # AO-RRT seed, (K,M)
    R1 = risk_eval.edge_risk(S_bar[:-1], U_bar, nsteps_k)      # SCP refined, (K,M)
    alpha = cfg.risk.alpha
    tr0 = total_risk(R0, alpha)
    tr1 = total_risk(R1, alpha)
    pct_r = 100.0 * (tr1 - tr0) / tr0 if tr0 != 0 else float("nan")
    print(f"\n[total risk / collision probability -- sum_k CVaR_{alpha}(R_k,:)]")
    print(f"  AO-RRT seed total risk   : {tr0:.5f}  (mean R={R0.mean():.4f}, max R={R0.max():.4f})")
    print(f"  SCP refined total risk   : {tr1:.5f}  (mean R={R1.mean():.4f}, max R={R1.max():.4f})  ({pct_r:+.1f}%)")

    # ---- joint collision probability across knots: 1 - prod_k(1 - p_fail(knot_k)) ----
    pj0, ls0 = joint_fail_prob(env, S_bar0)
    pj1, ls1 = joint_fail_prob(env, S_bar)
    print(f"\n[joint collision probability -- P_fail = 1 - prod_k(1 - p_fail(knot_k))]")
    print(f"  AO-RRT seed  P_fail (joint, {S_bar0.shape[0]} knots) : {pj0:.6f}  "
          f"(log P_survive = {ls0:.2f})")
    print(f"  SCP refined  P_fail (joint, {S_bar.shape[0]} knots)  : {pj1:.6f}  "
          f"(log P_survive = {ls1:.2f})")
    if pj0 == 1.0 and pj1 == 1.0:
        print(f"  NOTE: both round to 1.0 (float64 underflow -- {S_bar0.shape[0]}+ knots "
              f"at realistic per-knot risk always saturates this product). The REAL "
              f"comparison is log P_survive: SCP is exp({ls1 - ls0:+.2f}) = "
              f"{np.exp(ls1 - ls0):.3e}x as likely to survive as the AO-RRT seed.")

    # ---- plot: obstacle map (left) + risk field (right), both with both paths ----
    plot_map(env, grid, pl, start, goal,
             paths=[(S_bar0, "tab:cyan", f"AO-RRT seed (K={len(nsteps_k)})"),
                    (S_bar, "tab:red", "SCP refined")],
             obst_title=f"Obstacle map ({info['iters']} SCP iters, converged={info['converged']})",
             risk_title=f"Risk field  (total risk {tr0:.2f} -> {tr1:.2f}, {pct_r:+.1f}%)")


if __name__ == "__main__":
    main()
