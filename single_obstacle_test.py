"""
Special single-circular-obstacle course (start bottom-right corner, goal top-left
corner), planned TWICE for a direct, controlled comparison:

    1. risk-aware AO-RRT: CVaR edge cost (RiskParams.use_edge_cvar) at ALPHA, planned
       against the risk-inflated SDF (each obstacle's keep-out pushed out by
       kappa(ALPHA) * sigma -- map.risk_sdf.build_risk_sdf). The CVaR term itself is
       computed internally from a closed-loop tracked ensemble on every candidate
       edge (risk_planner.RiskSensitiveAORRT._edge_tracking_cvar) -- that's part of
       the planner's cost function, not a separate post-processing step here.
    2. plain AO-RRT: no CVaR edge cost -- pure control-effort/time cost, planned
       against the PLAIN (non-inflated) SDF.

Both runs share the SAME obstacle geometry, seed, and disc-radius body-collision
logic (Environment.path_free/collision_free, same cfg.env.disc_radius) -- only
env.sdf (risk-inflated vs nominal) and the planner's edge cost differ. The global
RNG is reseeded to SEED before each plan() call, so both trees sample the identical
extension sequence and any divergence between them is attributable to the cost/SDF
difference, not to different random draws.

Settable below: OBSTACLE_RADIUS, OBSTACLE_SIGMA_MIN/MAX, ALPHA, ITERS, SEED.
Everything else (cost mode, edge weights, tracking gains, command caps, ...) comes
from src.config.PlannerConfig's defaults, same as the rest of the repo.

    python single_obstacle_test.py

Opens two interactive plot windows (tree + path, one per run) -- does not save to disk.
"""
import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import Environment, clear_disc
from src.risk_planner import RiskSensitiveAORRT
from src.ao_rrt import AORRT
from map import Map2D, GridMap
from map.generator import disk
from map.risk_sdf import cvar_kappa

# ── settable knobs ──────────────────────────────────────────────────────────────
SEED = 33792
ITERS = 20000
ALPHA = 0.05                  # CVaR tail fraction (risk-aware run only)
OBSTACLE_RADIUS = 0.5         # m, single circular obstacle at the map center
OBSTACLE_SIGMA_MIN = 0.3     # m, boundary-uncertainty std -- single obstacle draws
OBSTACLE_SIGMA_MAX = 0.3     # ONE value, uniform in [MIN, MAX] (same seed both runs)
START_HEADING = math.pi / 2   # initial theta0 (rad); pi/2 = facing +y, instead of at the goal

# Trained-envelope command caps (same reasoning as main_test.py): commands fed to
# the residual NN must stay inside the range it was trained on.
V_CMD_CAP = 0.38
W_CMD_CAP = 0.9


def build_single_obstacle_grid(cfg, center, radius_m, start, goal, clear_radius):
    """Hand-placed single circular obstacle (radius_m, world `center`) -- no random
    rock field. Obstacle-free discs of `clear_radius` are carved around start/goal
    (sized by the caller to stay feasible under the risk-inflated SDF too)."""
    e = cfg.env
    grid = GridMap(Map2D(e.width, e.height, e.resolution))
    j0, i0 = grid.map.meters_to_indices(center[0], center[1])
    r_px = int(round(radius_m / grid.map.resolution_m))
    rr, cc = disk((j0, i0), r_px, shape=grid.features.shape)
    grid.features[rr, cc] = 1
    grid.occupancy[rr, cc] = 1
    grid.obstacle_id[rr, cc] = 1
    grid.compute_slope_map()
    for p in (start, goal):
        clear_disc(grid, p, clear_radius)
    return grid


def _plot_sdf(env, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ext = [0, env.width, 0, env.height]
    vmax = float(np.abs(env.sdf).max())
    im = ax.imshow(env.sdf, origin="lower", extent=ext, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="signed distance (m)")
    fig.tight_layout()


def _plot(grid, env, pl, path, start, goal, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ext = [0, env.width, 0, env.height]
    obst = np.ma.masked_where(grid.features == 0, grid.features)
    ax.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
              alpha=0.85, interpolation="bilinear")
    for e in pl.tree_edges():
        ax.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)
    ax.plot(path[:, 0], path[:, 1], "-", color="tab:cyan", lw=2.0, label="AO-RRT", zorder=3)
    ax.plot(*start, "o", color="lime", ms=12, mec="k")
    ax.plot(*goal, "*", color="magenta", ms=20, mec="k")
    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{title}\nnodes={len(pl.nodes)}, cost={pl.best_cost:.2f}")
    fig.tight_layout()


def main():
    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = ITERS
    cfg.aorrt.v_max = V_CMD_CAP
    cfg.aorrt.w_max = W_CMD_CAP
    cfg.risk.alpha = ALPHA

    W, H = cfg.env.width, cfg.env.height
    START = (0.85 * W, 0.13 * H)   # bottom-right corner
    GOAL = (0.15 * W, 0.87 * H)    # top-left corner
    CENTER = (W / 2.0, H / 2.0)

    kappa = cvar_kappa(ALPHA)
    # obstacle-free disc around start/goal sized for the WORST-CASE risk-inflated
    # keep-out too (same reasoning as environment.make_map_env), so the SAME cleared
    # grid stays feasible for both the risk-inflated and the plain env.
    clear_radius = cfg.env.disc_radius + 0.5 + kappa * OBSTACLE_SIGMA_MAX
    grid = build_single_obstacle_grid(cfg, CENTER, OBSTACLE_RADIUS, START, GOAL, clear_radius)

    env_risk = Environment(grid, disc_radius=cfg.env.disc_radius, clearance=cfg.env.clearance,
                           with_risk=False,
                           obstacle_sigma_min=OBSTACLE_SIGMA_MIN, obstacle_sigma_max=OBSTACLE_SIGMA_MAX,
                           sigma_seed=SEED, use_risk_sdf=True, risk_sdf_alpha=ALPHA,
                           risk_sdf_d_max=cfg.risk_sdf.d_max)
    env_plain = Environment(grid, disc_radius=cfg.env.disc_radius, clearance=cfg.env.clearance,
                            with_risk=False,
                            obstacle_sigma_min=OBSTACLE_SIGMA_MIN, obstacle_sigma_max=OBSTACLE_SIGMA_MAX,
                            sigma_seed=SEED, use_risk_sdf=False)

    print(f"single-obstacle course: {W}x{H} m, obstacle r={OBSTACLE_RADIUS} m @ center, "
          f"sigma=[{OBSTACLE_SIGMA_MIN},{OBSTACLE_SIGMA_MAX}] seed={SEED} | AO-RRT {ITERS} iters, "
          f"v_cmd cap=+-{V_CMD_CAP}, omega_cmd cap=+-{W_CMD_CAP}, "
          f"risk_sdf kappa={env_risk.risk_sdf_kappa:.3f} (alpha={ALPHA})")

    _plot_sdf(env_risk, f"Risk-inflated SDF (alpha={ALPHA}, kappa={env_risk.risk_sdf_kappa:.3f})")
    _plot_sdf(env_plain, "Plain SDF")

    # ---- run 1: risk-aware (CVaR edge cost + risk-inflated SDF) ---------------
    cfg_risk = copy.deepcopy(cfg)
    cfg_risk.risk.use_edge_cvar = True
    np.random.seed(SEED)
    pl1 = RiskSensitiveAORRT(env_risk, cfg_risk, START, GOAL, start_heading=START_HEADING)
    path1 = pl1.plan(verbose=True)
    if not pl1.goal_reached():
        print("WARNING (risk-aware run): goal not reached -- using best-effort path.")
    print(f"risk-aware AO-RRT: goal={pl1.goal_reached()} cost={pl1.best_cost:.3f} "
          f"nodes={len(pl1.nodes)} time={pl1.timers['total']:.2f}s")
    _plot(grid, env_risk, pl1, path1, START, GOAL,
          title=f"Risk-aware AO-RRT (CVaR, alpha={ALPHA}, risk-inflated SDF)")

    # ---- run 2: plain AO-RRT (no CVaR cost, plain SDF) -------------------------
    cfg_plain = copy.deepcopy(cfg)
    np.random.seed(SEED)
    pl2 = AORRT(env_plain, cfg_plain, START, GOAL, start_heading=START_HEADING)
    path2 = pl2.plan(verbose=True)
    if not pl2.goal_reached():
        print("WARNING (plain run): goal not reached -- using best-effort path.")
    print(f"plain AO-RRT: goal={pl2.goal_reached()} cost={pl2.best_cost:.3f} "
          f"nodes={len(pl2.nodes)} time={pl2.timers['total']:.2f}s")
    _plot(grid, env_plain, pl2, path2, START, GOAL,
          title="Plain AO-RRT (no CVaR, plain SDF)")

    plt.show()


if __name__ == "__main__":
    main()
