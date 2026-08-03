"""
Sanity/demo test for the closed-loop tracking ensemble (dynamics_vel.py) run
alongside a real AO-RRT tree. The tree itself is planned with RiskSensitiveAORRT;
pass "cvar" to also add the tracked-ensemble obstacle-CVaR term to the AO-RRT edge
cost (RiskParams.use_edge_cvar), or "nocvar" (default) to plan without it, for direct
comparison. This script:

    1. plans a short AO-RRT tree (edge CVaR cost on/off per the CLI flag),
    2. picks N_SEGMENTS parent/child edges spread evenly along the best-cost path,
    3. runs the feedback-linearization tracking ensemble on each edge (each member
       absorbing a different constant disturbance on (dot(v_b), dot(omega))),
    4. plots the full tree + path (with all inspected edges highlighted), and one
       zoomed-in panel per edge overlaying the disturbed/tracked trajectories against
       that edge's nominal reference,
    5. times one tracked_ensemble() call per edge, cold (incl. JIT compile) vs warm.

    python main_test.py [seed] [iters] [cvar|nocvar]

Saves results/plots/tracking_test.png.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.risk_planner import RiskSensitiveAORRT

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Trained-envelope command caps: commands fed to the closed-loop tracker must stay
# inside the range the residual net was trained on, else the NN extrapolates
# unreliably (see project notes). Overrides cfg.aorrt.v_max/w_max for THIS script
# only -- not a change to config.py's defaults.
V_CMD_CAP = 0.38
W_CMD_CAP = 0.9

N_SEGMENTS = 5
SEGMENT_COLORS = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]

def _disturbance_grid(cfg):
    r = cfg.risk
    ax = np.linspace(-r.ax_dist_max, r.ax_dist_max, r.dist_grid_n)
    yw = np.linspace(-r.yaw_dist_max, r.yaw_dist_max, r.dist_grid_n)
    AX, YW = np.meshgrid(ax, yw)
    return np.stack([AX.ravel(), YW.ravel()], axis=1)


def _pick_edges(pl, n_edges=N_SEGMENTS):
    """n_edges parent/child node pairs spread evenly along the best (or best-effort)
    path -- e.g. n_edges=5 gives roughly the 1/6, 2/6, 3/6, 4/6, 5/6 points."""
    chain = pl._chain()
    if len(chain) < 2:
        raise RuntimeError("best path has no edges to inspect -- try more iterations")
    n_edges = min(n_edges, len(chain) - 1)
    idxs = sorted(set(np.linspace(1, len(chain) - 1, n_edges, dtype=int).tolist()))
    return [(chain[i - 1], chain[i]) for i in idxs]


def main():
    args = sys.argv[1:]
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 6545
    iters = nums[1] if len(nums) > 1 else 9000
    cfg = PlannerConfig()
    if "cvar" in args:
        cfg.risk.use_edge_cvar = True
    elif "nocvar" in args:
        cfg.risk.use_edge_cvar = False
    cfg.aorrt.max_iterations = iters
    cfg.aorrt.v_max = V_CMD_CAP
    cfg.aorrt.w_max = W_CMD_CAP
    use_edge_cvar = cfg.risk.use_edge_cvar
    W, H = cfg.env.width, cfg.env.height
    START = (0.75 * W, 0.13 * H); GOAL = (0.15 * W, 0.87 * H)
    # use_risk_sdf=True: collision checking (path_free feasibility gate, the tracked-
    # ensemble CVaR kernel's clearance lookup, everything reading env.sdf) runs against
    # map.risk_sdf's per-obstacle risk-inflated field instead of the plain nominal SDF,
    # at cfg.risk.alpha -- the SAME tail fraction the CVaR edge cost uses.
    grid, env = make_map_env(cfg, seed, start=START, goal=GOAL, use_risk_sdf=True)
    print(f"map {W}x{H} m, {cfg.env.num_rocks} rocks, seed {seed} | AO-RRT {iters} iters, "
          f"v_cmd cap=+-{V_CMD_CAP}, omega_cmd cap=+-{W_CMD_CAP}, "
          f"risk_sdf alpha={cfg.risk.alpha} kappa={env.risk_sdf_kappa:.3f}, "
          f"edge_cvar={'ON' if use_edge_cvar else 'off'} "
          f"(alpha={cfg.risk.alpha}, weight={cfg.risk.edge_cvar_weight})")

    pl = RiskSensitiveAORRT(env, cfg, START, GOAL)
    path = pl.plan(verbose=True)
    if not pl.goal_reached():
        print("WARNING: goal not reached in this iteration budget -- inspecting "
              "edges from the best-effort path to the nearest node instead.")
    print(f"AO-RRT: goal={pl.goal_reached()} cost={pl.best_cost:.3f} nodes={len(pl.nodes)} "
          f"time={pl.timers['total']:.2f}s")

    # ---- pick N_SEGMENTS edges spread along the path, run the closed-loop tracking
    # ensemble on each (standalone -- not part of the planner's cost) ---------------
    edges = _pick_edges(pl, N_SEGMENTS)
    D = _disturbance_grid(cfg)
    r = cfg.risk
    dt = cfg.dyn.dt
    clip_lo = (r.ax_clip_lo, r.yaw_clip_lo)
    clip_hi = (r.ax_clip_hi, r.yaw_clip_hi)
    Kp = (r.kp_x, r.kp_y)
    u_max = (V_CMD_CAP, W_CMD_CAP)

    segments = []   # list of dicts: parent, child, u_ref, nsteps, Zref, Zens, t_cold, t_warm
    for k, (parent, child) in enumerate(edges):
        u_ref = np.clip(child.u, [-V_CMD_CAP, -W_CMD_CAP], [V_CMD_CAP, W_CMD_CAP])
        nsteps = child.nsteps

        t0 = time.perf_counter()
        Zens = pl.model.tracked_ensemble(parent.x, u_ref, nsteps, dt, D, r.track_gain,
                                          clip_lo, clip_hi, Kp, r.k_psi, r.v_eps_speed, u_max)
        t_cold = time.perf_counter() - t0
        n_warm = 5
        t0 = time.perf_counter()
        for _ in range(n_warm):
            pl.model.tracked_ensemble(parent.x, u_ref, nsteps, dt, D, r.track_gain,
                                       clip_lo, clip_hi, Kp, r.k_psi, r.v_eps_speed, u_max)
        t_warm = (time.perf_counter() - t0) / n_warm

        print(f"segment {k+1}/{len(edges)}: parent t={parent.t:.2f}s -> child t={child.t:.2f}s, "
              f"nsteps={nsteps}, u_ref={u_ref}, "
              f"tracked_ensemble(M={len(D)}): cold={t_cold*1e3:.2f} ms, warm={t_warm*1e3:.3f} ms/call")

        segments.append(dict(parent=parent, child=child, u_ref=u_ref, nsteps=nsteps,
                              Zref=child.edgeX, Zens=Zens, t_cold=t_cold, t_warm=t_warm))

    # ---- plot: full tree + path (all inspected edges highlighted), one zoomed-in
    # tracking-ensemble panel per segment ------------------------------------------
    ncols = len(segments)
    fig = plt.figure(figsize=(4.2 * ncols, 11))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.4, 1])
    axm = fig.add_subplot(gs[0, :])
    axz_list = [fig.add_subplot(gs[1, k]) for k in range(ncols)]

    ext = [0, env.width, 0, env.height]
    obst = np.ma.masked_where(grid.features == 0, grid.features)
    axm.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
               alpha=0.85, interpolation="bilinear")
    for e in pl.tree_edges():
        axm.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)
    axm.plot(path[:, 0], path[:, 1], "-", color="tab:cyan", lw=2.0, label="AO-RRT", zorder=3)
    for k, seg in enumerate(segments):
        col = SEGMENT_COLORS[k % len(SEGMENT_COLORS)]
        axm.plot(seg["Zref"][:, 0], seg["Zref"][:, 1], "-", color=col, lw=3.0, zorder=5,
                  label=f"segment {k+1}")
    axm.plot(*START, "o", color="lime", ms=12, mec="k"); axm.plot(*GOAL, "*", color="magenta", ms=20, mec="k")
    axm.set_xlim(0, env.width); axm.set_ylim(0, env.height); axm.set_aspect("equal")
    axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)"); axm.legend(loc="upper left", fontsize=9, ncol=2)
    axm.set_title(f"AO-RRT tree (nodes={len(pl.nodes)}, cost={pl.best_cost:.1f})")

    for k, (axz, seg) in enumerate(zip(axz_list, segments)):
        col = SEGMENT_COLORS[k % len(SEGMENT_COLORS)]
        Zref, Zens = seg["Zref"], seg["Zens"]
        cmap = plt.get_cmap("autumn")
        for m in range(len(D)):
            mcol = cmap(m / max(1, len(D) - 1))
            axz.plot(Zens[m, :, 0], Zens[m, :, 1], "-", color=mcol, lw=0.8, alpha=0.7, zorder=2,
                      label="tracked (disturbed)" if m == 0 else None)
        axz.plot(Zref[:, 0], Zref[:, 1], "-", color="black", lw=0.9, zorder=4, label="reference")
        axz.plot(*Zref[0, :2], "o", color="lime", ms=8, mec="k", zorder=5, label="parent")
        axz.plot(*Zref[-1, :2], "*", color="magenta", ms=13, mec="k", zorder=5, label="child")

        pts = np.vstack([Zref[:, :2]] + [Zens[m, :, :2] for m in range(len(D))])
        span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min(), 0.2)
        pad = 0.15 * span
        axz.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        axz.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        axz.set_aspect("equal")
        axz.set_xlabel("x (m)")
        if k == 0:
            axz.set_ylabel("y (m)")
            axz.legend(loc="best", fontsize=7)
        for spine in axz.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(2.5)
        axz.set_title(f"segment {k+1}: t={seg['parent'].t:.1f}-{seg['child'].t:.1f}s\n"
                       f"u_ref=[{seg['u_ref'][0]:.2f},{seg['u_ref'][1]:.2f}], "
                       f"nsteps={seg['nsteps']}", fontsize=9, color=col)

    fig.suptitle(f"Closed-loop tracking ensemble (M={len(D)}) on {ncols} path segments, "
                 f"K={r.track_gain}, Kp=({r.kp_x},{r.kp_y}), k_psi={r.k_psi}", fontsize=12)

    out = os.path.join(RESULTS_DIR, "tracking_test.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
