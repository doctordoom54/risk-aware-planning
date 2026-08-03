"""
Standard AO-RRT for the skid-steer rover (velocity-pose dynamics + learned residual).

    python main.py [cost_mode] [seed] [iters]
        cost_mode = time | control | both     (default: control)

Builds a map, plans with AO-RRT under the chosen cost, prints metrics, and saves results/plots/plan.png:
    left   trajectory over the map (AO-RRT)
    right  the [v_cmd, omega_cmd] control inputs vs time
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.ao_rrt import AORRT


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CONTROLS = ("v_cmd", "omega_cmd"); CCOL = ["tab:blue", "tab:orange"]


def main():
    args = [a for a in sys.argv[1:]]
    cost_mode = next((a for a in args if a in ("time", "control", "both")), "control")
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 60
    iters = nums[1] if len(nums) > 1 else 8000

    cfg = PlannerConfig()
    cfg.aorrt.cost_mode = cost_mode
    cfg.aorrt.max_iterations = iters
    # start/goal scale with the arena (near opposite corners, clear of the border)
    W, H = cfg.env.width, cfg.env.height
    START = (0.75 * W, 0.13 * H); GOAL = (0.15 * W, 0.87 * H)
    grid, env = make_map_env(cfg, seed, start=START, goal=GOAL)
    print(f"map {cfg.env.width}x{cfg.env.height} m, {cfg.env.num_rocks} rocks, seed {seed} | "
          f"AO-RRT cost_mode='{cost_mode}', {iters} iters")

    pl = AORRT(env, cfg, START, GOAL)
    path = pl.plan(verbose=True)
    print(f"AO-RRT: goal={pl.goal_reached()} cost={pl.best_cost:.3f} "
          f"nodes={len(pl.nodes)} time={pl.timers['total']:.2f}s "
          f"t_first={pl.t_first if pl.t_first else float('nan'):.2f}s")
    U = pl.extract_controls(); dt = cfg.dyn.dt

    # ---- plot -------------------------------------------------------------
    ext = [0, env.width, 0, env.height]
    obst = np.ma.masked_where(grid.features == 0, grid.features)
    fig, (axm, axu) = plt.subplots(1, 2, figsize=(15, 7))
    axm.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, alpha=0.85,interpolation="bilinear")
    for e in pl.tree_edges():
        axm.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)
    axm.plot(path[:, 0], path[:, 1], "-", color="tab:cyan", lw=2.0, label="AO-RRT", zorder=3)
    s = max(1, len(path) // 25)
    axm.quiver(path[::s, 0], path[::s, 1], np.cos(path[::s, 2]), np.sin(path[::s, 2]),
               color="tab:blue", scale=25, width=0.004, zorder=4)
    axm.plot(*START, "o", color="lime", ms=12, mec="k"); axm.plot(*GOAL, "*", color="magenta", ms=20, mec="k")
    axm.set_xlim(0, env.width); axm.set_ylim(0, env.height); axm.set_aspect("equal")
    axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)"); axm.legend(loc="upper left", fontsize=9)
    axm.set_title(f"AO-RRT ('{cost_mode}' cost = {pl.best_cost:.1f})")

    tu = np.arange(len(U)) * dt
    v_max, w_max = cfg.aorrt.v_max, cfg.aorrt.w_max
    axu2 = axu.twinx()
    axu.step(tu, U[:, 0], where="post", lw=1.0, color=CCOL[0], alpha=0.75, label="AO $v_{cmd}$")
    axu2.step(tu, U[:, 1], where="post", lw=1.0, color=CCOL[1], alpha=0.75, label="AO $\\omega_{cmd}$")

    axu.axhline(v_max, color=CCOL[0], ls=":", lw=0.8); axu.axhline(-v_max, color=CCOL[0], ls=":", lw=0.8)
    axu2.axhline(w_max, color=CCOL[1], ls=":", lw=0.8); axu2.axhline(-w_max, color=CCOL[1], ls=":", lw=0.8)

    axu.set_xlabel("time (s)")
    axu.set_ylabel("v_cmd (m/s)", color=CCOL[0]); axu.tick_params(axis="y", labelcolor=CCOL[0])
    axu2.set_ylabel("omega_cmd (rad/s)", color=CCOL[1]); axu2.tick_params(axis="y", labelcolor=CCOL[1])
    axu.set_title("Control inputs (v_cmd, omega_cmd)")
    lines1, labels1 = axu.get_legend_handles_labels(); lines2, labels2 = axu2.get_legend_handles_labels()
    axu.legend(lines1 + lines2, labels1 + labels2, fontsize=7, ncol=2)
    axu.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "plots", "plan.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
