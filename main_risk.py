"""
Risk-sensitive planning demo: risk-neutral vs risk-averse (CVaR) on a terrain-risk
map, for the Leo rover. Two side-by-side panels (one per CVaR level), each showing
the AO-RRT tree, the AO-RRT path, and the terrain-risk heatmap.

    python main_risk.py [--seed S] [--iters N] [--alphas 1.0 0.05]
                        [--slip 0.25] [--rocks 8] [--hills 4] [--no-tree]

    alpha = 1.0   risk-neutral (expectation)
    alpha = 0.05  risk-averse  (CVaR of the worst 5%)

Why two panels (not one overlay): each planner grows its own tree, so plotting them
separately keeps the trees legible.

IMPORTANT — for the comparison to be meaningful the terrain risk must have spatial
structure that is NOT merely obstacle proximity (otherwise the SDF planner already
avoids it and risk-aversion has nothing new to do). This demo therefore builds the
map with hills + slope/roughness risk weights, and uses a larger slip so the CVaR
tail separates from the mean. See the README "Risk-sensitive layer" notes.

Saves results/plots/risk_demo.png. Requires jax.
"""
import os
import sys
import argparse
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.risk_planner import RiskSensitiveAORRT, cvar

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _plan(cfg, env, start, goal, alpha):
    rp = replace(cfg.risk, alpha=alpha)
    pl = RiskSensitiveAORRT(env, cfg, start, goal, risk=rp)
    path = pl.plan(verbose=False)
    reached = pl.goal_reached()
    if reached:
        rvals = env.risk_vals(path[:, 0:2])
        clear = env.sdf_and_grad(path[:, 0:2])[0] - env.disc_radius
        min_clear = float(np.min(clear))
    else:
        rvals = np.array([np.nan])
        min_clear = float("nan")
    return dict(pl=pl, path=path, reached=reached,
                cost=pl.best_cost, mean_risk=float(np.mean(rvals)),
                tail_risk=cvar(rvals, alpha), min_clear=min_clear,
                margin=pl.clearance_margin)


def _panel(ax, grid, env, res, col, draw_tree):
    """Draw one planner result (res from _plan) over the terrain-risk map."""
    W, H = env.width, env.height
    ext = [0, W, 0, H]
    ax.imshow(env.risk.risk, origin="lower", extent=ext, cmap="YlOrRd",
              vmin=0, vmax=1, alpha=0.9, zorder=0)
    obst = np.ma.masked_where(grid.features == 0, grid.features)
    ax.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
              alpha=0.9, zorder=1)
    if draw_tree:
        for e in res["pl"].tree_edges():
            ax.plot(e[:, 0], e[:, 1], "-", color="0.6", lw=0.3, alpha=0.35, zorder=2)
    if res["reached"]:
        ax.plot(res["path"][:, 0], res["path"][:, 1], "-", color=col, lw=2.0,
                label="AO-RRT", zorder=3)


def main():
    ap = argparse.ArgumentParser(description="Risk-sensitive CVaR planning demo.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.15, 0.05],
                    help="one or more CVaR alpha levels to compare (e.g. 1.0 0.15 0.05)")
    ap.add_argument("--slip", type=float, default=0.3,
                    help="wheel-slip noise std (larger -> CVaR separates from mean)")
    ap.add_argument("--rocks", type=int, default=8)
    ap.add_argument("--hills", type=int, default=5,
                    help="terrain undulations -> slope/roughness risk (drives slip)")
    ap.add_argument("--obs-sigma-min", type=float, default=0.01,
                    help="stochastic-obstacle boundary-uncertainty std (m): low end of "
                         "the per-obstacle sampling range")
    ap.add_argument("--obs-sigma-max", type=float, default=0.01,
                    help="stochastic-obstacle boundary-uncertainty std (m): high end of "
                         "the per-obstacle sampling range -- each obstacle instance "
                         "draws its own std uniformly in [obs-sigma-min, obs-sigma-max]")
    ap.add_argument("--no-tree", action="store_true")
    args = ap.parse_args()

    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = args.iters
    cfg.risk.slip_sigma = args.slip
    W, H = cfg.env.width, cfg.env.height
    START = (0.85 * W, 0.13 * H); GOAL = (0.15 * W, 0.87 * H)
    grid, env = make_map_env(cfg, args.seed, num_rocks=args.rocks, start=START,
                             goal=GOAL, with_risk=True, undulation=args.hills,
                             risk_weights=(0.3, 0.3, 1.0),
                             obstacle_sigma_min=args.obs_sigma_min,
                             obstacle_sigma_max=args.obs_sigma_max)

    print(f"risk demo: seed {args.seed}, {args.iters} iters, slip_sigma={args.slip}, "
          f"slip_gain={cfg.risk.slip_gain}, "
          f"stochastic-obstacle sigma range=[{args.obs_sigma_min}, {args.obs_sigma_max}], "
          f"CVaR n_samples={cfg.risk.n_samples}, alphas={args.alphas}")

    palette = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    labels = [(f"α={a:g}", a, palette[i % len(palette)])
              for i, a in enumerate(args.alphas)]
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 7.5))
    if n == 1:
        axes = [axes]
    for ax, (label, alpha, col) in zip(axes, labels):
        res = _plan(cfg, env, START, GOAL, alpha)
        _panel(ax, grid, env, res, col, not args.no_tree)
        ax.plot(*START, "o", color="lime", ms=12, mec="k", zorder=6)
        ax.plot(*GOAL, "*", color="magenta", ms=20, mec="k", zorder=6)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        if res["reached"]:
            ax.set_title(f"{label}   keep-out margin=+{res['margin']:.2f} m\n"
                         f"mean-risk={res['mean_risk']:.3f}  CVaR-tail={res['tail_risk']:.3f}  "
                         f"min-clearance={res['min_clear']:.2f} m")
            ax.legend(loc="upper left", fontsize=9)
            print(f"  {label}: reached, margin=+{res['margin']:.3f}m, "
                  f"mean-risk={res['mean_risk']:.3f}, tail-risk={res['tail_risk']:.3f}, "
                  f"min-clear={res['min_clear']:.3f}m, nodes={len(res['pl'].nodes)}")
        else:
            ax.set_title(f"{label}\nNO GOAL")
            print(f"  {label}: NO GOAL reached")

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="YlOrRd"), ax=axes,
                        fraction=0.025, pad=0.02)
    cbar.set_label("terrain failure risk")
    fig.suptitle("Risk-sensitive planning: CVaR risk-neutral vs risk-averse "
                 "(tree faint, solid = AO-RRT path)", fontsize=13)
    out = os.path.join(RESULTS_DIR, "risk_demo.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
