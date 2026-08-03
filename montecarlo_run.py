"""
Monte Carlo sweep over obstacle density for the AO-RRT planner.

For each obstacle density we run three bottom -> top traverses across many random
maps (seeds), planning with AO-RRT, and log one row per trial to a CSV. Useful for
measuring how success rate, path cost, and planning time degrade as the arena gets
more cluttered.

    python montecarlo_run.py [--densities 5 10 15 20 25] [--seeds 20] [--iters 6000]
                              [--out results/montecarlo.csv] [--quiet]

Defaults: 5 densities x 3 traverses x 20 seeds = 300 trials, AO-RRT only.
The three traverses (bottom -> top):
    L  bottom-left  -> top-right   (diagonal)
    C  bottom-centre -> top-centre  (straight up the middle)
    R  bottom-right -> top-left    (diagonal)
make_map_env carves obstacle-free discs around the start/goal so they never land
on a rock.

Rows are flushed as they are produced, so a partial CSV survives an interrupt.

Each trial also saves a small .npz (obstacle grid + planned path) to <out>_trials/
by default, so any trial's map+path plot can be recreated later with
replot_trial.py -- pass --no-save-trials to skip this.
"""
import os
import sys
import csv
import math
import time
import argparse

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.ao_rrt import AORRT

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# bottom -> top traverses: (name, start_x_frac, goal_x_frac) as fractions of width.
# L and R cross diagonally (opposite corners); C goes straight up the middle.
COLUMNS = (
    ("L", 0.15, 0.85),   # bottom-left  -> top-right
    ("C", 0.50, 0.50),   # bottom-centre -> top-centre
    ("R", 0.85, 0.15),   # bottom-right -> top-left
)
BOTTOM_FRAC, TOP_FRAC = 0.13, 0.87

# CSV schema (fixed column order).
FIELDS = [
    "trial_id", "density", "column", "start_x_frac", "goal_x_frac", "seed",
    "start_x", "start_y", "goal_x", "goal_y",
    "reached", "cost", "nodes", "plan_time_s", "t_first_s",
    "nn_time_s", "prop_time_s", "collision_time_s",
]


def _save_trial_artifact(trials_dir, trial_id, cfg, grid, path, start, goal,
                          density, col_name, seed, cost, reached):
    """Persist just enough to recreate the map+path plot later, independent of
    whatever the planner/dynamics code looks like at replot time: the actual
    obstacle occupancy grid (not just the seed -- rock placement is regenerable
    from the seed today, but the planned path depends on the dynamics model in
    effect at run time, which can drift) and the accepted path."""
    np.savez_compressed(
        os.path.join(trials_dir, f"{trial_id}.npz"),
        grid_features=grid.features,
        path=path,
        start=np.array(start), goal=np.array(goal),
        width=cfg.env.width, height=cfg.env.height, resolution=cfg.env.resolution,
        density=density, column=col_name, seed=seed,
        cost=(cost if math.isfinite(cost) else np.nan), reached=reached,
    )


def run_trial(cfg, density, col_name, start_x_frac, goal_x_frac, seed, trials_dir=None):
    """One planning trial; returns a dict matching FIELDS."""
    W, H = cfg.env.width, cfg.env.height
    start = (start_x_frac * W, BOTTOM_FRAC * H)
    goal = (goal_x_frac * W, TOP_FRAC * H)
    trial_id = f"{density}_{col_name}_{seed}"

    grid, env = make_map_env(cfg, seed, num_rocks=density, start=start, goal=goal)

    pl = AORRT(env, cfg, start, goal)
    t0 = time.perf_counter()
    path = pl.plan(verbose=False)
    plan_time = time.perf_counter() - t0
    reached = pl.goal_reached()

    if trials_dir is not None:
        _save_trial_artifact(trials_dir, trial_id, cfg, grid, path, start, goal,
                              density, col_name, seed, pl.best_cost, reached)

    return dict(
        trial_id=trial_id,
        density=density, column=col_name, start_x_frac=start_x_frac, goal_x_frac=goal_x_frac, seed=seed,
        start_x=round(start[0], 4), start_y=round(start[1], 4),
        goal_x=round(goal[0], 4), goal_y=round(goal[1], 4),
        reached=reached,
        cost=(round(pl.best_cost, 6) if math.isfinite(pl.best_cost) else ""),
        nodes=len(pl.nodes),
        plan_time_s=round(plan_time, 4),
        t_first_s=(round(pl.t_first, 4) if pl.t_first is not None else ""),
        nn_time_s=round(pl.timers["nn"], 4),
        prop_time_s=round(pl.timers["prop"], 4),
        collision_time_s=round(pl.timers["collision"], 4),
    )


def summarize(rows):
    """Print a per-density aggregate to stdout (the CSV holds the raw trials)."""
    densities = sorted({r["density"] for r in rows})
    print("\n=== summary (per density) ===")
    print(f"{'rocks':>5} {'trials':>6} {'success':>8} {'mean_cost':>10} {'mean_plan_s':>11}")
    for d in densities:
        rs = [r for r in rows if r["density"] == d]
        n = len(rs)
        succ = [r for r in rs if r["reached"]]
        costs = [r["cost"] for r in succ if r["cost"] != ""]
        ptimes = [r["plan_time_s"] for r in rs]
        print(f"{d:>5} {n:>6} {len(succ) / n:>7.0%} "
              f"{(np.mean(costs) if costs else float('nan')):>10.3f} "
              f"{np.mean(ptimes):>11.3f}")


def main():
    ap = argparse.ArgumentParser(description="Obstacle-density Monte Carlo sweep.")
    ap.add_argument("--densities", type=int, nargs="+", default=[5,6,7,8,9],
                    help="rock counts to sweep")
    ap.add_argument("--seeds", type=int, default=30, help="random maps per cell")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (seeds are seed0..seed0+seeds-1)")
    ap.add_argument("--iters", type=int, default=10000, help="AO-RRT iterations")
    ap.add_argument("--cost-mode", default="control", choices=("time", "control", "both"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "montecarlo.csv"))
    ap.add_argument("--trials-dir", default=None,
                    help="dir to save per-trial map+path .npz artifacts "
                         "(default: <out>_trials/ next to the CSV)")
    ap.add_argument("--no-save-trials", action="store_true",
                    help="skip saving per-trial map+path artifacts")
    ap.add_argument("--quiet", action="store_true", help="suppress per-trial progress")
    args = ap.parse_args()

    cfg = PlannerConfig()
    cfg.aorrt.cost_mode = args.cost_mode
    cfg.aorrt.max_iterations = args.iters

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    total = len(args.densities) * len(COLUMNS) * len(seeds)

    trials_dir = None
    if not args.no_save_trials:
        trials_dir = args.trials_dir or (os.path.splitext(args.out)[0] + "_trials")
        os.makedirs(trials_dir, exist_ok=True)

    print(f"Monte Carlo: {len(args.densities)} densities x {len(COLUMNS)} columns x "
          f"{len(seeds)} seeds = {total} trials | cost='{args.cost_mode}', {args.iters} iters"
          + (f" | trial artifacts -> {trials_dir}" if trials_dir else " | trial artifacts off"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        i = 0
        for density in args.densities:
            for col_name, start_x_frac, goal_x_frac in COLUMNS:
                for seed in seeds:
                    i += 1
                    rec = run_trial(cfg, density, col_name, start_x_frac, goal_x_frac, seed, trials_dir)
                    writer.writerow(rec); fh.flush()
                    rows.append(rec)
                    if not args.quiet:
                        flag = "G" if rec["reached"] else "."
                        cost = rec["cost"] if rec["cost"] != "" else "  -  "
                        print(f"[{i:4d}/{total}] rocks={density:>3} col={col_name} "
                              f"seed={seed:>3} {flag} cost={cost} "
                              f"t={rec['plan_time_s']:.2f}s")

    print(f"\nsaved {len(rows)} trials -> {args.out}  ({time.perf_counter() - t_start:.1f}s)")
    summarize(rows)


if __name__ == "__main__":
    main()
