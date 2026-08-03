"""
Monte Carlo sweep for RiskAwareSCP (scp_vel.py), same trial grid as
montecarlo_test.py (num_rocks densities x L/C/R start/goal columns x seeds) but
running the test_scp_vel_plot.py workflow per trial instead of
RiskSensitiveAORRT: plain AORRT seed -> RiskAwareSCP.solve refine, then compare
duration-weighted control cost and total CVaR risk (AO-RRT seed vs SCP-refined),
exactly the same formulas test_scp_vel_plot.py prints for one seed/map.

Each trial's .npz artifact holds enough to recreate test_scp_vel_plot.py's
plot_map() panels later WITHOUT rerunning planning/SCP -- grid + risk raster +
both trajectories -- but deliberately NOT the AO-RRT tree edges (same call
montecarlo_test.py made: a tree's edges dominate storage and aren't needed to
redraw just the two paths). No replot script yet -- artifacts are saved so one
can be written later without re-running the sweep.

    python montecarlo_scp.py [--densities 5 6 7 8] [--seeds 20] [--master-seed N]
                              [--iters 10000] [--out results/montecarlo_scp.csv]
                              [--quiet]

Defaults: 4 densities x 3 columns x 20 seeds = 240 trials. Map seeds are drawn
randomly (not 0,1,2,...) via --master-seed (omit for a fresh draw each run,
pass a value to reproduce the same seed list later -- the draw is printed at
startup and every seed is also logged per-row in the CSV).
"""
import os
import sys
import csv
import math
import time
import argparse
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.ao_rrt import AORRT
from src.scp_vel import RiskAwareSCP, EdgeRiskEvaluator, edges_from_chain
from src.risk_planner import cvar

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# bottom -> top traverses: (name, start_x_frac, goal_x_frac) as fractions of width.
# SAME as montecarlo_test.py's COLUMNS/BOTTOM_FRAC/TOP_FRAC.
COLUMNS = (
    ("L", 0.15, 0.85),
    ("C", 0.50, 0.50),
    ("R", 0.85, 0.15),
)
BOTTOM_FRAC, TOP_FRAC = 0.13, 0.87

# same map-generation recipe test_scp_vel_plot.py uses -- terrain risk needs real
# spatial structure (undulation), not just the obstacle halo (risk_weights' 3rd
# term), for RiskAwareSCP's EdgeRiskEvaluator to have a meaningful map to refine
# against.
MAP_KW = dict(with_risk=True, num_craters=0, undulation=6,
              risk_weights=(0.3, 0.3, 1.0),
              obstacle_sigma_min=0.1, obstacle_sigma_max=0.5)
START_HEADING = math.pi / 2   # always face +y, same as test_scp_vel_plot.py

FIELDS = [
    "trial_id", "density", "column", "start_x_frac", "goal_x_frac", "seed",
    "start_x", "start_y", "goal_x", "goal_y",
    "aorrt_reached", "K_edges",
    "control_cost_aorrt", "control_cost_scp", "control_cost_pct_decrease",
    "risk_cost_aorrt", "risk_cost_scp", "risk_cost_pct_decrease",
    "log_survive_aorrt", "log_survive_scp", "log_survive_delta",
    "scp_converged", "scp_iters", "scp_n_solves", "scp_stop_reason",
    "plan_time_s", "scp_time_s", "total_time_s",
]


# ---- same formulas as test_scp_vel_plot.py, kept identical for comparability ----
def control_cost(U, nsteps_k, dt, r_v, r_omega):
    per_edge = (r_v * U[:, 0] ** 2 + r_omega * U[:, 1] ** 2) * (nsteps_k * dt)
    return float(np.sum(per_edge))


def total_risk(R, alpha):
    return float(sum(cvar(R[k], alpha) for k in range(R.shape[0])))


def joint_fail_prob(env, S):
    p_fail = env.risk_vals(S[:, 0:2])
    return float(np.sum(np.log1p(-p_fail)))


def _save_trial_artifact(trials_dir, trial_id, cfg, grid, env, start, goal,
                          density, col_name, seed, S_bar0, S_bar):
    """grid + risk raster + both trajectories -- enough to redraw
    test_scp_vel_plot.py's plot_map() obstacle/risk panels for this trial later.
    Deliberately no tree edges (see module docstring)."""
    np.savez_compressed(
        os.path.join(trials_dir, f"{trial_id}.npz"),
        grid_features=grid.features,
        risk_field=env.risk.risk,
        S_bar0=S_bar0, S_bar=S_bar,
        start=np.array(start), goal=np.array(goal),
        width=cfg.env.width, height=cfg.env.height, resolution=cfg.env.resolution,
        density=density, column=col_name, seed=seed,
    )


def run_trial(cfg, density, col_name, start_x_frac, goal_x_frac, seed,
              trials_dir=None):
    """One AO-RRT-seed + RiskAwareSCP-refine trial; returns a dict matching
    FIELDS. If AO-RRT doesn't reach the goal, SCP is never run and every
    cost/decrease/scp_* field is left blank ("") -- same "only if a trajectory
    was found" convention as montecarlo_test.py's cost field."""
    t_trial0 = time.perf_counter()
    W, H = cfg.env.width, cfg.env.height
    start = (start_x_frac * W, BOTTOM_FRAC * H)
    goal = (goal_x_frac * W, TOP_FRAC * H)
    trial_id = f"{density}_{col_name}_{seed}"

    grid, env = make_map_env(cfg, seed, num_rocks=density, start=start, goal=goal,
                              **MAP_KW)

    pl = AORRT(env, cfg, start, goal, start_heading=START_HEADING)
    t0 = time.perf_counter()
    pl.plan(verbose=False)
    plan_time = time.perf_counter() - t0
    reached = pl.goal_reached()

    row = dict(
        trial_id=trial_id, density=density, column=col_name,
        start_x_frac=start_x_frac, goal_x_frac=goal_x_frac, seed=seed,
        start_x=round(start[0], 4), start_y=round(start[1], 4),
        goal_x=round(goal[0], 4), goal_y=round(goal[1], 4),
        aorrt_reached=reached, K_edges="",
        control_cost_aorrt="", control_cost_scp="", control_cost_pct_decrease="",
        risk_cost_aorrt="", risk_cost_scp="", risk_cost_pct_decrease="",
        log_survive_aorrt="", log_survive_scp="", log_survive_delta="",
        scp_converged="", scp_iters="", scp_n_solves="", scp_stop_reason="",
        plan_time_s=round(plan_time, 4), scp_time_s="",
        total_time_s="",
    )

    if not reached:
        row["total_time_s"] = round(time.perf_counter() - t_trial0, 4)
        return row

    chain = pl._chain()
    S_bar0, U_bar0, nsteps_k = edges_from_chain(chain)

    scp = RiskAwareSCP(env, cfg, pl.model)
    t0 = time.perf_counter()
    S_bar, U_bar, info = scp.solve(chain, goal, verbose=False)
    scp_time = time.perf_counter() - t0

    svp = cfg.scp_vel
    dt = cfg.dyn.dt
    c0 = control_cost(U_bar0, nsteps_k, dt, svp.r_v, svp.r_omega)
    c1 = control_cost(U_bar, nsteps_k, dt, svp.r_v, svp.r_omega)
    c_pct = 100.0 * (c1 - c0) / c0 if c0 != 0 else float("nan")

    risk_eval = EdgeRiskEvaluator(env, cfg, pl.model)
    R0 = risk_eval.edge_risk(S_bar0[:-1], U_bar0, nsteps_k)
    R1 = risk_eval.edge_risk(S_bar[:-1], U_bar, nsteps_k)
    alpha = cfg.risk.alpha
    tr0 = total_risk(R0, alpha)
    tr1 = total_risk(R1, alpha)
    tr_pct = 100.0 * (tr1 - tr0) / tr0 if tr0 != 0 else float("nan")

    ls0 = joint_fail_prob(env, S_bar0)
    ls1 = joint_fail_prob(env, S_bar)

    if trials_dir is not None:
        _save_trial_artifact(trials_dir, trial_id, cfg, grid, env, start, goal,
                              density, col_name, seed, S_bar0, S_bar)

    row.update(
        K_edges=len(nsteps_k),
        control_cost_aorrt=round(c0, 6), control_cost_scp=round(c1, 6),
        control_cost_pct_decrease=round(c_pct, 3),
        risk_cost_aorrt=round(tr0, 6), risk_cost_scp=round(tr1, 6),
        risk_cost_pct_decrease=round(tr_pct, 3),
        log_survive_aorrt=round(ls0, 4), log_survive_scp=round(ls1, 4),
        log_survive_delta=round(ls1 - ls0, 4),
        scp_converged=bool(info["converged"]), scp_iters=info["iters"],
        scp_n_solves=info["n_solves"], scp_stop_reason=info["stop_reason"] or "",
        scp_time_s=round(scp_time, 4),
        total_time_s=round(time.perf_counter() - t_trial0, 4),
    )
    return row


def _signed_pct(vals):
    """Mean of vals plus a direction marker -- "v" (cost/risk DROPPED vs the
    AO-RRT seed) or "^" (GREW) -- so a glance tells you which way the number
    cuts without having to remember the sign convention
    (control_cost_pct_decrease/risk_cost_pct_decrease are POSITIVE when the
    SCP-refined trajectory's cost/risk is HIGHER than the seed's, e.g. the
    common control-cost-up/risk-down tradeoff). ASCII only -- Windows consoles
    here default to cp1252, which can't encode arrow unicode."""
    if not vals:
        return "    -    "
    m = float(np.mean(vals))
    marker = "v" if m < 0 else ("^" if m > 0 else "=")
    return f"{marker}{abs(m):6.2f}%"


def summarize(rows):
    densities = sorted({r["density"] for r in rows})
    print("\n=== summary (per density) ===")
    print("(ctrl/risk columns: mean % change vs the AO-RRT seed -- "
          "v = dropped (improved), ^ = grew)")
    print(f"{'rocks':>5} {'trials':>6} {'success':>8} {'converged':>10} "
          f"{'ctrl avg':>10} {'risk avg':>10} {'mean_total_s':>12}")
    for d in densities:
        rs = [r for r in rows if r["density"] == d]
        n = len(rs)
        reached = [r for r in rs if r["aorrt_reached"]]
        scored = [r for r in reached if r["control_cost_pct_decrease"] != ""]
        conv = [r for r in scored if r["scp_converged"]]
        ctrl_pct = [r["control_cost_pct_decrease"] for r in scored]
        risk_pct = [r["risk_cost_pct_decrease"] for r in scored]
        totals = [r["total_time_s"] for r in rs if r["total_time_s"] != ""]
        print(f"{d:>5} {n:>6} {len(reached) / n:>7.0%} "
              f"{(len(conv) / len(scored) if scored else float('nan')):>9.0%} "
              f"{_signed_pct(ctrl_pct):>10} {_signed_pct(risk_pct):>10} "
              f"{(np.mean(totals) if totals else float('nan')):>12.2f}")


def main():
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.join(RESULTS_DIR, f"montecarlo_scp_{run_stamp}", "montecarlo_scp.csv")

    ap = argparse.ArgumentParser(description="AO-RRT + RiskAwareSCP Monte Carlo sweep.")
    ap.add_argument("--densities", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--seeds", type=int, default=22,
                     help="number of map seeds to draw (randomly, not sequential -- see --master-seed)")
    ap.add_argument("--master-seed", type=int, default=None,
                     help="seeds the RNG that draws the actual per-trial map seeds; omit for a "
                          "fresh, non-reproducible draw each run (default), pass a value to "
                          "reproduce the same seed list later")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--out", default=default_out,
                     help="CSV output path; defaults to a fresh dated results/montecarlo_scp_<timestamp>/ folder")
    ap.add_argument("--trials-dir", default=None,
                     help="dir to save per-trial artifacts (default: <out>_trials/ next to the CSV)")
    ap.add_argument("--no-save-trials", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = args.iters
    cfg.risk.dist_grid_n = 10   # same override test_scp_vel_plot.py uses

    # random map seeds, not a sequential 0..N-1 run -- master_seed only controls
    # reproducibility of WHICH random seeds get drawn, drawn without replacement
    # from a wide range so collisions across --seeds draws are effectively impossible.
    master_seed = args.master_seed if args.master_seed is not None else np.random.SeedSequence().entropy
    seeds = np.random.default_rng(master_seed).choice(1_000_000, size=args.seeds, replace=False).tolist()
    total = len(args.densities) * len(COLUMNS) * len(seeds)

    trials_dir = None
    if not args.no_save_trials:
        trials_dir = args.trials_dir or (os.path.splitext(args.out)[0] + "_trials")
        os.makedirs(trials_dir, exist_ok=True)

    print(f"Monte Carlo (AO-RRT + RiskAwareSCP): {len(args.densities)} densities x "
          f"{len(COLUMNS)} columns x {len(seeds)} seeds = {total} trials | "
          f"{args.iters} AO-RRT iters"
          + (f" | trial artifacts -> {trials_dir}" if trials_dir else " | trial artifacts off"))
    # master_seed is printed so this exact random seed draw can be reproduced later
    # via --master-seed <value>; seeds themselves are logged per-row in the CSV too.
    print(f"master_seed={master_seed} seeds={seeds}")

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
                    rec = run_trial(cfg, density, col_name, start_x_frac, goal_x_frac,
                                     seed, trials_dir)
                    writer.writerow(rec); fh.flush()
                    rows.append(rec)
                    if not args.quiet:
                        prefix = (f"[{i:4d}/{total}] rocks={density:>3} col={col_name} "
                                  f"seed={seed:>3}")
                        if not rec["aorrt_reached"]:
                            print(f"{prefix} . (goal not reached)")
                        else:
                            cflag = "C" if rec["scp_converged"] else "c"
                            ctrl = f"{rec['control_cost_pct_decrease']:+.1f}%"
                            risk = f"{rec['risk_cost_pct_decrease']:+.1f}%"
                            print(f"{prefix} G{cflag} ctrl={ctrl:>7} risk={risk:>7} "
                                  f"total={rec['total_time_s']}s")

    print(f"\nsaved {len(rows)} trials -> {args.out}  ({time.perf_counter() - t_start:.1f}s)")
    summarize(rows)


if __name__ == "__main__":
    main()
