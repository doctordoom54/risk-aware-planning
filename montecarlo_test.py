"""
Monte Carlo sweep over obstacle density (same structure as montecarlo_run.py) with
the closed-loop tracking ensemble (dynamics_vel.py) ALSO exercised on each trial --
NOT wired into the planner's cost (that's risk_planner.py, still to come), purely for
gathering statistics: for every trial, N_SEGMENTS edges spread along the best (or
best-effort) path each get a tracked_ensemble() call, timed cold (first call for that
edge's nsteps -- includes JIT compile) and warm (repeat calls, already compiled).

Each trial's .npz artifact is a superset of montecarlo_run.py's (grid + path + the
usual metadata) PLUS everything needed to recreate a main_test.py-style tracking plot
later with replot_tracking_trial.py: each segment's reference trajectory, the full
tracked-ensemble trajectories, the disturbance grid, and the gains used. The full
gray background tree is NOT saved (same as montecarlo_run.py's existing artifact --
would dominate storage: a 10-30k-node tree's edge trajectories are ~10x larger than
everything else in this file combined) -- the replot recreates path + highlighted
segments + their tracking-ensemble panels, not the faint full-tree backdrop.

    python montecarlo_test.py [--densities 5 10 15 20 25] [--seeds 20] [--iters 6000]
                               [--n-segments 5] [--out results/montecarlo_test.csv]
                               [--quiet]

Defaults: 5 densities x 3 traverses x 20 seeds = 300 trials.
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
from src.risk_planner import RiskSensitiveAORRT
from map.risk_sdf import cvar_kappa

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# bottom -> top traverses: (name, start_x_frac, goal_x_frac) as fractions of width.
COLUMNS = (
    ("L", 0.15, 0.85),
    ("C", 0.50, 0.50),
    ("R", 0.85, 0.15),
)
BOTTOM_FRAC, TOP_FRAC = 0.13, 0.87

# Trained-envelope command caps (same as main_test.py): commands fed to the
# closed-loop tracker must stay inside the range the residual net was trained on.
# Overrides cfg.aorrt.v_max/w_max for THIS script only -- not config.py's defaults.
V_CMD_CAP = 0.38
W_CMD_CAP = 0.9
N_WARM = 5   # repeat calls per segment for the warm-timing average

FIELDS = [
    "trial_id", "density", "column", "start_x_frac", "goal_x_frac", "seed",
    "start_x", "start_y", "goal_x", "goal_y",
    "reached", "cost", "nodes", "plan_time_s", "t_first_s",
    "nn_time_s", "prop_time_s", "collision_time_s",
    "n_segments", "track_time_s", "track_cold_ms_total", "track_warm_ms_mean",
    "total_time_s",
]


def _disturbance_grid(cfg):
    r = cfg.risk
    ax = np.linspace(-r.ax_dist_max, r.ax_dist_max, r.dist_grid_n)
    yw = np.linspace(-r.yaw_dist_max, r.yaw_dist_max, r.dist_grid_n)
    AX, YW = np.meshgrid(ax, yw)
    return np.stack([AX.ravel(), YW.ravel()], axis=1)


def _pick_edges(pl, n_edges):
    """n_edges parent/child node pairs spread evenly along the best (or best-effort)
    path. Returns [] (not an error) if the tree has no edges to inspect -- some
    trials in a big sweep can end with a trivial tree; the sweep should keep going."""
    chain = pl._chain()
    if len(chain) < 2:
        return []
    n_edges = min(n_edges, len(chain) - 1)
    idxs = sorted(set(np.linspace(1, len(chain) - 1, n_edges, dtype=int).tolist()))
    return [(chain[i - 1], chain[i]) for i in idxs]


def _save_trial_artifact(trials_dir, trial_id, cfg, grid, path, start, goal,
                          density, col_name, seed, cost, reached, D, seg_data):
    """Persist the map+path (as montecarlo_run.py does) plus everything needed to
    recreate a tracking-ensemble plot later: per-segment reference trajectory,
    tracked-ensemble trajectories, timings, and the gains/disturbance grid used.
    Segments have different nsteps (ragged), so Zref/Zens are padded with NaN up to
    the longest segment in THIS trial -- replot code slices back to [:nsteps+1]."""
    r = cfg.risk
    n_seg = len(seg_data)
    M = len(D)
    max_len = (max((s["nsteps"] for s in seg_data), default=0) + 1)
    seg_nsteps = np.zeros(n_seg, dtype=int)
    seg_parent_t = np.zeros(n_seg)
    seg_child_t = np.zeros(n_seg)
    seg_u_ref = np.zeros((n_seg, 2))
    seg_cold_ms = np.zeros(n_seg)
    seg_warm_ms = np.zeros(n_seg)
    seg_Zref = np.full((n_seg, max_len, 5), np.nan)
    seg_Zens = np.full((n_seg, M, max_len, 5), np.nan)
    for k, s in enumerate(seg_data):
        L = s["nsteps"] + 1
        seg_nsteps[k] = s["nsteps"]
        seg_parent_t[k] = s["parent_t"]; seg_child_t[k] = s["child_t"]
        seg_u_ref[k] = s["u_ref"]
        seg_cold_ms[k] = s["cold_ms"]; seg_warm_ms[k] = s["warm_ms"]
        seg_Zref[k, :L] = s["Zref"]
        seg_Zens[k, :, :L] = s["Zens"]

    np.savez_compressed(
        os.path.join(trials_dir, f"{trial_id}.npz"),
        grid_features=grid.features,
        path=path,
        start=np.array(start), goal=np.array(goal),
        width=cfg.env.width, height=cfg.env.height, resolution=cfg.env.resolution,
        density=density, column=col_name, seed=seed,
        cost=(cost if math.isfinite(cost) else np.nan), reached=reached,
        D=D, dt=cfg.dyn.dt, track_gain=r.track_gain, kp_x=r.kp_x, kp_y=r.kp_y,
        k_psi=r.k_psi, v_eps_speed=r.v_eps_speed,
        seg_nsteps=seg_nsteps, seg_parent_t=seg_parent_t, seg_child_t=seg_child_t,
        seg_u_ref=seg_u_ref, seg_cold_ms=seg_cold_ms, seg_warm_ms=seg_warm_ms,
        seg_Zref=seg_Zref, seg_Zens=seg_Zens,
    )


def run_trial(cfg, density, col_name, start_x_frac, goal_x_frac, seed, D, n_segments,
              trials_dir=None):
    """One planning trial + closed-loop tracking-ensemble pass; returns a dict
    matching FIELDS. total_time_s covers the WHOLE trial (map generation, planning,
    tracking-ensemble segments, artifact save) -- plan_time_s/track_time_s are the
    finer-grained breakdown already returned; total_time_s is how long this one trial
    actually took wall-clock, start to finish."""
    t_trial0 = time.perf_counter()
    W, H = cfg.env.width, cfg.env.height
    start = (start_x_frac * W, BOTTOM_FRAC * H)
    goal = (goal_x_frac * W, TOP_FRAC * H)
    trial_id = f"{density}_{col_name}_{seed}"

    # use_risk_sdf=True: same as main_test.py -- collision checking (path_free, the
    # tracked-ensemble CVaR kernel's clearance lookup, everything reading env.sdf)
    # runs against map.risk_sdf's per-obstacle risk-inflated field, at cfg.risk.alpha.
    grid, env = make_map_env(cfg, seed, num_rocks=density, start=start, goal=goal,
                             use_risk_sdf=True)

    pl = RiskSensitiveAORRT(env, cfg, start, goal)
    t0 = time.perf_counter()
    path = pl.plan(verbose=False)
    plan_time = time.perf_counter() - t0
    reached = pl.goal_reached()

    r = cfg.risk; dt = cfg.dyn.dt
    clip_lo = (r.ax_clip_lo, r.yaw_clip_lo)
    clip_hi = (r.ax_clip_hi, r.yaw_clip_hi)
    Kp = (r.kp_x, r.kp_y)

    seg_data = []
    t_track0 = time.perf_counter()
    for parent, child in _pick_edges(pl, n_segments):
        u_ref = np.clip(child.u, [-V_CMD_CAP, -W_CMD_CAP], [V_CMD_CAP, W_CMD_CAP])
        nsteps = child.nsteps

        u_max = (V_CMD_CAP, W_CMD_CAP)
        t0 = time.perf_counter()
        Zens = pl.model.tracked_ensemble(parent.x, u_ref, nsteps, dt, D, r.track_gain,
                                          clip_lo, clip_hi, Kp, r.k_psi, r.v_eps_speed, u_max)
        t_cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(N_WARM):
            pl.model.tracked_ensemble(parent.x, u_ref, nsteps, dt, D, r.track_gain,
                                       clip_lo, clip_hi, Kp, r.k_psi, r.v_eps_speed, u_max)
        t_warm = (time.perf_counter() - t0) / N_WARM

        seg_data.append(dict(parent_t=parent.t, child_t=child.t, u_ref=u_ref,
                              nsteps=nsteps, Zref=child.edgeX, Zens=Zens,
                              cold_ms=t_cold * 1e3, warm_ms=t_warm * 1e3))
    track_time = time.perf_counter() - t_track0

    if trials_dir is not None:
        _save_trial_artifact(trials_dir, trial_id, cfg, grid, path, start, goal,
                              density, col_name, seed, pl.best_cost, reached, D, seg_data)

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
        n_segments=len(seg_data),
        track_time_s=round(track_time, 4),
        track_cold_ms_total=(round(sum(s["cold_ms"] for s in seg_data), 2) if seg_data else ""),
        track_warm_ms_mean=(round(float(np.mean([s["warm_ms"] for s in seg_data])), 3) if seg_data else ""),
        total_time_s=round(time.perf_counter() - t_trial0, 4),
    )


def summarize(rows):
    densities = sorted({r["density"] for r in rows})
    print("\n=== summary (per density) ===")
    print(f"{'rocks':>5} {'trials':>6} {'success':>8} {'mean_cost':>10} {'mean_plan_s':>11} "
          f"{'mean_track_s':>12} {'mean_total_s':>12}")
    for d in densities:
        rs = [r for r in rows if r["density"] == d]
        n = len(rs)
        succ = [r for r in rs if r["reached"]]
        costs = [r["cost"] for r in succ if r["cost"] != ""]
        ptimes = [r["plan_time_s"] for r in rs]
        ttimes = [r["track_time_s"] for r in rs]
        totals = [r["total_time_s"] for r in rs]
        print(f"{d:>5} {n:>6} {len(succ) / n:>7.0%} "
              f"{(np.mean(costs) if costs else float('nan')):>10.3f} "
              f"{np.mean(ptimes):>11.3f} {np.mean(ttimes):>12.3f} {np.mean(totals):>12.3f}")


def main():
    # dated results/montecarlo_<timestamp>/ folder by default, so a run never
    # overwrites a previous one -- pass --out explicitly to opt out.
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.join(RESULTS_DIR, f"montecarlo_{run_stamp}", "montecarlo_test.csv")

    ap = argparse.ArgumentParser(description="Obstacle-density Monte Carlo sweep with closed-loop tracking-ensemble stats.")
    ap.add_argument("--densities", type=int, nargs="+", default=[5,6,7,8])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--cost-mode", default="control", choices=("time", "control", "both"))
    ap.add_argument("--n-segments", type=int, default=5, help="tracking-ensemble segments per trial")
    ap.add_argument("--out", default=default_out,
                    help="CSV output path; defaults to a fresh dated results/montecarlo_<timestamp>/ folder")
    ap.add_argument("--trials-dir", default=None,
                    help="dir to save per-trial artifacts (default: <out>_trials/ next to the CSV)")
    ap.add_argument("--no-save-trials", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    # same on/off toggle as main_test.py's "cvar"/"nocvar" positional arg -- default
    # (neither flag given) leaves cfg.risk.use_edge_cvar at its config.py default.
    ap.add_argument("--cvar", dest="use_edge_cvar", action="store_true", default=None,
                    help="force RiskParams.use_edge_cvar ON for this sweep")
    ap.add_argument("--no-cvar", dest="use_edge_cvar", action="store_false",
                    help="force RiskParams.use_edge_cvar OFF for this sweep")
    args = ap.parse_args()

    cfg = PlannerConfig()
    cfg.aorrt.cost_mode = args.cost_mode
    cfg.aorrt.max_iterations = args.iters
    cfg.aorrt.v_max = V_CMD_CAP
    cfg.aorrt.w_max = W_CMD_CAP
    if args.use_edge_cvar is not None:
        cfg.risk.use_edge_cvar = args.use_edge_cvar

    D = _disturbance_grid(cfg)   # same for every trial -- built once

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    total = len(args.densities) * len(COLUMNS) * len(seeds)

    trials_dir = None
    if not args.no_save_trials:
        trials_dir = args.trials_dir or (os.path.splitext(args.out)[0] + "_trials")
        os.makedirs(trials_dir, exist_ok=True)

    print(f"risk_sdf: ON, alpha={cfg.risk.alpha} kappa={cvar_kappa(cfg.risk.alpha):.3f} | "
          f"edge_cvar={'ON' if cfg.risk.use_edge_cvar else 'off'} weight={cfg.risk.edge_cvar_weight}")
    print(f"Monte Carlo (+tracking ensemble): {len(args.densities)} densities x {len(COLUMNS)} columns x "
          f"{len(seeds)} seeds = {total} trials | cost='{args.cost_mode}', {args.iters} iters, "
          f"M={len(D)} disturbance grid, {args.n_segments} segments/trial"
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
                    rec = run_trial(cfg, density, col_name, start_x_frac, goal_x_frac,
                                     seed, D, args.n_segments, trials_dir)
                    writer.writerow(rec); fh.flush()
                    rows.append(rec)
                    if not args.quiet:
                        flag = "G" if rec["reached"] else "."
                        cost = rec["cost"] if rec["cost"] != "" else "  -  "
                        tk = rec["track_cold_ms_total"] if rec["track_cold_ms_total"] != "" else "  -  "
                        print(f"[{i:4d}/{total}] rocks={density:>3} col={col_name} "
                              f"seed={seed:>3} {flag} cost={cost} "
                              f"t={rec['plan_time_s']:.2f}s track={rec['track_time_s']:.2f}s "
                              f"total={rec['total_time_s']:.2f}s (cold_total={tk}ms)")

    print(f"\nsaved {len(rows)} trials -> {args.out}  ({time.perf_counter() - t_start:.1f}s)")
    summarize(rows)


if __name__ == "__main__":
    main()
