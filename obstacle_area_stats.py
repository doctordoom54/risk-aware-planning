"""
Average obstacle-area fraction (% of the 6x6 m course footprint covered by
obstacles) per rock density, computed from a montecarlo_test.py trial dump.

Reads every per-trial .npz artifact in a results/montecarlo_<timestamp>/
montecarlo_test_trials/ folder, groups them by `density` (= num_rocks passed
to make_map_env -- 5/6/7/8 by default), and for each trial computes
    obstacle_area_frac = (# grid cells with grid_features > 0) * resolution^2
                          / (width * height)
then averages that fraction across all trials (all start/goal columns and
seeds) sharing the same density. NOTE: grid_features is saved AFTER
make_map_env's clear_disc() carves obstacle-free discs around start/goal, so
this is the actual obstacle footprint the planner saw, not the raw
pre-clearing rock field.

    python obstacle_area_stats.py                                    # most recent results/montecarlo_* run
    python obstacle_area_stats.py results/montecarlo_20260714_004733  # explicit run dir
"""
import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _default_run_dir():
    candidates = [p for p in glob.glob(os.path.join(RESULTS_DIR, "montecarlo_*")) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError(f"no results/montecarlo_<timestamp>/ run folders found in {RESULTS_DIR}")
    return max(candidates, key=os.path.getmtime)


def obstacle_area_frac(npz_path):
    d = np.load(npz_path)
    features = d["grid_features"]
    resolution = d["resolution"].item()
    width = d["width"].item(); height = d["height"].item()
    obstacle_area = float(np.count_nonzero(features)) * resolution ** 2
    course_area = width * height
    return int(d["density"].item()), obstacle_area / course_area


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="results/montecarlo_<timestamp>/ folder "
                         f"(default: most recently modified in {RESULTS_DIR})")
    args = ap.parse_args()
    run_dir = args.run_dir or _default_run_dir()
    trials_dir = os.path.join(run_dir, "montecarlo_test_trials")
    npz_paths = sorted(glob.glob(os.path.join(trials_dir, "*.npz")))
    if not npz_paths:
        sys.exit(f"no .npz trial artifacts found in {trials_dir}")

    by_density = defaultdict(list)
    for p in npz_paths:
        density, frac = obstacle_area_frac(p)
        by_density[density].append(frac)

    print(f"run: {trials_dir}  ({len(npz_paths)} trials)\n")
    print(f"{'density':>7} {'n':>4} {'mean %':>8} {'std %':>7} {'min %':>7} {'max %':>7}")
    for density in sorted(by_density):
        fracs = np.array(by_density[density]) * 100.0
        print(f"{density:>7} {len(fracs):>4} {fracs.mean():>8.3f} {fracs.std():>7.3f} "
              f"{fracs.min():>7.3f} {fracs.max():>7.3f}")


if __name__ == "__main__":
    main()
