"""
Plot a saved SCP trajectory CSV (results/riskmap1_risk_aware_scp_trajectory_
seed<N>.csv, written by pcd_risk_aware_planner.py) directly over the riskmap5
boundary polygons (map boundary + rock outlines), in raw (x, y) world
coordinates.

Both the polygons (riskmap5_polygons.npz) and the trajectory CSV already
store x, y in the SAME true-world-coordinate frame -- neither needs an
origin shift to line up with the other. What DOES need care is the axis
window: this map's world origin is NOT (0, 0) (riskmap5's map boundary
polygon's own bounding box can start below zero, e.g. around x=-0.225,
y=-0.535 -- see MEMORY/prior discussion of this same origin issue in
pcd_risk_aware_planner.py). Hardcoding plot limits to start at (0, 0) would
silently crop part of the map or misalign it -- see this same repo's
pcd_risk_aware_planner.py history. To avoid that entirely, axis limits here
are derived directly from the map polygon's OWN bounding box (with a small
margin), never assumed to start at (0, 0).

Nothing outside hardwarexps/ is touched or imported from -- this script only
reads the polygons .npz and the trajectory .csv directly with numpy.

    python hardwarexps/plot_trajectory_on_polygons.py [seed]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in hardwarexps/)

PCD_DIR = r"C:\Users\skelkar37\GT\Research F26\pcd\data\processed_pcd"
POLY_NAME = "riskmap6_polygons_cleaned.npz"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CSV_NAME_FMT = "riskmap1_risk_aware_scp_trajectory_seed{seed}.csv"

MARGIN_FRAC = 0.05   # extra breathing room around the map polygon's own bbox, as a fraction of its span


def load_polygons(pcd_dir=PCD_DIR, poly_name=POLY_NAME):
    poly_npz = np.load(os.path.join(pcd_dir, poly_name), allow_pickle=True)
    return poly_npz["kinds"], poly_npz["polygons"]


def load_trajectory_csv(path):
    """x,y,theta,v_b,omega,t (header row, comma-delimited -- same layout
    pcd_risk_aware_planner.save_trajectory_csv writes). Returns the (N,6) array."""
    return np.loadtxt(path, delimiter=",", skiprows=1)


def map_bounds(kinds, polygons, margin_frac=MARGIN_FRAC):
    """(x_min, x_max, y_min, y_max) straight off the map polygon's own
    vertices, padded by margin_frac of its span -- NOT assumed to start at
    (0, 0), since this map's true world origin is negative in both axes."""
    map_poly = np.asarray(polygons[list(kinds).index("map")], float)
    x_min, y_min = map_poly.min(axis=0)
    x_max, y_max = map_poly.max(axis=0)
    mx = (x_max - x_min) * margin_frac
    my = (y_max - y_min) * margin_frac
    return x_min - mx, x_max + mx, y_min - my, y_max + my


def plot_trajectory_on_polygons(kinds, polygons, traj, seed, csv_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    for kind, poly in zip(kinds, polygons):
        poly = np.asarray(poly, float)
        closed = np.vstack([poly, poly[0]])
        if kind == "map":
            ax.plot(closed[:, 0], closed[:, 1], "-", color="tab:blue", lw=1.5, zorder=2, label="map boundary")
        else:
            ax.fill(closed[:, 0], closed[:, 1], color="tab:orange", alpha=0.55, zorder=1)
    ax.plot([], [], color="tab:orange", lw=6, alpha=0.55, label="rock")   # single legend entry for all rocks

    x, y = traj[:, 0], traj[:, 1]
    ax.plot(x, y, "-", color="tab:red", lw=2.0, zorder=3, label=f"Risk Aware SCP trajectory (seed {seed})")
    ax.plot(x[0], y[0], "o", color="lime", ms=12, mec="k", zorder=5, label="start")
    ax.plot(x[-1], y[-1], "*", color="magenta", ms=18, mec="k", zorder=5, label="end")

    x_min, x_max, y_min, y_max = map_bounds(kinds, polygons)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Risk Aware SCP trajectory (seed {seed}) over riskmap5 boundary polygons")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    max_omega = np.max(np.abs(traj[:, 4]))
    print(f"plotted {os.path.basename(csv_path)} over {POLY_NAME} "
          f"({len(x)} trajectory points, map bbox x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}])")
    print(f"max |omega| = {max_omega:.4f} rad/s")
    plt.show()


def main():
    args = sys.argv[1:]
    seed = int(args[0]) if args else 102

    csv_path = os.path.join(RESULTS_DIR, CSV_NAME_FMT.format(seed=seed))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist -- run pcd_risk_aware_planner.py with "
                                 f"this seed (and savecsv/SAVE_TRAJECTORY_CSV=True) first.")

    kinds, polygons = load_polygons()
    traj = load_trajectory_csv(csv_path)
    plot_trajectory_on_polygons(kinds, polygons, traj, seed, csv_path)


if __name__ == "__main__":
    main()
