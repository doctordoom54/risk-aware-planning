"""
Recreate a montecarlo_test.py trial's tracking-ensemble plot (map+path with
highlighted segments, one zoomed closed-loop tracking-ensemble panel per segment)
from its saved .npz artifact -- same idea as replot_trial.py, but for the extended
artifacts written by montecarlo_test.py. NOTE: the full gray background tree is NOT
in the artifact (not saved -- see montecarlo_test.py's module docstring), so this
plot shows path + highlighted segments, not the faint full-tree backdrop that
main_test.py's live run shows.

    python replot_tracking_trial.py                                        # most recent trial
    python replot_tracking_trial.py results/montecarlo_test_trials/8_C_5.npz

Opens an interactive plot window -- does not save to disk.
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TRIALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "montecarlo_test_trials")
SEGMENT_COLORS = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]


def _default_npz(trials_dir=DEFAULT_TRIALS_DIR):
    candidates = glob.glob(os.path.join(trials_dir, "*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"no .npz trial artifacts found in {trials_dir} -- "
            "run montecarlo_test.py first, or pass a path explicitly."
        )
    return max(candidates, key=os.path.getmtime)


def replot(npz_path):
    d = np.load(npz_path)
    grid_features = d["grid_features"]
    path = d["path"]
    start = d["start"]; goal = d["goal"]
    width = d["width"].item(); height = d["height"].item()
    density = d["density"].item(); column = d["column"].item(); seed = d["seed"].item()
    cost = d["cost"].item(); reached = bool(d["reached"].item())

    seg_nsteps = d["seg_nsteps"]
    seg_parent_t = d["seg_parent_t"]; seg_child_t = d["seg_child_t"]
    seg_u_ref = d["seg_u_ref"]; seg_Zref = d["seg_Zref"]; seg_Zens = d["seg_Zens"]
    track_gain = d["track_gain"].item(); kp_x = d["kp_x"].item(); kp_y = d["kp_y"].item()
    k_psi = d["k_psi"].item()
    n_seg = len(seg_nsteps)
    M = seg_Zens.shape[1] if n_seg else 0

    ext = [0, width, 0, height]
    obst = np.ma.masked_where(grid_features == 0, grid_features)

    ncols = max(n_seg, 1)
    fig = plt.figure(figsize=(4.2 * ncols, 11))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.4, 1])
    axm = fig.add_subplot(gs[0, :])
    axz_list = [fig.add_subplot(gs[1, k]) for k in range(ncols)]

    axm.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
               alpha=0.85, interpolation="bilinear")
    axm.plot(path[:, 0], path[:, 1], "-", color="tab:cyan", lw=2.0, label="AO-RRT", zorder=3)
    for k in range(n_seg):
        L = int(seg_nsteps[k]) + 1
        col = SEGMENT_COLORS[k % len(SEGMENT_COLORS)]
        Zref = seg_Zref[k, :L]
        axm.plot(Zref[:, 0], Zref[:, 1], "-", color=col, lw=3.0, zorder=5, label=f"segment {k + 1}")
    axm.plot(*start, "o", color="lime", ms=12, mec="k")
    axm.plot(*goal, "*", color="magenta", ms=20, mec="k")
    axm.set_xlim(0, width); axm.set_ylim(0, height); axm.set_aspect("equal")
    axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)"); axm.legend(loc="upper left", fontsize=9, ncol=2)
    axm.set_title(f"density={density} col={column} seed={seed} "
                  f"({'reached' if reached else 'not reached'}, cost={cost:.2f})")

    for k, axz in enumerate(axz_list):
        if k >= n_seg:
            axz.axis("off")
            continue
        L = int(seg_nsteps[k]) + 1
        col = SEGMENT_COLORS[k % len(SEGMENT_COLORS)]
        Zref = seg_Zref[k, :L]; Zens = seg_Zens[k, :, :L]
        cmap = plt.get_cmap("autumn")
        for m in range(M):
            mcol = cmap(m / max(1, M - 1))
            axz.plot(Zens[m, :, 0], Zens[m, :, 1], "-", color=mcol, lw=0.8, alpha=0.7, zorder=2,
                      label="tracked (disturbed)" if m == 0 else None)
        axz.plot(Zref[:, 0], Zref[:, 1], "-", color="black", lw=0.9, zorder=4, label="reference")
        axz.plot(*Zref[0, :2], "o", color="lime", ms=8, mec="k", zorder=5, label="parent")
        axz.plot(*Zref[-1, :2], "*", color="magenta", ms=13, mec="k", zorder=5, label="child")

        pts = np.vstack([Zref[:, :2]] + [Zens[m, :, :2] for m in range(M)])
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
        axz.set_title(f"segment {k + 1}: t={seg_parent_t[k]:.1f}-{seg_child_t[k]:.1f}s\n"
                       f"u_ref=[{seg_u_ref[k, 0]:.2f},{seg_u_ref[k, 1]:.2f}], "
                       f"nsteps={int(seg_nsteps[k])}", fontsize=9, color=col)

    fig.suptitle(f"Closed-loop tracking ensemble (M={M}) on {n_seg} path segments, "
                 f"K={track_gain}, Kp=({kp_x},{kp_y}), k_psi={k_psi}", fontsize=12)

    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Recreate a saved montecarlo_test.py trial's tracking-ensemble plot.")
    ap.add_argument("npz", nargs="?", default=None,
                    help="path to the trial's .npz artifact "
                         f"(default: most recently written .npz in {DEFAULT_TRIALS_DIR})")
    args = ap.parse_args()
    npz_path = args.npz or _default_npz()
    print(f"replotting {npz_path}")
    replot(npz_path)


if __name__ == "__main__":
    main()
