"""
Recreate a Monte Carlo trial's map+path plot from its saved .npz artifact
(written by montecarlo_run.py into <out>_trials/<trial_id>.npz, unless run
with --no-save-trials). trial_id is the "trial_id" column in the sweep CSV,
formatted as "{density}_{column}_{seed}".

    python replot_trial.py                                        # most recent trial in DEFAULT_TRIALS_DIR
    python replot_trial.py results/montecarlo_trials/10_L_3.npz
    python replot_trial.py results/montecarlo_trials/10_L_3.npz --out my_plot.png
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TRIALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "montecarlo_trials")


def _default_npz(trials_dir=DEFAULT_TRIALS_DIR):
    """Most recently written .npz in trials_dir -- lets the script run with no args."""
    candidates = glob.glob(os.path.join(trials_dir, "*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"no .npz trial artifacts found in {trials_dir} -- "
            "run montecarlo_run.py first, or pass a path explicitly."
        )
    return max(candidates, key=os.path.getmtime)


def replot(npz_path, out_path=None):
    d = np.load(npz_path)
    grid_features = d["grid_features"]
    path = d["path"]
    start = d["start"]; goal = d["goal"]
    width = d["width"].item(); height = d["height"].item()
    density = d["density"].item(); column = d["column"].item(); seed = d["seed"].item()
    cost = d["cost"].item(); reached = bool(d["reached"].item())

    ext = [0, width, 0, height]
    obst = np.ma.masked_where(grid_features == 0, grid_features)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(obst, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
              alpha=0.85, interpolation="bilinear")
    ax.plot(path[:, 0], path[:, 1], "-", color="tab:cyan", lw=2.0, label="AO-RRT", zorder=3)
    s = max(1, len(path) // 25)
    ax.quiver(path[::s, 0], path[::s, 1], np.cos(path[::s, 2]), np.sin(path[::s, 2]),
              color="tab:blue", scale=25, width=0.004, zorder=4)
    ax.plot(*start, "o", color="lime", ms=12, mec="k")
    ax.plot(*goal, "*", color="magenta", ms=20, mec="k")
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"density={density} col={column} seed={seed} "
                 f"({'reached' if reached else 'not reached'}, cost={cost:.2f})")

    if out_path is None:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")
        out_path = os.path.join(results_dir, f"replot_{base}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path

npz_path =  os.path.join(DEFAULT_TRIALS_DIR, "8_C_5.npz")


def main():
    ap = argparse.ArgumentParser(description="Recreate a saved Monte Carlo trial's map+path plot.")
    ap.add_argument("npz", nargs="?", default=None,
                    help="path to the trial's .npz artifact "
                         f"(default: most recently written .npz in {DEFAULT_TRIALS_DIR})")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: results/plots/replot_<trial_id>.png)")
    args = ap.parse_args()
    npz_path = args.npz or _default_npz()
    print(f"replotting {npz_path}")
    out = replot(npz_path, args.out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
