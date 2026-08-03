"""
Standalone visual smoke test for map.risk_sdf -- NOT a pytest file (no asserts,
opens an interactive matplotlib window). Not wired into any planner.

Generates ONE random map (obstacle count / radius range / boundary-uncertainty
range all read from src.config.PlannerConfig) and rebuilds the risk-inflated SDF
at several alpha (CVaR tail fraction) values off that SAME obstacle_masks/s_k, so
the panels are a controlled comparison of alpha's effect on kappa/inflation alone.
Shows 5 SDF panels (one per alpha) plus a 6th panel with the per-obstacle boundary
uncertainty s_k that produced them all. Does not save anything to disk.

    python -m map.risk_sdf_demo [--seed S]
    python map/risk_sdf_demo.py [--seed S]
"""
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Project root must come BEFORE any path Python added for us, so that "import map"
# resolves to the map/ PACKAGE (map/__init__.py) rather than the sibling module
# map/map.py -- when this file is run directly (not via `-m`), Python auto-inserts
# this file's own directory (map/) at sys.path[0], and map/map.py would shadow the
# package if the project root were only appended instead of inserted first.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from src.config import PlannerConfig
from map import Map2D, GridMap, LunarMapGenerator
from map.risk_sdf import build_risk_sdf


def build_random_map(cfg, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)   # LunarMapGenerator.generate uses the global RNG

    e = cfg.env
    grid = GridMap(Map2D(e.width, e.height, e.resolution))
    LunarMapGenerator(grid).generate(
        num_rocks=e.num_rocks, num_craters=0,
        rock_radius_min=e.rock_radius_min, rock_radius_max=e.rock_radius_max)

    K = int(grid.obstacle_id.max())
    obstacle_masks = np.stack([grid.obstacle_id == (k + 1) for k in range(K)], axis=0)
    s_k = rng.uniform(e.obstacle_sigma_min, e.obstacle_sigma_max, size=K)
    return grid, obstacle_masks, s_k


def sigma_display(obstacle_masks, s_k, shape):
    """(H, W) array of each cell's own obstacle's s_k, masked (NaN) over free space."""
    disp = np.full(shape, np.nan, dtype=np.float64)
    for k in range(obstacle_masks.shape[0]):
        disp[obstacle_masks[k]] = s_k[k]
    return disp


ALPHAS = [1.0, 0.8, 0.2, 0.5, 0.05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=29120)
    args = ap.parse_args()

    cfg = PlannerConfig()
    grid, obstacle_masks, s_k = build_random_map(cfg, args.seed)
    res = grid.map.resolution_m
    origin = (grid.map.origin_x, grid.map.origin_y)
    extent = [origin[0], origin[0] + grid.map.width_m,
              origin[1], origin[1] + grid.map.height_m]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    # same obstacle_masks/s_k for every panel -- only alpha (-> kappa) changes, so
    # the panels isolate what alpha does to the inflation/argmin, nothing else.
    results = [build_risk_sdf(obstacle_masks, s_k, res, alpha=a, d_max=cfg.risk_sdf.d_max)
               for a in ALPHAS]
    vmax = max(np.abs(r[0]).max() for r in results)

    for ax, alpha, (risk_sdf, _argmin_idx, kappa) in zip(axes, ALPHAS, results):
        im = ax.imshow(risk_sdf, origin="lower", extent=extent, cmap="RdBu",
                        vmin=-vmax, vmax=vmax)
        ax.set_title(f"alpha={alpha}, kappa={kappa:.2f}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label="clearance (m)")

    ax_sigma = axes[len(ALPHAS)]
    sigma_field = sigma_display(obstacle_masks, s_k, (grid.map.size_y, grid.map.size_x))
    cmap_sigma = plt.get_cmap("plasma").copy()
    cmap_sigma.set_bad(color="0.9")   # free space -> light gray
    im_sigma = ax_sigma.imshow(sigma_field, origin="lower", extent=extent, cmap=cmap_sigma,
                                vmin=cfg.env.obstacle_sigma_min, vmax=cfg.env.obstacle_sigma_max)
    ax_sigma.set_title("Per-obstacle boundary uncertainty s_k")
    ax_sigma.set_xlabel("x (m)"); ax_sigma.set_ylabel("y (m)")
    fig.colorbar(im_sigma, ax=ax_sigma, label="s_k (m)")

    fig.suptitle(f"Risk-inflated SDF vs. alpha (K={obstacle_masks.shape[0]} obstacles, seed={args.seed})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
