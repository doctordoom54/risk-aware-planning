"""
Multi-obstacle JOINT collision-probability risk map (review/visualization script).

Generates a terrain (hills/valleys via undulation, craters, rocks) with per-obstacle
boundary-uncertainty std s_k (map.sdf.ObstacleSigmaField, one draw per obstacle
INSTANCE), then builds a risk field that treats EVERY obstacle instance as its own
independent failure mode instead of TerrainRiskMap's default "nearest obstacle only"
approximation:

    r_k(p)   = exp(-0.5 * (sdf_k(p) / s_k)^2)     -- per-obstacle Gaussian halo,
                                                       sdf_k = NOMINAL (uninflated)
                                                       distance to obstacle k only,
                                                       clamped >=0 (0 at/inside k)
    r_k(p)  <- 0                                   where sdf_k(p) > CLIP_N_SIGMA*s_k
                                                       (negligible-contribution guard)
    p_obs(p) = 1 - prod_k (1 - w_obs * r_k(p))     -- joint OR over ALL obstacles

combined with the SAME terrain layers TerrainRiskMap already uses (slope, roughness --
which already encode hills/valleys/craters through surface elevation), via the
identical probabilistic-OR fusion:

    risk(p) = 1 - (1 - slope_risk)(1 - roughness_risk)(1 - p_obs(p))

This differs from TerrainRiskMap.obstacle_risk, which uses ONE combined (union) SDF
and the nearest obstacle's sigma only -- so it misses compounding risk in the gap
between two obstacles that are close together relative to their sigmas. See
map/risk_map.py and map/risk_sdf.py (the analogous per-obstacle-before-combine
argument for the HARD keep-out field) for the two approximations this generalizes.

Does not save anything -- opens two interactive plot windows:
    1. heatmap of the joint collision-probability risk field, everywhere on the map.
    2. terrain/obstacle context: elevation (hills/valleys/craters), rock/crater
       obstacle footprints, and slope contours (mobility-relevant thresholds).

Settable below: SEED, SIGMA_SEED, NUM_ROCKS, NUM_CRATERS, NUM_HILLS, obstacle sigma
range, and the layer weights.

    python joint_risk_map_test.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import Environment
from map import Map2D, GridMap, LunarMapGenerator
from map.sdf import distance_transform_edt

# ── settable knobs ──────────────────────────────────────────────────────────────
SEED = 69                    # terrain generation (rocks/craters/hills placement)
SIGMA_SEED = 13693              # per-obstacle boundary-uncertainty std draws

NUM_ROCKS = 3
NUM_CRATERS = 0
NUM_HILLS = 1                 # Gaussian hill/valley terrain undulation features

OBSTACLE_SIGMA_MIN = 0.08     # m, EACH obstacle instance draws its own std
OBSTACLE_SIGMA_MAX = 0.30     # uniform in [MIN, MAX] (independent per instance)

W_SLOPE = 0.3                 # layer weights in the joint OR-combine, in [0, 1]
W_ROUGH = 0.3
W_OBS = 1.0

CLIP_N_SIGMA = 6.0             # zero a per-obstacle halo beyond this many sigmas
                               # (negligible contribution, exp(-0.5*6^2) ~ 1.5e-8;
                               # numerical/efficiency guard, not a correctness fix)


def joint_obstacle_risk(obstacle_id, per_obstacle_sigma, res_m, w_obs, clip_n_sigma):
    """Joint (all-obstacles) collision-probability field via probabilistic OR.

    obstacle_id      : (H, W) int, 0 = free space, k = obstacle instance k (1-indexed)
    per_obstacle_sigma: (K,) boundary-uncertainty std (m), per_obstacle_sigma[k-1] is
                         instance k's std
    Returns (H, W) float64 in [0, 1].
    """
    H, W = obstacle_id.shape
    keep = np.ones((H, W), dtype=np.float64)     # running product of (1 - w_obs*r_k)
    K = int(per_obstacle_sigma.shape[0])
    for k in range(1, K + 1):
        mask_k = (obstacle_id == k)
        if not mask_k.any():
            continue
        s_k = float(per_obstacle_sigma[k - 1])
        # nominal (uninflated) distance to obstacle k ONLY; 0 at/inside its boundary,
        # exactly SignedDistanceField.compute's convention but per-obstacle instead
        # of the union mask.
        d_k = distance_transform_edt(~mask_k).astype(np.float64) * res_m
        r_k = np.exp(-0.5 * (d_k / s_k) ** 2)
        r_k[d_k > clip_n_sigma * s_k] = 0.0        # negligible-contribution guard
        keep *= (1.0 - np.clip(w_obs * r_k, 0.0, 1.0))
    return np.clip(1.0 - keep, 0.0, 1.0)


def build_grid(cfg):
    e = cfg.env
    np.random.seed(SEED)
    grid = GridMap(Map2D(e.width, e.height, e.resolution))
    gen = LunarMapGenerator(grid)
    gen.generate(num_rocks=NUM_ROCKS, num_craters=NUM_CRATERS,
                rock_radius_min=e.rock_radius_min, rock_radius_max=e.rock_radius_max)
    if NUM_HILLS > 0:
        gen.add_terrain_undulation(num_hills=NUM_HILLS, seed=SEED)
    grid.compute_slope_map()
    return grid


def plot_risk_heatmap(grid, risk, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    W, H = grid.map.width_m, grid.map.height_m
    ext = [0, W, 0, H]
    im = ax.imshow(risk, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="collision probability")
    xs = np.linspace(0, W, risk.shape[1])
    ys = np.linspace(0, H, risk.shape[0])
    ax.contour(xs, ys, (grid.features > 0).astype(float), levels=[0.5],
              colors="cyan", linewidths=0.6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(title)
    handles = [Line2D([0], [0], color="cyan", lw=0.6, label="obstacle boundary")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()


def plot_terrain_context(grid, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    W, H = grid.map.width_m, grid.map.height_m
    ext = [0, W, 0, H]
    im = ax.imshow(grid.surface, origin="lower", extent=ext, cmap="terrain")
    fig.colorbar(im, ax=ax, label="elevation (m)")

    xs = np.linspace(0, W, grid.features.shape[1])
    ys = np.linspace(0, H, grid.features.shape[0])

    rocks = np.ma.masked_where(grid.features != 1, np.ones_like(grid.features, dtype=float))
    ax.imshow(rocks, origin="lower", extent=ext, cmap=ListedColormap(["dimgray"]),
              vmin=0, vmax=1, alpha=0.85)
    ax.contour(xs, ys, (grid.features == 2).astype(float), levels=[0.5],
              colors="red", linewidths=1.2)

    slope_deg = np.degrees(grid.slope_map)
    cs = ax.contour(xs, ys, slope_deg, levels=[10, 20, 30], colors="gold",
                    linewidths=0.7, alpha=0.9)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d deg")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(title)
    handles = [
        Patch(facecolor="dimgray", alpha=0.85, label="rock obstacle"),
        Line2D([0], [0], color="red", lw=1.2, label="crater obstacle boundary"),
        Line2D([0], [0], color="gold", lw=0.7, label="slope contour (deg)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()


def main():
    cfg = PlannerConfig()

    grid = build_grid(cfg)

    env = Environment(grid, disc_radius=cfg.env.disc_radius, clearance=cfg.env.clearance,
                      with_risk=True, risk_weights=(W_SLOPE, W_ROUGH, 0.0),
                      obstacle_sigma_min=OBSTACLE_SIGMA_MIN,
                      obstacle_sigma_max=OBSTACLE_SIGMA_MAX, sigma_seed=SIGMA_SEED)

    K = int(grid.obstacle_id.max())
    p_obs = joint_obstacle_risk(grid.obstacle_id, env.per_obstacle_sigma,
                                env.res_m, W_OBS, CLIP_N_SIGMA)

    slope_risk = env.risk.slope_risk
    roughness_risk = env.risk.roughness_risk
    risk = 1.0 - (1.0 - slope_risk) * (1.0 - roughness_risk) * (1.0 - p_obs)
    risk = np.clip(risk, 0.0, 1.0).astype(np.float32)

    print(f"joint risk map: {cfg.env.width}x{cfg.env.height} m @ {cfg.env.resolution} m/px, "
          f"{K} obstacle instances (rocks+craters), sigma=[{OBSTACLE_SIGMA_MIN},{OBSTACLE_SIGMA_MAX}] m "
          f"seed={SEED} sigma_seed={SIGMA_SEED} | weights slope={W_SLOPE} rough={W_ROUGH} obs={W_OBS} "
          f"| risk range [{risk.min():.4f}, {risk.max():.4f}], mean={risk.mean():.4f}")

    plot_risk_heatmap(grid, risk,
                      f"Joint collision-probability risk map ({K} obstacles, "
                      f"sigma=[{OBSTACLE_SIGMA_MIN},{OBSTACLE_SIGMA_MAX}] m)")
    plot_terrain_context(grid, "Terrain context: elevation, obstacles, slope")

    plt.show()


if __name__ == "__main__":
    main()
