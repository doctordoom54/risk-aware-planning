"""
Gaussian-boundary-only risk map for the pcd-derived riskmap1 arena.

This is deliberately NOT the same thing as the hard-collision SDF used by
AORRT (hardwarexps/test_exp.py, env.collision_free / env.path_free): the
obstacles here are never geometrically inflated, disc_radius/clearance never
enter this file at all.

Each rock gets its OWN independent Gaussian halo -- its own distance-to-that-
rock's-boundary, its own sigma -- which decays to 0 smoothly in every
direction on its own:

    r_i(cell) = 1                                     if cell is inside rock i
              = exp(-0.5 * (dist_to_rock_i(cell) / sigma_i)^2)   otherwise

All 8 per-rock layers are then fused as independent failure modes
(probabilistic OR, same combination rule map/risk_map.py's TerrainRiskMap
uses to fuse its slope/roughness/obstacle layers):

    p_fail = 1 - prod_i (1 - r_i)

NOTE on an earlier version of this script: it instead assigned ONE sigma per
obstacle and broadcast it to "whichever cell is nearest that obstacle"
(map.sdf.ObstacleSigmaField.broadcast's Voronoi-partition approach, which is
what the rest of this repo uses elsewhere). That partitions the whole map by
nearest-obstacle-id, and sigma jumps discontinuously exactly at the boundary
between two obstacles' partitions -- even though distance-to-nearest-obstacle
itself is continuous there, so exp(-0.5*(d/sigma)^2) visibly jumps at that
seam (looks like a hard wall, and a slow-decaying big-sigma obstacle looks
like it "never decays" because its whole partition is smaller than a few
sigma). Giving every obstacle its own halo and OR-fusing them instead avoids
that: no partitioning, no seams, each halo genuinely decays to 0 on its own.

Manual sigma assignment (rock instance index = index into riskmap1_polygons.npz's
"polygons" array, 1..8, since index 0 is the "map" boundary entry):

    idx 1 -> 0.40   (higher uncertainty)
    idx 4 -> 0.09   (lower uncertainty)
    idx 7 -> 0.05   (lower uncertainty)
    all other rocks (2, 3, 5, 6, 8) -> sampled uniform(0.1, 0.5)

Nothing in src/ or map/ is edited.

    python hardwarexps/riskmap1_boundary_risk.py [--seed N] [--out path.png]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.ndimage import distance_transform_edt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in hardwarexps/)
sys.path.append(ROOT_DIR)

PCD_DIR = r"C:\Users\skelkar37\GT\Research F26\pcd\data\processed_pcd"
SDF_NAME = "riskmap6_polygons_cleaned_sdf.npz"
POLY_NAME = "riskmap6_polygons_cleaned.npz"

                      
MANUAL_SIGMA = {1: 0.08,4:0.1, 3:0.03,6:0.03, 5:0.08, 2: 0.01, 7:0.25}      # obstacle instance idx -> sigma (m)1
RANDOM_SIGMA_RANGE = (0.01, 0.15)                 # range for every other rock
DEFAULT_SEED = 0                                # reproducible draw for the "remaining" rocks


def load_map(pcd_dir=PCD_DIR, sdf_name=SDF_NAME, poly_name=POLY_NAME):
    sdf_npz = np.load(os.path.join(pcd_dir, sdf_name))
    sdf = sdf_npz["sdf"]
    resolution_m = float(sdf_npz["resolution_m"])
    origin = sdf_npz["origin"]
    extent = tuple(sdf_npz["extent"])

    poly_npz = np.load(os.path.join(pcd_dir, poly_name), allow_pickle=True)
    kinds = poly_npz["kinds"]
    polygons = poly_npz["polygons"]             # already in the SDF's frame, no _mirror_x needed
    return sdf, resolution_m, origin, extent, kinds, polygons


def rasterize_obstacle_id(sdf_shape, resolution_m, origin, kinds, polygons):
    """(H,W) int grid: 0 = free space, else the rock's own index into `polygons`
    (1..8) -- matches the numbering used for MANUAL_SIGMA and the earlier
    idx/centroid table. Only interior cells are labelled (a true polygon fill,
    not a nearest-neighbour assignment) -- each mask (obstacle_id == idx) is
    exactly that rock's own footprint, used below to build its own independent
    distance field."""
    H, W = sdf_shape
    xs = origin[0] + np.arange(W) * resolution_m    # cell coordinate convention matches
    ys = origin[1] + np.arange(H) * resolution_m    # src.environment's _bilin_cell (no +0.5 offset)
    gx, gy = np.meshgrid(xs, ys)                     # (H,W) each
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    obstacle_id = np.zeros((H, W), dtype=np.int32)
    for idx, (kind, poly) in enumerate(zip(kinds, polygons)):
        if kind != "rock":
            continue
        path = mpath.Path(np.asarray(poly, float))
        inside = path.contains_points(pts).reshape(H, W)
        obstacle_id[inside] = idx
    return obstacle_id


def assign_sigma(obstacle_id, manual_sigma, random_range, seed):
    """(K,) array, s_k[i] = obstacle id (i+1)'s sigma."""
    num_obs = int(obstacle_id.max())
    rng = np.random.default_rng(seed)
    s_k = rng.uniform(random_range[0], random_range[1], size=num_obs)
    for idx, sigma in manual_sigma.items():
        s_k[idx - 1] = sigma
    return s_k


def fused_obstacle_risk(obstacle_id, s_k, resolution_m):
    """Each obstacle instance gets its own independent Gaussian halo (own
    distance-to-ITS-OWN boundary, own sigma), then all instances are combined
    as independent failure modes (probabilistic OR) -- collision probability
    is always exactly 1 inside any obstacle, and every halo decays smoothly
    to 0 in isolation, with no cross-obstacle partitioning/seams."""
    keep = np.ones(obstacle_id.shape, dtype=np.float64)
    for i, sigma in enumerate(s_k, start=1):
        mask = obstacle_id == i
        if not mask.any():
            continue
        dist_m = distance_transform_edt(~mask) * resolution_m
        r_i = np.where(mask, 1.0, np.exp(-0.5 * (dist_m / sigma) ** 2))
        keep *= (1.0 - r_i)
    return 1.0 - keep


def plot_risk_map(obstacle_id, risk, extent, kinds, polygons, s_k, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im_kwargs = dict(origin="lower", extent=extent, aspect="equal")

    ax = axes[0]
    n_obs = int(obstacle_id.max())
    ax.imshow(obstacle_id, cmap="tab10", vmin=0, vmax=max(9, n_obs), **im_kwargs)
    for idx, (kind, poly) in enumerate(zip(kinds, polygons)):
        if kind != "rock":
            continue
        poly = np.asarray(poly, float)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        ax.text(cx, cy, f"{idx}\nsigma={s_k[idx - 1]:.2f}", color="white", fontsize=9,
                fontweight="bold", ha="center", va="center")
    ax.set_title("obstacle instance id + assigned sigma")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1]
    im1 = ax.imshow(risk, cmap="YlOrRd", vmin=0, vmax=1, **im_kwargs)
    for kind, poly in zip(kinds, polygons):
        poly = np.asarray(poly, float)
        closed = np.vstack([poly, poly[0]])
        color = "k" if kind == "map" else "cyan"
        ax.plot(closed[:, 0], closed[:, 1], "-", color=color, lw=0.8, alpha=0.8)
    fig.colorbar(im1, ax=ax, label="p_fail", fraction=0.046)
    ax.set_title("Gaussian-boundary risk map (per-obstacle halo, OR-fused)")
    ax.set_xlabel("x (m)")

    fig.suptitle("riskmap1 -- obstacle-boundary-only risk (no geometric inflation)")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved plot -> {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="RNG seed for the randomly-sampled (non-manual) rock sigmas")
    ap.add_argument("--out", default=None, help="save figure to this path instead of showing it")
    args = ap.parse_args()

    sdf, resolution_m, origin, extent, kinds, polygons = load_map()
    obstacle_id = rasterize_obstacle_id(sdf.shape, resolution_m, origin, kinds, polygons)
    s_k = assign_sigma(obstacle_id, MANUAL_SIGMA, RANDOM_SIGMA_RANGE, args.seed)

    print("assigned sigmas (obstacle idx -> sigma, m):")
    for i, sigma in enumerate(s_k, start=1):
        tag = " (manual)" if i in MANUAL_SIGMA else " (random)"
        print(f"  idx {i}: {sigma:.3f}{tag}")

    risk = fused_obstacle_risk(obstacle_id, s_k, resolution_m)

    plot_risk_map(obstacle_id, risk, extent, kinds, polygons, s_k, args.out)


if __name__ == "__main__":
    main()
