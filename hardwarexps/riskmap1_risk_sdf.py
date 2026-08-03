"""
Risk-inflated SDF for riskmap1 -- the EXACT mechanism src/risk_planner.py's
RiskSensitiveAORRT (the NN-dynamics risk-aware AO-RRT, _edge_tracking_cvar
path) reads via env.sdf_device when the environment is built with
use_risk_sdf=True (src/environment.py:145-159). This is a DIFFERENT
construction from hardwarexps/riskmap1_boundary_risk.py's Gaussian halo
probability field (that one matches map/risk_map.py's TerrainRiskMap, which
backs RiskSensitiveAORRT's OTHER, OLDER risk term -- _edge_risk, built on
src/dynamics.py's plain noisy rollout, not the NN model). This script instead
replicates map/risk_sdf.py's build_risk_sdf: NOT a soft [0,1] probability
layer -- a genuine signed-distance field, per obstacle inflated OUTWARD by a
CVaR margin, then combined by an elementwise minimum. That inflated field is
what a downstream planner's ordinary "clearance > robot_radius" hard check
runs against, so risk-aversion becomes a bigger hard keep-out, not a soft cost.

THE MATH (map/risk_sdf.py's build_risk_sdf, reproduced exactly, imported and
called unmodified -- not reimplemented):

    For EACH obstacle k independently (its OWN two-pass Euclidean distance
    transform, not a merged/joint distance-to-nearest-obstacle field):
        sdf_k(x) = dist(x, outside mask_k) - dist(x, inside mask_k)     (m)
                   -- positive outside obstacle k, negative inside it
        F_k(x)   = sdf_k(x) - kappa(alpha) * sigma_k                     (m)
                   -- obstacle k's OWN keep-out boundary pushed outward by
                      kappa(alpha)*sigma_k metres

    kappa(alpha) = phi(Phi^-1(1 - alpha)) / alpha
                   -- CVaR (superquantile) multiplier of a standard normal at
                      tail fraction alpha; NOT the plain quantile z_alpha =
                      Phi^-1(1-alpha) -- the CVaR is the (larger) mean of the
                      normal BEYOND that quantile, the correct multiplier for
                      a coherent (sub-additive) risk margin on sigma_k*Z.

    risk_sdf(x)   = min_k F_k(x)          -- combine across obstacles: at
                                              every cell, whichever obstacle's
                                              OWN inflated field is tightest
                                              wins, not necessarily the
                                              nominally nearest one
    argmin_idx(x) = argmin_k F_k(x)       -- which obstacle that was

    risk_sdf clipped to [-d_max, d_max]   (TSDF truncation)

Because this is a MIN of independently-computed continuous fields (not a
single merged distance field paired with a discontinuous nearest-instance
sigma lookup, which is what riskmap1_boundary_risk.py's ORIGINAL now-reverted
approach did and why it had visible seams), risk_sdf itself has NO value
discontinuities -- only the usual SDF-of-a-union gradient kink at points
equidistant (in inflated-field terms) between two obstacles, same as any
ordinary multi-obstacle SDF.

Nothing in src/ or map/ is edited. map.risk_sdf.build_risk_sdf/cvar_kappa are
imported and called completely unmodified; riskmap1_boundary_risk.py's own
rasterize_obstacle_id/assign_sigma/MANUAL_SIGMA/RANDOM_SIGMA_RANGE are reused
unmodified too, so both risk-map scripts start from the exact same obstacle
geometry and sigma assignment -- they only differ in the math applied on top.

    python hardwarexps/riskmap1_risk_sdf.py [--alpha A] [--d_max D] [--seed N] [--out path.png]
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in hardwarexps/)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(HERE)

from map.risk_sdf import build_risk_sdf, cvar_kappa                     # noqa: E402  (read-only reuse, not modified)
from riskmap1_boundary_risk import (                                    # noqa: E402  (read-only reuse, not modified)
    load_map, rasterize_obstacle_id, assign_sigma,
    MANUAL_SIGMA, RANDOM_SIGMA_RANGE, DEFAULT_SEED,
)

PCD_DIR = r"C:\Users\skelkar37\GT\Research F26\pcd\data\processed_pcd"
SDF_NAME = "riskmap6_polygons_cleaned_sdf.npz"
POLY_NAME = "riskmap6_polygons_cleaned.npz"      # explicit override -- riskmap1_boundary_risk.py's own
                                          # load_map() default currently points POLY_NAME at
                                          # riskmap2_polygons.npz; this script always keeps
                                          # riskmap1's sdf + riskmap1's own polygons together

DEFAULT_ALPHA = 0.05    # RiskParams.alpha's default -- same CVaR tail fraction RiskSensitiveAORRT uses
DEFAULT_D_MAX = 6.0     # RiskSDFParams.d_max's default


def build_obstacle_masks(obstacle_id):
    """(K,H,W) bool array, mask[k] = obstacle_id == (k+1) -- build_risk_sdf's
    required input shape, K = number of rock instances."""
    K = int(obstacle_id.max())
    return np.stack([obstacle_id == (k + 1) for k in range(K)], axis=0)


def plot_risk_sdf(sdf, risk_sdf, extent, kappa, alpha, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im_kwargs = dict(origin="lower", extent=extent, aspect="equal")
    vmax = max(abs(sdf.min()), abs(sdf.max()), abs(risk_sdf.min()), abs(risk_sdf.max()), 1e-6)

    ax = axes[0]
    im0 = ax.imshow(sdf, cmap="RdBu", vmin=-vmax, vmax=vmax, **im_kwargs)
    fig.colorbar(im0, ax=ax, label="sdf (m)", fraction=0.046)
    ax.set_title("sdf (no inflation)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1]
    im1 = ax.imshow(risk_sdf, cmap="RdBu", vmin=-vmax, vmax=vmax, **im_kwargs)
    fig.colorbar(im1, ax=ax, label="risk_sdf (m)", fraction=0.046)
    ax.set_title(f"risk-inflated sdf (alpha={alpha})")
    ax.set_xlabel("x (m)")

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved plot -> {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="CVaR tail fraction in (0,1]")
    ap.add_argument("--d_max", type=float, default=DEFAULT_D_MAX, help="TSDF truncation distance (m)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="RNG seed for the randomly-sampled (non-manual) rock sigmas")
    ap.add_argument("--out", default=None, help="save figure to this path instead of showing it")
    args = ap.parse_args()

    sdf, resolution_m, origin, extent, kinds, polygons = load_map(PCD_DIR, SDF_NAME, POLY_NAME)
    obstacle_id = rasterize_obstacle_id(sdf.shape, resolution_m, origin, kinds, polygons)
    s_k = assign_sigma(obstacle_id, MANUAL_SIGMA, RANDOM_SIGMA_RANGE, args.seed)

    kappa = cvar_kappa(args.alpha)
    print(f"alpha={args.alpha}  ->  kappa(alpha) = phi(Phi^-1(1-alpha))/alpha = {kappa:.4f}")
    print("assigned sigmas (obstacle idx -> sigma, m -> inflation kappa*sigma, m):")
    for i, sigma in enumerate(s_k, start=1):
        tag = " (manual)" if i in MANUAL_SIGMA else " (random)"
        print(f"  idx {i}: sigma={sigma:.3f}{tag}  ->  push-out={kappa * sigma:.3f} m")

    obstacle_masks = build_obstacle_masks(obstacle_id)
    risk_sdf, argmin_idx, kappa_check = build_risk_sdf(
        obstacle_masks, s_k, res=resolution_m, alpha=args.alpha, d_max=args.d_max)
    assert abs(kappa_check - kappa) < 1e-9

    plot_risk_sdf(sdf, risk_sdf, extent, kappa, args.alpha, args.out)


if __name__ == "__main__":
    main()
