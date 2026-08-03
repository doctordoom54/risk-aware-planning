"""
Risk-inflated signed distance field (standalone; NOT wired into any planner yet).

Nominal SDFs treat an obstacle boundary as certain. Here each obstacle k carries
its own boundary-uncertainty std s_k (drawn from the same range as
map.sdf.ObstacleSigmaField -- see src.config.RiskSDFParams / EnvParams.obstacle_sigma_min/
max), and the field is inflated PER OBSTACLE, before combining across obstacles, so
that the combination step (an elementwise min) always picks the obstacle whose
inflated keep-out is tightest at that cell -- not necessarily the nominally nearest
one. This module is split into an offline NumPy/SciPy build (Step 1) and an online
pure-JAX query (Step 2), plus a small collision predicate / ensemble reduction
(Step 3) and an integration-ready dataclass wrapper (Step 5).

Usage (build once, jit the query, evaluate an ensemble in one call):

    from map.risk_sdf import build_risk_sdf, RiskSDF

    risk_sdf, argmin_idx, kappa = build_risk_sdf(obstacle_masks, s_k, res=0.05,
                                                  alpha=0.05, d_max=2.0)
    field = RiskSDF(risk_sdf, origin=(0.0, 0.0), res=0.05, kappa=kappa,
                     argmin_idx=argmin_idx)
    query_jit = jax.jit(field.query)
    clearances = query_jit(traj_points)          # traj_points: (B, T, 2) world (x, y)
    L = ensemble_worst_clearance(clearances)      # (B,)
"""
import numpy as np
from scipy.stats import norm

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from .sdf import distance_transform_edt


# --------------------------------------------------------------------------- #
# Step 1 -- offline build (NumPy / SciPy)
# --------------------------------------------------------------------------- #
def cvar_kappa(alpha):
    """CVaR (superquantile) multiplier of a standard normal at tail fraction alpha:
        kappa = norm.pdf(norm.ppf(1 - alpha)) / alpha
    NOT the quantile z_alpha = norm.ppf(1 - alpha) itself -- see build_risk_sdf's
    docstring for the distinction. Exposed standalone (not just inlined inside
    build_risk_sdf) so a caller that needs the multiplier WITHOUT building a whole
    field can get it directly -- e.g. environment.make_map_env sizing the start/goal
    clear-disc radius wide enough for the worst-case inflation any nearby obstacle
    could apply, before the risk-SDF itself is even built.

    alpha=0 is rejected rather than silently producing kappa=NaN: NaN < x is always
    False in NumPy, so build_risk_sdf's running-min would silently never update for
    ANY obstacle, producing a flat d_max field (no obstacle keep-out at all) instead
    of erroring."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    z_alpha = norm.ppf(1.0 - alpha)          # VaR quantile (not returned/used further)
    return float(norm.pdf(z_alpha) / alpha)


def build_risk_sdf(obstacle_masks, s_k, res, alpha, d_max):
    """Per-obstacle risk-inflated signed distance field.

    For EACH obstacle k independently:
        sdf_k = (distance_transform_edt(~mask_k) - distance_transform_edt(mask_k)) * res
    (two EDT passes -- exterior distance minus interior distance -- so sdf_k is
    positive outside obstacle k and strictly negative inside it). A single EDT on
    the union of all masks is NOT used: per-obstacle identity has to survive until
    AFTER inflation, otherwise the boundary-uncertainty correction of the
    nominally-nearest obstacle would be applied even when a farther-but-more-
    uncertain obstacle is actually the tighter constraint (see Step 4 test (b)).

    kappa = norm.pdf(norm.ppf(1 - alpha)) / alpha is the CVaR (superquantile) of a
    standard normal at tail fraction alpha -- NOT the quantile z_alpha =
    norm.ppf(1 - alpha) itself. The quantile only tells you the threshold that is
    exceeded with probability alpha; the CVaR is the (larger) MEAN of the normal
    beyond that threshold, i.e. E[Z | Z > z_alpha], which is the right multiplier
    for a coherent (sub-additive, convex) risk margin on the boundary offset
    s_k * Z. Each obstacle's field is then

        F_k = sdf_k - kappa * s_k[k]

    i.e. the keep-out boundary is pushed outward by kappa * s_k[k] metres. The
    combination across obstacles is a running elementwise minimum (risk_sdf) with a
    parallel int argmin raster (argmin_idx) recording which obstacle is tightest at
    each cell -- computed one obstacle at a time so peak memory is O(running min +
    argmin + one current field), never O(K) fields at once.

    Finally the field is truncated to a TSDF: clipped to [-d_max, d_max].

    Args:
        obstacle_masks: (K, H, W) bool array, one boolean mask per obstacle.
        s_k:            (K,) boundary-uncertainty std per obstacle (metres).
        res:            metres per pixel.
        alpha:          CVaR tail fraction in (0, 1).
        d_max:          TSDF truncation distance (metres).

    Returns:
        risk_sdf   : (H, W) float64 array, the risk-inflated, truncated SDF (m).
        argmin_idx : (H, W) int64 array, index k of the tightest obstacle per cell.
        kappa      : float, the CVaR multiplier used (see above).
    """
    obstacle_masks = np.asarray(obstacle_masks, dtype=bool)
    if obstacle_masks.ndim != 3:
        raise ValueError(f"obstacle_masks must be (K, H, W), got shape {obstacle_masks.shape}")
    K, H, W = obstacle_masks.shape
    s_k = np.asarray(s_k, dtype=np.float64)
    if s_k.shape != (K,):
        raise ValueError(f"s_k must have shape ({K},), got {s_k.shape}")

    kappa = cvar_kappa(alpha)

    risk_sdf = np.full((H, W), np.inf, dtype=np.float64)
    argmin_idx = np.zeros((H, W), dtype=np.int64)

    for k in range(K):
        mask_k = obstacle_masks[k]
        sdf_k = (distance_transform_edt(~mask_k) -
                 distance_transform_edt(mask_k)) * res
        F_k = sdf_k - kappa * s_k[k]
        better = F_k < risk_sdf
        risk_sdf = np.where(better, F_k, risk_sdf)
        argmin_idx = np.where(better, k, argmin_idx)

    risk_sdf = np.clip(risk_sdf, -d_max, d_max)
    return risk_sdf.astype(np.float64), argmin_idx.astype(np.int64), kappa


# --------------------------------------------------------------------------- #
# Step 2 -- online batch lookup (pure JAX, jit / vmap safe)
# --------------------------------------------------------------------------- #
def query_risk_sdf(risk_sdf_jnp, points, origin, res):
    """Bilinearly interpolate a baked risk-SDF at world-frame points.

    points is (N, 2) or any (..., 2) batch of world (x, y) coordinates -- e.g.
    (B, T, 2) for B trajectories x T timesteps. World -> continuous grid
    coordinates via `origin` (ox, oy) and `res` (metres/pixel), then bilinear
    interpolation with jax.scipy.ndimage.map_coordinates(order=1, mode='nearest')
    (nearest-clamp outside the grid, so points just past the border still get a
    sane -- not NaN/garbage -- clearance).

    No Python branching on traced values: the only shape-dependent step is the
    flatten/reshape around map_coordinates, and shapes are static under jit, so
    this function is jax.jit-able and vmap-safe as-is (vmap just adds another
    static leading axis).

    Returns:
        clearances with the same leading batch shape as `points` (points.shape[:-1]).
    """
    ox, oy = origin
    points = jnp.asarray(points)
    batch_shape = points.shape[:-1]
    pts_flat = points.reshape(-1, 2)

    gx = (pts_flat[:, 0] - ox) / res
    gy = (pts_flat[:, 1] - oy) / res
    coords = jnp.stack([gy, gx], axis=0)   # (2, N): axis0 -> row (H), axis1 -> col (W)

    vals = map_coordinates(risk_sdf_jnp, coords, order=1, mode='nearest')
    return vals.reshape(batch_shape)


# --------------------------------------------------------------------------- #
# Step 3 -- collision predicate and ensemble reduction
# --------------------------------------------------------------------------- #
def collision_margin(clearances, robot_radius, sample_spacing):
    """Point-wise collision-free predicate with a tunneling-safe margin.

    A point is safe iff clearance > robot_radius + sample_spacing / 2. The field
    is 1-Lipschitz (it's a signed distance field), so clearance can drop by at
    most `sample_spacing` between two trajectory samples spaced `sample_spacing`
    apart; without the extra sample_spacing/2 margin, two adjacent "safe" discrete
    samples could straddle an obstacle whose penetration never registers at either
    sample point (tunneling). Requiring the half-spacing buffer rules that out.
    """
    clearances = jnp.asarray(clearances)
    return clearances > (robot_radius + sample_spacing / 2.0)


def ensemble_worst_clearance(clearances):
    """Worst-case (least-clearance) loss per trajectory in an ensemble.

    clearances: (B, T) or (B, T, ...) -- batch of B trajectories.
    Returns L_j = -min over all non-batch axes (t, and points, if present), shape
    (B,): the worst-alpha trajectory check (e.g. CVaR over the ensemble) becomes a
    single reduction over L.
    """
    clearances = jnp.asarray(clearances)
    reduce_axes = tuple(range(1, clearances.ndim))
    return -jnp.min(clearances, axis=reduce_axes)


# --------------------------------------------------------------------------- #
# Step 5 -- integration wrapper
# --------------------------------------------------------------------------- #
class RiskSDF:
    """Thin holder for a baked risk-inflated SDF plus its geometry.

    .query(points) is pure JAX (safe to wrap in jax.jit / jax.vmap at the call
    site; this class keeps no SciPy in the query path). .rebuild(alpha) reruns
    the offline NumPy/SciPy build (Step 1) and is REQUIRED whenever alpha or the
    map (obstacle_masks / s_k) changes: the argmin raster baked at build time is
    only valid for the alpha (and s_k) it was built with, since a different alpha
    rescales kappa and can flip which obstacle is tightest at a given cell (Step 4
    test (b)). A cheaper "cache s_argmin per cell and rescale" trick is only valid
    if the argmin is alpha-stable everywhere, which isn't guaranteed -- so this
    wrapper always rebakes the whole field instead of trying to patch it in place.
    """

    def __init__(self, risk_sdf, origin, res, kappa, argmin_idx,
                 obstacle_masks=None, s_k=None, d_max=None):
        self.field = jnp.asarray(risk_sdf)
        self.origin = tuple(origin)
        self.res = float(res)
        self.kappa = float(kappa)
        self.argmin_idx = np.asarray(argmin_idx)
        # kept only so .rebuild(alpha) can rerun Step 1; not used by .query
        self._obstacle_masks = obstacle_masks
        self._s_k = s_k
        self._d_max = d_max

    def query(self, points):
        return query_risk_sdf(self.field, points, self.origin, self.res)

    def rebuild(self, alpha):
        if self._obstacle_masks is None or self._s_k is None or self._d_max is None:
            raise ValueError(
                "RiskSDF.rebuild requires obstacle_masks, s_k and d_max to have "
                "been passed to __init__ (or set on the instance) -- it reruns "
                "the full offline build, it does not patch the baked field.")
        risk_sdf, argmin_idx, kappa = build_risk_sdf(
            self._obstacle_masks, self._s_k, self.res, alpha, self._d_max)
        self.field = jnp.asarray(risk_sdf)
        self.argmin_idx = argmin_idx
        self.kappa = kappa
        return self
