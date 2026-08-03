"""
Environment adapter: maps.GridMap (+ optional TerrainRiskMap) -> the queries the
planner needs, in metres.

    .width / .height        map extent (m)
    .collision_free(x, y)   disc-model clearance test (one bilinear SDF lookup)
    .in_bounds(x, y)        disc-radius margin to the map border
    .risk_at(x, y)          terrain failure-probability in [0,1] 

Optimal-cost planning only uses collision_free / in_bounds; risk_at is exposed so
the risk term can be switched on later without touching the planner.
"""
import math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

from map import Map2D, GridMap, LunarMapGenerator, TerrainRiskMap
from map.sdf import distance_transform_edt, ObstacleSigmaField
from map.risk_sdf import build_risk_sdf, cvar_kappa


def _bilin_cell(sdf, c):
    """Bilinear sample of `sdf` at cell coordinate c = [gx, gy]. Differentiable:
    floor() has zero gradient a.e., so jax.grad flows through the interpolation
    weights and returns the exact analytic bilinear gradient."""
    H, W = sdf.shape
    gx = jnp.clip(c[0], 0.0, W - 1.0)
    gy = jnp.clip(c[1], 0.0, H - 1.0)
    x0 = jnp.floor(gx).astype(jnp.int32); y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1); y1 = jnp.minimum(y0 + 1, H - 1)
    wx = gx - x0; wy = gy - y0
    v0 = sdf[y0, x0] * (1 - wx) + sdf[y0, x1] * wx
    v1 = sdf[y1, x0] * (1 - wx) + sdf[y1, x1] * wx
    return v0 * (1 - wy) + v1 * wy


# value + gradient of the bilinear SDF at one cell coordinate, vmapped over points
_sdf_value_and_grad = jit(vmap(jax.value_and_grad(_bilin_cell, argnums=1),
                               in_axes=(None, 0)))

# value-ONLY bilinear SDF lookup, vmapped -- no autodiff, so cheaper than
# _sdf_value_and_grad when the gradient isn't needed. Meant to be called from INSIDE
# another jitted function (e.g. RiskSensitiveAORRT's fused edge-CVaR kernel in
# risk_planner.py) so a whole rollout+SDF+reduction pipeline stays on-device.
_sdf_value_only = jit(vmap(_bilin_cell, in_axes=(None, 0)))


def _true_signed_sdf(features):
    """TRUE signed distance (cells): +distance to the obstacle in free space,
    -distance to the boundary INSIDE obstacles. Unlike SignedDistanceField.compute
    (which clamps the interior to 0), this keeps a non-zero gradient inside an
    obstacle, so the linearised SDF keep-out can push a knot back out."""
    obst = (np.asarray(features) > 0)
    if not obst.any():
        return np.full(obst.shape, max(obst.shape), dtype=np.float64)
    d_out = distance_transform_edt(~obst)      # free  -> distance to obstacle (0 on obstacle)
    d_in = distance_transform_edt(obst)        # obst  -> distance to free     (0 in free)
    return (d_out - d_in).astype(np.float64)   # >0 outside, <0 inside


def _bilinear(grid, gx, gy):
    H, W = grid.shape
    gx = min(max(gx, 0.0), W - 1.0); gy = min(max(gy, 0.0), H - 1.0)
    x0 = int(math.floor(gx)); y0 = int(math.floor(gy))
    x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1)
    wx = gx - x0; wy = gy - y0
    v0 = grid[y0, x0] * (1 - wx) + grid[y0, x1] * wx
    v1 = grid[y1, x0] * (1 - wx) + grid[y1, x1] * wx
    return v0 * (1 - wy) + v1 * wy


def _bilinear_batch(grid, gx, gy):
    """Vectorised bilinear lookup for arrays of grid coordinates."""
    H, W = grid.shape
    gx = np.clip(gx, 0.0, W - 1.0); gy = np.clip(gy, 0.0, H - 1.0)
    x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    wx = gx - x0; wy = gy - y0
    v0 = grid[y0, x0] * (1 - wx) + grid[y0, x1] * wx
    v1 = grid[y1, x0] * (1 - wx) + grid[y1, x1] * wx
    return v0 * (1 - wy) + v1 * wy


class Environment:
    def __init__(self, grid_map, disc_radius=0.4, clearance=0.0,
                 risk_weights=(0.0, 0.0, 1.0),
                 obstacle_sigma_min=0.1, obstacle_sigma_max=0.6, sigma_seed=None,
                 with_risk=False,
                 use_risk_sdf=False, risk_sdf_alpha=0.05, risk_sdf_d_max=2.0):
        """
        use_risk_sdf    : if True, self.sdf (and everything derived from it --
                          collision_free/path_free/sdf_and_grad/sdf_device/
                          body_clearance_batch) is map.risk_sdf.build_risk_sdf's
                          OUTPUT instead of the plain nominal SDF: each obstacle's
                          keep-out is inflated by kappa(risk_sdf_alpha) * s_k BEFORE
                          the per-obstacle min, so every hard-collision consumer gets
                          the boundary-uncertainty margin "for free" -- no separate
                          CVaR-gaussian term needs to be added downstream (see
                          risk_planner.edge_tracking_cvar). Off by default so existing
                          callers (e.g. main_risk.py, which compares two alphas
                          against the SAME built map/env) keep their current,
                          uncertainty-free hard-feasibility gate unless they opt in.
        risk_sdf_alpha  : CVaR tail fraction used to build the field when
                          use_risk_sdf=True. Bake-time only -- unlike RiskParams.alpha
                          elsewhere, this is fixed for the env's lifetime (see
                          RiskSDF.rebuild's docstring in map/risk_sdf.py for why the
                          field must be REBUILT, not patched, if alpha should change).
        risk_sdf_d_max  : TSDF truncation distance (m) when use_risk_sdf=True.
        """
        self.grid = grid_map
        self.res_m = grid_map.map.resolution_m
        self.resolution = 1.0 / self.res_m          # cells per metre
        self.width = grid_map.map.width_m
        self.height = grid_map.map.height_m
        self.disc_radius = disc_radius
        self.clearance = clearance
        # stochastic-obstacle boundary-uncertainty std (m): obstacles have UNCERTAIN
        # boundaries (the Gaussian halo in TerrainRiskMap, and -- when
        # use_risk_sdf=True -- the inflated keep-out in self.sdf itself). EACH obstacle
        # instance draws its own std ~ uniform(obstacle_sigma_min, obstacle_sigma_max)
        # rather than sharing one constant -- see map.sdf.ObstacleSigmaField.
        # per_obstacle_sigma (K,) and obstacle_sigma_map (per-cell broadcast) share the
        # SAME draw (sample_per_obstacle then broadcast) so the risk-SDF build below and
        # TerrainRiskMap's halo never silently diverge on what s_k a given obstacle has.
        self.obstacle_sigma_min = obstacle_sigma_min
        self.obstacle_sigma_max = obstacle_sigma_max
        obstacle_id = getattr(grid_map, "obstacle_id",
                              (grid_map.features > 0).astype(np.int32))
        sigma_rng = np.random.default_rng(sigma_seed)
        self.per_obstacle_sigma = ObstacleSigmaField.sample_per_obstacle(
            obstacle_id, obstacle_sigma_min, obstacle_sigma_max, rng=sigma_rng)
        self.obstacle_sigma_map = np.ascontiguousarray(
            ObstacleSigmaField.broadcast(obstacle_id, self.per_obstacle_sigma,
                                         obstacle_sigma_max))
        self._obstacle_sigma_dev = jnp.asarray(self.obstacle_sigma_map)

        # TRUE signed distance (negative inside obstacles) so the linearised SDF
        # keep-out has a valid outward gradient everywhere.
        sdf_px = _true_signed_sdf(grid_map.features)
        nominal_sdf = np.ascontiguousarray(sdf_px * self.res_m)         # metres, signed
        self.use_risk_sdf = use_risk_sdf
        if use_risk_sdf:
            # obstacle_masks (K,H,W): mask k is obstacle instance (k+1) -- same
            # instance numbering as per_obstacle_sigma, so s_k[k] lines up with
            # obstacle_masks[k]. K=0 (no obstacles) is handled by build_risk_sdf
            # itself (empty loop -> field stays at d_max everywhere).
            K = int(self.per_obstacle_sigma.shape[0])
            obstacle_masks = np.stack([obstacle_id == (k + 1) for k in range(K)], axis=0) \
                if K > 0 else np.zeros((0,) + obstacle_id.shape, dtype=bool)
            self.sdf, self._risk_sdf_argmin, self.risk_sdf_kappa = build_risk_sdf(
                obstacle_masks, self.per_obstacle_sigma, self.res_m,
                alpha=risk_sdf_alpha, d_max=risk_sdf_d_max)
        else:
            self.sdf = nominal_sdf
            self._risk_sdf_argmin = None
            self.risk_sdf_kappa = None
        self.H, self.W = self.sdf.shape
        self._sdf_dev = jnp.asarray(self.sdf)        # device copy for autodiff lookups
        # risk map is built only if needed (the risk-aware / CVaR phase); share the
        # SAME per-cell sigma field so it isn't resampled with a different draw.
        self.risk = (TerrainRiskMap(grid_map, sdf=sdf_px, weights=risk_weights,
                                    obstacle_sigma_map=self.obstacle_sigma_map)
                     if with_risk else None)
        # device copy of the terrain-risk grid for autodiff value+gradient lookups
        self._risk_dev = (jnp.asarray(self.risk.risk.astype(np.float64))
                          if self.risk is not None else None)
        # terrain field that DRIVES wheel slip: slope + roughness (the physical causes
        # of slip), normalised to [0,1]. Falls back to the combined risk if those
        # layers are flat. Used by the risk-sensitive planners for terrain-dependent
        # slip, so rough/steep ground is both costlier AND more uncertain.
        self._slip_field = None
        if self.risk is not None:
            sl = np.asarray(getattr(self.risk, "slope_risk", 0.0), float)
            ro = np.asarray(getattr(self.risk, "roughness_risk", 0.0), float)
            f = sl + ro
            if np.ndim(f) == 0 or float(np.max(f)) <= 1e-9:
                f = np.asarray(self.risk.risk, float)
            mx = float(np.max(f))
            self._slip_field = (f / mx) if mx > 1e-9 else np.zeros_like(self.sdf)

    def collision_free(self, x, y):
        return _bilinear(self.sdf, x * self.resolution, y * self.resolution) > \
            self.clearance + self.disc_radius

    def in_bounds(self, x, y):
        r = self.disc_radius
        return (r <= x <= self.width - r) and (r <= y <= self.height - r)

    def valid(self, x, y):
        return self.in_bounds(x, y) and self.collision_free(x, y)

    def path_free(self, X, extra_margin=0.0):
        """Vectorised: True iff every point of trajectory X (m,>=2) is in bounds and
        disc-clearance-free. One batched SDF lookup for the whole edge. `extra_margin`
        inflates the required clearance — used for risk-aware planning against
        stochastic (uncertain-boundary) obstacles."""
        px = X[:, 0]; py = X[:, 1]; r = self.disc_radius + extra_margin
        if (px.min() < r or px.max() > self.width - r or
                py.min() < r or py.max() > self.height - r):
            return False
        vals = _bilinear_batch(self.sdf, px * self.resolution, py * self.resolution)
        return bool(np.all(vals > self.clearance + r))

    def risk_at(self, x, y):
        return 0.0 if self.risk is None else self.risk.risk_at(x, y)

    def risk_and_grad(self, pts):
        """Batched terrain failure-risk + EXACT spatial gradient (autodiff) for
        points `pts` (N,2) in metres -> (vals (N,), grads (N,2)). Risk is in [0,1]
        (higher = riskier); grad is the gradient of that field w.r.t. world position.
        Returns zeros if the environment was built without a risk map."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self._risk_dev is None:
            return np.zeros(len(pts)), np.zeros((len(pts), 2))
        cells = jnp.asarray(pts) * self.resolution
        v, g = _sdf_value_and_grad(self._risk_dev, cells)
        return np.asarray(v), np.asarray(g) * self.resolution

    def risk_vals(self, pts):
        """Batched terrain risk values only (N,) in metres-space; zeros if no risk."""
        return self.risk_and_grad(pts)[0]

    def slip_field_vals(self, pts):
        """Terrain slip-driver in [0,1] (slope+roughness) at points (N,2) m; zeros if
        the environment has no risk map."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self._slip_field is None:
            return np.zeros(len(pts))
        return _bilinear_batch(self._slip_field, pts[:, 0] * self.resolution,
                               pts[:, 1] * self.resolution)

    def slip_scale(self, pts, gain):
        """Local wheel-slip multiplier  1 + gain * terrain[0,1]  at points (N,2) m.
        gain=0 (or no risk map) -> ones (uniform slip everywhere)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        if self._slip_field is None or gain <= 0:
            return np.ones(len(pts))
        return 1.0 + gain * self.slip_field_vals(pts)

    def sdf_at(self, x, y):
        """Signed distance to the nearest obstacle (metres), bilinear-interpolated."""
        return _bilinear(self.sdf, x * self.resolution, y * self.resolution)

    def obstacle_sigma_at(self, x, y):
        """Boundary-uncertainty std (metres) of whichever obstacle is nearest to
        (x, y), bilinear-interpolated over the per-cell sigma field."""
        return _bilinear(self.obstacle_sigma_map, x * self.resolution, y * self.resolution)

    @property
    def obstacle_sigma_device(self):
        """On-device per-cell sigma field, for callers building fused JAX kernels
        (e.g. RiskSensitiveAORRT's edge-CVaR kernel) that need the LOCAL obstacle's
        boundary std rather than the map-wide constant this used to be."""
        return self._obstacle_sigma_dev

    def sdf_and_grad(self, pts):
        """Batched signed distance + EXACT spatial gradient (autodiff) for points
        `pts` (N,2) in metres. Returns (vals (N,), grads (N,2)), both in metres —
        grad is the gradient of the SDF w.r.t. world position. Used by SCP for the
        linearised keep-out:  sdf(p) + grad . (p - p_prev) >= margin."""
        pts = np.atleast_2d(np.asarray(pts, float))
        cells = jnp.asarray(pts) * self.resolution
        v, g = _sdf_value_and_grad(self._sdf_dev, cells)
        return np.asarray(v), np.asarray(g) * self.resolution

    @property
    def sdf_device(self):
        """Raw on-device signed-distance array (metres, TRUE/unclamped), for callers
        building their OWN fused JAX kernels (e.g. RiskSensitiveAORRT's edge-CVaR
        kernel) that need to stay on-device across a rollout + SDF lookup + reduction
        -- unlike sdf_and_grad/body_clearance_batch, which always round-trip through
        numpy and compute an autodiff gradient even when only the value is needed."""
        return self._sdf_dev

    def body_clearance_batch(self, X):
        """Batched, UNCLAMPED body-surface clearance (signed distance minus
        disc_radius) for an ensemble of trajectories `X` (..., >=2) in metres — e.g.
        (M, T, 5) tracked-ensemble rollouts from VelPoseDynamics.tracked_ensemble.
        Only the leading [..., 0:2] (x, y) is used. One batched sdf_and_grad call over
        all points, so this stays a single jitted lookup regardless of ensemble size.

        Because self.sdf is the TRUE signed distance (negative inside obstacles, not
        clamped to 0 — see _true_signed_sdf), subtracting disc_radius here is a pure
        shift: it re-centres clearance on the rover's DISC SURFACE rather than its
        center point, so a member whose disc actually overlaps an obstacle reports a
        genuine negative clearance (by exactly how many metres), not a clipped flag.
        Returns an array shaped like X.shape[:-1], e.g. (M, T)."""
        X = np.asarray(X, dtype=float)
        lead_shape = X.shape[:-1]
        pts = X[..., 0:2].reshape(-1, 2)
        sdf_vals, _ = self.sdf_and_grad(pts)
        return sdf_vals.reshape(lead_shape) - self.disc_radius


def clear_disc(grid_map, p, radius_m):
    """Zero out obstacle features within radius_m of world point p — used to keep
    the start / goal regions obstacle-free so planning stays feasible."""
    res = grid_map.map.resolution_m
    r = int(math.ceil(radius_m / res))
    j0, i0 = grid_map.map.meters_to_indices(p[0], p[1])
    H, W = grid_map.features.shape
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            if dj * dj + di * di <= r * r:
                j, i = j0 + dj, i0 + di
                if 0 <= j < H and 0 <= i < W:
                    grid_map.features[j, i] = 0
                    grid_map.occupancy[j, i] = 0
                    grid_map.obstacle_id[j, i] = 0


def make_map_env(cfg, seed, num_rocks=None, start=None, goal=None,
                 clear_radius=None, with_risk=False, num_craters=0, undulation=0,
                 risk_weights=None, obstacle_sigma_min=None, obstacle_sigma_max=None,
                 rock_radius_min=None, rock_radius_max=None,
                 use_risk_sdf=False, risk_sdf_alpha=None, risk_sdf_d_max=None):
    """Build a map + Environment for a seed, carving obstacle-free discs around the
    start and goal so they never fall on a rock. Returns (grid, env).

    For risk-aware planning, give the terrain risk genuine spatial structure that is
    NOT just obstacle proximity (the default risk_weights (0,0,1) makes risk == the
    obstacle halo, which the SDF planner already avoids). Use:
        `undulation` > 0  -> smooth Gaussian hills/valleys (slope + roughness), the
                             right tool for the small Leo arena;
        `num_craters` > 0 -> craters (auto-scaled to fit the map);
        `risk_weights` = (slope, roughness, obstacle), e.g. (0.6, 0.4, 0.3).
    The slope/roughness layers also drive terrain-dependent wheel slip (RiskParams).
    rock_radius_min/max (m): override cfg.env.rock_radius_min/max for this call.
    obstacle_sigma_min/max (m): override cfg.env.obstacle_sigma_min/max for this call --
    EACH obstacle instance draws its own boundary-uncertainty std uniform in this range
    (see map.sdf.ObstacleSigmaField), rather than sharing one constant.

    use_risk_sdf : if True, env.sdf is map.risk_sdf.build_risk_sdf's per-obstacle
                   risk-inflated field instead of the plain nominal SDF -- see
                   Environment.__init__. Off by default (opt-in per call).
    risk_sdf_alpha : CVaR tail fraction for the risk-SDF build; defaults to
                   cfg.risk.alpha (the SAME tail fraction the rest of the risk-aware
                   planner uses for this run), NOT cfg.risk_sdf.alpha -- keeping one
                   source of truth for "how risk-averse is this run" rather than two
                   alphas that could silently diverge. Override explicitly if a run
                   genuinely needs the SDF built at a different alpha than its CVaR
                   edge cost.
    risk_sdf_d_max : TSDF truncation distance (m); defaults to cfg.risk_sdf.d_max."""
    e = cfg.env
    obstacle_sigma_max_eff = e.obstacle_sigma_max if obstacle_sigma_max is None else obstacle_sigma_max
    risk_sdf_alpha_eff = cfg.risk.alpha if risk_sdf_alpha is None else risk_sdf_alpha
    np.random.seed(seed)
    grid = GridMap(Map2D(e.width, e.height, e.resolution))
    gen = LunarMapGenerator(grid)
    gen.generate(num_rocks=e.num_rocks if num_rocks is None else num_rocks,
                 num_craters=num_craters,
                 rock_radius_min=e.rock_radius_min if rock_radius_min is None else rock_radius_min,
                 rock_radius_max=e.rock_radius_max if rock_radius_max is None else rock_radius_max)
    if undulation > 0:
        gen.add_terrain_undulation(num_hills=undulation, seed=seed)
    cr = (e.disc_radius + 0.5) if clear_radius is None else clear_radius
    if use_risk_sdf:
        # clear_disc only erases the RAW obstacle grid within cr of start/goal, sized
        # for the plain nominal SDF requirement (disc_radius + a 0.5m buffer). It has
        # no idea about the EXTRA kappa*s_k inflation build_risk_sdf applies on top,
        # so a nearby obstacle's inflated keep-out can still reach past the cleared
        # disc and leave start/goal permanently infeasible (path_free fails on every
        # sampled edge, tree stuck at the root). Widening cr by the WORST-CASE
        # inflation (kappa * obstacle_sigma_max -- the largest any obstacle's std can
        # be) restores the same guaranteed-safe 0.5m bubble around start/goal under
        # the risk-inflated field: for any point q within `cr` of start/goal, no
        # obstacle can be closer than (cr - dist(start, q)), so even at the worst-case
        # sigma the inflated clearance there is still >= disc_radius + 0.5 - dist.
        cr = cr + cvar_kappa(risk_sdf_alpha_eff) * obstacle_sigma_max_eff
    for p in (start, goal):
        if p is not None:
            clear_disc(grid, p, cr)
    grid.compute_slope_map()
    kw = {} if risk_weights is None else {"risk_weights": tuple(risk_weights)}
    env = Environment(grid, disc_radius=e.disc_radius, clearance=e.clearance,
                      with_risk=with_risk,
                      obstacle_sigma_min=e.obstacle_sigma_min if obstacle_sigma_min is None else obstacle_sigma_min,
                      obstacle_sigma_max=obstacle_sigma_max_eff,
                      sigma_seed=seed,
                      use_risk_sdf=use_risk_sdf,
                      risk_sdf_alpha=risk_sdf_alpha_eff,
                      risk_sdf_d_max=cfg.risk_sdf.d_max if risk_sdf_d_max is None else risk_sdf_d_max,
                      **kw)
    return grid, env
