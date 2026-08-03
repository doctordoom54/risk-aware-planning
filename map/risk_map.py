"""
Terrain-aware risk map.

Fuses three terrain-derived hazard layers into a single per-cell failure
probability in [0, 1] for risk-aware planning on lunar/planetary terrain:

    1. Slope risk        - tip-over / slip on steep grades (from the slope map).
    2. Roughness risk    - rocks and undulation (local elevation variability).
    3. Obstacle risk     - collision with uncertain obstacle boundaries
                           (Gaussian halo around the signed distance field).

The layers are combined as independent failure modes (probabilistic OR):
        p_fail = 1 - (1 - r_slope)(1 - r_rough)(1 - r_obstacle)
so the result is a bounded, monotone, interpretable risk field that drops into
the same role as a collision-probability map for the MCTS / SCP planners.
"""
import numpy as np

from .sdf import SignedDistanceField, ObstacleSigmaField


# --------------------------------------------------------------------------- #
# local-window mean (for roughness); scipy if available, else O(HW) cumsum box
# --------------------------------------------------------------------------- #
try:
    from scipy.ndimage import uniform_filter

    def _box_mean(a, radius):
        return uniform_filter(a, size=2 * radius + 1, mode="nearest")
except ImportError:
    def _box_mean(a, radius):
        r = int(radius)
        if r < 1:
            return a.astype(np.float64)
        H, W = a.shape
        ap = np.pad(a.astype(np.float64), r, mode="edge")
        S = np.zeros((H + 2 * r + 1, W + 2 * r + 1))
        S[1:, 1:] = np.cumsum(np.cumsum(ap, axis=0), axis=1)
        win = 2 * r + 1
        total = (S[win:, win:] - S[:-win, win:] - S[win:, :-win] + S[:-win, :-win])
        return total / (win * win)


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class TerrainRiskMap:
    def __init__(self, grid_map, sdf=None,
                 slope_critical_deg=20.0, slope_width_deg=5.0,
                 roughness_radius_m=0.5, roughness_scale_m=0.08,
                 obstacle_sigma_map=None, obstacle_sigma_min=0.1, obstacle_sigma_max=0.6,
                 sigma_rng=None,
                 weights=(1.0, 1.0, 1.0)):
        """
        grid_map           : maps.GridMap (provides surface, features, slope_map, map geometry)
        sdf                : optional precomputed signed distance field (pixels);
                             computed from grid.features if None
        slope_critical_deg : slope at which slope-risk = 0.5 (rover mobility limit)
        slope_width_deg    : softness of the slope-risk ramp
        roughness_radius_m : window radius for local elevation roughness
        roughness_scale_m  : roughness (std, m) that maps to risk ~1
        obstacle_sigma_map : optional precomputed per-cell boundary-uncertainty std (m),
                             e.g. Environment's shared field -- avoids resampling per
                             obstacle sigma twice. Computed from grid_map.obstacle_id via
                             ObstacleSigmaField if None (each obstacle instance gets its
                             own std ~ uniform(obstacle_sigma_min, obstacle_sigma_max)).
        obstacle_sigma_min,
        obstacle_sigma_max : sampling range used when obstacle_sigma_map is None
        sigma_rng           : optional np.random.Generator for the sigma sampling
        weights            : (slope, roughness, obstacle) layer weights in [0,1]
        """
        self.grid = grid_map
        self.res = grid_map.map.resolution_m
        self.slope_crit = np.radians(slope_critical_deg)
        self.slope_width = np.radians(max(slope_width_deg, 1e-3))
        self.rough_radius_px = max(1, int(round(roughness_radius_m / self.res)))
        self.rough_scale = roughness_scale_m
        if obstacle_sigma_map is not None:
            self.obs_sigma = np.asarray(obstacle_sigma_map, dtype=np.float64)
        else:
            obstacle_id = getattr(grid_map, "obstacle_id",
                                  (grid_map.features > 0).astype(np.int32))
            self.obs_sigma = ObstacleSigmaField.compute(
                obstacle_id, obstacle_sigma_min, obstacle_sigma_max, rng=sigma_rng)
        self.w_slope, self.w_rough, self.w_obs = weights

        self.sdf = sdf if sdf is not None else SignedDistanceField.compute(grid_map.features)

        self.slope_risk = None
        self.roughness_risk = None
        self.obstacle_risk = None
        self.risk = None
        self.compute()

    # ------------------------------------------------------------------ #
    def compute(self):
        # 1. slope risk: logistic ramp past the mobility-limit slope
        slope = self.grid.slope_map                         # radians (H, W)
        self.slope_risk = self.w_slope * _logistic((slope - self.slope_crit) / self.slope_width)

        # 2. roughness risk: local std of surface elevation
        z = self.grid.surface.astype(np.float64)
        mean = _box_mean(z, self.rough_radius_px)
        meansq = _box_mean(z * z, self.rough_radius_px)
        std = np.sqrt(np.clip(meansq - mean * mean, 0.0, None))
        self.roughness_risk = self.w_rough * np.clip(std / self.rough_scale, 0.0, 1.0)

        # 3. obstacle-proximity risk: Gaussian halo OUTSIDE the boundary, saturating
        # to the layer max INSIDE it. d_m is the TRUE signed distance (negative
        # inside obstacles) -- squaring it in a bare Gaussian would make the risk
        # symmetric around the boundary, i.e. decay back toward 0 again deep inside
        # solid rock (as if the interior were "far away and safe" the same way open
        # space far outside is). Clamping to w_obs for d_m<=0 fixes that: the
        # boundary crossing stays C1 (both value AND slope equal w_obs*1 / 0 from
        # either side, since d_m=0 is exactly the Gaussian's zero-slope peak), and
        # the entire interior is (correctly) certain-collision risk, not a ring.
        d_m = self.sdf.astype(np.float64) * self.res        # SDF is in pixels
        self.obstacle_risk = self.w_obs * np.where(
            d_m <= 0, 1.0, np.exp(-0.5 * (d_m / self.obs_sigma) ** 2))

        # combine as independent failure modes (probabilistic OR)
        keep = (1.0 - np.clip(self.slope_risk, 0, 1)) * \
               (1.0 - np.clip(self.roughness_risk, 0, 1)) * \
               (1.0 - np.clip(self.obstacle_risk, 0, 1))
        self.risk = np.clip(1.0 - keep, 0.0, 1.0).astype(np.float32)
        return self.risk

    # ------------------------------------------------------------------ #
    def risk_at(self, x_m, y_m, bilinear=True):
        """Risk at a world coordinate, with optional bilinear interpolation."""
        gx = (x_m - self.grid.map.origin_x) / self.res
        gy = (y_m - self.grid.map.origin_y) / self.res
        H, W = self.risk.shape
        if not bilinear:
            j, i = self.grid.map.meters_to_indices(x_m, y_m)
            if 0 <= j < H and 0 <= i < W:
                return float(self.risk[j, i])
            return 1.0
        gx = min(max(gx, 0.0), W - 1.0)
        gy = min(max(gy, 0.0), H - 1.0)
        x0 = int(np.floor(gx)); y0 = int(np.floor(gy))
        x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1)
        wx = gx - x0; wy = gy - y0
        r = self.risk
        v0 = r[y0, x0] * (1 - wx) + r[y0, x1] * wx
        v1 = r[y1, x0] * (1 - wx) + r[y1, x1] * wx
        return float(v0 * (1 - wy) + v1 * wy)

    def layers(self):
        return {
            "slope_risk": self.slope_risk,
            "roughness_risk": self.roughness_risk,
            "obstacle_risk": self.obstacle_risk,
            "total_risk": self.risk,
        }
