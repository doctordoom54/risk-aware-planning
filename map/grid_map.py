import numpy as np
from .map import Map2D


class GridMap:
    def __init__(self, map2d: Map2D):
        self.map = map2d
        self.surface = np.zeros(self.map.shape(), dtype=np.float32)
        self.features = np.zeros(self.map.shape(), dtype=np.uint8)
        self.heightmap = np.zeros(self.map.shape(), dtype=np.float32)
        self.occupancy = np.zeros(self.map.shape(), dtype=np.uint8)
        # per-instance obstacle label: 0 = free space, k = the k-th obstacle placed
        # (rock/crater/wall) -- lets each obstacle carry its OWN boundary-uncertainty
        # std instead of one constant for every obstacle (see map.sdf.ObstacleSigmaField).
        self.obstacle_id = np.zeros(self.map.shape(), dtype=np.int32)
        self._slope_map = None  # cached slope angles (radians)

    # ── Terrain slope ───────────────────────────────────────────
    def compute_slope_map(self):
        """Compute terrain slope angle (radians) from surface heightmap.

        Uses central finite differences on the surface elevation grid:
            ∂z/∂x, ∂z/∂y  →  slope = arctan(√((∂z/∂x)² + (∂z/∂y)²))

        The result is cached; call again after modifying self.surface
        to refresh.
        """
        res = self.map.resolution_m
        # np.gradient returns (∂z/∂row, ∂z/∂col)
        dz_dy, dz_dx = np.gradient(self.surface, res, res)
        self._slope_map = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        return self._slope_map

    @property
    def slope_map(self):
        """Slope angle at each cell in radians.  Computed lazily."""
        if self._slope_map is None:
            self.compute_slope_map()
        return self._slope_map

    def slope_at(self, x_m, y_m):
        """Return slope angle (radians) at a world coordinate."""
        j, i = self.map.meters_to_indices(x_m, y_m)
        sm = self.slope_map
        if 0 <= j < sm.shape[0] and 0 <= i < sm.shape[1]:
            return float(sm[j, i])
        return 0.0

    def set_feature(self, x_m, y_m, value):
        j, i = self.map.meters_to_indices(x_m, y_m)
        if 0 <= j < self.map.size_y and 0 <= i < self.map.size_x:
            self.features[j, i] = value

    def get_feature(self, x_m, y_m):
        j, i = self.map.meters_to_indices(x_m, y_m)
        if 0 <= j < self.map.size_y and 0 <= i < self.map.size_x:
            return self.features[j, i]
        return None

    def world_to_grid(self, x_m, y_m):
        return self.map.meters_to_indices(x_m, y_m)

    def grid_to_world(self, j, i):
        return self.map.indices_to_meters(j, i)

    def can_place(self, rr, cc):
        return np.all(
            (rr >= 0) & (rr < self.map.size_y) &
            (cc >= 0) & (cc < self.map.size_x) &
            (self.occupancy[rr, cc] == 0)
        )
