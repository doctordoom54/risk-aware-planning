import numpy as np
from .grid_map import GridMap

try:
    from skimage.draw import disk
except ImportError:
    def disk(center, radius, shape=None):
        r0, c0 = int(round(center[0])), int(round(center[1]))
        radius = int(round(radius))
        rr, cc = [], []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    r, c = r0 + dr, c0 + dc
                    if shape is not None:
                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            rr.append(r)
                            cc.append(c)
                    else:
                        rr.append(r)
                        cc.append(c)
        return np.array(rr, dtype=int), np.array(cc, dtype=int)

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(arr, sigma=1.5):
        result = arr.copy()
        k = max(1, int(round(sigma * 2)))
        for _ in range(3):
            padded = np.pad(result, k, mode='edge')
            out = np.zeros_like(result)
            count = 0
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    out += padded[k + di:k + di + result.shape[0],
                                  k + dj:k + dj + result.shape[1]]
                    count += 1
            result = out / count
        return result


class LunarMapGenerator:
    def __init__(self, grid: GridMap):
        self.grid = grid

    def generate(self, num_rocks=120, num_craters=60, rock_radius_min=0.25, rock_radius_max=0.8):
        # crater radius scales with the map so it fits small arenas too (the old fixed
        # 2-5 m radius overflowed a 6 m map -> randint(low>=high) crash)
        max_r_m = 0.30 * min(self.grid.map.width_m, self.grid.map.height_m)
        next_id = int(self.grid.obstacle_id.max()) + 1   # continue numbering past any
                                                           # obstacles already on the grid
        for _ in range(num_craters):
            for _try in range(100):
                r_m = np.random.uniform(min(2.0, 0.3 * max_r_m), max(0.5, max_r_m))
                r = int(np.round(r_m / self.grid.map.resolution_m))
                if self.grid.map.size_x - r <= r or self.grid.map.size_y - r <= r:
                    continue                       # crater too big for this map; retry
                x = np.random.randint(r, self.grid.map.size_x - r)
                y = np.random.randint(r, self.grid.map.size_y - r)
                rr, cc = disk((y, x), r)
                if self.grid.can_place(rr, cc):
                    depth = np.random.uniform(0.2, 0.5)
                    self.grid.surface[rr, cc] -= depth
                    self.grid.features[rr, cc] = 2
                    self.grid.occupancy[rr, cc] = 1
                    self.grid.obstacle_id[rr, cc] = next_id
                    next_id += 1
                    break

        for _ in range(num_rocks):
            for _try in range(100):
                r_m = np.random.uniform(rock_radius_min, rock_radius_max)
                r = int(np.round(r_m / self.grid.map.resolution_m))
                if self.grid.map.size_x - r <= r or self.grid.map.size_y - r <= r:
                    continue                       # rock too big for this map; retry
                x = np.random.randint(r, self.grid.map.size_x - r)
                y = np.random.randint(r, self.grid.map.size_y - r)
                rr, cc = disk((y, x), r)
                if self.grid.can_place(rr, cc):
                    height = np.random.uniform(0.01, 0.2)
                    self.grid.surface[rr, cc] += height
                    self.grid.features[rr, cc] = 1
                    self.grid.occupancy[rr, cc] = 1
                    self.grid.obstacle_id[rr, cc] = next_id
                    next_id += 1
                    break

        self.grid.surface = gaussian_filter(self.grid.surface, sigma=1.5)

    def add_terrain_undulation(self, num_hills=3, height_range=(0.3, 1.0),
                                wavelength_range=(4.0, 10.0), seed=None):
        """Add smooth sinusoidal hills and valleys to the surface.

        Creates realistic terrain undulation with slopes that are
        significant enough to affect rover mobility (10-30° locally).

        Args:
            num_hills: number of Gaussian hill/valley features.
            height_range: (min, max) peak height in meters.
            wavelength_range: (min, max) spatial extent in meters.
            seed: optional RNG seed for reproducibility.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        res = self.grid.map.resolution_m
        H, W = self.grid.surface.shape

        for _ in range(num_hills):
            # Random center
            cx = rng.uniform(0.2 * W, 0.8 * W)
            cy = rng.uniform(0.2 * H, 0.8 * H)
            # Random height (positive = hill, negative = depression)
            h = rng.uniform(*height_range)
            if rng.rand() < 0.3:
                h = -h  # 30% chance of valley
            # Random width in pixels
            wl = rng.uniform(*wavelength_range) / res
            sigma_x = wl / 2.0
            sigma_y = wl / (1.5 + rng.rand())  # slight asymmetry

            yy, xx = np.mgrid[0:H, 0:W]
            gauss = h * np.exp(-((xx - cx)**2 / (2 * sigma_x**2) +
                                  (yy - cy)**2 / (2 * sigma_y**2)))
            self.grid.surface += gauss.astype(np.float32)

        # Smooth to avoid sharp edges
        self.grid.surface = gaussian_filter(self.grid.surface, sigma=2.0)

    def add_wall(self, x_m, y_m, length_m, thickness_m, angle_deg=0):
        """Place a rectangular wall obstacle to create narrow corridors.

        Args:
            x_m, y_m: center of the wall in meters
            length_m: wall length
            thickness_m: wall thickness
            angle_deg: rotation angle (0 = horizontal, 90 = vertical)
        """
        res = self.grid.map.resolution_m
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        half_l = length_m / 2
        half_t = thickness_m / 2

        next_id = int(self.grid.obstacle_id.max()) + 1    # wall is one obstacle instance
        H, W = self.grid.surface.shape
        for j in range(H):
            for i in range(W):
                wx, wy = self.grid.map.indices_to_meters(j, i)
                # Transform to wall-local coords
                dx = wx - x_m
                dy = wy - y_m
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                if abs(lx) <= half_l and abs(ly) <= half_t:
                    self.grid.features[j, i] = 1
                    self.grid.occupancy[j, i] = 1
                    self.grid.obstacle_id[j, i] = next_id
