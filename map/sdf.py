import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    # ── Efficient O(n) Euclidean Distance Transform ──────────────
    # Felzenszwalb & Huttenlocher (2012) algorithm — separable 1D
    # parabola envelope approach.  Memory: O(max(H,W)) auxiliary.

    def _edt_1d(f, n):
        """1D squared-distance transform of f[0..n-1].

        Uses lower-envelope of parabolas.  Returns d[i] = min_j (f[j] + (i-j)^2).
        """
        d = np.empty(n, dtype=np.float64)
        v = np.empty(n, dtype=np.int32)   # parabola locations
        z = np.empty(n + 1, dtype=np.float64)  # intersection points
        k = 0
        v[0] = 0
        z[0] = -1e20
        z[1] = 1e20

        for q in range(1, n):
            fq = f[q]
            while True:
                vk = v[k]
                s = ((fq + q * q) - (f[vk] + vk * vk)) / (2.0 * q - 2.0 * vk)
                if s > z[k]:
                    break
                k -= 1
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = 1e20

        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            vk = v[k]
            d[q] = (q - vk) * (q - vk) + f[vk]
        return d

    def distance_transform_edt(binary_input):
        """Euclidean distance transform — O(H*W) time and memory.

        For each True cell, returns the Euclidean distance to the nearest
        False cell.  Uses separable 1D transforms along rows then columns.
        """
        arr = np.asarray(binary_input, dtype=bool)
        H, W = arr.shape

        INF = float(H * H + W * W + 1)
        # Initialize: 0 for True cells, INF for False cells
        # (we want distance from True to nearest False)
        grid = np.where(arr, INF, 0.0)

        # Transform along rows (axis=1)
        tmp = np.empty_like(grid)
        for i in range(H):
            tmp[i, :] = _edt_1d(grid[i, :], W)

        # Transform along columns (axis=0)
        result = np.empty_like(grid)
        col_buf = np.empty(H, dtype=np.float64)
        for j in range(W):
            col_buf[:] = tmp[:, j]
            result[:, j] = _edt_1d(col_buf, H)

        return np.sqrt(result).astype(np.float32)

    def binary_dilation(arr, structure=None, iterations=1):
        arr = np.asarray(arr, dtype=bool)
        if structure is None:
            structure = np.ones((3, 3), dtype=bool)
        result = arr.copy()
        sh, sw = structure.shape
        ph, pw = sh // 2, sw // 2
        for _ in range(iterations):
            padded = np.pad(result, max(ph, pw), mode='constant',
                            constant_values=False)
            new = np.zeros_like(result)
            for di in range(sh):
                for dj in range(sw):
                    if structure[di, dj]:
                        shifted = padded[di:di + result.shape[0],
                                         dj:dj + result.shape[1]]
                        new = np.logical_or(new, shifted)
            result = new
        return result


class SignedDistanceField:
    @staticmethod
    def crater_dilated_mask(features, buffer_radius=5):
        craters = (features == 2)
        structure = np.ones((buffer_radius, buffer_radius))
        return binary_dilation(craters, structure=structure)

    @staticmethod
    def compute(features, crater_buffer=5):
        is_obstacle = (features > 0).astype(np.uint8)
        if crater_buffer > 0:
            crater_mask = SignedDistanceField.crater_dilated_mask(
                features, buffer_radius=crater_buffer)
            is_obstacle[crater_mask] = 1
        sdf_pos = distance_transform_edt(1 - is_obstacle)
        sdf_neg = distance_transform_edt(is_obstacle)
        sdf = sdf_pos - sdf_neg
        # Clamp interior to zero — obstacles read as 0, not negative
        sdf[sdf < 0] = 0.0
        return sdf


def _nearest_obstacle_indices(obstacle_id):
    """Pure-numpy multi-source BFS fallback (scipy unavailable): for every cell,
    the (row, col) of the nearest cell with obstacle_id > 0. 4-connected, so only
    approximately Euclidean -- fine for assigning a per-obstacle scalar, unlike the
    SDF itself which needs the exact distance transform."""
    from collections import deque
    H, W = obstacle_id.shape
    src_r, src_c = np.nonzero(obstacle_id)
    ind_r = np.full((H, W), -1, dtype=np.int64)
    ind_c = np.full((H, W), -1, dtype=np.int64)
    ind_r[src_r, src_c] = src_r
    ind_c[src_r, src_c] = src_c
    visited = obstacle_id > 0
    q = deque(zip(src_r.tolist(), src_c.tolist()))
    while q:
        r, c = q.popleft()
        nr0, nc0 = ind_r[r, c], ind_c[r, c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                visited[nr, nc] = True
                ind_r[nr, nc] = nr0
                ind_c[nr, nc] = nc0
                q.append((nr, nc))
    return ind_r, ind_c


class ObstacleSigmaField:
    """Per-obstacle boundary-uncertainty std, broadcast spatially.

    Each obstacle INSTANCE (grid.obstacle_id, one label per rock/crater/wall) draws
    its own std ~ uniform(sigma_min, sigma_max); every cell then takes the std of
    whichever obstacle instance is nearest to it. This replaces one constant sigma
    for the whole map with a field that varies obstacle-to-obstacle, while staying a
    single scalar lookup wherever a "the local obstacle sigma" value is needed
    (TerrainRiskMap's Gaussian halo, Environment.obstacle_sigma_at, and the (K,)
    per-instance array map.risk_sdf.build_risk_sdf takes as its s_k argument)."""

    @staticmethod
    def sample_per_obstacle(obstacle_id, sigma_min, sigma_max, rng=None):
        """Draw ONE std per obstacle INSTANCE, ids 1..K where K = obstacle_id.max().
        Returns a (K,) array: s_k[i] is obstacle id (i+1)'s std. Split out of
        .compute() so a caller that needs the raw per-instance array directly (e.g.
        Environment building map.risk_sdf's s_k argument) can share the EXACT same
        draw used for the broadcast per-cell field below, instead of re-sampling
        independently and silently diverging from it."""
        rng = np.random.default_rng() if rng is None else rng
        obstacle_id = np.asarray(obstacle_id, dtype=np.int32)
        num_obs = int(obstacle_id.max())
        if num_obs == 0:
            return np.zeros(0, dtype=np.float64)
        return rng.uniform(sigma_min, sigma_max, size=num_obs)

    @staticmethod
    def broadcast(obstacle_id, s_k, free_fill):
        """Per-cell field from a (K,) per-instance array (s_k[i] = obstacle id
        (i+1)'s std, from sample_per_obstacle): every cell takes the std of
        whichever obstacle instance is nearest to it. free_fill is a placeholder for
        id 0 (free space) -- never actually selected, since every cell's NEAREST
        obstacle id is >= 1 by construction of the distance transform below."""
        obstacle_id = np.asarray(obstacle_id, dtype=np.int32)
        s_k = np.asarray(s_k, dtype=np.float64)
        num_obs = s_k.shape[0]
        if num_obs == 0:
            return np.full(obstacle_id.shape, free_fill, dtype=np.float64)

        per_obs_sigma = np.empty(num_obs + 1, dtype=np.float64)
        per_obs_sigma[0] = free_fill
        per_obs_sigma[1:] = s_k

        if _HAVE_SCIPY:
            _, inds = distance_transform_edt(obstacle_id == 0, return_indices=True)
            ind_r, ind_c = inds[0], inds[1]
        else:
            ind_r, ind_c = _nearest_obstacle_indices(obstacle_id)
        nearest_ids = obstacle_id[ind_r, ind_c]
        return per_obs_sigma[nearest_ids]

    @staticmethod
    def compute(obstacle_id, sigma_min, sigma_max, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        s_k = ObstacleSigmaField.sample_per_obstacle(obstacle_id, sigma_min, sigma_max, rng=rng)
        return ObstacleSigmaField.broadcast(obstacle_id, s_k, sigma_max)
