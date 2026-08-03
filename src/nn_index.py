"""
Approximate nearest-neighbour index for the kinodynamic RRT*.

The true edge metric is the control-optimal cost, which is asymmetric and not a
valid distance, so it can't be indexed directly. Standard practice: retrieve a
candidate set with a cheap Euclidean metric on a weighted state
[x, y, vel_weight*vx, vel_weight*vy], then evaluate the exact control-optimal cost
only on those candidates. This module wraps that retrieval with three backends,
picked by availability:

    hnswlib   -> incremental approximate NN graph (fastest at scale)
    scipy     -> cKDTree, periodically rebuilt + brute tail for new points
    (none)    -> vectorised numpy brute force

All backends expose add(state) and knn(state, k) -> candidate node indices.
"""
import numpy as np


class NeighborIndex:
    def __init__(self, dim=3, weight=None, max_elements=100000):
        self.w = np.ones(dim) if weight is None else np.asarray(weight, float)
        self.dim = dim
        self.n = 0
        self.V = np.empty((max(16, max_elements), dim))   # weighted states (all backends keep this)
        self.backend = "brute"
        self._hnsw = None
        self._kdt = None; self._kdt_n = 0; self._rebuild = 256
        self._cKDTree = None
        try:
            import hnswlib
            self._hnsw = hnswlib.Index(space="l2", dim=dim)
            self._hnsw.init_index(max_elements=max_elements, ef_construction=200, M=16)
            self._hnsw.set_ef(64)
            self.backend = "hnswlib"
        except Exception:
            try:
                from scipy.spatial import cKDTree
                self._cKDTree = cKDTree
                self.backend = "scipy"
            except Exception:
                self.backend = "brute"

    def add(self, x):
        v = np.asarray(x, float) * self.w
        if self.n >= len(self.V):
            self.V = np.vstack([self.V, np.empty_like(self.V)])
        self.V[self.n] = v
        if self.backend == "hnswlib":
            self._hnsw.add_items(v[None, :], np.array([self.n]))
        self.n += 1

    def knn(self, x, k):
        n = self.n
        k = min(k, n)
        if k <= 0:
            return np.empty(0, dtype=int)
        q = np.asarray(x, float) * self.w
        if self.backend == "hnswlib":
            labels, _ = self._hnsw.knn_query(q[None, :], k=k)
            return labels[0].astype(int)
        if self.backend == "scipy":
            if self._kdt is None or (n - self._kdt_n) > self._rebuild:
                self._kdt = self._cKDTree(self.V[:n]); self._kdt_n = n
            idx = np.atleast_1d(self._kdt.query(q, k=min(k, self._kdt_n))[1])
            cand = list(idx)
            if self._kdt_n < n:                       # brute tail since last rebuild
                cand.extend(range(self._kdt_n, n))
            return np.array(cand, dtype=int)
        # brute force (vectorised)
        d = ((self.V[:n] - q) ** 2).sum(axis=1)
        if k >= n:
            return np.arange(n)
        return np.argpartition(d, k - 1)[:k]