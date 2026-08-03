import numpy as np

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


class ScienceTargets:
    @staticmethod
    def add(features, num_targets=5, target_radius_px=3):
        size_y, size_x = features.shape
        for _ in range(num_targets):
            for _try in range(100):
                x = np.random.randint(target_radius_px,
                                      size_x - target_radius_px)
                y = np.random.randint(target_radius_px,
                                      size_y - target_radius_px)
                rr, cc = disk((y, x), target_radius_px)
                valid = (rr >= 0) & (rr < size_y) & (cc >= 0) & (cc < size_x)
                rr, cc = rr[valid], cc[valid]
                if np.all(features[rr, cc] == 0):
                    features[rr, cc] = 3
                    break
        return features
