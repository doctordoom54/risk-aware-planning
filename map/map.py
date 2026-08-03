import numpy as np


class Map2D:
    def __init__(self, width_m, height_m, resolution_m,
                 origin_x=0.0, origin_y=0.0):
        self.width_m = width_m
        self.height_m = height_m
        self.resolution_m = resolution_m
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.size_x = int(np.ceil(width_m / resolution_m))
        self.size_y = int(np.ceil(height_m / resolution_m))

    def meters_to_indices(self, x_m, y_m):
        i = int(np.floor((x_m - self.origin_x) / self.resolution_m))
        j = int(np.floor((y_m - self.origin_y) / self.resolution_m))
        return j, i

    def indices_to_meters(self, j, i):
        x_m = self.origin_x + i * self.resolution_m
        y_m = self.origin_y + j * self.resolution_m
        return x_m, y_m

    def shape(self):
        return (self.size_y, self.size_x)

    def is_within_bounds(self, x_m, y_m):
        return (self.origin_x <= x_m < self.origin_x + self.width_m and
                self.origin_y <= y_m < self.origin_y + self.height_m)
