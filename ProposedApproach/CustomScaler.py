import numpy as np

class Scaler:
    def __init__(self):
        self.min = [100000, 100000]
        self.max = [-100000, -100000]

    def fit(self, X):
        for x in X:
            if x[0] <= self.min[0]:
                self.min[0] = x[0]
            elif x[0] >= self.max[0]:
                self.max[0] = x[0]

            if x[1] <= self.min[1]:
                self.min[1] = x[1]
            elif x[1] >= self.max[1]:
                self.max[1] = x[1]

    def transform(self, X):
        return [[(x[0] - self.min[0]) / (self.max[0] - self.min[0]),
                 (x[1] - self.min[1]) / (self.max[1] - self.min[1])]
                for x in X]

    def transform_trajectory(self, trajectory):
        return np.array([self.transform([p])[0] for p in trajectory])

    def coords_to_grid(self, coords, grid_scale):
        grid_coords = [int(coords[0]*grid_scale), int(coords[1]*grid_scale)]
        return grid_coords

    def grid_cell_equals(self, gc1, gc2):
        str1 = str(gc1[0]) + "," + str(gc1[1])
        str2 = str(gc2[0]) + "," + str(gc2[1])
        return str1 == str2

    def remove_repetitions(self, traj):
        previous = traj[0]
        result = [traj[0]]
        for p in traj[1:]:
            if not self.grid_cell_equals(p, previous):
                result.append(p)
            previous = p
        return result

    def trajectory_to_grid(self, trajectory, grid_scale):
        trajectory = [self.coords_to_grid(p, grid_scale) for p in trajectory]
        trajectory = self.remove_repetitions(trajectory)
        return np.array(trajectory)

