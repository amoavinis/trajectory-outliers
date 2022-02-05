from sklearn.preprocessing import MinMaxScaler

class GridCreator:
    def __init__(self, grid_scale):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.grid_scale = grid_scale

    def take_points(self, trips):
        all_points = []
        for t in trips:
            for p in t:
                all_points.append([p.latitude, p.longitude])
        return all_points

    def fit(self, trips):
        self.scaler.fit(self.take_points(trips))

    def fit_transform(self, trips):
        self.scaler.fit(self.take_points(trips))
        all_trajectories_transformed = self.transform(trips)
        return all_trajectories_transformed

    def transform(self, trips):
        all_trajectories_transformed = []
        for t in trips:
            t1 = self.scaler.transform([[p.latitude, p.longitude] for p in t]).tolist()
            all_trajectories_transformed.append(self.coords_to_grid(t1))
        return all_trajectories_transformed

    def coords_to_grid(self, coords):
        grid_coords = [
            str(int(coords[0] * self.grid_scale)),
            str(int(coords[1] * self.grid_scale))
        ]
        return '-'.join(grid_coords)