from CustomScaler import Scaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm
import os
from collections import Counter

class Normalizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_trajectories = []
        self.scaler = Scaler()
        self.traj_scaler = MinMaxScaler()
        self.grid_scale = 100
        self.trajectories_with_grid = []

    def process_file(self, f):
        file = open(f, 'r')
        lines = file.readlines()[6:]

        result = []

        for line in lines:
            split_line = line.split(",")
            latitude = float(split_line[0])
            if latitude > 100:
                latitude -= 360
            longitude = float(split_line[1])
            result.append([longitude, latitude])

        return result
    
    def create_trajectories(self):
        for i in tqdm(os.listdir(self.data_path)):
            for j in os.listdir(self.data_path+i+'/Trajectory/'):
                self.all_trajectories.append(
                    self.process_file(self.data_path+i+'/Trajectory/'+j))

    def take_points(self):
        all_points = []
        for t in self.all_trajectories:
            for p in t:
                all_points.append(p[:2])
        return all_points

    def coords_to_grid(self, coords, grid_scale):
        grid_coords = [str(int(coords[0]*grid_scale)), str(int(coords[1]*grid_scale))]
        return '-'.join(grid_coords)

    def fit_point_scaler(self):
        self.scaler.fit(self.take_points())

    def remove_duplicates(self, x):
        y = [x[0]]
        for i in range(1, len(x)):
            if y[-1] != x[i]:
                y.append(x[i])
        return y

    def transform_trajectory_to_grid(self, traj, grid_scale):
        trajectory_transformed = []
        for p in traj:
            trajectory_transformed.append(self.coords_to_grid(self.scaler.transform([p])[0], grid_scale))
        return self.remove_duplicates(trajectory_transformed)

    def transform_all_trajectories(self):
        for t in self.all_trajectories:
            self.trajectories_with_grid.append([
                t, self.transform_trajectory_to_grid(t, self.grid_scale)
            ])

    def trajectories_to_pickle(self):
        data = self.trajectories_with_grid
        pickle.dump(data, open(os.getcwd()+'/trajectories_with_grid.pkl', 'wb'))

    def preprocess(self):
        if os.path.exists("trajectories_raw.pkl"):
            print("Reading trajectories from disk...")
            self.all_trajectories = pickle.load(open("trajectories_raw.pkl", "rb"))
            print("Read trajectories from disk.")
        else:
            print("Creating trajectories...")
            self.create_trajectories()
            pickle.dump(self.all_trajectories, open("trajectories_raw.pkl", "wb"))
            print("Trajectories created.")
        print("Fitting point scaler...")
        self.fit_point_scaler()
        print("Fitted point scaler.")
        print("Transforming all trajectories...")
        self.transform_all_trajectories()
        print("Transformed all trajectories.")
        self.trajectories_to_pickle()
        print("Trajectories output to trajectories_with_grid.pkl")
    
    def trajectory_statistics(self):
        simple_lengths = [len(t[0]) for t in self.trajectories_with_grid]
        grid_lengths = [len(t[1]) for t in self.trajectories_with_grid]
        print("Lengths of simple paths:", Counter(simple_lengths))
        print("Lengths of grid paths:", Counter(grid_lengths))


DATA_PREFIX = "Datasets/Geolife Trajectories 1.3/Data/"
nm = Normalizer(os.getcwd()+"/"+DATA_PREFIX)
nm.preprocess()
nm.trajectory_statistics()