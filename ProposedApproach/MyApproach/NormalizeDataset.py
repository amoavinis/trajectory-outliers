from CustomScaler import Scaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm
import os
from collections import Counter
from matplotlib import pyplot as plt
import sys

class Normalizer:
    def __init__(self, dataset, data_path, grid_scale):
        self.data_path = data_path
        self.dataset = dataset
        self.all_trajectories = []
        self.scaler = Scaler()
        self.traj_scaler = MinMaxScaler()
        self.grid_scale = grid_scale
        self.trajectories_with_grid = []

    def process_file(self, f):
        file = open(f, 'r')
        lines = file.readlines()[6:]

        result = []

        for line in lines:
            split_line = line.split(",")
            latitude = float(split_line[0])
            longitude = float(split_line[1])
            result.append([longitude, latitude])

        return result

    def create_trajectories(self):
        for i in tqdm(os.listdir(self.data_path)):
            i_path = self.data_path+i+"/"
            if self.dataset == "geolife":
                i_path += "Trajectory/"
            for j in os.listdir(i_path):
                trajectory = self.process_file(i_path+j)
                valid = True
                for p in trajectory:
                    if p[0] < 116 or p[0] > 116.7 or p[1] < 39.7 or p[1] > 40.1:
                        valid = False
                        break
                if valid:
                    self.all_trajectories.append(trajectory)

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
        #print(self.scaler.min, self.scaler.max)

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
        if os.path.exists("trajectories_raw_"+self.dataset+".pkl"):
            print("Reading trajectories from disk...")
            self.all_trajectories = pickle.load(open("trajectories_raw_"+self.dataset+".pkl", "rb"))
            print("Read trajectories from disk.")
        else:
            print("Creating trajectories...")
            self.create_trajectories()
            pickle.dump(self.all_trajectories, open("trajectories_raw_"+self.dataset+".pkl", "wb"))
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
        #print("Lengths of simple paths:", Counter(simple_lengths))
        print("Average simple path length:", sum(simple_lengths)/len(simple_lengths))
        print("Lengths of grid paths:", Counter(grid_lengths))

    def analyze_sd_pairs(self):
        sd_pairs = dict()
        sd_pairs_list = []
        for t in self.trajectories_with_grid:
            s = t[1][0]
            d = t[1][-1]
            sd = s+"->"+d
            sd_pairs_list.append(sd)
            if sd in sd_pairs:
                sd_pairs[sd] += 1
            else:
                sd_pairs[sd] = 1
        print(sum(sd_pairs.values())/len(sd_pairs))
        vals = list(sd_pairs.values())
        #print(sd_pairs)
        with open('values', 'w') as f:
            f.write(str(vals))
        plt.hist(vals, bins=5)
        plt.savefig('hist.jpg')
        #print(Counter(sd_pairs))

dataset = 'geolife'
grid_scale = 2000
data_prefixes = {
    "geolife": "Datasets/Geolife Trajectories 1.3/Data/",
    "tdrive": "Datasets/T-Drive/taxi_log_2008_by_id/"
}
if len(sys.argv) > 2:
    grid_scale = int(sys.argv[1])
    dataset = sys.argv[2]

nm = Normalizer(dataset, os.getcwd()+"/"+data_prefixes[dataset], grid_scale)
nm.preprocess()
#nm.trajectory_statistics()
#nm.analyze_sd_pairs()
