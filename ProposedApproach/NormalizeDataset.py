import argparse
from datetime import datetime
from CustomScaler import Scaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm
import os
from collections import Counter
from matplotlib import pyplot as plt

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
        if self.dataset == "geolife":
            lines = file.readlines()[6:]

            result = []

            for line in lines:
                split_line = line.split(",")
                latitude = float(split_line[0])
                longitude = float(split_line[1])
                date = split_line[5]
                time = split_line[6]
                dt = date + " " + time
                timestamp = datetime.strptime(dt.strip(), "%Y-%m-%d %H:%M:%S").timestamp()
                result.append([longitude, latitude, timestamp])

            return result
        else:
            lines = file.readlines()

            result = []

            for line in lines:
                split_line = line.split(",")[2:]
                latitude = float(split_line[1])
                longitude = float(split_line[0])
                result.append([longitude, latitude])

            return result

    def create_trajectories(self):
        for i in tqdm(os.listdir(self.data_path)):
            if self.dataset == "geolife":
                i_path = self.data_path+i+"/Trajectory/"
                for j in os.listdir(i_path):
                    trajectory = self.process_file(i_path+j)
                    valid = True
                    for p in trajectory:
                        #if p[0] < 115.5 or p[0] > 117.5 or p[1] < 39.4 or p[1] > 40.5:
                        if p[0] < 115.42 or p[0] > 117.5 or p[1] < 39.4 or p[1] > 41.1:
                            valid = False
                            break
                    if valid and len(trajectory) > 0:
                        self.all_trajectories.append(trajectory)
            else:
                trajectory = self.process_file(self.data_path+i)
                valid = True
                for p in trajectory:
                    if p[0] < 114.5 or p[0] > 119.5 or p[1] < 38 or p[1] > 43:
                        valid = False
                        break
                if valid and len(trajectory) > 0:
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

    def remove_duplicates(self, x):
        y = [x[0]]
        for i in range(1, len(x)):
            if self.dataset == "geolife" and y[-1][0] != x[i][0] or self.dataset != "geolife" and y[-1] != x[i]:
                y.append(x[i])
        return y

    def transform_trajectory_to_grid(self, traj, grid_scale):
        trajectory_transformed = []
        for p in traj:
            to_append = None
            if self.dataset == "geolife":
                to_append = (self.coords_to_grid(self.scaler.transform([p[:2]])[0], grid_scale), p[2])
            else:
                to_append = self.coords_to_grid(self.scaler.transform([p[:2]])[0], grid_scale)
            trajectory_transformed.append(to_append)
        return self.remove_duplicates(trajectory_transformed)

    def transform_all_trajectories(self):
        for t in self.all_trajectories:
            self.trajectories_with_grid.append([
                t, self.transform_trajectory_to_grid(t, self.grid_scale)
            ])

    def trajectories_to_pickle(self):
        data = self.trajectories_with_grid
        pickle.dump(data, open(os.getcwd()+"/trajectories_with_grid_"+self.dataset+".pkl", 'wb'))

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

        plot = False
        # Uncomment below line to plot the GPS points  
        # plot = True
        
        if plot:
            all_points = self.take_points()
            x = [p[0] for p in all_points]
            y = [p[1] for p in all_points]
            plt.scatter(x, y)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()
        
        print("Fitting point scaler...")
        self.fit_point_scaler()
        print("Fitted point scaler.")
        print("Transforming all trajectories...")
        self.transform_all_trajectories()
        print("Transformed all trajectories.")
        self.trajectories_to_pickle()
        print("Trajectories output to trajectories_with_grid_"+self.dataset+".pkl")

    def trajectory_statistics(self):
        simple_lengths = [len(t[0]) for t in self.trajectories_with_grid]
        grid_lengths = [len(t[1]) for t in self.trajectories_with_grid]

        print("Average simple path length:", sum(simple_lengths)/len(simple_lengths))
        print("Average grid path length:", sum(grid_lengths)/len(grid_lengths))
        print("Lengths of grid paths:", Counter(grid_lengths))

data_prefixes = {
    "geolife": "Datasets/Geolife Trajectories 1.3/Data/",
    "tdrive": "Datasets/T-Drive/taxi_log_2008_by_id/"
}

parser = argparse.ArgumentParser(description="Filter out trajectories and produce the grid representation of the dataset.")
parser.add_argument("--dataset",
                    help="Specify the dataset to use",
                    default="geolife")
parser.add_argument("--gridScale", help="The number of grid cells per dimension", default="20")
args = parser.parse_args()

grid_scale = int(args.gridScale)
dataset = args.dataset

nm = Normalizer(dataset, os.getcwd()+"/"+data_prefixes[dataset], grid_scale)
nm.preprocess()
nm.trajectory_statistics()
