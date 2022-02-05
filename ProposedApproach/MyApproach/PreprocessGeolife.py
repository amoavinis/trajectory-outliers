import os
import geopy.distance
from sklearn.preprocessing import MinMaxScaler
import datetime
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
import pickle

class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_trajectories = []
        self.scaler = MinMaxScaler()
        self.representation_pca = None

    def process_file(self, f):
        file = open(f, 'r')
        lines = file.readlines()[6:]

        result = []

        for line in lines:
            split_line = line.split(",")
            latitude = float(split_line[0])
            longitude = float(split_line[1])
            timestamp = ' '.join(split_line[5:]).strip()
            timestamp = datetime.datetime.strptime(
                timestamp, '%Y-%m-%d %H:%M:%S').timestamp()
            result.append([latitude, longitude, timestamp])

        return result

    def create_trajectories(self):
        for i in tqdm(os.listdir(self.data_path)[:4]):
            for j in os.listdir(self.data_path+i+'/Trajectory/'):
                self.all_trajectories.append(
                    self.process_file(self.data_path+i+'/Trajectory/'+j))

    def dist_of_transition(self, transition):
        return geopy.distance.distance(transition[0], transition[1]).meters

    def distance_of_trajectory(self, traj):
        dist = 0.0
        for i in range(len(traj) - 1):
            dist += self.dist_of_transition([traj[i][:2], traj[i+1][:2]])
        return dist

    def speeds_in_traj(self, traj):
        speeds = []
        for i in range(len(traj)-1):
            dx = self.dist_of_transition([traj[i][:2], traj[i+1][:2]])
            dt = traj[i+1][2] - traj[i][2]
            if dt > 0:
                speeds.append(dx/dt)
        return {
            'min_speed': min(speeds),
            'max_speed': max(speeds),
            'avg_speed': sum(speeds)/len(speeds)
        }

    def slant(self, traj):
        if traj[-1][0]-traj[0][0] == 0:
            return 0
        else:
            return (traj[-1][1]-traj[0][1])/(traj[-1][0]-traj[0][0])

    def get_features(self):
        trajs = []
        for traj in tqdm(self.all_trajectories):
            speeds = self.speeds_in_traj(traj)
            t = [
                traj[0][0],
                traj[0][1],
                traj[-1][0],
                traj[-1][1],
                self.slant(traj),
                self.distance_of_trajectory(traj),
                speeds['min_speed'],
                speeds['max_speed'],
                speeds['avg_speed']
            ]
            trajs.append(t)
        self.all_trajectories = trajs

    def convert_to_dataframe(self):
        df = DataFrame(self.all_trajectories, columns=[
                       'starting_x', 'starting_y', 'ending_x', 'ending_y', 'slant', 'distance', 'min_speed', 'max_speed', 'avg_speed'])
        return df

    def trajectories_to_pickle(self):
        pickle.dump(list(self.all_trajectories), open(os.getcwd()+'/trajectories_features.pkl', 'wb'))

    def preprocess(self):
        print("Creating trajectories...")
        self.create_trajectories()
        print("Trajectories created.")
        print("Extracting features...")
        self.get_features()
        print("Features extracted.")
        self.trajectories_to_pickle()
        print("Trajectories output to trajectories_features.pkl")

DATA_PREFIX = "Geolife Trajectories 1.3/Data/"
p = Preprocessor(os.getcwd()+"/"+DATA_PREFIX)
p.preprocess()
