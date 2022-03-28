import os
import geopy.distance
from CustomScaler import Scaler
from sklearn.preprocessing import MinMaxScaler
import datetime
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm
import pickle
from math import sqrt
import numpy as np

class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_trajectories = []
        self.scaler = Scaler()
        self.traj_scaler = MinMaxScaler()
        self.paths = []
        self.dist_clustering = 0.5
        self.grouping_grid_scale = 20
        self.inliers = []
        self.outliers = []

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
            #timestamp = ' '.join(split_line[5:]).strip()
            #timestamp = datetime.datetime.strptime(
            #    timestamp, '%Y-%m-%d %H:%M:%S').timestamp()
            result.append([longitude, latitude])

        return result

    def create_trajectories(self):
        for i in tqdm(os.listdir(self.data_path)):
            for j in os.listdir(self.data_path+i+'/Trajectory/'):
                self.all_trajectories.append(
                    self.process_file(self.data_path+i+'/Trajectory/'+j))

    def distance_of_transition(self, transition):
        return geopy.distance.great_circle(list(reversed(transition[0])), list(reversed(transition[1]))).meters

    def distance_of_trajectory(self, traj):
        dist = 0.0
        for i in range(len(traj) - 1):
            dist += self.distance_of_transition([traj[i][:2], traj[i+1][:2]])
        return dist

    def speeds_in_traj(self, traj):
        speeds = []
        for i in range(len(traj)-1):
            dx = self.distance_of_transition([traj[i][:2], traj[i+1][:2]])
            dt = traj[i+1][2] - traj[i][2]
            if dt > 0:
                speeds.append(dx/dt)
        return {
            'min_speed': min(speeds),
            'max_speed': max(speeds),
            'avg_speed': sum(speeds)/len(speeds)
        }

    def slant(self, traj):
        sd_len = sqrt((traj[-1][1]-traj[0][1])**2 + (traj[-1][0]-traj[0][0])**2)
        dx = traj[-1][0] - traj[0][0]
        if sd_len > 0:
            sine = dx / sd_len
            return sine
        else:
            return 0

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

    def transform_trajectory_to_grid(self, traj, grid_scale):
        trajectory_transformed = []
        for p in traj:
            trajectory_transformed.append(self.coords_to_grid(self.scaler.transform([p])[0], grid_scale))
        return trajectory_transformed

    def group_by_sd_pairs(self, trajectories, threshold):
        sd_pairs = dict()
        for traj in trajectories:
            s = self.coords_to_grid(traj[0], self.grouping_grid_scale)
            d = self.coords_to_grid(traj[-1], self.grouping_grid_scale)
            sd_pair = s+'->'+d
            if sd_pair in sd_pairs:
                sd_pairs[sd_pair].append(traj)
            else:
                sd_pairs[sd_pair] = [traj]
        filtered_dict = dict()
        for sd in sd_pairs:
            if len(sd_pairs[sd]) >= threshold:
                filtered_dict[sd] = sd_pairs[sd]
            else:
                self.outliers.extend(sd_pairs[sd])

        return filtered_dict

    def intersection(self, lst1, lst2):
        return set(lst1).intersection(lst2)

    def union(self, lst1, lst2):
        return set(lst1).union(lst2)

    def custom_distance(self, x1, x2, debug=False):
        jaccard_sq = 1 - len(self.intersection(x1, x2))/len(self.union(x1, x2))
        return jaccard_sq

    def clustering_trajectories(self):
        trajectories = [self.transform_trajectory_to_grid(t, 200) for t in self.all_trajectories]
        filtered_sd = self.group_by_sd_pairs(trajectories, 2)
        print(len(trajectories))
        #print(len(list(filtered_sd.values())))
        print(len(self.outliers))
        for k in filtered_sd:
            to_cluster = filtered_sd[k]
            #total_dist = 0.0
            #for x in to_cluster[:100]:
            #    for y in to_cluster[:100]:
            #        total_dist += self.custom_distance(x, y)
            #print(total_dist/(len(to_cluster[:100])**2))
            #print(len(to_cluster))
            print(to_cluster[0])
            linked = linkage(to_cluster, method='complete', metric=self.custom_distance)
            clusters = fcluster(linked, t=self.dist_clustering, criterion='distance')

            clusters_grouped = dict()
            for i in range(len(clusters)):
                if clusters[i] in clusters_grouped:
                    clusters_grouped[clusters[i]].append(filtered_sd[k][i])
                else:
                    clusters_grouped[clusters[i]] = [filtered_sd[k][i]]
            for cluster in clusters_grouped:
                if len(clusters_grouped[cluster])/len(filtered_sd[k]) > 0.03:
                    self.inliers.extend(clusters_grouped[cluster])
                else:
                    self.outliers.extend(clusters_grouped[cluster])

    def trajectories_to_pickle(self):
        res = []
        for inlier in self.inliers:
            res.append((inlier, 0))
        for outlier in self.outliers:
            res.append((outlier, 1))
        pickle.dump(res, open(os.getcwd()+'/trajectories_features_labels.pkl', 'wb'))

    def preprocess(self):
        if os.path.exists("trajectories_raw.pkl"):
            self.all_trajectories = pickle.load(open("trajectories_raw.pkl", "rb"))
        else:
            print("Creating trajectories...")
            self.create_trajectories()
            pickle.dump(self.all_trajectories, open("trajectories_raw.pkl", "wb"))
            print("Trajectories created.")
        print("Fitting point scaler...")
        self.fit_point_scaler()
        print("Fitted point scaler.")
        print("Clustering trajectories...")
        self.clustering_trajectories()
        print("Clustered trajectories.")
        self.trajectories_to_pickle()
        print("Trajectories output to trajectories_features_labels.pkl")
        print(len(self.inliers))
        print(len(self.outliers))

DATA_PREFIX = "Datasets/Geolife Trajectories 1.3/Data/"
p = Preprocessor(os.getcwd()+"/"+DATA_PREFIX)
p.preprocess()
