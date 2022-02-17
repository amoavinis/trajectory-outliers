import os
import geopy.distance
from CustomScaler import Scaler
from sklearn.preprocessing import MinMaxScaler
import datetime
from scipy.cluster.hierarchy import linkage, fcluster
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
        self.representation_pca = None
        self.grid_scale = 100
        self.inliers = []
        self.outliers = []

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
            result.append([longitude, latitude, timestamp])

        return result

    def create_trajectories(self):
        for i in tqdm(os.listdir(self.data_path)[:4]):
            for j in os.listdir(self.data_path+i+'/Trajectory/'):
                self.all_trajectories.append(
                    self.process_file(self.data_path+i+'/Trajectory/'+j))

    def distance_of_transition(self, transition):
        return geopy.distance.distance(transition[0], transition[1]).meters

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
        sine = dx / sd_len
        
        return sine

    def get_features(self):
        trajs = []
        for traj in tqdm(self.all_trajectories):
            speeds = self.speeds_in_traj(traj)
            t = [
                [x[:2] for x in traj],
                self.slant(traj),
                traj[0][0],
                traj[0][1],
                traj[-1][0],
                traj[-1][1],
                self.distance_of_trajectory(traj),
                speeds['min_speed'],
                speeds['max_speed'],
                speeds['avg_speed'],
                [x[2] for x in traj]
            ]
            trajs.append(t)
        self.all_trajectories = trajs

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
            s = self.coords_to_grid(traj[0][0], 100)
            d = self.coords_to_grid(traj[0][-1], 100)
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
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3
    
    def union(self, lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list        

    def normalize_trajectory_features(self):
        trajectories = self.all_trajectories
        temp = [traj[2:-1] for traj in trajectories]
        temp = self.traj_scaler.fit_transform(temp)
        transformed = [trajectories[i][:2] + temp[i].tolist() + trajectories[i][-1] for i in range(len(trajectories))]
        self.all_trajectories = transformed

    def custom_distance(self, x1, x2):
        grid_x1 = self.transform_trajectory_to_grid(x1[0], 1000)
        grid_x2 = self.transform_trajectory_to_grid(x2[0], 1000)
        jaccard_sq = (1 - len(self.intersection(grid_x1, grid_x2))/len(self.union(grid_x1, grid_x2)))**2
        d_slant_sq = ((x1[1] - x2[1])/2)**2
        rest_squared = sum([(x1[i] - x2[i])**2 for i in range(2, len(x1)-1)])
        sum_sq = jaccard_sq + d_slant_sq + rest_squared

        return sqrt(sum_sq)

    def clustering_trajectories(self):
        trajectories = self.all_trajectories
        filtered_sd = self.group_by_sd_pairs(trajectories, 100)
        for k in filtered_sd:
            linked = linkage(filtered_sd[k], method='complete', metric=self.custom_distance, optimal_ordering=True)
            clusters = fcluster(linked, t=0.2, criterion='distance')

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
        print("Creating trajectories...")
        self.create_trajectories()
        print("Trajectories created.")
        print("Fitting point scaler...")
        self.fit_point_scaler()
        print("Fitted point scaler.")
        print("Extracting features...")
        self.get_features()
        print("Features extracted.")
        print("Normalizing trajectory features...")
        self.normalize_trajectory_features()
        print("Normalized trajectory features.")
        print("Clustering trajectories...")
        self.normalize_trajectory_features()
        print("Clustered trajectories.")
        #self.trajectories_to_pickle()
        #print("Trajectories output to trajectories_features.pkl")

DATA_PREFIX = "Geolife Trajectories 1.3/Data/"
p = Preprocessor(os.getcwd()+"/"+DATA_PREFIX)
p.preprocess()
