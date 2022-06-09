import os
import argparse
import json
from sklearn.metrics import silhouette_score
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
import pickle
from datetime import datetime
from CustomScaler import Scaler
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt

class Normalizer:
    def __init__(self, dataset, data_path, grid_scale):
        self.data_path = data_path
        self.dataset = dataset
        self.all_trajectories = []
        self.scaler = Scaler()
        self.grid_scale = grid_scale
        self.trajectories_with_grid = []

    def process_file(self, f):
        file = open(f, 'r')
        if self.dataset == "geolife":
            lines = file.readlines()[6:]
            file.close()

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
            print("Not implemented")

    def process_skg_json(self, json_text):
        obj = json.loads(json_text)
        traj_raw = obj["signals"]
        traj = []
        for p in traj_raw:
            latitude = p["latitude"]/10**5
            longitude = p["longitude"]/10**5
            traj.append([longitude, latitude])
        return traj

    def create_trajectories(self):
        if self.dataset == "geolife":
            for i in tqdm(os.listdir(self.data_path)):
                i_path = self.data_path+i+"/Trajectory/"
                for j in os.listdir(i_path):
                    trajectory = self.process_file(i_path+j)
                    valid = True
                    for p in trajectory:
                        if p[0] < 116.1 or p[0] > 116.6 or p[1] < 39.65 or p[1] > 40.3:
                            valid = False
                            break
                    if valid and len(trajectory) > 0:
                        self.all_trajectories.append(trajectory)
        elif self.dataset == "thessaloniki":
            file =  open(self.data_path+"decoded.txt", "r")
            lines = file.readlines()
            file.close()

            self.all_trajectories = [self.process_skg_json(line) for line in lines]

            points = []
            for t in self.all_trajectories:
                for p in t:
                    points.append(p)
            plt.scatter([p[0] for p in points], [p[1] for p in points])
            plt.show()
        else:
            print("Not implemented")

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
            to_append = self.coords_to_grid(self.scaler.transform([p[:2]])[0], grid_scale)
            trajectory_transformed.append(to_append)
        return self.remove_duplicates(trajectory_transformed)

    def transform_all_trajectories(self):
        for t in self.all_trajectories:
            self.trajectories_with_grid.append([
                t, self.transform_trajectory_to_grid(t, self.grid_scale)
            ])

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

    def trajectory_statistics(self):
        simple_lengths = [len(t[0]) for t in self.trajectories_with_grid]
        grid_lengths = [len(t[1]) for t in self.trajectories_with_grid]

        print("Average simple path length:", sum(simple_lengths)/len(simple_lengths))
        print("Average grid path length:", sum(grid_lengths)/len(grid_lengths))
        print("Lengths of grid paths:", Counter(grid_lengths))

data_prefixes = {
    "geolife": "Datasets/Geolife Trajectories 1.3/Data/",
    "thessaloniki": "Datasets/ThessalonikiDataset/"
}

class Labeling:
    def __init__(self, dataset, thr, minThr, distClustering):
        self.all_trajectories = []
        self.paths = []
        self.dist_clustering = distClustering
        self.inliers = []
        self.outliers = []
        self.dataset = dataset
        self.thr = thr
        self.minThr = minThr

    def group_by_sd_pairs(self, trajectories, threshold):
        sd_pairs = dict()
        for traj in trajectories:
            s = traj[1][0]
            d = traj[1][-1]
            sd_pair = s+"->"+d
            if sd_pair in sd_pairs:
                sd_pairs[sd_pair].append(traj)
            else:
                sd_pairs[sd_pair] = [traj]
        filtered_dict = dict()
        for sd in sd_pairs:
            if len(sd_pairs[sd]) >= threshold:
                filtered_dict[sd] = sd_pairs[sd]
            else:
                self.outliers.extend([t[0] for t in sd_pairs[sd]])

        return filtered_dict

    def intersection(self, lst1, lst2):
        return set(lst1).intersection(lst2)

    def union(self, lst1, lst2):
        return set(lst1).union(lst2)

    def custom_distance(self, x1, x2):
        X1 = set([p[0] for p in self.paths[int(x1[0])]])
        X2 = set([p[0] for p in self.paths[int(x2[0])]])
        jaccard_sq = 1 - len(X1.intersection(X2))/len(X1.union(X2))
        return jaccard_sq

    def clustering_trajectories(self):
        filtered_sd = self.group_by_sd_pairs(self.all_trajectories, self.minThr)
        print("Total number of trajectories:", len(self.all_trajectories))
        print("Number of step 1 outliers:", len(self.outliers))
        score = []

        for k in filtered_sd:
            self.paths = [f[1] for f in filtered_sd[k]]
            to_cluster = [[i] for i in range(len(self.paths))]
            linked = linkage(to_cluster, method='complete', metric=self.custom_distance)
            clusters = fcluster(linked, t=self.dist_clustering, criterion='distance')

            if len(set(clusters)) >= 2 and len(set(clusters)) <= len(to_cluster)-1:
                silhouette_score_1 = silhouette_score(to_cluster, clusters, metric=self.custom_distance)
                score.append(silhouette_score_1)

            clusters_grouped = dict()
            for i in range(len(clusters)):
                if clusters[i] in clusters_grouped:
                    clusters_grouped[clusters[i]].append(filtered_sd[k][i])
                else:
                    clusters_grouped[clusters[i]] = [filtered_sd[k][i]]
            for cluster in clusters_grouped:
                if len(clusters_grouped[cluster])/len(filtered_sd[k]) > self.thr:
                    self.inliers.extend([t[0] for t in clusters_grouped[cluster]])
                else:
                    self.outliers.extend([t[0] for t in clusters_grouped[cluster]])
        print("Average silhouette score:", sum(score)/len(score))

    def trajectories_to_pickle(self):
        res = []
        for inlier in self.inliers:
            res.append((inlier, 0))
        for outlier in self.outliers:
            res.append((outlier, 1))
        pickle.dump(res, open(os.getcwd()+"/trajectories_labeled_"+self.dataset+".pkl", 'wb'))

    def start(self, data):
        self.all_trajectories = data
        print("Clustering trajectories...")
        self.clustering_trajectories()
        print("Clustered trajectories.")
        self.trajectories_to_pickle()
        print("Trajectories output to trajectories_labeled_"+self.dataset+".pkl")
        print("Total number of inliers:", len(self.inliers))
        print("Total number of outliers:", len(self.outliers))

parser = argparse.ArgumentParser(description="Automatic annotation of the selected dataset.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
parser.add_argument("--G", help="The number of grid cells per dimension", default="5")
parser.add_argument("--thr", help="Percentage threshold for acceptable cluster size.", default="0.03")
parser.add_argument("--minThr", help="Count threshold for acceptable sd-pair size.", default="15")
parser.add_argument("--dist", help="The distance threshold for forming clusters with the complete linkage algorithm.", default="0.4")
args = parser.parse_args()

dataset = args.dataset
thr = float(args.thr)
minThr = int(args.minThr)
distClustering = float(args.dist)
grid_scale = int(args.G)

nm = Normalizer(dataset, os.getcwd()+"/"+data_prefixes[dataset], grid_scale)
nm.preprocess()
nm.trajectory_statistics()

l = Labeling(dataset, thr, minThr, distClustering)
l.start(nm.trajectories_with_grid)
