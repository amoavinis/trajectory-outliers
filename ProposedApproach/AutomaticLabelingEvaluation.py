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

cyprus_trajectories = pickle.load(open("trajectories_labeled_cyprus.pkl", "rb"))
manual_outliers = pickle.load(open("manual_outliers.pkl", "rb"))

class GridConverter:
    def __init__(self, trajectories, grid_scale):
        self.grid_scale = grid_scale
        self.scaler = Scaler()
        self.all_trajectories = trajectories

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
        grid_trajectories = []
        for t in self.all_trajectories:
            grid_trajectories.append(self.transform_trajectory_to_grid(t, self.grid_scale))
        return grid_trajectories

class Labeling:
    def __init__(self, thr, minThr, distClustering):
        self.all_trajectories = []
        self.paths = []
        self.dist_clustering = distClustering
        self.inliers = []
        self.outliers = []
        self.thr = thr
        self.minThr = minThr

    def intersection(self, lst1, lst2):
        return set(lst1).intersection(lst2)

    def union(self, lst1, lst2):
        return set(lst1).union(lst2)

    def custom_distance(self, x1, x2):
        X1 = set([p[0] for p in self.paths[int(x1[0])]])
        X2 = set([p[0] for p in self.paths[int(x2[0])]])
        jaccard_sq = 1 - len(X1.intersection(X2))/len(X1.union(X2))
        return jaccard_sq

    def group_by_sd_pairs(self, trajectories, threshold):
        sd_pairs = dict()
        outlier_indices = []
        for i in range(len(trajectories)):
            traj = trajectories[i]
            s = traj[0]
            d = traj[-1]
            sd_pair = s+"->"+d
            if sd_pair in sd_pairs:
                sd_pairs[sd_pair].append({"index": i, "trajectory": traj})
            else:
                sd_pairs[sd_pair] = [{"index": i, "trajectory": traj}]
        filtered_dict = dict()
        for sd in sd_pairs:
            if len(sd_pairs[sd]) >= threshold:
                filtered_dict[sd] = sd_pairs[sd]
            else:
                outlier_indices.extend([sd_pairs[sd][i]["index"] for i in range(len(sd_pairs[sd]))])

        return filtered_dict, outlier_indices

    def clustering_trajectories(self, trajectories):
        filtered_sd, outlier_indices_step1 = self.group_by_sd_pairs(trajectories, self.minThr)

        for k in filtered_sd:
            self.paths = [f["trajectory"] for f in filtered_sd[k]]
            to_cluster = [[i] for i in range(len(self.paths))]
            linked = linkage(to_cluster, method='complete', metric=self.custom_distance)
            clusters = fcluster(linked, t=self.dist_clustering, criterion='distance')

            inliers = []
            outliers = []

            clusters_grouped = dict()
            for i in range(len(clusters)):
                if clusters[i] in clusters_grouped:
                    clusters_grouped[clusters[i]].append(filtered_sd[k][i])
                else:
                    clusters_grouped[clusters[i]] = [filtered_sd[k][i]]
            for cluster in clusters_grouped:
                if len(clusters_grouped[cluster])/len(filtered_sd[k]) > self.thr:
                    inliers.extend([t["index"] for t in clusters_grouped[cluster]])
                else:
                    outliers.extend([t["index"] for t in clusters_grouped[cluster]])
            return inliers, outliers+outlier_indices_step1

cyprus_trajectories = [t[0] for t in cyprus_trajectories]

all_trajectories = cyprus_trajectories+manual_outliers

manual_outlier_indices = list(range(len(cyprus_trajectories), len(all_trajectories)))

grid_converter = GridConverter(all_trajectories, 10)
grid_converter.fit_point_scaler()

trajectories_grid = grid_converter.transform_all_trajectories()

l = Labeling(0.03, 15, 0.4)
(inlier_indices, outlier_indices) = l.clustering_trajectories(trajectories_grid)

count = 0
outlier_indices_set = set(outlier_indices)
for idx in manual_outlier_indices:
    if idx in outlier_indices_set:
        print("True outlier found at index:", idx-len(cyprus_trajectories))
        count += 1
    else:
        print("True outlier missed at index:", idx-len(cyprus_trajectories))
        print("Grid length:", len(trajectories_grid[idx]))
        pass

print("True outliers missed:", len(manual_outliers) - count)
print(count/len(manual_outliers))
