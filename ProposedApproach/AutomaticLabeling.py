import os
import argparse
from sklearn.metrics import silhouette_score
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
import pickle

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
            s = traj[1][0][0] if self.dataset == "geolife" else traj[1][0]
            d = traj[1][-1][0] if self.dataset == "geolife" else traj[1][-1]
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
        if self.dataset == "geolife":
            X1 = set([p[0] for p in self.paths[int(x1[0])]])
            X2 = set([p[0] for p in self.paths[int(x2[0])]])
        else:
            X1 = set(self.paths[int(x1[0])])
            X2 = set(self.paths[int(x2[0])])
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

    def start(self):
        print("Reading trajectories from disk...")
        self.all_trajectories = pickle.load(open("trajectories_with_grid_"+self.dataset+".pkl", "rb"))
        print("Read trajectories from disk.")
        print("Clustering trajectories...")
        self.clustering_trajectories()
        print("Clustered trajectories.")
        self.trajectories_to_pickle()
        print("Trajectories output to trajectories_labeled_"+self.dataset+".pkl")
        print("Total number of inliers:", len(self.inliers))
        print("Total number of outliers:", len(self.outliers))

parser = argparse.ArgumentParser(description="Automatic annotation of the selected dataset.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
parser.add_argument("--thr", help="Percentage threshold for acceptable cluster size.", default="0.03")
parser.add_argument("--minThr", help="Count threshold for acceptable sd-pair size.", default="2")
parser.add_argument("--dist", help="The distance threshold for forming clusters with the complete linkage algorithm.", default="0.8")
args = parser.parse_args()

dataset = args.dataset
thr = float(args.thr)
minThr = int(args.minThr)
distClustering = float(args.dist)
l = Labeling(dataset, thr, minThr, distClustering)
l.start()
