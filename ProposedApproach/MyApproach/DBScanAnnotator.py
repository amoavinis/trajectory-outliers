import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pickle
import itertools
import matplotlib.pyplot as plt
import os

class DBScanAnnotator:
    def __init__(self, filename):
        self.X = np.array(pickle.load(open(filename, "rb")))
        self.scaler = MinMaxScaler()
        self.dbscan = DBSCAN(eps=0.1, min_samples=10)

    def fit(self):
        self.X = self.scaler.fit_transform(self.X)
        self.dbscan.fit(self.X)

    def printStats(self):
        labels = self.dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("Silhouette Coefficient: %0.3f" %
              metrics.silhouette_score(self.X, labels))

    def scatterPlots(self):
        core_samples_mask = np.zeros_like(self.dbscan.labels_, dtype=bool)
        core_samples_mask[self.dbscan.core_sample_indices_] = True

        # #############################################################################
        # Plot result

        pairs = list(itertools.combinations(list(range(self.X.shape[1])), 2))

        for p in pairs:
            # Black removed and is used for noise instead.
            unique_labels = set(self.dbscan.labels_)
            colors = [plt.cm.Spectral(each) for each in np.linspace(
                0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = self.dbscan.labels_ == k

                xy = self.X[class_member_mask & core_samples_mask]
                plt.plot(
                    xy[:, p[0]],
                    xy[:, p[1]],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                )

                xy = self.X[class_member_mask & ~core_samples_mask]
                plt.plot(
                    xy[:, p[0]],
                    xy[:, p[1]],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                )

            plt.title("Viewing dimensions "+str(p[0])+" and "+str(p[1]))
            plt.show()

    def saveResults(self):
        X_y = []
        for i in range(len(self.X)):
            label = None
            if self.dbscan.labels_[i] >= 0:
                label = 0
            else:
                label = 1
            X_y.append(list(self.X[i])+[label])

        pickle.dump(X_y, open(os.getcwd()+"/pickles/trajectories_labeled.pkl", "wb"))

annotator = DBScanAnnotator(os.getcwd()+"/pickles/trajectories_features.pkl")
annotator.fit()
annotator.printStats()
annotator.saveResults()
