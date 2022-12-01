from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap
import pickle
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample, seed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
seed(2)

parser = argparse.ArgumentParser(
    description="Run traffic insights.")
parser.add_argument("--dataset",
                    help="Specify the dataset to use",
                    default="geolife")
parser.add_argument(
    "--K", help="The K parameter of the K-Means algorithm.", default="5")
parser.add_argument("--plot_nn_outliers",
                    help="Show a plot of the outliers and their nearest neighbors (0 for no, 1 for yes).", default="0")
parser.add_argument("--plot_manual_outliers",
                    help="Show a plot of the manually added outliers (0 for no, 1 for yes).", default="0")
args = parser.parse_args()
dataset = args.dataset
K = int(args.K)
plot_nn_outliers = args.plot_nn_outliers == "1"
plot_manual_outliers = args.plot_manual_outliers == "1"

data = pickle.load(open("trajectory_features_labeled_"+dataset+".pkl", "rb"))

X = [[d[0][0], d[0][1], d[0][2], d[0][3], d[0][4]] for d in data]
paths = [d[1] for d in data]
y = np.array([d[2] for d in data])

knn_model = NearestNeighbors(n_neighbors=51)
knn_model.fit(X)
_, neighbors = knn_model.kneighbors(X)
outlier_indices = [i for i, x in enumerate(y) if x == 1]
sampled = sample(outlier_indices, 10)

if plot_manual_outliers and dataset == "cyprus":
    manual_outliers = pickle.load(open("manual_outliers.pkl", "rb"))
    hsv = get_cmap('hsv', 256)
    cmap = colors.ListedColormap(hsv(np.linspace(0, 1, len(manual_outliers))))
    for i in range(len(manual_outliers)):
        plt.plot([p[0] for p in manual_outliers[i]], [
                 p[1] for p in manual_outliers[i]], color=cmap.colors[i], label="Outlier "+str(i+1))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper left", prop={'size': 18}, bbox_to_anchor=(1, 1.05))
    plt.show()


def plot_outlier(x, neighbors_dict):
    plt.plot([p[0] for p in x], [p[1] for p in x], color='r')
    for n in neighbors_dict:
        plt.plot([p[0] for p in neighbors_dict[n][0]], [
                 p[1] for p in neighbors_dict[n][0]], color='b' if neighbors_dict[n][1] == 0 else 'r')
    plt.show()


if plot_nn_outliers:
    for id in sampled:
        neighbors_dict = {}
        for neighbor in neighbors[id][1:]:
            neighbors_dict[neighbor] = [paths[neighbor], y[neighbor]]
        plot_outlier(paths[id], neighbors_dict)


def clustering(data, paths, e):
    kmeans = KMeans(e, random_state=1)
    y_pred = kmeans.fit_predict(data[:, :4])

    print(e, "-", silhouette_score(data, y_pred))

    # Uncomment below lines to plot starting and ending coordinates of trajectories.
    """ for class_id in range(e):
        class_data = [data[i]
                      for i in range(len(data)) if y_pred[i] == class_id]
        points = []
        for row in class_data:
            points.append(row[:2])
            points.append(row[2:])
        plt.scatter([p[0] for p in points], [p[1]
                    for p in points], label="Cluster "+str(class_id+1))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show() """

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    labels_used = set()
    for label in range(e):
        class_label = [paths[i]
                       for i in range(len(y_pred)) if y_pred[i] == label]
        sample_limit = min(1000, len(class_label))
        for x in sample(class_label, sample_limit):
            if label in labels_used:
                plt.plot([p[0] for p in x], [p[1]
                         for p in x], colors[label % 7])
            else:
                plt.plot([p[0] for p in x], [p[1] for p in x],
                         colors[label % 7], label="Cluster "+str(label+1))
                labels_used.add(label)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(prop={'size': 20})
    plt.show()

    return y_pred


outlier_indices_set = set(outlier_indices)
inliers = []
inlier_paths = []
for i in range(len(X)):
    if y[i] == 0:
        inliers.append(X[i])
        inlier_paths.append(paths[i])
inliers = np.array(inliers)
cluster_ids = clustering(inliers, inlier_paths, K)


def normalizedToOriginal(min, max, value):
    return value*(max-min) + min


minmax_values = pickle.load(open(dataset+"_minmax.pkl", "rb"))
for class_id in range(K):
    cluster_data = np.array(
        [x for i, x in enumerate(inliers) if cluster_ids[i] == class_id])
    print("Cluster " + str(class_id+1))
    print("Starting latitude:", normalizedToOriginal(minmax_values[0][0], minmax_values[0][1], min(
        cluster_data[:, 0])), "->", normalizedToOriginal(minmax_values[0][0], minmax_values[0][1], max(cluster_data[:, 0])))
    print("Starting longitude", normalizedToOriginal(minmax_values[1][0], minmax_values[1][1], min(
        cluster_data[:, 1])), "->", normalizedToOriginal(minmax_values[1][0], minmax_values[1][1], max(cluster_data[:, 1])))
    print("Ending latitude", normalizedToOriginal(minmax_values[2][0], minmax_values[2][1], min(
        cluster_data[:, 2])), "->", normalizedToOriginal(minmax_values[2][0], minmax_values[2][1], max(cluster_data[:, 2])))
    print("Ending longitude", normalizedToOriginal(minmax_values[3][0], minmax_values[3][1], min(
        cluster_data[:, 3])), "->", normalizedToOriginal(minmax_values[3][0], minmax_values[3][1], max(cluster_data[:, 3])))
    print("Distance", normalizedToOriginal(minmax_values[4][0], minmax_values[4][1], min(
        [d for d in cluster_data[:, 4] if d > 0.0001])), "->", normalizedToOriginal(minmax_values[4][0], minmax_values[4][1], max(cluster_data[:, 4])))
