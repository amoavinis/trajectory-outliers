import pickle
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from ClusterSegments import ClusterSegments
from CustomScaler import Scaler
from Utils import hausdorff_dist
from sklearn.cluster import DBSCAN
from random import sample, seed
import pickle
from time import perf_counter
from collections import Counter
seed(1997)

parser = argparse.ArgumentParser(
    description="Compare performance of clustering algorithms.")
parser.add_argument("--dataset",
                    help="Specify the dataset to use",
                    default="geolife")
args = parser.parse_args()

dataset = args.dataset

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
data = sample(data, 1000)

X = [[p[:2] for p in d[0]] for d in data]
y = np.array([d[1] for d in data])

def average_length_of_sequences(sequences):
    s = sum([len(x) for x in sequences])
    avg = s / len(sequences)
    return avg

print("Average length of raw sequences:", average_length_of_sequences(X))

grid_scale = 30

scaler = Scaler()
points = []
for x in X:
    points.extend(x)
scaler.fit(points)
X = [scaler.transform_trajectory(x) for x in X]
X_grid = [scaler.trajectory_to_grid(x, grid_scale) for x in X]

print("Average length of size "+str(grid_scale)+" grid cell sequences:", average_length_of_sequences(X_grid))

def distance_function_paths(t1, t2):
    trip1_grid = X_grid[int(t1[0])]
    trip2_grid = X_grid[int(t2[0])]
    dist = hausdorff_dist(trip1_grid, trip2_grid)
    return dist

indices = np.array(range(len(X)))
y = y[indices]

dbscan = DBSCAN(eps=1.2, min_samples=2, metric=distance_function_paths)

t = perf_counter()

labels = [0 for _ in indices]#dbscan.fit_predict(indices.reshape((-1, 1)))
print(Counter(labels))
print("Fitting time:", perf_counter() - t)

y_pred_1 = [1 if l==-1 else 0 for l in labels]

cluster_segments = ClusterSegments()
labels = s=cluster_segments.fit_predict(X)

#minmax = MinMaxScaler()
#X_features = [trajectory_features(X[i]) for i in indices]
#X_features = minmax.fit_transform(X_features)

#dbscan = DBSCAN(eps=0.15, min_samples=2)
#labels = dbscan.fit_predict(X_features)
#kmeans = CustomKMeans(5)
#kmeans.fit(X_features)
#labels = kmeans.predict(X_features)

#print(Counter(labels))
print("Fitting time:", perf_counter() - t)

y_pred_2 = labels#calculate_labels(labels, -1)

y_pred = []
for i in range(len(y_pred_1)):
    if y_pred_2[i] == 1 or y_pred_2[i] == 1:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(accuracy_score(y, y_pred))
print(f1_score(y, y_pred, average="macro"))
print(confusion_matrix(y, y_pred))