import pickle
import argparse
from tqdm import tqdm
from time import perf_counter
from numba import njit
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from CustomScaler import Scaler
from Utils import average_length_of_sequences, distance_of_trajectory
from GSP import GSPModule

@njit
def hausdorff_dist(A, B):
    dist = np.float32(-1.0)
    for a in A:
        minimum = np.float32(10000.0)
        for b in B:
            d = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            if d < minimum:
                minimum = d
        if minimum > dist:
            dist = minimum
    return dist

@njit
def calculate_distances(X):
    distances = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1):
            d = max(hausdorff_dist(X[i], X[j]), hausdorff_dist(X[j], X[i]))
            distances[i, j] = d
            distances[j, i] = d
    return distances

parser = argparse.ArgumentParser(
    description="Compare performance of clustering algorithms.")
parser.add_argument("--dataset",
                    help="Specify the dataset to use",
                    default="geolife")
parser.add_argument("--G", help="Specify the grid size", default="40")
parser.add_argument("--eps", help="Specify the eps", default="1.5")
parser.add_argument("--C", help="The C parameter.", default="8000")
parser.add_argument("--gamma", help="The gamma parameter.", default="scale")
parser.add_argument("--kernel", help="The SVM kernel.", default="rbf")
parser.add_argument("--method", help="clustering, svm or both", default="svm")
args = parser.parse_args()

dataset = args.dataset
grid_scale = int(args.G)
eps = float(args.eps)
C = int(args.C)
gamma = args.gamma
kernel = args.kernel
method = args.method

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))

X_init = [[p[:2] for p in d[0]] for d in data]
y = np.array([d[1] for d in data])

print("Average length of raw sequences:", average_length_of_sequences(X_init))

scaler = Scaler()
points = []
for x in X_init:
    points.extend(x)
scaler.fit(points)
X = [scaler.transform_trajectory(x) for x in X_init]

X_grid = []
for x in tqdm(X):
    X_grid.append(scaler.trajectory_to_grid(x, grid_scale))
print("Average length of size " + str(grid_scale) + " grid cell sequences:",
    average_length_of_sequences(X_grid))

t = perf_counter()

if method != "svm":
    distances = calculate_distances(X_grid)
    dbscan = DBSCAN(eps=eps, metric="precomputed", n_jobs=-1, min_samples=2)
    labels = dbscan.fit_predict(distances)
    y_pred1 = np.array([1 if l == -1 else 0 for l in labels])
    print("Finished path clustering")

if method != "cluster":
    gsp = GSPModule()
    gsp_dists = gsp.deviation_from_frequent(X, 0.05)

    X_features = []
    for i, x in enumerate(X):
        X_features.append([
            x[0][0], x[0][1], x[-1][0], x[-1][1],
            distance_of_trajectory(x),
            gsp_dists[i]
        ])

    X_features = MinMaxScaler().fit_transform(X_features)
    lg = SVC(C=C, gamma=gamma, kernel=kernel)
    lg.fit(X_features, y)
    y_pred2 = lg.predict(X_features)

    print("Finished feature training")

if method == "both":
    y_pred_concat = np.concatenate((y_pred1.reshape((-1, 1)), y_pred2.reshape((-1, 1))), axis=1)
    logreg = LogisticRegression()
    logreg.fit(y_pred_concat, y)
    y_pred = logreg.predict(y_pred_concat)
    print(logreg.coef_) 
elif method == "cluster":
    y_pred = y_pred1
else:
    y_pred = y_pred2

print("Running time:", round(perf_counter()-t, 1), "seconds")
print("Accuracy score:", round(accuracy_score(y, y_pred), 4))
print("F1 score:", round(f1_score(y, y_pred, average="macro"), 4))
print(confusion_matrix(y, y_pred))

output = []
for i, x in enumerate(X_init):
    output.append([X_features[i], X_init[i], y_pred[i]])
pickle.dump(output, open("trajectory_features_labeled.pkl", "wb"))

