import pickle
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from CustomScaler import Scaler
from Utils import average_length_of_sequences, distance_of_trajectory, slant
from sklearn.cluster import DBSCAN
from time import perf_counter
from numba import njit
import warnings
warnings.filterwarnings("ignore")

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

def path_clustering(X, eps):
    distances = calculate_distances(X)
    dbscan = DBSCAN(eps=eps, metric="precomputed", n_jobs=-1, min_samples=2)
    labels = dbscan.fit_predict(distances)
    y_pred = np.array([1 if l == -1 else 0 for l in labels])

    return y_pred

def feature_training(X, y):
    X_features = [[
        x[0][0], x[0][1], x[-1][0], x[-1][1],
        slant(x),
        distance_of_trajectory(x)
    ] for x in X]

    X_features = MinMaxScaler().fit_transform(X_features)
    lg = SVC(C=4000)
    lg.fit(X_features, y)
    y_pred = lg.predict(X_features)

    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare performance of clustering algorithms.")
    parser.add_argument("--dataset",
                        help="Specify the dataset to use",
                        default="geolife")
    parser.add_argument("--G", help="Specify the grid size", default="50")
    parser.add_argument("--eps", help="Specify the eps", default="2.5")
    args = parser.parse_args()

    dataset = args.dataset
    grid_scale = int(args.G)
    Eps = float(args.eps)

    data_file = "trajectories_labeled_" + dataset + ".pkl"
    data = pickle.load(open(data_file, "rb"))

    X = [[p[:2] for p in d[0]] for d in data]
    y = np.array([d[1] for d in data])

    print("Average length of raw sequences:", average_length_of_sequences(X))

    scaler = Scaler()
    points = []
    for x in X:
        points.extend(x)
    scaler.fit(points)
    X = [scaler.transform_trajectory(x) for x in X]
    
    X_grid = []
    for x in tqdm(X):
        X_grid.append(scaler.trajectory_to_grid(x, grid_scale))
    print("Average length of size " + str(grid_scale) + " grid cell sequences:",
        average_length_of_sequences(X_grid))
    
    t = perf_counter()
    y_pred1 = path_clustering(X, Eps)
    print("Finished path clustering")
    y_pred2 = feature_training(X, y)
    print("Finished feature training")

    y_pred_concat = np.concatenate((y_pred1.reshape((-1, 1)), y_pred2.reshape((-1, 1))), axis=1)
    logreg = LogisticRegression()
    logreg.fit(y_pred_concat, y)
    y_pred = logreg.predict(y_pred_concat)

    print("Running time:", round(perf_counter()-t, 1), "seconds")
    print("Accuracy score:", round(accuracy_score(y, y_pred), 4))
    print("F1 score:", round(f1_score(y, y_pred, average="macro"), 4))
    print(confusion_matrix(y, y_pred))
