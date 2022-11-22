from math import ceil
import os
from GSP import GSPModule
from Utils import average_length_of_sequences, distance_of_trajectory
from CustomScaler import Scaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pickle
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from hilbertcurve.hilbertcurve import HilbertCurve
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


def calculate_distances(X, distance_fn):
    distances = np.zeros((len(X), len(X)))
    hilbert = HilbertCurve(8, 2)

    for i in range(len(X)):
        X[i] = np.array(X[i])
    for i in range(len(X)):
        """ if i % 200 == 0:
            print(round(100*i/len(X), 1), "%") """
        for j in range(i + 1):
            d = 0
            if distance_fn == "hausdorff":
                d = max(hausdorff_dist(X[i], X[j]), hausdorff_dist(X[j], X[i]))
            elif distance_fn == "dtw":
                d, _ = fastdtw(X[i], X[j], dist=euclidean)
            elif distance_fn == "dtw_hilbert":
                xh = hilbert.distances_from_points([p for p in X[i]])
                yh = hilbert.distances_from_points([p for p in X[j]])
                d, _ = fastdtw(xh, yh, dist=lambda p1, p2: abs(p1-p2))
            else:
                raise Exception("Incorrect distance metric")
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
parser.add_argument(
    "--minPts", help="The DBSCAN minPts parameter.", default="2")
parser.add_argument(
    "--distance_fn", help="The distance function used for the path clustering method (hausdorff or dtw)", default="hausdorff")
parser.add_argument("--C", help="The C parameter.", default="8000")
parser.add_argument("--gamma", help="The gamma parameter.", default="scale")
parser.add_argument("--kernel", help="The SVM kernel.", default="rbf")
parser.add_argument("--method", help="clustering, svm or both", default="svm")
parser.add_argument("--do_gsp", help="0 or 1", default="0")
parser.add_argument("--gsp_support", default="0.05")
parser.add_argument("--seed", default="999")
args = parser.parse_args()

dataset = args.dataset
grid_scale = int(args.G)
eps = float(args.eps)
minPts = int(args.minPts)
distance_fn = args.distance_fn
C = int(args.C)
gamma = args.gamma
kernel = args.kernel
method = args.method
do_gsp = bool(int(args.do_gsp))
gsp_support = float(args.gsp_support)
seed = int(args.seed)

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))

manual_outliers = []
if dataset == "cyprus":
    try:
        manual_outliers = pickle.load(open("manual_outliers.pkl", "rb"))
    except:
        pass

X_init = [[p[:2] for p in d[0]] for d in data]
y = np.array([d[1] for d in data])

x_init_train, x_init_test, y_train, y_test = train_test_split(
    X_init, y, train_size=0.75, random_state=seed)

print("Average length of raw sequences:", average_length_of_sequences(X_init))

t = perf_counter()

scaler = Scaler()
points = []
for x in X_init:
    points.extend(x)
scaler.fit(points)

x_train = [scaler.transform_trajectory(x) for x in x_init_train]
x_test = [scaler.transform_trajectory(x) for x in x_init_test]
if dataset == "cyprus":
    manual_outliers = [scaler.transform_trajectory(x) for x in manual_outliers]


def calc_features(X, gsp_dists=[], gsp=False, isCoordinates=False):
    feature_list = []
    # FEATURES: MAYBE ADD TRIP TIME
    for i, x in enumerate(X):
        features = [x[0][0], x[0][1], x[-1][0], x[-1][1],
                    distance_of_trajectory(np.array(x), isCoordinates=isCoordinates)]
        if gsp:
            features.append(gsp_dists[i])
        feature_list.append(features)
    return feature_list


x_train_features = np.array(calc_features(x_init_train, isCoordinates=True))
minmax_values = [(np.min(x_train_features[:, j]), np.max(
    x_train_features[:, j])) for j in range(5)]

pickle.dump(minmax_values, open(dataset+"_minmax.pkl", "wb"))

X_grid_train = []
X_grid_test = []
X_grid_manual = []
for x in x_train:
    X_grid_train.append(scaler.trajectory_to_grid(x, grid_scale))
print("Average length of size " + str(grid_scale) + " grid cell sequences:",
      average_length_of_sequences(X_grid_train))

gsp = GSPModule()
if method in ["svm", "both"] and do_gsp:
    gsp.find_frequent_subsequences(
        X_grid_train+X_grid_test+X_grid_manual, gsp_support, False)

for x in x_test:
    X_grid_test.append(scaler.trajectory_to_grid(x, grid_scale))

for x in manual_outliers:
    X_grid_manual.append(scaler.trajectory_to_grid(x, grid_scale))


def sample_trajectory(X, n):
    X = np.array(X)
    sampled = X[[round(i) for i in np.linspace(0, len(X)-1, num=n)]]
    return sampled.tolist()


if method in ["clustering", "both"]:
    if distance_fn == "dtw" or distance_fn == "dtw_hilbert":
        samples = ceil(average_length_of_sequences(X_grid_train)) + 1
        X_grid_train = [sample_trajectory(x, samples) if len(
            x) > samples else x for x in X_grid_train]
        X_grid_test = [sample_trajectory(x, samples) if len(
            x) > samples else x for x in X_grid_test]
        X_grid_manual = [sample_trajectory(x, samples) if len(
            x) > samples else x for x in X_grid_manual]

    distances = []

    try:
        distances = pickle.load(
            open(dataset+'_'+distance_fn+'_'+'distances.pkl', 'rb'))
    except:
        distances = calculate_distances(
            X_grid_train+X_grid_test+X_grid_manual, distance_fn)
        pickle.dump(distances, open(
            dataset+'_'+distance_fn+'_'+'distances.pkl', 'wb'))

    distances_train = distances[:len(X_grid_train), :len(X_grid_train)]

    dbscan = DBSCAN(eps=eps, metric="precomputed",
                    n_jobs=-1, min_samples=minPts)
    labels_train = dbscan.fit_predict(distances_train)

    distances_test_pred = distances[len(X_grid_train):len(
        X_grid_train)+len(X_grid_test), :len(X_grid_train)]
    distances_manual_pred = distances[len(
        X_grid_train)+len(X_grid_test):, :len(X_grid_train)]
    labels_test = [labels_train[np.argmin(
        distances_test_pred[i])] for i in range(len(X_grid_test))]
    labels_manual = [labels_train[np.argmin(
        distances_manual_pred[i])] for i in range(len(X_grid_manual))]

    y_pred_train1 = np.array([1 if l == -1 else 0 for l in labels_train])
    y_pred_test1 = np.array([1 if l == -1 else 0 for l in labels_test])
    y_pred_manual1 = np.array([1 if l == -1 else 0 for l in labels_manual])
    print("Finished path clustering")

if method in ["svm", "both"]:
    gsp_dists_train = []
    gsp_dists_test = []
    gsp_dists_manual = []
    if do_gsp:
        gsp_dists_train = gsp.deviation_from_frequent(X_grid_train)
        gsp_dists_test = gsp.deviation_from_frequent(X_grid_test)
        gsp_dists_manual = gsp.deviation_from_frequent(X_grid_manual)

    X_features_train = calc_features(x_train, gsp_dists_train, do_gsp)
    X_features_test = calc_features(x_test, gsp_dists_test, do_gsp)
    X_features_manual = calc_features(
        manual_outliers, gsp_dists_manual, do_gsp)

    minmax = MinMaxScaler()
    X_features_train = minmax.fit_transform(X_features_train)
    X_features_test = minmax.transform(X_features_test)
    if dataset == "cyprus":
        X_features_manual = minmax.transform(X_features_manual)

    svm = SVC(C=C, gamma=gamma, kernel=kernel)
    svm.fit(X_features_train, y_train)
    y_pred_train2 = svm.predict(X_features_train)
    y_pred_test2 = svm.predict(X_features_test)
    if dataset == "cyprus":
        y_pred_manual2 = svm.predict(X_features_manual)

    print("Finished feature training")

if method == "both":
    y_pred_train_concat = np.concatenate(
        (y_pred_train1.reshape((-1, 1)), y_pred_train2.reshape((-1, 1))), axis=1)
    y_pred_test_concat = np.concatenate(
        (y_pred_test1.reshape((-1, 1)), y_pred_test2.reshape((-1, 1))), axis=1)
    if dataset == "cyprus":
        y_pred_manual_concat = np.concatenate(
            (y_pred_manual1.reshape((-1, 1)), y_pred_manual2.reshape((-1, 1))), axis=1)
    logreg = LogisticRegression()
    logreg.fit(y_pred_train_concat, y_train)
    y_pred_train = logreg.predict(y_pred_train_concat)
    y_pred_test = logreg.predict(y_pred_test_concat)
    if dataset == "cyprus":
        y_pred_manual = logreg.predict(y_pred_manual_concat)
    # print(logreg.coef_)
elif method == "clustering":
    y_pred_train = y_pred_train1
    y_pred_test = y_pred_test1
    if dataset == "cyprus":
        y_pred_manual = y_pred_manual1
else:  # SVM
    y_pred_train = y_pred_train2
    y_pred_test = y_pred_test2
    if dataset == "cyprus":
        y_pred_manual = y_pred_manual2

print("Running time:", round(perf_counter()-t, 1), "seconds")
print("Train accuracy score:", round(accuracy_score(y_train, y_pred_train), 4))
print("Train F1 score:", round(f1_score(y_train, y_pred_train, average="macro"), 4))
print("Test accuracy score:", round(accuracy_score(y_test, y_pred_test), 4))
print("Test F1 score:", round(f1_score(y_test, y_pred_test, average="macro"), 4))
print(confusion_matrix(y_test, y_pred_test))
if dataset == "cyprus" and len(manual_outliers) > 0:
    print(y_pred_manual)
    print("Percentage of manual outliers found by the system: {pct}%".format(
        pct=100*len([y for y in y_pred_manual if y == 1])/len(y_pred_manual)))

if method != "clustering":
    output = []
    for i, x in enumerate(x_init_test):
        output.append([X_features_test[i], x_init_test[i], y_pred_test[i]])
    pickle.dump(output, open(
        f"trajectory_features_labeled_{dataset}.pkl", "wb"))
