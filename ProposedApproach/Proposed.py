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
from tqdm import tqdm
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
parser.add_argument("--do_gsp", help="0 or 1", default="0")
parser.add_argument("--gsp_support", default="0.05")
parser.add_argument("--seed", default="999")
args = parser.parse_args()

dataset = args.dataset
grid_scale = int(args.G)
eps = float(args.eps)
C = int(args.C)
gamma = args.gamma
kernel = args.kernel
method = args.method
do_gsp = bool(int(args.do_gsp))
gsp_support = float(args.gsp_support)
seed = int(args.seed)

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))

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

X_grid_train = []
X_grid_test = []
for x in x_train:
    X_grid_train.append(scaler.trajectory_to_grid(x, grid_scale))
print("Average length of size " + str(grid_scale) + " grid cell sequences:",
      average_length_of_sequences(X_grid_train))

gsp = GSPModule()
if method != "cluster" and do_gsp:
    gsp.find_frequent_subsequences(X_grid_train+X_grid_test, gsp_support, True)

for x in x_test:
    X_grid_test.append(scaler.trajectory_to_grid(x, grid_scale))

if method != "svm":
    distances = calculate_distances(X_grid_train+X_grid_test)
    distances_train = distances[:len(X_grid_train), :len(X_grid_train)]

    dbscan = DBSCAN(eps=eps, metric="precomputed", n_jobs=-1, min_samples=2)
    labels_train = dbscan.fit_predict(distances_train)

    distances_test_pred = distances[len(X_grid_train):, :len(X_grid_train)]
    labels_test = [labels_train[np.argmin(
        distances_test_pred[i])] for i in range(len(X_grid_test))]

    y_pred_train1 = np.array([1 if l == -1 else 0 for l in labels_train])
    y_pred_test1 = np.array([1 if l == -1 else 0 for l in labels_test])
    print("Finished path clustering")


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

if method != "cluster":
    gsp_dists_train = []
    gsp_dists_test = []
    if do_gsp:
        gsp_dists_train = gsp.deviation_from_frequent(X_grid_train)
        gsp_dists_test = gsp.deviation_from_frequent(X_grid_test)

    X_features_train = calc_features(x_train, gsp_dists_train, do_gsp)
    X_features_test = calc_features(x_test, gsp_dists_test, do_gsp)

    minmax = MinMaxScaler()
    X_features_train = minmax.fit_transform(X_features_train)
    X_features_test = minmax.transform(X_features_test)

    svm = SVC(C=C, gamma=gamma, kernel=kernel)
    svm.fit(X_features_train, y_train)
    y_pred_train2 = svm.predict(X_features_train)
    y_pred_test2 = svm.predict(X_features_test)

    print("Finished feature training")

if method == "both":
    y_pred_train_concat = np.concatenate(
        (y_pred_train1.reshape((-1, 1)), y_pred_train2.reshape((-1, 1))), axis=1)
    y_pred_test_concat = np.concatenate(
        (y_pred_test1.reshape((-1, 1)), y_pred_test2.reshape((-1, 1))), axis=1)
    logreg = LogisticRegression()
    logreg.fit(y_pred_train_concat, y_train)
    y_pred_train = logreg.predict(y_pred_train_concat)
    y_pred_test = logreg.predict(y_pred_test_concat)
    print(logreg.coef_)
elif method == "cluster":
    y_pred_train = y_pred_train1
    y_pred_test = y_pred_test1
else:
    y_pred_train = y_pred_train2
    y_pred_test = y_pred_test2

print("Running time:", round(perf_counter()-t, 1), "seconds")
print("Train accuracy score:", round(accuracy_score(y_train, y_pred_train), 4))
print("Train F1 score:", round(f1_score(y_train, y_pred_train, average="macro"), 4))
print("Test accuracy score:", round(accuracy_score(y_test, y_pred_test), 4))
print("Test F1 score:", round(f1_score(y_test, y_pred_test, average="macro"), 4))
print(confusion_matrix(y_test, y_pred_test))

if method != "cluster":
    output = []
    for i, x in enumerate(x_init_test):
        output.append([X_features_test[i], x_init_test[i], y_pred_test[i]])
    pickle.dump(output, open(
        f"trajectory_features_labeled_{dataset}.pkl", "wb"))
