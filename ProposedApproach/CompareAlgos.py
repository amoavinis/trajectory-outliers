import os
import pickle
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from Utils import *
from sklearn.cluster import DBSCAN
from random import sample, seed
from SimplifyTrajectories import Simplifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description="Compare performance of clustering algorithms.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
args = parser.parse_args()

dataset = args.dataset

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
seed(1997)
sampled = sample(data, int(len(data)/10) if len(data) > 10000 else len(data))
X = [[p[:2] for p in d[0]] for d in data]
y = [d[1] for d in data]

def average_length_of_sequences(sequences):
    s = sum([len(x) for x in sequences])
    avg = s/len(sequences)
    return avg

print("Average length of raw sequences:", average_length_of_sequences(X))

# thresold in rads
simplifier = Simplifier(150)
X = simplifier.simplify(X)

print("Average length of simplified sequences:", average_length_of_sequences(X))

distances = []
distances_pairwise = dict()

if os.path.exists("main_trajectory_features_"+dataset+".pkl"):
    loaded = pickle.load(open("main_trajectory_features_"+dataset+".pkl", "rb"))
    distances = loaded[0]
    distances_pairwise = loaded[1]
else:
    print('Scaling distances...')
    distances = [[distance_of_trajectory(t)] for t in X]
    distance_scaler = MinMaxScaler()
    sampled_X = [[p[:2] for p in d[0]] for d in sampled]
    sampled_distances = [[distance_of_trajectory(t)] for t in sampled_X]
    distance_scaler.fit(sampled_distances)
    distances = distance_scaler.fit_transform(distances)

    print("Scaling Hausdorff pairwise distances...")
    distances_pairwise = dict()
    distances_pairwise_list = []
    for i in tqdm(range(len(X))):
        temp_dict = dict()
        for j in range(len(X)):
            d = hausdorff_dist(np.array(X[i]), np.array(X[j]))
            distances_pairwise_list.append([d])
            temp_dict[j] = d
        distances_pairwise[i] = temp_dict
    distances_pairwise_scaler = MinMaxScaler()
    distances_pairwise_list = distances_pairwise_scaler.fit_transform(distances_pairwise_list)
    for i in range(len(X)):
        for j in range(len(X)):
            distances_pairwise[i][j] = distances_pairwise_list[i*len(X) + j]

    pickle.dump((distances, distances_pairwise), open("main_trajectory_features_"+dataset+".pkl", "rb"))

def distance_function(t1, t2):
    trip1 = np.array(X[int(t1[0])])
    trip2 = np.array(X[int(t2[0])])
    dist = distances_pairwise[int(t1[0])][int(t2[0])]
    start_dx = trip1[0][0] - trip2[0][0]
    end_dx = trip1[-1][0] - trip2[-1][0]
    start_dy = trip1[0][1] - trip2[0][1]
    end_dy = trip1[-1][1] - trip2[-1][1]
    d_slant = (slant(trip1) - slant(trip2))/2
    d_dist = distances[int(t1[0])] - distances[int(t2[0])]

    s = dist**2 + start_dx**2 + end_dx**2 + start_dy**2 + end_dy**2 + d_slant**2 + d_dist**2
    return sqrt(s)

dbscan = DBSCAN(eps=0.2, metric=distance_function)

print("Clustering...")
labels = dbscan.fit_predict(np.array(range(len(X))).reshape((-1, 1)))
y_pred = [0 if x > -1 else 1 for x in labels]

print(accuracy_score(y, y_pred))
print(f1_score(y, y_pred, average="macro"))
print(confusion_matrix(y, y_pred))