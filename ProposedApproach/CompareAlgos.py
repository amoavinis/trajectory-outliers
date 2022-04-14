import pickle
import argparse
import numpy as np
#import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from Utils import *
#import pandas as pd
#from CustomKMeans import KMeans
from sklearn.cluster import DBSCAN
from minisom import MiniSom
from random import sample

parser = argparse.ArgumentParser(description="Compare performance of clustering algorithms.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
args = parser.parse_args()

dataset = args.dataset

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
data = sample(data, 2000)
X = [[p[:2] for p in d[0]] for d in data]
y = [d[1] for d in data]

def trim_none(traj):
    trimmed = []
    for p in traj:
        if p != [None, None]:
            trimmed.append(p)
        else:
            break
    return trimmed

def distance_function(t1, t2):
    t1 = X[int(t1[0])]
    t2 = X[int(t2[0])]
    dist = average_distance_of_trips(t1, t2, False)
    start_dx = t1[0][0] - t2[0][0]
    end_dx = t1[-1][0] - t2[-1][0]
    start_dy = t1[0][1] - t2[0][1]
    end_dy = t1[-1][1] - t2[-1][1]
    d_slant = (slant(t1) - slant(t2))/2
    #d_dist = distance_of_trajectory(t1) - distance_of_trajectory(t2)

    s = dist**2 + start_dx**2 + end_dx**2 + start_dy**2 + end_dy**2 + d_slant**2# + d_dist**2
    return sqrt(s)

dbscan = DBSCAN(eps=0.2, metric=distance_function)

#max_len = max([len(x) for x in X])
#X_padded = []
#for x in X:
#	xp = x + [[None, None]]*(max_len-len(x))
#	X_padded.append(xp)

labels = dbscan.fit_predict(np.array(range(len(X))).reshape((-1, 1)))
y_pred = [0 if x > -1 else 1 for x in labels]

print(accuracy_score(y, y_pred))
print(f1_score(y, y_pred, average="macro"))
print(confusion_matrix(y, y_pred))