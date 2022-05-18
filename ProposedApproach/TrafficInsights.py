from matplotlib import pyplot as plt
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample, seed
seed(1997)

data = pickle.load(open("trajectory_features_labeled.pkl", "rb"))

X = [d[0] for d in data]
paths = [d[1] for d in data]
y = np.array([d[2] for d in data])

knn_model = NearestNeighbors(n_neighbors=20)
knn_model.fit(X)
_, neighbors = knn_model.kneighbors(X)
outlier_indices = [i for i,x in enumerate(y) if x==1]
sampled = sample(outlier_indices, 5)

def plot_outlier(x, neighbors_dict):
    plt.plot([p[0] for p in x], [p[1] for p in x], color='r')
    for n in neighbors_dict:
        plt.plot([p[0] for p in neighbors_dict[n][0]], [p[1] for p in neighbors_dict[n][0]], color='b' if neighbors_dict[n][1]==0 else 'r')
    plt.show()

for id in sampled:
    neighbors_dict = {}
    for neighbor in neighbors[id]:
        neighbors_dict[neighbor] = [paths[neighbor], y[neighbor]]
    plot_outlier(paths[id], neighbors_dict)
