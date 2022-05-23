from matplotlib import pyplot as plt
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample, seed
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
seed(2)

data = pickle.load(open("trajectory_features_labeled.pkl", "rb"))

X = [[d[0][0], d[0][1], d[0][2], d[0][3], d[0][5]] for d in data]
paths = [d[1] for d in data]
y = np.array([d[2] for d in data])

knn_model = NearestNeighbors(n_neighbors=51)
knn_model.fit(X)
_, neighbors = knn_model.kneighbors(X)
outlier_indices = [i for i,x in enumerate(y) if x==1]
sampled = sample(outlier_indices, 10)

def plot_outlier(x, neighbors_dict):
    plt.plot([p[0] for p in x], [p[1] for p in x], color='r')
    for n in neighbors_dict:
        plt.plot([p[0] for p in neighbors_dict[n][0]], [p[1] for p in neighbors_dict[n][0]], color='b' if neighbors_dict[n][1]==0 else 'r')
    plt.show()

for id in sampled:
    neighbors_dict = {}
    for neighbor in neighbors[id][1:]:
        neighbors_dict[neighbor] = [paths[neighbor], y[neighbor]]
    plot_outlier(paths[id], neighbors_dict)

def limits_of_cluster(cluster_points):
    minmax = np.zeros((cluster_points.shape[1], 2))
    for i in range(cluster_points.shape[1]):
        minmax[i, 0] = np.min(cluster_points[:, i])
        minmax[i, 1] = np.max(cluster_points[:, i])
    minmax[0] = 116.1 + minmax[0]*(116.599-116.1)
    minmax[1] = 39.655 + minmax[1]*(40.2957-39.655)
    minmax[2] = 116.1 + minmax[2]*(116.599-116.1)
    minmax[3] = 39.655 + minmax[3]*(40.2957-39.655)
    minmax[4] = minmax[4]*252807
    return minmax

def clustering(data, e):
    kmeans = KMeans(e)
    y_pred = kmeans.fit_predict(data)

    print(e, "-", silhouette_score(data, y_pred))
    
    labels = set(y_pred)
    for label in labels:
        if label >= 0:
            cluster_points = []
            for i in range(len(y_pred)):
                if y_pred[i] == label:
                    cluster_points.append(data[i])
            print(label, "-", limits_of_cluster(np.array(cluster_points)))
    
    class0 = [paths[i] for i in range(len(y_pred)) if y_pred[i] == 0]
    class1 = [paths[i] for i in range(len(y_pred)) if y_pred[i] == 1]

    sample_limit = 1500

    for x in sample(class0, sample_limit):
        plt.plot([p[0] for p in x], [p[1] for p in x], 'k')
    
    for x in sample(class1, sample_limit):
        plt.plot([p[0] for p in x], [p[1] for p in x], 'y')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

clustering(X, 2)
