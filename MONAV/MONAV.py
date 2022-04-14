import pickle
import geopy.distance
import networkx as nx
import osmnx as ox
import tqdm
#ox.config(use_cache=True, log_console=False)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
from sklearn.metrics import accuracy_score, f1_score


dataset = "geolife"
data_file = "trajectories_labeled_" + dataset + ".pkl"
all_data = pickle.load(open(data_file, "rb"))

X = [t[0] for t in all_data]
y = [t[1] for t in all_data]
X = [[p[:2][::-1] for p in t] for t in X]

# get a graph
G = ox.graph_from_place('Beijing, China', network_type='all')

# impute missing edge speed and add travel times
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

def trajectory_distance(traj):
    dist = 0.0
    for i in range(len(traj) - 1):
        dist += geopy.distance.great_circle(traj[i], traj[i+1]).meters
    return dist

def shortest_path(G, start, finish):
    start = [round(start[0], 6), round(start[1], 6)]
    finish = [round(finish[0], 6), round(finish[1], 6)]
    # calculate shortest path minimizing travel time
    orig_node = ox.get_nearest_node(G, start)
    target_node = ox.get_nearest_node(G, finish)
    try:
        length = nx.shortest_path_length(G=G, source=orig_node, target=target_node, weight="length")
        return length
    except Exception as e:
        print(e)
        return 0

dists = []
if os.path.exists("monav_dists_"+dataset+".pkl"):
    dists = pickle.load(open("monav_dists_"+dataset+".pkl", "rb"))
else:
    i = 0
    for trajectory in tqdm.tqdm(X):
        dist = trajectory_distance(trajectory)
        shortest = shortest_path(G, trajectory[0], trajectory[-1])
        ratio = 0
        if shortest > 0:
            ratio = dist/shortest
        dists.append((trajectory, ratio, y[i]))
        i += 1
    pickle.dump(dists, open("monav_dists_"+dataset+".pkl", "wb"))

threshold = 20

outputs = [d[1] for d in dists]
y_pred = [1 if x > threshold else 0 for x in outputs]

print("Accuracy score:", accuracy_score(y, y_pred))
print("F1 score:", f1_score(y, y_pred, average="macro")
