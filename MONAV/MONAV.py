import os
import geopy.distance
import networkx as nx
import osmnx as ox
import tqdm
ox.config(use_cache=True, log_console=False)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

all_data = []

#DATA_PREFIX = "release/taxi_log_2008_by_id/"
DATA_PREFIX = "Datasets/Geolife Trajectories 1.3/Data/"

def process_file(f):
    file = open(f, 'r')
    lines = file.readlines()[6:]

    result = []

    for line in lines:
        split_line = line.split(",")
        latitude = float(split_line[0])
        longitude = float(split_line[1])
        result.append((latitude, longitude))

    return result

for i in os.listdir(DATA_PREFIX)[:20]:
    for j in os.listdir(DATA_PREFIX+i+'/Trajectory/'):
        all_data.append(process_file(DATA_PREFIX+i+'/Trajectory/'+j))

#print(len(all_data))
# get a graph
G = ox.graph_from_place('Beijing, China', network_type='drive')

# impute missing edge speed and add travel times
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

def trajectory_distance(traj):
    dist = 0.0
    for i in range(len(traj) - 1):
        dist += geopy.distance.distance(traj[i], traj[i+1]).kilometers
    return dist

def shortest_path(G, start, finish):
    # calculate shortest path minimizing travel time
    orig_node = ox.get_nearest_node(G, start)
    target_node = ox.get_nearest_node(G, finish)
    try:
        length = nx.shortest_path_length(G=G, source=orig_node, target=target_node, weight='length')
        return length
    except Exception:
        return 0
    

dists = []
for trajectory in tqdm.tqdm(all_data):
    dist = trajectory_distance(trajectory)
    shortest = shortest_path(G, trajectory[0], trajectory[-1]) / 1000
    if shortest > 0:
        print(dist/shortest)
    dists.append(dist)
plt.hist(dists, 10)
plt.show()
