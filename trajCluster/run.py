from partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning
from point import Point
from cluster import line_segment_clustering, representative_trajectory_generation
import pickle
import argparse
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from time import time
import tqdm

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Train and predict using the TOP model.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
parser.add_argument("--theta", help="The theta parameter", default="5")
parser.add_argument("--epsilon", help="The epsilon parameter", default="5")
parser.add_argument("--seed", default="9999")
args = parser.parse_args()

dataset = args.dataset
theta = float(args.theta)
epsilon = float(args.epsilon)
seed = int(args.seed)

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))

data = data[:1000]

X = [[Point(p[1]*100, p[0]*100) for p in d[0]] for d in data]
y = [d[1] for d in data]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)

# Part 1: partition

all_segs = []
for i in tqdm.tqdm(range(len(X))):
    all_segs.extend(approximate_trajectory_partitioning(X[i], theta=theta, traj_id=i))

# Part 2: cluster

norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=3, epsilon=epsilon)
for k, v in remove_cluster.items():
    print("remove cluster: the cluster %d, the segment number %d" % (k, len(v)))

cluster_s_x, cluster_s_y = [], []
for k, v in norm_cluster.items():
    cluster_s_x.extend([s.start.x for s in v])
    cluster_s_x.extend([s.end.x for s in v])

    cluster_s_y.extend([s.start.y for s in v])
    cluster_s_y.extend([s.end.y for s in v])
    print("using cluster: the cluster %d, the segment number %d" % (k, len(v)))
