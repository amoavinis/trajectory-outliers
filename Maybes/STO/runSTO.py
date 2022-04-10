from CustomScaler import Scaler
from STO import STO
import pickle
import argparse
from sklearn.metrics import f1_score, accuracy_score

parser = argparse.ArgumentParser(description="Train and predict using the STO model.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
parser.add_argument("--gridScale", help="The number of grid cells per dimension", default="10")
parser.add_argument("--timebin", help="The duration of a timebin in seconds", default="3600")
parser.add_argument("--W", help="The window size", default="3")
args = parser.parse_args()

dataset = args.dataset
grid_scale = int(args.gridScale)
timebin = int(args.timebin)
W = int(args.W)

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
X = [d[0] for d in data]
y = [d[1] for d in data]

scaler = Scaler()
points = []
for x in X:
    points.extend(x)
scaler.fit(points)

X = [scaler.trajectory_to_grid(scaler.transform_trajectory(x), grid_scale) for x in X]

sto = STO(X, timebin, W)
sto.fit()
