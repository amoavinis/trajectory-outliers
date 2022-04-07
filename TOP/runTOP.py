from CustomScaler import Scaler
from CreateSearchSpaces import Preprocessor
from TOP import TOPClassifier
import pickle
import argparse
from sklearn.metrics import f1_score, accuracy_score
#from collections import Counter

parser = argparse.ArgumentParser(description="Train and predict using the TOP model.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
parser.add_argument("--gridScale", help="The number of grid cells per dimension", default="10")
parser.add_argument("--minSup", help="The minimum support for a CF pattern", default="5")
parser.add_argument("--seqGap", help="The seqGap parameter", default="2")
args = parser.parse_args()

dataset = args.dataset
grid_scale = int(args.gridScale)
minSup = int(args.minSup)
seqGap = int(args.seqGap)

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

preprocessor = Preprocessor(X, minSup, seqGap)

print("Preprocessing started")
preprocessor.preprocess()
print("Preprocessing ended")

top = TOPClassifier(preprocessor.all_trajectories, preprocessor.search_spaces,
          preprocessor.freq_events, minSup, seqGap)
print("Fitting...")
top.fit()
print("Fitting complete.")

#freqPatterns = top.freq_patterns
#print(Counter([len(cf) for cf in freqPatterns]))

y_pred = top.predict(X)

print("F1 score:", f1_score(y, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y, y_pred))
