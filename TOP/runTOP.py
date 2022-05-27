from CustomScaler import Scaler
from CreateSearchSpaces import Preprocessor
from TOP import TOPClassifier
import pickle
import argparse
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from time import time

parser = argparse.ArgumentParser(description="Train and predict using the TOP model.")
parser.add_argument("--dataset", help="Specify the dataset to use", default="geolife")
parser.add_argument("--G", help="The number of grid cells per dimension", default="5")
parser.add_argument("--minSup", help="The minimum support for a CF pattern", default="5")
parser.add_argument("--seqGap", help="The seqGap parameter", default="2")
args = parser.parse_args()

dataset = args.dataset
grid_scale = int(args.G)
minSup = int(args.minSup)
seqGap = int(args.seqGap)

data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
X = [[p[:2] for p in d[0]] for d in data]
y = [d[1] for d in data]

scaler = Scaler()
points = []
for x in X:
    points.extend(x)
scaler.fit(points)
X = [scaler.trajectory_to_grid(scaler.transform_trajectory(x), grid_scale) for x in X]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=10)

preprocessor = Preprocessor(x_train, minSup, seqGap)

t = time()
print("Preprocessing started")
preprocessor.preprocess()
print("Preprocessing ended")

top = TOPClassifier(preprocessor.all_trajectories, preprocessor.search_spaces,
          preprocessor.freq_events, minSup, seqGap)
print("Fitting...")
top.fit()
print("Fitting complete.")
print("Fitting time:", round(time() - t, 2), "seconds")

#freqPatterns = top.freq_patterns
#print(Counter([len(cf) for cf in freqPatterns]))

y_pred_train = top.predict(x_train)
y_pred_test = top.predict(x_test)

print("Training accuracy:", accuracy_score(y_train, y_pred_train))
print("Training F1 score:", f1_score(y_train, y_pred_train, average='macro'))
print("Testing accuracy:", accuracy_score(y_test, y_pred_test))
print("Testing F1 score:", f1_score(y_test, y_pred_test, average='macro'))
print(confusion_matrix(y_test, y_pred_test))