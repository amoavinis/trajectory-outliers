from CustomScaler import Scaler
from CreateSearchSpaces import Preprocessor
from TOP import TOPClassifier
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

dataset = "geolife"
if len(sys.argv) > 1:
    dataset = sys.argv[1]
data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
X = [d[0] for d in data]
y = [d[1] for d in data]

grid_scale = 20
scaler = Scaler()
points = []
for x in X:
    points.extend(x)
scaler.fit(points)
X = [scaler.trajectory_to_grid(scaler.transform_trajectory(x), grid_scale) for x in X]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

minSup = 5
seqGap = 4
preprocessor = Preprocessor(x_train, minSup, seqGap)

print("Preprocessing started")
preprocessor.preprocess()
print("Preprocessing ended")

top = TOPClassifier(preprocessor.all_trajectories, preprocessor.search_spaces,
          preprocessor.freq_events, minSup, seqGap)
print("Fitting...")
top.fit()
print("Fitting complete.")

freqPatterns = top.freq_patterns
from collections import Counter
print(Counter([len(cf) for cf in freqPatterns]))

print(freqPatterns[:10])

y_pred_train = top.predict(x_train)
y_pred_test = top.predict(x_test)

print(f1_score(y_train, y_pred_train, average='macro'))
print(accuracy_score(y_train, y_pred_train))
print(f1_score(y_test, y_pred_test, average='macro'))
print(accuracy_score(y_test, y_pred_test))
