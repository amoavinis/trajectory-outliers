from CreateSearchSpaces import Preprocessor
from TOP import TOPClassifier
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

dataset = "geolife"
if len(sys.argv) > 1:
    dataset = sys.argv[1]
data_file = "trajectories_labeled_" + dataset + ".pkl"
data = pickle.load(open(data_file, "rb"))
X = [d[0] for d in data]
y = [d[1] for d in data]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

grid_scale = 20
minSup = 5
seqGap = 4
preprocessor = Preprocessor(x_train, grid_scale, minSup, seqGap)

print("Preprocessing started")
preprocessor.preprocess()
print("Preprocessing ended")

top = TOPClassifier(preprocessor.all_trajectories, preprocessor.search_spaces,
          preprocessor.freq_events, minSup, seqGap)
top.fit()

freqPatterns = top.freq_patterns

print(freqPatterns[:10])

y_pred_train = top.predict(x_train)
y_pred_test = top.predict(x_test)

print(f1_score(y_train, y_pred_train, average='macro'))
print(f1_score(y_test, y_pred_test, average='macro'))
