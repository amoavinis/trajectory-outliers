import os
import pickle
import numpy as np
from gsppy.gsp import GSP
from hausdorff import hausdorff_distance

class GSPModule:
    def __init__(self):
        pass

    def stringify_grid_trajectory(self, trajectory):
        return [f"{p}" for p in trajectory]

    def find_frequent_subsequences(self, X, t):
        output = GSP(X).search(t)
        freq_subsequences = []
        for d in output:
            freq_subsequences.extend(d.keys())

        return freq_subsequences

    def intify_grid_trajectories(self, X):
        intified = []
        for x in X:
            x_int = []
            for p in x:
                l = p.replace("[", "").replace("]", "").split()
                x_int.append([int(l[0]), int(l[1])])
            intified.append(np.array(x_int))

        return intified

    def deviation_from_frequent(self, X, t):
        stringified = [self.stringify_grid_trajectory(x) for x in X]

        frequent = []
        if os.path.exists("freq_subs.pkl"):
            dicts = pickle.load(open("freq_subs.pkl", "rb"))
            for d in dicts:
                frequent.extend(d.keys())
        else:
            frequent = self.find_frequent_subsequences(stringified, t)
            pickle.dump(frequent, open("freq_subs.pkl", "wb"))
        intified_freq = self.intify_grid_trajectories(frequent)
        distances = np.zeros((len(X), len(frequent)))
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i, j] = hausdorff_distance(X[i], intified_freq[j])

        mins = np.amin(distances, axis=1)

        return mins

