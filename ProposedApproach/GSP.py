import os
import pickle
import numpy as np
from gsppy.gsp import GSP
from hausdorff import hausdorff_distance

class GSPModule:
    def __init__(self):
        self.freq_subsequences = []

    def stringify_grid_trajectory(self, trajectory):
        return [f"{p}" for p in trajectory]

    def intify_grid_trajectories(self, X):
        intified = []
        for x in X:
            x_int = []
            for p in x:
                l = p.replace("[", "").replace("]", "").split()
                x_int.append([int(l[0]), int(l[1])])
            intified.append(np.array(x_int))

        return intified

    def find_frequent_subsequences(self, X, t, read_pkl=False):
        frequent = []
        if read_pkl and os.path.exists(f"freq_subs_{str(t).replace('.', '')}.pkl"):
            frequent = pickle.load(open(f"freq_subs_{str(t).replace('.', '')}.pkl", "rb"))   
        else:
            stringified = [self.stringify_grid_trajectory(x) for x in X]
            output = GSP(stringified).search(t)
            for d in output:
                frequent.extend(d.keys())
            pickle.dump(frequent, open(f"freq_subs_{str(t).replace('.', '')}.pkl", "wb"))
        intified_freq = self.intify_grid_trajectories(frequent)
        
        self.freq_subsequences = intified_freq

    def deviation_from_frequent(self, X):
        distances = np.zeros((len(X), len(self.freq_subsequences)))
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i, j] = hausdorff_distance(np.array(X[i]), self.freq_subsequences[j])

        mins = np.amin(distances, axis=1)

        return mins

