import numpy as np

def average_length_of_sequences(sequences):
    s = sum([len(x) for x in sequences])
    avg = s / len(sequences)
    return round(avg, 2)

def distance_of_trajectory(traj):
    dist = 0.0
    for i in range(len(traj) - 1):
        dist += np.linalg.norm(traj[i]-traj[i+1])
    return dist
