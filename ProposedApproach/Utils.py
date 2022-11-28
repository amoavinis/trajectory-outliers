import numpy as np
import geopy.distance as distance


def average_length_of_sequences(sequences):
    s = sum([len(x) for x in sequences])
    avg = s / len(sequences)
    return round(avg, 2)


def distance_of_trajectory(traj, isCoordinates=False):
    dist = 0.0
    for i in range(len(traj) - 1):
        dist += np.linalg.norm(traj[i]-traj[i+1]) if isCoordinates == False else distance.great_circle(
            traj[i][::-1], traj[i+1][::-1]).meters
    return dist
