import pickle
import numpy as np
from CustomScaler import Scaler
from SimplifyTrajectories import Simplifier
from Utils import distance_of_trajectory

def average_distance(X):
    s = sum([distance_of_trajectory(x) for x in X])
    return s/len(X)

def analyze_grid_paths(X):
    grid_scale = 50

    scaler = Scaler()
    points = []
    for x in X:
        points.extend(x)
    scaler.fit(points)
    X = [scaler.transform_trajectory(x) for x in X]
    avg_dist = average_distance(X)

    X_grid = [np.array(scaler.trajectory_to_grid(x, grid_scale)) for x in X]
    X_grid_new = []
    for x in X_grid:
        X_grid_new.append(np.array([[(p[0]+0.5)/grid_scale, (p[1]+0.5)/grid_scale] for p in x]))

    avg_dist_grid = average_distance(X_grid_new)
    print("Grid distance ratio:", avg_dist_grid/avg_dist)

def analyze_angle_paths(X):
    threshold = 150
    
    scaler = Scaler()
    points = []
    for x in X:
        points.extend(x)
    scaler.fit(points)
    X = [scaler.transform_trajectory(x) for x in X]
    avg_dist = average_distance(X)

    simplifier = Simplifier(threshold)
    X_angle = simplifier.simplify(X)
    avg_dist_angle = average_distance(X_angle)
    print("Angle distance ratio:", avg_dist_angle/avg_dist)


if __name__ == '__main__':
    data = pickle.load(open("trajectories_labeled_geolife.pkl", "rb"))
    X = [np.array(d[0]) for d in data]
    analyze_grid_paths(X)
    analyze_angle_paths(X)