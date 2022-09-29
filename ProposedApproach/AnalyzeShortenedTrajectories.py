import pickle
import numpy as np
from CustomScaler import Scaler
from SimplifyTrajectories import Simplifier
from Utils import distance_of_trajectory, average_length_of_sequences

def average_distance(X):
    s = sum([distance_of_trajectory(x) for x in X])
    return s/len(X)

def preprocess(X):
    scaler = Scaler()
    points = []
    for x in X:
        points.extend(x)
    scaler.fit(points)
    X = [scaler.transform_trajectory(x) for x in X]
    
    return X, scaler

def analyze_grid_paths(X, scaler, grid_scale):    
    X_grid = [np.array(scaler.trajectory_to_grid(x, grid_scale)) for x in X]
    print("Average length of sequences:", average_length_of_sequences(X_grid))

def analyze_angle_paths(X, threshold):
    avg_dist = average_distance(X)

    simplifier = Simplifier(threshold)
    X_angle = simplifier.simplify(X)
    avg_dist_angle = average_distance(X_angle)
    print("Angle distance ratio:", avg_dist_angle/avg_dist)
    print("Average length of sequences:", average_length_of_sequences(X_angle))

if __name__ == '__main__':
    data = pickle.load(open("trajectories_labeled_geolife.pkl", "rb"))
    X = [np.array(d[0]) for d in data]

    X, scaler = preprocess(X)

    analyze_grid_paths(X, scaler, 10)
    analyze_grid_paths(X, scaler, 20)
    analyze_grid_paths(X, scaler, 30)
    analyze_grid_paths(X, scaler, 40)
    analyze_grid_paths(X, scaler, 50)
    analyze_grid_paths(X, scaler, 60)
    #analyze_angle_paths(X, 15)
    #analyze_angle_paths(X, 30)
    #analyze_angle_paths(X, 45)
    #analyze_angle_paths(X, 60)
    #analyze_angle_paths(X, 90)