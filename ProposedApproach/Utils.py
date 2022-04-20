import geopy.distance
from math import sqrt
from hausdorff import hausdorff_distance
import numpy as np

def euclidean(A, B):
    dist = [(a - b)**2 for a, b in zip(A, B)]
    dist = sqrt(sum(dist))
    return dist

def hausdorff_dist(t1, t2):
    return hausdorff_distance(t1, t2)

def average_distance_of_trips(t1, t2):
    D = []
    for p1 in t1:
        for p2 in t2:
            D.append(euclidean(p1, p2))
    return sum(D)/len(D)

def distance_of_line(a, b):
    a = np.flip(a)
    b = np.flip(b)
    return geopy.distance.great_circle(a, b).meters

def distance_of_trajectory(traj, geo_coords=False):
    dist = 0.0
    for i in range(len(traj) - 1):
        if geo_coords:
            dist += distance_of_line(traj[i], traj[i+1])
        else:
            dist += sqrt((traj[i][0]-traj[i+1][0])**2 + (traj[i][1]-traj[i+1][1])**2)
    return dist

def slant(traj):
    sd_len = sqrt((traj[-1][1]-traj[0][1])**2 + (traj[-1][0]-traj[0][0])**2)
    dx = traj[-1][0] - traj[0][0]
    if sd_len > 0:
        sine = dx / sd_len
        return sine
    else:
        return 0
