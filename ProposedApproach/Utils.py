from hausdorff import hausdorff_distance
import numpy as np

def hausdorff_dist(t1, t2):
    return hausdorff_distance(t1, t2)

def average_distance_of_trips(t1, t2):
    D = []
    for p1 in t1:
        for p2 in t2:
            D.append(np.linalg.norm(p1-p2))
    return sum(D)/len(D)

def distance_of_trajectory(traj):
    dist = 0.0
    for i in range(len(traj) - 1):
        dist += np.linalg.norm(traj[i]-traj[i+1])
    return dist

def slant(traj):
    sd_len = np.linalg.norm(traj[-1]-traj[0])
    dx = traj[-1][0] - traj[0][0]
    if sd_len > 0:
        sine = dx / sd_len
        return sine
    else:
        return 0
