import geopy.distance
from math import sqrt

def distance_of_transition(transition):
    return geopy.distance.great_circle(list(reversed(transition[0])), list(reversed(transition[1]))).meters

def distance_of_trajectory(traj):
    dist = 0.0
    for i in range(len(traj) - 1):
        dist += distance_of_transition([traj[i][:2], traj[i+1][:2]])
    return dist

def speeds_in_traj(traj):
    speeds = []
    for i in range(len(traj)-1):
        dx = distance_of_transition([traj[i][:2], traj[i+1][:2]])
        dt = traj[i+1][2] - traj[i][2]
        if dt > 0:
            speeds.append(dx/dt)
    return {
        'min_speed': min(speeds),
        'max_speed': max(speeds),
        'avg_speed': sum(speeds)/len(speeds)
    }

def slant(traj):
    sd_len = sqrt((traj[-1][1]-traj[0][1])**2 + (traj[-1][0]-traj[0][0])**2)
    dx = traj[-1][0] - traj[0][0]
    if sd_len > 0:
        sine = dx / sd_len
        return sine
    else:
        return 0