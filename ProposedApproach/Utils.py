import geopy.distance
from math import sqrt

def euclidean(A, B):
    dist = [(a - b)**2 for a, b in zip(A, B)]
    dist = sqrt(sum(dist))
    return dist

def average_distance_of_trips(t1, t2, geo_coords=False):
    D = []
    for p1 in t1:
        for p2 in t2:
            d = distance_of_line([p1, p2]) if geo_coords else euclidean(p1, p2)
            D.append(d)
    return sum(D)/len(D)

def distance_of_line(transition):
    return geopy.distance.great_circle(list(reversed(transition[0])), list(reversed(transition[1]))).meters

def distance_of_trajectory(traj, geo_coords=False):
    dist = 0.0
    for i in range(len(traj) - 1):
        if geo_coords:
            dist += distance_of_line([traj[i], traj[i+1]])
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

''' def speeds_in_traj(traj):
    speeds = []
    for i in range(len(traj)-1):
        dx = distance_of_line([traj[i][:2], traj[i+1][:2]])
        dt = traj[i+1][2] - traj[i][2]
        if dt > 0:
            speeds.append(dx/dt)
    return {
        'min_speed': min(speeds),
        'max_speed': max(speeds),
        'avg_speed': sum(speeds)/len(speeds)
    } '''