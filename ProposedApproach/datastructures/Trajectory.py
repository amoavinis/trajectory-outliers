from Point import Point

class Trajectory:
    def __init__(self, id, path, datetimes, label):
        self.id = id
        self.path = path
        self.label = label
        self.datetimes = datetimes
        self.slant = self.slant(self.path)
        self.distance = self.distance_of_trajectory(self.path)
        speeds_obj = self.speeds_in_traj(self.path)
        self.min_speed = speeds_obj['min_speed']
        self.avg_speed = speeds_obj['avg_speed']
        self.max_speed = speeds_obj['max_speed']

    def calc_slant(self, traj):
        if traj[-1][0]-traj[0][0] == 0:
            return 0
        else:
            return (traj[-1][1]-traj[0][1])/(traj[-1][0]-traj[0][0])

    def dist_of_transition(self, transition):
        return Point.distance(transition[0], transition[1]).meters

    def distance_of_trajectory(self, traj):
        dist = 0.0
        for i in range(len(traj) - 1):
            dist += self.dist_of_transition([traj[i], traj[i+1]])
        return dist

    def speeds_in_traj(self, traj):
        speeds = []
        for i in range(len(traj)-1):
            dx = self.dist_of_transition([traj[i], traj[i+1]])
            dt = traj[i+1][2] - traj[i][2]
            if dt > 0:
                speeds.append(dx/dt)
        return {
            'min_speed': min(speeds),
            'max_speed': max(speeds),
            'avg_speed': sum(speeds)/len(speeds)
        }
    