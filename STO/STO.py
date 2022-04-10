import datetime
import pytz
import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt

class STO:
    def __init__(self, trajectories, timebin_duration, weeks_before_and_after):
        self.all_trajectories = trajectories
        self.T = timebin_duration
        self.W = weeks_before_and_after
        self.all_transitions_indexed = dict()
        self.transitions_4features = dict()

    def getUnixDay(self, t):
        return (datetime.datetime.fromtimestamp(t) - datetime.datetime(1970, 1, 1, 0, 0, 0)).days
    
    def mean_transition_time(self, t1, t2):
        mean_timestamp = (t1 + t2) / 2
        return mean_timestamp

    def timebin_of_day(self, t):
        dt = datetime.datetime.fromtimestamp(t)
        total_seconds = 3600*dt.hour + 60*dt.minute + dt.second
        return total_seconds%self.T

    def index_transitions(self):
        all_transitions_indexed = dict()
        for i in range(len(self.all_trajectories)):
            for j in range(1, len(self.all_trajectories[i])):
                transition = [self.all_trajectories[i][j-1], self.all_trajectories[i][j]]
                time_of_transition = self.mean_transition_time(transition[0][1], transition[1][1])
                day_of_transition = self.getUnixDay(time_of_transition)
                timebin = self.timebin_of_day(time_of_transition)
                transition_str = transition[0][0] + "->" + transition[1][0]
                
                if day_of_transition in all_transitions_indexed:
                    if timebin in all_transitions_indexed[day_of_transition]:
                        all_transitions_indexed[day_of_transition][timebin].append(transition_str)
                    else:
                        all_transitions_indexed[day_of_transition][timebin] = [transition_str]
                else:
                    all_transitions_indexed[day_of_transition] = {timebin: [transition_str]}
        self.all_transitions_indexed = all_transitions_indexed           

    def count_incoming(self, x, arr):
        count = 0
        for t in arr:
            if t.split("->")[1] == x:
                count += 1
        return count

    def count_outgoing(self, x, arr):
        count = 0
        for t in arr:
            if t.split("->")[0] == x:
                count += 1
        return count

    def calculate_3features(self, transitions):
        feature1 = dict()
        feature2 = dict()
        feature3 = dict()
        aggregation = dict()

        for t in transitions:
            if t in feature1:
                feature1[t] += 1
            else:
                feature1[t] = 1
        for t in transitions:
            split_str = t.split("->")
            if split_str[1] in feature2:
                feature2[split_str[1]] += 1
            else:
                feature2[split_str[1]] = 1
            if split_str[0] in feature3:
                feature3[split_str[0]] += 1
            else:
                feature3[split_str[0]] = 1

        for t in feature1:
            split_str = t.split("->")
            agg = [feature1[t], feature1[t]/feature2[split_str[1]], feature1[t]/feature3[split_str[0]]]
            aggregation[t] = agg

        return aggregation

    def euclidean_diff(self, x, y):
        s = 0.0
        for i in range(len(x)):
            s += (x[i] - y[i])**2
        return s**0.5

    def calculate_min_distort(self, x, array):
        return min([self.euclidean_diff(x, y) for y in array])
    
    def calculate_features(self):
        transitions_4features = self.all_transitions_indexed.copy()
        for d in transitions_4features:
            for h in transitions_4features[d]:
                t_dict = dict()
                for t in transitions_4features[d][h]:
                    t_dict[t] = []
                transitions_4features[d][h] = t_dict
        for i in range(len(self.all_transitions_indexed.keys())):
            d = list(self.all_transitions_indexed.keys())[i]
            for h in self.all_transitions_indexed[d]:
                _3features = self.calculate_3features(self.all_transitions_indexed[d][h])
                _3features_for_window = [_3features]
                for w in range(-self.W, self.W+1):
                    if i + w >= 0 and w != 0 and i + w < len(self.all_transitions_indexed.keys()) and i+w in self.all_transitions_indexed and h in self.all_transitions_indexed[i+w]:
                        _3features_for_window.append(self.calculate_3features(self.all_transitions_indexed[i+w][h]))
                for transition in _3features:
                    minDistort = self.calculate_min_distort(_3features[transition], [_3f[transition] for _3f in _3features_for_window])
                    transitions_4features[d][h][transition] = _3features[transition] + [minDistort]
        self.transitions_4features =  transitions_4features

    def outlier_transition_detection(self):
        for d in self.transitions_4features:
            for h in self.transitions_4features[d]:
                X = list(self.transitions_4features[d][h].values())
                print(X[0])
                S = np.array([x[:3] for x in X])
                T = [[x[3]] for x in X]

                # Covariance matrix
                covariance  = np.cov(S , rowvar=False)

                # Covariance matrix power of -1
                covariance_pm1 = np.linalg.matrix_power(covariance, -1)

                # Center point
                centerpoint = np.mean(S , axis=0)

                distances = []
                for val in S:
                    p1 = val
                    p2 = centerpoint
                    distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
                    distances.append(distance)
                distances = np.array(distances)

                # Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
                cutoff = chi2.ppf(0.95, S.shape[1])

                # Index of outliers
                outlierIndexes = np.where(distances > cutoff)
                print(len(S), len(outlierIndexes))


    def fit(self):
        self.index_transitions()
        self.calculate_features()
        self.outlier_transition_detection()

    def predict(self, X):
        labels = []
        
        return labels

