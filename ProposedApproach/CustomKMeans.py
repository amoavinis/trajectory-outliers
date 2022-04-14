from Utils import *
import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, X):
        self.X = X

    def calculate_error(self, t1, t2):
        t1 = self.X[t1[0]]
        t2 = self.X[t2[0]]
        dist = average_distance_of_trips(t1, t2, False)
        start_dx = t1[0][0] - t2[0][0]
        end_dx = t1[-1][0] - t2[-1][0]
        start_dy = t1[0][1] - t2[0][1]
        end_dy = t1[-1][1] - t2[-1][1]
        d_slant = (slant(t1) - slant(t2))/2
        d_dist = distance_of_trajectory(t1) - distance_of_trajectory(t2)

        s = dist**2 + start_dx**2 + end_dx**2 + start_dy**2 + end_dy**2 + d_slant**2 + d_dist**2
        return sqrt(s)

    def assign_centroid(self, data, centroids):
        '''
        Receives a dataframe of data and centroids and returns a list assigning each observation a centroid.
        data: a dataframe with all data that will be used.
        centroids: a dataframe with the centroids. For assignment the index will be used.
        '''

        n_observations = data.shape[0]
        centroid_assign = []
        centroid_errors = []
        k = centroids.shape[0]


        for observation in range(n_observations):

            # Calculate the errror
            errors = np.array([])
            for centroid in range(k):
                error = self.calculate_error(centroids.iloc[centroid, :2], data.iloc[observation,:2])
                errors = np.append(errors, error)

            # Calculate closest centroid & error 
            closest_centroid =  np.where(errors == np.amin(errors))[0].tolist()[0]
            centroid_error = np.amin(errors)

            # Assign values to lists
            centroid_assign.append(closest_centroid)
            centroid_errors.append(centroid_error)

        return (centroid_assign,centroid_errors)

    def initialize_centroids(self, k, data):
        
        n_dims = data.shape[1]
        centroid_min = data.min().min()
        centroid_max = data.max().max()
        centroids = []

        for centroid in range(k):
            centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
            centroids.append(centroid)

        centroids = pd.DataFrame(centroids, columns = data.columns)

        return centroids

    def kmeans(self, data, k):
        # Initialize centroids and error
        centroids = self.initialize_centroids(k, data)
        error = []
        compr = True
        i = 0

        while(compr):
            # Obtain centroids and error
            data['centroid'], iter_error = self.assign_centroid(data,centroids)
            error.append(sum(iter_error))
            # Recalculate centroids
            centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)

            # Check if the error has decreased
            if(len(error)<2):
                compr = True
            else:
                if(round(error[i],3) !=  round(error[i-1],3)):
                    compr = True
                else:
                    compr = False
            i = i + 1 

        data['centroid'], iter_error = self.assign_centroid(data,centroids)
        centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)
        return (data['centroid'], iter_error, centroids)