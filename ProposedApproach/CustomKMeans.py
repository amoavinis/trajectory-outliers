import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class CustomKMeans:
    def __init__(self, K):
        self.K = K
        self.model = KMeans(n_clusters=K)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        labels = self.model.predict(X)
        centroids = self.model.cluster_centers_

        outlyingness = []
        for i, x in enumerate(X):
            outlyingness.append([np.linalg.norm(x - centroids[labels[i]])])

        outlyingness = MinMaxScaler((0, 1)).fit_transform(outlyingness).reshape(-1)

        max_outlyingness = np.max(outlyingness)
        min_outlyingness = np.min(outlyingness)
        mean_outlyingness = np.average(outlyingness)

        threshold = max_outlyingness - abs(
            (mean_outlyingness - (max_outlyingness + min_outlyingness) / 2))

        transformed_labels = []
        for i, o in enumerate(outlyingness):
            if o < threshold:
                transformed_labels.append(labels[i])
            else:
                transformed_labels.append(-1)
        return transformed_labels
