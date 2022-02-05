from sklearn.ensemble import RandomForestClassifier

class EnsembleModel:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, x_train, y_train):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        self.model.fit(x_train, y_train)

    def get_most_uncertain_example(self, X):
        probabilities = self.model.predict_proba(X)
        min_index = -1
        min_diff = 1
        for i in range(len(probabilities)):
            if abs(probabilities[i][0] - probabilities[i][1]) <= min_diff:
                min_diff = abs(probabilities[i][0] - probabilities[i][1])
                min_index = i
        if (min_index >= 0):
            return X[min_index]
        else:
            return None
