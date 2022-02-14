class Scaler:

    def __init__(self):
        self.min = [100000, 100000]
        self.max = [-100000, -100000]

    def fit(self, X):
        for x in X:
            if x[0] <= self.min[0]:
                self.min[0] = x[0]
            elif x[0] >= self.max[0]:
                self.max[0] = x[0]

            if x[1] <= self.min[1]:
                self.min[1] = x[1]
            elif x[1] >= self.max[1]:
                self.max[1] = x[1]

    def transform(self, X):
        return [[(x[0] - self.min[0]) / (self.max[0] - self.min[0]),
                 (x[1] - self.min[1]) / (self.max[1] - self.min[1])]
                for x in X]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
