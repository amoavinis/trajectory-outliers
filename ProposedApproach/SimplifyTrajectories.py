import numpy as np
import matplotlib.pyplot as plt
class Simplifier:
    def __init__(self, threshold):
        self.threshold = threshold
        self.angles = []

    def to_polar(self, O, x):
        return np.array(x) - np.array(O)

    def angle(self, a, b):
        if a[0] == a[1] or b[0] == b[1]:
            return 0
        
        v1_u = a / np.linalg.norm(a)
        v2_u = b / np.linalg.norm(b)
        
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1, 1)))
    
    def simplify_trajectory(self, x):
        y = [x[0]]

        if len(x) < 3:
            return x
        
        for i in range(1, len(x)-2):
            ab = self.to_polar(x[i+1], x[i])
            bc = self.to_polar(x[i+1], x[i+2]) 
            angle = self.angle(ab, bc)
            self.angles.append(angle)
            if angle < self.threshold:
                y.append(x[i+1])
        y.append(x[-1])

        return y

    def simplify(self, X):
        res =  [self.simplify_trajectory(x) for x in X]
        #plt.hist(self.angles, bins=10)
        #plt.show()
        return res