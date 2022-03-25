import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
def calculate_area(polygon):
    # using the shoelace method
    A = 0
    for i in range(len(polygon)):
        a = polygon[i]
        b = None
        if i == len(polygon) - 1:
            b = polygon[0]
        else:
            b = polygon[i+1]
        A += np.linalg.det(np.array([a, b]))
    return A/2

x = np.array([[0, 0], [1, 0], [0, 0.66], [0, 1.33], [1, 2], [0, 2], [0, 1.33], [-0.5, 1], [0, 0.66]])

for i in range(10):
    np.random.shuffle(x)
    #print(calculate_area(x))
    print(x)
    print(Polygon(zip(x[:, 0], x[:, 1])).area)
#plt.plot(x[:, 0], x[:, 1])
#plt.show()