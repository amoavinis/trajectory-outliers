from matplotlib import pyplot as plt
from math import log

grid_lengths_5 = {1: 13898, 2: 1471, 3: 522, 4: 282, 5: 208, 6: 156, 7: 129, 8: 90, 9: 82, 10: 50, 11: 43, 13: 32, 12: 19, 15: 14, 17: 10, 14: 7, 19: 5, 16: 2, 23: 2, 18: 1, 21: 1, 31: 1, 33: 1, 29: 1}
grid_lengths_10 = {1: 11219, 2: 2250, 3: 1427, 4: 562, 5: 436, 6: 251, 7: 246, 9: 140, 8: 139, 11: 87, 10: 67, 13: 57, 12: 37, 15: 26, 17: 19, 19: 15, 14: 11, 16: 8, 18: 6, 21: 5, 23: 5, 29: 4, 20: 3, 25: 2, 30: 1, 24: 1, 31: 1, 35: 1, 27: 1}
grid_lengths_20 = {1: 6774, 2: 3172, 3: 2060, 5: 1410, 4: 965, 7: 629, 6: 399, 9: 323, 11: 230, 8: 187, 13: 145, 10: 118, 15: 106, 12: 84, 17: 67, 14: 53, 19: 45, 23: 44, 16: 31, 21: 30, 18: 21, 20: 18, 25: 17, 22: 13, 31: 10, 35: 9, 27: 9, 28: 7, 26: 7, 29: 6, 33: 5, 24: 4, 39: 4, 34: 3, 30: 3, 46: 2, 45: 2, 36: 2, 32: 2, 41: 1, 50: 1, 43: 1, 38: 1, 55: 1, 42: 1, 52: 1, 47: 1, 48: 1, 37: 1, 51: 1}

def plot_bar(data_dict, G, log_values=False):
    x = data_dict.keys()
    y = data_dict.values()

    if log_values:
        y = [log(v) for v in y]

    plt.bar(x, y)
    plt.xlabel("Length of path in grid tiles, G="+str(G))
    if log_values:
        plt.ylabel("Log of number of paths")
    else:
        plt.ylabel("Number of paths")

plt.suptitle("Experimentation with grid size G")

plt.subplot(231)
plot_bar(grid_lengths_5, 5)
plt.subplot(234)
plot_bar(grid_lengths_5, 5, True)

plt.subplot(232)
plot_bar(grid_lengths_10, 10)
plt.subplot(235)
plot_bar(grid_lengths_10, 10, True)

plt.subplot(233)
plot_bar(grid_lengths_20, 20)
plt.subplot(236)
plot_bar(grid_lengths_20, 20, True)

plt.show()