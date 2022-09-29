from matplotlib import pyplot as plt
from math import log

grid_lengths_5 = {1: 8411, 2: 3528, 3: 1573, 4: 879, 5: 721, 7: 244, 6: 219, 9: 125, 11: 100, 8: 91, 13: 50, 10: 46, 12: 30, 15: 27, 17: 27, 14: 18, 16: 13, 23: 11, 19: 8, 18: 6, 21: 5, 22: 4, 28: 4, 41: 2, 20: 2, 39: 2, 33: 2, 26: 2, 86: 1, 51: 1, 49: 1, 29: 1, 30: 1, 24: 1, 43: 1, 96: 1, 25: 1, 115: 1, 73: 1, 69: 1, 99: 1, 60: 1, 62: 1, 77: 1, 95: 1, 171: 1, 70: 1, 32: 1}
grid_lengths_8 = {1: 5201, 2: 2790, 3: 2101, 4: 1428, 5: 1238, 7: 760, 6: 663, 9: 362, 8: 323, 11: 226, 10: 179, 13: 168, 12: 111, 15: 103, 14: 75, 17: 73, 16: 58, 19: 42, 18: 40, 21: 34, 22: 24, 20: 21, 25: 21, 23: 18, 29: 12, 27: 11, 26: 9, 33: 9, 24: 7, 32: 6, 31: 6, 36: 5, 35: 5, 28: 5, 37: 4, 43: 3, 30: 3, 34: 3, 38: 3, 55: 2, 60: 2, 42: 2, 45: 2, 39: 2, 61: 1, 81: 1, 49: 1, 40: 1, 59: 1, 41: 1, 47: 1, 53: 1, 66: 1, 67: 1}
grid_lengths_10 = {1: 4283, 2: 2198, 3: 2065, 4: 1750, 5: 1392, 7: 910, 6: 835, 9: 477, 8: 454, 11: 278, 10: 237, 13: 187, 12: 152, 15: 119, 17: 108, 14: 102, 16: 86, 19: 67, 18: 58, 21: 44, 20: 41, 23: 40, 25: 39, 22: 31, 27: 23, 24: 22, 26: 18, 28: 15, 32: 14, 29: 12, 35: 11, 34: 10, 31: 9, 33: 8, 37: 6, 43: 6, 39: 6, 30: 5, 42: 4, 81: 3, 47: 3, 41: 3, 40: 3, 65: 3, 60: 2, 38: 2, 45: 2, 192: 2, 54: 2, 36: 2, 57: 1, 61: 1, 88: 1, 51: 1, 59: 1, 68: 1, 56: 1, 151: 1, 46: 1, 55: 1, 49: 1, 97: 1, 115: 1, 74: 1, 100: 1, 66: 1, 116: 1, 72: 1, 178: 1, 71: 1, 67: 1}

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
plot_bar(grid_lengths_8, 8)
plt.subplot(235)
plot_bar(grid_lengths_8, 8, True)

plt.subplot(233)
plot_bar(grid_lengths_10, 10)
plt.subplot(236)
plot_bar(grid_lengths_10, 10, True)

plt.show()