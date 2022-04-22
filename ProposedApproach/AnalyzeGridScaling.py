from matplotlib import pyplot as plt
from math import log

grid_lengths_10 = {
    1: 11425,
    2: 3032,
    3: 1668,
    4: 348,
    5: 308,
    7: 144,
    6: 112,
    9: 68,
    8: 58,
    11: 40,
    10: 32,
    12: 15,
    13: 15,
    15: 9,
    19: 6,
    14: 6,
    17: 6,
    16: 4,
    25: 1,
    24: 1,
    23: 1,
    18: 1,
    33: 1,
    20: 1,
    21: 1,
    26: 1
}
grid_lengths_20 = {
    1: 10059,
    2: 2957,
    3: 2043,
    4: 687,
    5: 539,
    7: 241,
    6: 198,
    9: 125,
    8: 97,
    11: 82,
    10: 56,
    13: 39,
    15: 36,
    12: 28,
    17: 27,
    14: 18,
    16: 13,
    18: 12,
    19: 12,
    21: 8,
    20: 7,
    22: 5,
    29: 5,
    24: 3,
    26: 3,
    23: 2,
    27: 1,
    33: 1
}
grid_lengths_30 = {
    1: 9148,
    2: 3565,
    3: 1741,
    4: 992,
    5: 577,
    7: 273,
    6: 216,
    9: 149,
    8: 117,
    11: 103,
    10: 63,
    13: 63,
    12: 35,
    15: 34,
    14: 31,
    17: 23,
    16: 21,
    20: 20,
    19: 18,
    21: 14,
    23: 13,
    18: 12,
    27: 10,
    26: 9,
    22: 8,
    25: 8,
    24: 6,
    29: 6,
    32: 4,
    31: 4,
    30: 3,
    28: 3,
    33: 3,
    39: 3,
    35: 2,
    44: 2,
    37: 1,
    38: 1,
    36: 1,
    42: 1,
    61: 1
}


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
plot_bar(grid_lengths_10, 10)
plt.subplot(234)
plot_bar(grid_lengths_10, 10, True)

plt.subplot(232)
plot_bar(grid_lengths_20, 20)
plt.subplot(235)
plot_bar(grid_lengths_20, 20, True)

plt.subplot(233)
plot_bar(grid_lengths_30, 30)
plt.subplot(236)
plot_bar(grid_lengths_30, 30, True)

plt.show()