import pickle
from random import sample
from matplotlib import pyplot as plt

data = pickle.load(open("trajectories_labeled_cyprus.pkl", "rb"))
X = [[p[:2] for p in d[0]] for d in data]
y = [d[1] for d in data]

inliers = [X[i] for i in range(len(X)) if y[i]==0]
outliers = [X[i] for i in range(len(X)) if y[i]==1]

sample_inliers = sample(inliers, 1000)
sample_outliers = sample(outliers, 200)

def plot_one(t, color):
    c1 = [s[0] for s in t]
    c2 = [s[1] for s in t]
    plt.plot(c1, c2, color=color)

def plot_group(X, color):
    for x in X:
        plot_one(x, color)

plot_group(sample_inliers, 'b')
plot_group(sample_outliers, 'r')

plt.show()