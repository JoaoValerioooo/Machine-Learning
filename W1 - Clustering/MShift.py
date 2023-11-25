import numpy as np
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt


class MShift:

    def __init__(self, X):
        self.X = X.to_numpy()
        self.K = 2
        self.centers = np.empty(0)
        self.clusters = np.empty(0)
        self.random_seed = 1234

    def getCenters(self):
        return self.centers

    def getClusters(self):
        return self.clusters

    def applyMShift(self, k, bin_seeding=False, seed=1234):
        self.K = k

        mean_shift_clustering = MeanShift(max_iter=100, bin_seeding=bin_seeding)
        # Training the model
        mean_shift_model = mean_shift_clustering.fit(self.X)
        # Labels of the points to the centroids
        self.clusters = mean_shift_model.labels_
        # Corrdinates of the centroids
        self.centers = mean_shift_clustering.cluster_centers_

    def plotMeanShift(self, model, **kwargs):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        print(self.X)
        ax.scatter(self.X['sepallength'], self.X['sepalwidth'], self.X['petallength'], marker='o')

        ax.scatter(self.centers[:, 0], self.centers[:, 1],
                   self.centers[:, 2], marker='x', color='red',
                   s=300, linewidth=5, zorder=10)

        plt.show()
