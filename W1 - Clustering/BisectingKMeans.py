import numpy as np
from KMeans import KMeans


class BisectingKMeans:
    def __init__(self, X):
        self.n = 5
        self.df = X
        self.X = X.to_numpy()
        self.K = 2
        self.centers = np.empty(0)
        self.clusters = np.zeros((len(X), 1))
        self.sses = np.empty(0)
        self.random_seed = 1234

    def getCenters(self):
        return self.centers

    def getClusters(self):
        return self.clusters

    def setRandomSeed(self, seed=1234):
        self.random_seed = seed
        np.random.seed(self.random_seed)

    # Compute sum of squared errors from the points to the centers
    def SSEs(self, df, centroids, indexes):
        self.sses = np.zeros(len(centroids))
        arr = df.to_numpy()
        for i in range(len(indexes)):
            self.sses[indexes[i]] += np.linalg.norm(arr[i] - centroids[indexes[i]])
        return self.sses

        # Run the kmeans algorithm n times in order to obtain the clusters with larges similarities

    def executeKmeans(self, df, k, n, seed):
        model = KMeans(df)
        self.setRandomSeed(seed)
        nseeds = np.random.randint(0, len(df), n)
        lowestSSE = np.inf
        for i in range(n):
            model.applyKMeans(k=k, seed=nseeds[i])
            centroids = model.getCenters()
            indexes = model.getClusters()
            sses = self.SSEs(df, centroids, indexes)
            if sum(sses) < lowestSSE:
                bestCentroids = centroids
                bestIndexes = indexes
                bestSSEs = sses
                lowestSSE = sum(sses)
        return bestCentroids, bestIndexes, bestSSEs

    def applyBisectingKMeans(self, k, n=5, cluserSelectionType="sse", seed=1234):
        clusters = [None]
        if k > 1:
            centroids, indexes, sses = self.executeKmeans(self.df, k=2, n=n, seed=seed)

        # split a cluser as long as the number of clusters are below k
        while max(indexes) + 1 < k:
            # Get the cluster with hichest sum of squared error, the largest cluster, or the largest cluster with a sse
            # above average
            selectedCluster = 0
            if cluserSelectionType == "sse":
                selectedCluster = np.where(sses == max(sses))[0][0]
            elif cluserSelectionType == "largest":
                selectedCluster = np.argmax(np.bincount(indexes))
            elif cluserSelectionType == "mix":
                meanSseIndex = np.where(sses >= np.mean(sses))[0]
                selectedCluster = np.argmax(np.bincount([i for i in indexes if i in meanSseIndex]))

            # Apply the k-means on the cluster
            newcentroids, newindexes, newSses = self.executeKmeans(
                (self.df.loc[np.array(indexes) == selectedCluster]).reset_index(drop=True), 2, n=n, seed=seed)
            newindexes = [selectedCluster if x == 0 else max(indexes) + 1 for x in newindexes]

            # put the new clusters and certers into the lists
            indexes = [x if x != selectedCluster else newindexes.pop(0) for x in indexes]
            centroids[selectedCluster] = newcentroids[0]
            centroids = np.append(centroids, [newcentroids[1]], axis=0)
            sses[selectedCluster] = newSses[0]
            sses = np.append(sses, [newSses[1]], axis=0)
        self.clusters = np.array(indexes)
        self.centers = centroids
