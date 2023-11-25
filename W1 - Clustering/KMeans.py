import numpy as np
from scipy.spatial.distance import cosine


class KMeans:

    def __init__(self, X):
        self.X = X.to_numpy()
        self.K = 0
        self.centers = np.empty(0)
        self.mem = np.empty(0)
        self.clusters = np.zeros((len(X), 1))
        self.random_seed = 1234
        self.distance = "euclidean"
        self.initialization = "plusPlus"
        me = locals()["self"]
        self.distance_function = getattr(me, self.distance+"_distance")
        self.initialization_function = getattr(me, self.initialization+"Initialization")

    def getCenters(self):
        return self.centers

    def getClusters(self):
        return self.clusters

    def setDistanceFucntion(self, distance):
        self.distance = distance
        me = locals()["self"]
        self.distance_function = getattr(me, self.distance + "_distance")

    def setInitializationFunction(self, initialization):
        self.initialization = initialization
        me = locals()["self"]
        self.initialization_function = getattr(me, self.initialization + "Initialization")

    def setRandomSeed(self, seed=1234):
        self.random_seed = seed
        np.random.seed(self.random_seed)

    def euclidean_distance(self, j):
        return np.linalg.norm(self.X - self.centers[j], axis=1)

    def manhattan_distance(self, j):
        return np.linalg.norm(self.X - self.centers[j], axis=1, ord=1)

    def cosine_distance(self, j):
        A = np.linalg.norm(self.X, axis=1)
        B = np.linalg.norm(self.centers[j])
        rows = len(self.X)
        cosine_distance = np.zeros(rows)
        for i in range(rows):
            cosine_distance[i] = cosine(self.X[i], self.centers[j])
        return cosine_distance

    def forgyInitialization(self, seed=1234):
        self.setRandomSeed(seed)
        n = len(self.X)
        k = self.K
        self.centers = self.X[np.random.randint(0, n, k)]

    def plusPlusInitialization(self, seed=1234):
        n = len(self.X)
        k = self.K
        cols = self.X.shape[1]
        centroids = np.zeros((k, cols))
        if seed != 1234:
            self.setRandomSeed(seed)
        centroids[0] = self.X[np.random.randint(0, n)]
        i = 0
        while i < k - 1:
            distances = np.zeros((n, i + 1))
            for j in range(i + 1):
                distances[:, j] = np.maximum(np.linalg.norm(self.X - centroids[j], axis=1), 1e-3)
            minimum_distances = np.min(distances, axis=1)
            next_centroid_index = np.argmax(minimum_distances)
            centroids[i + 1] = self.X[next_centroid_index]
            i += 1
        self.centers = centroids

    def recomputeCenters(self):
        n = len(self.X)
        k = self.K
        newCenters = np.zeros_like(self.centers)
        for j in range(k):
            num = np.sum((self.mem[:, j]).reshape(n, 1) * self.X, axis=0)
            den = np.sum(self.mem[:, j])
            newCenters[j] = num / den
        self.centers = newCenters

    def memberships(self):
        '''
        mKHM(c_j | x_i) = mem[i][j]
        '''
        n = len(self.X)
        k = self.K
        mem = np.zeros((n, k))
        norms = np.zeros((n, k))

        for j in range(k):
            norma = np.maximum(self.distance_function(j), 1e-3)
            norms[:, j] = pow(norma, 2)
        minClusters = np.argmin(norms, axis=1)

        for j in range(k):
            mem[minClusters == j, j] = 1

        self.mem = mem

    def applyKMeans(self, k, seed=1234, distance="euclidean", initialization="plusPlus"):
        self.setDistanceFucntion(distance)
        self.setInitializationFunction(initialization)
        self.K = k
        self.initialization_function(seed)

        iteration = 0
        difference = 1000
        epsilon = 1e-4
        while iteration < 100 and difference >= epsilon:
            self.memberships()
            prevCenters = self.centers.copy()
            self.recomputeCenters()
            difference = np.linalg.norm(self.centers - prevCenters)
            iteration += 1

        self.clusters = np.argmax(self.mem, axis=1)

