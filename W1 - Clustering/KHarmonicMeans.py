import numpy as np


class KHarmonicMeans:

    def __init__(self, X):
        self.X = X.to_numpy()
        self.K = 2
        self.p = 2
        self.centers = np.empty(0)
        self.mem = np.empty(0)
        self.weig = np.empty(0)
        self.clusters = np.zeros((len(X), 1))
        self.random_seed = 1234


    def getCenters(self):
        return self.centers

    def getClusters(self):
        return self.clusters

    def setRandomSeed(self, seed=1234):
        self.random_seed = seed
        np.random.seed(self.random_seed)

    def randomInitialization(self, seed=1234):
        self.setRandomSeed(seed)
        n = len(self.X)
        k = self.K
        self.centers = self.X[np.random.randint(0, n, k)]

    def memberships(self):
        '''
        mKHM(c_j | x_i) = mem[i][j]
        '''
        n = len(self.X)
        k = self.K
        norms = np.zeros((n, k))

        for j in range(k):
            norma = np.maximum(np.linalg.norm(self.X - self.centers[j], axis=1), 1e-3)
            norms[:, j] = pow(norma, -self.p - 2)

        sumNorms = np.sum(norms, axis=1)
        mem = np.zeros((n, k))

        for j in range(k):
            mem[:, j] = norms[:, j] / sumNorms

        self.mem = mem

    def weights(self):
        '''
        wKHM(x_i) = weig[i]
        '''
        n = len(self.X)
        k = self.K
        norms = np.zeros((n, k))

        for j in range(k):
            norms[:, j] = np.maximum(np.linalg.norm(self.X - self.centers[j], axis=1), 1e-3)

        den = pow(np.sum(pow(norms, -self.p), axis=1), 2)
        num = np.sum(pow(norms, -self.p - 2), axis=1)
        weig = num / den

        self.weig = weig

    def performance(self):
        n = len(self.X)
        k = self.K
        norms = np.zeros((n, k))

        for j in range(k):
            norms[:, j] = np.maximum(np.linalg.norm(self.X - self.centers[j], axis=1), 1e-3)

        den = np.sum(pow(norms, -self.p), axis=1)
        div = k / den
        perf = np.sum(div)
        return perf

    def recomputeCenters(self):
        n = len(self.X)
        k = self.K
        newCenters = np.zeros_like(self.centers)
        for j in range(0, k):
            num = np.sum((self.mem[:, j] * self.weig).reshape(n, 1) * self.X, axis=0)
            den = np.dot(self.mem[:, j], self.weig)
            newCenters[j] = num / den
        self.centers = newCenters

    def applyKHarmonicMeans(self, k, p=5, seed=1234):
        self.K = k
        self.p = p
        self.randomInitialization(seed)

        iter = 0
        difference = 1000
        epsilon = 1e-4
        pnow = 0

        while iter < 100 and difference >= epsilon:
            self.memberships()
            self.weights()
            prevCenters = self.centers.copy()
            self.recomputeCenters()
            difference = np.linalg.norm(self.centers - prevCenters)
            iter += 1
        self.clusters = np.argmax(self.mem, axis=1)


