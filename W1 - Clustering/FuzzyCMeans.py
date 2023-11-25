import numpy as np


class FuzzyCMeans:

    def __init__(self, X):
        self.X = X.to_numpy()
        self.K = 2
        self.m = 2
        self.V = np.empty(0)
        self.U = np.empty(0)
        self.clusters = np.empty(0)
        self.random_seed = 1234

    def getCenters(self):
        return self.V

    def getClusters(self):
        return self.clusters

    def setRandomSeed(self, seed=1234):
        self.random_seed = seed
        np.random.seed(self.random_seed)

    def randomInitialization(self, seed=1234):
        self.setRandomSeed(seed)
        n = len(self.X)
        k = self.K
        self.V = self.X[np.random.randint(0, n, k)]

    def computeU(self):
        n = len(self.X)
        c = self.K
        norms = np.zeros((n, c))

        for j in range(c):
            norms[:, j] = np.maximum(np.linalg.norm(self.X - self.V[j], axis=1), 1e-3)

        U = np.zeros((c, n))
        for j in range(c):
            division = np.zeros((n, c))
            for k in range(c):
                division[:, k] = pow(norms[:, j] / norms[:, k], 2 // (self.m - 1))

            summary = np.sum(division, axis=1)
            U[j, :] = pow(summary, -1)
        self.U = U

    def computeV(self):
        n = len(self.X)
        c = self.K
        V = np.zeros_like(self.X[:c, :])
        for j in range(c):
            mPowerOfUj = pow(self.U[j, :], self.m)
            mult = (mPowerOfUj.reshape(n, 1)) * self.X
            summary = np.sum(mult, axis=0)
            V[j, :] = summary / np.sum(mPowerOfUj)
        self.V = V

    def applyFuzzyCMeans(self, k, m=2, seed=1234):
        self.K = k
        self.m = m
        self.randomInitialization(seed)

        iteration = 0
        difference = 1000
        epsilon = 1e-4
        while iteration < 100 and difference >= epsilon:
            self.computeU()
            prevV = self.V.copy()
            self.computeV()
            difference = np.linalg.norm(self.V - prevV)
            iteration += 1
        self.clusters = np.argmax(self.U, axis=0)

