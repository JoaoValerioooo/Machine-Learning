import numpy as np


class PCA:

    """
    Method that initialize the PCA class using the current dataframe
    """
    def __init__(self, X):
        self.X = X.to_numpy()
        self.transformedData = self.X
        self.eigenvectors = np.empty(0)
        self.eigenvalues = np.empty(0)
        self.explainedVariances = np.empty(0)

    """
    Method that returns the eigenvectors of the dataset
    """
    def getEigenvectors(self):
        return self.eigenvectors

    """
    Method that returns the eigenvalues of the dataset
    """
    def getEigenvalues(self):
        return self.eigenvalues

    """
    It returns the transformed data after applying PCA
    """
    def getTransformedData(self):
        return self.transformedData

    """
    It returns the explained variance of all the components after PCA
    """
    def getExplainedVariance(self):
        return self.explainedVariances

    """
    It computes the explained variance of all the components using the eigenvalues
    """
    def computeExplainedVariances(self):
        total_egnvalues = sum(self.eigenvalues)
        self.explainedVariances = [(i / total_egnvalues) for i in sorted(self.eigenvalues, reverse=True)]

    """
    Function that applies the PCA basing in two parameters. If we indicate number of components, we will strictly 
    return the transformed data using that number of components. On the other hand, if we use minimum variance, we 
    will return the transformed data that allows that quantity of minimum variance in the new data.
    """
    def applyPCA(self, k=-1, minimum_variance=0.0):
        means = np.mean(self.X, axis=0)
        covariance_matrix = np.cov(self.X - means, rowvar=False)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(covariance_matrix)

        print("." * 60)
        print("Unsorted eigenvectors and eigenvalues:")
        self.eigenvectors = self.eigenvectors.T
        nvectors = len(self.eigenvalues)
        for i in range(nvectors):
            print(f"Eigenvector {i} with an eigenvalue of {self.eigenvalues[i]}:\n{self.eigenvectors[i]} ")

        self.computeExplainedVariances()

        print("." * 60)
        print("Sorted eigenvectors and eigenvalues:")
        sort_indexes = self.eigenvalues.argsort()
        sorted_eigenvalues = self.eigenvalues[sort_indexes[::-1]]
        sorted_eigenvectors = self.eigenvectors[sort_indexes[::-1]]

        components = k
        if components == -1:
            totalEigenvalues = sum(self.eigenvalues)
            accumVariance = 0
            i = 0
            while accumVariance < minimum_variance:
                accumVariance += sorted_eigenvalues[i] / totalEigenvalues
                i += 1
            components = i

        sorted_eigenvectors = sorted_eigenvectors[:components]
        sorted_eigenvalues = sorted_eigenvalues[:components]
        nvectors = len(sorted_eigenvalues)
        for i in range(nvectors):
            print(f"Eigenvector {i} with an eigenvalue of {sorted_eigenvalues[i]}:\n{sorted_eigenvectors[i]} ")
        sorted_eigenvectors = sorted_eigenvectors.T
        self.transformedData = np.matmul(self.X - means, sorted_eigenvectors)
        # Inverse the process
        # reconsData = np.matmul(transData, sorted_eigenvectors.T)
        # reconsData += means

