import numpy as np
import Preprocessing as prep
from collections import Counter
from sklearn import feature_selection


class kNNAlgorithm:

    def __init__(self, K=3):
        self.K = K
        self.X_train_df = None
        self.X_train = None
        self.y_train = None
        self.distances = None
        self.distance_type = "Minkowski"
        self.distance_function = None
        self.voting_type = "Majority"
        self.voting_function = None
        self.weights = None
        self.weighting_type = "Equal"
        self.weighting_function = None
        self.P = None

    def setDistanceFucntion(self, distance):
        self.distance_type = distance
        thisClass = locals()["self"]
        self.distance_function = getattr(thisClass, f"get{self.distance_type}DistanceMatrix")

    def getMinkowskiDistanceMatrix(self, X):
        differences = np.zeros_like(self.X_train)
        for index, feature in enumerate(X):
            if type(feature) != str:
                diffs = abs(self.X_train[:, index] - feature)
                differences[:, index] = self.weights[index] * (diffs.astype(np.float32))**2
            else:
                diffs = self.X_train[:, index] != feature
                differences[:, index] = self.weights[index] * diffs.astype(np.float32)

        suma = np.sum(differences, axis=1)
        self.distances = pow(suma, 1 / 2)

    def getCanberraDistanceMatrix(self, X):
        differences = np.zeros_like(self.X_train)
        for index, feature in enumerate(X):
            if type(feature) != str:
                up = abs(self.X_train[:, index] - feature)
                down = abs(self.X_train[:, index] + feature)
                indexes_0 = np.argwhere(down == 0)
                indexes_n_0 = np.argwhere(down != 0)
                differences[indexes_0, index] = 0
                differences[indexes_n_0, index] = self.weights[index] * (up[indexes_n_0]/down[indexes_n_0])
            else:
                diffs = self.X_train[:, index] != feature
                differences[:, index] = self.weights[index] * diffs.astype(np.float32)

        self.distances = np.sum(differences, axis=1)


    def getCosineDistanceMatrix(self, X):
        ABmult = np.zeros_like(self.X_train)
        Asquare = np.ones_like(self.X_train)
        Bsqaure = np.ones_like(X)
        for index, feature in enumerate(X):
            if type(feature) != str:
                ABmult[:, index] = self.weights[index] * self.X_train[:, index] * feature
                Asquare[:, index] = self.weights[index] * pow(self.X_train[:, index], 2)
                Bsqaure[index] = self.weights[index] * feature * feature
            else:
                equal = self.X_train[:, index] == feature
                ABmult[:, index] = self.weights[index] * equal.astype(np.float32)
        Anorm = pow(np.sum(Asquare, axis=1), 1 / 2)
        Bnorm = pow(np.sum(Bsqaure), 1 / 2)
        ABsum = np.sum(ABmult, axis=1)
        cosine_similarity = ABsum / (Anorm * Bnorm)
        self.distances = 1 - cosine_similarity

    def setVotingFunction(self, voting):
        self.voting_type = voting
        thisClass = locals()["self"]
        self.voting_function = getattr(thisClass, f"get{self.voting_type}Voting")

    def getMajorityVoting(self, distance_matrix):
        count_values = Counter([col[1] for col in distance_matrix][:self.K])
        most_frequent_labels = [key for key, value in count_values.items() if value == max(count_values.values())]
        if len(most_frequent_labels) > 1:
            return self.compute_weighted_vote(
                [element for element in distance_matrix if element[1] in most_frequent_labels],
                max(count_values.values()) * len(most_frequent_labels),
                lambda x: len(self.y_train) - x)
        else:
            return most_frequent_labels[0]

    def getInverseDistanceWeightedVoting(self, distance_matrix):
        return self.compute_weighted_vote(distance_matrix, self.K, lambda x: np.inf if x == 0 else 1 / (x ** self.P))

    def getSheppardsVoting(self, distance_matrix):
        return self.compute_weighted_vote(distance_matrix, self.K, lambda x: np.exp(-x))

    def compute_weighted_vote(self, distance_matrix, k, equation):
        distance_sums = {}
        max_keys = []
        for i in range(len(distance_matrix)):
            distance_sums[distance_matrix[i][1]] = \
                distance_sums.setdefault(distance_matrix[i][1], 0) + equation(distance_matrix[i][0])
            if i + 1 >= k:
                max_keys = [key for key, value in distance_sums.items() if value == max(distance_sums.values())]
                if len(max_keys) == 1:
                    return max_keys[0]
        return max_keys[0]

    def setWeightingFunction(self, weighting):
        self.weighting_type = weighting
        thisClass = locals()["self"]
        self.weighting_function = getattr(thisClass, f"{self.weighting_type}Weight")

    def scaler(self, weights):
        return np.asarray([(value - min(weights)) / (max(weights) - min(weights)) for value in weights])
    def EqualWeight(self):
        self.weights = np.ones(len(self.X_train_df.columns))

    # Information Gain (IG)
    def InformationGainWeight(self, neighbors=5, seed=20):
        self.weights = self.scaler(feature_selection.mutual_info_classif(self.X_train_df.to_numpy(), self.y_train,
                                                     discrete_features=prep.getColumnTypeBool(self.X_train_df),
                                                     n_neighbors=neighbors, copy=True, random_state=seed))

    def ChiWeight(self):
        self.weights = self.scaler(feature_selection.chi2(self.X_train, self.y_train)[0])

    def fit(self, X, y, distance="Minkowski", voting="Majority", P=1, weighting="Equal"):
        #Print possible options for each parameter
        self.X_train_df = X
        self.X_train = X.values
        self.y_train = y
        self.P = P
        self.setDistanceFucntion(distance)
        self.setVotingFunction(voting)
        self.setWeightingFunction(weighting)
        self.weighting_function()

    def predict(self, X):
        X = X.values
        y_pred = [self.getClass(x) for x in X]
        return y_pred

    def getClass(self, x):
        self.distance_function(x)
        K_neighbours = np.argsort(self.distances)
        classes = [[self.distances[index], self.y_train[index]] for index in K_neighbours]
        pred_class = self.voting_function(classes)
        return pred_class



