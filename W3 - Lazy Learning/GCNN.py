import itertools
import numpy as np
from scipy.stats import mode

class GCNN:

    def __init__(self, X_train, y_train, rho=0.001, distance="Canberra"):
        self.X_df = X_train
        self.X_train = X_train.to_numpy()
        self.y_train = y_train
        self.rho = rho
        self.distances = None
        self.distance_function = None
        self.setDistanceFunction(distance)

    def setDistanceFunction(self, distance):
        self.distance_type = distance
        thisClass = locals()["self"]
        self.distance_function = getattr(thisClass, f"get{self.distance_type}DistanceMatrix")

    def getMinkowskiDistanceMatrix(self, X):
        differences = np.zeros_like(self.X_train)
        for index, feature in enumerate(X):
            if type(feature) != str:
                diffs = abs(self.X_train[:, index] - feature)
                differences[:, index] = diffs ** 2
            else:
                diffs = self.X_train[:, index] != feature
                differences[:, index] = diffs

        suma = np.sum(differences, axis=1)
        self.distances = pow (suma, 1 / 2)

    def getCanberraDistanceMatrix(self, X):
        differences = np.zeros_like(self.X_train)
        for index, feature in enumerate(X):
            if type(feature) != str:
                up = abs(self.X_train[:, index] - feature)
                down = abs(self.X_train[:, index] + feature)
                indexes_0 = np.argwhere(down == 0)
                indexes_n_0 = np.argwhere(down != 0)
                differences[indexes_0, index] = 0
                differences[indexes_n_0, index] = up[indexes_n_0] / down[indexes_n_0]
            else:
                diffs = self.X_train[:, index] != feature
                differences[:, index] = diffs

        self.distances = np.sum(differences, axis=1)

    def getDistances(self, P):
        return self.distance_function(P)

    def applyGCNN(self):
        label_count = len(set(self.y_train))
        prototypes = [[] for i in range(label_count)]
        unabsorbed = [[] for i in range(label_count)]
        votes = [[] for i in range(label_count)]
        DistanceMatrix = [None] * len(self.y_train)
        # G1 initialization.
        for i in range(len(self.y_train)):
            x = self.X_train[i]
            label = self.y_train[i]
            # Fill a matrix with the distances to all the instances
            self.distance_function(np.array(x))
            DistanceMatrix[i] = np.copy(self.distances)
            low = np.inf
            # Each instance makes a vote to the nearest neighbour of the same label
            for j in range(len(self.y_train)):
                if self.y_train[j] == label and i != j and self.distances[j] < low:
                    low = self.distances[j]
                    nn = j
            votes[label].append(nn)
        all_absorbed = False
        const = np.amax(DistanceMatrix)
        # As long as not all instances are absorbed by a prototype, there will be added new prototypes.
        while not all_absorbed:
            # For each label the instance with most votes are added to the prototype list
            for i in range(len(votes)):
                if (votes[i]):
                    prototypes[i].append(mode(votes[i])[0][0])
            # G2 Absorption Check
            for i in range(len(DistanceMatrix)):
                if not any(i in sublist for sublist in prototypes):
                    label = self.y_train[i]
                    other_labeled_prototypes = prototypes.copy()
                    other_labeled_prototypes.pop(label)
                    other_labeled_prototypes = list(itertools.chain(*other_labeled_prototypes))
                    distance_same_label = min(DistanceMatrix[i][prototypes[label]])
                    distance_other_label = min(DistanceMatrix[i][other_labeled_prototypes])
                    # If the distance to the nearest prototype of the same label + rho is less than to the nearest
                    # prototype of another label, the example is defined as absorbed
                    if (distance_other_label - distance_same_label) < (self.rho * const):
                        unabsorbed[self.y_train[i]].append(i)
            # G3 Prototype Augmentation
            if not any(unabsorbed) or len(other_labeled_prototypes) == len(DistanceMatrix):
                all_absorbed = True
            else:
                votes = [[] for i in range(label_count)]
                # Each unabsorbed instance makes a vote to the closest non-prototype instance of the same label
                for label in range(len(unabsorbed)):
                    for i in unabsorbed[label]:
                        low = np.inf
                        for j in unabsorbed[label]:
                            if i != j and DistanceMatrix[i][j] < low:
                                low = DistanceMatrix[i][j]
                                nn = j
                            elif len(unabsorbed[label]) == 1:
                                nn = j
                        # The instances with most votes will be added to the prototype list in the next iteration
                        votes[label].append(nn)
                unabsorbed = [[] for i in range(label_count)]
        new_data = list(itertools.chain(*prototypes))
        self.X_train = self.X_df.iloc[new_data]
        self.y_train = self.y_train[new_data]
        return self.X_train, self.y_train.astype(int)


