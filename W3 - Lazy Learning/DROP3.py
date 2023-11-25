import time
import numpy as np
import pandas as pd
import kNNAlgorithm as KNN
import ENNTh
import Visualize as vis


class DROP3:

    def __init__(self, X, y_train, K=5, Knn_k=5, distance="Canberra"):
        ennth = ENNTh.ENNTh(X, y_train)
        X, y_train = ennth.applyENNTh()
        self.X_df = X
        self.X_train = X.to_numpy()
        self.y_train = y_train
        self.K = K
        self.Knn_k = Knn_k
        self.distance_type = distance
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
        return pow(suma, 1 / 2)

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

        return np.sum(differences, axis=1)

    def getDistances(self, P):
        return self.distance_function(P)

    def findNeighbours(self, P):
        distances = self.getDistances(P)
        sorted_indexes = np.argsort(distances)[1:self.K + 2]
        return sorted_indexes

    def classifiedCorrectly(self, X, y, individuals, ind_classes, kwargs):
        test = pd.DataFrame(individuals)
        knn = KNN.kNNAlgorithm(self.Knn_k)
        knn.fit(X, y, **kwargs)
        predictions = knn.predict(test)
        return np.sum(predictions == ind_classes)

    def neighboursSortedByDistance(self):
        n = len(self.X_train)
        neighbours = []
        list_of_distances = []
        for index, P in enumerate(self.X_train):
            distances = self.getMinkowskiDistanceMatrix(P)
            sorted_neighbours = np.argsort(distances)
            neighbours.append(sorted_neighbours)
            list_of_distances.append(distances)
        return neighbours, list_of_distances

    def sortedbyNearestEnemies(self, X_train, neighbours_list, distances, classes):
        n = len(X_train)
        DistancesToNearestEnemy = np.zeros(n)
        for P in range(n):
            points_order = neighbours_list[P]
            classes_order = classes[points_order]
            myClass = classes[P]
            firstDifferentClass = np.argwhere(classes_order != myClass)[0]
            nearestEnemy = points_order[firstDifferentClass].squeeze()
            DistancesToNearestEnemy[P] = distances[P][nearestEnemy]
        SortingOrder = np.argsort(DistancesToNearestEnemy)
        # [::-1][:n]
        return X_train.iloc[SortingOrder]

    def applyDROP3(self, **kwargs):
        T = self.X_df
        S = T.copy()
        classes = pd.DataFrame(self.y_train)
        neighbours_list, distances = self.neighboursSortedByDistance()
        T = self.sortedbyNearestEnemies(T, neighbours_list, distances, np.array(self.y_train))

        n = len(T)
        associates = np.zeros((n, n))
        neighMatrix = np.zeros((n, n))
        erased = []
        for index, P in enumerate(S.values):
            neighbours = self.findNeighbours(P)
            neighMatrix[index, neighbours] = 1
            associates[neighbours, index] = 1
        real_indexes = T.index
        ini = time.time()
        for index, P in enumerate(T.values):
            r_index = real_indexes[index]
            pAssociates = np.argwhere(associates[r_index] == 1).squeeze()
            AssocFeatures = self.X_df.iloc[pAssociates].values.squeeze()
            AssocClasses = classes.iloc[pAssociates].values
            TwithoutP = T.drop(index)
            YwithoutP = classes.drop(index).values.squeeze()
            n_with = self.classifiedCorrectly(T, self.y_train, AssocFeatures, AssocClasses, kwargs)
            n_without = self.classifiedCorrectly(TwithoutP, YwithoutP, AssocFeatures, AssocClasses, kwargs)
            if n_without >= n_with:
                S.drop(index, inplace=True)
                erased.append(index)
                if pAssociates.size == 1:
                    pAssociates = np.expand_dims(pAssociates, axis=0)
                for A in pAssociates:
                    neighMatrix[A, index] = 0
                sorted_neighbours = neighbours_list[A]
                new_neighbours = [neigh for neigh in sorted_neighbours if neigh not in erased]
                new_neighbours = new_neighbours[1:self.K + 2]
                new_neigh = new_neighbours[-1]
                neighMatrix[A, new_neigh] = 1
                associates[new_neigh, A] = 1
            time.sleep(0.1)
            vis.printProgressBar(index + 1, n, prefix='           ', suffix='Complete', length=50, begin=ini)

        new_y_train = classes.drop(erased).values.squeeze()
        return S, new_y_train



