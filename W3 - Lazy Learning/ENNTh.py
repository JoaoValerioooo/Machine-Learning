import numpy as np
import pandas as pd

class ENNTh:

    def __init__(self, data, labels, k=5, mu=0.15):
        self.data = data
        self.X_train = data.to_numpy()
        self.labels = labels
        self.K = k
        self.mu = mu

    # Creat the P_i dataframe
    def get_P_i_dataframe(self):
        # Number of classes
        classes = np.unique(self.labels)
        # Probability Dataframe
        P_i_dataframe = pd.DataFrame()
        for num in classes:
            P_i_dataframe['P_i_class_' + str(num)] = ''
        P_i_dataframe['P_i_nearest_neighboor_'] = ''
        return P_i_dataframe, classes

    # Get the term p_i_j (probability of the nearest point belong to the class of the
    # current iteration -> belongs p_i_j=1, does not belong p_i_j=0)
    def get_p_i_j(self, KNN_idx, num):
        p_i_j = []
        for i in KNN_idx:
            if self.labels[i] == num:
                p_i_j.append(1)
            else:
                p_i_j.append(0)
        return p_i_j

    # Get the nearest neighbors distances and indices
    def get_nearest_neighbors(self, dist):
        KNN_idx = np.argpartition(dist[:, 0], self.K)[:self.K]
        KNN = dist[:, 0][KNN_idx]
        return KNN_idx, KNN

    # Probability of a point belong to a class
    def calculate_P_i(self, p_i_j, KNN):
        return sum(p_i_j * (1 / (1 + KNN)))

    def getMinkowskiDistanceMatrix(self, X):
        differences = np.zeros_like(self.X_train)
        for index, feature in enumerate(X):
            if type(feature) != str:
                diffs = abs(self.X_train[:, index] - feature)
                differences[:, index] = diffs**2
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
                differences[indexes_n_0, index] = up[indexes_n_0]/down[indexes_n_0]
            else:
                diffs = self.X_train[:, index] != feature
                differences[:, index] = diffs

        return np.sum(differences, axis=1)

    # Get the probability that a sample x belongs to a class i
    def get_P_i(self, data_numpy):
        # Get the P_i dataframe
        P_i_dataframe, classes = self.get_P_i_dataframe()

        for id in range(len(data_numpy)):
            # Distance to all the points
            # dist = distance.cdist(data_numpy[:, :], [data_numpy[id, :]], 'euclidean')
            dist = self.getCanberraDistanceMatrix(data_numpy[id, :])
            dist = dist.reshape(-1, 1)
            P_i = []
            for num in classes:
                # Get the KNN for each class
                KNN_idx, KNN = self.get_nearest_neighbors(dist)
                # Get p_i_j
                p_i_j = self.get_p_i_j(KNN_idx, num)
                # Calculate the probability and save in the dataframe
                P_i.append(self.calculate_P_i(p_i_j, KNN))
                P_i_dataframe.at[id, 'P_i_class_' + str(num)] = P_i[num]
                P_i_dataframe.at[id, 'P_i_nearest_neighboor_'] = KNN_idx
        return P_i_dataframe

    ############ P_i auiliar functions ############
    ############ P_i auiliar functions ############
    ############ P_i auiliar functions ############

    # Creat the p_i dataframe
    def get_p_i_dataframe(self):
        # Number of classes
        classes = np.unique(self.labels)
        # Probability Dataframe
        p_i_dataframe = pd.DataFrame()
        for num in classes:
            p_i_dataframe['p_i_class_' + str(num)] = ''
        return p_i_dataframe, classes

    # Calculate p_i per class
    def calculate_p_i(self, id, num, P_i_dataframe, p_i_dataframe):
        if P_i_dataframe.at[id, 'P_i_class_' + str(num)] == 0:
            p_i_dataframe.at[id, 'p_i_class_' + str(num)] = 0
        else:
            P_i_sum = sum(
                [P_i_dataframe.at[i, 'P_i_class_' + str(num)] for i in P_i_dataframe.at[id, 'P_i_nearest_neighboor_']])
            p_i_dataframe.at[id, 'p_i_class_' + str(num)] = P_i_dataframe.at[id, 'P_i_class_' + str(num)] / P_i_sum

    # Weighted average of the probabilities that its k-nearest neighbors belong to that class
    def get_p_i(self, P_i_dataframe):
        # Get the p_i dataframe
        p_i_dataframe, classes = self.get_p_i_dataframe()
        # Create new_data and new_labels
        new_data, new_labels = pd.DataFrame(columns=self.data.columns), []

        # Get the valid observations
        for id in range(len(P_i_dataframe)):
            # Calculate p_i
            [self.calculate_p_i(id, num, P_i_dataframe, p_i_dataframe) for num in classes]
            # Condition to add the observation to the new Dataframe
            array = np.asarray(p_i_dataframe.iloc[id])
            if self.labels[id] == np.argmax(array) and max(array) > self.mu:
                new_data = new_data.append(self.data.iloc[id], ignore_index=True)
                new_labels.append(self.labels[id])
        return new_data, new_labels

    ############ p_i auiliar functions ############
    ############ p_i auiliar functions ############
    ############ p_i auiliar functions ############

    ## Main function
    # Editing Algorithm Estimating Class Probabilities and Threshold (WilsonTh)
    def applyENNTh(self):
        # Condition
        if self.K < 1 or self.K >= len(self.data) or self.mu < 0 or self.mu > 1: return -1

        # Get all the P_i values
        P_i_dataframe = self.get_P_i(np.asarray(self.data))
        # display(P_i_dataframe)

        # Get all the p_i values and new_data+new_labels
        new_data, new_labels = self.get_p_i(P_i_dataframe)

        return new_data, new_labels

