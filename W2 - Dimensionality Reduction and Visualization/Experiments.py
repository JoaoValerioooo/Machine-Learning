import time
import psutil
import pandas as pd
import numpy as np
import Preprocessing as prep
from matplotlib import pyplot as plt
import KMeans
import Agglomerative
import sklearn.decomposition as decom
import Visualize as vis
import PCA as pca
from sklearn.manifold import TSNE
from sklearn import cluster

pd.options.display.float_format = '{:.2f}'.format
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class Experiments:

    """
    Method that allows to initialize the experiment class, the names of the parameters are self describing
    """
    def __init__(self, X, labels, model_name, dataset_name, seed=1234):
        module = globals()[model_name]
        self.class_ = getattr(module, model_name)
        self.X = X
        self.model = self.class_(X)
        self.seed = seed
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.true_labels = labels
        self.K = len(np.unique(labels))
        self.visu = vis.Visualize(self.dataset_name, model_name=self.model_name)

    """
    Method that allows us to set the model used in clustering with the new data, mainly used for applying clustering 
    after some feature reduction technique
    """
    def setModelWithNewData(self, data):
        df = pd.DataFrame(data)
        self.model = self.class_(df)

    """
    Method that resets the dataframe of the clustering model
    """
    def resetDataframeModel(self):
        self.model = self.class_(self.X)

    """
    Method that generates the confusion matrix after applying clustering over the data in the class
    """
    def getConfusionMatrixK(self, plot_title="Default", directory="./Plots"):
        k = self.K
        applyClustering = getattr(self.model, "apply" + self.model_name)
        applyClustering(k, seed=self.seed)
        pred_labels = self.model.getClusters()
        self.visu.saveConfusionMatrix(true_labels=self.true_labels, pred_labels=pred_labels, plot_title=plot_title,
                                      directory=directory)

    """
    Method that allows us to get the new clustering labels after applying clustering. It renames the labels in order to
    match with the ones that are in the confusion matrix.
    """
    def getClusteringLabels(self):
        k = self.K
        applyClustering = getattr(self.model, "apply" + self.model_name)
        applyClustering(k, seed=self.seed)
        pred_labels = self.model.getClusters()
        clusters_distributions = np.zeros((k, k))
        # Find cluster distributions
        for i in range(k):
            cluster_i = pred_labels[self.true_labels == i]
            cluster_i_distr = np.zeros(k)
            for j in range(k):
                cluster_i_distr[j] = np.sum(cluster_i == j)
            clusters_distributions[i] = cluster_i_distr

        # Find new name for clusters

        newLabels = np.zeros_like(pred_labels)
        available_clusters = list(range(k))
        for i in range(k):
            best_distr_as_i = -1
            max_points = -np.inf
            for j in available_clusters:
                total_individuals = np.sum(clusters_distributions[j])
                diagonal = clusters_distributions[j, i]
                points = diagonal - (total_individuals - diagonal)
                if points > max_points:
                    best_distr_as_i = j
                    max_points = points

            indexes = np.where(pred_labels == i)
            newLabels[indexes] = best_distr_as_i

            available_clusters.remove(best_distr_as_i)
        return newLabels

    """
    Method that saves the explained variance plots
    """
    def studyExplainedVariance(self):
        PersonalPca = pca.PCA(self.X)
        f = self.X.shape[1]
        PersonalPca.applyPCA(k=f)
        explainedVariance = PersonalPca.getExplainedVariance()
        self.visu.plotExplainedVariance(explainedVariance)

    """
    Method that applies PCA and saves the confusion matrix
    """
    def PCAandSaveConfusionMatrix(self, k=-1, minimum_variance=-1):
        PersonalPca = pca.PCA(self.X)
        PersonalPca.applyPCA(k=k, minimum_variance=minimum_variance)
        newData = PersonalPca.getTransformedData()
        plot_title = ""
        if k != -1:
            plot_title = f"Confusion Matrix of {self.model_name} with {self.dataset_name} and {k} PCA\'s"
        else:
            plot_title = f"Confusion Matrix of {self.model_name} with {self.dataset_name} and {minimum_variance} variance"
        self.setModelWithNewData(newData)
        self.getConfusionMatrixK(plot_title=plot_title, directory="./Plots/ConfusionMatrix/PCA")

    """
    Method that doesn't apply PCA and saves the confusion matrix
    """
    def NoPCAandSaveConfusionMatrix(self):
        plot_title = f"Confusion Matrix of {self.model_name} with {self.dataset_name} without PCA"
        self.getConfusionMatrixK(plot_title=plot_title, directory="./Plots/ConfusionMatrix/NoPCA")

    """
    Method that plots the data using the top 3 principal components in the data
    """
    def studyFeatures(self, k=-1, minimum_variance=-1.0):
        PersonalPca = pca.PCA(self.X)
        PersonalPca.applyPCA(k=k, minimum_variance=minimum_variance)
        newData = PersonalPca.getTransformedData()
        self.visu.plotPrincipalFeatures(newData[:, 0], newData[:, 1], newData[:, 2])

    """
    Method that plots the data using the top 3 principal components in the data and coloring the points depending on 
    their cluster
    """
    def studyFeaturesWithClustering(self, k=-1, minimum_variance=-1.0):
        PersonalPca = pca.PCA(self.X)
        PersonalPca.applyPCA(k=k, minimum_variance=1)
        newData = PersonalPca.getTransformedData()
        self.setModelWithNewData(newData)
        newlabels = self.getClusteringLabels()
        sub_title = f"Without dimensionality reduction, {newData.shape[1]} features & {100}% explained variance"
        self.visu.plotPrincipalFeaturesWithClustering(newData[:, 0], newData[:, 1], newData[:, 2], labels=newlabels,
                                                      sub_title=sub_title)

        PersonalPca = pca.PCA(self.X)
        PersonalPca.applyPCA(k=k, minimum_variance=minimum_variance)
        newData = PersonalPca.getTransformedData()
        self.setModelWithNewData(newData)
        newlabels = self.getClusteringLabels()
        sub_title = f"With dimensionality reduction, {newData.shape[1]} features & {minimum_variance * 100}% explained variance"
        self.visu.plotPrincipalFeaturesWithClustering(newData[:, 0], newData[:, 1], newData[:, 2], labels=newlabels,
                                                      sub_title=sub_title)

    """
    Method that applies PCA of the Sklearn libray using the components passed as parameter
    """
    def sklearn_PCA(self, components):
        print("*" * 70)
        print('sklearn - PCA')
        print("*" * 70)
        self.model_output(decom.PCA(n_components=components).fit(self.X))

    """
    Method that applies incremental PCA of the Sklearn libray using the components passed as parameter
    """
    def sklearn_IncrementalPCA(self, components):
        print("*" * 70)
        print('sklearn - Incremenatal PCA')
        print("*" * 70)
        self.model_output(decom.IncrementalPCA(n_components=components).fit(self.X))

    """
    Method that applies our PCA implementation using the components passed as parameter
    """
    def implementation_PCA(self, components):
        print("*" * 70)
        print('Our implementation of PCA')
        print("*" * 70)
        pca.PCA(self.X).applyPCA(k=components)

    """
    Method used to print important information about Sklearn PCA methods
    """
    def model_output(self, model):
        # print('Number of Components: \n', len(model.components_))
        point = "."
        print('Components: \n', model.components_)
        print(point * 70)
        print('Sorted Eigenvalues: \n', model.explained_variance_)
        print(point * 70)
        print('Explained variance ratio: \n', model.explained_variance_ratio_)
        print(point * 70)
        print('Sorted Eigenvectors: \n', model.components_)
        print(point * 70)
        print('Mean of each feature: \n', model.mean_)
        print(point * 70)
        print('Estimated noise covariance: \n', model.noise_variance_)
        print(point * 70)

    """
    Method that allows us to know the memory usage and the time spent by Sklearn PCA or Sklearn IPCA
    """
    def memory_and_time_used(self, model, components, n=10):
        times = np.zeros(n)
        memories = np.zeros(n)
        RAMs = np.zeros(n)
        for i in range(n):
            # Start time
            st = time.time()
            # Apply dimensionality reduction model
            model(n_components=components).fit(self.X)
            et = time.time()
            # Execution time
            times[i] = et - st
            # Getting % usage of virtual_memory
            memories[i] = psutil.virtual_memory()[2]
            # Getting usage of virtual_memory in GB
            RAMs[i] = psutil.virtual_memory()[3] / 1000000000
        print('Average execution time for', n, 'times:', times.mean(), 'seconds')
        print('Average RAM memory % used:', n, 'times:', memories.mean())
        print('Average RAM Used (GB):', n, 'times:', RAMs.mean())

    """
    Method that applies PCA over the data, mainly used in the option 0 of the main program
    """
    def doPCA(self, components):
        print("Executing PCA ...")
        point = "."
        print(point * 70)
        PersonalPca = pca.PCA(self.X)
        PersonalPca.applyPCA(k=components)
        print(point * 70)
        newData = PersonalPca.getTransformedData()
        print("First 5 rows of transformed data:")
        print(newData[:5])

    """
    Method that applies Sklearn PCA over the data, mainly used in the option 0 of the main program
    """
    def doSklearnPCA(self, components):
        print("Executing Sklean-PCA ...")
        point = "."
        print(point * 70)
        model = decom.PCA(n_components=components).fit(self.X)
        print('Sorted Eigenvalues: \n', model.explained_variance_)
        print(point * 70)
        print('Sorted Eigenvectors: \n', model.components_)
        print(point * 70)
        print("First 5 rows of transformed data:")
        newData = model.transform(self.X)
        print(newData[:5])

    """
    Method that applies Sklearn IPCA over the data, mainly used in the option 0 of the main program
    """
    def doSklearnIPCA(self, components):
        print("Executing Sklean-IPCA ...")
        point = "."
        print(point * 70)
        model = decom.IncrementalPCA(n_components=components).fit(self.X)
        print('Sorted Eigenvalues: \n', model.explained_variance_)
        print(point * 70)
        print('Sorted Eigenvectors: \n', model.components_)
        print(point * 70)
        print("First 5 rows of transformed data:")
        newData = model.transform(self.X)
        print(newData[:5])

    """
    Method that applies feature agglomeration over the data, mainly used in the option 0 of the main program
    """
    def doFeatureAgglomeration(self, components):
        print("Executing Feature Agglomeration ...")
        point = "."
        agglo = cluster.FeatureAgglomeration(n_clusters=components)
        newData = agglo.fit_transform(self.X)
        print(point * 70)
        print("First 5 rows of transformed data:")
        print(newData[:5])

    """
    Method that applies t-SNE over the data, mainly used in the option 0 of the main program
    """
    def dotSNE(self, components):
        print("Executing t-SNE ...")
        point = "."
        newData = TSNE(n_components=components, learning_rate='auto', init='random').fit_transform(self.X)
        print(point * 70)
        print("First 5 rows of transformed data:")
        print(newData[:5])

    """
    Method that allows reading the number of clusters for feature agglomeration 
    """
    def readNumberOfClusters(self):
        return int(input(f"Enter the number of clusters between 1 and {self.X.shape[0]}:\n"))

    """
    Method that prints the clustering results for the KMeans algorithm
    """
    def printClusteringResults(self):
        print(
            f"The {self.K} centers of the {self.dataset_name} dataset with the {self.model_name} model are:")
        print(self.model.getCenters())

    """
    Method that applies a clustering method over the data available in the model
    """
    def applyClustering(self, k):
        self.K = k
        applyClustering = getattr(self.model, "apply" + self.model_name)
        print(f"Executing {self.model_name} clustering ...")
        if self.model_name != "KMeans":
            self.model.setDatasetName(self.dataset_name)
            applyClustering(k, seed=self.seed, dendro=True)
        else:
            applyClustering(k, seed=self.seed)
            self.printClusteringResults()

    """
    Method that applies PCA over the data, and then applies clustering with the transformed data. Mainly used in the 
    option 1 of the main program
    """
    def doPCAClustering(self, components):
        k = self.readNumberOfClusters()
        if 1 <= k <= self.X.shape[0]:
            print(f"Executing PCA ...")
            PersonalPca = pca.PCA(self.X)
            PersonalPca.applyPCA(k=components)
            newData = PersonalPca.getTransformedData()
            self.setModelWithNewData(newData)
            self.applyClustering(k)
        else:
            print("Please try again")

    """
    Method that applies Sklearn PCA over the data, and then applies clustering with the transformed data. Mainly used in the 
    option 1 of the main program
    """
    def doSklearnPCAClustering(self, components):
        k = self.readNumberOfClusters()
        if 1 <= k <= self.X.shape[0]:
            print(f"Executing Sklearn-PCA ...")
            newData = decom.PCA(n_components=components).fit_transform(self.X)
            self.setModelWithNewData(newData)
            self.applyClustering(k)
        else:
            print("Please try again")

    """
    Method that applies Sklearn IPCA over the data, and then applies clustering with the transformed data. Mainly used in the 
    option 1 of the main program
    """
    def doSklearnIPCAClustering(self, components):
        k = self.readNumberOfClusters()
        if 1 <= k <= self.X.shape[0]:
            print(f"Executing Sklearn-IPCA ...")
            newData = decom.IncrementalPCA(n_components=components).fit_transform(self.X)
            self.setModelWithNewData(newData)
            self.applyClustering(k)
        else:
            print("Please try again")

    """
    Method that applies feature agglomeration over the data, and then applies clustering with the transformed data. Mainly used in the 
    option 1 of the main program
    """
    def doFeatureAgglomerationClustering(self, components):
        k = self.readNumberOfClusters()
        if 1 <= k <= self.X.shape[0]:
            print(f"Executing Feature agglomeration ...")
            newData = cluster.FeatureAgglomeration(n_clusters=components).fit_transform(self.X)
            self.setModelWithNewData(newData)
            self.applyClustering(k)
        else:
            print("Please try again")

    """
    Method that applies t-SNE over the data, and then applies clustering with the transformed data. Mainly used in the 
    option 1 of the main program
    """
    def dotSNEClustering(self, components):
        k = self.readNumberOfClusters()
        if 1 <= k <= self.X.shape[0]:
            print(f"Executing t-SNE ...")
            newData = TSNE(n_components=components, learning_rate='auto', init='random').fit_transform(self.X)
            self.setModelWithNewData(newData)
            self.applyClustering(k)
        else:
            print("Please try again")

    """
    Method that obtains the f1 score after applying feature agglomeration on data and then applying clustering.
    """
    def GetF1ScoreFeatureAgglomeration(self, k):
        if k > 0:
            # clustering_k = max(self.true_labels) + 1
            agglomerated_data = cluster.FeatureAgglomeration(n_clusters=k).fit_transform(self.X)
            new_data = pd.DataFrame(agglomerated_data)
            self.setModelWithNewData(new_data)
            applyClustering = getattr(self.model, "apply" + self.model_name)
            applyClustering(max(self.true_labels) + 1, seed=self.seed)
            pred_labels = self.model.getClusters()
            matrix = self.visu.getComConfusionMatrix(true_labels=self.true_labels, pred_labels=pred_labels)
            F1Score = self.visu.computeF1Score(matrix)
            return F1Score

    """
    Method the prints memory and time usage for the feature agglomeration method
    """
    def memory_and_time_used_agglo(self, model, data, k, n):
        times = np.zeros(n)
        memories = np.zeros(n)
        RAMs = np.zeros(n)
        for i in range(n):
            # Start time
            st = time.time()
            # Apply dimensionality reduction model
            agglo = model(n_clusters=k).fit_transform(self.X)
            et = time.time()
            # Execution time
            times[i] = et - st
            # Getting % usage of virtual_memory
            memories[i] = psutil.virtual_memory()[2]
            # Getting usage of virtual_memory in GB
            RAMs[i] = psutil.virtual_memory()[3] / 1000000000
        print('Average ecution time for', n, 'times:', times.mean(), 'seconds')
        print('Average RAM memory % used:', n, 'times:', memories.mean())
        print('Average RAM Used (GB):', n, 'times:', RAMs.mean())

    """
    Method that plots the points in the three principal components basing on their cluster and the feature reduction 
    technique used. It prints all the possible combinations
    """
    def plot_clusters(self, k=-1, minimum_variance=-1.0):
        PersonalPca = pca.PCA(self.X)
        PersonalPca.applyPCA(k=k, minimum_variance=minimum_variance)
        PCA_Data = PersonalPca.getTransformedData()

        t_sne = TSNE(random_state=123, n_components=3, learning_rate=200)
        T_sne_Data = t_sne.fit_transform(self.X)

        fa = cluster.FeatureAgglomeration(n_clusters=PCA_Data.shape[1])
        fa_data = fa.fit_transform(self.X)

        inc_pca = decom.IncrementalPCA(n_components=PCA_Data.shape[1])
        inc_pca_data = inc_pca.fit_transform(self.X)

        sub_titles = ["without dimensionality reduction",
                      f"With PCA for dim red to {PCA_Data.shape[1]} features & {minimum_variance * 100}% exp. variance",
                      f"With t-SNE for dim reduction to {PCA_Data.shape[1]} features",
                      f"With feature agglomeration for dim reduction to {PCA_Data.shape[1]} features",
                      f"with incremetal PCA for dim reduction to {PCA_Data.shape[1]} features"]

        for data, sub_title in zip([self.X, PCA_Data, T_sne_Data, fa_data, inc_pca_data], sub_titles):
            self.setModelWithNewData(data)
            new_labels = self.getClusteringLabels()
            self.visu.plotPrincipalFeaturesWithClustering(PCA_Data[:, 0], PCA_Data[:, 1], PCA_Data[:, 2],
                                                          labels=new_labels,
                                                          sub_title=f"PCA visualization {sub_title}")
            self.visu.plotPrincipalFeaturesWithClustering(T_sne_Data[:, 0], T_sne_Data[:, 1], T_sne_Data[:, 2], labels=new_labels,
                                                          sub_title=f"t-SNE visualization {sub_title}")


# Global variables

random_seed = 99999
datasets = ["cmc", "hepatitis", "satimage"]
# datasets= ["cmc"]
models = ["KMeans", "Agglomerative"]
'''
#######################################################################
                         Experiments
#######################################################################
'''

"""
Save the explained variance of all the datasets with all the components
"""
def saveExplainedVariancePlots():
    model_name = "KMeans"
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                 seed=random_seed)
        Experiment.studyExplainedVariance()



"""
Plot the three most important componenets (interactive)
"""
def plotMostImportantFeatures():
    model_name = "KMeans"
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                 seed=random_seed)
        # Experiment.studyFeatures(k=-1, minimum_variance=0.9)
        Experiment.studyFeatures(k=3, minimum_variance=-1)



"""
Plot the three most important componenets after applying clustering (interactive)
"""
def plotMostImportantFeaturesWithClustering(variance):
    for model_name in models:
        for dataset_name in datasets:
            df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
            Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=random_seed)
            Experiment.studyFeaturesWithClustering(k=-1, minimum_variance=variance)


"""
Save all the confusion matrices of all the datasets, using or not using PCA, with the KMeans and Agglomerative models
"""
def getConfusionMatrices(variance):
    for model_name in models:
        for dataset_name in datasets:
            df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
            Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=random_seed)
            Experiment.PCAandSaveConfusionMatrix(k=-1, minimum_variance=variance)
            Experiment.NoPCAandSaveConfusionMatrix()



"""
Print all the important data to compare the different techniques (part b)
"""
def compare_PCA():
    # Definition of the components depending on the dataset
    model_name = "KMeans"
    n_components_satimage = np.array([10, 5, 3])
    n_components_cmc = np.array([7, 5, 4, 2])
    n_components_hepatitis = np.array([15, 10, 5, 3])
    components_array = np.array([n_components_satimage, n_components_cmc, n_components_hepatitis])
    dataset_names = ["satimage", "cmc", "hepatitis"]
    for i, dataset_name in enumerate(dataset_names):
        print("=" * 80)
        print(f"{dataset_name} Dataset")
        print("=" * 80)
        # Consider different components
        n_components = components_array[i]
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                 seed=random_seed)
        for components in n_components:
            print(f"\nThe number of components is {components}:\n")
            Experiment.sklearn_PCA(components)
            Experiment.sklearn_IncrementalPCA(components)
            Experiment.implementation_PCA(components)
        print("-" * 70)

"""
Prints the memory usage of PCA and IPCA in all the dataasets
"""
def getMemoryUsageOfPCAandIPCA(components=3):
    model_name = "KMeans"
    for dataset_name in datasets:
        print("*" * 60)
        print(f"{dataset_name} Dataset")
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                 seed=random_seed)
        print("Sklearn-PCA memory and time stats: ")
        print("." * 60)
        Experiment.memory_and_time_used(decom.PCA, components=components, n=10)
        print("." * 60)
        print("Sklearn-IPCA memory and time stats: ")
        print("." * 60)
        Experiment.memory_and_time_used(decom.IncrementalPCA, components=components, n=10)




"""
Print the f-score from feature agglomeration with different number of features and find time and memory usage. (part d)
"""
def analyzeFeatureAgglomeration():
    random_seed = 33
    prosentages_features = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # f1 scores matrix. rows = dataset, colums = k-values
    f1scores = np.zeros((3, 9))
    dfs = []
    labels = []
    for dataset in datasets:
        df_labels = prep.preprocessDataset(f"./Datasets/{dataset}.arff")
        dfs.append(df_labels[0])
        labels.append(df_labels[1])
    for model in models:
        for i, dataset_name in zip(range(len(f1scores[0])), datasets):
            df = dfs[i]
            label = labels[i]
            feature_number = np.ones(len(prosentages_features))
            feature_number *= prosentages_features * len(
                df.columns)  # Computes the number for K equal to the percentage of the of the original features
            for j in range(len(feature_number)):
                # Compute F1 scores
                Exp = Experiments(df, labels=label, model_name=model, dataset_name=dataset_name, seed=random_seed)
                f1scores[i, j] = Exp.GetF1ScoreFeatureAgglomeration(k=round(feature_number[j]))
            plt.plot(f1scores[i], label=dataset_name)
            # Print the result from the best k value
            largestIndex = max(range(1, len(f1scores[0])), key=f1scores[i].__getitem__)
            print(
                f"Best number of clusters for {model} in {dataset_name} is {round(feature_number[largestIndex])} / {prosentages_features[largestIndex] * 100}% with a F1-score of {f1scores[i, largestIndex]}")

        plt.title(f"F1 scores for FeatureAgglomeration  and {model}")
        plt.legend()
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["5", "10", "20", "30", "40", "50", "60", "70", "80"])
        plt.xlabel("% fetures")
        plt.ylabel("F1 score")
        plt.show()

    # Compute the time and memory useage for feature agglomeration with different number of features.
    for k in [4, 8, 12]:
        print("\nPerformance for Feature agglomaeration k=", k, "Satimage data set")
        Exp.memory_and_time_used_agglo(cluster.FeatureAgglomeration, k=k, data=dfs[2], n=10)  # Done 1000 times in the test


"""
Plot a 3D scatter plot for each combination of models dimentional reduction, no reduction and visualized with pca and t-SNE (part e)
"""
def plot_all_cluster_combinations(variance):
    for model_name in models:
        for dataset_name in datasets:
            df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
            Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=random_seed)

            Experiment.plot_clusters(k=-1, minimum_variance=variance)


