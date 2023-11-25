import os
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
import Preprocessing as prep
from matplotlib import pyplot as plt
import KMeans
import KHarmonicMeans
import FuzzyCMeans
import Agglomerative
import MShift
import BisectingKMeans

pd.options.display.float_format = '{:.2f}'.format


class Experiments:

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

    def normalizeArray(self, data, min_def=-2, max_def=-2):
        Min = min_def
        Max = max_def
        if min_def == -2 and max_def == -2:
            Min = np.min(data)
            Max = np.max(data)
        if Max == Min: return data
        return (data - Min) / (Max - Min)

    def resetModel(self):
        self.model = self.class_(self.X)

    def getConfusionMatrixK(self, k, plot_title="Default", directory="./Plots"):
        # self.resetModel()
        applyClustering = getattr(self.model, "apply" + self.model_name)
        applyClustering(k, seed=self.seed)
        pred_labels = self.model.getClusters()
        self.saveConfusionMatrix(k, pred_labels, plot_title=plot_title, directory=directory)

    def saveConfusionMatrix(self, k, pred_labels, plot_title, directory="./Plots"):

        clusters_distributions = np.zeros((k, k))
        # Find cluster distributions
        for i in range(k):
            cluster_i = pred_labels[self.true_labels == i]
            cluster_i_distr = np.zeros(k)
            for j in range(k):
                cluster_i_distr[j] = np.sum(cluster_i == j)
            clusters_distributions[i] = cluster_i_distr

        # Find best sorting of clusters
        newDistribution = np.zeros((k, k))

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

            newDistribution[i] = clusters_distributions[best_distr_as_i]
            available_clusters.remove(best_distr_as_i)

        self.plotConfusionMatrix(newDistribution, k, plot_title=plot_title, directory=directory)

    def plotConfusionMatrix(self, newDistribution, k, plot_title, directory):
        # Plot the confusion matrix
        # accuracy = round(np.trace(newDistribution) * 100 / np.sum(newDistribution), 2)
        f1score = self.computeF1Score(newDistribution)
        for j in range(k):
            newDistribution[j] = newDistribution[j] / np.sum(newDistribution[j])

        names = [f"Cluster {i}" for i in range(k)]
        palette = sns.color_palette("light:b", as_cmap=True)
        df_cm = pd.DataFrame(newDistribution, index=names, columns=names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap=palette, vmin=0, vmax=1, annot_kws={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'sans'
        })
        plt.xticks(fontsize=14, fontfamily="sans")
        plt.yticks(fontsize=14, fontfamily="sans", rotation=0)
        plt.suptitle(plot_title, fontsize=16, fontweight="bold", fontfamily="sans")
        plt.title(f"With f1-score of {f1score}", fontsize=15, pad=20)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{plot_title}.png', bbox_inches='tight')
        print(f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset is saved on {directory}")
        plt.show()

    def computeF1Score(self, confusionMatrix):
        k = len(confusionMatrix)
        precisions = np.zeros(k)
        recalls = np.zeros(k)
        for i in range(k):
            TP = confusionMatrix[i, i]
            FP = np.sum(confusionMatrix[:, i]) - TP
            FN = np.sum(confusionMatrix[i]) - TP
            if TP + FP != 0:
                precisions[i] = TP / (TP + FP)
            if TP + FN != 0:
                recalls[i] = TP / (TP + FN)
        avg_precision = np.average(precisions)
        avg_recall = np.average(recalls)
        return round((2 * avg_recall * avg_precision) / (avg_recall + avg_precision), 2)

    def studyKValues(self, Ks):
        davies_bouldin_scores = np.empty(0)
        silhouette_scores = np.empty(0)
        adjusted_rand_scores = np.empty(0)
        adjusted_mutual_scores = np.empty(0)

        for k in Ks:
            self.K = k
            print(f"Computing metrics for K = {k}")
            applyClustering = getattr(self.model, "apply" + self.model_name)
            applyClustering(k, seed=self.seed)
            predicted_labels = self.model.getClusters()

            score = davies_bouldin_score(self.X, predicted_labels)
            davies_bouldin_scores = np.append(davies_bouldin_scores, score)

            score = silhouette_score(self.X, predicted_labels)
            silhouette_scores = np.append(silhouette_scores, score)

            score = adjusted_rand_score(self.true_labels, predicted_labels)
            adjusted_rand_scores = np.append(adjusted_rand_scores, score)

            score = adjusted_mutual_info_score(self.true_labels, predicted_labels)
            adjusted_mutual_scores = np.append(adjusted_mutual_scores, score)

            self.resetModel()

        davies_bouldin_scores = self.normalizeArray(davies_bouldin_scores, 0, np.max(davies_bouldin_scores))
        silhouette_scores = self.normalizeArray(silhouette_scores, -1, 1)
        adjusted_rand_scores = self.normalizeArray(adjusted_rand_scores, -1, 1)
        adjusted_mutual_scores = self.normalizeArray(adjusted_mutual_scores, -1, 1)

        davies_k_point = np.argmin(davies_bouldin_scores)
        silhouette_k_point = np.argmax(silhouette_scores)
        adjusted_rande_k_point = np.argmax(adjusted_rand_scores)
        adjusted_mutual_k_point = np.argmax(adjusted_mutual_scores)

        plt.plot(Ks, davies_bouldin_scores, label="Davies Bouldin (Min.)", linestyle=":")
        plt.plot(Ks[davies_k_point], davies_bouldin_scores[davies_k_point], color="black", marker="*")
        plt.plot(Ks, silhouette_scores, label="Silhouette (Max.)", linestyle="-.")
        plt.plot(Ks[silhouette_k_point], silhouette_scores[silhouette_k_point], color="black", marker="*")
        plt.plot(Ks, adjusted_rand_scores, label="Adjusted Rand (Max.)", linestyle="-")
        plt.plot(Ks[adjusted_rande_k_point], adjusted_rand_scores[adjusted_rande_k_point], color="black", marker="*")
        plt.plot(Ks, adjusted_mutual_scores, label="Adjusted Mutual Info (Max.)", linestyle="--")
        plt.plot(Ks[adjusted_mutual_k_point], adjusted_mutual_scores[adjusted_mutual_k_point], color="black",
                 marker="*")

        plot_title = f"Num. of clusters for {self.dataset_name} dataset with {self.model_name}"
        plt.xlabel("K values", fontsize=14, fontfamily="sans")
        plt.ylabel("Scores", fontsize=14, fontfamily="sans")
        plt.legend(title="Clustering Metric", title_fontproperties={'weight': "bold"})
        plt.xticks(Ks, fontsize=12, fontfamily="sans")
        plt.yticks(fontsize=12, fontfamily="sans")
        plt.title(plot_title, fontsize=16, fontweight="bold", fontfamily="sans", pad=20)
        MetricsPath = './Plots/Metrics'
        if not os.path.exists(MetricsPath):
            os.makedirs(MetricsPath)
        plt.savefig(f'{MetricsPath}/{self.dataset_name}/{plot_title}.png', bbox_inches='tight')
        print(
            f"Metrics plot of {self.model_name} model with {self.dataset_name} dataset is saved on {MetricsPath}/{self.dataset_name}")
        plt.show()

    def KMeansDistances(self):
        if self.model_name != "KMeans":
            print("This experiment is not possible with this model")
            return

        applyClustering = getattr(self.model, "apply" + self.model_name)
        distances = ["euclidean", "manhattan", "cosine"]
        for dist in distances:
            applyClustering(self.K, seed=self.seed, distance=dist)
            predicted_labels = self.model.getClusters()
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset and {dist} distance"
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/KMeansDistances")
            self.resetModel()

    def KMeansInitializations(self):
        if self.model_name != "KMeans":
            print("This experiment is not possible with this model")
            return

        applyClustering = getattr(self.model, "apply" + self.model_name)
        Initializations = ["plusPlus", "forgy"]
        for initi in Initializations:
            applyClustering(self.K, seed=self.seed, initialization=initi)
            predicted_labels = self.model.getClusters()
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset and {initi} initialization"
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/KMeansInitializations")
            self.resetModel()

    def AgglomerativeLinkages(self):
        if self.model_name != "Agglomerative":
            print("This experiment is not possible with this model")
            return

        applyClustering = getattr(self.model, "apply" + self.model_name)
        linkages = ['complete', 'ward', 'average', 'single']
        for linkage in linkages:
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset and {linkage} linkage"
            applyClustering(self.K, seed=self.seed, linkage=linkage)
            predicted_labels = self.model.getClusters()
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/AgglomerativeLinkages")

    def MeanShiftBinSeeding(self):
        if self.model_name != "MShift":
            print("This experiment is not possible with this model")
            return

        applyClustering = getattr(self.model, "apply" + self.model_name)
        bin_seedings = [False, True]
        for value in bin_seedings:
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset when bin_seeding = {value}"
            applyClustering(self.K, bin_seeding=value)
            predicted_labels = self.model.getClusters()
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/MeanShiftBinSeading")

    def StudyPValue(self, Prange):
        if self.model_name != "KHarmonicMeans":
            print("This experiment is not possible with this model")
            return

        for p in Prange:
            applyClustering = getattr(self.model, "apply" + self.model_name)
            applyClustering(self.K, p=p, seed=self.seed)
            predicted_labels = self.model.getClusters()
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset and p = {p}"
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/KHarmonicPValues")
            self.resetModel()

    def StudyMValue(self, Mrange):
        if self.model_name != "FuzzyCMeans":
            print("This experiment is not possible with this model")
            return

        for m in Mrange:
            applyClustering = getattr(self.model, "apply" + self.model_name)
            applyClustering(self.K, m=m, seed=self.seed)
            predicted_labels = self.model.getClusters()
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset and m = {m}"
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/FuzzyCMeansMValues")
            self.resetModel()

    def getClusters(self, k):
        self.K = k
        applyClustering = getattr(self.model, "apply" + self.model_name)
        applyClustering(self.K, seed=self.seed)
        print(
            f"The centers of the {self.dataset_name} dataset and {self.K} clusters with the {self.model_name} models are:")
        print(self.model.getCenters())

    def BisectingKMeansChoice(self):
        if self.model_name != "BisectingKMeans":
            print("This experiment is not possible with this model")
            return

        applyClustering = getattr(self.model, "apply" + self.model_name)
        choices = ["sse", "largest", "mix"]
        for choice in choices:
            applyClustering(self.K, seed=self.seed, cluserSelectionType=choice)
            predicted_labels = self.model.getClusters()
            plot_title = f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset and {choice} distance"
            self.saveConfusionMatrix(self.K, predicted_labels, plot_title=plot_title,
                                     directory="./Plots/BisectingKMeansChoices")
            self.resetModel()

    def computeDendrogram(self):
        self.model.setDatasetName(self.dataset_name)
        applyClustering = getattr(self.model, "apply" + self.model_name)
        applyClustering(k=2, seed=self.seed, dendro=True)


# Global variables

random_seed = 99999
datasets = ["cmc", "hepatitis", "satimage"]
models = ["KMeans", "BisectingKMeans", "KHarmonicMeans", "FuzzyCMeans", "Agglomerative"]


# Plot metrics and confusion matrix of each model for each k
def saveMetricsAndConfusionMatrixAll():
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        for model_name in models:
            Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=random_seed)
            if model_name != "MShift":
                print(f"Computing metrics of: {model_name}")
                Experiment.studyKValues(range(2, 9))

            print(f"Computing confusion matrix of: {model_name}")
            k = len(np.unique(labels))
            plot_title = f"Confusion Matrix of {model_name} model with {dataset_name} dataset"
            Experiment.getConfusionMatrixK(k, plot_title=plot_title,
                                           directory=f"./Plots/ConfusionMatrix/{dataset_name}")


# Best distance for KMeans
def runKMeansDistancesExperiment():
    model_name = "KMeans"
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=random_seed)
        Experiment.KMeansDistances()


# Best initialization for KMeans
def runKMeansIntializationExperiment():
    model_name = "KMeans"
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=random_seed)
        Experiment.KMeansInitializations()


# Best linkage for agglomerative
def runAgglomerativeLinkagExperiment():
    model_name = "Agglomerative"
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=random_seed)
        Experiment.AgglomerativeLinkages()


# Best choice method for BisectingKMeans
def runBisectingKMeansChoiceExperiment():
    model_name = "BisectingKMeans"
    for dataset_name in datasets:
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=random_seed)
        Experiment.BisectingKMeansChoice()


# Best P for KHarmonicMeans
def runKHarmonicMeansPExperiment():
    minM = int(input(f"Enter the starting p (p >= 2): \n"))
    maxM = int(input(f"Enter the ending p (p >= 2): \n"))
    if maxM >= minM >= 2:
        model_name = "KHarmonicMeans"
        for dataset_name in datasets:
            df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
            Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=random_seed)
            Experiment.StudyPValue(range(minM, maxM + 1))
    else:
        print("Please try again")


# Best M for FuzzyCMeans
def runFuzzyCMeansMExperiment():
    minM = int(input(f"Enter the starting m (m >= 2): \n"))
    maxM = int(input(f"Enter the ending m (m >= 2): \n"))
    if maxM >= minM >= 2:
        model_name = "FuzzyCMeans"
        for dataset_name in datasets:
            df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
            Experiment = Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=random_seed)
            Experiment.StudyMValue(range(minM, maxM + 1))
    else:
        print("Please try again")
