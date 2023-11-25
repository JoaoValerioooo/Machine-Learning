import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


class Visualize:

    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name

    def plotPrincipalFeatures(self, feature1, feature2, feature3):
        mpl.use('Qt5Agg')
        directory = "./Plots/FeaturesPlots"
        ax = plt.axes(projection='3d')
        ax.grid(color='w', linestyle='solid')
        for spine in ax.spines.values():
            spine.set_visible(False)

        plot_title = f"Three principal components of the {self.dataset_name} dataset"
        ax.set_title(plot_title, fontweight="bold", fontfamily='sans-serif')
        ax.scatter(feature1, feature2, feature3, s=4, edgecolors='None')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.show()

    def plotPrincipalFeaturesWithClustering(self, feature1, feature2, feature3, labels, sub_title=""):
        mpl.use('Qt5Agg')
        directory = "./Plots/FeaturesPlots"
        ax = plt.axes(projection='3d')
        ax.grid(color='w', linestyle='solid')
        for spine in ax.spines.values():
            spine.set_visible(False)

        plot_title = f"Clustering of {self.dataset_name} dataset using {self.model_name}"
        plt.suptitle(plot_title, fontsize=14, fontweight="bold", fontfamily='sans-serif')
        ax.set_title(sub_title, fontsize=12, fontfamily='sans-serif')

        k = len(np.unique(labels))
        for label in range(k):
            ax.scatter(feature1[labels == label], feature2[labels == label], feature3[labels == label], s=4,
                       edgecolors='None', label=f"Cluster {label}")
        lgnd = ax.legend()
        for handle in lgnd.legendHandles:
            handle.set_sizes([30])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.show()

    def plotPrincipalFeaturesWithClustering_subTitles(self, feature1, feature2, feature3, sub_title, labels=[]):
        mpl.use('Qt5Agg')
        directory = "./Plots/FeaturesPlots"
        ax = plt.axes(projection='3d')
        ax.grid(color='w', linestyle='solid')
        for spine in ax.spines.values():
            spine.set_visible(False)

        plot_title = f"Clustering of {self.dataset_name} dataset using {self.model_name}"
        undertitle = sub_title
        plt.suptitle(plot_title, fontsize=14, fontweight="bold", fontfamily='sans-serif')
        ax.set_title(undertitle, fontsize=12, fontfamily='sans-serif')

        k = len(np.unique(labels))
        for label in range(k):
            ax.scatter(feature1[labels == label], feature2[labels == label], feature3[labels == label], s=4,
                       edgecolors='None', label=f"Cluster {label}")
        lgnd = ax.legend()
        for handle in lgnd.legendHandles:
            handle.set_sizes([30])
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.show()

    def plotExplainedVariance(self, explainedVariance):
        cum_sum_exp = np.cumsum(explainedVariance)

        ax = plt.axes(facecolor='#E6E6E6')
        plot_title = f"Explained variance of the {self.dataset_name} dataset"
        directory = "./Plots/ExplainedVariance"
        ax.set_title(plot_title, fontsize=14, fontweight="bold", fontfamily='sans-serif')
        # ax = plt.gca()

        ax.set_axisbelow(True)
        ax.grid(color='w', linestyle='solid')
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('gray')
        for tick in ax.get_yticklabels():
            tick.set_color('gray')

        ax.bar(range(0, len(explainedVariance)), explainedVariance, align='center',
               label='Component explained variance', color='#EE6666')
        ax.step(range(0, len(cum_sum_exp)), cum_sum_exp, where='mid', label='Cumulative explained variance',
                color='#EE6666')
        ax.set_ylabel('Explained variance ratio')
        ax.set_xlabel('Principal component index')
        ax.set_xticks(ticks=range(0, len(explainedVariance), 2))
        ax.set_yticks(ticks=np.linspace(0, 1, 11))
        ax.legend(loc='best')
        if not os.path.exists(directory):
            os.makedirs(directory)
        ax.figure.savefig(f'{directory}/{plot_title}.png', bbox_inches='tight')
        print(f"Saving {plot_title} in the directory {directory}")
        ax.figure.show()

    def saveConfusionMatrix(self, true_labels, pred_labels, plot_title, directory="./Plots/ConfusionMatrix"):
        k = len(np.unique(true_labels))
        clusters_distributions = np.zeros((k, k))
        # Find cluster distributions
        for i in range(k):
            cluster_i = pred_labels[true_labels == i]
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
        plt.suptitle(plot_title, fontsize=16, fontweight="bold", fontfamily='sans-serif')
        plt.title(f"With f1-score of {f1score}", fontsize=15, pad=20)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{plot_title}.png', bbox_inches='tight')
        print(f"Confusion Matrix of {self.model_name} model with {self.dataset_name} dataset is saved on {directory}")
        # plt.show()

    def getComConfusionMatrix(self, true_labels, pred_labels):
        k = len(np.unique(true_labels))
        clusters_distributions = np.zeros((k, k))
        # Find cluster distributions
        for i in range(k):
            cluster_i = pred_labels[true_labels == i]
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
        return newDistribution

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

