from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import os

class Agglomerative:

    def __init__(self, X):
        self.X = X.to_numpy()
        self.K = 2
        self.clusters = np.empty(0)
        self.random_seed = 1234
        self.dataset_name = "Default"

    def getClusters(self):
        return self.clusters

    def setDatasetName(self, name):
        self.dataset_name = name

    def applyAgglomerative(self, k, seed=123456, linkage='ward', dendro=False):
        self.K = k
        agglomerative_clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        if dendro:
            agglomerative_clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage)
            agglomerative_clustering_model = agglomerative_clustering.fit(self.X)
            self.plot_dendrogram(agglomerative_clustering_model, truncate_mode="level", p=0)
        else:
            agglomerative_clustering_model = agglomerative_clustering.fit(self.X)
            self.clusters = agglomerative_clustering_model.labels_

    def plot_dendrogram(self, model, **kwargs):
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        plot_title = f"Dendrogram of  {self.dataset_name} dataset"
        MetricsPath = './Plots/Dendrograms'
        if not os.path.exists(MetricsPath):
            os.makedirs(MetricsPath)
        plt.title(plot_title, fontsize=16, fontweight="bold", fontfamily="sans", pad=20)
        plt.savefig(f'{MetricsPath}/{plot_title}.png', bbox_inches='tight')
        print(
            f"{plot_title} is saved on {MetricsPath}")
        plt.show()
