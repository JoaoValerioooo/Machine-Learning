import numpy as np
import Experiments as Exp
import Preprocessing as prep


def print_long_line(n):
    print("=" * n)


dataset_list = ["cmc", "hepatitis", "satimage"]


def get_model_dataset_seed(questions, models_list):
    print_long_line(50)
    print(questions[0])
    for i, model in enumerate(models_list):
        print(f"{i}. {model}")

    m = int(input("Enter a number between 0 and 5:\n"))
    print(questions[1])
    for i, dataset in enumerate(dataset_list):
        print(f"{i}. {dataset}")
    d = int(input("Enter a number between 0 and 2:\n"))
    seed = int(input("Enter a seed number (default = 99999):\n"))
    return m, d, seed


def operation_0():
    models_list = ["MShift", "KMeans", "BisectingKMeans", "KHarmonicMeans", "FuzzyCMeans"]
    questions = ["Select the model to compute the clustering:",
                 "Select the dataset you are interested in:"]
    m, d, seed = get_model_dataset_seed(questions=questions, models_list=models_list)
    if 0 <= m <= 5 and 0 <= d <= 2:
        dataset_name = dataset_list[d]
        model_name = models_list[m]
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        k = 2
        if model_name != "MShift":
            k = int(input(f"Enter the number of clusters (between 2 and {len(df)}): \n"))
        if 2 <= k <= len(df):
            Experiment = Exp.Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=seed)
            print("Computing clusters centers...")
            print("Please wait...")
            Experiment.getClusters(k)
        else:
            print("Please try again")
    else:
        print("Please try again")


def operation_1():
    print("Select the dataset you are interested in:")
    for i, dataset in enumerate(dataset_list):
        print(f"{i}. {dataset}")
    d = int(input("Enter a number between 0 and 2:\n"))
    if 0 <= d <= 2:
        dataset_name = dataset_list[d]
        model_name = "Agglomerative"
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Exp.Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name,
                                     seed=99999)
        print("Computing dendrogram...")
        print("Please wait...")
        Experiment.computeDendrogram()
    else:
        print("Please try again")


def operation_2():
    models_list = ["Agglomerative", "KMeans", "BisectingKMeans", "KHarmonicMeans", "FuzzyCMeans"]
    questions = ["Select the model of which you want the clustering metrics:",
                 "Select the dataset you are interested in:"]
    m, d, seed = get_model_dataset_seed(questions=questions, models_list=models_list)
    if 0 <= m <= 5 and 0 <= d <= 2:
        dataset_name = dataset_list[d]
        model_name = models_list[m]
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        n = len(df)
        print("Now we will define the range of the possible Ks")
        minK = int(input(f"Enter the minimum k (between 2 and {n}): \n"))
        maxK = int(input(f"Enter the maximum k (between 2 and {n}): \n"))
        if 2 <= minK <= n and 2 <= maxK <= n and minK <= maxK:
            Experiment = Exp.Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=seed)
            print("Computing clustering metrics...")
            print("Please wait...")
            Experiment.studyKValues(range(minK, maxK + 1))
        else:
            print("Please try again")
    else:
        print("Please try again")


def operation_3():
    models_list = ["Agglomerative", "MeanShift", "KMeans", "BisectingKMeans", "KHarmonicMeans", "FuzzyCMeans"]
    questions = ["Select the model of which you want the confusion matrix:",
                 "Select the dataset you are interested in:"]
    m, d, seed = get_model_dataset_seed(questions=questions, models_list=models_list)
    if 0 <= m <= 5 and 0 <= d <= 2:
        dataset_name = dataset_list[d]
        model_name = models_list[m]
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        k = len(np.unique(labels))
        Experiment = Exp.Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=seed)
        plot_title = f"Confusion Matrix of {model_name} model with {dataset_name} dataset"
        directory = f"./Plots/ConfusionMatrix/{dataset_name}"
        print("Computing confusion matrix...")
        print("Please wait...")
        Experiment.getConfusionMatrixK(k, plot_title, directory=directory)
    else:
        print("Please try again")


def operation_4():
    print_long_line(50)
    Experiments_names = ["Plot the metrics and the confusion matrix for all the models and all the datasets",
                         "Distances for KMeans", "Initialization techniques for KMeans",
                         "Methods for picking the cluster to split in BisectingKMeans",
                         "Linkages for Agglomerative clustering", "Try different values of P for KHarmonicMeans",
                         "Try different values of M for FuzzyCMeans"]
    print("Select the experiment you want you want to run:")
    for i, experi in enumerate(Experiments_names):
        print(f"{i}. {experi}")
    n = len(Experiments_names)
    k = int(input(f"Enter the number of the experiment (between 0 and {n - 1}): \n"))
    if 0 <= k <= n:
        if k == 0:
            Exp.saveMetricsAndConfusionMatrixAll()
        elif k == 1:
            Exp.runKMeansDistancesExperiment()
        elif k == 2:
            Exp.runKMeansIntializationExperiment()
        elif k == 3:
            Exp.runBisectingKMeansChoiceExperiment()
        elif k == 4:
            Exp.runAgglomerativeLinkagExperiment()
        elif k == 5:
            Exp.runKHarmonicMeansPExperiment()
        elif k == 6:
            Exp.runFuzzyCMeansMExperiment()
    else:
        print("Please try again")


def main():
    print_long_line(50)
    print("What of the following actions you want to do?")
    print("0. Compute the cluster centers of a model with a particular dataset")
    print("1. Compute the dendrogram of a particular dataset")
    print("2. Compute clustering validation metrics")
    print("3. Compute Confusion Matrix of a model with a particular dataset")
    print("4. Run Experiments")
    op = int(input("Enter a number between 0 and 4:\n"))
    if op == 0:
        operation_0()
    elif op == 1:
        operation_1()
    elif op == 2:
        operation_2()
    elif op == 3:
        operation_3()
    elif op == 4:
        operation_4()


if __name__ == "__main__":
    main()
