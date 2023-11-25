import numpy as np
import Experiments as Exp
import Preprocessing as prep


def print_long_line(n):
    print("=" * n)


dataset_list = ["cmc", "hepatitis", "satimage"]
model_list = ["KMeans", "Agglomerative"]


def get_answers(questions, listOptions):
    print_long_line(50)
    answers = []
    for i, question in enumerate(questions):
        print(question)
        options = listOptions[i]
        for j, op in enumerate(options):
            print(f"{j}. {op}")
        x = int(input(f"Enter a number between 0 and {len(options) - 1}:\n"))
        answers.append(x)
    return answers


def operation_0():
    model_name = "KMeans"
    seed = 9999
    fReductionOptions = ["PCA", "Sklearn-PCA", "Sklearn-IPCA", "Feature Agglomeration", "t-SNE"]
    questions = ["Select the dataset you are interested in:",
                 "Select the feature reduction option you are interested in:"]
    listOptions = [dataset_list, fReductionOptions]
    answ = get_answers(questions, listOptions)
    if 0 <= answ[0] <= 2 and 0 <= answ[1] <= 4:
        dataset_name = dataset_list[answ[0]]
        option = fReductionOptions[answ[1]]
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Exp.Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=seed)
        if option == "PCA":
            components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
            Experiment.doPCA(components)
        elif option == "Sklearn-PCA":
            components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
            Experiment.doSklearnPCA(components)
        elif option == "Sklearn-IPCA":
            components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
            Experiment.doSklearnIPCA(components)
        elif option == "Feature Agglomeration":
            components = int(input(f"Enter the number of feature clusters between 1 and {df.shape[1]}:\n"))
            Experiment.doFeatureAgglomeration(components)
        elif option == "t-SNE":
            components = int(input(f"Enter the number of components between 1 and 3:\n"))
            Experiment.dotSNE(components)
    else:
        print("Please try again")


def operation_1():
    model_name = "KMeans"
    seed = 9999
    fReductionOptions = ["PCA", "Sklearn-PCA", "Sklearn-IPCA", "Feature Agglomeration", "t-SNE"]
    questions = ["Select the dataset you are interested in:",
                 "Select the model you want to use:",
                 "Select the feature reduction option you are interested in:"]
    listOptions = [dataset_list, model_list, fReductionOptions]
    answ = get_answers(questions, listOptions)
    if 0 <= answ[0] <= 2 and 0 <= answ[1] <= 1 and 0 <= answ[2] <= 4:

        dataset_name = dataset_list[answ[0]]
        model_name = model_list[answ[1]]
        option = fReductionOptions[answ[2]]
        df, labels = prep.preprocessDataset(f"./Datasets/{dataset_name}.arff")
        Experiment = Exp.Experiments(df, labels=labels, model_name=model_name, dataset_name=dataset_name, seed=seed)
        if option == "PCA":
            components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
            Experiment.doPCAClustering(components)
        elif option == "Sklearn-PCA":
            components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
            Experiment.doSklearnPCAClustering(components)
        elif option == "Sklearn-IPCA":
            components = int(input(f"Enter the number of components between 1 and {df.shape[1]}:\n"))
            Experiment.doSklearnIPCAClustering(components)
        elif option == "Feature Agglomeration":
            components = int(input(f"Enter the number of feature clusters between 1 and {df.shape[1]}:\n"))
            Experiment.doFeatureAgglomerationClustering(components)
        elif option == "t-SNE":
            components = int(input(f"Enter the number of components between 1 and 3:\n"))
            Experiment.dotSNEClustering(components)
    else:
        print("Please try again")


def operation_2():
    print_long_line(50)
    Experiments_names = ["Compute the explained variance for all the datasets",
                         "Plot the 3 most important features of the datasets using PCA",
                         "Plot the 3 most important features of the datasets with clustering methods",
                         "Get all the confusion matrices for all models (with or without PCA)",
                         "Print the results in the terminal to compare all the PCA techniques",
                         "Print memory usage of PCA and IPCA",
                         "Print all the data necessary for feature agglomeration study",
                         "Plot all the clustering possibilities with all the techniques (Be aware that there are a lot)"]
    print("Select the experiment you want you want to run:")
    for i, experi in enumerate(Experiments_names):
        print(f"{i}. {experi}")
    n = len(Experiments_names)
    k = int(input(f"Enter the number of the experiment (between 0 and {n - 1}): \n"))
    if 0 <= k <= n:
        if k == 0:
            Exp.saveExplainedVariancePlots()
        elif k == 1:
            Exp.plotMostImportantFeatures()
        elif k == 2:
            variance = float(input(f"Enter the minimum explained variance (between 0 and 1):\n"))
            if 0 <= variance <= 1:
                Exp.plotMostImportantFeaturesWithClustering(variance)
            else:
                print("Please try again")
        elif k == 3:
            variance = float(input(f"Enter the minimum explained variance (between 0 and 1):\n"))
            if 0 <= variance <= 1:
                Exp.getConfusionMatrices(variance)
            else:
                print("Please try again")
        elif k == 4:
            Exp.compare_PCA()
        elif k == 5:
            Exp.getMemoryUsageOfPCAandIPCA()
        elif k == 6:
            Exp.analyzeFeatureAgglomeration()
        elif k == 7:
            variance = float(input(f"Enter the minimum explained variance (between 0 and 1):\n"))
            if 0 <= variance <= 1:
                Exp.plot_all_cluster_combinations(variance)
            else:
                print("Please try again")
    else:
        print("Please try again")


def main():
    print_long_line(50)
    print("What of the following actions you want to do?")
    print("0. Apply feature reduction to a particular dataset")
    print("1. Apply feature reduction to a dataset and compute the clustering")
    print("2. Run Experiments")
    op = int(input("Enter a number between 0 and 2:\n"))
    if op == 0:
        operation_0()
    elif op == 1:
        operation_1()
    elif op == 2:
        operation_2()


if __name__ == "__main__":
    main()
