import datetime
import time
from sklearn.metrics import accuracy_score
import GCNN
import ENNTh
import DROP3
import Preprocessing as prep
import kNNAlgorithm as kNNalgo
import reductionKNNAlgorithm as redKNN
import numpy as np
import pandas as pd
import os
import StatisticalTests as stats


def saveReducedFold(X_train, X_test, y_train, y_test, dataset_name, fold_number, option, tech="GCNN"):
    name_op, value_op = option
    new_train = X_train.copy()
    new_train['class'] = y_train
    same_test = X_test.copy()
    same_test['class'] = y_test

    directory = f"./ReducedDatasets/{tech}/{dataset_name}/{name_op}{value_op}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fold_name = f"{dataset_name}.fold.00000{int(fold_number)}"
    new_train.to_csv(f'{directory}/{fold_name}.train.csv', index=False)
    same_test.to_csv(f'{directory}/{fold_name}.test.csv', index=False)


def RunExperiments(dataset_name, Ks, weights, distances, votings, extraname="nothing"):
    assert dataset_name in ["satimage", "cmc"], "Please give a dataset name which is satimage or cmc"
    assert Ks, "Ks list is empty.\n" + " " * 3 + "Possible options are: 1, 3, 5, 7"
    assert weights, "Weights list is empty.\n" + " " * 3 + "Possible options are: Equal, InformationGain, Chi"
    assert distances, "distances list is empty.\n" + " " * 3 + "Possible options are: Minkowski, Cosine, Canberra"
    assert votings, "votings list is empty.\n" + " " * 3 + "Possible options are: Majority, " \
                                                           "InverseDistanceWeighted, Sheppards "
    firstStart = time.time()
    print(f"Reading and preprocessing the crossfolds of {dataset_name} dataset ...")
    Crossfolds = prep.preprocessCrossFolds(f"./datasets/{dataset_name}", encodeCategorical=False)

    acc_df = pd.DataFrame()
    times_df = pd.DataFrame()

    folds_names = list(range(10))

    for K in Ks:
        print("-" * 50)
        print(f"Doing experiments with K = {K}:")
        print("-" * 50)
        acc_values = []
        time_values = []
        for distance in distances:
            print(" " * 2 + f"Distance: {distance}")
            for weight in weights:
                print(" " * 4 + f"Weight: {weight}")
                for voting in votings:
                    print(" " * 6 + f"Voting: {voting}")
                    comb_name = f"{K}_{distance}_{weight}_{voting}"
                    accs = np.zeros(10)
                    times = np.zeros(10)
                    i = 0
                    for train_data, test_data in Crossfolds:
                        # print(" " * 8 + f"fold nº {i} ...")
                        X_train, y_train = train_data
                        X_test, y_test = test_data
                        begin = time.time()
                        Knn = kNNalgo.kNNAlgorithm(K)
                        Knn.fit(X_train, y_train, voting=voting, distance=distance, weighting=weight)
                        y_pred = Knn.predict(X_test)
                        times[i] = time.time() - begin
                        accs[i] = accuracy_score(y_test, y_pred, normalize=True)
                        i += 1
                    acc_df[comb_name] = accs
                    times_df[comb_name] = times

    acc_df.index = folds_names
    times_df.index = folds_names
    print("-" * 50)
    TotalElapsed = time.time() - firstStart
    ExecutionTime = time.strftime("%Hh%Mm%Ss", time.gmtime(TotalElapsed))
    print(f"Total execution time of the experiments: {ExecutionTime}")
    directory = "./ResultFiles"
    if not os.path.exists(directory):
        os.makedirs(directory)

    extra_name = ""
    if extraname != "nothing":
        timestamp = '{:%d_%m_%Y__%Hh_%Mm_%Ss}'.format(datetime.datetime.now())
        extra_name = f"{extraname}_{timestamp}"
    acc_df.to_csv(f'{directory}/{dataset_name}_accuracies{extra_name}.csv', index=False)
    times_df.to_csv(f'{directory}/{dataset_name}_times{extra_name}.csv', index=False)
    print("-" * 50)
    print(f"The following result files were saved in the directory {directory}")
    print(" " * 3 + f"{dataset_name}_accuracies{extra_name}.csv")
    print(" " * 3 + f"{dataset_name}_times{extra_name}.csv")


def RunAllCombinations(dataset_name="cmc"):
    Ks = [1, 3, 5, 7]
    weights = ["Equal", "InformationGain", "Chi"]
    distances = ["Minkowski", "Cosine", "Canberra"]
    votings = ["Majority", "InverseDistanceWeighted", "Sheppards"]
    RunExperiments(dataset_name, Ks=Ks, weights=weights, distances=distances, votings=votings)


def RunParticularCombination(dataset_name, Ks, weights, distances, votings):
    assert dataset_name in ["satimage", "cmc"], "Please give a dataset name which is satimage or cmc"
    assert Ks, "Ks list is empty.\n" + " " * 3 + "Possible options are: 1, 3, 5, 7"
    assert weights, "Weights list is empty.\n" + " " * 3 + "Possible options are: Equal, InformationGain, Chi"
    assert distances, "distances list is empty.\n" + " " * 3 + "Possible options are: Minkowski, Cosine, Canberra"
    assert votings, "votings list is empty.\n" + " " * 3 + "Possible options are: Majority, " \
                                                           "InverseDistanceWeighted, Sheppards "
    extra_name = "_particular_combination"
    RunExperiments(dataset_name, Ks=Ks, weights=weights, distances=distances, votings=votings, extraname=extra_name)


def readAndSort(dataset_name):
    assert dataset_name in ["satimage", "cmc"], "Please give a dataset name which is satimage or cmc"
    assert os.path.exists(
        f"./ResultFiles/{dataset_name}/{dataset_name}_accuracies.csv"), "Please run experiments of all the " \
                                                                        f"combinations of {dataset_name} dataset"

    accuracies = pd.read_csv(f"./ResultFiles/{dataset_name}/{dataset_name}_accuracies.csv")
    times = pd.read_csv(f"./ResultFiles/{dataset_name}/{dataset_name}_times.csv")
    accuracies.index = accuracies["K"]
    times.index = times["K"]
    jointDataset = pd.DataFrame(columns=["combination", "accuracy", "time"])
    Ks = times["K"]
    for k in Ks:
        for col in accuracies.columns[1:]:
            comb = f"{int(k)}_{col}"
            acc = accuracies.loc[k, col]
            tim = times.loc[k, col]
            row = [comb, acc, tim]
            jointDataset.loc[comb] = row
    jointDataset.reset_index(drop=True, inplace=True)
    accuracies_sorted = jointDataset.sort_values(by="accuracy", ascending=False)
    times_sorted = jointDataset.sort_values(by="time")

    directory = f"./ResultFiles/{dataset_name}"
    jointDataset.to_csv(f'{directory}/{dataset_name}_results.csv', index=False)
    accuracies_sorted.to_csv(f'{directory}/{dataset_name}_results_by_accuracy.csv', index=False)
    times_sorted.to_csv(f'{directory}/{dataset_name}_results_by_time.csv', index=False)
    print(f"The following result files were saved in the directory {directory}")
    print(" " * 3 + f"{dataset_name}_results.csv")
    print(" " * 3 + f"{dataset_name}_results_by_accuracy.csv")
    print(" " * 3 + f"{dataset_name}_results_by_time.csv")


def RunGCNNExperiment(rho=0.1, dataset_name="cmc", K=5, reduce=True, save=True, **kwargs):
    firstStart = time.time()
    print(f"Reading and preprocessing the crossfolds of {dataset_name} dataset ...")
    Crossfolds = prep.preprocessCrossFolds(f"./datasets/{dataset_name}", encodeCategorical=False)

    accuacies = np.zeros(10)
    redu_times = np.zeros(10)
    pred_times = np.zeros(10)
    storage = np.zeros(10)
    classes_distr = []
    i = 0
    for train_data, test_data in Crossfolds:
        print(" " * 8 + f"Applying reduction on fold nº {i} ...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        old_n_instances = len(X_train)

        if reduce:
            begin = time.time()
            gcnn = GCNN.GCNN(X_train, y_train, rho=rho)
            X_train, y_train = gcnn.applyGCNN()
            redu_times[i] = time.time() - begin
        else:
            redu_times[i] = 0
        new_n_instances = len(X_train)

        if save:
            saveReducedFold(X_train, X_test, y_train, y_test, dataset_name=dataset_name,
                            fold_number=i, tech="GCNN", option=("rho", str(rho)))

        _, counts = np.unique(y_train, return_counts=True)
        classes_distr.append(counts / new_n_instances)

        storage[i] = (new_n_instances * 100) / old_n_instances

        begin = time.time()
        Knn = kNNalgo.kNNAlgorithm(K)
        Knn.fit(X_train, y_train, **kwargs)
        y_pred = Knn.predict(X_test)
        pred_times[i] = time.time() - begin

        accuacies[i] = accuracy_score(y_test, y_pred, normalize=True)
        i += 1

    infodata = pd.DataFrame()
    infodata["Fold"] = list(range(0, 10))
    infodata["Accuracy"] = accuacies
    infodata["Reduction_time"] = redu_times
    infodata["Prediction_time"] = pred_times
    infodata["Storage"] = storage
    infodata["Distributions"] = classes_distr
    infodata.loc["Average Results"] = ["Folds Average", accuacies.mean(), redu_times.mean(),
                                       pred_times.mean(), storage.mean(), []]

    print("-" * 50)
    TotalElapsed = time.time() - firstStart
    ExecutionTime = time.strftime("%Hh%Mm%Ss", time.gmtime(TotalElapsed))
    print(f"Total execution time of the experiments: {ExecutionTime}")

    directory = f"./ResultFiles/{dataset_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if reduce:
        name = f"{dataset_name}_GCNN_Reduction_rho{rho}"
    else:
        name = f"{dataset_name}_without_GCNN_Reduction"

    infodata.to_csv(f'{directory}/{name}.csv', index=False)
    print("-" * 50)
    print(f"The following result files were saved in the directory {directory}")
    print(" " * 3 + f"{name}.csv")
    if save:
        tech = "GCNN"
        name_op = "rho"
        value_op = str(rho)
        directory = f"./ReducedDatasets/{tech}/{dataset_name}/{name_op}{value_op}"
        print(f"The reduced cross folds were saved in the directory:")
        print(" " * 3 + f"{directory}")


def RunENNThExperiment(dataset_name="cmc", K=5, mu=0.15, reduce=True, save=True, **kwargs):
    firstStart = time.time()
    print(f"Reading and preprocessing the crossfolds of {dataset_name} dataset ...")
    Crossfolds = prep.preprocessCrossFolds(f"./datasets/{dataset_name}", encodeCategorical=False)

    accuacies = np.zeros(10)
    redu_times = np.zeros(10)
    pred_times = np.zeros(10)
    storage = np.zeros(10)
    classes_distr = []
    i = 0
    for train_data, test_data in Crossfolds:
        print(" " * 8 + f"Applying reduction on fold nº {i} ...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        old_n_instances = len(X_train)

        if reduce:
            begin = time.time()
            ennth = ENNTh.ENNTh(X_train, y_train, mu=mu, k=K)
            X_train, y_train = ennth.applyENNTh()
            redu_times[i] = time.time() - begin
        else:
            redu_times[i] = 0
        new_n_instances = len(X_train)

        if save:
            saveReducedFold(X_train, X_test, y_train, y_test, dataset_name=dataset_name,
                            fold_number=i, tech="ENNTh", option=("mu", str(mu)))

        _, counts = np.unique(y_train, return_counts=True)
        classes_distr.append(counts / new_n_instances)

        storage[i] = (new_n_instances * 100) / old_n_instances

        begin = time.time()
        Knn = kNNalgo.kNNAlgorithm(K)
        Knn.fit(X_train, y_train, **kwargs)
        y_pred = Knn.predict(X_test)
        pred_times[i] = time.time() - begin

        accuacies[i] = accuracy_score(y_test, y_pred, normalize=True)
        i += 1

    infodata = pd.DataFrame()
    infodata["Fold"] = list(range(0, 10))
    infodata["Accuracy"] = accuacies
    infodata["Reduction_time"] = redu_times
    infodata["Prediction_time"] = pred_times
    infodata["Storage"] = storage
    infodata["Distributions"] = classes_distr
    infodata.loc["Average Results"] = ["Folds Average", accuacies.mean(), redu_times.mean(),
                                       pred_times.mean(), storage.mean(), []]

    print("-" * 50)
    TotalElapsed = time.time() - firstStart
    ExecutionTime = time.strftime("%Hh%Mm%Ss", time.gmtime(TotalElapsed))
    print(f"Total execution time of the experiments: {ExecutionTime}")

    directory = f"./ResultFiles/{dataset_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if reduce:
        name = f"{dataset_name}_ENNTh_Reduction_mu{mu}"
    else:
        name = f"{dataset_name}_without_ENNTh_Reduction"

    infodata.to_csv(f'{directory}/{name}.csv', index=False)
    print("-" * 50)
    print(f"The following result files were saved in the directory {directory}")
    print(" " * 3 + f"{name}.csv")
    if save:
        tech = "EENTh"
        name_op = "mu"
        value_op = str(mu)
        directory = f"./ReducedDatasets/{tech}/{dataset_name}/{name_op}{value_op}"
        print(f"The reduced cross folds were saved in the directory:")
        print(" " * 3 + f"{directory}")


def RunDROP3Experiment(dataset_name="cmc", K=5, reduce=True, save=True, **kwargs):
    firstStart = time.time()
    print(f"Reading and preprocessing the crossfolds of {dataset_name} dataset ...")
    Crossfolds = prep.preprocessCrossFolds(f"./datasets/{dataset_name}", encodeCategorical=False)

    accuacies = np.zeros(10)
    redu_times = np.zeros(10)
    pred_times = np.zeros(10)
    storage = np.zeros(10)
    classes_distr = []
    i = 0
    for train_data, test_data in Crossfolds:
        print(" " * 8 + f"Applying reduction on fold nº {i} ...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        old_n_instances = len(X_train)

        if reduce:
            begin = time.time()
            drop3 = DROP3.DROP3(X_train, y_train, K=K, Knn_k=K)
            X_train, y_train = drop3.applyDROP3(**kwargs)
            redu_times[i] = time.time() - begin
        else:
            redu_times[i] = 0
        new_n_instances = len(X_train)

        if save:
            saveReducedFold(X_train, X_test, y_train, y_test, dataset_name=dataset_name,
                            fold_number=i, tech="DROP3", option=("K", str(K)))

        _, counts = np.unique(y_train, return_counts=True)
        classes_distr.append(counts / new_n_instances)

        storage[i] = (new_n_instances * 100) / old_n_instances

        begin = time.time()
        Knn = kNNalgo.kNNAlgorithm(K)
        Knn.fit(X_train, y_train, **kwargs)
        y_pred = Knn.predict(X_test)
        pred_times[i] = time.time() - begin

        accuacies[i] = accuracy_score(y_test, y_pred, normalize=True)
        i += 1

    infodata = pd.DataFrame()
    infodata["Fold"] = list(range(0, 10))
    infodata["Accuracy"] = accuacies
    infodata["Reduction_time"] = redu_times
    infodata["Prediction_time"] = pred_times
    infodata["Storage"] = storage
    infodata["Distributions"] = classes_distr
    infodata.loc["Average Results"] = ["Folds Average", accuacies.mean(), redu_times.mean(),
                                       pred_times.mean(), storage.mean(), []]

    print("-" * 50)
    TotalElapsed = time.time() - firstStart
    ExecutionTime = time.strftime("%Hh%Mm%Ss", time.gmtime(TotalElapsed))
    print(f"Total execution time of the experiments: {ExecutionTime}")

    directory = f"./ResultFiles/{dataset_name}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if reduce:
        name = f"{dataset_name}_Drop3_Reduction_K{K}"
    else:
        name = f"{dataset_name}_without_Drop3_Reduction"

    infodata.to_csv(f'{directory}/{name}.csv', index=False)
    print("-" * 50)
    print(f"The following result files were saved in the directory {directory}")
    print(" " * 3 + f"{name}.csv")
    if save:
        tech = "DROP3"
        name_op = "K"
        value_op = str(K)
        directory = f"./ReducedDatasets/{tech}/{dataset_name}/{name_op}{value_op}"
        print(f"The reduced cross folds were saved in the directory:")
        print(" " * 3 + f"{directory}")


def getCrossfolds(directory):
    options = os.listdir(directory)
    if len(options) != 0:

        print("List of possible reduced datasets:")
        for i, file_name in enumerate(options):
            print(" " * 3 + f"{i}. {file_name}")
        x = -1
        tam = len(options) - 1
        while not (0 <= x <= tam):
            x = int(input(f"Enter a number between 0 and {len(options) - 1} to select a dataset:\n"))
            if not (0 <= x <= tam):
                print("Please try again")

        parts = directory.split("/")
        dataset_name = parts[-1]
        CrossFolds = []
        name = options[x]
        for i in range(0, 10):
            filename = f"{directory}/{name}/{dataset_name}.fold.00000{i}."
            train = pd.read_csv(filename + "train.csv", sep=",")
            y_train = train.iloc[:, -1]
            train = train.drop(train.columns[-1], axis=1)
            test = pd.read_csv(filename + "test.csv", sep=",")
            y_test = test.iloc[:, -1]
            test = test.drop(test.columns[-1], axis=1)
            CrossFolds.append(((train, y_train), (test, y_test)))
        return CrossFolds


    else:
        print("No reduction datasets has been fund with this combination")


'''
#####################################################################################################
                                   Experiments
#####################################################################################################
'''


# Experiments

def RunAllMuENNTh(dataset_name, mus, K=5, **kwargs):
    for mu in mus:
        print("-" * 50)
        print(f"Experiment with {dataset_name} dataset and mu = {mu}")
        print("-" * 50)
        RunENNThExperiment(dataset_name=dataset_name, K=K, mu=mu, **kwargs)
    print("-" * 50)
    print(f"Experiment with {dataset_name} dataset and no reduction")
    print("-" * 50)
    RunENNThExperiment(dataset_name=dataset_name, K=K, reduce=False, save=False, **kwargs)


def RunAllThoGCNNWithCMCKnn():
    kwargs = {
        'distance': "Canberra",
        'voting': "Majority",
        'weighting': "InformationGain"
    }
    dataset_name = "cmc"
    K = 7
    rhos = [0.0001, 0.001, 0.01]
    for rho in rhos:
        RunGCNNExperiment(rho=rho, dataset_name=dataset_name, K=K, reduce=True, save=True, **kwargs)
    RunGCNNExperiment(dataset_name=dataset_name, K=K, reduce=False, save=False, **kwargs)


def RunAllThoGCNNWithSATIMAGEKnn():
    kwargs = {
        'distance': "Canberra",
        'voting': "InverseDistanceWeighted",
        'weighting': "Equal"
    }
    dataset_name = "satimage"
    K = 3
    rhos = [0.0001, 0.001, 0.01]
    for rho in rhos:
        RunGCNNExperiment(rho=rho, dataset_name=dataset_name, K=K, reduce=True, save=True, **kwargs)
    RunGCNNExperiment(dataset_name=dataset_name, K=K, reduce=False, save=False, **kwargs)


def RunAllMuENNThWithCMCKnn():
    kwargs = {
        'distance': "Canberra",
        'voting': "Majority",
        'weighting': "InformationGain"
    }
    mus = np.linspace(100, 160, 4) / 1000
    dataset_name = "cmc"
    RunAllMuENNTh(dataset_name=dataset_name, K=7, mus=mus, **kwargs)


def RunAllMuENNThWithSATIMAGEKnn():
    kwargs = {
        'distance': "Canberra",
        'voting': "InverseDistanceWeighted",
        'weighting': "Equal"
    }
    mus = np.linspace(300, 400, 6) / 1000
    dataset_name = "satimage"
    RunAllMuENNTh(dataset_name=dataset_name, K=3, mus=mus, **kwargs)


def RunDROP3CMCKnn():
    kwargs = {
        'distance': "Canberra",
        'voting': "Majority",
        'weighting': "InformationGain"
    }
    dataset_name = "cmc"
    K = 7
    RunDROP3Experiment(dataset_name=dataset_name, K=K, reduce=True, save=True, **kwargs)
    RunDROP3Experiment(dataset_name=dataset_name, K=K, reduce=False, save=False, **kwargs)


def RunDROP3SATIMAGEKnn():
    K = 3
    dataset_name = "satimage"
    kwargs = {
        'distance': "Canberra",
        'voting': "InverseDistanceWeighted",
        'weighting': "Equal"
    }
    RunDROP3Experiment(dataset_name=dataset_name, K=K, reduce=True, save=True, **kwargs)
    RunDROP3Experiment(dataset_name=dataset_name, K=K, reduce=False, save=False, **kwargs)


'''
#####################################################################################################
                                   Main options
#####################################################################################################
'''


def runKnnWithParticularCombination(dataset_name, k, **kwargs):
    print(f"Reading and preprocessing the crossfolds of {dataset_name} dataset ...")
    Crossfolds = prep.preprocessCrossFolds(f"./datasets/{dataset_name}", encodeCategorical=False)
    accs = np.zeros(10)
    times = np.zeros(10)
    i = 0
    for train_data, test_data in Crossfolds:
        print(f"Applying Knn to fold nº {i} ...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        begin = time.time()
        Knn = kNNalgo.kNNAlgorithm(k)
        Knn.fit(X_train, y_train, **kwargs)
        y_pred = Knn.predict(X_test)
        times[i] = time.time() - begin
        accs[i] = accuracy_score(y_test, y_pred, normalize=True)
        print(" " * 3 + f"Execution time: {round(times[i], 3)} s")
        print(" " * 3 + f"Accuracy      : {round(accs[i], 4)}")
        i += 1
    print("-" * 50)
    print(f"Average execution time: {round(times.mean(), 3)} s")
    print(f"Average accuracy      : {round(accs.mean(), 4)}")


def runKnnWithParticularCombinationAndReduction(dataset_name, k, reduction, **kwargs):
    if reduction == "GCNN":
        value = float(input(f"Enter the rho value (between 0 and 1, please consider that bigger values take more "
                            f"time):\n"))
    elif reduction == "ENNTh":
        value = float(input(f"Enter the mu value (between 0 and 1):\n"))
    else:
        value = int(input(f"Enter the K value for the drop3:\n"))

    print(f"Reading and preprocessing the crossfolds of {dataset_name} dataset ...")
    Crossfolds = prep.preprocessCrossFolds(f"./datasets/{dataset_name}", encodeCategorical=False)
    accs = np.zeros(10)
    times = np.zeros(10)
    red_times = np.zeros(10)
    i = 0
    for train_data, test_data in Crossfolds:
        print(f"Applying reduction and Knn to fold nº {i} ...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        begin = time.time()
        Knn = redKNN.reductionKNNAlgorithm(k)
        print(" " * 3 + f"Appling reduction with {reduction} ...")
        Knn.fit(X_train, y_train, reduction=reduction, value=value, **kwargs)
        red_times[i] = time.time() - begin
        begin = time.time()
        print(" " * 3 + f"Predicting with Knn ...")
        y_pred = Knn.predict(X_test)
        times[i] = time.time() - begin
        accs[i] = accuracy_score(y_test, y_pred, normalize=True)
        print(" " * 3 + f"Reduction time : {round(red_times[i], 3)} s")
        print(" " * 3 + f"Prediction time: {round(times[i], 3)} s")
        print(" " * 3 + f"Accuracy       : {round(accs[i], 4)}")
        i += 1
    print("-" * 50)
    print(f"Average reduction time : {round(red_times.mean(), 3)} s")
    print(f"Average prediction time: {round(times.mean(), 3)} s")
    print(f"Average accuracy       : {round(accs.mean(), 4)}")


def runKnnWithParticularCombinationWithReducedDataset(dataset_name, k, reduction, **kwargs):
    Crossfolds = getCrossfolds(f"./ReducedDatasets/{reduction}/{dataset_name}")
    print("Cross folds read correctly")
    accs = np.zeros(10)
    times = np.zeros(10)
    i = 0
    for train_data, test_data in Crossfolds:
        print(f"Applying Knn to fold nº {i} ...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        begin = time.time()
        Knn = kNNalgo.kNNAlgorithm(k)
        Knn.fit(X_train, y_train, **kwargs)
        y_pred = Knn.predict(X_test)
        times[i] = time.time() - begin
        accs[i] = accuracy_score(y_test, y_pred, normalize=True)
        print(" " * 3 + f"Execution time: {round(times[i], 3)} s")
        print(" " * 3 + f"Accuracy      : {round(accs[i], 4)}")
        i += 1
    print("-" * 50)
    print(f"Average execution time: {round(times.mean(), 3)} s")
    print(f"Average accuracy      : {round(accs.mean(), 4)}")


def StatisticalTestsWithoutReduction(dataset_name):
    stats.get_best_model(dataset_name)


def StatisticalTestsWithReduction(dataset_name):
    stats.get_best_reduction_model(dataset_name)
