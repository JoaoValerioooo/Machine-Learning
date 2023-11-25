import csv
import pandas as pd
from scipy.stats import rankdata
import numpy as np
from scipy import stats
import Orange
import os
from tabulate import tabulate

# perform Friedman Test


def get_best_reduction_model(dataset_name):
    directory_files = os.listdir('ResultFiles/' + dataset_name)
    headers = []
    data_raw = pd.DataFrame()
    time_raw = pd.DataFrame()
    storage_raw = pd.DataFrame()
    for file in directory_files:
        if ("Reduction" in file) and not "without" in file:
            with open("ResultFiles/" + dataset_name + "/" + file, newline='') as f:
                reader = csv.reader(f)
                non_dt = [row for row in list(reader)[1:]]
                data = pd.DataFrame([row[1] for row in non_dt])
                time = pd.DataFrame([row[3] for row in non_dt])
                storage = pd.DataFrame([row[4] for row in non_dt])
                time_raw = pd.concat([time_raw, time], axis=1)
                data_raw = pd.concat([data_raw, data], axis=1)
                storage_raw = pd.concat([storage_raw, storage], axis=1)
                headers.append(file)
    data_raw = np.array(data_raw)
    time_raw = np.array(time_raw)
    storage_raw = np.array(storage_raw)
    headers = np.array(headers)

    ranks_accuracy = []
    inverted_data = [[float(y) * -1 for y in x] for x in data_raw[1:]]
    data = [[float(y) for y in x] for x in data_raw[1:]]
    for i in range(len(inverted_data)):
        ranks_accuracy.append(rankdata(inverted_data[i]))
    column_mean = np.mean(ranks_accuracy, axis=0)
    sorted_indexes_accuracy = np.argsort(column_mean)

    ranks_time = []
    times = np.array([[float(y) for y in x] for x in time_raw])
    for i in range(len(times)):
        ranks_time.append(rankdata(times[i]))
    column_mean_time = np.mean(ranks_time, axis=0)
    sorted_indexes_time = np.argsort(column_mean_time)

    ranks_storage = []
    storages = np.array([[float(y) for y in x] for x in storage_raw])
    for i in range(len(storages)):
        ranks_storage.append(rankdata(storages[i]))
    column_mean_sorage = np.mean(ranks_storage, axis=0)
    sorted_indexes_storage = np.argsort(column_mean_sorage)

    print_info(column_mean, np.mean(data, axis=0), headers, column_mean, sorted_indexes_accuracy, data,
               "Mean accuracy rank;Mean accuracy;Mean time rank;model",
               dataset_name + " best_reduced_model_accuracy")
    print_info(column_mean_time, np.mean(times, axis=0), headers, column_mean_time, sorted_indexes_time, times,
               "Mean time rank;Mean time;Mean accuracy rank;model", dataset_name + " best_reduced_model_time")
    print_info(column_mean_sorage, np.mean(storages, axis=0), headers, column_mean_sorage, sorted_indexes_storage,
               storages,
               "Mean storage rank;Mean storage;Mean time rank;model", dataset_name + " best_reduced_model_storage")


def get_best_model(dataset_name):
    with open(f'ResultFiles/{dataset_name}/{dataset_name}_times.csv', newline='') as f:
        reader = csv.reader(f)
        times_raw = list(reader)
    times = np.array([[float(y) for y in x] for x in times_raw[1:]])
    ranks_time = []
    for i in range(len(times)):
        ranks_time.append(rankdata(times[i]))
    column_mean_time = np.mean(ranks_time, axis=0)
    sorted_indexes_time = np.argsort(column_mean_time)

    ranks_accuracy = []
    with open(f'ResultFiles/{dataset_name}/{dataset_name}_accuracies.csv', newline='') as f:
        reader = csv.reader(f)
        data_raw = list(reader)
    headers = np.array(data_raw[0])
    inverted_data = [[float(y) * -1 for y in x] for x in data_raw[1:]]
    data = [[float(y) for y in x] for x in data_raw[1:]]

    for i in range(len(inverted_data)):
        ranks_accuracy.append(rankdata(inverted_data[i]))

    column_mean = np.mean(ranks_accuracy, axis=0)
    sorted_indexes_accuracy = np.argsort(column_mean)
    print_info(column_mean, np.mean(data, axis=0), headers, column_mean_time, sorted_indexes_accuracy, data,
               "Mean accuracy rank;Mean accuracy;Mean time rank;combination", dataset_name + " best_model_accuracy")
    print_info(column_mean_time, np.mean(times, axis=0), headers, column_mean, sorted_indexes_time, times,
               "Mean time rank;Mean time;Mean accuracy rank;combination", dataset_name + " best_model_time")


def print_info(column_mean_ranked, column_mean_value, headers, column_mean_secondary, sorted_indexes, friedman_data,
               header_text, printed_name):
    print("\n", printed_name, "\n")
    # print(header_text)
    column_names = header_text.split(";")
    table = [column_names]
    for rankvalue, average_accuracy, name, time in zip(column_mean_ranked[sorted_indexes],
                                                       column_mean_value[sorted_indexes], headers[sorted_indexes],
                                                       column_mean_secondary[sorted_indexes]):
        # print(round(rankvalue, 1), "\t", round(average_accuracy, 3), "\t", round(time, 1), "\t", name)
        row = [round(rankvalue, 1), round(average_accuracy, 3), round(time, 1), name]
        table.append(row)

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', numalign="center", colalign=["center"]*len(column_names)))

    stats_result = stats.friedmanchisquare(*[k for k in np.transpose(friedman_data)])
    print("Friedman test result P-value :", stats_result[1])

    names = [name.split(".csv")[0] for name in headers[sorted_indexes][:10]]
    avranks = column_mean_ranked[sorted_indexes][:10]
    cd = Orange.evaluation.scoring.compute_CD(avranks, 10)  # tested on 30 datasets
    print("CD:", cd)

    parts = printed_name.split(" ")
    dataset_name = parts[0]
    directory = f"./ResultFiles/{dataset_name}/StatisticalImages"
    if not os.path.exists(directory):
        os.makedirs(directory)

    Orange.evaluation.scoring.graph_ranks(filename=f"{directory}/{printed_name}.png", avranks=avranks, names=names,
                                          cd=cd, width=18, textspace=4.5)

    print(f"The following test image was saved in the directory {directory}")
    print(" " * 3 + f"{printed_name}.png")

