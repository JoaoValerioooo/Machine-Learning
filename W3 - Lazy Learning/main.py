import Experiments as Exp


def print_long_line(n):
    print("=" * n)


dataset_list = ["cmc", "satimage"]
Ks = [1, 3, 5, 7]
weights = ["Equal", "InformationGain", "Chi"]
distances = ["Minkowski", "Cosine", "Canberra"]
votings = ["Majority", "InverseDistanceWeighted", "Sheppards"]


def get_answers(questions, listOptions):
    print_long_line(50)
    answers = {}
    for name, question in questions.items():
        print(question)
        options = listOptions[name]
        for j, op in enumerate(options):
            print(" " * 3 + f"{j}. {op}")
        x = -1
        tam = len(options) - 1
        while not (0 <= x <= tam):
            x = int(input(f"Enter a number between 0 and {len(options) - 1}:\n"))
            if not (0 <= x <= tam):
                print("Please try again")
        answers[name] = options[x]
    return answers


def operation_0():
    questions = {"dataset": "Select the dataset you are interested in:",
                 "k": "Please select the k value of the KNN:",
                 "distance": "Please select the distance type:",
                 "voting": "Please select the voting type:",
                 "weighting": "Please select the weights type:"}
    listOptions = {"dataset": dataset_list,
                 "k": Ks,
                 "distance": distances,
                 "voting": votings,
                 "weighting": weights}
    answers = get_answers(questions, listOptions)
    dataset_name = answers['dataset']
    k = answers['k']
    del answers['dataset']
    del answers['k']
    Exp.runKnnWithParticularCombination(dataset_name=dataset_name, k=k, **answers)

def operation_1():
    reduction_methods = ["GCNN", "ENNTh", "DROP3"]
    questions = {"dataset": "Select the dataset you are interested in:",
                 "k": "Please select the k value of the KNN:",
                 "distance": "Please select the distance type:",
                 "voting": "Please select the voting type:",
                 "weighting": "Please select the weights type:",
                 "reduction": "Please select the reduction method:"}
    listOptions = {"dataset": dataset_list,
                   "k": Ks,
                   "distance": distances,
                   "voting": votings,
                   "weighting": weights,
                   "reduction": reduction_methods}
    answers = get_answers(questions, listOptions)
    dataset_name = answers['dataset']
    k = answers['k']
    reduction = answers['reduction']
    del answers['dataset']
    del answers['k']
    del answers['reduction']
    Exp.runKnnWithParticularCombinationAndReduction(dataset_name=dataset_name, k=k, reduction=reduction, **answers)



def operation_2():
    reduction_methods = ["GCNN", "ENNTh", "DROP3"]
    questions = {"k": "Please select the k value of the KNN:",
                 "distance": "Please select the distance type:",
                 "voting": "Please select the voting type:",
                 "weighting": "Please select the weights type:",
                 "dataset": "Select the dataset you are interested in:",
                 "reduction": "Please select the reduction method:"}
    listOptions = {"k": Ks,
                   "distance": distances,
                   "voting": votings,
                   "weighting": weights,
                   "dataset": dataset_list,
                   "reduction": reduction_methods}
    answers = get_answers(questions, listOptions)
    dataset_name = answers['dataset']
    k = answers['k']
    reduction = answers['reduction']
    del answers['dataset']
    del answers['k']
    del answers['reduction']
    Exp.runKnnWithParticularCombinationWithReducedDataset(dataset_name=dataset_name, k=k, reduction=reduction, **answers)


def operation_3():
    print_long_line(50)
    Experiments_names = ["Compute all the combinations for a particular dataset",
                         "Run the Friedman and Nemenyi tests over all the combinations without reduction",
                         "Run the Friedman and Nemenyi tests over all the combinations with reduction",
                         "Run GCNN reduction technique experiments with a particular dataset",
                         "Run ENNTh reduction technique experiments with a particular dataset",
                         "Run DROP3 reduction technique experiments with a particular dataset"]
    print("Select the experiment you want you want to run:")
    for i, experi in enumerate(Experiments_names):
        print(" "*3+f"{i}. {experi}")
    n = len(Experiments_names)
    print()
    print("!!!!! Please be aware that some experiments may take a lot of time to execute !!!")
    print()
    k = int(input(f"Enter the number of the experiment (between 0 and {n - 1}): \n"))
    if 0 <= k <= n:
        questions = {"dataset": "Select the dataset you are interested in:"}
        listOptions = {"dataset": dataset_list}
        answers = get_answers(questions, listOptions)
        dataset_name = answers['dataset']
        if k == 0:
            Exp.RunAllCombinations(dataset_name)
        elif k == 1:
            Exp.StatisticalTestsWithoutReduction(dataset_name)
        elif k == 2:
            Exp.StatisticalTestsWithReduction(dataset_name)
        elif k == 3:
            if dataset_name == "cmc":
                Exp.RunAllThoGCNNWithCMCKnn()
            else:
                Exp.RunAllThoGCNNWithSATIMAGEKnn()
        elif k == 4:
            if dataset_name == "cmc":
                Exp.RunAllMuENNThWithCMCKnn()
            else:
                Exp.RunAllMuENNThWithSATIMAGEKnn()
        elif k == 5:
            if dataset_name == "cmc":
                Exp.RunDROP3CMCKnn()
            else:
                Exp.RunDROP3SATIMAGEKnn()
    else:
        print("Please try again")


def main():
    print_long_line(50)
    print("What of the following actions you want to do?")
    print("0. Compute the KNN over a particular dataset")
    print("1. Apply instance reduction techniques and do the KNN")
    print("2. Apply Knn to previously reduced datasets")
    print("3. Run Experiments")
    op = int(input("Enter a number between 0 and 3:\n"))
    if op == 0:
        operation_0()
    elif op == 1:
        operation_1()
    elif op == 2:
        operation_2()
    elif op == 3:
        operation_3()


if __name__ == "__main__":
    main()
