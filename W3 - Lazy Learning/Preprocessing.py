import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.impute import KNNImputer


# Analysis functions
def readDataframe(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    return df


def valueCountCategorical(filename):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    for col in df.columns:
        if df[col].dtype == object:
            print("Col name: ", col)
            df[col] = df[col].str.decode('utf-8')
            print("Valores Unicos: \n", df[col].unique())


def missingsDetails(df):
    print("Categorical columns info:")
    print()
    print("=" * 32)
    for col in df.columns:
        if df[col].dtype == object:
            print(f"Column: {col}")
            print(df[col].value_counts())
            print("=" * 32)
    print("Number of NA in dataset")
    print()
    print("=" * 32)
    for col in df.columns:
        if df[col].dtype != object:
            print(f"Column: {col}")
            print(sum(df[col].isna()))
            print("=" * 32)


#############################################################
# Preprocessing functions

def replaceCategoricalMissings(df, missingCode, replaceCode):
    for col in df.columns:
        if df[col].dtype == object:
            df.loc[df[col] == missingCode, col] = replaceCode
    return df


def eraseClassColumn(df):
    dfaux = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    return dfaux, labels


def applyOneHotEncoding(df):
    categorical = []
    for col in df.columns:
        if df[col].dtype == object:
            categorical.append(col)
    df = pd.get_dummies(df, columns=categorical)
    return df


def applyLabelEncoding(df):
    label_encoder = preprocessing.LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = label_encoder.fit_transform(df[col])
    return df


def normalizeDataset(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        if df[col].dtype != object:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    return df_norm


def findMinMaxForColumns(df):
    min_max = {}
    for col in df.columns:
        if df[col].dtype != object:
            minimum = df[col].min()
            maximum = df[col].max()
            min_max[col] = (minimum, maximum)
    return min_max


def normalizeDatasetWithMinMax(df, min_max):
    df_norm = df.copy()
    for col in df_norm.columns:
        if df[col].dtype != object:
            minimum, maximum = min_max[col]
            df_norm[col] = (df_norm[col] - minimum) / (maximum - minimum)
    return df_norm


def replaceNumericalMissings(df):
    numerical = []
    df_copy = df.copy()
    for ind, col in enumerate(df.columns.values):
        if df[col].dtype != object:
            numerical.append(ind)
    if len(numerical) == 0:
        return df_copy
    dd = df.iloc[:, numerical]
    colnames = dd.columns
    imputer = KNNImputer(weights='distance')
    imputer.fit(dd)
    ddarray = imputer.transform(dd)
    ddclean = pd.DataFrame(ddarray, columns=colnames)
    for col in ddclean.columns:
        df_copy[col] = ddclean[col]
    return df_copy


def getColumnTypeBool(df):
    variableTypes = []
    for col in df.columns:
        if df[col].dtype == object:
            variableTypes.append(True)
        else:
            variableTypes.append(False)
    return variableTypes

def eraseBytesFormat(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    return df

def readCrossFolds(dir_path):
    CrossFolds = []
    dirformat = dir_path.split("/")
    name = dirformat[2]
    for i in range(0, 10):
        filename = f"{dir_path}/{name}.fold.00000{i}."
        train_arff = arff.loadarff(filename + "train.arff")
        test_arff = arff.loadarff(filename + "test.arff")
        train = eraseBytesFormat(pd.DataFrame(train_arff[0]))
        test = eraseBytesFormat(pd.DataFrame(test_arff[0]))
        CrossFolds.append((train, test))
    return CrossFolds


def preprocessDataset(filename, replaceCategorical=True, replaceNumerical=True, encodeCategorical=True,
                      categoricalEncondingType="Label", normalize=True):
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    df, labels = eraseClassColumn(df)
    if replaceCategorical:
        df = replaceCategoricalMissings(df, "?", "Unknown")
    if replaceNumerical:
        df = replaceNumericalMissings(df)
    if encodeCategorical:
        if categoricalEncondingType == "One":
            df = applyOneHotEncoding(df)
        else:
            df = applyLabelEncoding(df)
    if normalize:
        df = normalizeDataset(df)
    return df, labels


def preprocessDatasetPart(part, min_max, replaceCategorical=True, replaceNumerical=True, encodeCategorical=True,
                          categoricalEncondingType="Label", normalize=True):
    X, y = eraseClassColumn(part)
    if replaceCategorical:
        X = replaceCategoricalMissings(X, "?", "Unknown")
    if replaceNumerical:
        X = replaceNumericalMissings(X)
    if encodeCategorical:
        if categoricalEncondingType == "One":
            X = applyOneHotEncoding(X)
        else:
            X = applyLabelEncoding(X)
    if normalize:
        X = normalizeDatasetWithMinMax(X, min_max)
    return X, y


def preprocessCrossFolds(dir_path, replaceCategorical=True, replaceNumerical=True, encodeCategorical=True,
                         categoricalEncondingType="Label", normalize=True):
    CrossFolds = readCrossFolds(dir_path)
    processedCrossFolds = []
    for train, test in CrossFolds:
        df = pd.concat([train, test])
        min_max = findMinMaxForColumns(df)
        X_train, y_train = preprocessDatasetPart(train, encodeCategorical=encodeCategorical,
                                               replaceCategorical=replaceCategorical,
                                               replaceNumerical=replaceNumerical,
                                               categoricalEncondingType=categoricalEncondingType,
                                               normalize=normalize, min_max=min_max)
        X_test, y_test = preprocessDatasetPart(test, encodeCategorical=encodeCategorical,
                                               replaceCategorical=replaceCategorical,
                                               replaceNumerical=replaceNumerical,
                                               categoricalEncondingType=categoricalEncondingType,
                                               normalize=normalize, min_max=min_max)
        processedCrossFolds.append(((X_train, y_train), (X_test, y_test)))
    return processedCrossFolds
