import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn import preprocessing
import numpy as np
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
    print("="*32)
    for col in df.columns:
        if df[col].dtype == object:
            print(f"Column: {col}")
            print(df[col].value_counts())
            print("="*32)
    print("Number of NA in dataset")
    print()
    print("="*32)
    for col in df.columns:
        if df[col].dtype != object:
            print(f"Column: {col}")
            print(sum(df[col].isna()))
            print("="*32)

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
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
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

    # for col in numerical_names:
    #     mean = df[col].mean(skipna=True)
    #     df[col].fillna(mean, inplace=True)
    # return df
    return df_copy


def preprocessDataset(filename, replaceCategorical=True, replaceNumerical=True, encodeCategorical=True, categoricalEncondingType = "Label", normalize=True):
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
