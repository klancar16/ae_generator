import numpy as np
import pandas as pd
from collections import defaultdict
import functools
MIN = "min"
MAX = "max"


def fill_missing(df):
    df = df.fillna(df.median())
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    return df


def check_numerical(x):
    try:
        int_value = int(x)
        return True
    except ValueError:
        return False


def from_dummies(data, categories, prefix_sep='_'):
    out = data.copy()
    for l in categories:
        cols, labs = [[c.replace(x,"") for c in data.columns if c.startswith(l+prefix_sep)] for x in ["", l+prefix_sep]]
        only_num = functools.reduce(lambda a, b: a and check_numerical(b), labs, True)
        if only_num:
            labs = [int(x) for x in labs]
        out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
        out.drop(cols, axis=1, inplace=True)
    return out


def convert_nominal(df):
    df = pd.get_dummies(df, prefix_sep="_")
    return df


def reconstruct_nominal(df, nominal_cols):
    df = from_dummies(df, categories=nominal_cols)
    return df


def normalize(df, numerical_cols):
    norm_map = defaultdict()
    for col in numerical_cols:
        norm_map[col] = defaultdict()
        norm_map[col][MIN] = df[col].min()
        norm_map[col][MAX] = df[col].max()
    df_norm = df[numerical_cols]
    df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
    df_norm = df_norm.fillna(0)
    df[numerical_cols] = df_norm
    return df, norm_map


def denormalize(df, numerical_cols, norm_map):
    for col in numerical_cols:
        df[col] = df[col] * (norm_map[col][MAX] - norm_map[col][MIN]) + norm_map[col][MIN]
    return df


def preprocess(df):
    columns = list(df.columns)
    numerical_cols = list(df._get_numeric_data().columns)
    nominal_cols = list(set(columns) - set(numerical_cols))
    df = df[numerical_cols + nominal_cols]
    df, norm_map = normalize(df, numerical_cols)
    df = convert_nominal(df)
    return df, columns, numerical_cols, norm_map, nominal_cols


def postprocess(df, numerical_cols, norm_map, nominal_cols):
    df = denormalize(df, numerical_cols, norm_map)
    df = reconstruct_nominal(df, nominal_cols)
    return df
