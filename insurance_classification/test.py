import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},"test": {"X": X_test, "y": y_test}, "validate": {"X": X_test, "y": y_test}}
    return data