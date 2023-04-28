import os
from argparse import ArgumentParser as AP

import numpy as np
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import LabelEncoder

# Get the run context
new_run = Run.get_context()

# Get the workspace from the run
ws = new_run.experiment.workspace

# Read the input dataset
df = new_run.input_datasets["insurance_ds"].to_pandas_dataframe()

# Check the missing values
dataNull = df.isnull().sum()

all_cols = df.columns


def Preprocessing(df):
    """Data Pre-processing"""
    # if '?' in the datset which we have to remove by NaN Values
    df = df.replace("?", np.NaN)

    # missing value treatment using fillna

    # we will replace the '?' by the most common collision type as we are unaware of the type.
    df["collision_type"].fillna(df["collision_type"].mode()[0], inplace=True)

    # Encoding property_damage variable
    df["property_damage"].fillna("NO", inplace=True)

    # Encoding police_report_available variable
    df["police_report_available"].fillna("NO", inplace=True)

    # let's extrat days, month and year from policy bind date
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"], errors="coerce")

    # dropping unimportant columns
    df.drop(["_c39"], axis=1, inplace=True)

    numeric_data = df._get_numeric_data()
    cat_data = df.select_dtypes(include=["object"])

    for c in cat_data:
        lbl = LabelEncoder()
        lbl.fit(cat_data[c].values)
        cat_data[c] = lbl.transform(cat_data[c].values)
    clean_data = pd.concat([numeric_data, cat_data], axis=1)

    return clean_data


dataPrep = Preprocessing(df)
parser = AP()
parser.add_argument("--datafolder", type=str)
args = parser.parse_args()


# Create the folder if it does not exist
os.makedirs(args.datafolder, exist_ok=True)

# Create the path
path = os.path.join(args.datafolder, "InsuranceData_prep_2.csv")

# Write the data preparation output as csv file
dataPrep.to_csv(path, index=False)

# Log null values
for column in all_cols:
    new_run.log(column, dataNull[column])


# Complete the run
new_run.complete()
