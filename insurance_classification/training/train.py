"""
Training Code
"""

import os
import argparse
import pandas as pd
import joblib
from azureml.core import Run
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Split the dataframe into test and train data
def split_data(df, Y):
    """Split data into train and test."""
    X = df.drop(Y, axis=1).values
    y = df[Y].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


def data_balance(X_train, y_train):
    """Balancing imbalanced data using SMOTE."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    data_balanced = {'balanced_X':X_res, 'balanced_y':y_res}
    return data_balanced

# Train the model, return the model
def model_training(X_res, y_res, ne, msl, md):
    "Model Traing with RFC."
    rfc = RandomForestClassifier(
        n_estimators=ne, min_samples_leaf=msl, random_state=20, max_depth=md
    )
    rfc.fit(X_res, y_res)  # fit on training data
    return rfc


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    """Get classification matrics."""
    preds = model.predict(data["test"]["X"])
    accuracy = accuracy_score(data["test"]["y"], preds)
    cm = confusion_matrix(data["test"]["y"], preds)
    recall = recall_score(data["test"]["y"], preds)
    precision = precision_score(data["test"]["y"], preds)
    f1_sco = f1_score(data["test"]["y"], preds)
    # Create the confusion matrix dictionary
    cm_dict = {
        "schema_type": "confusion_matrix",
        "schema_version": "v1",
        "data": {"class_labels": ["N", "Y"], "matrix": cm.tolist()},
    }
    metrics = {"accuracy": accuracy, "confusion_matrix":cm, 
               "recall_score":recall, "precision_score": precision, 
               "f1_score": f1_sco, "confusion_matrix_dict": cm_dict}
    
    return metrics


def main():
    print("Running train.py")

    # Get the context of the experiment run
    new_run = Run.get_context()

    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--min_samples_leaf", type=int)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--datafolder", type=str)

    args, unknown = parser.parse_known_args()

    ne = args.n_estimators
    msl = args.min_samples_leaf
    md = args.max_depth

    # Read the data from the previous step
    path = os.path.join(args.datafolder, "InsuranceData_prep_2.csv")

    # Reading data
    dataPrep = pd.read_csv(path)

    # data splitting 
    data_split = split_data(dataPrep, "fraud_reported")

    # Balancing data
    df_dict_bal = data_balance(data_split['train']['X'], data_split['train']['y'])

    # Model training
    fitted_ob = model_training(df_dict_bal['balanced_X'], df_dict_bal['balanced_y'], ne, msl, md)

    # model evaluation
    model_mtrics_dict = get_model_metrics(fitted_ob, data_split)

    # logging metrics on workspace
    new_run.log("TotalObservations", len(dataPrep))
    new_run.log_confusion_matrix("ConfusionMatrix", model_mtrics_dict['confusion_matrix_dict'])
    new_run.log("Accuracy", model_mtrics_dict['accuracy'])
    new_run.log("Precision", model_mtrics_dict['precision_score'])
    new_run.log("Recall", model_mtrics_dict['recall_score'])
    new_run.log("F1 Score", model_mtrics_dict['f1_score'])

    # Save the model in the run outputs
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(value=fitted_ob, filename="outputs/insurance_classification.pkl")

    # Complete the run
    new_run.complete()


if __name__ == '__main__':
    main()
