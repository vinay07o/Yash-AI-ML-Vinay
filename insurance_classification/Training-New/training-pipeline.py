# Import required classes from Azureml
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from azureml.core import Run
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Get the context of the experiment run
new_run = Run.get_context()

# Access the Workspace
ws = new_run.experiment.workspace

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

dataPrep = pd.read_csv(path)

X = dataPrep.drop("fraud_reported", axis=1)
y = dataPrep["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=1234
)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Baseline Random forest based Model
rfc = RandomForestClassifier(
    n_estimators=ne, min_samples_leaf=msl, random_state=20, max_depth=md
)
rfcg = rfc.fit(X_res, y_res)  # fit on training data
Y_predict = rfcg.predict(X_test)

# Get the probability score - Scored Probabilities
Y_prob = rfcg.predict_proba(X_test)[:, 1]

# Get Confusion matrix and the accuracy/score - Evaluate
cm = confusion_matrix(y_test, Y_predict)
accuracy = accuracy_score(y_test, Y_predict)
recall = recall_score(y_test, Y_predict)
precision = precision_score(y_test, Y_predict)
f1_sco = f1_score(y_test, Y_predict)

# Create the confusion matrix dictionary
cm_dict = {
    "schema_type": "confusion_matrix",
    "schema_version": "v1",
    "data": {"class_labels": ["N", "Y"], "matrix": cm.tolist()},
}

new_run.log("TotalObservations", len(dataPrep))
new_run.log_confusion_matrix("ConfusionMatrix", cm_dict)
new_run.log("Accuracy", accuracy)
new_run.log("Precision", precision)
new_run.log("Recall", recall)
new_run.log("F1 Score", f1_sco)
# new_run.log('AUC', np.float(auc))

# Save the model in the run outputs
os.makedirs("outputs", exist_ok=True)
joblib.dump(value=rfc, filename="outputs/insurance_classification.pkl")

# Complete the run
new_run.complete()
