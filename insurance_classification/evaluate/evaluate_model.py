import os
from azureml.core import Run
import argparse
import traceback
from azureml.core import Run
from azureml.core import Workspace, Experiment
from azureml.core.model import Model as AMLModel

run = Run.get_context()

if (run.id.startswith('OfflineRun')):
    print("exception")

    workspace_name = os.environ.get("AML-WorkSpace")
    experiment_name = "mlopspython"
    resource_group = ""
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    aml_workspace = Workspace.get(
        name="AML-WorkSpace",
        subscription_id="329034d6-eb63-442d-8652-e7e83d49a345",
        resource_group='V1-ML-RG',
    )
    ws = aml_workspace
    exp = Experiment(ws, "mlopspython")
else:
    ws = run.experiment.workspace
    exp = run.experiment
    run_id = 'amlcompute'

parser = argparse.ArgumentParser("evaluate")

parser.add_argument(
    "--run_id",
    type=str,
    help="Training run ID",
)

parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="diabetes_model.pkl",
)

parser.add_argument(
    "--allow_run_cancel",
    type=str,
    help="Set this to false to avoid evaluation step from cancelling run after an unsuccessful evaluation",  # NOQA: E501
    default="true",
)

args = parser.parse_args()
model_name = args.model_name
metric_eval = "accuracy"

allow_run_cancel = args.allow_run_cancel

# Parameterize the matrices on which the models should be compared
# Add golden data set on which all the model performance can be evaluated
try:
    firstRegistration = False
    tag_name = 'experiment_name'

    models = AMLModel.list(
            ws, name=model_name, tags=None, latest=True)
    if len(models) == 1:
        model = models[0]
    elif len(models) > 1:
        raise Exception("Expected only one model")
    
    if (model is not None):
        production_model_accuracy = 10000
        if (metric_eval in model.tags):
            production_model_accuracy = float(model.tags[metric_eval])
        try:
            run_pipeline = [i for i in exp.get_runs()][0]
            training_job = [i for i in run_pipeline.get_children()][2]
            best_run_value = [best_run.get_children_sorted_by_primary_metric()[0] for best_run in training_job.get_children()]

            new_model_accuracy = float(best_run_value[0]['best_primary_metric'])
        except TypeError:
            new_model_accuracy = None
        if (production_model_accuracy is None or new_model_accuracy is None):
            print("Unable to find ", metric_eval, " metrics, "
                  "exiting evaluation")
            if((allow_run_cancel).lower() == 'true'):
                run.parent.cancel()
        else:
            print(
                "Current Production model {}: {}, ".format(
                    metric_eval, production_model_accuracy) +
                "New trained model {}: {}".format(
                    metric_eval, new_model_accuracy
                )
            )

        if (new_model_accuracy < production_model_accuracy):
            print("New trained model performs better, "
                  "thus it should be registered")
        else:
            print("New trained model metric is worse than or equal to "
                  "production model so skipping model registration.")
            if((allow_run_cancel).lower() == 'true'):
                run.parent.cancel()
    
    else:
        print("This is the first model, "
              "thus it should be registered")

except Exception:
    traceback.print_exc(limit=None, file=None, chain=True)
    print("Something went wrong trying to evaluate. Exiting.")
    raise