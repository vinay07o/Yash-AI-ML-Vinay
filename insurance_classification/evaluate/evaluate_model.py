from azureml.core import Run
import argparse
import traceback
from model_helper import get_model

run = Run.get_context()

exp = run.experiment
ws = run.experiment.workspace
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
if (args.run_id is not None):
    run_id = args.run_id
if (run_id == 'amlcompute'):
    run_id = run.parent.id
model_name = args.model_name
metric_eval = "accuracy"

allow_run_cancel = args.allow_run_cancel

# Parameterize the matrices on which the models should be compared
# Add golden data set on which all the model performance can be evaluated
try:
    firstRegistration = False
    tag_name = 'experiment_name'

    model = get_model(
                model_name=model_name,
                tag_name=tag_name,
                tag_value=exp.name,
                aml_workspace=ws)
    
    if (model is not None):
        production_model_mse = 10000
        if (metric_eval in model.tags):
            production_model_mse = float(model.tags[metric_eval])
        try:
            new_model_mse = float(run.parent.get_metrics().get(metric_eval))
        except TypeError:
            new_model_mse = None
        if (production_model_mse is None or new_model_mse is None):
            print("Unable to find ", metric_eval, " metrics, "
                  "exiting evaluation")
            if((allow_run_cancel).lower() == 'true'):
                run.parent.cancel()
        else:
            print(
                "Current Production model {}: {}, ".format(
                    metric_eval, production_model_mse) +
                "New trained model {}: {}".format(
                    metric_eval, new_model_mse
                )
            )

        if (new_model_mse < production_model_mse):
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