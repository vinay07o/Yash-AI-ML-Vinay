"""Model register codes"""
import json
import os
import argparse
from azureml.core import Run, Experiment, Workspace

def main():

    run = Run.get_context()
    if (run.id.startswith('OfflineRun')):
        from dotenv import load_dotenv
        # For local development, set values in this section
        load_dotenv()
        workspace_name = os.environ.get("WORKSPACE_NAME")
        experiment_name = os.environ.get("EXPERIMENT_NAME")
        resource_group = os.environ.get("RESOURCE_GROUP")
        subscription_id = os.environ.get("SUBSCRIPTION_ID")
        # run_id useful to query previous runs
        run_id = "bd184a18-2ac8-4951-8e78-e290bef3b012"
        aml_workspace = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group
        )
        ws = aml_workspace
        exp = Experiment(ws, experiment_name)
    else:
        ws = run.experiment.workspace
        exp = run.experiment
        run_id = 'amlcompute'

    parser = argparse.ArgumentParser("register")

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="insurance_classification.pkl",
    )

    parser.add_argument(
        "--step_input",
        type=str,
        help=("input from previous steps")
    )

    args = parser.parse_args()
    if (args.run_id is not None):
        run_id = args.run_id
    if (run_id == 'amlcompute'):
        run_id = run.parent.id
    model_name = args.model_name
    model_path = args.step_input

    print("Getting registration parameters")

    # Load the registration parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        register_args = pars["registration"]
    except KeyError:
        print("Could not load registration values from file")
        register_args = {"tags": []}

    model_tags = {}
    for tag in register_args["tags"]:
        try:
            mtag = run.parent.get_metrics()[tag]
            model_tags[tag] = mtag
        except KeyError:
            print(f"Could not find {tag} metric on parent run.")

    # load the model
    print("Loading model from " + model_path)

    run_pipeline = [i for i in exp.get_runs()][0]

    training_job = [i for i in run_pipeline.get_children()][2]

    best_run_value = [best_run.get_children_sorted_by_primary_metric()[0] for best_run in training_job.get_children()]

    ran_ob = Run(exp, best_run_value[0]['run_id'])

    tags = {'accuracy': best_run_value[0]['best_primary_metric'], "area": "insurance_classification", "experiment_name": exp.name}

    ran_ob.get_metrics()['ConfusionMatrix']

    ran_ob.register_model(model_path='outputs/insurance_classification.pkl', model_name=model_name,
                        properties={'Accuracy': best_run_value[0]['best_primary_metric'], 'ConfusionMatrix': ran_ob.get_metrics()['ConfusionMatrix']},
                        tags = tags )
