# testing files
import os

from azureml.core import (
    Dataset,
    Datastore,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import HyperDriveStep, PythonScriptStep
from azureml.train.hyperdrive import (
    HyperDriveConfig,
    PrimaryMetricGoal,
    RandomParameterSampling,
    choice,
)

from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment


def main():
    e = Env()
    # Get Azure machine learning workspace
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    print("get_workspace:")
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(aml_workspace, e.compute_name, e.vm_size)
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    # Create a reusable Azure ML environment
    environment = get_environment(
        aml_workspace,
        e.aml_env_name,
        conda_dependencies_file=e.aml_env_train_conda_dep_file,
        create_new=True,
    )  
    run_config = RunConfiguration()
    run_config.environment = environment
    run_config.target = aml_compute

    registered_env = Environment.get(aml_workspace, e.aml_env_name)

    if e.datastore_name:
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables[
        "DATASTORE_NAME"
    ] = datastore_name  # NOQA: E501

    # Get dataset name
    dataset_name = e.dataset_name

    # Check to see if dataset exists
    if dataset_name not in aml_workspace.datasets:
        # This call creates an example CSV from sklearn sample data. If you
        # have already bootstrapped your project, you can comment this line
        # out and use your own CSV

        # Use a CSV to read in the data set.
        file_name = "data/insurance_claims.csv"

        if not os.path.exists(file_name):
            raise Exception(
                'Could not find CSV dataset at "%s". If you have bootstrapped your project, you will need to provide a CSV.'  # NOQA: E501
                % file_name
            )  # NOQA: E501

        # Upload file to default datastore in workspace
        datatstore = Datastore.get(aml_workspace, datastore_name)

        datatstore.upload_files(
            files=[file_name],
            target_path="",
            overwrite=True,
            show_progress=False,
        )

        # Register dataset
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datatstore, "insurance_claims.csv")
        )
        dataset = dataset.register(
            workspace=aml_workspace,
            name=dataset_name,
            description="insurance training data",
            tags={"format": "CSV"},
            create_new_version=True,
        )
    else:
        print("Dataset already registered.")

    # Create a PipelineData to pass data between steps
    pipeline_data = PipelineData(
        "pipeline_data", datastore=aml_workspace.get_default_datastore()
    )

    # Getting Data
    input_ds = aml_workspace.datasets.get("insurance_ds")

    dataFolder = PipelineData(
        "datafolder", datastore=aml_workspace.get_default_datastore()
    )
    dataFolderoutput = PipelineData(
        "datafolderoutput", datastore=aml_workspace.get_default_datastore()
    )

    # creating Hyperdrive sampling
    hyper_params = RandomParameterSampling(
        {
            "--n_estimators": choice(range(10, 100, 10)),
            "--min_samples_leaf": choice(range(1, 10)),
            "--max_depth": choice(range(1, 10)),
        }
    )

    script_config = ScriptRunConfig(
        source_directory=".",
        script="insurance_classification/training/train.py",
        arguments=["--datafolder", dataFolder],
        environment=registered_env,
        compute_target=aml_compute,
    )

    hd_config = HyperDriveConfig(
        run_config=script_config,
        hyperparameter_sampling=hyper_params,
        policy=None,
        primary_metric_name="Accuracy",
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=2,
    )
    # Step 01 - Data Preparation
    dataPrep_step = PythonScriptStep(
        name="01 Data Preparation",
        source_directory=".",
        script_name="insurance_classification/Pre-processing/Pre-Processing.py",
        inputs=[input_ds.as_named_input("insurance_ds")],
        outputs=[dataFolder],
        runconfig=run_config,
        arguments=["--datafolder", dataFolder],
    )

    # Setup Hyperdrivestep for pipeline creation
    hd_step = HyperDriveStep(
        name="02 Train & Tune Model",
        hyperdrive_config=hd_config,
        inputs=[dataFolder],
        metrics_output=dataFolderoutput,
    )

    evaluate_step = PythonScriptStep(
        name="03 Evaluate Model",
        script_name=e.evaluate_script_path,
        source_directory=e.sources_directory_train,
        arguments=[
            "--model_name",
            "insurance_classification.pkl",
            "--allow_run_cancel",
            e.allow_run_cancel,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    # Creating Pipeline
    train_pipeline = Pipeline(workspace=aml_workspace, steps=[dataPrep_step, hd_step, evaluate_step])

    train_pipeline._set_experiment_name
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=e.pipeline_name,
        description="Model training/retraining pipeline",
        version=e.build_id,
    )
    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")


if __name__ == "__main__":
    main()
