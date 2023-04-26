# testing files
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.runconfig import RunConfiguration
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os


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
        create_new=e.rebuild_env,
    )  #
    run_config = RunConfiguration()
    run_config.environment = environment

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


if __name__ == "__main__":
    main()
