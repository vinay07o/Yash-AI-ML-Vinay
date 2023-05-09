## Repo Details

### Directory Structure

High level directory structure for this repository:

```bash
├── .pipelines                  <- Azure DevOps YAML pipelines for CI, PR and model training and deployment.
├── data                        <- Initial set of data to train and evaluate model. Not for use to store data.
├── insurance_classification    <- The top-level folder for the ML project.
│   ├── evaluate                <- Python script to evaluate trained ML model.
|   ├── Pre-processing          <- Python script to pre-process the data.
│   ├── register                <- Python script to register trained ML model with Azure Machine Learning Service.
│   ├── scoring           <- Python score.py to deploy trained ML model.
│   ├── training          <- Python script to train ML model.
│   ├── util              <- Python script for various utility operations specific to this ML project.
├── docs                  <- Extensive markdown documentation for entire project.
├── environment_setup     <- The top-level folder for everything related to infrastructure.
│   ├── arm-templates     <- Azure Resource Manager(ARM) templates to build infrastructure needed for this project. 
├── ml_service            <- The top-level folder for all Azure Machine Learning resources.
│   ├── pipelines         <- Python script that builds Azure Machine Learning pipelines.
│   ├── util              <- Python script for various utility operations specific to Azure Machine Learning.
├── .env.example          <- Example .env file with environment for local development experience.  
├── .gitignore            <- A gitignore file specifies intentionally un-tracked files that Git should ignore.  
├── README.md             <- The top-level README for developers using this project.  
```

The repository provides a template with folders structure suitable for maintaining multiple ML projects. There are common folders such as ***.pipelines***, ***environment_setup***, ***ml_service*** and folders containing the code base for each ML project. This repository contains a single sample ML project in the ***insurance_classification*** folder.

### Environment Setup

- `environment_setup/iac-*-arm.yml, arm-templates` : Infrastructure as Code piplines to create required resources using ARM, along with corresponding arm-templates. Infrastructure as Code can be deployed with this template or with the Terraform template.

- `environment_setup/iac-remove-environment.yml` : Infrastructure as Code piplines to delete the created required resources.

### Pipelines

- `.pipelines/code-quality-template.yml` : a pipeline template used by the CI and PR pipelines. It contains steps performing linting, data and unit testing.
- `.pipelines/insurance_classification-ci.yml` : a pipeline triggered when the code is merged into **master**. It performs linting, data integrity testing, unit testing, building and publishing an ML pipeline.
- `.pipelines/insurance_classification-cd.yml` : a pipeline triggered when the code is merged into **master** and the `.pipelines/insurance_classification-ci.yml` completes. Deploys the model to ACI, AKS or Webapp.
- `.pipelines/insurance_classification-get-model-id-artifact-template.yml` : a pipeline template used by the `.pipelines/insurance_classification-cd.yml` pipeline. It takes the model metadata artifact published by the previous pipeline and gets the model ID.
- `.pipelines/insurance_classification-publish-model-artifact-template.yml` : a pipeline template used by the `.pipelines/insurance_classification-ci.yml` pipeline. It finds out if a new model was registered and publishes a pipeline artifact containing the model metadata.
- `.pipelines/pr.yml` : a pipeline triggered when a **pull request** to the **master** branch is created. It performs linting, data integrity testing and unit testing only.

### ML Services

- `ml_service/pipelines/insurance_classification_build_train_pipeline.py` : builds and publishes an ML training pipeline. It uses Python on ML Compute.
- `ml_service/pipelines/run_train_pipeline.py` : invokes a published ML training pipeline (Python on ML Compute) via REST API.
- `ml_service/util` : contains common utility functions used to build and publish an ML training pipeline.

### Environment Definitions

- `insurance_classification/conda_dependencies.yml` : Conda environment definition for the environment used for both training and scoring (Docker image in which train.py and score.py are run).
- `insurance_classification/ci_dependencies.yml` : Conda environment definition for the CI environment.

### Pre-processing Step

- `insurance_classification/Pre-processing/pre-processing.py` : Data Pre processing python script.

### Training Step

- `insurance_classification/training/train.py` : ML functionality called by train_aml.py
- `insurance_classification/training/R/test_train.py` : a unit test for the training script(s)

### Evaluation Step

- `insurance_classification/evaluate/evaluate_model.py` : an evaluating step which cancels the pipeline in case of non-improvement.

### Registering Step

- `insurance_classification/register/register_model.py` : registers a new trained model if evaluation shows the new model is more performant than the previous one.

### Scoring

- `insurance_classification/scoring/score.py` : a scoring script which is about to be packed into a Docker Image along with a model while being deployed to QA/Prod environment.
- `insurance_classification/scoring/inference_config.yml`, `deployment_config_aci.yml` : configuration files for the [AML Model Deploy](https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.private-vss-services-azureml&ssr=false#overview) pipeline task for ACI deployment targets.

