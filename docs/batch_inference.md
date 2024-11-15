# Batch Inference Conventions and Ideals

In this document, we will describe our conventions on how to setup batch inference for different applications. As of 11/14/24 some of this has been implemented, while some is aspirational.

## Key Features

- **Platform Agnostic**: Easily be able to deploy code to Beaker or a cloud provider
- **[rslearn](https://github.com/allenai/rslearn)**: All the core logic for data ingestion,preparation and inference lives in rslearn
- **Explicit Pipeline Steps**: Each step has explicit I/O requirements
- **Easy to Test**: We include examples with cropped and small amounts of data to enable easily working on pipelines
- **Docker-based Execution**: Containerized steps for reproducibility
- **Single Configuration for pipeline**: Each project has a single configuration for its core pipeline


### Design Principles

1. **Pipeline Configuration**
    - Each pipeline should have a single config file that is ingested/validated by a python class
        -  The model and data configurations should be stored in a canonical location for each pipeline, i.e we change the configuration in this location, not where we point
    - All hardware related configurations both within rslp and within rslearn should be exposed to the pipeline, i.e number of workers for a given step or choice to run on gpu
    - Geographic locations to run the pipeline on should be configurable (either by setting windows, or specifying the input file area to look for)
        - We should be able to eaisly run the pipeline on a small image or area only


2. **Dataset Prepare/Ingestion/Materialize**
   - All dataset preparation steps happens through [rslearn](https://github.com/allenai/rslearn)
   - We do not use Earth Engine
   - Expected I/O for rslearn based steps should be explicit (especially with assumed directory structures)

2. **Model Predict**
   - Prefer running model based components on beaker
   - Prefer using shared beaker launching features where possible
   - Use beaker-py to split up data across variety of jobs
   - Strive for balance between task length and number of tasks being launched
   - All tasks ran on beaker should be pre-emtible and low priority
   - Set max time limit to take advantage of scheduling advantages where possible

5. **Deployment**
    - Use rslp_main and workflows to create entrypoints into pipelines
    - All pipelines should be portable on docker
    - Data ingestion can be run on GCP or via a beaker job
    - Scheduled pipelines are intiated in gha

6. **Testing**
    - Each step should have an integration test
    - Each entire pipeline should have an integration test
    - Prefer to create crops to run these tests for easy debugging etc

7. **Miscellaneous**
    -  Expected I/O for all steps in the pipeline should be explicit (especially with assumed directory structures) and all required env vars should be clearly validated




## Open Questions
- How do we manage rolling updates of outputs as new data is available? Do we update history?
