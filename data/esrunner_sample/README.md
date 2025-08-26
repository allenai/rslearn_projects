# ES Runner Local Development Guide

## What is esrunner?

ESRunner provides the [EsPredictRunner](https://github.com/allenai/earth-system-run/blob/josh/esrunner/src/esrun/runner/local/predict_runner.py) (and perhaps eventually the EsFineTuneRunner) class which can be used to run predictions and fine-tuning pipelines outside of the esrun service.

## Setting up your environment

- Install `esrunner` (earth-system-run) in your development environment.
  ```
  pip install earth-system-run @ git+https://github.com/allenai/earth-system-run.git
  ```
- Following the project structure below, create a directory in the `rslearn-projects/data/` directory. This directory will contain all the necessary files for your prediction or fine-tuning pipeline.

## Project Structure
- `checkpoint.ckpt`: (Optional)
- `dataset.json`: This is the rslearn dataset definition file.
- `model.yaml`: This is the rslearn (pytorch) model definition file.
- `partition_strategies.yaml`:
- `postprocessing_strategies.yaml`: This file defines how the esrunner will post-process the predictions.
- `requirements.txt`: This file contains the additional Python packages required for the pipeline. It should include any dependencies that are not part of the base environment.
- `prediction/test-request1.geojson`: This directory contains the prediction requests in GeoJSON format. Each file represents a set of prediction requests for a specific region or time period.  Many different prediction requests can be defined within a single file as separate features in the feature collection. The esrunner will partition these requests into smaller tasks based on the partition strategies defined in `partition_strategies.yaml`.
- `run_pipeline.py`: This script is used to run the prediction pipeline. It will read the configuration files and execute the necessary steps to perform predictions or fine-tuning. You can customize this script to suit your specific needs, such as adding additional logging or error handling.

## Partitioning Strategies
This file defines how the esrunner will break the inference request into multiple request geometries for compute parallelization (equivalent to rslearn window groups) and prediction window geometries.

Partitioning strategies can be mixed and matched for flexible development.

  - partition_request_geometry
  - prepare_window_geometries

Available partitioners:
- `FixedWindowPartitioner` - Given a fixed window size, this partitioner will create partitions of that size for each lat/lon or polygon centroid in the prediction request.
- `GridPartitioner` - Given a grid size, this partitioner will create partitions based on the grid cells that intersect with the prediction request.
- NoopPartitioner - Does not partition the prediction request. This is useful for testing or when you want to run the entire prediction request as a single task.

Example `partition_strategies.yaml`. This will leave the original input as a single partition, but will create individual windows of size 128x128 pixels for each feature.
```yaml
partition_request_geometry:
  class_path: esrun.tools.partitioners.noop_partitioner.NoopPartitioner
  init_args:

prepare_window_geometries:
  class_path: esrun.tools.partitioners.fixed_window_partitioner.FixedWindowPartitioner
  init_args:
    window_size: 128 # intended to be a pixel value
```

## Post-Processing Strategies
There are 3 different stages to postprocessing:
  - `postprocess_window()` - This is the stage where individual model outputs are converted into a digestible artifact for the next stage.
  - `postprocess_partition()` - This is the stage where the outputs from the window postprocessors are combined into a single per-partition artifact.
  - `postprocess_dataset()` - This is the final stage of postprocessing where the partition level outputs are combined into a artifact.

## Samples

### Run a pipeline end-to-end

```python file=run_pipeline.py
from rslp.espredict_runner import EsPredictRunner

runner = EsPredictRunner(
    'model.yaml',
    'dataset.json',
    'partition_strategies.yaml',
    'postprocessing_strategies.yaml',
    'prediction/test-request1.geojson',
    scratch_path='scratch/'
)
partitions = runner.partition()
for partition_id in partitions:
    runner.build_dataset(partition_id)
    runner.run_inference(partition_id)
    runner.postprocess(partition_id)

runner.combine(partitions)
```

### Run dataset building for the entire prediction request.
```python file=run_dataset_building.py
from rslp.espredict_runner import EsPredictRunner

runner = EsPredictRunner(
    'model.yaml',
    'dataset.json',
    'partition_strategies.yaml',
    'postprocessing_strategies.yaml',
    'prediction/test-request1.geojson',
    scratch_path='scratch/'
)

for partition_id in runner.partition():
    runner.build_dataset(partition_id)
```

### Run inference for a single partition.
(Assumes you have an existing materialized dataset for the partition.)
```python file=run_inference_single_partition.py
from rslp.espredict_runner import EsPredictRunner

runner = EsPredictRunner(
    'model.yaml',
    'dataset.json',
    'partition_strategies.yaml',
    'postprocessing_strategies.yaml',
    'prediction/test-request1.geojson',
    scratch_path='scratch/'
)
partition_id = 'my-existing-partition-id'  # Replace with the actual partition ID you want to run
runner.run_inference(partition_id)
```

### Run inference for a single window.
Since we don't expose window-level inference via the runner API, you can configure your partitioners to produce limited sets of partitions and windows.

```yaml file=partition_strategies.yaml
strategy_large:
  class_path: esrun.tools.partitioners.noop_partitioner.NoopPartitioner
  init_args:

strategy_small:
  class_path: esrun.tools.partitioners.fixed_window_partitioner.FixedWindowPartitioner
  init_args:
    window_size: 128 # intended to be a pixel value
    limit: 1  # This will limit window generation to a single window per large partition, effectively allowing you to run inference on a single window.
```

```python file=run_inference_single_window.py
from rslp.espredict_runner import EsPredictRunner

runner = EsPredictRunner(
    'model.yaml',
    'dataset.json',
    'partition_strategies.yaml',
    'postprocessing_strategies.yaml',
    'prediction/test-request1.geojson',
    scratch_path='scratch/'
)
partition_id = 'my-existing-partition-id'  # Replace with the actual partition ID you want to run
partitions = runner.partition()
for partition_id in partitions:
    runner.run_inference(partition_id)
```

## Writing Your Own Partitioners
You may supply your own partitioners by creating a new class that implements the ` PartitionInterface` class in the `esrun.tools.partitioners.partition_interface` module.  You can then specify your custom partitioner in the `partition_strategies.yaml` file.  This class must exist on your PYTHONPATH and be importable by the esrunner.  As such we recommend you place your custom partitioner in the `rslp/common/partitioners` directory of this repository to ensure it gets installed into the final Dockerimage artifact.

## Writing your own post-processing strategies
You may supply your own post-processing strategies by creating a new class that implements the `PostprocessInterface` class in the `esrun.tools.postprocessors.postprocess_inferface` module.  You can then specify your custom post-processing strategy in the `postprocessing_strategies.yaml` file.  This class must exist on your `PYTHONPATH` and be importable by the esrunner.  As such we recommend you place your custom post-processing strategy in the `rslp/common/postprocessing` directory of this repository to ensure it gets installed into the final Docker image artifact.

### Testing Partitioner & Post-Processing Implementations
See the [earth-system-run](https://github.com/allenai/earth-system-run) repository for tests covering existing [partitioner](https://github.com/allenai/earth-system-run/tree/v1-develop/tests/unit/tools/partitioners) and [post-processor](https://github.com/allenai/earth-system-run/tree/v1-develop/tests/unit/tools/postprocessors) implementations.

## Longer Term Vision / Model Development Workflow
1. ML folk will create the requisite configs in a directory like this one.
2. Any additional or alternate requirements will be specified in a requirements.txt file in the same directory.
3. When a PR is created, CI will perform a docker build using the main Dockerfile in the root of the repo, but ensure any deviations from the main requirements.txt are merged into the main requirements.txt at build time so that the docker image is built with the correct requirements. This will allow developers to use this docker image for things like beaker runs or other executions (if needed.)
4. When the PR is merged, the docker build from above will be performed again, but the final image will be published to esrun as a new "model" (model version?) using the configurations in this directory.  (TODO: Should we consider "versioning" models in esrun?)
5. Once the "model" has been published to esrun, fine-tuning can be performed using esrun. (Longer term I think we can use a standard versioned helios image for this, but for now we can use the bespoke images created in the previous step.)
6. (Presumably) Once the fine-tuning is complete, esrun will publish the final model (with weights) to esrun as a (new?) model (version?).  Esrun can then be used to run predictions with this final model.
