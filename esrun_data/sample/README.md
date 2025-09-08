# ES Runner Local Development Guide

## What is esrunner?

ESRunner provides:

- the [EsPredictRunner](https://github.com/allenai/earth-system-run/blob/josh/esrunner/src/esrun/runner/local/predict_runner.py)
- the [EsFineTuneRunner](https://github.com/allenai/earth-system-run/blob/josh/esrunner/src/esrun/runner/local/fine_tune_runner.py)

classes, which can be used to run prediction and fine-tuning pipelines outside of the esrun service architecture


## Setting up your environment

- Install `esrunner` (earth-system-run) in your development environment.
  ```
  pip install earth-system-run @ git+https://github.com/allenai/earth-system-run.git
  ```
- Following the project structure below, create a directory in the `rslearn-projects/esrun_data/` directory. This directory will contain all the necessary files for your prediction or fine-tuning pipeline.

## Project Structure
- `checkpoint.ckpt`:  This is the model checkpoint file. It is required for running inference. If you are only building datasets, this file is not required.  Note: You probably don't want to check this file into git repository.
- `dataset.json`: This is the rslearn dataset definition file.
- `esrun.yaml`: This file defines the behavior of the esrunner including partitioning, postprocessing, training window prep, etc..
- `model.yaml`: This is the rslearn (pytorch) model definition file.
- `annotation_features.geojson`: Labeled annotation feature collection, exported from Studio. Only required for labeled window prep.
- `annotation_task_features.geojson`: Studio tasks for the annotation features, also exported from Studio. Only required for labeled window prep.
- `prediction/test-request1.geojson`: This directory contains the prediction requests in GeoJSON format. Each file represents a set of prediction requests for a specific region or time period.  Many different prediction requests can be defined within a single file as separate features in the feature collection. The esrunner will partition these requests into smaller tasks based on the partition strategies defined in `esrun.yaml#partition_strategies`

## Fine-Tuning

Fine-tuning is encapsulated in the Fine Tuning Workflow, accessible through `EsFineTuningRunner`. It currently only exposes a method for preparing labeled RSLearn windows from geojson feature collections exported through Earth System Studio. Using it requires your `esrun.yaml` to define the following data processing pipeline:

```yaml
window_prep:
  sampler:
  labeled_window_preparer:
  data_splitter:
```

### sampler

Technically optional, defaulting to `NoopSampler`. These classes receive a `list[AnnotationTask]` and are expected to return the same, filtered down by whatever needs your application has.

### labeled_window_preparer

Transforms individual `AnnotationTask` instances to `list[LabeledWindow[LabeledSTGeometry]]` or `list[LabeledWindow[ndarray]]` depending on whether vector or raster label output layers are desired.

Available window preparers:
  - `PointToPixelWindowPreparer` - Converts each annotation feature in a Studio task to a 1x1pixel window with a vector class label
  - `PolygonToRasterWindowPreparer` - Converts a Studio task + its (multi/)polygon annotations into a uint8 2d class matrix

### data_splitter

Given a `LabeledWindow`, assign it to `train`, `val`, or `test`.

Available data splitters:
  - `RandomDataSplitter` - weighted random assignment

### Run a pipeline end-to-end

A fully functional `esrun.yaml` and set of `.geojson` files is available in `esrun_data/sample` as a reference example.
Exercise it via:

```
python -m rslp.main esrun prepare_labeled_windows \
    --project_path esrun_data/sample \
    --scratch_path /tmp/scratch
```

to produce labeled training windows at:

```
/tmp/scratch/dataset
```

### Getting the geojson files

Window labeling requires ES Studio Task + Annotation-formatted FeatureCollection files. The best way to get compliant
data is to upload your raw data via Studio's Command Center "Add Dataset" feature, and export to the desired
format via the "Export Annotations" tab. This will create the required data files in gcs, that you can then download to your working location.

### Writing Your Own Samplers
You may supply your own data samplers by creating a new class that implements the `SamplerInterface` class in the `esrun.runner.tools.samplers.sampler_interface` module. You can then specify your custom sampler in the `esrun.yaml` file. This
class must be importable via your PYTHONPATH. Include it as code in this repository or as a new implementation in earth-system-run.git.

### Writing Your Own LabeledWindowPreparers
You may supply new implementations for converting raw Studio Tasks + Annotations into LabeledWindows. To do so, implement
either `esrun.runner.tools.labeled_window_preparers.labeled_window_preparer.RasterLabelsWindowPreparer` (for rasterized targets) or `esrun.runner.tools.labeled_window_preparers.labeled_window_preparer.VectorLabelsWindowPreparer` (for vector targets). As with Samplers, these must be importable from your PYTHONPATH and can be referenced by class path in `esrun.yaml`. Include as code in this repository or contribute directly to earth-system-run.git.

### Writing Your Own DataPartitioners
You may supply your own data partitioners to determine test/eval/train split assignment for a LabeledWindow. To do so, implement `esrun.runner.tools.data_splitter.data_splitter_interface.DataSplitterInterface`.

## Inference

Inference is encapsulated in the Prediction Workflow, accessible through `EsPredictRunner`. It requires your `esrun.yaml` define:

- partitioning strategy
- post-processing strategy

### Partitioning Strategies
These stanzas defines how esrunner will break the inference request into multiple request geometries for compute parallelization (equivalent to rslearn window groups) and prediction window geometries.

Partitioning strategies can be mixed and matched for flexible development.
  - partition_request_geometry
  - prepare_window_geometries

Available partitioners:
- `FixedWindowPartitioner` - Given a fixed window size, this partitioner will create partitions of that size for each lat/lon or polygon centroid in the prediction request.
- `GridPartitioner` - Given a grid size, this partitioner will create partitions based on the grid cells that intersect with the prediction request.
- NoopPartitioner - Does not partition the prediction request. This is useful for testing or when you want to run the entire prediction request as a single task.

Example `esrun.yaml`. This will leave the original input as a single partition, but will create individual windows of size 128x128 pixels for each feature.
```yaml
partition_request_geometry:
  class_path: esrun.tools.partitioners.noop_partitioner.NoopPartitioner
  init_args:

prepare_window_geometries:
  class_path: esrun.tools.partitioners.fixed_window_partitioner.FixedWindowPartitioner
  init_args:
    window_size: 128 # intended to be a pixel value
```

### Post-Processing Strategies
There are 3 different stages to postprocessing:
  - `postprocess_window()` - This is the stage where individual model outputs are converted into a digestible artifact for the next stage.
  - `postprocess_partition()` - This is the stage where the outputs from the window postprocessors are combined into a single per-partition artifact.
  - `postprocess_dataset()` - This is the final stage of postprocessing where the partition level outputs are combined into a artifact.

### Samples

#### Run a pipeline end-to-end

The simplest way to run a pipeline is to use the `esrun-local-predict` CLI command.  This command will run the entire pipeline end-to-end including partitioning, dataset building, inference, post-processing, and combining the final outputs.
```
$ esrun-local-predict
```

If you want more flexibility, you can use the `EsPredictRunner` class directly.  The following example shows how to run the entire pipeline end-to-end using the `EsPredictRunner` class.  Note: This example may become out of date very quickly due to ongoing changes in the EsPredictRunner class.  Refer to the esrun repo for the most up-to-date information.

```python file=run_pipeline.py
from pathlib import Path
from esrun.runner.local.predict_runner import EsPredictRunner

config_path = Path(__file__).parent

runner = EsPredictRunner(
    project_path=config_path,
    scratch_path=config_path / "scratch",
)
partitions = runner.partition()
for partition_id in partitions:
    runner.build_dataset(partition_id)
    runner.run_inference(partition_id)
    runner.postprocess(partition_id)

runner.combine(partitions)
```

#### Run dataset building for the entire prediction request.
```python file=run_dataset_building.py
from pathlib import Path
from esrun.runner.local.predict_runner import EsPredictRunner

config_path = Path(__file__).parent

runner = EsPredictRunner(
    project_path=config_path,
    scratch_path=config_path / "scratch",
)

for partition_id in runner.partition():
    runner.build_dataset(partition_id)
```

#### Run inference for a single partition.
(Assumes you have an existing materialized dataset for the partition.)
```python file=run_inference_single_partition.py
from pathlib import Path
from esrun.runner.local.predict_runner import EsPredictRunner

config_path = Path(__file__).parent

runner = EsPredictRunner(
    project_path=config_path,
    scratch_path=config_path / "scratch",
)
partition_id = 'my-existing-partition-id'  # Replace with the actual partition ID you want to run
runner.run_inference(partition_id)
```

#### Run inference for a single window.
Since we don't expose window-level inference via the runner API, you can configure your partitioners to produce limited sets of partitions and windows.

```yaml file=esrun.yaml
partition_request_geometry:
  class_path: esrun.runner.tools.partitioners.noop_partitioner.NoopPartitioner
  init_args:

prepare_window_geometries:
  class_path: esrun.runner.tools.partitioners.fixed_window_partitioner.FixedWindowPartitioner
  init_args:
    window_size: 128 # intended to be a pixel value
    limit: 1  # This will limit window generation to a single window per large partition, effectively allowing you to run inference on a single window.
```

```python file=run_inference_single_window.py
from pathlib import Path
from esrun.runner.local.predict_runner import EsPredictRunner

config_path = Path(__file__).parent

runner = EsPredictRunner(
    project_path=config_path,
    scratch_path=config_path / "scratch",
)
partition_id = 'my-existing-partition-id'  # Replace with the actual partition ID you want to run
partitions = runner.partition()
for partition_id in partitions:
    runner.run_inference(partition_id)
```

### Writing Your Own Partitioners
You may supply your own partitioners by creating a new class that implements the ` PartitionInterface` class in the `esrun.runner.tools.partitioners.partition_interface` module.  You can then specify your custom partitioner in the `esrun.yaml` file.  This class must exist on your PYTHONPATH and be importable by the esrunner.  As such we recommend you place your custom partitioner in the `rslp/common/partitioners` directory of this repository to ensure it gets installed into the final Dockerimage artifact.

### Writing your own post-processing strategies
You may supply your own post-processing strategies by creating a new class that implements the `PostprocessInterface` class in the `esrun.runner.tools.postprocessors.postprocess_inferface` module.  You can then specify your custom post-processing strategy in the `postprocessing_strategies.yaml` file.  This class must exist on your `PYTHONPATH` and be importable by the esrunner.  As such we recommend you place your custom post-processing strategy in the `rslp/common/postprocessing` directory of this repository to ensure it gets installed into the final Docker image artifact.

#### Testing Partitioner & Post-Processing Implementations
See the [earth-system-run](https://github.com/allenai/earth-system-run) repository for tests covering existing [partitioner](https://github.com/allenai/earth-system-run/tree/v1-develop/tests/unit/runner/tools/partitioners) and [post-processor](https://github.com/allenai/earth-system-run/tree/v1-develop/tests/unit/runner/tools/postprocessors) implementations.

## Longer Term Vision / Model Development Workflow
1. ML folk will create the requisite configs in a directory like this one.
2. Any additional or alternate requirements will be specified in a requirements.txt file in the same directory.
3. When a PR is created, CI will perform a docker build using the main Dockerfile in the root of the repo, but ensure any deviations from the main requirements.txt are merged into the main requirements.txt at build time so that the docker image is built with the correct requirements. This will allow developers to use this docker image for things like beaker runs or other executions (if needed.)
4. When the PR is merged, the docker build from above will be performed again, but the final image will be published to esrun as a new "model" (model version?) using the configurations in this directory.  (TODO: Should we consider "versioning" models in esrun?)
5. Once the "model" has been published to esrun, fine-tuning can be performed using esrun. (Longer term I think we can use a standard versioned helios image for this, but for now we can use the bespoke images created in the previous step.)
6. (Presumably) Once the fine-tuning is complete, esrun will publish the final model (with weights) to esrun as a (new?) model (version?).  Esrun can then be used to run predictions with this final model.
