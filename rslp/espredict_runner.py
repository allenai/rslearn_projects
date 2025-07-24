# must be checked into main
# CI builds a docker image
# tagging the commit can cause CI to push the model & configs to ESRun. ???
import logging
import shutil
import uuid
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from tempfile import mkdtemp

from esrun.api.models.step import StepResponse
from esrun.api.models.task import TaskResponseWithStepAndWorkflow
from esrun.api.models.workflow import WorkflowResponse
from esrun.shared.models.prediction_scratch_space import PredictionScratchSpace
from esrun.shared.models.status import Status
from esrun.shared.models.step_type import StepType
from esrun.shared.models.task_args import (
    CombinePartitionsTaskArgs,
    CreatePartitionsTaskArgs,
    DatasetBuildTaskArgs,
    PostprocessPartitionTaskArgs,
    RunInferenceTaskArgs,
    _CommonInferenceTaskArgs,
)
from esrun.shared.models.task_results import (
    CombinePartitionsTaskResults,
    InferenceResultsDataType,
    PostprocessPartitionTaskResults,
)
from esrun.shared.models.workflow_args import PredictionWorkflowArgs
from esrun.shared.models.workflow_type import WorkflowType
from esrun.steps.combine_partitions_step_definition import CombinePartitionsStepDefinition
from esrun.steps.create_partitions_step_definition import CreatePartitionsStepDefinition
from esrun.steps.dataset_build_step_definition import DatasetBuildStepDefinition
from esrun.steps.postprocess_partition_step_definition import PostprocessPartitionStepDefinition
from esrun.steps.run_inference_step_definition import RunInferenceStepDefinition


logger = logging.getLogger(__name__)


class EsPredictRunner:
    def __init__(self, model_config_path: PathLike, dataset_config_path: PathLike, partition_path: PathLike,
                 postprocess_path: PathLike, prediction_request_geometry_path: PathLike,
                 scratch_path: PathLike | None = None):

        self.scratch_path = Path(scratch_path or mkdtemp())

        self.scratch = PredictionScratchSpace(root_path=str(self.scratch_path))

        logger.info(f"scratch path: {self.scratch_path}")

        self.dataset_idx = 0
        Path(self.scratch.get_dataset_dir(self.dataset_idx)).mkdir(parents=True, exist_ok=True)

        # Copy configs into the scratch space
        shutil.copy(prediction_request_geometry_path, self.scratch_path / "prediction_request_geometry.geojson")
        shutil.copy(dataset_config_path, self.scratch.get_dataset_config_path(0))
        shutil.copy(model_config_path, self.scratch_path / "model.yaml")
        shutil.copy(partition_path, self.scratch_path / "partition_strategies.yaml")
        shutil.copy(postprocess_path, self.scratch_path / "postprocessing_strategies.yaml")
        shutil.copy(prediction_request_geometry_path, self.scratch_path / "prediction_request_geometry.geojson")

        workflow_args = PredictionWorkflowArgs(
            workflow_type=WorkflowType.PREDICTION,
            model_pipeline_id=uuid.uuid4(),
            geometry=self.scratch.get_prediction_request_geometry()
        )

        workflow_id = uuid.uuid4()
        self.workflow = WorkflowResponse(
            id=workflow_id,
            args=workflow_args,
            external_id=None,
            progress=0,
            status=Status.PENDING,
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
        )

    def _create_step(self, step_type: StepType) -> StepResponse:
        step_id = uuid.uuid4()
        return StepResponse(
            id=step_id,
            workflow_id=self.workflow.id,
            step_type=step_type,
            step_index=0,
            status=Status.PENDING,
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
            completed_at=None
        )

    def _create_task(self, step_type: StepType, args: _CommonInferenceTaskArgs) -> TaskResponseWithStepAndWorkflow:
        step = self._create_step(step_type)
        return TaskResponseWithStepAndWorkflow(
            id=uuid.uuid4(),
            step_id=step.id,
            step=step,
            workflow=self.workflow,
            status=Status.PENDING,
            args=args,
            created_at=datetime.now(tz=UTC),
            updated_at=datetime.now(tz=UTC),
        )

    def partition(self) -> list[str]:
        (self.scratch_path / "dataset_0").mkdir(parents=True, exist_ok=True)
        partition_args = CreatePartitionsTaskArgs(
            scratch_path=str(self.scratch_path),
            model_stage_path=str(Path(self.scratch_path)),
            model_stage_id=uuid.uuid4(),
            dataset_path=self.scratch.get_dataset_dir(self.dataset_idx),
        )
        task = self._create_task(StepType.CREATE_PARTITIONS, partition_args)

        # Logic to partition the dataset
        step = CreatePartitionsStepDefinition()
        return step.run(task).partition_ids

    def build_dataset(self, partition_id: str):
        dataset_build_args = DatasetBuildTaskArgs(
            partition_id=partition_id,
            scratch_path=str(self.scratch_path),
            model_stage_id=uuid.uuid4(),
            model_stage_path=str(self.scratch_path),
            dataset_path=self.scratch.get_dataset_dir(self.dataset_idx),
        )
        task = self._create_task(StepType.CREATE_PARTITIONS, dataset_build_args)

        # Logic to build the model for the given partition
        step = DatasetBuildStepDefinition()
        step.run(task)

    def run_inference(self, partition_id: str):
        # Logic to run inference on the given partition
        stage_id = uuid.uuid4()
        inference_args = RunInferenceTaskArgs(
            partition_id=partition_id,
            scratch_path=str(self.scratch_path),
            model_stage_id=stage_id,
            model_stage_path=str(self.scratch_path / "models" / f"stage_{stage_id}"),
            dataset_path=self.scratch.get_dataset_dir(self.dataset_idx)
        )
        task = self._create_task(StepType.RUN_INFERENCE, inference_args)

        step = RunInferenceStepDefinition()
        step.run(task)

    def postprocess(self, partition_id: str) -> PostprocessPartitionTaskResults:
        stage_id = uuid.uuid4()

        postprocess_args = PostprocessPartitionTaskArgs(
            partition_id=partition_id,
            scratch_path=str(self.scratch_path),
            model_stage_id=stage_id,
            model_stage_path=str(self.scratch_path / "models" / f"stage_{stage_id}"),
            dataset_path=str(self.scratch_path / "dataset_0"),
            inference_results_data_type=InferenceResultsDataType.VECTOR,
        )
        task = self._create_task(StepType.POSTPROCESS_PARTITION, postprocess_args)

        # Logic to postprocess the results of inference
        step = PostprocessPartitionStepDefinition()
        return step.run(task)

    def combine(self, partition_ids) -> CombinePartitionsTaskResults:
        stage_id = uuid.uuid4()
        inference_files: list[str] = []
        combine_args = CombinePartitionsTaskArgs(
            partition_ids=partition_ids,
            partition_result_file_paths=inference_files,
            scratch_path=str(self.scratch_path),
            model_stage_id=stage_id,
            model_stage_path=str(self.scratch_path / "models" / f"stage_{stage_id}"),
            dataset_path=self.scratch.get_dataset_dir(self.dataset_idx),
            inference_results_data_type=InferenceResultsDataType.VECTOR,
        )
        task = self._create_task(StepType.COMBINE_PARTITIONS, combine_args)

        # Logic to combine results from all partitions
        step = CombinePartitionsStepDefinition()
        return step.run(task)
