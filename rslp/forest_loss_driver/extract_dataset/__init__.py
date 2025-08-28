"""Dataset extraction pipeline."""

import multiprocessing
from dataclasses import dataclass, field

from upath import UPath

from rslp.utils.rslearn import (
    ApplyWindowsArgs,
    IngestArgs,
    MaterializeArgs,
    MaterializePipelineArgs,
    PrepareArgs,
    materialize_dataset,
)

from .extract_alerts import ExtractAlertsArgs, extract_alerts
from .least_cloudy_image_selector import (
    SelectLeastCloudyImagesArgs,
    select_least_cloudy_images_pipeline,
)

VISUALIZATION_ONLY_LAYERS = [
    "planet_pre_0",
    "planet_pre_1",
    "planet_pre_2",
    "planet_post_0",
    "planet_post_1",
    "planet_post_2",
]

INFERENCE_LAYERS = [
    "pre_0",
    "pre_1",
    "pre_2",
    "pre_3",
    "pre_4",
    "pre_5",
    "pre_6",
    "post",
]

DEFAULT_VIS_LAYER_WORKERS = 32


def get_default_workers() -> int:
    """Get the default number of workers."""
    # Past 128 workers, the disk I/O and network throughput will likely be bottleneck.
    return min(multiprocessing.cpu_count(), 128)


@dataclass
class InferenceLayerMaterializeArgs(MaterializePipelineArgs):
    """Arguments for materialize_dataset, with defaults for non-visualization layers.

    Args:
        disabled_layers: the list of layers to disable for prepare/ingest/materialize.
        prepare_args: the arguments for the prepare step.
        ingest_args: the arguments for the ingest step.
        materialize_args: the arguments for the materialize step.
    """

    disabled_layers: list[str] = field(
        default_factory=lambda: list(VISUALIZATION_ONLY_LAYERS)
    )
    prepare_args: PrepareArgs = field(
        default_factory=lambda: PrepareArgs(
            apply_windows_args=ApplyWindowsArgs(
                use_initial_job=True, workers=get_default_workers()
            ),
        )
    )
    ingest_args: IngestArgs = field(
        default_factory=lambda: IngestArgs(
            ignore_errors=True,
            apply_windows_args=ApplyWindowsArgs(workers=get_default_workers()),
        )
    )
    materialize_args: MaterializeArgs = field(
        default_factory=lambda: MaterializeArgs(
            ignore_errors=True,
            apply_windows_args=ApplyWindowsArgs(workers=get_default_workers()),
        ),
    )


@dataclass
class VisLayerMaterializeArgs(MaterializePipelineArgs):
    """Arguments for materialize_dataset, with defaults for the visualization layers.

    These layers require fewer workers to operate properly due to API rate limit.

    Args:
        disabled_layers: the list of layers to disable for prepare/ingest/materialize.
        prepare_args: the arguments for the prepare step.
        ingest_args: the arguments for the ingest step.
        materialize_args: the arguments for the materialize step.
    """

    disabled_layers: list[str] = field(default_factory=lambda: list(INFERENCE_LAYERS))
    prepare_args: PrepareArgs = field(
        default_factory=lambda: PrepareArgs(
            apply_windows_args=ApplyWindowsArgs(
                use_initial_job=True, workers=DEFAULT_VIS_LAYER_WORKERS
            ),
            retry_max_attempts=5,
        ),
    )
    ingest_args: IngestArgs = field(
        default_factory=lambda: IngestArgs(
            apply_windows_args=ApplyWindowsArgs(workers=DEFAULT_VIS_LAYER_WORKERS),
            retry_max_attempts=5,
        )
    )
    materialize_args: MaterializeArgs = field(
        default_factory=lambda: MaterializeArgs(
            ignore_errors=True,
            apply_windows_args=ApplyWindowsArgs(workers=DEFAULT_VIS_LAYER_WORKERS),
        ),
    )


def extract_dataset(
    ds_path: str | UPath,
    extract_alerts_args: ExtractAlertsArgs = ExtractAlertsArgs(),
    select_least_cloudy_images_args: SelectLeastCloudyImagesArgs = SelectLeastCloudyImagesArgs(),
    inference_materialize_args: InferenceLayerMaterializeArgs = InferenceLayerMaterializeArgs(),
    vis_materialize_args: VisLayerMaterializeArgs = VisLayerMaterializeArgs(),
) -> None:
    """Integrated dataset extraction pipeline.

    The pipeline runs these steps:

    1. Create initial dataset by extracting from GLAD alerts.
    2. Materialize the inference layers (Sentinel-2 images).
    3. Materialize the visualization layers (Planet Labs images).
    4. Select least cloudy images within each window.

    Args:
        ds_path: the dataset path to write to.
        extract_alerts_args: configuration for processing GLAD alerts.
        select_least_cloudy_images_args: configuration for selecting the least cloudy
            images.
        inference_materialize_args: materialization arguments for inference
            layers.
        vis_materialize_args: materialization arguments for visualization layers.
    """
    ds_path = UPath(ds_path) if not isinstance(ds_path, UPath) else ds_path
    extract_alerts(ds_path, extract_alerts_args)
    materialize_dataset(ds_path, inference_materialize_args)
    materialize_dataset(ds_path, vis_materialize_args)
    select_least_cloudy_images_pipeline(ds_path, select_least_cloudy_images_args)
