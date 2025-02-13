"""Config for the forest loss driver inference/predict pipeline."""

import multiprocessing
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from upath import UPath

from rslp.utils.rslearn import (
    ApplyWindowsArgs,
    IngestArgs,
    MaterializeArgs,
    MaterializePipelineArgs,
    PrepareArgs,
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

FOREST_LOSS_GEOTIFF_FILENAMES = [
    "070W_10S_060W_00N.tif",
    "070W_20S_060W_10S.tif",
    "080W_10S_070W_00N.tif",
    "080W_20S_070W_10S.tif",
]

DEFAULT_MODEL_CFG_FNAME = "data/forest_loss_driver/config.yaml"
DEFAULT_VIS_LAYER_WORKERS = 32


def get_default_workers() -> int:
    """Get the default number of workers."""
    # Past 128 workers, the disk I/O and network throughput will likely be bottleneck.
    return min(multiprocessing.cpu_count(), 128)


@dataclass
class ExtractAlertsArgs:
    """Arguments for extract_alerts_pipeline.

    Args:
        gcs_tiff_filenames: the list of GCS TIFF filenames to extract alerts from.
        country_data_path: the path to the country data to clip the alerts to.
        conf_prefix: the prefix for the confidence raster of the forest loss alerts.
        date_prefix: the prefix for the date raster of the forest loss alerts.
        prediction_utc_time: the UTC time of the prediction.
        min_confidence: the minimum confidence threshold.
        days: the number of days to consider before the prediction time.
        min_area: the minimum area threshold for an event to be extracted.
        max_number_of_events: the maximum number of events to extract.
    """

    @staticmethod
    def _default_peru_shape_data_path() -> str:
        """The path to the Peru shape data."""
        return UPath(
            f"gcs://{os.environ.get('RSLP_BUCKET', 'rslearn-eai')}/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp"
        )

    gcs_tiff_filenames: list[str] = field(
        default_factory=lambda: FOREST_LOSS_GEOTIFF_FILENAMES
    )
    country_data_path: UPath | None = field(
        default_factory=_default_peru_shape_data_path
    )
    conf_prefix: str = "gs://earthenginepartners-hansen/S2alert/alert/"
    date_prefix: str = "gs://earthenginepartners-hansen/S2alert/alertDate/"
    prediction_utc_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    min_confidence: int = 2
    days: int = 365
    min_area: float = 16.0
    max_number_of_events: int | None = None
    group: str | None = None
    workers: int = get_default_workers()

    # Parameters to fill in for the dataset configuration file.
    # Absolute paths are preferred here so that these directories can be shared across
    # different runs of the pipeline.
    # The default empty string results in using relative path within the dataset root.
    index_cache_dir: str = ""
    tile_store_dir: str = ""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.country_data_path = UPath(self.country_data_path)
        if self.min_confidence < 0:
            raise ValueError("min_confidence must be non-negative")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.min_area <= 0:
            raise ValueError("min_area must be positive")


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
        )
    )
    ingest_args: IngestArgs = field(
        default_factory=lambda: IngestArgs(
            ignore_errors=True,
            apply_windows_args=ApplyWindowsArgs(workers=DEFAULT_VIS_LAYER_WORKERS),
        )
    )
    materialize_args: MaterializeArgs = field(
        default_factory=lambda: MaterializeArgs(
            ignore_errors=True,
            apply_windows_args=ApplyWindowsArgs(workers=DEFAULT_VIS_LAYER_WORKERS),
        ),
    )


@dataclass
class SelectLeastCloudyImagesArgs:
    """Arguments for select_least_cloudy_images_pipeline.

    Args:
        min_choices: the minimum number of images to select.
        num_outs: the number of best images to select.
        workers: the number of workers to use.
    """

    min_choices: int = 5
    num_outs: int = 3
    workers: int = get_default_workers()


@dataclass
class PredictPipelineConfig:
    """Prediction pipeline config for forest loss driver classification.

    These are the arguments that are used to run the pipeline. They can all
    be overridden by the inference yaml file or via command line args.

    Args:
        ds_root: the root path to the dataset.
        model_cfg_fname: the model configuration filename to apply.
        extract_alerts_args: the arguments for the extract_alerts step.
        inference_materialize_args: arguments for materializing inference layers.
        vis_materialize_args: arguments for materializing visualization layers.
        select_least_cloudy_images_args: the arguments for the select_least_cloudy_images step.
    """

    ds_root: str
    model_cfg_fname: str = DEFAULT_MODEL_CFG_FNAME
    extract_alerts_args: ExtractAlertsArgs = field(default_factory=ExtractAlertsArgs)
    inference_materialize_args: InferenceLayerMaterializeArgs = field(
        default_factory=InferenceLayerMaterializeArgs
    )
    vis_materialize_args: VisLayerMaterializeArgs = field(
        default_factory=VisLayerMaterializeArgs
    )

    select_least_cloudy_images_args: SelectLeastCloudyImagesArgs = field(
        default_factory=SelectLeastCloudyImagesArgs
    )

    @property
    def path(self) -> UPath:
        """The path to the dataset."""
        return UPath(self.ds_root)
