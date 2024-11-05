"""Config for the forest loss driver inference/predict pipeline."""

import os
from datetime import datetime, timezone

from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)


# Where to obtain the forest loss alert data.
GCS_CONF_PREFIX = "gs://earthenginepartners-hansen/S2alert/alert/"
GCS_DATE_PREFIX = "gs://earthenginepartners-hansen/S2alert/alertDate/"


class PredictPipelineConfig:
    """Prediction pipeline config for forest loss driver classification."""

    # Maybe put this in init or must be in a constants file?
    rslp_bucket = os.environ.get("RSLP_BUCKET", "rslearn-eai")

    peru_shape_data_path = f"gcs://{rslp_bucket}/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp"

    def __init__(
        self,
        ds_root: str,
        model_cfg_fname: str,
        gcs_tiff_filenames: list[str],
        workers: int = 1,
        days: int = 365,
        min_confidence: int = 2,
        min_area: float = 16,
        country_data_path: UPath | None = None,
        date_prefix: str = GCS_DATE_PREFIX,
        conf_prefix: str = GCS_CONF_PREFIX,
        prediction_utc_time: datetime = datetime.now(timezone.utc),
        disabled_layers: list[str] = [],
        max_number_of_events: int | None = None,
        group: str | None = None,
    ) -> None:
        """Create a new PredictPipelineConfig.

        Args:
            ds_root: dataset root to write the dataset.
            model_cfg_fname: the model configuration file name.
            gcs_tiff_filenames: the list of GCS TIFF filenames that we want to extract alerts from
            workers: number of workers.
            days: only consider forest loss events in this many past days.
            min_confidence: threshold on the GLAD alert confidence.
            min_area: minimum area in pixels of forest loss polygons. Pixels are
                roughly 10x10 m.
            country_data_path: the path to access country boundary data, so we can
                select the subset of forest loss events that are within Peru.
            date_prefix: the prefix for the date raster where the alert date is stored.
            conf_prefix: the prefix for the confidence raster where the alert confidence is stored.
            prediction_utc_time: the utc time to look back from for forest loss events.
            disabled_layers: layers to disable in the dataset.
            max_number_of_events: the maximum number of events to write to the dataset,
                if None, all extractedevents are written.
            group: the group to use for the dataset.
        """
        if country_data_path is None:
            logger.info(
                f"using default peru shape data path: {self.peru_shape_data_path}"
            )
            country_data_path = self.peru_shape_data_path
        self.path = UPath(ds_root)
        self.model_cfg_fname = model_cfg_fname
        self.gcs_tiff_filenames = gcs_tiff_filenames
        self.workers = workers
        self.days = days
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.country_data_path = UPath(country_data_path)
        self.date_prefix = date_prefix
        self.conf_prefix = conf_prefix
        self.prediction_utc_time = prediction_utc_time
        self.disabled_layers = disabled_layers
        self.max_number_of_events = max_number_of_events
        self.group = group

    def __str__(self) -> str:
        """Return a string representation of the config."""
        return (
            f"PredictPipelineConfig(path={self.path}, model_cfg_fname={self.model_cfg_fname}, "
            f"gcs_tiff_filenames={self.gcs_tiff_filenames}, workers={self.workers}, "
            f"days={self.days}, min_confidence={self.min_confidence}, "
            f"min_area={self.min_area}, country_data_path={self.country_data_path}, "
            f"date_prefix={self.date_prefix}, conf_prefix={self.conf_prefix}, "
            f"prediction_utc_time={self.prediction_utc_time}, "
            f"disabled_layers={self.disabled_layers}, "
            f"max_number_of_events={self.max_number_of_events}, "
            f"group={self.group})"
        )
