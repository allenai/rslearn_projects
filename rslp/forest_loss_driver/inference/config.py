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
        workers: int = 1,
        days: int = 365,
        min_confidence: int = 2,
        min_area: float = 16,
        country_data_path: UPath | None = None,
        date_prefix: str = GCS_DATE_PREFIX,
        conf_prefix: str = GCS_CONF_PREFIX,
        prediction_utc_time: datetime = datetime.now(timezone.utc),
        disabled_layers: list[str] = [],
        group: str | None = None,
    ) -> None:
        """Create a new PredictPipelineConfig.

        Args:
            ds_root: dataset root to write the dataset.
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
            group: the group to use for the dataset.
        """
        if country_data_path is None:
            logger.info(
                f"using default peru shape data path: {self.peru_shape_data_path}"
            )
            country_data_path = self.peru_shape_data_path
        self.path = UPath(ds_root)
        self.workers = workers
        self.days = days
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.country_data_path = UPath(country_data_path)
        self.date_prefix = date_prefix
        self.conf_prefix = conf_prefix
        self.prediction_utc_time = prediction_utc_time
        self.disabled_layers = disabled_layers
        self.group = group

    def __str__(self) -> str:
        """Return a string representation of the config."""
        return (
            f"PredictPipelineConfig(path={self.path}, workers={self.workers}, "
            f"days={self.days}, min_confidence={self.min_confidence}, "
            f"min_area={self.min_area}, country_data_path={self.country_data_path}, "
            f"date_prefix={self.date_prefix}, conf_prefix={self.conf_prefix}, "
            f"prediction_utc_time={self.prediction_utc_time}, "
            f"disabled_layers={self.disabled_layers}, group={self.group})"
        )
