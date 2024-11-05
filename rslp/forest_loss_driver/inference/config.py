"""Config for the forest loss driver inference/predict pipeline."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from upath import UPath


@dataclass
class PredictPipelineConfig:
    """Prediction pipeline config for forest loss driver classification.

    Required parameters:
        ds_root: Dataset root to write the dataset
        model_cfg_fname: Model configuration file name
        gcs_tiff_filenames: List of GCS TIFF filenames to extract alerts from
    """

    # Required fields (no default values)
    ds_root: str
    model_cfg_fname: str
    gcs_tiff_filenames: list[str]

    # Optional fields with defaults
    workers: int = 1
    days: int = 365
    min_confidence: int = 2
    min_area: float = 16.0
    country_data_path: UPath | None = None
    date_prefix: str = "gs://earthenginepartners-hansen/S2alert/alertDate/"
    conf_prefix: str = "gs://earthenginepartners-hansen/S2alert/alert/"
    prediction_utc_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    disabled_layers: list[str] = field(default_factory=list)
    max_number_of_events: int | None = None
    group: str | None = None

    # Constants that shouldn't be overridden
    rslp_bucket: str = field(
        default=os.environ.get("RSLP_BUCKET", "rslearn-eai"),
        init=False,  # This makes it not appear as a constructor parameter
    )

    @property
    def peru_shape_data_path(self) -> str:
        """The path to the Peru shape data."""
        return f"gcs://{self.rslp_bucket}/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp"

    @property
    def path(self) -> UPath:
        """The path to the dataset."""
        return UPath(self.ds_root)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.min_confidence < 0:
            raise ValueError("min_confidence must be non-negative")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.min_area <= 0:
            raise ValueError("min_area must be positive")
        if self.country_data_path is None:
            self.country_data_path = UPath(self.peru_shape_data_path)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PredictPipelineConfig":
        """Create a new config from a YAML file.

        Raises:
            ValueError: If required fields are missing
            yaml.YAMLError: If YAML file is invalid
        """
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Check for required fields
        required_fields = {"ds_root", "model_cfg_fname", "gcs_tiff_filenames"}
        missing_fields = required_fields - set(config_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {missing_fields}")

        # Convert string datetime to datetime object if present
        if "prediction_utc_time" in config_dict:
            if isinstance(config_dict["prediction_utc_time"], str):
                config_dict["prediction_utc_time"] = datetime.fromisoformat(
                    config_dict["prediction_utc_time"].replace("Z", "+00:00")
                )
        if "model_cfg_fname" in config_dict:
            if not config_dict["model_cfg_fname"].startswith(
                "gs://"
            ) and not config_dict["model_cfg_fname"].startswith("/"):
                config_dict["model_cfg_fname"] = str(
                    Path(__file__).resolve().parents[3] / config_dict["model_cfg_fname"]
                )

        return cls(**config_dict)

    def __str__(self) -> str:
        """Return a string representation of the config."""
        return (
            f"PredictPipelineConfig(\n"
            f"  Required:\n"
            f"    ds_root={self.ds_root}\n"
            f"    model_cfg_fname={self.model_cfg_fname}\n"
            f"    gcs_tiff_filenames={self.gcs_tiff_filenames}\n"
            f"  Optional:\n"
            f"    workers={self.workers}\n"
            f"    days={self.days}\n"
            f"    min_confidence={self.min_confidence}\n"
            f"    min_area={self.min_area}\n"
            f"    country_data_path={self.country_data_path}\n"
            f"    prediction_utc_time={self.prediction_utc_time}\n"
            f"    disabled_layers={self.disabled_layers}\n"
            f"    max_number_of_events={self.max_number_of_events}\n"
            f"    group={self.group}\n"
            f")"
        )
