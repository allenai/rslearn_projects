"""Prediction pipeline for Satlas models."""

import json
import shutil
from datetime import datetime
from enum import Enum

from rslearn.dataset import Window
from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from rslp.utils.rslearn import materialize_dataset, run_model_predict

DATASET_CONFIG_FNAME = "convert_satlas_webmercator_to_rslearn/{application}/config.json"
MODEL_CONFIG_FNAME = "convert_satlas_webmercator_to_rslearn/{application}/config.yaml"


class Application(Enum):
    """Specifies the various Satlas applications."""

    SOLAR_FARM = "solar_farm"
    WIND_TURBINE = "wind_turbine"
    MARINE_INFRA = "marine_infra"
    TREE_COVER = "tree_cover"


APP_IS_RASTER = {
    Application.SOLAR_FARM: True,
    Application.WIND_TURBINE: False,
    Application.MARINE_INFRA: False,
    Application.SOLAR_FARM: True,
}


def get_output_fname(
    application: Application, out_path: str, projection: Projection, bounds: PixelBounds
) -> UPath:
    """Get output filename to use for this application and task.

    Args:
        application: the application.
        out_path: the output path.
        projection: the projection of this task.
        bounds: the bounds of this task.

    Returns:
        the output filename.
    """
    if APP_IS_RASTER[application]:
        out_fname = (
            UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.tif"
        )
    else:
        out_fname = (
            UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.geojson"
        )
    return out_fname


def predict_pipeline(
    application: Application,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    scratch_path: str,
) -> None:
    """Compute outputs of a Satlas model on this tile.

    The tile is one part of a UTM zone.

    Args:
        application: the application for which to compute outputs.
        projection_json: JSON-encoded projection, normally a UTM zone with 10 m/pixel
            resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
        time_range: time range to apply model on.
        out_path: where to write the outputs. It will either be a GeoTIFF or GeoJSON,
            named based on the bounds.
        scratch_path: where to store the dataset.
    """
    dataset_config_fname = DATASET_CONFIG_FNAME.format(application=application.value)
    model_config_fname = MODEL_CONFIG_FNAME.format(application=application.value)

    # Check if the output was already computed.
    projection = Projection.deserialize(json.loads(projection_json))
    out_fname = get_output_fname(application, out_path, projection, bounds)
    if out_fname.exists():
        print(f"output file {out_fname} already exists")
        return

    # Initialize an rslearn dataset.
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)
    with open(dataset_config_fname) as f:
        ds_cfg = json.load(f)
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_cfg, f)

    # Create a window corresponding to the specified projection and bounds.
    group = "predict"
    window_path = ds_path / "windows" / group / "default"
    window = Window(
        path=window_path,
        group=group,
        name="default",
        projection=projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    # Populate the window.
    print("materialize dataset")
    materialize_dataset(ds_path, group=group)

    # Run the model.
    run_model_predict(model_config_fname, ds_path)

    if APP_IS_RASTER[application]:
        src_fname = window_path / "layers" / "output" / "output" / "geotiff.tif"
    else:
        src_fname = window_path / "layers" / "output" / "data.geojson"

    with src_fname.open("rb") as src:
        with out_fname.open("wb") as dst:
            shutil.copyfileobj(src, dst)
