"""Convert Fields of the World dataset to rslearn format."""

import hashlib
import json
import multiprocessing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.fsspec import open_rasterio_upath_reader
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    get_raster_projection_and_bounds,
)
from upath import UPath

SRC_DIR = "/weka/dfive-default/rslearn-eai/artifacts/fields_of_the_world/"
# Path to write dataset that uses projection and images from original dataset.
# They are WGS84 at roughly 6 m/pixel at the equator (but resolution is in degrees).
ORIG_DST_DIR = (
    "/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_orig/"
)
# Path to write dataset that uses UTM projection.
UTM_DST_DIR = (
    "/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/"
)
# Order of bands in the s2_a/s2_b GeoTIFFs.
# The source dataset only has these bands.
S2_BANDS = ["B04", "B03", "B02", "B08"]
# Duration for the window time ranges.
DURATION = timedelta(days=30 * 8)


@dataclass
class Example:
    """One example in the source dataset.

    This contains all the information needed to convert the example.
    """

    # Country name for this example.
    country_name: str
    # 3-class semantic GeoTIFF path.
    semantic_fname: UPath
    # window_a GeoTIFF path.
    # The dataset specifies two windows, one at the first half of the agricultural
    # season and one at the second half. One image is provided for each window.
    s2_a_fname: UPath
    # window_b GeoTIFF path.
    s2_b_fname: UPath
    # Time range to set.
    time_range: tuple[datetime, datetime]


def season_to_time_range(season: dict) -> tuple[datetime, datetime]:
    """Get the window time range from the season dict.

    We compute it by adding DURATION days to the start of the first (a) time window.
    """
    ts = datetime.strptime(season["window_a"]["start"], "%Y-%m-%d").replace(tzinfo=UTC)
    return (ts, ts + DURATION)


def create_window(
    orig_dataset: Dataset, utm_dataset: Dataset, example: Example
) -> None:
    """Create an rslearn window from the given example."""
    suffix = example.semantic_fname.name.split(".tif")[0]
    window_name = f"{example.country_name}_{suffix}"

    # Use hash to determine val vs test.
    hash_hex_char = hashlib.sha256(window_name.encode()).hexdigest()[0]
    is_val = hash_hex_char in ["0", "1", "2", "3"]
    is_test = hash_hex_char in [
        "4",
        "5",
        "6",
        "7",
    ]
    if is_val:
        split = "val"
    elif is_test:
        split = "test"
    else:
        split = "train"

    # Get initial projection and bounds from the semantic GeoTIFF.
    with open_rasterio_upath_reader(example.semantic_fname) as raster:
        projection, bounds = get_raster_projection_and_bounds(raster)

    # Change it to be UTM.
    initial_geom = STGeometry(projection, shapely.box(*bounds), None)
    wgs84_geom = initial_geom.to_projection(WGS84_PROJECTION)
    utm_proj = get_utm_ups_projection(
        wgs84_geom.shp.centroid.x, wgs84_geom.shp.centroid.y, 10, -10
    )
    utm_geom = initial_geom.to_projection(utm_proj)
    utm_bounds = (
        int(utm_geom.shp.bounds[0]),
        int(utm_geom.shp.bounds[1]),
        int(utm_geom.shp.bounds[2]),
        int(utm_geom.shp.bounds[3]),
    )

    # Create the original window (keep source projection).
    orig_window = Window(
        storage=orig_dataset.storage,
        group=example.country_name,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=example.time_range,
        options={"split": split},
    )
    orig_window.save()

    # Copy in the rasters.
    raster_format = GeotiffRasterFormat()
    semantic_array = raster_format.decode_raster(
        example.semantic_fname.parent,
        projection,
        bounds,
        fname=example.semantic_fname.name,
        nodata_val=3,
    )
    s2_a_array = raster_format.decode_raster(
        example.s2_a_fname.parent, projection, bounds, fname=example.s2_a_fname.name
    )
    s2_b_array = raster_format.decode_raster(
        example.s2_b_fname.parent, projection, bounds, fname=example.s2_b_fname.name
    )
    raster_format.encode_raster(
        orig_window.get_raster_dir("label", ["label"]),
        projection,
        bounds,
        semantic_array,
        nodata_val=3,
    )
    raster_format.encode_raster(
        orig_window.get_raster_dir("s2_a", S2_BANDS), projection, bounds, s2_a_array
    )
    raster_format.encode_raster(
        orig_window.get_raster_dir("s2_b", S2_BANDS), projection, bounds, s2_b_array
    )
    orig_window.mark_layer_completed("label")
    orig_window.mark_layer_completed("s2_a")
    orig_window.mark_layer_completed("s2_b")

    # Create the UTM window and copy the label only.
    # The images should be obtained with prepare/ingest/materialize.
    utm_window = Window(
        storage=utm_dataset.storage,
        group=example.country_name,
        name=window_name,
        projection=utm_proj,
        bounds=utm_bounds,
        time_range=example.time_range,
        options={"split": split},
    )
    utm_window.save()
    semantic_array = raster_format.decode_raster(
        example.semantic_fname.parent,
        utm_proj,
        utm_bounds,
        fname=example.semantic_fname.name,
        nodata_val=3,
    )
    raster_format.encode_raster(
        utm_window.get_raster_dir("label", ["label"]),
        utm_proj,
        utm_bounds,
        semantic_array,
        nodata_val=3,
    )
    utm_window.mark_layer_completed("label")


if __name__ == "__main__":
    # Enumerate all of the examples in the source dataset.
    examples: list[Example] = []
    country_configs = list(UPath(SRC_DIR).glob("*/data_config_*.json"))
    for country_config_fname in tqdm.tqdm(
        country_configs, desc="Enumerating examples across countries"
    ):
        country_dir = country_config_fname.parent
        country_name = country_dir.name

        # Extract time range from the data config.
        with country_config_fname.open() as f:
            data_config = json.load(f)
            seasons = data_config["seasons"]
            grids = data_config["grids"]
        # seasons can either be a list with one dict per grid, or just a single dict.
        if isinstance(seasons, list):
            if len(seasons) != len(grids):
                raise ValueError(
                    f"got {len(seasons)} seasons but {len(grids)} grids, expected the lists to correspond"
                )
            season_by_grid_id = {
                grid["id"]: season for grid, season in zip(grids, seasons)
            }
        else:
            season_by_grid_id = {grid["id"]: seasons for grid in grids}

        # Now get the examples in this country.
        for semantic_fname in (
            country_dir / "label_masks" / "semantic_3class"
        ).iterdir():
            s2_a_fname = country_dir / "s2_images" / "window_a" / semantic_fname.name
            s2_b_fname = country_dir / "s2_images" / "window_b" / semantic_fname.name

            # When there is only one grid, sometimes the filenames don't start with the grid.
            # In any case we use the grid ID to get the season, which we convert to time range.
            if len(grids) == 1:
                grid_id = grids[0]["id"]
            else:
                grid_id = semantic_fname.name.split("_")[0]
            time_range = season_to_time_range(season_by_grid_id[grid_id])

            examples.append(
                Example(
                    country_name=country_name,
                    semantic_fname=semantic_fname,
                    s2_a_fname=s2_a_fname,
                    s2_b_fname=s2_b_fname,
                    time_range=time_range,
                )
            )

    # Run the conversion in parallel.
    orig_dataset = Dataset(UPath(ORIG_DST_DIR))
    utm_dataset = Dataset(UPath(UTM_DST_DIR))
    p = multiprocessing.Pool(128)
    outputs = star_imap_unordered(
        p,
        create_window,
        [
            dict(
                orig_dataset=orig_dataset,
                utm_dataset=utm_dataset,
                example=example,
            )
            for example in examples
        ],
    )
    for _ in tqdm.tqdm(outputs, total=len(examples)):
        pass
    p.close()
