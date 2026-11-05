"""GeoZarr store for scaled OlmoEarth LCC predictions.

One store holds the predictions for one reference timestamp. It uses the
``utm_zones`` layout: one group per UTM zone number named ``utm{NN}``, each in its
northern CRS (EPSG:326NN) with a continuous northing axis spanning both hemispheres
(see tiling.get_zone_grid). Each zone group holds two arrays with dimensions
``(<array>_band, y, x)`` plus band-name and x/y coordinate arrays:

- ``change`` (uint8, CHANGE_BANDS): binary change probability (0-255), pre/post land
  cover class (0 = nodata, otherwise an index into LC_CLASS_NAMES), pre/post change
  category (0 = no prediction, 1 = none, otherwise an index into
  PRE/POST_CHANGE_CATEGORY_NAMES), and the post change category's probability
  (0-255).
- ``change_dates`` (uint16, CHANGE_DATES_BANDS): predicted change start and end in
  days since 1970-01-01 (0 = nodata).

Where both change categories are "none" and the binary change probability is below
128, the land cover classes, post change score, and dates are zeroed, and binary
change probabilities below 16 are stored as 0 (see
postprocess.geozarr_window_arrays).

The store is created once by ``init_store`` (root, all zone groups, arrays, and
consolidated metadata). Prediction workers then only write data regions via
``write_window``, never mutating metadata. The shard spatial size equals the window
size (PATCH_SIZE), so each window writes exactly one shard per array and concurrent
writes of disjoint windows are safe without locking.
"""

from datetime import datetime
from typing import Any, cast

import numpy as np
import zarr
from rslearn.utils.geometry import PixelBounds, Projection
from zarr.codecs import BloscCodec, BloscShuffle, ZstdCodec

from rslp.log_utils import get_logger

from .postprocess import (
    GEOZARR_BINARY_ZERO_THRESHOLD,
    GEOZARR_CHANGE_THRESHOLD,
    LC_CLASS_NAMES,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from .tiling import (
    NORTH_EPSG_BASE,
    PATCH_SIZE,
    RESOLUTION,
    TILE_SIZE,
    get_zone_grid,
)

logger = get_logger(__name__)

CHANGE_ARRAY = "change"
CHANGE_BANDS = [
    "binary_change",
    "pre_lc",
    "post_lc",
    "pre_change_category",
    "post_change_category",
    "post_change_score",
]
CHANGE_DATES_ARRAY = "change_dates"
CHANGE_DATES_BANDS = ["change_start", "change_end"]
# The two arrays have different band counts, so each gets its own band dimension
# (xarray requires a dimension to have one size within a group).
CHANGE_BAND_DIM = "change_band"
CHANGE_DATES_BAND_DIM = "change_dates_band"

# Value used for nodata in both arrays (also the zarr fill value, so all-nodata
# chunks are not stored).
NODATA_VALUE = 0

SHARD_SIZE = PATCH_SIZE
DEFAULT_CHUNK_SIZE = 512
DEFAULT_COMPRESSOR = "zstd"
DEFAULT_COMPRESSION_LEVEL = 9
COMPRESSORS = ["zstd", "blosc_zstd", "blosc_lz4", "none"]

# Chunk size (in elements) for the 1-D x/y coordinate arrays.
COORD_CHUNK_SIZE = 65536

SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial",
    "description": "Spatial coordinate information",
}
PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data",
}
ZARR_CONVENTIONS = [SPATIAL_CONVENTION, PROJ_CONVENTION]

REFERENCE_TIMESTAMP_ATTR = "reference_timestamp"


def zone_group_name(zone_number: int) -> str:
    """Get the group name for a UTM zone number (e.g. 10 -> 'utm10')."""
    return f"utm{zone_number:02d}"


def get_compressors(compressor: str, level: int) -> list[Any]:
    """Get the zarr compressors list for a compressor name.

    Args:
        compressor: one of COMPRESSORS.
        level: the compression level.

    Returns:
        the list of codecs to pass as compressors.
    """
    if compressor == "zstd":
        return [ZstdCodec(level=level)]
    if compressor == "blosc_zstd":
        return [BloscCodec(cname="zstd", clevel=level, shuffle=BloscShuffle.shuffle)]
    if compressor == "blosc_lz4":
        return [BloscCodec(cname="lz4", clevel=level, shuffle=BloscShuffle.shuffle)]
    if compressor == "none":
        return []
    raise ValueError(f"unknown compressor {compressor}, expected one of {COMPRESSORS}")


def build_zone_spatial_attrs(
    projection: Projection,
    origin_px: tuple[int, int],
    shape_px: tuple[int, int],
) -> dict[str, Any]:
    """Build the proj:/spatial: attributes for one zone group.

    Args:
        projection: the northern UTM projection of the zone.
        origin_px: the (x, y) pixel coordinate of the array's top-left corner.
        shape_px: the (height, width) of the array in pixels.

    Returns:
        a dict of proj: and spatial: attributes.
    """
    origin_x, origin_y = origin_px
    height, width = shape_px
    x_res = projection.x_resolution
    y_res = projection.y_resolution
    # Affine mapping array (col, row) -> CRS (easting, northing). The pixel grid has
    # its origin at CRS (0, 0), and y_res is negative so northing decreases with row.
    corner_x = origin_x * x_res
    corner_y = origin_y * y_res
    far_x = (origin_x + width) * x_res
    far_y = (origin_y + height) * y_res
    return {
        "proj:code": f"EPSG:{projection.crs.to_epsg()}",
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": [x_res, 0.0, corner_x, 0.0, y_res, corner_y],
        "spatial:shape": [height, width],
        "spatial:bbox": [
            min(corner_x, far_x),
            min(corner_y, far_y),
            max(corner_x, far_x),
            max(corner_y, far_y),
        ],
        "spatial:registration": "pixel",
    }


def init_store(
    store_path: str,
    timestamp: datetime,
    model_config: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compressor: str = DEFAULT_COMPRESSOR,
    level: int = DEFAULT_COMPRESSION_LEVEL,
    zone_numbers: list[int] | None = None,
    overwrite: bool = False,
) -> None:
    """Create the GeoZarr store skeleton (root, zone groups, arrays, metadata).

    This must run once before any prediction workers write to the store. Workers only
    write data regions, so all metadata is created here to avoid races.

    Args:
        store_path: the Zarr store path or URL (e.g. gs://bucket/lcc_20250601.zarr).
        timestamp: the reference timestamp of the predictions in this store.
        model_config: the model config used for prediction, recorded in the
            attributes.
        chunk_size: inner chunk spatial size; must divide SHARD_SIZE.
        compressor: the compressor to use, one of COMPRESSORS.
        level: the compression level.
        zone_numbers: the UTM zone numbers to create groups for (default all 60).
        overwrite: whether to overwrite an existing store.
    """
    if SHARD_SIZE % chunk_size != 0:
        raise ValueError(
            f"shard size {SHARD_SIZE} must be a multiple of chunk_size {chunk_size}"
        )
    if zone_numbers is None:
        zone_numbers = list(range(1, 61))
    compressors = get_compressors(compressor, level)

    root = zarr.open_group(store=store_path, mode="w" if overwrite else "w-")
    root.attrs.update(
        {
            "zarr_conventions": ZARR_CONVENTIONS,
            REFERENCE_TIMESTAMP_ATTR: timestamp.isoformat(),
            "model_config": model_config,
            "spatial_layout": "utm_zones",
            "gsd": float(RESOLUTION),
            "lc_class_names": LC_CLASS_NAMES,
            "pre_change_category_names": PRE_CHANGE_CATEGORY_NAMES,
            "post_change_category_names": POST_CHANGE_CATEGORY_NAMES,
            "no_change_threshold": GEOZARR_CHANGE_THRESHOLD,
            "binary_change_zero_threshold": GEOZARR_BINARY_ZERO_THRESHOLD,
            "description": (
                "OlmoEarth LCC land cover change predictions. change: binary_change "
                "and post_change_score are probabilities scaled 0-255; pre_lc/post_lc "
                "index lc_class_names (0 = nodata); pre/post_change_category index "
                "pre/post_change_category_names (0 = no prediction, 1 = none). "
                "change_dates: days since 1970-01-01 (0 = nodata). Where both change "
                "categories are none and binary_change < no_change_threshold, "
                "pre_lc, post_lc, post_change_score and change_dates are 0. "
                "binary_change < binary_change_zero_threshold is stored as 0."
            ),
        }
    )

    for zone_number in zone_numbers:
        projection, origin_px, shape_px = get_zone_grid(
            zone_number, RESOLUTION, TILE_SIZE
        )
        height, width = shape_px
        zone_group = root.create_group(zone_group_name(zone_number))
        zone_group.attrs.update(
            {
                "zarr_conventions": ZARR_CONVENTIONS,
                **build_zone_spatial_attrs(projection, origin_px, shape_px),
            }
        )

        array_specs: list[tuple[str, list[str], str, str, dict[str, Any]]] = [
            (
                CHANGE_ARRAY,
                CHANGE_BANDS,
                CHANGE_BAND_DIM,
                "uint8",
                {
                    "bands": CHANGE_BANDS,
                    "class_names": {
                        "pre_lc": LC_CLASS_NAMES,
                        "post_lc": LC_CLASS_NAMES,
                        "pre_change_category": PRE_CHANGE_CATEGORY_NAMES,
                        "post_change_category": POST_CHANGE_CATEGORY_NAMES,
                    },
                },
            ),
            (
                CHANGE_DATES_ARRAY,
                CHANGE_DATES_BANDS,
                CHANGE_DATES_BAND_DIM,
                "uint16",
                {"bands": CHANGE_DATES_BANDS, "units": "days since 1970-01-01"},
            ),
        ]
        for name, bands, band_dim, dtype, attrs in array_specs:
            zone_group.create_array(
                name,
                shape=(len(bands), height, width),
                chunks=(len(bands), chunk_size, chunk_size),
                shards=(len(bands), SHARD_SIZE, SHARD_SIZE),
                dtype=dtype,
                fill_value=NODATA_VALUE,
                compressors=compressors,
                dimension_names=(band_dim, "y", "x"),
                attributes={**attrs, "nodata": NODATA_VALUE},
            )
            band_coord = zone_group.create_array(
                band_dim, shape=(len(bands),), dtype=str, dimension_names=(band_dim,)
            )
            band_coord[:] = np.array(bands)

        # Coordinate arrays (pixel centers) for xarray friendliness.
        origin_x, origin_y = origin_px
        x_coord = zone_group.create_array(
            "x",
            shape=(width,),
            dtype="float64",
            chunks=(min(width, COORD_CHUNK_SIZE),),
            compressors=[ZstdCodec(level=DEFAULT_COMPRESSION_LEVEL)],
            dimension_names=("x",),
        )
        x_coord[:] = (np.arange(width) + origin_x + 0.5) * projection.x_resolution
        y_coord = zone_group.create_array(
            "y",
            shape=(height,),
            dtype="float64",
            chunks=(min(height, COORD_CHUNK_SIZE),),
            compressors=[ZstdCodec(level=DEFAULT_COMPRESSION_LEVEL)],
            dimension_names=("y",),
        )
        y_coord[:] = (np.arange(height) + origin_y + 0.5) * projection.y_resolution
        logger.debug(
            "created zone group %s with shape %s",
            zone_group_name(zone_number),
            shape_px,
        )

    zarr.consolidate_metadata(root.store)
    logger.info("initialized store %s with %d zones", store_path, len(zone_numbers))


def get_store_timestamp(store_path: str) -> datetime | None:
    """Get the reference timestamp of an existing store.

    Args:
        store_path: the Zarr store path or URL.

    Returns:
        the store's reference timestamp, or None if the store does not exist.
    """
    try:
        root = zarr.open_group(store=store_path, mode="r")
    except FileNotFoundError:
        return None
    return datetime.fromisoformat(str(root.attrs[REFERENCE_TIMESTAMP_ATTR]))


def ensure_store(
    store_path: str, timestamp: datetime, model_config: str | None = None
) -> None:
    """Create the store with default settings unless it already exists.

    Args:
        store_path: the Zarr store path or URL.
        timestamp: the reference timestamp of the predictions.
        model_config: the model config used for prediction.

    Raises:
        ValueError: if the store exists but was created for a different timestamp.
    """
    store_timestamp = get_store_timestamp(store_path)
    if store_timestamp is None:
        init_store(store_path, timestamp, model_config=model_config)
        return
    if store_timestamp != timestamp:
        raise ValueError(
            f"store {store_path} is for reference timestamp {store_timestamp}, "
            f"not {timestamp}"
        )
    logger.info("using existing store %s", store_path)


def write_window(
    store_path: str,
    projection: Projection,
    bounds: PixelBounds,
    change: np.ndarray,
    change_dates: np.ndarray,
) -> None:
    """Write one window's outputs into its zone's arrays.

    The window must be aligned to the shard grid (bounds are multiples of
    SHARD_SIZE) so this writes whole shards, which is safe under concurrent writes
    of disjoint windows.

    Args:
        store_path: the Zarr store path or URL.
        projection: the window projection, which must be a northern UTM projection
            (EPSG:326NN) at RESOLUTION.
        bounds: the window's pixel bounds.
        change: the uint8 (len(CHANGE_BANDS), H, W) array.
        change_dates: the uint16 (len(CHANGE_DATES_BANDS), H, W) array.
    """
    epsg_code = projection.crs.to_epsg()
    if epsg_code is None or not NORTH_EPSG_BASE < epsg_code <= NORTH_EPSG_BASE + 60:
        raise ValueError(
            f"expected a northern UTM projection (EPSG:326NN) but got {projection}"
        )
    if projection.x_resolution != RESOLUTION or projection.y_resolution != -RESOLUTION:
        raise ValueError(f"expected resolution {RESOLUTION} but got {projection}")
    if any(value % SHARD_SIZE != 0 for value in bounds):
        raise ValueError(f"bounds {bounds} must be multiples of {SHARD_SIZE}")

    group = zarr.open_group(
        store=store_path,
        path=zone_group_name(epsg_code - NORTH_EPSG_BASE),
        mode="r+",
    )
    transform = cast(list[float], group.attrs["spatial:transform"])
    shape = cast(list[int], group.attrs["spatial:shape"])
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])
    row = bounds[1] - origin_y
    col = bounds[0] - origin_x
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    if row < 0 or col < 0 or row + height > shape[0] or col + width > shape[1]:
        raise ValueError(f"bounds {bounds} are outside the zone grid of {projection}")

    for name, array, dtype in [
        (CHANGE_ARRAY, change, np.uint8),
        (CHANGE_DATES_ARRAY, change_dates, np.uint16),
    ]:
        if array.dtype != dtype or array.shape[1:] != (height, width):
            raise ValueError(
                f"{name}: expected {dtype} array of size {height}x{width} but got "
                f"{array.dtype} {array.shape}"
            )
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        zarr_array[:, row : row + height, col : col + width] = array
