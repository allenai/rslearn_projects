"""GeoZarr store for scaled OlmoEarth LCC predictions.

A store usually holds the predictions for one reference timestamp (recorded in the
store name), but random-tile runs write tiles with different timestamps into one
store. Each tile's reference timestamp is recorded in its ``_meta.json``. The store
uses the ``utm_zones`` layout: one group per UTM zone number named ``utm{NN}``, each in
its northern CRS (EPSG:326NN) with a continuous northing axis spanning both
hemispheres (see tiling.get_zone_grid). Each zone group holds two arrays with
dimensions ``(<array>_band, y, x)`` plus band-name and x/y coordinate arrays:

- ``change`` (uint8, CHANGE_BANDS): binary change probability (0-255), pre/post land
  cover class (0 = nodata, otherwise an index into LC_CLASS_NAMES), pre/post change
  category (0 = no prediction, 1 = none, otherwise an index into
  PRE/POST_CHANGE_CATEGORY_NAMES), and the pre/post change categories'
  probabilities (0-255).
- ``change_dates`` (uint16, CHANGE_DATES_BANDS): predicted change start and end in
  days since 1970-01-01 (0 = nodata).

Where both change categories are "none" and the binary change probability is below
128, the land cover classes, change scores, and dates are zeroed, and binary
change probabilities below 16 are stored as 0 (see
geozarr_window_arrays).

Stores have a detail mode (the ``detail`` root attribute). Compact stores hold only the
two arrays above. Full stores (meant for small runs) also hold one uint8 array per
classification head with the softmax probabilities scaled 0-255 and no zeroing (see
PROB_ARRAYS): ``src_probs`` and ``dst_probs`` over the land cover classes, and
``pre_change_probs`` and ``post_change_probs`` over "none" plus the change categories.
The nodata class of each head is not stored.

The store is created once by ``init_store`` (root, all zone groups, arrays, and
consolidated metadata). It builds the skeleton in memory and uploads each object
once, the root zarr.json last (see write_store_skeleton), since on GCS a cached
read of an overwritten metadata document can be stale. Prediction workers then only
write data regions via
``write_window``, never mutating metadata. The shard spatial size equals the window
size (PATCH_SIZE), so each window writes exactly one shard per array and concurrent
writes of disjoint windows are safe without locking.
"""

import posixpath
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import fsspec
import numpy as np
import zarr
from fsspec.implementations.local import LocalFileSystem
from rslearn.utils.geometry import PixelBounds, Projection
from zarr.codecs import ZstdCodec
from zarr.core.buffer import Buffer
from zarr.storage import MemoryStore

from rslp.log_utils import get_logger

from .output_bands import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
    LC_CLASS_NAMES,
    NONE_CATEGORY_IDX,
    NUM_LC_CLASSES,
    POST_CHANGE_BAND_OFFSET,
    POST_CHANGE_BANDS,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_BAND_OFFSET,
    PRE_CHANGE_BANDS,
    PRE_CHANGE_CATEGORY_NAMES,
    SRC_BAND_OFFSET,
    TS_POST_DAYS_BAND,
    TS_PRE_DAYS_BAND,
    UNIX_EPOCH_DAY_OFFSET,
)
from .tiling import (
    NORTH_EPSG_BASE,
    PATCH_SIZE,
    RESOLUTION,
    TILE_SIZE,
    check_bounds_aligned,
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
    "pre_change_score",
    "post_change_score",
]
CHANGE_DATES_ARRAY = "change_dates"
CHANGE_DATES_BANDS = ["change_start", "change_end"]
# The two arrays have different band counts, so each gets its own band dimension
# (xarray requires a dimension to have one size within a group).
CHANGE_BAND_DIM = "change_band"
CHANGE_DATES_BAND_DIM = "change_dates_band"

DETAIL_COMPACT = "compact"
DETAIL_FULL = "full"
DETAILS = [DETAIL_COMPACT, DETAIL_FULL]


@dataclass(frozen=True)
class ProbArray:
    """A full-mode probability array: one classification head's softmax outputs."""

    name: str
    # Class names of the stored bands (the head's classes without nodata).
    bands: list[str]
    # Index of the first stored band within output_bands.OUTPUT_BANDS.
    output_band_offset: int

    @property
    def band_dim(self) -> str:
        """The name of the array's band dimension."""
        return f"{self.name}_band"


PROB_ARRAYS = [
    ProbArray("src_probs", LC_CLASS_NAMES[1:NUM_LC_CLASSES], SRC_BAND_OFFSET + 1),
    ProbArray("dst_probs", LC_CLASS_NAMES[1:NUM_LC_CLASSES], DST_BAND_OFFSET + 1),
    ProbArray(
        "pre_change_probs", PRE_CHANGE_CATEGORY_NAMES[1:], PRE_CHANGE_BAND_OFFSET + 1
    ),
    ProbArray(
        "post_change_probs",
        POST_CHANGE_CATEGORY_NAMES[1:],
        POST_CHANGE_BAND_OFFSET + 1,
    ),
]

# Binary change probability (0-255) below which pixels with no change category get
# their land cover classes, post change score, and dates zeroed in the GeoZarr output.
GEOZARR_CHANGE_THRESHOLD = 128

# Binary change probability (0-255) below which the GeoZarr binary_change band itself
# is zeroed (in any pixel). Low probabilities carry little information but are most
# of the compressed output size; values at or above this keep full 0-255 resolution.
GEOZARR_BINARY_ZERO_THRESHOLD = 16

# Value used for nodata in both arrays (also the zarr fill value, so all-nodata
# chunks are not stored).
NODATA_VALUE = 0

SHARD_SIZE = PATCH_SIZE
CHUNK_SIZE = 512
# zstd level for every array. Decompression speed barely depends on the level, so a
# high level only costs CPU in the (background) writer threads.
ZSTD_LEVEL = 9

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

DETAIL_ATTR = "detail"

# Name of the metadata document of each group and array.
ZARR_JSON = "zarr.json"


def zone_group_name(zone_number: int) -> str:
    """Get the group name for a UTM zone number (e.g. 10 -> 'utm10')."""
    return f"utm{zone_number:02d}"


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


@dataclass(frozen=True)
class ArraySpec:
    """A (band, y, x) data array within a zone group."""

    name: str
    bands: list[str]
    band_dim: str
    dtype: str
    attrs: dict[str, Any]


CHANGE_ARRAY_SPECS = [
    ArraySpec(
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
    ArraySpec(
        CHANGE_DATES_ARRAY,
        CHANGE_DATES_BANDS,
        CHANGE_DATES_BAND_DIM,
        "uint16",
        {"bands": CHANGE_DATES_BANDS, "units": "days since 1970-01-01"},
    ),
]

PROB_ARRAY_SPECS = [
    ArraySpec(
        prob_array.name,
        prob_array.bands,
        prob_array.band_dim,
        "uint8",
        {"bands": prob_array.bands, "units": "probability scaled 0-255"},
    )
    for prob_array in PROB_ARRAYS
]


def get_array_specs(detail: str) -> list[ArraySpec]:
    """Get the data arrays of each zone group for a store detail mode."""
    if detail == DETAIL_COMPACT:
        return CHANGE_ARRAY_SPECS
    if detail == DETAIL_FULL:
        return CHANGE_ARRAY_SPECS + PROB_ARRAY_SPECS
    raise ValueError(f"unknown detail {detail}, expected one of {DETAILS}")


def create_zone_group(
    parent: zarr.Group,
    path: str,
    projection: Projection,
    origin_px: tuple[int, int],
    shape_px: tuple[int, int],
    array_specs: list[ArraySpec],
    chunk_size: int,
    shard_size: int,
    attrs: dict[str, Any] | None = None,
) -> zarr.Group:
    """Create one zone (or pyramid level) group with its arrays and coordinates.

    Args:
        parent: the group to create it in.
        path: the path of the new group within parent.
        projection: the group's projection (its resolution is the pixel size).
        origin_px: the (x, y) pixel coordinate of the arrays' top-left corner.
        shape_px: the (height, width) of the arrays in pixels.
        array_specs: the data arrays to create.
        chunk_size: inner chunk spatial size; must divide shard_size.
        shard_size: shard spatial size.
        attrs: additional group attributes.

    Returns:
        the new group.
    """
    height, width = shape_px
    group = parent.create_group(path)
    group.attrs.update(
        {
            "zarr_conventions": ZARR_CONVENTIONS,
            **build_zone_spatial_attrs(projection, origin_px, shape_px),
            **(attrs or {}),
        }
    )

    for spec in array_specs:
        group.create_array(
            spec.name,
            shape=(len(spec.bands), height, width),
            chunks=(len(spec.bands), chunk_size, chunk_size),
            shards=(len(spec.bands), shard_size, shard_size),
            dtype=spec.dtype,
            fill_value=NODATA_VALUE,
            compressors=[ZstdCodec(level=ZSTD_LEVEL)],
            dimension_names=(spec.band_dim, "y", "x"),
            attributes={**spec.attrs, "nodata": NODATA_VALUE},
        )
        band_coord = group.create_array(
            spec.band_dim,
            shape=(len(spec.bands),),
            dtype=str,
            dimension_names=(spec.band_dim,),
        )
        band_coord[:] = np.array(spec.bands)

    # Coordinate arrays (pixel centers) for xarray friendliness.
    origin_x, origin_y = origin_px
    x_coord = group.create_array(
        "x",
        shape=(width,),
        dtype="float64",
        chunks=(min(width, COORD_CHUNK_SIZE),),
        compressors=[ZstdCodec(level=ZSTD_LEVEL)],
        dimension_names=("x",),
    )
    x_coord[:] = (np.arange(width) + origin_x + 0.5) * projection.x_resolution
    y_coord = group.create_array(
        "y",
        shape=(height,),
        dtype="float64",
        chunks=(min(height, COORD_CHUNK_SIZE),),
        compressors=[ZstdCodec(level=ZSTD_LEVEL)],
        dimension_names=("y",),
    )
    y_coord[:] = (np.arange(height) + origin_y + 0.5) * projection.y_resolution
    return group


def write_store_skeleton(
    store_path: str, build: Callable[[zarr.Group], None], overwrite: bool = False
) -> None:
    """Create a store's skeleton, writing each object to the store only once.

    ``build`` creates the groups, arrays and attributes in an in-memory store, which
    is then consolidated and uploaded, with the root zarr.json last so that a store
    whose root exists is complete. Building in place would write placeholder group
    metadata, read it back (zarr checks each parent before creating a child), and
    only then overwrite it. GCS caches publicly readable objects, so a later read,
    e.g. by consolidate_metadata, could get the stale placeholder.

    With overwrite, objects at the same paths are written again, so readers of a
    public bucket may still get cached copies of the old store for a while.

    Args:
        store_path: the Zarr store path or URL.
        build: creates the hierarchy under the given (empty) root group.
        overwrite: whether to replace an existing store.

    Raises:
        FileExistsError: if the store exists and overwrite is False.
    """
    store_dict: dict[str, Buffer] = {}
    root = zarr.open_group(store=MemoryStore(store_dict), mode="w")
    build(root)
    zarr.consolidate_metadata(root.store)

    fs, path = fsspec.core.url_to_fs(store_path)
    path = path.rstrip("/")
    root_path = f"{path}/{ZARR_JSON}"
    if fs.exists(root_path) and not overwrite:
        raise FileExistsError(f"store {store_path} already exists")
    if overwrite and fs.exists(path):
        fs.rm(path, recursive=True)

    objects = {f"{path}/{key}": value.to_bytes() for key, value in store_dict.items()}
    root_data = objects.pop(root_path)
    if isinstance(fs, LocalFileSystem):
        # Object stores have no directories, but local files need their parents.
        for parent in sorted({posixpath.dirname(p) for p in objects}):
            fs.makedirs(parent, exist_ok=True)
    fs.pipe(objects)
    fs.pipe_file(root_path, root_data)


def init_store(
    store_path: str,
    model_config: str | None = None,
    detail: str = DETAIL_COMPACT,
    zone_numbers: list[int] | None = None,
    overwrite: bool = False,
) -> None:
    """Create the GeoZarr store skeleton (root, zone groups, arrays, metadata).

    This must run once before any prediction workers write to the store. Workers only
    write data regions, so all metadata is created here to avoid races. Each object
    is written once (see write_store_skeleton).

    Args:
        store_path: the Zarr store path or URL (e.g. gs://bucket/lcc_20250601.zarr).
        model_config: the model config used for prediction, recorded in the
            attributes.
        detail: the detail mode, one of DETAILS.
        zone_numbers: the UTM zone numbers to create groups for (default all 60).
        overwrite: whether to overwrite an existing store.
    """
    array_specs = get_array_specs(detail)
    zones = list(range(1, 61)) if zone_numbers is None else zone_numbers

    description = (
        "OlmoEarth Land Change predictions. binary_change, pre_change_score and "
        "post_change_score are probabilities scaled 0-255; pre_lc/post_lc index lc_class_names (0 = nodata); "
        "pre/post_change_category index pre/post_change_category_names (0 = no prediction, 1 = none). "
        "change_dates: days since 1970-01-01 (0 = nodata). Where both change "
        "categories are none and binary_change < no_change_threshold, "
        "pre_lc, post_lc, pre_change_score, post_change_score and change_dates "
        "are 0. "
        "binary_change < binary_change_zero_threshold is stored as 0."
    )
    if detail == DETAIL_FULL:
        description += (
            " The *_probs arrays hold each head's softmax probabilities scaled "
            "0-255 (without the nodata class and without zeroing)."
        )

    def build(root: zarr.Group) -> None:
        root.attrs.update(
            {
                "zarr_conventions": ZARR_CONVENTIONS,
                DETAIL_ATTR: detail,
                "model_config": model_config,
                "spatial_layout": "utm_zones",
                "gsd": float(RESOLUTION),
                "lc_class_names": LC_CLASS_NAMES,
                "pre_change_category_names": PRE_CHANGE_CATEGORY_NAMES,
                "post_change_category_names": POST_CHANGE_CATEGORY_NAMES,
                "no_change_threshold": GEOZARR_CHANGE_THRESHOLD,
                "binary_change_zero_threshold": GEOZARR_BINARY_ZERO_THRESHOLD,
                "description": description,
            }
        )

        for zone_number in zones:
            projection, origin_px, shape_px = get_zone_grid(
                zone_number, RESOLUTION, TILE_SIZE
            )
            create_zone_group(
                root,
                zone_group_name(zone_number),
                projection,
                origin_px,
                shape_px,
                array_specs,
                chunk_size=CHUNK_SIZE,
                shard_size=SHARD_SIZE,
            )
            logger.debug(
                "created zone group %s with shape %s",
                zone_group_name(zone_number),
                shape_px,
            )

    write_store_skeleton(store_path, build, overwrite=overwrite)
    logger.info("initialized store %s with %d zones", store_path, len(zones))


def read_store_detail(store_path: str) -> str | None:
    """Read the detail mode of an existing store.

    Args:
        store_path: the Zarr store path or URL.

    Returns:
        the detail mode (one of DETAILS), or None if the store does not exist.
    """
    try:
        root = zarr.open_group(store=store_path, mode="r")
    except FileNotFoundError:
        return None
    return str(root.attrs[DETAIL_ATTR])


def ensure_store(
    store_path: str,
    model_config: str | None = None,
    detail: str = DETAIL_COMPACT,
) -> None:
    """Create the store unless it already exists.

    Args:
        store_path: the Zarr store path or URL.
        model_config: the model config used for prediction.
        detail: the detail mode, one of DETAILS.

    Raises:
        ValueError: if the store exists but was created with a different detail
            mode.
    """
    existing_detail = read_store_detail(store_path)
    if existing_detail is None:
        init_store(store_path, model_config=model_config, detail=detail)
        return
    if existing_detail != detail:
        raise ValueError(
            f"store {store_path} has detail {existing_detail}, not {detail}"
        )
    logger.info("using existing store %s", store_path)


def _head_argmax(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the argmax class (skipping the nodata class 0) and its probability.

    Args:
        probs: (num classes, H, W) probabilities of one head, scaled 0-255.

    Returns:
        tuple of the uint8 (H, W) class indices (at least 1) and uint8 (H, W)
        probabilities of those classes.
    """
    classes = probs[1:].argmax(axis=0) + 1
    scores = np.take_along_axis(probs, classes[None], axis=0)[0]
    return classes.astype(np.uint8), np.clip(scores, 0, 255).astype(np.uint8)


def geozarr_window_arrays(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive the GeoZarr change and change_dates arrays from a full window array.

    See CHANGE_BANDS and CHANGE_DATES_BANDS for the layout. Each head is reduced to
    its argmax class (skipping nodata). Pixels with no prediction are all zero.
    Where both change categories are "none" and the binary change probability is
    below GEOZARR_CHANGE_THRESHOLD, the land cover classes, change scores, and
    change dates are zeroed too. binary_change values below
    GEOZARR_BINARY_ZERO_THRESHOLD are set to 0.

    Args:
        arr: (len(OUTPUT_BANDS), H, W) array with probabilities scaled 0-255.

    Returns:
        tuple of the (len(CHANGE_BANDS), H, W) uint8 change array and the (2, H, W) uint16
        change_dates array (days since 1970-01-01, 0 = nodata).
    """
    # Pixels with no prediction were left all-zero; use the binary head to detect
    # them.
    valid = arr[0 : BINARY_CHANGE_BAND + 1].max(axis=0) > 0
    binary_change = np.clip(arr[BINARY_CHANGE_BAND], 0, 255).astype(np.uint8)
    src, _ = _head_argmax(arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES])
    dst, _ = _head_argmax(arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES])
    pre, pre_score = _head_argmax(
        arr[PRE_CHANGE_BAND_OFFSET : PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)]
    )
    post, post_score = _head_argmax(
        arr[POST_CHANGE_BAND_OFFSET : POST_CHANGE_BAND_OFFSET + len(POST_CHANGE_BANDS)]
    )

    change = np.stack([binary_change, src, dst, pre, post, pre_score, post_score])
    change[:, ~valid] = 0

    dates = np.stack(
        [
            arr[TS_PRE_DAYS_BAND].astype(np.int64) + UNIX_EPOCH_DAY_OFFSET,
            arr[TS_POST_DAYS_BAND].astype(np.int64) + UNIX_EPOCH_DAY_OFFSET,
        ]
    )
    dates = np.clip(dates, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    dates[:, ~valid] = 0

    no_change = (
        (pre == NONE_CATEGORY_IDX)
        & (post == NONE_CATEGORY_IDX)
        & (binary_change < GEOZARR_CHANGE_THRESHOLD)
    )
    # Land cover classes (bands 1, 2) and the change scores (bands 5, 6).
    change[1:3, no_change] = 0
    change[5:7, no_change] = 0
    dates[:, no_change] = 0
    # Binary change (band 0); valid pixels stay distinguishable from nodata through
    # the change category bands.
    change[0, change[0] < GEOZARR_BINARY_ZERO_THRESHOLD] = 0
    return change, dates


def prob_window_arrays(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Derive the full-mode probability arrays from a full output window array.

    Args:
        arr: (len(OUTPUT_BANDS), H, W) array with probabilities scaled 0-255.

    Returns:
        map from PROB_ARRAYS name to its uint8 (num bands, H, W) array.
    """
    return {
        prob_array.name: np.clip(
            arr[
                prob_array.output_band_offset : prob_array.output_band_offset
                + len(prob_array.bands)
            ],
            0,
            255,
        ).astype(np.uint8)
        for prob_array in PROB_ARRAYS
    }


def check_zone_projection(projection: Projection) -> int:
    """Check that a projection is a northern UTM zone at RESOLUTION.

    Args:
        projection: the projection to check.

    Returns:
        the UTM zone number.
    """
    epsg_code = projection.crs.to_epsg()
    if epsg_code is None or not NORTH_EPSG_BASE < epsg_code <= NORTH_EPSG_BASE + 60:
        raise ValueError(
            f"expected a northern UTM projection (EPSG:326NN) but got {projection}"
        )
    if projection.x_resolution != RESOLUTION or projection.y_resolution != -RESOLUTION:
        raise ValueError(f"expected resolution {RESOLUTION} but got {projection}")
    return epsg_code - NORTH_EPSG_BASE


def get_group_offset(group: zarr.Group, bounds: PixelBounds) -> tuple[int, int]:
    """Get the (row, col) array offset of pixel bounds within a zone group.

    Args:
        group: the zone (or pyramid level) group.
        bounds: pixel bounds at the group's resolution.

    Returns:
        the (row, col) of the bounds' top-left corner within the group's arrays.
    """
    transform = cast(list[float], group.attrs["spatial:transform"])
    shape = cast(list[int], group.attrs["spatial:shape"])
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])
    row = bounds[1] - origin_y
    col = bounds[0] - origin_x
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    if row < 0 or col < 0 or row + height > shape[0] or col + width > shape[1]:
        raise ValueError(f"bounds {bounds} are outside the grid of group {group.path}")
    return row, col


def write_arrays(
    group: zarr.Group, bounds: PixelBounds, arrays: dict[str, np.ndarray]
) -> None:
    """Write (band, y, x) arrays into a group's arrays at the given bounds.

    Args:
        group: the zone (or pyramid level) group.
        bounds: pixel bounds at the group's resolution.
        arrays: map from array name to the data to write, which must match the
            zarr array's dtype and band count and the bounds' size.
    """
    row, col = get_group_offset(group, bounds)
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    for name, array in arrays.items():
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        expected_shape = (zarr_array.shape[0], height, width)
        if array.dtype != zarr_array.dtype or array.shape != expected_shape:
            raise ValueError(
                f"{name}: expected {zarr_array.dtype} array of shape "
                f"{expected_shape} but got {array.dtype} {array.shape}"
            )
        zarr_array[:, row : row + height, col : col + width] = array


def write_window(
    store_path: str,
    projection: Projection,
    bounds: PixelBounds,
    change: np.ndarray,
    change_dates: np.ndarray,
    prob_arrays: dict[str, np.ndarray] | None = None,
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
        prob_arrays: the full-mode probability arrays (see prob_window_arrays), for
            full stores.
    """
    zone_number = check_zone_projection(projection)
    check_bounds_aligned(bounds, SHARD_SIZE)
    group = zarr.open_group(
        store=store_path, path=zone_group_name(zone_number), mode="r+"
    )
    write_arrays(
        group,
        bounds,
        {CHANGE_ARRAY: change, CHANGE_DATES_ARRAY: change_dates, **(prob_arrays or {})},
    )


def read_window(
    store_path: str, projection: Projection, bounds: PixelBounds
) -> dict[str, np.ndarray]:
    """Read the data arrays of a region of a zone.

    Args:
        store_path: the Zarr store path or URL.
        projection: the northern UTM projection (EPSG:326NN) at RESOLUTION.
        bounds: the pixel bounds to read.

    Returns:
        map from array name to its (band, y, x) data: change and change_dates, plus
        the probability arrays in full stores.
    """
    zone_number = check_zone_projection(projection)
    group = zarr.open_group(
        store=store_path, path=zone_group_name(zone_number), mode="r"
    )
    row, col = get_group_offset(group, bounds)
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    names = [CHANGE_ARRAY, CHANGE_DATES_ARRAY] + [
        prob_array.name for prob_array in PROB_ARRAYS if prob_array.name in group
    ]
    arrays = {}
    for name in names:
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        arrays[name] = np.asarray(zarr_array[:, row : row + height, col : col + width])
    return arrays
