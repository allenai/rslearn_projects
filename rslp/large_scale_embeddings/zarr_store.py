"""GeoZarr store for global quantized OlmoEarth embeddings.

Writes embeddings to a Zarr v3 store following the geoemb embeddings-zarr-convention
(https://github.com/geo-embeddings/embeddings-zarr-convention). The store uses the
``utm_zones`` spatial layout: one group per UTM zone number named ``utm{NN}``, each in
its northern CRS (EPSG:326NN) with a continuous northing axis spanning both
hemispheres (see tiling.get_zone_grid). Each zone group holds one ``embeddings`` array
with dimensions ``(time, band, y, x)`` where ``band`` is the 128-dim embedding vector
and ``time`` is the annual reference years.

The store is created once by ``init_store`` (root + all zone groups, arrays, and
consolidated metadata). Prediction workers then only write disjoint data regions via
``write_window_region``, never mutating metadata, so concurrent writes are safe. To
keep concurrent writes on disjoint shards, the shard spatial size must equal the
window (PATCH_SIZE) size so each window owns exactly one shard.

Quantization is the signed-power (AlphaEarth-style) scheme from model.py; it is not
expressible with the convention's scalar/array scale objects, so it is described with
a custom ``method`` plus a ``link`` to the formula (see the project README).
"""

import numpy as np
import zarr
from rslearn.utils.geometry import PixelBounds, Projection
from zarr.codecs import ZstdCodec

from rslp.log_utils import get_logger

from .model import NODATA_VALUE
from .tiling import get_zone_grid

logger = get_logger(__name__)

# Name of the embedding array within each zone group.
EMBEDDINGS_ARRAY = "embeddings"

# Derived false-color RGB array, written alongside the embeddings for tile serving.
# It shares the shard grid so one worker owns both of a window's objects and neither
# write needs locking. 0 is reserved as nodata, so valid pixels occupy 1-255.
PCA_ARRAY = "pca_rgb"
PCA_DIMENSIONS = ("time", "rgb", "y", "x")
PCA_NODATA_VALUE = 0
PCA_BANDS = 3

# Zarr v3 dimension order for the embedding array.
EMBEDDING_DIMENSIONS = ("time", "band", "y", "x")

# Default chunk (inner) and shard (outer storage unit) spatial sizes, and zstd level.
# The shard size must equal the window size (PATCH_SIZE) so each prediction window
# writes exactly one shard, keeping concurrent region writes on disjoint objects.
# chunk=256 was chosen from a small-AOI read benchmark on real embeddings: it roughly
# halves the bytes read per interactive AOI versus 512, at ~14% more storage.
DEFAULT_CHUNK_SIZE = 256
DEFAULT_SHARD_SIZE = 2048
DEFAULT_ZSTD_LEVEL = 1
# Chunk size (in elements) for the 1-D x/y coordinate arrays. They are linear ramps
# so they compress to almost nothing under zstd.
COORD_CHUNK_SIZE = 65536

# Registered zarr_conventions entries (see the geoemb spec and its tessera example).
GEOEMB_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/geo-embeddings/embeddings-zarr-convention/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/geo-embeddings/embeddings-zarr-convention/blob/v1/README.md",
    "uuid": "61c12cc5-0e28-4056-999a-480cf3fb7e4c",
    "name": "geoemb:",
    "description": "Geoembeddings convention for geospatial embedding arrays with model provenance",
}
SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate information",
}
PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}
ZARR_CONVENTIONS = [GEOEMB_CONVENTION, SPATIAL_CONVENTION, PROJ_CONVENTION]

# Default URL documenting the signed-power quantization/dequantization formula.
DEFAULT_QUANTIZATION_LINK = (
    "https://github.com/allenai/rslearn_projects/blob/master/"
    "rslp/large_scale_embeddings/README.md"
)


def zone_group_name(zone_number: int) -> str:
    """Get the group name for a UTM zone number (e.g. 10 -> 'utm10')."""
    return f"utm{zone_number:02d}"


def get_store_years(store_path: str, storage_options: dict | None = None) -> list[int]:
    """Read the store's time axis (reference years) from the first zone group.

    Args:
        store_path: the Zarr store path or URL.
        storage_options: fsspec storage options for remote stores.

    Returns:
        the list of years defining the store's time axis.
    """
    root = zarr.open_group(store=store_path, mode="r", storage_options=storage_options)
    for name in sorted(root.keys()):
        if name.startswith("utm"):
            return [int(value) for value in root[name]["time"][:]]
    raise ValueError(f"no utm zone groups found in store {store_path}")


def _quantization_attrs(link: str) -> dict:
    """Build the geoemb:quantization object for the signed-power scheme."""
    return {
        "method": "signed_power",
        "original_dtype": "float32",
        "quantized_dtype": "int8",
        "link": link,
    }


def build_geoemb_attrs(
    dimensions: int,
    model_url: str,
    source_data: list[str],
    gsd: float,
    build_version: str,
    quantization_link: str = DEFAULT_QUANTIZATION_LINK,
) -> dict:
    """Build the geoemb convention attributes (shared by the root and zone groups).

    Args:
        dimensions: the embedding vector dimensionality.
        model_url: URL reference to the encoder model.
        source_data: URLs of the source datasets.
        gsd: ground sample distance in meters.
        build_version: version of the software that built the store.
        quantization_link: URL documenting the dequantization formula.

    Returns:
        a dict of geoemb: attributes.
    """
    return {
        "geoemb:type": "pixel",
        "geoemb:dimensions": dimensions,
        "geoemb:model": model_url,
        "geoemb:source_data": list(source_data),
        "geoemb:data_type": "int8",
        "geoemb:gsd": gsd,
        "geoemb:spatial_layout": "utm_zones",
        "geoemb:quantization": _quantization_attrs(quantization_link),
        "geoemb:build_version": build_version,
    }


def build_zone_spatial_attrs(
    projection: Projection,
    origin_px: tuple[int, int],
    shape_px: tuple[int, int],
) -> dict:
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
    # Affine mapping array (col, row) -> CRS (easting, northing): the pixel grid has
    # its origin at CRS (0, 0), so the array corner is at (origin_x * x_res,
    # origin_y * y_res). y_res is negative so northing decreases with row.
    corner_x = origin_x * x_res
    corner_y = origin_y * y_res
    transform = [x_res, 0.0, corner_x, 0.0, y_res, corner_y]
    far_x = (origin_x + width) * x_res
    far_y = (origin_y + height) * y_res
    bbox = [
        min(corner_x, far_x),
        min(corner_y, far_y),
        max(corner_x, far_x),
        max(corner_y, far_y),
    ]
    return {
        "proj:code": f"EPSG:{projection.crs.to_epsg()}",
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": transform,
        "spatial:shape": [height, width],
        "spatial:bbox": bbox,
        "spatial:registration": "pixel",
    }


def init_store(
    store_path: str,
    zone_numbers: list[int],
    years: list[int],
    model_url: str,
    source_data: list[str],
    resolution: float,
    tile_size: int,
    dimensions: int,
    gsd: float | None = None,
    build_version: str = "0.0.1",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    quantization_link: str = DEFAULT_QUANTIZATION_LINK,
    create_pca_array: bool = True,
    overwrite: bool = False,
    storage_options: dict | None = None,
) -> None:
    """Create the GeoZarr store skeleton (root, zone groups, arrays, metadata).

    This must be run once before any prediction workers write to the store. Workers
    then only write data regions, so all metadata is created here to avoid races.

    Args:
        store_path: the Zarr store path or URL (e.g. gs://bucket/embeddings.zarr).
        zone_numbers: the UTM zone numbers (1-60) to create groups for.
        years: the annual reference years, defining the time axis (order preserved).
        model_url: URL reference to the encoder model.
        source_data: URLs of the source datasets.
        resolution: the projection resolution in m/pixel.
        tile_size: the tile size in pixels (must be a multiple of shard_size).
        dimensions: the embedding vector dimensionality (the band dimension size).
        gsd: ground sample distance in meters; defaults to resolution.
        build_version: version of the software that built the store.
        chunk_size: inner chunk spatial size for the embedding array.
        shard_size: outer shard spatial size; must equal the window (PATCH_SIZE) size
            and be a multiple of chunk_size, and tile_size a multiple of it.
        zstd_level: zstd compression level for the embedding and coordinate arrays.
        quantization_link: URL documenting the dequantization formula.
        create_pca_array: whether to also create the derived pca_rgb array. It costs
            nothing until written, and creating it up front keeps prediction workers
            from ever having to mutate store metadata.
        overwrite: whether to overwrite an existing store.
        storage_options: fsspec storage options for remote stores.
    """
    if tile_size % shard_size != 0:
        raise ValueError(
            f"tile_size {tile_size} must be a multiple of shard_size {shard_size}"
        )
    if shard_size % chunk_size != 0:
        raise ValueError(
            f"shard_size {shard_size} must be a multiple of chunk_size {chunk_size}"
        )
    if gsd is None:
        gsd = float(resolution)

    root = zarr.open_group(
        store=store_path,
        mode="w" if overwrite else "w-",
        storage_options=storage_options,
    )
    geoemb_attrs = build_geoemb_attrs(
        dimensions=dimensions,
        model_url=model_url,
        source_data=source_data,
        gsd=gsd,
        build_version=build_version,
        quantization_link=quantization_link,
    )
    root.attrs.update({"zarr_conventions": ZARR_CONVENTIONS, **geoemb_attrs})

    years_arr = np.array(years, dtype="int32")
    for zone_number in zone_numbers:
        projection, origin_px, shape_px = get_zone_grid(
            zone_number, resolution, tile_size
        )
        height, width = shape_px
        zone_group = root.create_group(zone_group_name(zone_number))
        zone_group.attrs.update(
            {
                "zarr_conventions": ZARR_CONVENTIONS,
                **geoemb_attrs,
                **build_zone_spatial_attrs(projection, origin_px, shape_px),
            }
        )

        zone_group.create_array(
            EMBEDDINGS_ARRAY,
            shape=(len(years), dimensions, height, width),
            chunks=(1, dimensions, chunk_size, chunk_size),
            shards=(1, dimensions, shard_size, shard_size),
            dtype="int8",
            fill_value=NODATA_VALUE,
            compressors=[ZstdCodec(level=zstd_level)],
            dimension_names=EMBEDDING_DIMENSIONS,
        )

        if create_pca_array:
            # Same shard grid as the embeddings, so a window's two objects are always
            # written together by one worker. uint8 and 3 bands, so this is about
            # 2.3% of the embedding volume.
            zone_group.create_array(
                PCA_ARRAY,
                shape=(len(years), PCA_BANDS, height, width),
                chunks=(1, PCA_BANDS, chunk_size, chunk_size),
                shards=(1, PCA_BANDS, shard_size, shard_size),
                dtype="uint8",
                fill_value=PCA_NODATA_VALUE,
                compressors=[ZstdCodec(level=zstd_level)],
                dimension_names=PCA_DIMENSIONS,
            )

        # Coordinate arrays (pixel centers) for xarray/GeoZarr friendliness.
        origin_x, origin_y = origin_px
        time_coord = zone_group.create_array(
            "time", shape=(len(years),), dtype="int32", dimension_names=("time",)
        )
        time_coord[:] = years_arr
        x_coord = zone_group.create_array(
            "x",
            shape=(width,),
            dtype="float64",
            chunks=(min(width, COORD_CHUNK_SIZE),),
            compressors=[ZstdCodec(level=zstd_level)],
            dimension_names=("x",),
        )
        x_coord[:] = (np.arange(width) + origin_x + 0.5) * projection.x_resolution
        y_coord = zone_group.create_array(
            "y",
            shape=(height,),
            dtype="float64",
            chunks=(min(height, COORD_CHUNK_SIZE),),
            compressors=[ZstdCodec(level=zstd_level)],
            dimension_names=("y",),
        )
        y_coord[:] = (np.arange(height) + origin_y + 0.5) * projection.y_resolution
        logger.info(
            "created zone group %s with shape %s",
            zone_group_name(zone_number),
            shape_px,
        )

    zarr.consolidate_metadata(root.store)
    logger.info("initialized store %s with %d zones", store_path, len(zone_numbers))


def _window_offsets(
    group: "zarr.Group", window_bounds: PixelBounds, patch_size: int
) -> tuple[int, int]:
    """Map a window's input-pixel bounds to (row, col) offsets in the zone array.

    The array origin is read back from the zone group's spatial:transform so it always
    matches what init_store wrote.

    Args:
        group: the opened zone group.
        window_bounds: window pixel bounds in the zone's northern CRS, at the input
            resolution.
        patch_size: the encoder patch size; the store grid is at 1/patch_size of the
            input resolution.

    Returns:
        tuple of (row_offset, col_offset) in output pixels.
    """
    transform = group.attrs["spatial:transform"]
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])
    return (
        window_bounds[1] // patch_size - origin_y,
        window_bounds[0] // patch_size - origin_x,
    )


def write_pca_window_region(
    store_path: str,
    zone_number: int,
    window_bounds: PixelBounds,
    time_index: int,
    rgb: np.ndarray,
    patch_size: int = 1,
    storage_options: dict | None = None,
) -> None:
    """Write one window's false-color RGB raster into the zone's pca_rgb array.

    Shares the embedding array's shard grid, so this writes whole shards and is safe
    under concurrent writes of disjoint windows.

    Args:
        store_path: the Zarr store path or URL.
        zone_number: the UTM zone number (1-60) whose group to write into.
        window_bounds: the window's pixel bounds in the zone's northern CRS, at the
            input resolution.
        time_index: the index into the time axis for this reference year.
        rgb: uint8 array of shape (PCA_BANDS, height, width) at the output resolution.
        patch_size: the encoder patch size.
        storage_options: fsspec storage options for remote stores.
    """
    if rgb.shape[0] != PCA_BANDS:
        raise ValueError(f"expected {PCA_BANDS} bands, got {rgb.shape[0]}")
    group = zarr.open_group(
        store=store_path,
        path=zone_group_name(zone_number),
        mode="r+",
        storage_options=storage_options,
    )
    if PCA_ARRAY not in group:
        raise KeyError(
            f"{PCA_ARRAY} array missing from {zone_group_name(zone_number)}; the store "
            "was created with create_pca_array=False"
        )
    row_offset, col_offset = _window_offsets(group, window_bounds, patch_size)
    _, height, width = rgb.shape
    group[PCA_ARRAY][
        time_index,
        :,
        row_offset : row_offset + height,
        col_offset : col_offset + width,
    ] = rgb


def write_window_region(
    store_path: str,
    zone_number: int,
    window_bounds: PixelBounds,
    time_index: int,
    embeddings: np.ndarray,
    patch_size: int = 1,
    storage_options: dict | None = None,
) -> None:
    """Write one window's embedding raster into the zone array region.

    The window must be aligned to the store's shard grid (guaranteed when window
    bounds are multiples of PATCH_SIZE and the shard spatial size equals the window's
    output size PATCH_SIZE / patch_size), so this writes whole shards and is safe
    under concurrent writes of disjoint windows. The array origin is read back from
    the zone group's spatial:transform so it always matches what init_store wrote.

    Args:
        store_path: the Zarr store path or URL.
        zone_number: the UTM zone number (1-60) whose group to write into.
        window_bounds: the window's pixel bounds in the zone's northern CRS, at the
            input resolution (i.e. multiples of PATCH_SIZE).
        time_index: the index into the time axis for this reference year.
        embeddings: the int8 embedding array of shape (band, height, width), at the
            output resolution (1/patch_size of the input resolution).
        patch_size: the encoder patch size; the store grid is at 1/patch_size of the
            input resolution, so the window bounds are divided by it to locate the
            output region.
        storage_options: fsspec storage options for remote stores.
    """
    group = zarr.open_group(
        store=store_path,
        path=zone_group_name(zone_number),
        mode="r+",
        storage_options=storage_options,
    )
    # The store grid is at the output resolution, so map the input-pixel window bounds
    # down to output pixels before offsetting against the (output-pixel) array origin.
    row_offset, col_offset = _window_offsets(group, window_bounds, patch_size)
    _, height, width = embeddings.shape
    array = group[EMBEDDINGS_ARRAY]
    array[
        time_index,
        :,
        row_offset : row_offset + height,
        col_offset : col_offset + width,
    ] = embeddings
