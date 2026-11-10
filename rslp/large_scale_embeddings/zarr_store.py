"""GeoZarr store for global quantized OlmoEarth embeddings.

Follows the geoemb embeddings-zarr-convention
(https://github.com/geo-embeddings/embeddings-zarr-convention) with the ``utm_zones``
layout: one ``utm{NN}`` group per UTM zone in its northern CRS, each holding an int8
``embeddings`` array with dimensions ``(time, band, y, x)``.

``init_store`` writes all metadata once. Workers then only write whole shards, one per
prediction window, so concurrent writes of disjoint windows are safe.
"""

import importlib.metadata
import pathlib
import subprocess  # nosec: only ever run with a fixed argv and no shell
from datetime import timedelta

import numpy as np
import zarr
from fsspec.core import url_to_fs
from rslearn.utils.geometry import PixelBounds, Projection
from zarr.codecs import ZstdCodec
from zarr.core.buffer import Buffer
from zarr.storage import FsspecStore
from zarr.storage._fsspec import _dereference_path

from rslp.large_scale_embeddings.model import (
    NODATA_VALUE,
    QUANTIZE_POWER,
    QUANTIZE_SCALE,
)
from rslp.large_scale_embeddings.tiling import get_zone_grid
from rslp.log_utils import get_logger

logger = get_logger(__name__)

EMBEDDINGS_ARRAY = "embeddings"
EMBEDDING_DIMENSIONS = ("time", "band", "y", "x")

# The shard must equal one prediction window so each window writes exactly one shard.
DEFAULT_CHUNK_SIZE = 256
DEFAULT_SHARD_SIZE = 2048
DEFAULT_ZSTD_LEVEL = 3
# Dimensions per inner chunk along the band axis. Matches the narrowest Matryoshka
# width, so a prefix read fetches exactly one chunk.
DEFAULT_BAND_CHUNK = 64

# Chunks are written once and never change, so they can be cached as immutable.
DEFAULT_CHUNK_MAX_AGE = timedelta(days=365)
DEFAULT_METADATA_MAX_AGE = timedelta(minutes=10)

# Chunk size for the 1-D x/y coordinate arrays.
COORD_CHUNK_SIZE = 65536

# Prefix widths the distilled encoder is trained to emit.
DEFAULT_MATRYOSHKA_DIMS = [128, 64]

GEOEMB_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/geo-embeddings/embeddings-zarr-convention/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/geo-embeddings/embeddings-zarr-convention/blob/v1/README.md",
    "uuid": "61c12cc5-0e28-4056-999a-480cf3fb7e4c",
    "name": "geoemb:",
    "description": "Geoembeddings convention for geospatial embedding arrays with model provenance",
}
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
LICENSE_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/clbarnes/zarr-convention-license/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/clbarnes/zarr-convention-license/blob/v1/README.md",
    "uuid": "b77365e5-2b0c-4141-b917-c03b7c68e935",
    "name": "license",
    "description": "License specifier for Zarr data",
}
ZARR_CONVENTIONS = [
    GEOEMB_CONVENTION,
    SPATIAL_CONVENTION,
    PROJ_CONVENTION,
    LICENSE_CONVENTION,
]

# Recorded as geoemb:source_data, keyed by the modality tokens in an input name.
SOURCE_DATA_URLS = {
    "s2": "https://sentinel.esa.int/web/sentinel/missions/sentinel-2",
    "s1": "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
    "landsat": "https://www.usgs.gov/landsat-missions",
}

# This repository and the vendored dependencies whose versions identify a build, with
# their checkout paths locally and in the image.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_COMPONENTS = (
    ("rslearn", "docker_build/rslearn", "/opt/rslearn"),
    (
        "olmoearth_pretrain",
        "docker_build/olmoearth_pretrain",
        "/opt/olmoearth_pretrain",
    ),
)

# Filesystems whose writes accept per-object Cache-Control metadata.
CACHE_CONTROL_PROTOCOLS = {"gs", "gcs"}


def cache_control(max_age: timedelta, immutable: bool) -> str:
    """The Cache-Control header for an object with this lifetime.

    Args:
        max_age: how long a cache may serve the object without revalidating.
        immutable: whether the object can never change.

    Returns:
        the header value.
    """
    parts = ["public", f"max-age={int(max_age.total_seconds())}"]
    if immutable:
        parts.append("immutable")
    return ", ".join(parts)


class CacheControlStore(FsspecStore):
    """A GCS-backed store that sets Cache-Control on each object as it is written."""

    def __init__(
        self,
        *args: object,
        chunk_max_age: timedelta = DEFAULT_CHUNK_MAX_AGE,
        metadata_max_age: timedelta = DEFAULT_METADATA_MAX_AGE,
        **kwargs: object,
    ) -> None:
        """Wrap an FsspecStore with the two cache lifetimes.

        Args:
            args: passed to FsspecStore.
            chunk_max_age: lifetime for a data chunk, served immutable.
            metadata_max_age: lifetime for zarr.json.
            kwargs: passed to FsspecStore.
        """
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._chunk_header = cache_control(chunk_max_age, immutable=True)
        self._metadata_header = cache_control(metadata_max_age, immutable=False)

    def _header_for(self, key: str) -> str:
        """The Cache-Control value for one store key."""
        return self._chunk_header if "/c/" in key else self._metadata_header

    async def set(
        self, key: str, value: Buffer, byte_range: tuple | None = None
    ) -> None:
        """Write one object with its Cache-Control, mirroring FsspecStore.set.

        Args:
            key: the store key.
            value: the bytes to write.
            byte_range: unsupported, as in the parent.

        Raises:
            NotImplementedError: if a byte range is given.
        """
        if not self._is_open:
            await self._open()
        self._check_writable()
        if byte_range is not None:
            raise NotImplementedError
        await self.fs._pipe_file(
            _dereference_path(self.path, key),
            value.to_bytes(),
            fixed_key_metadata={"cache_control": self._header_for(key)},
        )


def writable_store(
    store_path: str,
    storage_options: dict | None = None,
    chunk_max_age: timedelta = DEFAULT_CHUNK_MAX_AGE,
    metadata_max_age: timedelta = DEFAULT_METADATA_MAX_AGE,
) -> FsspecStore | str:
    """The store to open for writing, setting Cache-Control on GCS.

    Args:
        store_path: the store path or URL.
        storage_options: fsspec storage options for remote stores.
        chunk_max_age: lifetime for a data chunk, served immutable.
        metadata_max_age: lifetime for zarr.json.

    Returns:
        a CacheControlStore on GCS, an FsspecStore on another remote filesystem, or
        the path unchanged for a local one.
    """
    if "://" not in store_path:
        return store_path
    if store_path.split("://", 1)[0] not in CACHE_CONTROL_PROTOCOLS:
        return FsspecStore.from_url(
            store_path, storage_options=storage_options, read_only=False
        )
    fs, path = url_to_fs(store_path, asynchronous=True, **(storage_options or {}))
    return CacheControlStore(
        fs,
        path=path,
        read_only=False,
        chunk_max_age=chunk_max_age,
        metadata_max_age=metadata_max_age,
    )


def _git_commit(*candidates: str | pathlib.Path) -> str | None:
    """Short commit of the first candidate path that is a git checkout.

    Args:
        candidates: paths to try, in order.

    Returns:
        the short commit, or None if none of the paths is a checkout.
    """
    for path in candidates:
        if not pathlib.Path(path, ".git").exists():
            continue
        try:
            out = subprocess.run(  # nosec: fixed argv, no shell
                ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return out.stdout.strip() or None
    return None


def _installed_version(name: str) -> str | None:
    """Version of an installed distribution, or None if it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_build_version() -> str:
    """Identify the code that built a store, for geoemb:build_version.

    Uses git commits where a checkout is present, else installed versions.

    Returns:
        a string naming this repository and each vendored dependency.
    """
    project = _installed_version("rslearn_projects") or "0"
    commit = _git_commit(REPO_ROOT)
    parts = [f"rslearn_projects {project}" + (f"+{commit}" if commit else "")]
    for name, local_path, image_path in BUILD_COMPONENTS:
        version = _git_commit(REPO_ROOT / local_path, image_path) or (
            _installed_version(name)
        )
        parts.append(f"{name} {version or 'unknown'}")
    return "; ".join(parts)


def source_data_for(inputs: str) -> list[str]:
    """The source-dataset URLs for an input name like ``s2_s1_landsat_distilled``.

    Args:
        inputs: an `EmbeddingInputs` value.

    Returns:
        the URLs to record as geoemb:source_data.
    """
    tokens = set(inputs.split("_"))
    return [url for name, url in SOURCE_DATA_URLS.items() if name in tokens]


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


def _quantization_attrs() -> dict:
    """Build the geoemb:quantization object for the signed-power scheme.

    `formula`, `power`, `valid_range` and `nodata` extend the convention, whose
    quantization object only models a linear scale and offset.
    """
    return {
        "method": "signed_power",
        "power": QUANTIZE_POWER,
        "formula": f"(abs(q) / {QUANTIZE_SCALE}) ** {QUANTIZE_POWER} * sign(q)",
        "original_dtype": "float32",
        "quantized_dtype": "int8",
        "valid_range": [-127, 127],
        "nodata": NODATA_VALUE,
    }


def build_geoemb_attrs(
    dimensions: int,
    model_url: str,
    source_data: list[str],
    gsd: float,
    build_version: str,
    matryoshka_dims: list[int] | None = None,
) -> dict:
    """Build the geoemb convention attributes (shared by the root and zone groups).

    Args:
        dimensions: the embedding vector dimensionality.
        model_url: URL reference to the encoder model.
        source_data: URLs of the source datasets.
        gsd: ground sample distance in meters.
        build_version: version of the software that built the store.
        matryoshka_dims: prefix widths the model is trained to emit, if any.

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
        "geoemb:quantization": _quantization_attrs(),
        "geoemb:build_version": build_version,
        **(
            {"geoemb:matryoshka_dims": sorted(matryoshka_dims, reverse=True)}
            if matryoshka_dims
            else {}
        ),
    }


def build_license_attrs(license_url: str) -> dict:
    """Build the license convention attribute.

    Args:
        license_url: URL of the full licence text.

    Returns:
        a dict holding the license attribute.
    """
    return {"license": {"url": license_url}}


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
    # Affine (col, row) -> (easting, northing); the pixel grid's origin is CRS (0, 0).
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
    license_url: str,
    gsd: float | None = None,
    build_version: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    band_chunk: int = DEFAULT_BAND_CHUNK,
    matryoshka_dims: list[int] | None = None,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    overwrite: bool = False,
    storage_options: dict | None = None,
    chunk_max_age: timedelta = DEFAULT_CHUNK_MAX_AGE,
    metadata_max_age: timedelta = DEFAULT_METADATA_MAX_AGE,
) -> None:
    """Create the GeoZarr store skeleton (root, zone groups, arrays, metadata).

    Run once before any prediction workers write to the store.

    Args:
        store_path: the Zarr store path or URL (e.g. gs://bucket/embeddings.zarr).
        zone_numbers: the UTM zone numbers (1-60) to create groups for.
        years: the annual reference years, defining the time axis (order preserved).
        model_url: URL reference to the encoder model.
        source_data: URLs of the source datasets.
        resolution: the projection resolution in m/pixel.
        tile_size: the tile size in pixels (must be a multiple of shard_size).
        dimensions: the embedding vector dimensionality (the band dimension size).
        license_url: URL of the licence text, recorded under the license convention.
        gsd: ground sample distance in meters; defaults to resolution.
        build_version: version of the software that built the store; defaults to
            get_build_version().
        chunk_size: inner chunk spatial size for the embedding array.
        shard_size: outer shard spatial size; must equal one prediction window's
            output size and be a multiple of chunk_size.
        band_chunk: dimensions per inner chunk along the band axis; must divide
            dimensions.
        matryoshka_dims: prefix widths the model supports.
        zstd_level: zstd compression level for the embedding and coordinate arrays.
        overwrite: whether to overwrite an existing store.
        storage_options: fsspec storage options for remote stores.
        chunk_max_age: how long a cache may hold a data chunk.
        metadata_max_age: how long a cache may hold zarr.json.
    """
    if tile_size % shard_size != 0:
        raise ValueError(
            f"tile_size {tile_size} must be a multiple of shard_size {shard_size}"
        )
    if shard_size % chunk_size != 0:
        raise ValueError(
            f"shard_size {shard_size} must be a multiple of chunk_size {chunk_size}"
        )
    if dimensions % min(band_chunk, dimensions) != 0:
        raise ValueError(f"band_chunk {band_chunk} must divide dimensions {dimensions}")
    if gsd is None:
        gsd = float(resolution)

    root = zarr.open_group(
        store=writable_store(
            store_path, storage_options, chunk_max_age, metadata_max_age
        ),
        mode="w" if overwrite else "w-",
    )
    geoemb_attrs = build_geoemb_attrs(
        dimensions=dimensions,
        model_url=model_url,
        source_data=source_data,
        matryoshka_dims=matryoshka_dims,
        gsd=gsd,
        build_version=build_version
        if build_version is not None
        else get_build_version(),
    )
    license_attrs = build_license_attrs(license_url)
    root.attrs.update(
        {"zarr_conventions": ZARR_CONVENTIONS, **geoemb_attrs, **license_attrs}
    )

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
                **license_attrs,
                **build_zone_spatial_attrs(projection, origin_px, shape_px),
            }
        )

        zone_group.create_array(
            EMBEDDINGS_ARRAY,
            shape=(len(years), dimensions, height, width),
            chunks=(1, min(band_chunk, dimensions), chunk_size, chunk_size),
            shards=(1, dimensions, shard_size, shard_size),
            dtype="int8",
            fill_value=NODATA_VALUE,
            compressors=[ZstdCodec(level=zstd_level)],
            dimension_names=EMBEDDING_DIMENSIONS,
        )

        # Pixel-center coordinate arrays, for xarray.
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

    Args:
        group: the opened zone group.
        window_bounds: window pixel bounds in the zone's northern CRS, at the input
            resolution.
        patch_size: input pixels per output pixel.

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


def write_window_region(
    store_path: str,
    zone_number: int,
    window_bounds: PixelBounds,
    time_index: int,
    embeddings: np.ndarray,
    patch_size: int = 1,
    storage_options: dict | None = None,
    chunk_max_age: timedelta = DEFAULT_CHUNK_MAX_AGE,
    metadata_max_age: timedelta = DEFAULT_METADATA_MAX_AGE,
) -> None:
    """Write one window's embeddings into the zone array as exactly one shard.

    Args:
        store_path: the Zarr store path or URL.
        zone_number: the UTM zone number (1-60) whose group to write into.
        window_bounds: the window's pixel bounds in the zone's northern CRS, at the
            input resolution.
        time_index: the index into the time axis for this reference year.
        embeddings: int8 array of shape (band, height, width) at the output
            resolution.
        patch_size: input pixels per output pixel.
        storage_options: fsspec storage options for remote stores.
        chunk_max_age: how long a cache may hold a data chunk.
        metadata_max_age: how long a cache may hold zarr.json.

    Raises:
        ValueError: if the write would not cover exactly one shard, since a partial
            shard write races with the window that shares it.
    """
    group = zarr.open_group(
        store=writable_store(
            store_path, storage_options, chunk_max_age, metadata_max_age
        ),
        path=zone_group_name(zone_number),
        mode="r+",
    )
    row_offset, col_offset = _window_offsets(group, window_bounds, patch_size)
    _, height, width = embeddings.shape
    array = group[EMBEDDINGS_ARRAY]
    shard_height, shard_width = array.shards[-2:]
    if (
        (height, width) != (shard_height, shard_width)
        or row_offset % shard_height != 0
        or col_offset % shard_width != 0
    ):
        raise ValueError(
            f"window at ({row_offset}, {col_offset}) of size {height}x{width} does not "
            f"cover exactly one {shard_height}x{shard_width} shard"
        )
    array[
        time_index,
        :,
        row_offset : row_offset + height,
        col_offset : col_offset + width,
    ] = embeddings
