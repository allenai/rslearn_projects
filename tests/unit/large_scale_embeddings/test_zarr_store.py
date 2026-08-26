"""Unit tests for rslp.large_scale_embeddings.zarr_store."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from rslp.large_scale_embeddings import zarr_store as zs
from rslp.large_scale_embeddings.tiling import get_zone_grid

RESOLUTION = 10
TILE_SIZE = 32768
MODEL_URL = "https://huggingface.co/allenai/OlmoEarth-v1_2-Small"
SOURCE_DATA = ["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]

# Required root attributes per the geoemb convention schema.
REQUIRED_GEOEMB_FIELDS = [
    "geoemb:type",
    "geoemb:dimensions",
    "geoemb:model",
    "geoemb:source_data",
    "geoemb:data_type",
]


def _init_small_store(store_path: str, dimensions: int = 4) -> None:
    """Initialize a store for zone 10 with small dims/shards for fast tests."""
    zs.init_store(
        store_path=store_path,
        zone_numbers=[10],
        years=[2024, 2025],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=RESOLUTION,
        tile_size=TILE_SIZE,
        dimensions=dimensions,
        shard_size=512,
        chunk_size=256,
    )


def test_build_geoemb_attrs_has_required_fields() -> None:
    """build_geoemb_attrs includes all required geoemb fields with correct values."""
    attrs = zs.build_geoemb_attrs(
        dimensions=128,
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        gsd=10.0,
        build_version="0.0.1",
    )
    for field in REQUIRED_GEOEMB_FIELDS:
        assert field in attrs
    assert attrs["geoemb:type"] == "pixel"
    assert attrs["geoemb:dimensions"] == 128
    assert attrs["geoemb:data_type"] == "int8"
    assert attrs["geoemb:spatial_layout"] == "utm_zones"
    quant = attrs["geoemb:quantization"]
    # Quantization object requires method and original_dtype.
    assert quant["method"] == "signed_power"
    assert quant["original_dtype"] == "float32"


def test_build_zone_spatial_attrs_transform() -> None:
    """The zone affine maps the array corner to the expected CRS coordinates."""
    projection, origin, shape = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    attrs = zs.build_zone_spatial_attrs(projection, origin, shape)
    origin_x, origin_y = origin
    height, width = shape
    assert attrs["proj:code"] == "EPSG:32610"
    assert attrs["spatial:shape"] == [height, width]
    # transform = [x_res, 0, corner_easting, 0, y_res, corner_northing].
    transform = attrs["spatial:transform"]
    assert transform[0] == RESOLUTION
    assert transform[4] == -RESOLUTION
    assert transform[2] == origin_x * RESOLUTION
    assert transform[5] == origin_y * -RESOLUTION


def test_init_store_root_and_zone_attrs(tmp_path: Path) -> None:
    """init_store writes conforming root and zone-group attributes."""
    store = str(tmp_path / "emb.zarr")
    _init_small_store(store)

    root = zarr.open_group(store=store, mode="r")
    assert isinstance(root.attrs["zarr_conventions"], list)
    convention_names = {entry["name"] for entry in root.attrs["zarr_conventions"]}
    assert "geoemb:" in convention_names
    for field in REQUIRED_GEOEMB_FIELDS:
        assert field in root.attrs

    zone = zarr.open_group(store=store, path="utm10", mode="r")
    assert zone.attrs["proj:code"] == "EPSG:32610"
    assert len(zone.attrs["spatial:transform"]) == 6
    # geoemb attrs are replicated on the zone group so it is self-describing.
    assert zone.attrs["geoemb:type"] == "pixel"


def test_init_store_array_layout(tmp_path: Path) -> None:
    """The embedding array has the expected dims, dtype, fill value, and sharding."""
    store = str(tmp_path / "emb.zarr")
    _init_small_store(store, dimensions=4)
    _, _, shape = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    height, width = shape

    array = zarr.open_group(store=store, path="utm10", mode="r")[zs.EMBEDDINGS_ARRAY]
    assert array.shape == (2, 4, height, width)
    assert array.dtype == np.int8
    assert array.fill_value == zs.NODATA_VALUE
    assert array.shards == (1, 4, 512, 512)
    assert array.chunks == (1, 4, 256, 256)


def test_write_window_region_roundtrip(tmp_path: Path) -> None:
    """A written window round-trips, and unwritten regions read as nodata."""
    store = str(tmp_path / "emb.zarr")
    _init_small_store(store, dimensions=4)
    _, origin, _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    origin_x, origin_y = origin

    bx0 = origin_x + 10 * 512
    by0 = origin_y + 100 * 512
    window_bounds = (bx0, by0, bx0 + 512, by0 + 512)
    embeddings = np.random.randint(-127, 128, size=(4, 512, 512)).astype(np.int8)
    zs.write_window_region(
        store, 10, window_bounds, time_index=1, embeddings=embeddings
    )

    array = zarr.open_group(store=store, path="utm10", mode="r")[zs.EMBEDDINGS_ARRAY]
    col0 = bx0 - origin_x
    row0 = by0 - origin_y
    written = array[1, :, row0 : row0 + 512, col0 : col0 + 512]
    assert np.array_equal(written, embeddings)
    # Same location at the other (unwritten) time index is nodata.
    assert (array[0, :, row0 : row0 + 4, col0 : col0 + 4] == zs.NODATA_VALUE).all()


def test_disjoint_writes_do_not_interfere(tmp_path: Path) -> None:
    """Two windows written to disjoint shards both round-trip correctly."""
    store = str(tmp_path / "emb.zarr")
    _init_small_store(store, dimensions=4)
    _, origin, _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    origin_x, origin_y = origin

    windows = []
    for shard_col, shard_row in [(10, 100), (12, 100)]:
        bx0 = origin_x + shard_col * 512
        by0 = origin_y + shard_row * 512
        emb = np.random.randint(-127, 128, size=(4, 512, 512)).astype(np.int8)
        zs.write_window_region(
            store, 10, (bx0, by0, bx0 + 512, by0 + 512), time_index=0, embeddings=emb
        )
        windows.append((bx0, by0, emb))

    array = zarr.open_group(store=store, path="utm10", mode="r")[zs.EMBEDDINGS_ARRAY]
    for bx0, by0, emb in windows:
        col0 = bx0 - origin_x
        row0 = by0 - origin_y
        back = array[0, :, row0 : row0 + 512, col0 : col0 + 512]
        assert np.array_equal(back, emb)


def test_write_window_region_patch_size(tmp_path: Path) -> None:
    """With patch_size>1 the store grid is at the output resolution.

    The store is created at 1/patch_size resolution and an input-pixel window lands at
    its input bounds divided by patch_size.
    """
    patch_size = 4
    window_size_px = 2048  # PATCH_SIZE, in input pixels.
    out_px = window_size_px // patch_size
    store = str(tmp_path / "emb.zarr")
    out_resolution = RESOLUTION * patch_size
    out_tile_size = TILE_SIZE // patch_size
    zs.init_store(
        store_path=store,
        zone_numbers=[10],
        years=[2024],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=out_resolution,
        tile_size=out_tile_size,
        dimensions=4,
        shard_size=out_px,
        chunk_size=256,
    )

    # The input grid (10 m) origin maps to the output grid origin by // patch_size.
    _, in_origin, _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    _, out_origin, _ = get_zone_grid(10, out_resolution, out_tile_size)
    assert out_origin[0] == in_origin[0] // patch_size
    assert out_origin[1] == in_origin[1] // patch_size

    # An input-pixel window yields an out_px raster placed at input_bounds//patch_size.
    bx0 = in_origin[0] + 3 * window_size_px
    by0 = in_origin[1] + 5 * window_size_px
    window_bounds = (bx0, by0, bx0 + window_size_px, by0 + window_size_px)
    embeddings = np.random.randint(-127, 128, size=(4, out_px, out_px)).astype(np.int8)
    zs.write_window_region(
        store,
        10,
        window_bounds,
        time_index=0,
        embeddings=embeddings,
        patch_size=patch_size,
    )

    array = zarr.open_group(store=store, path="utm10", mode="r")[zs.EMBEDDINGS_ARRAY]
    out_ox, out_oy = out_origin
    col0 = bx0 // patch_size - out_ox
    row0 = by0 // patch_size - out_oy
    back = array[0, :, row0 : row0 + out_px, col0 : col0 + out_px]
    assert np.array_equal(back, embeddings)


def test_init_store_band_chunking(tmp_path: Path) -> None:
    """The band axis is chunked, so a Matryoshka prefix read is proportionally cheap.

    With the whole width in one chunk, reading the first N dimensions still fetches all
    of them and discards the rest. This asserts the inner chunk is narrower than the
    band axis while the shard still spans it, which is what keeps one window in one
    object for concurrent writers.
    """
    store_path = str(tmp_path / "store.zarr")
    zs.init_store(
        store_path=store_path,
        zone_numbers=[10],
        years=[2024],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=RESOLUTION,
        tile_size=TILE_SIZE,
        dimensions=128,
        band_chunk=32,
        shard_size=512,
        chunk_size=256,
    )
    array = zarr.open_group(store=store_path, path="utm10", mode="r")[
        zs.EMBEDDINGS_ARRAY
    ]
    assert array.chunks == (1, 32, 256, 256)
    assert array.shards == (1, 128, 512, 512)
    # Four band groups per shard, so the trained prefixes 32/64/96/128 land exactly.
    assert array.shards[1] // array.chunks[1] == 4


def test_init_store_rejects_band_chunk_that_does_not_divide(tmp_path: Path) -> None:
    """A band_chunk that does not divide the width would make an invalid shard grid."""
    with pytest.raises(ValueError, match="must divide dimensions"):
        zs.init_store(
            store_path=str(tmp_path / "bad.zarr"),
            zone_numbers=[10],
            years=[2024],
            model_url=MODEL_URL,
            source_data=SOURCE_DATA,
            resolution=RESOLUTION,
            tile_size=TILE_SIZE,
            dimensions=100,
            band_chunk=32,
            shard_size=512,
            chunk_size=256,
        )
