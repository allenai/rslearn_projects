"""Coverage mask deciding which ground the pipeline computes embeddings for.

A global EPSG:4326 raster at 1/120 degree, rasterised with ALL_TOUCHED so small
islands are kept. Loaded once per process and bit-packed.
"""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.windows

COVERAGE_RES_DEG = 1.0 / 120.0
COVERAGE_HEIGHT = 21600
COVERAGE_WIDTH = 43200
# Rows read per band when loading, so the full mask is never held as bool.
MASK_READ_ROWS = 1080
MASK_GDAL_CACHE_MB = 64

COVERAGE_MASK_RELPATH = "data/large_scale_embeddings/coverage_mask.tif"


def resolve_mask_path() -> Path:
    """Find the coverage mask relative to the working directory or the source tree.

    Returns:
        the path to the mask.

    Raises:
        FileNotFoundError: if the mask is in neither place.
    """
    candidates = [
        Path.cwd() / COVERAGE_MASK_RELPATH,
        Path(__file__).resolve().parents[2] / COVERAGE_MASK_RELPATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"coverage mask not found; looked in {[str(c) for c in candidates]}"
    )


@functools.lru_cache(maxsize=1)
def _packed_mask() -> npt.NDArray[np.uint8]:
    """Load the mask once per process, bit-packed.

    Returns:
        the mask as one bit per cell, row-major from (90N, 180W).
    """
    with (
        rasterio.Env(GDAL_CACHEMAX=MASK_GDAL_CACHE_MB),
        rasterio.open(resolve_mask_path()) as src,
    ):
        if (src.height, src.width) != (COVERAGE_HEIGHT, COVERAGE_WIDTH):
            raise ValueError(
                f"coverage mask is {src.height}x{src.width}, expected "
                f"{COVERAGE_HEIGHT}x{COVERAGE_WIDTH}"
            )
        bands = []
        for row in range(0, src.height, MASK_READ_ROWS):
            window = rasterio.windows.Window(
                0, row, src.width, min(MASK_READ_ROWS, src.height - row)
            )
            bands.append(np.packbits(src.read(1, window=window).astype(bool)))
    return np.concatenate(bands)


def is_covered(lats: npt.ArrayLike, lons: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether each point falls inside the covered area.

    Args:
        lats: latitudes in degrees.
        lons: longitudes in degrees; wrapped, so the antimeridian needs no special case.

    Returns:
        a boolean array of the same shape.
    """
    packed = _packed_mask()
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    rows = np.clip(
        ((90.0 - lats_arr) / COVERAGE_RES_DEG).astype(np.int64), 0, COVERAGE_HEIGHT - 1
    )
    cols = (((lons_arr + 180.0) % 360.0) / COVERAGE_RES_DEG).astype(
        np.int64
    ) % COVERAGE_WIDTH
    flat = rows * COVERAGE_WIDTH + cols
    bits = (packed[flat >> 3] >> (7 - (flat & 7))) & 1
    return bits.astype(bool)
