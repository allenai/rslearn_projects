"""The ground the run covers, as a baked-in raster mask.

Replaces the point-sampled `global_land_mask` test. Two things were wrong with that
mask and both are fixed here rather than separately, because either alone still loses
land:

  * The mask itself omits sub-kilometre features. It reports Padre Island, the Florida
    Keys and Male as ocean, so no amount of finer sampling recovers them.
  * The sampling was coarser than the data it queried: a 2.56 km lattice against a
    926 m mask, so a narrow island could fall between sample points even when the mask
    knew about it.

This mask is built by rasterising with ALL_TOUCHED, so any polygon touching a cell
lights it, and it is a strict superset of the previous mask: every cell the old rule
called land is set here. That is what keeps already-computed blocks valid.

Packed to one bit per cell (117 MB) and cached per process, so workers share one
read-only lookup rather than rebuilding geometry.
"""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio

# Degrees per cell. The mask is a global EPSG:4326 grid at this resolution, so a cell
# is about 926 m at the equator.
COVERAGE_RES_DEG = 1.0 / 120.0
COVERAGE_HEIGHT = 21600
COVERAGE_WIDTH = 43200

COVERAGE_MASK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "large_scale_embeddings"
    / "coverage_mask.tif"
)


@functools.lru_cache(maxsize=1)
def _packed_mask() -> npt.NDArray[np.uint8]:
    """Load the mask once per process, bit-packed.

    Returns:
        the mask as one bit per cell, row-major from (90N, 180W).
    """
    with rasterio.open(COVERAGE_MASK_PATH) as src:
        if (src.height, src.width) != (COVERAGE_HEIGHT, COVERAGE_WIDTH):
            raise ValueError(
                f"coverage mask is {src.height}x{src.width}, expected "
                f"{COVERAGE_HEIGHT}x{COVERAGE_WIDTH}"
            )
        arr = src.read(1).astype(bool)
    return np.packbits(np.reshape(arr, -1))


def is_covered(lats: npt.ArrayLike, lons: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether each point falls inside the covered area.

    Signature matches `global_land_mask.globe.is_land` so it drops into the same call
    site, and longitudes wrap so the antimeridian needs no special case.

    Args:
        lats: latitudes in degrees.
        lons: longitudes in degrees.

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
