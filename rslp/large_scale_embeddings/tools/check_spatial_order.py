"""Confirm a written store's embeddings are spatially ordered.

Why this exists
---------------
The model wrapper flattens the encoder's output registers, and the register layout has
changed upstream (``[B, HxW, d]`` to ``[B, H, W, d]``). Flattening on the wrong layout
puts embeddings on the wrong pixels while the store stays healthy by every check
normally run: markers appear, byte counts match, and a dequantized vector still has L2
norm near 1.0, because a scrambled image is still a valid image.

What this checks, and what it does not
--------------------------------------
Embeddings of neighbouring ground are strongly correlated and decorrelate with distance.
A flattening error destroys that: adjacent pixels become arbitrary pairs and the near
field collapses toward the far field. So this catches scrambling, the catastrophic case,
and needs no knowledge of what is on the ground.

It does **not** catch a pure height/width transpose, which preserves autocorrelation and
only swaps the axes; windows and crops are square, so the reshape would not even error.
Catching that needs external geometry: render a PCA tile over imagery and look, or use
``check_zone_agreement`` below.

Usage
-----
    python -m rslp.large_scale_embeddings.tools.check_spatial_order
        --store_path https://.../embeddings.zarr --zone utm36

The store reads anonymously over https, worth preferring here because gcloud credentials
expire often and this check should not fail for that reason.
"""

import numpy as np

from rslp.log_utils import get_logger

logger = get_logger(__name__)

RESOLUTION = 10
NODATA = -128

# Separations to measure, in pixels. 1 is the near field; the largest is the far
# field the near field is compared against.
SEPARATIONS = (1, 2, 4, 8, 16, 32, 64, 128)

# The near-field value carries the test. A correct store measures 0.96 to 0.98 between
# adjacent pixels; a scrambled one collapses to its far-field value of 0.4 to 0.7.
MIN_NEAR = 0.90
# Guards only the degenerate case where every pixel is identical, which would pass
# MIN_NEAR with a perfect 1.0. Kept loose: homogeneous ground decorrelates slowly and
# that is not a defect.
MIN_GAP = 0.10


def measure_decay(patch: np.ndarray, valid: np.ndarray) -> dict[int, float]:
    """Mean cosine between pixel pairs at each horizontal separation.

    Args:
        patch: dequantized embeddings shaped (dims, height, width).
        valid: boolean mask of pixels holding data, shaped (height, width).

    Returns:
        separation in pixels mapped to mean cosine over valid pairs.
    """
    unit = patch / (np.linalg.norm(patch, axis=0, keepdims=True) + 1e-9)
    out: dict[int, float] = {}
    for step in SEPARATIONS:
        if step >= patch.shape[2]:
            continue
        both = valid[:, :-step] & valid[:, step:]
        if both.sum() < 1000:
            continue
        left = unit[:, :, :-step][:, both]
        right = unit[:, :, step:][:, both]
        out[step] = float((left * right).sum(0).mean())
    if not out:
        raise ValueError("not enough valid pixel pairs to measure decay")
    return out


def check_spatial_order(
    store_path: str,
    zone: str = "utm36",
    row: int = 951296,
    col: int = 69376,
    time_index: int = 0,
    dims: int = 64,
    size: int = 256,
) -> dict[int, float]:
    """Assert a store's embeddings are spatially autocorrelated.

    Args:
        store_path: the embeddings store, https or gs.
        zone: zone group within the store, e.g. utm36.
        row: top row of the patch to read. Chunk-aligned by default.
        col: left column of the patch to read. Chunk-aligned by default.
        time_index: which reference year to read.
        dims: leading dimensions to compare. 64 is the release candidate's
            Matryoshka width, so it is the width a consumer would use.
        size: patch edge in pixels. One inner chunk by default, so the whole
            check costs a single chunk read.

    Returns:
        the measured separation-to-cosine mapping, for logging or comparison.

    Raises:
        ValueError: if the patch is empty, or if near-field correlation is absent,
            which is what a flattening error looks like.
    """
    import zarr

    array = zarr.open_array(f"{store_path}/{zone}/embeddings", mode="r")
    block = np.asarray(
        array[time_index, :dims, row : row + size, col : col + size]
    ).astype(np.float32)
    valid = block[0] != NODATA
    logger.info(
        "%s %s: patch %dx%d at row %d col %d, %d/%d valid",
        zone,
        array.shape,
        size,
        size,
        row,
        col,
        int(valid.sum()),
        valid.size,
    )
    if valid.sum() < 1000:
        raise ValueError(
            f"patch at row {row} col {col} holds {int(valid.sum())} valid pixels; "
            "pick a location inside the run's footprint"
        )

    decay = measure_decay(block, valid)
    logger.info("mean cosine by separation:")
    for step, value in decay.items():
        logger.info("  %4d px  %6d m   %.4f", step, step * RESOLUTION, value)

    near = decay[min(decay)]
    far = decay[max(decay)]
    gap = near - far
    logger.info(
        "near %.4f, far %.4f, gap %.4f (need near > %.2f and gap > %.2f)",
        near,
        far,
        gap,
        MIN_NEAR,
        MIN_GAP,
    )

    if near < MIN_NEAR or gap < MIN_GAP:
        raise ValueError(
            f"embeddings are not spatially autocorrelated: adjacent pixels average "
            f"{near:.4f} and distant ones {far:.4f}, a gap of {gap:.4f}. Neighbouring "
            "ground should be far more alike than ground a kilometre apart, so the "
            "output is probably being written to the wrong pixels. Check whether the "
            "image's olmoearth_pretrain straddles d3e0941, which reshaped the output "
            "registers from [B, HxW, d] to [B, H, W, d]. Note this check cannot see a "
            "plain height/width transpose; for that, render a PCA tile over imagery."
        )
    logger.info("spatial order looks correct")
    return decay


# Leaves headroom for genuine resampling differences at a zone seam while sitting far
# above the ambient similarity between unrelated ground, which is 0.4 to 0.7.
MIN_CROSS_ZONE = 0.90


def check_zone_agreement(
    store_path: str,
    zone_a: str,
    zone_b: str,
    points: list[tuple[float, float]],
    time_index: int = 0,
    dims: int = 64,
) -> float:
    """Assert two zones agree on ground they both cover.

    This is the check that catches a transpose, which the decay measurement above
    cannot. Shards snap outward to 2048 px, so a strip either side of a zone
    boundary is written twice, by separate jobs whose 16x16 crop grids land at
    different ground offsets. Reorient the registers inside a crop and the same
    ground picks up a different neighbour's embedding in each zone, so agreement
    falls away. Handled correctly, the two are near-identical.

    Args:
        store_path: the embeddings store, https or gs.
        zone_a: first zone group, e.g. utm36.
        zone_b: second zone group, e.g. utm37.
        points: lon/lat pairs inside the overlap. For the utm36/utm37 seam that is
            roughly longitude 35.99 to 36.05.
        time_index: which reference year to read.
        dims: leading dimensions to compare.

    Returns:
        mean cosine across the points that had data in both zones.

    Raises:
        ValueError: if no point has data in both zones, or if agreement is too low,
            which is what a reorientation looks like.
    """
    import pyproj
    import zarr

    arrays = {
        z: zarr.open_array(f"{store_path}/{z}/embeddings", mode="r")
        for z in (zone_a, zone_b)
    }
    transformers = {
        z: pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{32600 + int(z.removeprefix('utm'))}", always_xy=True
        )
        for z in (zone_a, zone_b)
    }

    def read(zone: str, lon: float, lat: float):
        x, y = transformers[zone].transform(lon, lat)
        col, row = int(x // RESOLUTION), int((9502720 - y) // RESOLUTION)
        raw = np.asarray(arrays[zone][time_index, :dims, row, col]).astype(np.float32)
        if bool(np.all(raw == NODATA)):
            return None
        return raw / (np.linalg.norm(raw) + 1e-9)

    cosines = []
    for lon, lat in points:
        left, right = read(zone_a, lon, lat), read(zone_b, lon, lat)
        if left is None or right is None:
            logger.info("  %.4f %.4f  skipped, one zone has no data", lon, lat)
            continue
        value = float(left @ right)
        cosines.append(value)
        logger.info("  %.4f %.4f  cosine %.4f", lon, lat, value)

    if not cosines:
        raise ValueError(
            f"no point had data in both {zone_a} and {zone_b}; the overlap strip is "
            "only about 10 km wide, so pick longitudes within it"
        )

    mean = float(np.mean(cosines))
    logger.info(
        "%s vs %s: mean %.4f, min %.4f over %d points (need > %.2f)",
        zone_a,
        zone_b,
        mean,
        min(cosines),
        len(cosines),
        MIN_CROSS_ZONE,
    )
    if mean < MIN_CROSS_ZONE:
        raise ValueError(
            f"{zone_a} and {zone_b} disagree on shared ground: mean cosine {mean:.4f} "
            f"over {len(cosines)} points, below {MIN_CROSS_ZONE}. The same ground is "
            "written twice here by jobs with differently aligned crop grids, so "
            "disagreement means the output is being placed in the wrong pixel within "
            "its crop. Check whether the image's olmoearth_pretrain straddles d3e0941."
        )
    logger.info("zones agree; orientation looks correct")
    return mean
