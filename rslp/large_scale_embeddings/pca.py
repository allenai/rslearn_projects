"""Fit and apply a global PCA basis for false-color RGB rendering of embeddings.

The RGB layer is served to map clients, so its colors must mean the same thing
everywhere. That is the whole design constraint: both the basis and the normalization
bounds have to be global, or the same color encodes different things in different
places.

Methodology follows olmoearth_run's embedding PCA artifact: three components mapped to
RGB, fitted incrementally on a random sample of valid pixels, with per-component
2nd/98th percentile bounds computed on the transformed fit sample and applied globally
at render time. Two deliberate deviations:

1. The artifact is ``.npz`` (mean, components, bounds) plus JSON metadata rather than a
   pickled scikit-learn estimator. Projection is three dot products, so there is no
   reason to couple a multi-terabyte archive to a pickle protocol or an sklearn version.
2. The fit sample is drawn from the archive itself, stratified across UTM zones. A basis
   fitted on one region does not transfer: fitting on one region and applying it to a
   distant one captures a small fraction of the variance, and the normalization bounds
   come out nearly disjoint. Sampling must span zones, not just tiles.

Expectation setting: three components capture roughly 21-40% of local variance for
128-dimensional embeddings. This is a visualization of the embeddings, not a
reduced-dimension version of them.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import zarr
from upath import UPath

from rslp.large_scale_embeddings.model import (
    NODATA_VALUE,
    QUANTIZE_POWER,
    QUANTIZE_SCALE,
)
from rslp.large_scale_embeddings.zarr_store import (
    EMBEDDINGS_ARRAY,
    PCA_NODATA_VALUE,
    zone_group_name,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Three components, mapped to R, G and B.
PCA_N_COMPONENTS = 3

# Outlier clipping for the normalization bounds. Matches olmoearth_run.
NORM_PERCENTILE_LOW = 2
NORM_PERCENTILE_HIGH = 98

# Random valid pixels sampled per source block. 50k per block over dozens of blocks is
# far more than a 128-dimensional covariance needs, and keeps the fit cheap.
DEFAULT_PIXELS_PER_BLOCK = 50_000

# Blocks sampled per UTM zone that has data. Spread matters more than depth, since the
# basis is dominated by between-region variation.
DEFAULT_BLOCKS_PER_ZONE = 8

# Windows to try per marker before treating it as unusable. A marker can name windows
# that are entirely nodata, so one miss should not cost a sample slot.
MAX_WINDOW_ATTEMPTS = 4

ARTIFACT_ARRAYS_NAME = "arrays.npz"
ARTIFACT_METADATA_NAME = "metadata.json"


def dequantize(quantized: np.ndarray) -> np.ndarray:
    """Invert the signed-power int8 quantization back to float32.

    Args:
        quantized: int8 array of any shape.

    Returns:
        float32 array of the same shape.
    """
    rescaled = quantized.astype(np.float32) / QUANTIZE_SCALE
    return np.sign(rescaled) * np.abs(rescaled) ** QUANTIZE_POWER


@dataclass
class PcaArtifact:
    """A fitted global PCA basis plus the bounds used to render it.

    Attributes:
        mean: the fit sample mean, shape (dimensions,).
        components: the top components, shape (PCA_N_COMPONENTS, dimensions).
        norm_bounds: per-component percentile bounds, shape (2, PCA_N_COMPONENTS);
            row 0 is the low bound and row 1 the high bound.
        explained_variance_ratio: fraction of fit-sample variance per component.
        metadata: provenance, recorded onto the output array's attributes so the
            pixels are interpretable without this file.
    """

    mean: np.ndarray
    components: np.ndarray
    norm_bounds: np.ndarray
    explained_variance_ratio: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the array shapes so a malformed artifact fails at load time."""
        dims = self.mean.shape[0]
        if self.components.shape != (PCA_N_COMPONENTS, dims):
            raise ValueError(
                f"components must be ({PCA_N_COMPONENTS}, {dims}), "
                f"got {self.components.shape}"
            )
        if self.norm_bounds.shape != (2, PCA_N_COMPONENTS):
            raise ValueError(
                f"norm_bounds must be (2, {PCA_N_COMPONENTS}), "
                f"got {self.norm_bounds.shape}"
            )
        if not np.all(self.norm_bounds[1] > self.norm_bounds[0]):
            raise ValueError("norm_bounds high must exceed low for every component")

    def save(self, artifact_path: str) -> None:
        """Write the artifact to a directory, as arrays plus JSON metadata.

        Args:
            artifact_path: directory path or URL to write into.
        """
        root = UPath(artifact_path)
        root.mkdir(parents=True, exist_ok=True)
        with (root / ARTIFACT_ARRAYS_NAME).open("wb") as f:
            np.savez(
                f,
                mean=self.mean,
                components=self.components,
                norm_bounds=self.norm_bounds,
                explained_variance_ratio=self.explained_variance_ratio,
            )
        with (root / ARTIFACT_METADATA_NAME).open("w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info("wrote PCA artifact to %s", artifact_path)

    @classmethod
    def load(cls, artifact_path: str) -> "PcaArtifact":
        """Read an artifact previously written by save.

        Args:
            artifact_path: directory path or URL to read from.

        Returns:
            the loaded artifact.
        """
        root = UPath(artifact_path)
        arrays_fname = root / ARTIFACT_ARRAYS_NAME
        if not arrays_fname.exists():
            raise FileNotFoundError(
                f"no PCA artifact at {artifact_path}; run the fit_pca workflow first"
            )
        with arrays_fname.open("rb") as f:
            data = np.load(f)
            arrays = {k: data[k] for k in data.files}
        metadata: dict = {}
        metadata_fname = root / ARTIFACT_METADATA_NAME
        if metadata_fname.exists():
            with metadata_fname.open() as f:
                metadata = json.load(f)
        return cls(
            mean=arrays["mean"],
            components=arrays["components"],
            norm_bounds=arrays["norm_bounds"],
            explained_variance_ratio=arrays["explained_variance_ratio"],
            metadata=metadata,
        )


def fit_basis(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the top PCA_N_COMPONENTS components of a pixel sample.

    Uses a thin SVD of the centered sample, which is exact and needs no iteration at
    this size. The sample is expected to be a few million rows at most.

    Args:
        samples: float32 array of shape (pixels, dimensions).

    Returns:
        tuple of (mean, components, explained_variance_ratio).
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (pixels, dims), got {samples.shape}")
    if samples.shape[0] <= samples.shape[1]:
        raise ValueError(
            f"need more pixels than dimensions to fit a basis, got {samples.shape}"
        )
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    return (
        mean.astype(np.float32),
        right[:PCA_N_COMPONENTS].astype(np.float32),
        (variance / variance.sum()).astype(np.float32),
    )


def compute_norm_bounds(
    samples: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Compute global per-component percentile bounds on the transformed sample.

    These are what make colors comparable across tiles; without them each tile would
    be stretched to its own range.

    Args:
        samples: the fit sample, shape (pixels, dimensions).
        mean: the fit sample mean.
        components: the fitted components.

    Returns:
        float32 array of shape (2, PCA_N_COMPONENTS).
    """
    transformed = (samples - mean) @ components.T
    return np.percentile(
        transformed, [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH], axis=0
    ).astype(np.float32)


def project_to_rgb(embeddings: np.ndarray, artifact: PcaArtifact) -> np.ndarray:
    """Project an int8 embedding block to a uint8 RGB block.

    Nodata is preserved: any pixel whose embedding vector is the nodata value maps to
    0, which is reserved, and valid pixels are scaled into 1-255.

    Args:
        embeddings: int8 array of shape (band, height, width).
        artifact: the fitted global artifact.

    Returns:
        uint8 array of shape (PCA_N_COMPONENTS, height, width).
    """
    if embeddings.ndim != 3:
        raise ValueError(f"expected (band, height, width), got {embeddings.shape}")
    bands, height, width = embeddings.shape
    if bands != artifact.mean.shape[0]:
        raise ValueError(
            f"embedding has {bands} bands but artifact expects {artifact.mean.shape[0]}"
        )

    valid = embeddings[0] != NODATA_VALUE
    out = np.zeros((PCA_N_COMPONENTS, height, width), dtype=np.uint8)
    if not valid.any():
        return out

    pixels = dequantize(embeddings[:, valid]).T  # (n_valid, bands)
    transformed = (pixels - artifact.mean) @ artifact.components.T
    low, high = artifact.norm_bounds[0], artifact.norm_bounds[1]
    scaled = (transformed - low) / (high - low)
    # Reserve 0 for nodata, so valid pixels occupy 1-255.
    levels = np.clip(np.rint(scaled * 254.0) + 1.0, 1.0, 255.0).astype(np.uint8)
    out[:, valid] = levels.T
    return out


def downsample_rgb(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Mean-downsample a uint8 RGB block, ignoring nodata pixels.

    Averaging over valid pixels only matters at coastlines and coverage edges: a plain
    block mean would drag the reserved 0 into the average and darken every edge pixel.
    A block with no valid pixels stays nodata.

    Args:
        rgb: uint8 array of shape (bands, height, width). Height and width must be
            divisible by factor.
        factor: integer downsample factor.

    Returns:
        uint8 array of shape (bands, height // factor, width // factor).
    """
    if factor == 1:
        return rgb
    bands, height, width = rgb.shape
    if height % factor or width % factor:
        raise ValueError(f"shape {rgb.shape} is not divisible by factor {factor}")
    out_h, out_w = height // factor, width // factor

    # Validity is carried by any band; project_to_rgb sets all three to 0 together.
    valid = (rgb[0] != PCA_NODATA_VALUE).reshape(out_h, factor, out_w, factor)
    counts = valid.sum(axis=(1, 3))
    blocks = rgb.reshape(bands, out_h, factor, out_w, factor).astype(np.uint32)
    sums = (blocks * valid[None, :, :, :, :]).sum(axis=(2, 4))

    out = np.zeros((bands, out_h, out_w), dtype=np.uint8)
    keep = counts > 0
    if keep.any():
        means = sums[:, keep] / counts[keep]
        # Stay inside 1-255 so a downsampled pixel never collides with nodata.
        out[:, keep] = np.clip(np.rint(means), 1, 255).astype(np.uint8)
    return out


def build_pyramid(rgb: np.ndarray, max_level: int) -> dict[int, np.ndarray]:
    """Build every pyramid level for one window from its full-resolution RGB.

    Levels are produced by repeated halving of the previous level rather than by
    downsampling the original each time, which is both cheaper and what a viewer
    stepping through zooms will visually expect.

    Args:
        rgb: uint8 array of shape (bands, height, width) at level 0.
        max_level: deepest level to produce, downsampled 2**max_level.

    Returns:
        mapping of level to its uint8 array, including level 0.
    """
    levels = {0: rgb}
    current = rgb
    for level in range(1, max_level + 1):
        current = downsample_rgb(current, 2)
        levels[level] = current
    return levels


def _sample_block_pixels(
    block: np.ndarray, pixels_per_block: int, rng: random.Random
) -> np.ndarray | None:
    """Sample random valid pixel vectors from one embedding block.

    Args:
        block: int8 array of shape (band, height, width).
        pixels_per_block: maximum pixels to keep.
        rng: seeded RNG, used to derive a numpy generator.

    Returns:
        float32 array of shape (pixels, band), or None when the block is all nodata.
    """
    valid = block[0] != NODATA_VALUE
    count = int(valid.sum())
    if count == 0:
        return None
    pixels = dequantize(block[:, valid]).T
    if count > pixels_per_block:
        generator = np.random.default_rng(rng.getrandbits(63))
        keep = generator.choice(count, size=pixels_per_block, replace=False)
        pixels = pixels[keep]
    return pixels


def summarize_artifact(artifact: PcaArtifact) -> str:
    """Render a one-line human summary of a fitted artifact.

    Args:
        artifact: the artifact to describe.

    Returns:
        a summary string for logging.
    """
    evr = artifact.explained_variance_ratio[:PCA_N_COMPONENTS]
    return (
        f"{PCA_N_COMPONENTS} components, "
        f"explained variance {evr.sum():.4f} ({', '.join(f'{v:.4f}' for v in evr)}), "
        f"bounds low={np.round(artifact.norm_bounds[0], 4).tolist()} "
        f"high={np.round(artifact.norm_bounds[1], 4).tolist()}"
    )


def build_metadata(
    store_path: str,
    zones: list[int],
    pixels: int,
    blocks: int,
    seed: int,
    explained_variance_ratio: np.ndarray,
    dimensions: int | None = None,
) -> dict:
    """Assemble the provenance recorded with the artifact and on the output array.

    Args:
        store_path: the archive the sample was drawn from.
        zones: UTM zone numbers contributing to the fit.
        pixels: total pixels in the fit sample.
        blocks: total blocks sampled.
        seed: the sampling seed.
        explained_variance_ratio: per-component explained variance.
        dimensions: embedding width the basis was fitted on.

    Returns:
        a JSON-serializable metadata dict.
    """
    return {
        "geoemb:pca_components": PCA_N_COMPONENTS,
        "geoemb:pca_source_store": store_path,
        "geoemb:pca_source_zones": sorted(zones),
        "geoemb:pca_fit_pixels": int(pixels),
        "geoemb:pca_fit_blocks": int(blocks),
        "geoemb:pca_fit_seed": int(seed),
        "geoemb:pca_norm_percentiles": [NORM_PERCENTILE_LOW, NORM_PERCENTILE_HIGH],
        "geoemb:pca_explained_variance_ratio": [
            float(v) for v in explained_variance_ratio[:PCA_N_COMPONENTS]
        ],
        "geoemb:pca_dimensions": dimensions,
        "geoemb:pca_fitted_at": datetime.now(UTC).isoformat(),
        "geoemb:pca_note": (
            "False-color visualization. Three components capture only a minority of "
            "embedding variance; do not use these bands as features."
        ),
    }


def _markers_by_zone(completed_paths: list[str]) -> dict[int, list[UPath]]:
    """Group completion markers by UTM zone number.

    Markers are named ``{crs}_{x}_{y}.json``, so the zone comes from the filename and
    no marker has to be opened to stratify the sample.

    Args:
        completed_paths: marker directories, typically one per reference year.

    Returns:
        mapping of zone number to marker paths.
    """
    by_zone: dict[int, list[UPath]] = {}
    for completed_path in completed_paths:
        root = UPath(completed_path)
        if not root.exists():
            logger.warning(
                "marker directory %s does not exist, skipping", completed_path
            )
            continue
        for fname in root.iterdir():
            if fname.name.endswith(".json"):
                epsg = fname.name.split("_")[0]
                zone = int(epsg.split(":")[1]) % 100
                by_zone.setdefault(zone, []).append(fname)
    return by_zone


def _sample_from_marker(
    marker_fname: UPath,
    store_path: str,
    chunk_size: int,
    pixels_per_block: int,
    rng: random.Random,
    storage_options: dict | None,
) -> np.ndarray | None:
    """Read one chunk from a written window named by a marker and sample its pixels.

    Reads a single inner chunk rather than a whole shard: a chunk is a few megabytes
    against a shard's few hundred, and 65,536 pixels is already more than the sample
    needs.

    Args:
        marker_fname: the completion marker to sample from.
        store_path: the Zarr store path or URL.
        chunk_size: inner chunk spatial size to read.
        pixels_per_block: maximum pixels to keep from this chunk.
        rng: seeded RNG.
        storage_options: fsspec storage options.

    Returns:
        float32 array of shape (pixels, dimensions), or None if nothing usable.
    """
    with marker_fname.open() as f:
        marker = json.load(f)
    written = marker.get("written") or []
    if not written:
        return None

    epsg = marker_fname.name.split("_")[0]
    zone = int(epsg.split(":")[1]) % 100
    group = zarr.open_group(
        store=store_path,
        path=zone_group_name(zone),
        mode="r",
        storage_options=storage_options,
    )
    transform = group.attrs["spatial:transform"]
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])

    # Try windows in random order rather than betting on one: a window can be entirely
    # nodata (ocean, or missing imagery), and giving up on the first miss would waste
    # this marker's slot in the stratified sample.
    candidates = list(written)
    rng.shuffle(candidates)
    for x, y in candidates[:MAX_WINDOW_ATTEMPTS]:
        row = y - origin_y
        col = x - origin_x
        block = np.asarray(
            group[EMBEDDINGS_ARRAY][
                marker["time_index"], :, row : row + chunk_size, col : col + chunk_size
            ]
        )
        pixels = _sample_block_pixels(block, pixels_per_block, rng)
        if pixels is not None and len(pixels) > 0:
            return pixels
    return None


def fit_pca(
    store_path: str,
    completed_paths: list[str],
    artifact_path: str,
    blocks_per_zone: int = DEFAULT_BLOCKS_PER_ZONE,
    pixels_per_block: int = DEFAULT_PIXELS_PER_BLOCK,
    chunk_size: int = 256,
    seed: int = 42,
    storage_options: dict | None = None,
) -> None:
    """Fit a global PCA basis from an existing archive and write the artifact.

    Samples are stratified across every UTM zone that has data, because a basis fitted
    on one region does not transfer to another. Sampling reads one inner chunk per
    selected window, so the whole fit costs on the order of a gigabyte of reads rather
    than a pass over the archive.

    Run this once per store, before writing any pca_rgb output. Refitting produces a
    different basis, which invalidates every previously written RGB pixel.

    Args:
        store_path: the GeoZarr store to sample from.
        completed_paths: completion-marker directories, one per reference year.
        artifact_path: directory path or URL to write the artifact into.
        blocks_per_zone: windows to sample per zone with data.
        pixels_per_block: pixels to keep per sampled chunk.
        chunk_size: inner chunk spatial size to read per sample.
        seed: sampling seed, recorded in the artifact metadata.
        storage_options: fsspec storage options for remote stores.
    """
    rng = random.Random(seed)
    by_zone = _markers_by_zone(completed_paths)
    if not by_zone:
        raise ValueError(
            f"no completion markers found under {completed_paths}; nothing to fit on"
        )
    logger.info("found markers in %d zone(s): %s", len(by_zone), sorted(by_zone))

    collected: list[np.ndarray] = []
    zones_used: list[int] = []
    blocks = 0
    for zone in sorted(by_zone):
        markers = sorted(by_zone[zone], key=lambda p: p.name)
        rng.shuffle(markers)
        taken = 0
        for marker_fname in markers:
            if taken >= blocks_per_zone:
                break
            try:
                pixels = _sample_from_marker(
                    marker_fname,
                    store_path,
                    chunk_size,
                    pixels_per_block,
                    rng,
                    storage_options,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("sampling %s failed: %s", marker_fname.name, exc)
                continue
            if pixels is None or len(pixels) == 0:
                continue
            collected.append(pixels)
            taken += 1
            blocks += 1
        if taken:
            zones_used.append(zone)
        logger.info("zone %02d: sampled %d block(s)", zone, taken)

    if not collected:
        raise ValueError("sampled no valid pixels; check store_path and the markers")

    samples = np.concatenate(collected, axis=0)
    logger.info(
        "fitting on %d pixels from %d blocks across %d zones",
        len(samples),
        blocks,
        len(zones_used),
    )
    mean, components, explained = fit_basis(samples)
    norm_bounds = compute_norm_bounds(samples, mean, components)
    artifact = PcaArtifact(
        mean=mean,
        components=components,
        norm_bounds=norm_bounds,
        explained_variance_ratio=explained,
        metadata=build_metadata(
            store_path=store_path,
            zones=zones_used,
            pixels=len(samples),
            blocks=blocks,
            seed=seed,
            explained_variance_ratio=explained,
            dimensions=int(samples.shape[1]),
        ),
    )
    logger.info("fitted %s", summarize_artifact(artifact))
    artifact.save(artifact_path)
