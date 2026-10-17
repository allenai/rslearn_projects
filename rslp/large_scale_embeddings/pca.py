"""Apply olmoearth_run's global PCA basis for false-color RGB rendering of embeddings.

The RGB layer is served to map clients, so its colors must mean the same thing
everywhere. That is the whole design constraint: both the basis and the normalization
bounds have to be global, or the same color encodes different things in different
places.

The basis is olmoearth_run's per-foundation-model ``embedding_pca.pkl`` (fitted with its
``fit-embedding-pca``), read directly so both products render with one basis: three
components mapped to RGB, with per-component 2nd/98th percentile bounds from the fit
sample. It is fitted on the int8 values, and both repos quantize identically, so it is
applied to the int8 values here too.

Expectation setting: three components capture roughly 21-40% of local variance for
128-dimensional embeddings. This is a visualization of the embeddings, not a
reduced-dimension version of them.
"""

import pickle  # nosec
from dataclasses import dataclass, field

import numpy as np
from upath import UPath

from rslp.large_scale_embeddings.model import NODATA_VALUE
from rslp.large_scale_embeddings.zarr_store import PCA_NODATA_VALUE
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Three components, mapped to R, G and B.
PCA_N_COMPONENTS = 3

# Outlier clipping for the normalization bounds, as olmoearth_run fits them.
NORM_PERCENTILE_LOW = 2
NORM_PERCENTILE_HIGH = 98

# The only classes an olmoearth_run artifact may contain, each loaded as a plain state
# holder so that neither olmoearth_run nor scikit-learn is needed to read it.
_OLMOEARTH_RUN_CLASSES = {
    ("olmoearth_run.shared.tools.pca_artifact", "PcaArtifact"),
    ("sklearn.decomposition._incremental_pca", "IncrementalPCA"),
}
# What numpy needs to rebuild arrays, under its 1.x and 2.x module names.
_NUMPY_GLOBALS = {
    (module, name)
    for module in ("numpy.core.multiarray", "numpy._core.multiarray")
    for name in ("_reconstruct", "scalar")
} | {("numpy", "ndarray"), ("numpy", "dtype")}


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

    @classmethod
    def load(cls, artifact_path: str) -> "PcaArtifact":
        """Read olmoearth_run's embedding_pca.pkl.

        An IncrementalPCA without whitening projects as (x - mean_) @ components_.T,
        which is what project_to_rgb computes, so only its fitted arrays are kept.

        Args:
            artifact_path: the path of the foundation model's embedding_pca.pkl.

        Returns:
            the loaded artifact.
        """
        upath = UPath(artifact_path)
        if not upath.exists():
            raise FileNotFoundError(
                f"no PCA artifact at {artifact_path}; fit one with olmoearth_run's "
                "fit-embedding-pca"
            )
        with upath.open("rb") as f:
            wrapper = _OlmoEarthRunUnpickler(f).load()
        pca = wrapper.state["pca"].state
        if pca.get("whiten"):
            raise ValueError(f"{artifact_path} is a whitened PCA, which is unsupported")
        explained = np.asarray(pca["explained_variance_ratio_"], dtype=np.float32)
        return cls(
            mean=np.asarray(pca["mean_"], dtype=np.float32),
            components=np.asarray(pca["components_"], dtype=np.float32),
            norm_bounds=np.asarray(wrapper.state["norm_bounds"], dtype=np.float32),
            explained_variance_ratio=explained,
            metadata={
                "geoemb:pca_components": PCA_N_COMPONENTS,
                "geoemb:pca_source_artifact": artifact_path,
                "geoemb:pca_fit_pixels": int(pca["n_samples_seen_"]),
                "geoemb:pca_norm_percentiles": [
                    NORM_PERCENTILE_LOW,
                    NORM_PERCENTILE_HIGH,
                ],
                "geoemb:pca_explained_variance_ratio": [float(v) for v in explained],
                "geoemb:pca_dimensions": int(pca["mean_"].shape[0]),
                "geoemb:pca_input_space": "int8",
                "geoemb:pca_note": (
                    "False-color visualization. Three components capture only a "
                    "minority of embedding variance; do not use these bands as "
                    "features."
                ),
            },
        )


class _PickledState:
    """Stand-in for a pickled class, keeping only the state it was saved with."""

    def __setstate__(self, state: dict) -> None:
        """Keep the state.

        Args:
            state: the instance __dict__ the object was pickled with.
        """
        self.state = state


class _OlmoEarthRunUnpickler(pickle.Unpickler):  # nosec
    """Unpickles an olmoearth_run artifact, refusing anything else it might contain."""

    def find_class(self, module: str, name: str) -> type:
        """Resolve a global, allowing only the artifact's own classes and numpy.

        Args:
            module: the global's module.
            name: the global's name.

        Returns:
            the class to construct.

        Raises:
            pickle.UnpicklingError: for any other global.
        """
        if (module, name) in _OLMOEARTH_RUN_CLASSES:
            return _PickledState
        if (module, name) in _NUMPY_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name} from a PCA artifact"
        )


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

    pixels = embeddings[:, valid].astype(np.float32).T  # (n_valid, bands)
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
