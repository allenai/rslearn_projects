"""Unit tests for rslp.large_scale_embeddings.pca and the pca_rgb store array."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rslp.large_scale_embeddings import pca
from rslp.large_scale_embeddings import zarr_store as zs
from rslp.large_scale_embeddings.model import quantize_embeddings
from tests.unit.large_scale_embeddings.pca_fixtures import (
    fit_int8_basis,
    write_olmoearth_run_artifact,
)

DIMS = 8


def _random_embeddings(rng: np.random.Generator, n: int, dims: int) -> np.ndarray:
    """Build L2-normalized float vectors with anisotropic structure to find."""
    scale = np.linspace(1.0, 0.05, dims)
    x = rng.normal(size=(n, dims)) * scale
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _artifact(rng: np.random.Generator, dims: int = DIMS) -> pca.PcaArtifact:
    floats = _random_embeddings(rng, 5000, dims).astype(np.float32)
    return fit_int8_basis(np.asarray(quantize_embeddings(torch.from_numpy(floats))))


def test_project_to_rgb_reserves_zero_for_nodata() -> None:
    rng = np.random.default_rng(3)
    artifact = _artifact(rng)

    floats = _random_embeddings(rng, 16, DIMS).astype(np.float32)
    block = np.asarray(quantize_embeddings(torch.from_numpy(floats))).T.reshape(
        DIMS, 4, 4
    )
    # Mark one pixel as nodata across all bands.
    block[:, 0, 0] = zs.NODATA_VALUE

    rgb = pca.project_to_rgb(block, artifact)

    assert rgb.shape == (pca.PCA_N_COMPONENTS, 4, 4)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[:, 0, 0] == zs.PCA_NODATA_VALUE)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    # Valid pixels never collide with the reserved nodata value.
    assert rgb[:, valid].min() >= 1


def test_project_to_rgb_all_nodata_returns_zeros() -> None:
    rng = np.random.default_rng(4)
    artifact = _artifact(rng)
    block = np.full((DIMS, 4, 4), zs.NODATA_VALUE, dtype=np.int8)
    rgb = pca.project_to_rgb(block, artifact)
    assert rgb.shape == (pca.PCA_N_COMPONENTS, 4, 4)
    assert not rgb.any()


def test_project_to_rgb_is_deterministic_across_blocks() -> None:
    """The same vector must render to the same color wherever it appears.

    This is the property that global norm bounds exist to guarantee.
    """
    rng = np.random.default_rng(5)
    artifact = _artifact(rng)
    floats = _random_embeddings(rng, 4, DIMS).astype(np.float32)
    quant = np.asarray(quantize_embeddings(torch.from_numpy(floats))).T
    block_a = quant.reshape(DIMS, 2, 2)
    block_b = quant[:, ::-1].reshape(DIMS, 2, 2)

    rgb_a = pca.project_to_rgb(np.ascontiguousarray(block_a), artifact)
    rgb_b = pca.project_to_rgb(np.ascontiguousarray(block_b), artifact)
    np.testing.assert_array_equal(rgb_a[:, 0, 0], rgb_b[:, 1, 1])


def test_artifact_load_missing_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fit-embedding-pca"):
        pca.PcaArtifact.load(str(tmp_path / "embedding_pca.pkl"))


def test_artifact_roundtrip(tmp_path: Path) -> None:
    """What the fixture writer pickles is what the loader reads back."""
    artifact = _artifact(np.random.default_rng(6))
    path = write_olmoearth_run_artifact(tmp_path / "embedding_pca.pkl", artifact)
    loaded = pca.PcaArtifact.load(path)

    np.testing.assert_allclose(loaded.mean, artifact.mean)
    np.testing.assert_allclose(loaded.components, artifact.components)
    np.testing.assert_array_equal(loaded.norm_bounds, artifact.norm_bounds)
    assert loaded.metadata["geoemb:pca_components"] == pca.PCA_N_COMPONENTS


def test_artifact_validates_shapes() -> None:
    with pytest.raises(ValueError, match="components must be"):
        pca.PcaArtifact(
            mean=np.zeros(8, np.float32),
            components=np.zeros((2, 8), np.float32),
            norm_bounds=np.array([[0, 0, 0], [1, 1, 1]], np.float32),
            explained_variance_ratio=np.zeros(3, np.float32),
        )
    with pytest.raises(ValueError, match="high must exceed low"):
        pca.PcaArtifact(
            mean=np.zeros(8, np.float32),
            components=np.zeros((3, 8), np.float32),
            norm_bounds=np.array([[1, 1, 1], [0, 0, 0]], np.float32),
            explained_variance_ratio=np.zeros(3, np.float32),
        )


def test_downsample_rgb_ignores_nodata() -> None:
    """A block mean must not average the reserved 0 into valid pixels."""
    rgb = np.full((3, 8, 8), 200, dtype=np.uint8)
    rgb[:, :4, :4] = zs.PCA_NODATA_VALUE
    out = pca.downsample_rgb(rgb, 2)

    assert out.shape == (3, 4, 4)
    # The fully-nodata quadrant stays nodata.
    assert (out[:, :2, :2] == zs.PCA_NODATA_VALUE).all()
    # Valid areas keep their value rather than being darkened toward 0.
    assert (out[:, 2:, 2:] == 200).all()


def test_downsample_rgb_partial_block_uses_valid_pixels_only() -> None:
    rgb = np.zeros((3, 2, 2), dtype=np.uint8)
    rgb[:, 0, 0] = 100  # one valid pixel of four
    out = pca.downsample_rgb(rgb, 2)
    # Mean over the single valid pixel, not over all four.
    assert out.shape == (3, 1, 1)
    assert (out[:, 0, 0] == 100).all()


def test_downsample_rgb_never_emits_the_nodata_sentinel() -> None:
    rgb = np.ones((3, 4, 4), dtype=np.uint8)  # darkest valid value
    out = pca.downsample_rgb(rgb, 2)
    assert out.min() >= 1


def test_downsample_rgb_rejects_indivisible_shape() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        pca.downsample_rgb(np.zeros((3, 5, 5), np.uint8), 2)


def test_build_pyramid_halves_each_level() -> None:
    rgb = np.full((3, 2048, 2048), 128, dtype=np.uint8)
    levels = pca.build_pyramid(rgb, 3)
    assert sorted(levels) == [0, 1, 2, 3]
    assert [levels[k].shape[1] for k in range(4)] == [2048, 1024, 512, 256]
    assert levels[0] is rgb  # level 0 is the input, not a copy


# A real olmoearth_run artifact, pickled by its own PcaArtifact and scikit-learn 1.8
# IncrementalPCA, with sklearn's transform of a few int8 pixels to compare against.
OLMOEARTH_RUN_ARTIFACT = (
    Path(__file__).parent / "data" / "olmoearth_run_embedding_pca.pkl"
)
OLMOEARTH_RUN_EXPECTED = (
    Path(__file__).parent / "data" / "olmoearth_run_embedding_pca_expected.npz"
)


def test_olmoearth_run_artifact_projects_like_sklearn() -> None:
    """The same artifact must give the same components as olmoearth_run computes."""
    artifact = pca.PcaArtifact.load(str(OLMOEARTH_RUN_ARTIFACT))
    expected = np.load(OLMOEARTH_RUN_EXPECTED)
    ours = (
        expected["pixels"].astype(np.float32) - artifact.mean
    ) @ artifact.components.T
    np.testing.assert_allclose(ours, expected["transformed"], atol=1e-3)
    np.testing.assert_array_equal(artifact.norm_bounds, expected["norm_bounds"])


def test_olmoearth_run_artifact_is_applied_to_int8_values() -> None:
    """olmoearth_run fits on int8 values, so render must not dequantize first."""
    artifact = pca.PcaArtifact.load(str(OLMOEARTH_RUN_ARTIFACT))
    expected = np.load(OLMOEARTH_RUN_EXPECTED)
    pixels = expected["pixels"]  # (n, bands)
    block = pixels.T.reshape(pixels.shape[1], 1, pixels.shape[0])
    rgb = pca.project_to_rgb(block, artifact)

    low, high = expected["norm_bounds"]
    scaled = (expected["transformed"] - low) / (high - low)
    want = np.clip(np.rint(scaled * 254.0) + 1.0, 1.0, 255.0).astype(np.uint8)
    np.testing.assert_array_equal(rgb[:, 0, :], want.T)


def test_olmoearth_run_artifact_records_its_provenance() -> None:
    """Annotate copies this onto the store, so it must say where the basis came from."""
    artifact = pca.PcaArtifact.load(str(OLMOEARTH_RUN_ARTIFACT))
    assert artifact.metadata["geoemb:pca_source_artifact"] == str(
        OLMOEARTH_RUN_ARTIFACT
    )
    assert artifact.metadata["geoemb:pca_dimensions"] == 128
    assert artifact.metadata["geoemb:pca_input_space"] == "int8"


def test_olmoearth_run_loader_refuses_other_classes(tmp_path: Path) -> None:
    """A pickle can run code on load, so only the artifact's own classes are allowed."""
    import collections
    import pickle

    path = tmp_path / "embedding_pca.pkl"
    path.write_bytes(pickle.dumps(collections.OrderedDict(a=1)))
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        pca.PcaArtifact.load(str(path))
