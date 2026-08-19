"""Unit tests for rslp.large_scale_embeddings.pca and the pca_rgb store array."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from rslp.large_scale_embeddings import pca
from rslp.large_scale_embeddings import zarr_store as zs
from rslp.large_scale_embeddings.model import quantize_embeddings
from rslp.large_scale_embeddings.tiling import get_zone_grid

RESOLUTION = 10
TILE_SIZE = 2048
SHARD = 512
CHUNK = 256
DIMS = 8
MODEL_URL = "https://huggingface.co/allenai/OlmoEarth-v1_2-Small"
SOURCE_DATA = ["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]


def _random_embeddings(rng: np.random.Generator, n: int, dims: int) -> np.ndarray:
    """Build L2-normalized float vectors with anisotropic structure to find."""
    scale = np.linspace(1.0, 0.05, dims)
    x = rng.normal(size=(n, dims)) * scale
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _artifact(rng: np.random.Generator, dims: int = DIMS) -> pca.PcaArtifact:
    samples = _random_embeddings(rng, 5000, dims).astype(np.float32)
    mean, comps, evr = pca.fit_basis(samples)
    bounds = pca.compute_norm_bounds(samples, mean, comps)
    return pca.PcaArtifact(
        mean=mean, components=comps, norm_bounds=bounds, explained_variance_ratio=evr
    )


def test_fit_basis_recovers_leading_directions() -> None:
    rng = np.random.default_rng(0)
    samples = _random_embeddings(rng, 8000, DIMS).astype(np.float32)
    mean, comps, evr = pca.fit_basis(samples)

    assert comps.shape == (pca.PCA_N_COMPONENTS, DIMS)
    assert mean.shape == (DIMS,)
    # Components are orthonormal.
    np.testing.assert_allclose(comps @ comps.T, np.eye(3), atol=1e-4)
    # Explained variance is sorted and sums to at most 1.
    assert evr[0] >= evr[1] >= evr[2]
    assert evr.sum() <= 1.0 + 1e-5
    # The construction puts most variance in the first dimensions, so 3 components
    # should capture a clear majority.
    assert evr[:3].sum() > 0.5


def test_fit_basis_rejects_degenerate_input() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="more pixels than dimensions"):
        pca.fit_basis(_random_embeddings(rng, 4, DIMS).astype(np.float32))
    with pytest.raises(ValueError, match="2-D"):
        pca.fit_basis(np.zeros((2, 3, 4), dtype=np.float32))


def test_norm_bounds_bracket_the_sample() -> None:
    rng = np.random.default_rng(2)
    samples = _random_embeddings(rng, 4000, DIMS).astype(np.float32)
    mean, comps, _ = pca.fit_basis(samples)
    bounds = pca.compute_norm_bounds(samples, mean, comps)

    assert bounds.shape == (2, pca.PCA_N_COMPONENTS)
    assert np.all(bounds[1] > bounds[0])
    transformed = (samples - mean) @ comps.T
    # p2/p98 by construction: about 2% of mass falls below the low bound.
    below = (transformed < bounds[0]).mean(axis=0)
    np.testing.assert_allclose(below, 0.02, atol=0.005)


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


def test_artifact_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    artifact = _artifact(rng)
    artifact.metadata = pca.build_metadata(
        "gs://bucket/store.zarr",
        [10, 18],
        1234,
        7,
        42,
        artifact.explained_variance_ratio,
    )
    artifact.save(str(tmp_path / "pca"))
    loaded = pca.PcaArtifact.load(str(tmp_path / "pca"))

    np.testing.assert_array_equal(loaded.mean, artifact.mean)
    np.testing.assert_array_equal(loaded.components, artifact.components)
    np.testing.assert_array_equal(loaded.norm_bounds, artifact.norm_bounds)
    assert loaded.metadata["geoemb:pca_source_zones"] == [10, 18]
    assert loaded.metadata["geoemb:pca_components"] == pca.PCA_N_COMPONENTS


def test_artifact_load_missing_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fit_pca"):
        pca.PcaArtifact.load(str(tmp_path / "absent"))


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


def _init_small_store(path: Path, create_pca: bool = True) -> str:
    store_path = str(path / "s2.zarr")
    zs.init_store(
        store_path=store_path,
        zone_numbers=[10],
        years=[2024],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=RESOLUTION,
        tile_size=TILE_SIZE,
        dimensions=DIMS,
        chunk_size=CHUNK,
        shard_size=SHARD,
        create_pca_array=create_pca,
    )
    return store_path


def test_init_store_creates_pca_array(tmp_path: Path) -> None:
    store_path = _init_small_store(tmp_path)
    group = zarr.open_group(store=store_path, path="utm10", mode="r")

    assert zs.PCA_ARRAY in group
    arr = group[zs.PCA_ARRAY]
    assert arr.shape == (1, zs.PCA_BANDS, *group[zs.EMBEDDINGS_ARRAY].shape[2:])
    assert arr.dtype == np.uint8
    assert arr.fill_value == zs.PCA_NODATA_VALUE
    # Shares the embedding shard grid so one worker owns both objects per window.
    assert arr.shards[2:] == group[zs.EMBEDDINGS_ARRAY].shards[2:]
    assert arr.chunks[2:] == group[zs.EMBEDDINGS_ARRAY].chunks[2:]


def test_init_store_can_skip_pca_array(tmp_path: Path) -> None:
    store_path = _init_small_store(tmp_path, create_pca=False)
    group = zarr.open_group(store=store_path, path="utm10", mode="r")
    assert zs.PCA_ARRAY not in group


def test_write_pca_window_roundtrip(tmp_path: Path) -> None:
    store_path = _init_small_store(tmp_path)
    _, (origin_x, origin_y), _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    bx0 = origin_x + 3 * SHARD
    by0 = origin_y + 5 * SHARD
    bounds = (bx0, by0, bx0 + SHARD, by0 + SHARD)

    rng = np.random.default_rng(7)
    rgb = rng.integers(1, 256, size=(zs.PCA_BANDS, SHARD, SHARD), dtype=np.uint8)
    zs.write_pca_window_region(store_path, 10, bounds, 0, rgb)

    group = zarr.open_group(store=store_path, path="utm10", mode="r")
    col0 = bx0 - origin_x
    row0 = by0 - origin_y
    got = np.asarray(
        group[zs.PCA_ARRAY][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD]
    )
    np.testing.assert_array_equal(got, rgb)
    # Neighbouring region is untouched.
    other = np.asarray(group[zs.PCA_ARRAY][0, :, row0 + SHARD : row0 + 2 * SHARD, col0])
    assert not other.any()


def test_write_pca_window_rejects_wrong_band_count(tmp_path: Path) -> None:
    store_path = _init_small_store(tmp_path)
    _, (origin_x, origin_y), _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    bounds = (origin_x, origin_y, origin_x + SHARD, origin_y + SHARD)
    bad = np.zeros((zs.PCA_BANDS + 1, SHARD, SHARD), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected 3 bands"):
        zs.write_pca_window_region(store_path, 10, bounds, 0, bad)


def test_write_pca_window_without_array_is_actionable(tmp_path: Path) -> None:
    store_path = _init_small_store(tmp_path, create_pca=False)
    _, (origin_x, origin_y), _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    bounds = (origin_x, origin_y, origin_x + SHARD, origin_y + SHARD)
    rgb = np.zeros((zs.PCA_BANDS, SHARD, SHARD), dtype=np.uint8)
    with pytest.raises(KeyError, match="create_pca_array=False"):
        zs.write_pca_window_region(store_path, 10, bounds, 0, rgb)


def _write_marker(
    completed: Path,
    epsg: int,
    x: int,
    y: int,
    written: list[list[int]],
    time_index: int,
) -> None:
    """Write a completion marker in the shape the prediction pipeline produces."""
    completed.mkdir(parents=True, exist_ok=True)
    fname = completed / f"EPSG:{epsg}_{x}_{y}.json"
    with fname.open("w") as f:
        import json

        json.dump(
            {
                "bounds": [x, y, x + SHARD, y + SHARD],
                "time_range": [
                    "2024-01-01T00:00:00+00:00",
                    "2024-01-01T00:00:00+00:00",
                ],
                "time_index": time_index,
                "written": written,
                "skipped_no_data": [],
                "skipped_longitude": [],
                "num_filtered_crops": 0,
            },
            f,
        )


def test_fit_pca_end_to_end(tmp_path: Path) -> None:
    """Write real blocks, fit from the markers, then render RGB through the artifact."""
    store_path = _init_small_store(tmp_path)
    _, (origin_x, origin_y), _ = get_zone_grid(10, RESOLUTION, TILE_SIZE)
    rng = np.random.default_rng(11)

    written: list[list[int]] = []
    for i in range(3):
        bx0 = origin_x + i * SHARD
        by0 = origin_y + 2 * SHARD
        floats = _random_embeddings(rng, SHARD * SHARD, DIMS).astype(np.float32)
        quant = np.asarray(quantize_embeddings(torch.from_numpy(floats)))
        block = quant.T.reshape(DIMS, SHARD, SHARD)
        zs.write_window_region(
            store_path, 10, (bx0, by0, bx0 + SHARD, by0 + SHARD), 0, block
        )
        written.append([bx0, by0])

    completed = tmp_path / "completed_2024"
    for bx0, by0 in written:
        _write_marker(completed, 32610, bx0, by0, [[bx0, by0]], 0)

    artifact_path = str(tmp_path / "pca_artifact")
    pca.fit_pca(
        store_path=store_path,
        completed_paths=[str(completed)],
        artifact_path=artifact_path,
        blocks_per_zone=3,
        pixels_per_block=5_000,
        chunk_size=CHUNK,
        seed=7,
    )

    artifact = pca.PcaArtifact.load(artifact_path)
    assert artifact.mean.shape == (DIMS,)
    assert artifact.components.shape == (pca.PCA_N_COMPONENTS, DIMS)
    assert artifact.metadata["geoemb:pca_source_zones"] == [10]
    assert artifact.metadata["geoemb:pca_fit_blocks"] == 3
    assert artifact.metadata["geoemb:pca_dimensions"] == DIMS
    assert artifact.metadata["geoemb:pca_fit_seed"] == 7

    # Render one of the written windows and store it, then read it back.
    bx0, by0 = written[0]
    group = zarr.open_group(store=store_path, path="utm10", mode="r")
    col0, row0 = bx0 - origin_x, by0 - origin_y
    block = np.asarray(
        group[zs.EMBEDDINGS_ARRAY][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD]
    )
    rgb = pca.project_to_rgb(block, artifact)
    zs.write_pca_window_region(
        store_path, 10, (bx0, by0, bx0 + SHARD, by0 + SHARD), 0, rgb
    )

    got = np.asarray(
        zarr.open_group(store=store_path, path="utm10", mode="r")[zs.PCA_ARRAY][
            0, :, row0 : row0 + SHARD, col0 : col0 + SHARD
        ]
    )
    np.testing.assert_array_equal(got, rgb)
    assert got.min() >= 1  # this window is fully valid, so nothing is nodata
    # A real basis spreads the data across the byte range rather than collapsing it.
    assert got.max() > 200
    assert len(np.unique(got)) > 50


def test_fit_pca_without_markers_is_actionable(tmp_path: Path) -> None:
    store_path = _init_small_store(tmp_path)
    with pytest.raises(ValueError, match="no completion markers"):
        pca.fit_pca(
            store_path=store_path,
            completed_paths=[str(tmp_path / "absent")],
            artifact_path=str(tmp_path / "artifact"),
        )
