"""Unit tests for the three-step embedding flow's render step."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from rslp.large_scale_embeddings import pca, render_pca
from rslp.large_scale_embeddings import zarr_store as zs
from rslp.large_scale_embeddings.model import quantize_embeddings
from rslp.large_scale_embeddings.tiling import get_zone_grid

RESOLUTION = 10
TILE_SIZE = 2048
SHARD = 512
CHUNK = 256
DIMS = 8
ZONE = 10
EPSG = 32610
MODEL_URL = "https://huggingface.co/allenai/OlmoEarth-v1_2-Small"
SOURCE_DATA = ["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]
PROJECTION = {
    "crs": f"EPSG:{EPSG}",
    "x_resolution": RESOLUTION,
    "y_resolution": -RESOLUTION,
}


def _unit_vectors(rng: np.random.Generator, n: int, dims: int) -> np.ndarray:
    scale = np.linspace(1.0, 0.05, dims)
    x = rng.normal(size=(n, dims)) * scale
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _build_run(tmp_path: Path, n_windows: int = 3, nodata_last: bool = False) -> dict:
    """Run step 1 and step 2 for a small store, returning the paths involved."""
    store_path = str(tmp_path / "s2.zarr")
    zs.init_store(
        store_path=store_path,
        zone_numbers=[ZONE],
        years=[2024],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=RESOLUTION,
        tile_size=TILE_SIZE,
        dimensions=DIMS,
        chunk_size=CHUNK,
        shard_size=SHARD,
    )
    _, (origin_x, origin_y), _ = get_zone_grid(ZONE, RESOLUTION, TILE_SIZE)
    rng = np.random.default_rng(3)

    written = []
    for i in range(n_windows):
        bx0 = origin_x + i * SHARD
        by0 = origin_y + 2 * SHARD
        if nodata_last and i == n_windows - 1:
            block = np.full((DIMS, SHARD, SHARD), zs.NODATA_VALUE, dtype=np.int8)
        else:
            floats = _unit_vectors(rng, SHARD * SHARD, DIMS)
            block = np.asarray(quantize_embeddings(torch.from_numpy(floats))).T.reshape(
                DIMS, SHARD, SHARD
            )
        zs.write_window_region(
            store_path, ZONE, (bx0, by0, bx0 + SHARD, by0 + SHARD), 0, block
        )
        written.append([bx0, by0])

    # One step 1 marker naming every window, as a single block would produce.
    source_completed = tmp_path / "s2_2024_completed"
    source_completed.mkdir()
    source_marker = (
        source_completed / f"EPSG:{EPSG}_{written[0][0]}_{written[0][1]}.json"
    )
    with source_marker.open("w") as f:
        json.dump(
            {
                "projection": PROJECTION,
                "bounds": [
                    written[0][0],
                    written[0][1],
                    written[-1][0] + SHARD,
                    written[0][1] + SHARD,
                ],
                "time_index": 0,
                "written": written,
                "skipped_no_data": [],
                "skipped_longitude": [],
                "num_filtered_crops": 0,
            },
            f,
        )

    artifact_path = str(tmp_path / "pca_artifact")
    pca.fit_pca(
        store_path=store_path,
        completed_paths=[str(source_completed)],
        artifact_path=artifact_path,
        blocks_per_zone=3,
        pixels_per_block=4_000,
        chunk_size=CHUNK,
        seed=5,
    )
    return {
        "store_path": store_path,
        "source_completed": str(source_completed),
        "source_marker": str(source_marker),
        "artifact_path": artifact_path,
        "completed_path": str(tmp_path / "pca_2024_completed"),
        "written": written,
        "origin": (origin_x, origin_y),
    }


def test_render_writes_rgb_for_every_written_window(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    render_pca.render_pca_pipeline(
        store_path=run["store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
    )

    group = zarr.open_group(store=run["store_path"], path=f"utm{ZONE}", mode="r")
    origin_x, origin_y = run["origin"]
    for bx0, by0 in run["written"]:
        col0, row0 = bx0 - origin_x, by0 - origin_y
        got = np.asarray(
            group[zs.PCA_ARRAY][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD]
        )
        assert got.shape == (zs.PCA_BANDS, SHARD, SHARD)
        assert got.min() >= 1  # fully valid windows never emit the nodata sentinel
        assert len(np.unique(got)) > 50

    marker = json.loads(
        (Path(run["completed_path"]) / Path(run["source_marker"]).name).read_text()
    )
    assert marker["rendered"] == run["written"]
    assert marker["skipped_empty"] == []
    assert marker["artifact_path"] == run["artifact_path"]


def test_render_is_idempotent(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    kwargs = dict(
        store_path=run["store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
    )
    render_pca.render_pca_pipeline(**kwargs)
    marker_path = Path(run["completed_path"]) / Path(run["source_marker"]).name
    first = marker_path.read_text()

    # Corrupt the array, then re-run: the marker short-circuits so nothing is rewritten.
    group = zarr.open_group(store=run["store_path"], path=f"utm{ZONE}", mode="r+")
    origin_x, origin_y = run["origin"]
    bx0, by0 = run["written"][0]
    col0, row0 = bx0 - origin_x, by0 - origin_y
    group[zs.PCA_ARRAY][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD] = 7

    render_pca.render_pca_pipeline(**kwargs)
    assert marker_path.read_text() == first
    still = np.asarray(
        zarr.open_group(store=run["store_path"], path=f"utm{ZONE}", mode="r")[
            zs.PCA_ARRAY
        ][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD]
    )
    assert (still == 7).all()


def test_render_skips_all_nodata_windows(tmp_path: Path) -> None:
    run = _build_run(tmp_path, nodata_last=True)
    render_pca.render_pca_pipeline(
        store_path=run["store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
    )
    marker = json.loads(
        (Path(run["completed_path"]) / Path(run["source_marker"]).name).read_text()
    )
    assert marker["skipped_empty"] == [run["written"][-1]]
    assert len(marker["rendered"]) == len(run["written"]) - 1


def test_get_render_jobs_excludes_completed(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    jobs = render_pca.get_render_jobs(
        store_path=run["store_path"],
        artifact_path=run["artifact_path"],
        source_completed_paths=[run["source_completed"]],
        completed_path=run["completed_path"],
    )
    assert len(jobs) == 1
    assert "--source_marker" in jobs[0]

    render_pca.render_pca_pipeline(
        store_path=run["store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
    )
    assert (
        render_pca.get_render_jobs(
            store_path=run["store_path"],
            artifact_path=run["artifact_path"],
            source_completed_paths=[run["source_completed"]],
            completed_path=run["completed_path"],
        )
        == []
    )


def test_render_without_artifact_is_actionable(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="fit_pca"):
        render_pca.render_pca_pipeline(
            store_path=run["store_path"],
            artifact_path=str(tmp_path / "absent"),
            source_marker=run["source_marker"],
            completed_path=run["completed_path"],
        )


def test_annotate_pca_array_records_provenance(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    render_pca.annotate_pca_array(run["store_path"], run["artifact_path"])

    group = zarr.open_group(store=run["store_path"], path=f"utm{ZONE}", mode="r")
    attrs = dict(group[zs.PCA_ARRAY].attrs)
    assert attrs["geoemb:pca_components"] == pca.PCA_N_COMPONENTS
    assert attrs["geoemb:pca_artifact_path"] == run["artifact_path"]
    assert attrs["geoemb:pca_source_zones"] == [ZONE]
    assert "do not use these bands as features" in attrs["geoemb:pca_note"]
