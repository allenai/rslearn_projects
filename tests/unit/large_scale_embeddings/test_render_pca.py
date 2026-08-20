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
MAX_LEVEL = 2
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
    pca_store_path = str(tmp_path / "pca_v1.zarr")
    zs.init_pca_store(
        pca_store_path=pca_store_path,
        zone_numbers=[ZONE],
        years=[2024],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=RESOLUTION,
        tile_size=TILE_SIZE,
        chunk_size=CHUNK,
        shard_size=SHARD,
        max_level=MAX_LEVEL,
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
        "pca_store_path": pca_store_path,
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
        pca_store_path=run["pca_store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
        max_level=MAX_LEVEL,
    )

    group = zarr.open_group(store=run["pca_store_path"], path=f"utm{ZONE}", mode="r")
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
        (
            Path(run["completed_path"])
            / render_pca.pca_marker_name(Path(run["source_marker"]))
        ).read_text()
    )
    assert marker["rendered"] == run["written"]
    assert marker["skipped_empty"] == []
    assert marker["artifact_path"] == run["artifact_path"]


def test_render_is_idempotent(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    kwargs = dict(
        store_path=run["store_path"],
        pca_store_path=run["pca_store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
        max_level=MAX_LEVEL,
    )
    render_pca.render_pca_pipeline(**kwargs)
    marker_path = Path(run["completed_path"]) / render_pca.pca_marker_name(
        Path(run["source_marker"])
    )
    first = marker_path.read_text()

    # Corrupt the array, then re-run: the marker short-circuits so nothing is rewritten.
    group = zarr.open_group(store=run["pca_store_path"], path=f"utm{ZONE}", mode="r+")
    origin_x, origin_y = run["origin"]
    bx0, by0 = run["written"][0]
    col0, row0 = bx0 - origin_x, by0 - origin_y
    group[zs.PCA_ARRAY][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD] = 7

    render_pca.render_pca_pipeline(**kwargs)
    assert marker_path.read_text() == first
    still = np.asarray(
        zarr.open_group(store=run["pca_store_path"], path=f"utm{ZONE}", mode="r")[
            zs.PCA_ARRAY
        ][0, :, row0 : row0 + SHARD, col0 : col0 + SHARD]
    )
    assert (still == 7).all()


def test_render_skips_all_nodata_windows(tmp_path: Path) -> None:
    run = _build_run(tmp_path, nodata_last=True)
    render_pca.render_pca_pipeline(
        store_path=run["store_path"],
        pca_store_path=run["pca_store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
        max_level=MAX_LEVEL,
    )
    marker = json.loads(
        (
            Path(run["completed_path"])
            / render_pca.pca_marker_name(Path(run["source_marker"]))
        ).read_text()
    )
    assert marker["skipped_empty"] == [run["written"][-1]]
    assert len(marker["rendered"]) == len(run["written"]) - 1


def test_get_render_jobs_excludes_completed(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    jobs = render_pca.get_render_jobs(
        store_path=run["store_path"],
        artifact_path=run["artifact_path"],
        pca_store_path=run["pca_store_path"],
        source_completed_paths=[run["source_completed"]],
        completed_path=run["completed_path"],
    )
    assert len(jobs) == 1
    assert "--source_marker" in jobs[0]

    render_pca.render_pca_pipeline(
        store_path=run["store_path"],
        pca_store_path=run["pca_store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
        max_level=MAX_LEVEL,
    )
    assert (
        render_pca.get_render_jobs(
            store_path=run["store_path"],
            pca_store_path=run["pca_store_path"],
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
            pca_store_path=run["pca_store_path"],
            artifact_path=str(tmp_path / "absent"),
            source_marker=run["source_marker"],
            completed_path=run["completed_path"],
        )


def test_annotate_pca_store_records_provenance(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    render_pca.annotate_pca_store(
        run["pca_store_path"], run["artifact_path"], max_level=MAX_LEVEL
    )

    group = zarr.open_group(store=run["pca_store_path"], path=f"utm{ZONE}", mode="r")
    attrs = dict(group[zs.PCA_ARRAY].attrs)
    assert attrs["geoemb:pca_components"] == pca.PCA_N_COMPONENTS
    assert attrs["geoemb:pca_artifact_path"] == run["artifact_path"]
    assert attrs["geoemb:pca_source_zones"] == [ZONE]
    assert "do not use these bands as features" in attrs["geoemb:pca_note"]

    # Every level carries the provenance, since any of them can be served.
    for level in range(MAX_LEVEL + 1):
        name = zs.pca_level_array_name(level)
        assert name in group, name
        assert (
            dict(group[name].attrs)["geoemb:pca_artifact_path"] == run["artifact_path"]
        )


def test_render_writes_every_pyramid_level(tmp_path: Path) -> None:
    """A window must land at every level, each covering the same ground."""
    run = _build_run(tmp_path)
    render_pca.render_pca_pipeline(
        store_path=run["store_path"],
        pca_store_path=run["pca_store_path"],
        artifact_path=run["artifact_path"],
        source_marker=run["source_marker"],
        completed_path=run["completed_path"],
        max_level=MAX_LEVEL,
    )

    group = zarr.open_group(store=run["pca_store_path"], path=f"utm{ZONE}", mode="r")
    origin_x, origin_y = run["origin"]
    bx0, by0 = run["written"][0]
    col0, row0 = bx0 - origin_x, by0 - origin_y

    for level in range(MAX_LEVEL + 1):
        factor = 2**level
        name = zs.pca_level_array_name(level)
        size = SHARD // factor
        got = np.asarray(
            group[name][
                0,
                :,
                row0 // factor : row0 // factor + size,
                col0 // factor : col0 // factor + size,
            ]
        )
        assert got.shape == (zs.PCA_BANDS, size, size), name
        # Fully-valid window, so no level may emit the reserved nodata value.
        assert got.min() >= 1, name
        # One shard per window footprint at every level keeps writes lock-free.
        assert group[name].shards[2] == size, name

    marker = json.loads(
        (
            Path(run["completed_path"])
            / render_pca.pca_marker_name(Path(run["source_marker"]))
        ).read_text()
    )
    assert marker["max_level"] == MAX_LEVEL
    assert marker["levels"] == [
        zs.pca_level_array_name(k) for k in range(MAX_LEVEL + 1)
    ]
    assert marker["pca_store_path"] == run["pca_store_path"]


def test_init_pca_store_rejects_indivisible_max_level(tmp_path: Path) -> None:
    """Every level must keep one shard per window, so the shard must divide down."""
    with pytest.raises(ValueError, match="divisible by 2\\*\\*max_level"):
        zs.init_pca_store(
            pca_store_path=str(tmp_path / "bad.zarr"),
            zone_numbers=[ZONE],
            years=[2024],
            model_url=MODEL_URL,
            source_data=SOURCE_DATA,
            resolution=RESOLUTION,
            tile_size=TILE_SIZE,
            chunk_size=CHUNK,
            shard_size=512,
            max_level=10,
        )


def test_init_pca_store_records_multiscales(tmp_path: Path) -> None:
    path = str(tmp_path / "pca_v1.zarr")
    zs.init_pca_store(
        pca_store_path=path,
        zone_numbers=[ZONE],
        years=[2024],
        model_url=MODEL_URL,
        source_data=SOURCE_DATA,
        resolution=RESOLUTION,
        tile_size=TILE_SIZE,
        chunk_size=CHUNK,
        shard_size=SHARD,
        max_level=MAX_LEVEL,
    )
    root = zarr.open_group(store=path, mode="r")
    scales = dict(root.attrs)["geoemb:multiscales"]
    assert [s["factor"] for s in scales] == [2**k for k in range(MAX_LEVEL + 1)]
    assert [s["resolution"] for s in scales] == [
        RESOLUTION * 2**k for k in range(MAX_LEVEL + 1)
    ]
    # The embeddings store is untouched by this: no pca arrays leak into it.
    assert "geoemb:multiscales" in dict(root[f"utm{ZONE}"].attrs)


def test_two_years_do_not_collide_on_one_marker(tmp_path: Path) -> None:
    """Step 1 gives a block the same file name in every year's directory.

    Keying step 3's marker on the file name alone therefore collapses all years onto one
    marker: the first year written makes the rest look already done and they are silently
    skipped. This asserts each year gets its own marker and its own rendered output.
    """
    run = _build_run(tmp_path)
    source_2022 = Path(run["source_marker"])

    # A second year, same block, identical file name in a sibling directory.
    completed_2023 = tmp_path / "s2_2023_completed"
    completed_2023.mkdir()
    source_2023 = completed_2023 / source_2022.name
    marker = json.loads(source_2022.read_text())
    marker["time_index"] = 0  # single-year store in this fixture
    source_2023.write_text(json.dumps(marker))
    assert source_2022.name == source_2023.name, "fixture must reuse the name"

    for src in (source_2022, source_2023):
        render_pca.render_pca_pipeline(
            store_path=run["store_path"],
            pca_store_path=run["pca_store_path"],
            artifact_path=run["artifact_path"],
            source_marker=str(src),
            completed_path=run["completed_path"],
            max_level=MAX_LEVEL,
        )

    written = sorted(p.name for p in Path(run["completed_path"]).iterdir())
    assert len(written) == 2, f"expected one marker per year, got {written}"

    # Each marker points back at its own year's source.
    sources = {
        json.loads((Path(run["completed_path"]) / n).read_text())["source_marker"]
        for n in written
    }
    assert sources == {str(source_2022), str(source_2023)}

    # And enumeration considers both done rather than re-offering them.
    remaining = render_pca.get_render_jobs(
        store_path=run["store_path"],
        pca_store_path=run["pca_store_path"],
        artifact_path=run["artifact_path"],
        source_completed_paths=[run["source_completed"], str(completed_2023)],
        completed_path=run["completed_path"],
    )
    assert remaining == [], f"expected nothing left to render, got {len(remaining)}"


def test_annotation_survives_consolidated_metadata(tmp_path: Path) -> None:
    """Annotations must be visible to a default reader, not just on the raw arrays.

    init_pca_store consolidates metadata at creation and readers prefer that snapshot,
    so writing array attrs without re-consolidating leaves the provenance invisible to
    everyone who opens the store normally.
    """
    run = _build_run(tmp_path)
    render_pca.annotate_pca_store(
        run["pca_store_path"], run["artifact_path"], max_level=MAX_LEVEL
    )

    # The default read path is the one that matters.
    group = zarr.open_group(store=run["pca_store_path"], mode="r")[f"utm{ZONE}"]
    for name in (zs.pca_level_array_name(k) for k in range(MAX_LEVEL + 1)):
        attrs = dict(group[name].attrs)
        assert attrs.get("geoemb:pca_components") == pca.PCA_N_COMPONENTS, name
        assert "do not use these bands as features" in attrs["geoemb:pca_note"], name
