import numpy as np
import pytest

from rslp.large_scale_embeddings.tools.sanity_checks import (
    TileResult,
    classification_metrics,
    cross_tile_duplicates,
    dem_tile,
    evaluate_gates,
    knn_predict,
    participation_ratio,
    region_of,
    ridge_oof_r2,
    silhouette_cosine,
    stage1_counts,
    uniformity,
    worldcover_tile,
)


def test_participation_ratio_isotropic_vs_rank_one() -> None:
    rng = np.random.default_rng(0)
    iso = rng.normal(size=(5000, 16))
    pr_iso, dims_iso = participation_ratio(iso)
    assert pr_iso > 0.9
    assert dims_iso >= 13

    line = rng.normal(size=(5000, 1)) * np.ones((1, 16)) + 1e-3 * iso
    pr_line, dims_line = participation_ratio(line)
    assert pr_line < 0.1
    assert dims_line == 1


def test_uniformity_lower_when_spread() -> None:
    rng = np.random.default_rng(0)
    spread = rng.normal(size=(500, 8))
    clumped = np.ones((500, 8)) + 0.01 * spread
    assert uniformity(spread) < uniformity(clumped)


def test_knn_separates_clusters() -> None:
    rng = np.random.default_rng(0)
    centers = np.eye(4) * 5
    y = rng.integers(0, 4, 400)
    x = centers[y] + rng.normal(size=(400, 4)) * 0.3
    pred = knn_predict(x[:300], y[:300], x[300:], 4)
    assert (pred == y[300:]).mean() > 0.95


def test_ridge_recovers_linear_target() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 10))
    y = x @ rng.normal(size=10) + 0.01 * rng.normal(size=1000)
    folds = np.arange(1000) % 5
    assert ridge_oof_r2(x, y, folds) > 0.99


def test_silhouette_sign() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    x = np.where(y[:, None] == 0, [1.0, 0.0], [0.0, 1.0]) + 0.05 * rng.normal(
        size=(200, 2)
    )
    assert silhouette_cosine(x, y) > 0.8
    assert silhouette_cosine(x, rng.permutation(y)) < 0.1


def test_classification_metrics_balanced() -> None:
    y = np.array([0, 0, 0, 1])
    pred = np.array([0, 0, 0, 0])
    m = classification_metrics(y, pred, 2)
    assert m["accuracy"] == 0.75
    assert m["balanced_accuracy"] == 0.5
    assert m["confusion"] == [[3, 0], [1, 0]]


def test_cross_tile_duplicates_ignores_same_tile() -> None:
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    best, count = cross_tile_duplicates(x, np.array([0, 0, 1, 2]))
    # Rows 0 and 1 share a tile; rows 0/1 and 3 are the cross-tile duplicates.
    assert best == pytest.approx(1.0)
    assert count == 2


def test_stage1_counts() -> None:
    q = np.ones((4, 2, 2), dtype=np.int8)
    q[:, 0, 0] = -128
    q[1, 0, 1] = -128
    q[:, 1, 0] = 0
    q[0, 1, 1] = 127
    res = TileResult(zone="utm01", y0=0, x0=0)
    valid = stage1_counts(q, res)
    assert valid.tolist() == [[False, False], [True, True]]
    assert res.partial_nodata == 1
    assert res.all_zero == 1
    assert res.saturated == 1
    assert res.values == 8


def test_tile_names() -> None:
    assert worldcover_tile(-122.4, 37.8) == "N36W123"
    assert worldcover_tile(151.2, -33.9) == "S36E150"
    assert dem_tile(-122.4, 37.8) == "Copernicus_DSM_COG_10_N37_00_W123_00_DEM"
    assert dem_tile(151.2, -33.9) == "Copernicus_DSM_COG_10_S34_00_E151_00_DEM"


def test_regions_partition() -> None:
    lon, lat = np.meshgrid(np.arange(-180, 180, 5.0), np.arange(-60, 85, 5.0))
    regions = region_of(lon.ravel(), lat.ravel())
    assert (regions != "").all()
    assert len(set(regions)) == 9


def test_s2_gates_only_when_measured() -> None:
    assert "reconstruction_r2" not in evaluate_gates({})
    gates = evaluate_gates(
        {"nodata_with_imagery": 0, "data_without_imagery": 3, "reconstruction_r2": 0.8}
    )
    assert gates["nodata_with_imagery"]["pass"]
    assert not gates["data_without_imagery"]["pass"]
    assert gates["reconstruction_r2"]["pass"]
