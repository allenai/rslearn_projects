"""Sanity checks for the embeddings store.

Inspired by LGND's sanity checks for geospatial embeddings
(https://lgnd.ai/resources/sanity-checks-for-geospatial-embeddings), which are
written for chip embeddings. This adapts them to a pixel store:

- Neighboring pixels are near-identical by construction, so duplicates are only
  sought between different 256 px tiles.
- Values are int8, so "finite" becomes no partial nodata, no all-zero vectors, and
  little saturation at +-127.
- Vectors are not L2-normalized, so norm outliers use the raw dequantized length.
- Years exist, so adjacent years are compared at the same pixels.

Stages:
    1. pipeline: valid values, duplicates. (Baseline agreement needs a GPU rerun of
       fixed windows and is not done here.)
    2. geometry: participation ratio, dead dimensions, redundancy, uniformity, norm
       outliers, on all 128 dimensions and on the 64-dim Matryoshka prefix.
    3. semantics: kNN land cover against ESA WorldCover 2021, and linear probes for
       elevation (Copernicus DEM) and climate (WorldClim 2.1) on land pixels. Train and test folds
       are grouped by tile, and tiles are scattered worldwide.
    4. transfer: leave-one-region-out land cover against a random tile split.

With --s2_tiles, stage 3 also probes reconstruction: ridge R^2 from the embedding to
each band's annual median Sentinel-2 reflectance, read through the workers' own
Sentinel-2 layer. That needs olmoearth_run and a datasets service token, so it runs on
the worker image.

Labels are 2021 while the main sample is 2025 (the only full-world year), which
costs land cover some accuracy wherever the ground changed.

Usage:
    python -m rslp.large_scale_embeddings.tools.sanity_checks --output results.json
"""

import argparse
import hashlib
import io
import json
import os
import random
import subprocess  # nosec: fixed argv, no shell
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import rasterio
import requests
import zarr
from fsspec.core import url_to_fs
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.windows import Window as RasterWindow
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.large_scale_embeddings.model import (
    NODATA_VALUE,
    QUANTIZE_POWER,
    QUANTIZE_SCALE,
)
from rslp.large_scale_embeddings.predict_pipeline import (
    DATASET_CONFIG_FNAME,
    MATERIALIZE_PIPELINE_ARGS,
    PREDICTION_GROUP,
    SENTINEL2_LAYER,
    EmbeddingInputs,
)
from rslp.log_utils import get_logger
from rslp.utils.rslearn import materialize_dataset

logger = get_logger(__name__)

# The public v1.3 store on Source Cooperative, read anonymously through its S3 API.
DEFAULT_STORE = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
DEFAULT_STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {"endpoint_url": "https://data.source.coop"},
}
TILE = 256
SHARD = 2048
PREFIX_DIMS = 64

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"
WORLDCLIM_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
)
# BIO1 is annual mean temperature, BIO12 annual precipitation.
WORLDCLIM_VARS = {"temperature": 1, "precipitation": 12}

WORLDCOVER_CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}

# Gates. LGND's are kept where they discriminate; where our pilot sat far inside
# theirs, ours are tightened to catch a regression rather than only a collapse.
GATES = {
    "partial_nodata_pixels": ("==", 0),
    "all_zero_pixels": ("==", 0),
    "saturated_fraction": ("<=", 0.01),
    "cross_tile_pairs_ge_0_99999": ("==", 0),
    "duplicate_tiles": ("==", 0),
    "duplicate_tiles_across_years": ("==", 0),
    "participation_ratio": (">=", 0.08),
    "dead_dims": ("==", 0),
    "mean_abs_offdiag_corr": ("<=", 0.3),
    "uniformity": ("<=", -1.5),
    "norm_outlier_fraction": ("<=", 0.005),
    "year_cosine_median_min": (">=", 0.9),
    "knn_accuracy": (">=", 0.55),
    "elevation_r2": (">=", 0.3),
    "transfer_ratio": (">=", 0.7),
}
# Gated only when --s2_tiles runs the Sentinel-2 probe. LGND's reconstruction bound
# has no units, so ours is R^2, set below our first run's 0.857.
S2_GATES = {
    "nodata_with_imagery": ("==", 0),
    "data_without_imagery": ("==", 0),
    "reconstruction_r2": (">=", 0.75),
}

KNN_K = 10
FOLDS = 5
RIDGE_ALPHA = 1.0
NORM_ROBUST_Z = 3.5
DUPLICATE_COSINE = 0.99999
# A dimension is dead if its spread is under this fraction of the median dimension's.
DEAD_DIM_FRACTION = 0.01


def dequantize(q: np.ndarray) -> np.ndarray:
    """Invert the signed-power quantization, as model.dequantize_embeddings does."""
    r = q.astype(np.float32) / QUANTIZE_SCALE
    return np.abs(r) ** QUANTIZE_POWER * np.sign(r)


def unit(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- geometry metrics


def participation_ratio(x: np.ndarray) -> tuple[float, int]:
    """Effective dimensionality as a fraction of the total, and dims for 90% variance.

    Args:
        x: samples by dimensions.

    Returns:
        (sum(l)^2 / sum(l^2) / d, components needed for 90% of variance).
    """
    xc = x - x.mean(0)
    lam = np.clip(np.linalg.eigvalsh(xc.T @ xc / (len(x) - 1))[::-1], 0, None)
    pr = lam.sum() ** 2 / (lam**2).sum() / x.shape[1]
    dims90 = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.9) + 1)
    return float(pr), dims90


def uniformity(x: np.ndarray, t: float = 2.0) -> float:
    """Wang and Isola's uniformity: log mean exp(-t * squared distance) on the sphere."""
    u = unit(x)
    sq = np.clip(2 - 2 * u @ u.T, 0, None)
    iu = np.triu_indices(len(u), 1)
    return float(np.log(np.mean(np.exp(-t * sq[iu]))))


def geometry(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Stage 2 metrics for one set of vectors."""
    pr, dims90 = participation_ratio(x)
    std = x.std(0)
    corr = np.corrcoef(x.T)
    off = np.abs(corr[~np.eye(len(corr), dtype=bool)])
    sub = x[rng.choice(len(x), min(len(x), 5000), replace=False)]
    norms = np.linalg.norm(x, axis=1)
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) * 1.4826
    return {
        "participation_ratio": pr,
        "dims_for_90pct_variance": dims90,
        "dead_dims": int((std < DEAD_DIM_FRACTION * np.median(std)).sum()),
        "dim_std_min_over_median": float(std.min() / np.median(std)),
        "mean_abs_offdiag_corr": float(off.mean()),
        "max_abs_offdiag_corr": float(off.max()),
        "uniformity": uniformity(sub),
        "norm_median": float(med),
        "norm_outlier_fraction": float(
            (np.abs(norms - med) / mad > NORM_ROBUST_Z).mean()
        ),
    }


# ---------------------------------------------------------------- probes


def knn_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, n_classes: int
) -> np.ndarray:
    """Cosine kNN with similarity-weighted votes.

    Args:
        train_x: training vectors.
        train_y: training class indices.
        test_x: vectors to classify.
        n_classes: number of classes.

    Returns:
        predicted class index per test vector.
    """
    tr = unit(train_x)
    te = unit(test_x)
    out = np.empty(len(te), dtype=np.int64)
    for s in range(0, len(te), 2048):
        sims = te[s : s + 2048] @ tr.T
        top = np.argpartition(-sims, KNN_K, axis=1)[:, :KNN_K]
        w = np.take_along_axis(sims, top, 1)
        votes = np.zeros((len(sims), n_classes))
        np.add.at(votes, (np.arange(len(sims))[:, None], train_y[top]), w)
        out[s : s + 2048] = votes.argmax(1)
    return out


def ridge_oof_r2(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Out-of-fold R^2 of a ridge regression on standardized features."""
    pred = np.empty_like(y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        a = (x[tr] - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(
            a.T @ a + RIDGE_ALPHA * np.eye(a.shape[1]), a.T @ (y[tr] - ym)
        )
        pred[te] = ((x[te] - mu) / sd) @ w + ym
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def silhouette_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Mean silhouette under cosine distance."""
    u = unit(x)
    d = 1 - u @ u.T
    classes = np.unique(y)
    means = np.stack([d[:, y == c].mean(1) for c in classes], 1)
    counts = np.array([(y == c).sum() for c in classes])
    own = np.searchsorted(classes, y)
    # Exclude self (distance 0) from the own-class mean.
    a = means[np.arange(len(y)), own] * counts[own] / np.maximum(counts[own] - 1, 1)
    means[np.arange(len(y)), own] = np.inf
    b = means.min(1)
    s = (b - a) / np.maximum(a, b)
    return float(s[counts[own] > 1].mean())


def classification_metrics(y: np.ndarray, pred: np.ndarray, n: int) -> dict:
    """Accuracy, balanced accuracy and confusion matrix (rows are truth)."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    present = cm.sum(1) > 0
    return {
        "accuracy": float(np.diag(cm).sum() / cm.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "recall": recall.tolist(),
        "confusion": cm.tolist(),
    }


# ---------------------------------------------------------------- regions


def region_of(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Assign each point to one of nine coarse regions (a full partition)."""
    r = np.full(lon.shape, "", dtype=object)
    w = lon < -30
    r[w & (lat >= 15)] = "north_america"
    r[w & (lat < 15)] = "south_america"
    mid = (lon >= -30) & (lon < 60)
    r[(lon >= -30) & (lon < 45) & (lat >= 35)] = "europe"
    r[mid & (lat >= 15) & (lat < 35)] = "n_africa_middle_east"
    r[mid & (lat < 15)] = "sub_saharan_africa"
    r[(lon >= 45) & (lon < 100) & (lat >= 35)] = "central_asia"
    r[(lon >= 60) & (lon < 100) & (lat < 35)] = "south_asia"
    r[(lon >= 100) & (lat >= -10)] = "east_se_asia"
    r[(lon >= 100) & (lat < -10)] = "oceania"
    return r


# ---------------------------------------------------------------- label rasters


def worldcover_tile(lon: float, lat: float) -> str:
    """WorldCover's 3 degree tile name, keyed by its south-west corner."""
    la, lo = int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)
    return (
        f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    )


def dem_tile(lon: float, lat: float) -> str:
    """Copernicus GLO-30's 1 degree tile name, keyed by its south-west corner."""
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    return (
        f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_"
        f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    )


def sample_tiled(
    lon: np.ndarray,
    lat: np.ndarray,
    tile_of: Callable[[float, float], str],
    url_of: Callable[[str], str],
    missing: float,
    workers: int = 1,
) -> np.ndarray:
    """Nearest-pixel values of a tiled EPSG:4326 raster at points.

    Each overlapping tile is read once, over a window bounding its points. A tile
    that does not exist (open ocean, for both sources here) yields ``missing``.
    """
    out = np.full(lon.shape, missing, dtype=np.float32)
    keys = np.array([tile_of(a, b) for a, b in zip(lon, lat)])

    def one(key: str) -> tuple[np.ndarray, np.ndarray] | None:
        sel = keys == key
        try:
            with rasterio.open(url_of(key)) as src:
                rows, cols = rasterio.transform.rowcol(
                    src.transform, lon[sel], lat[sel]
                )
                rows = np.clip(np.asarray(rows), 0, src.height - 1)
                cols = np.clip(np.asarray(cols), 0, src.width - 1)
                r0, c0 = rows.min(), cols.min()
                win = RasterWindow(c0, r0, cols.max() - c0 + 1, rows.max() - r0 + 1)
                raw = src.read(1, window=win)[rows - r0, cols - c0]
                vals = raw.astype(np.float32)
                if src.nodata is not None:
                    vals[raw == src.nodata] = missing
                return sel, vals
        except RasterioIOError:
            return None

    unique = np.unique(keys)
    with ThreadPoolExecutor(min(workers, len(unique)) or 1) as ex:
        for got in ex.map(one, unique):
            if got is not None:
                out[got[0]] = got[1]
    return out


def worldclim_paths(cache_dir: Path) -> dict[str, Path]:
    """Download WorldClim 2.1 bioclim at 10 arc-minutes once, returning the layers used."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: cache_dir / f"wc2.1_10m_bio_{n}.tif" for name, n in WORLDCLIM_VARS.items()
    }
    if not all(p.exists() for p in paths.values()):
        logger.info("downloading %s", WORLDCLIM_URL)
        resp = requests.get(WORLDCLIM_URL, timeout=600)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for p in paths.values():
                (cache_dir / p.name).write_bytes(zf.read(p.name))
    return paths


# ---------------------------------------------------------------- store access


@dataclass
class Zone:
    """One zone's array and the pieces needed to place its pixels on the ground."""

    array: zarr.Array
    years: list[int]
    crs: str
    transform: list[float]
    to_lonlat: pyproj.Transformer


@dataclass
class Store:
    """The opened store and its zones."""

    path: str
    root: zarr.Group
    zones: dict[str, Zone] = field(default_factory=dict)

    def zone(self, name: str) -> Zone:
        """Open a zone lazily; consolidated metadata makes this cheap."""
        if name not in self.zones:
            group = self.root[name]
            attrs = group.attrs
            self.zones[name] = Zone(
                array=group["embeddings"],
                years=[int(v) for v in group["time"][:]],
                crs=attrs["proj:code"],
                transform=list(attrs["spatial:transform"]),
                to_lonlat=pyproj.Transformer.from_crs(
                    attrs["proj:code"], "EPSG:4326", always_xy=True
                ),
            )
        return self.zones[name]

    def lonlat(self, name: str, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-center lon/lat for a TILE x TILE block at (y0, x0)."""
        z = self.zone(name)
        a, _, c, _, e, f = z.transform
        cols = np.arange(x0, x0 + TILE) + 0.5
        rows = np.arange(y0, y0 + TILE) + 0.5
        xx, yy = np.meshgrid(c + a * cols, f + e * rows)
        return z.to_lonlat.transform(xx, yy)


def list_shards(
    store_path: str, time_index: int, storage_options: dict[str, Any]
) -> list[tuple[str, int, int]]:
    """Every written shard for one time index, as (zone, shard row, shard col)."""
    fs, root = url_to_fs(store_path, **storage_options)
    zones = [
        p.rsplit("/", 1)[-1]
        for p in fs.ls(root)
        if p.rsplit("/", 1)[-1].startswith("utm")
    ]

    def one(zone: str) -> list[tuple[str, int, int]]:
        prefix = f"{root}/{zone}/embeddings/c/{time_index}/0/"
        try:
            keys = fs.find(prefix)
        except FileNotFoundError:
            return []
        out = []
        for k in keys:
            sy, sx = k[len(prefix) :].split("/")
            out.append((zone, int(sy), int(sx)))
        return out

    with ThreadPoolExecutor(32) as ex:
        return [s for part in ex.map(one, zones) for s in part]


# ---------------------------------------------------------------- per-tile work


@dataclass
class TileResult:
    """What one sampled tile contributes."""

    zone: str
    y0: int
    x0: int
    partial_nodata: int = 0
    all_zero: int = 0
    saturated: int = 0
    values: int = 0
    digest: str = ""
    uniform_q: np.ndarray | None = None
    uniform_yx: np.ndarray | None = None
    valid_bits: np.ndarray | None = None
    uniform_lonlat: np.ndarray | None = None
    uniform_label: np.ndarray | None = None
    strat_q: np.ndarray | None = None
    strat_label: np.ndarray | None = None
    strat_lonlat: np.ndarray | None = None


def tile_origin(zone: str, sy: int, sx: int, rng: random.Random) -> tuple[int, int]:
    """A random inner tile within a shard."""
    k = SHARD // TILE
    return sy * SHARD + TILE * rng.randrange(k), sx * SHARD + TILE * rng.randrange(k)


def stage1_counts(q: np.ndarray, res: TileResult) -> np.ndarray:
    """Record value checks for a tile and return its valid-pixel mask."""
    nod = q == NODATA_VALUE
    any_nod, all_nod = nod.any(0), nod.all(0)
    res.partial_nodata = int((any_nod & ~all_nod).sum())
    valid = ~any_nod
    v = q[:, valid]
    res.all_zero = int((v == 0).all(0).sum())
    res.saturated = int((np.abs(v.astype(np.int16)) == 127).sum())
    res.values = int(v.size)
    if valid.any():
        res.digest = hashlib.sha1(np.where(valid, q, 0).tobytes()).hexdigest()
    return valid


def process_tile(
    store: Store,
    shard: tuple[str, int, int],
    time_index: int,
    uniform_per_tile: int,
    strat_per_class: int,
) -> TileResult:
    """Read one tile, run value checks, and draw the labeled samples."""
    zone, sy, sx = shard
    rng = random.Random(f"{zone}/{sy}/{sx}")
    y0, x0 = tile_origin(zone, sy, sx, rng)
    res = TileResult(zone=zone, y0=y0, x0=x0)
    q = np.asarray(
        store.zone(zone).array[time_index, :, y0 : y0 + TILE, x0 : x0 + TILE]
    )
    valid = stage1_counts(q, res)
    res.valid_bits = np.packbits(valid)
    if not valid.any():
        return res
    lon, lat = store.lonlat(zone, y0, x0)
    ys, xs = np.nonzero(valid)
    pick = rng.sample(range(len(ys)), min(uniform_per_tile, len(ys)))
    res.uniform_q = q[:, ys[pick], xs[pick]].T
    res.uniform_yx = np.stack([ys[pick], xs[pick]], 1)
    res.uniform_lonlat = np.stack([lon[ys[pick], xs[pick]], lat[ys[pick], xs[pick]]], 1)

    labels = sample_tiled(
        lon[valid],
        lat[valid],
        worldcover_tile,
        lambda t: WORLDCOVER_URL.format(tile=t),
        0,
    ).astype(np.int64)
    res.uniform_label = labels[pick]
    chosen = []
    for code in WORLDCOVER_CLASSES:
        idx = np.nonzero(labels == code)[0]
        if len(idx):
            chosen.extend(rng.sample(list(idx), min(strat_per_class, len(idx))))
    if chosen:
        chosen = np.array(chosen)
        res.strat_q = q[:, ys[chosen], xs[chosen]].T
        res.strat_label = labels[chosen]
        res.strat_lonlat = np.stack([lon[valid][chosen], lat[valid][chosen]], 1)
    return res


def year_tile(store: Store, shard: tuple[str, int, int], pixels: int) -> dict:
    """Read one tile across all years: adjacent-year cosine and cross-year repeats."""
    zone, sy, sx = shard
    rng = random.Random(f"years/{zone}/{sy}/{sx}")
    y0, x0 = tile_origin(zone, sy, sx, rng)
    z = store.zone(zone)
    q = np.asarray(z.array[:, :, y0 : y0 + TILE, x0 : x0 + TILE])
    valid = (q != NODATA_VALUE).all(1)
    digests = [
        hashlib.sha1(np.where(valid[t], q[t], 0).tobytes()).hexdigest()
        if valid[t].any()
        else ""
        for t in range(len(q))
    ]
    pairs = {}
    for t in range(len(q) - 1):
        both = valid[t] & valid[t + 1]
        ys, xs = np.nonzero(both)
        if len(ys) < pixels:
            continue
        sel = rng.sample(range(len(ys)), pixels)
        a = unit(dequantize(q[t][:, ys[sel], xs[sel]].T))
        b = unit(dequantize(q[t + 1][:, ys[sel], xs[sel]].T))
        pairs[f"{z.years[t]}-{z.years[t + 1]}"] = ((a * b).sum(1), a, b)
    return {"digests": [d for d in digests if d], "pairs": pairs}


def sentinel2_medians(
    store: Store, tiles: list[TileResult], year: int, ds_path: Path, workers: int
) -> tuple[list[str], list[np.ndarray | None], dict]:
    """Annual median Sentinel-2 reflectance at each tile's uniform pixels.

    Materializes the workers' Sentinel-2 layer over each tile, so the targets are the
    model's own inputs: up to twelve monthly mosaics, of which the median is taken per
    band over the months with data.

    The pipeline writes nodata exactly where every mosaic is empty, so the tiles'
    nodata masks are also compared against the imagery. Mosaics are built per window,
    and these windows are smaller than the workers', so a few scenes can differ.

    Returns:
        the band names; per tile a (pixels, bands) array (NaN where a pixel has no
        data) or None if nothing was materialized; and nodata agreement counts.
    """
    config = json.loads(
        Path(
            DATASET_CONFIG_FNAME.format(
                inputs=EmbeddingInputs.S2_S1_LANDSAT_DISTILLED.value
            )
        ).read_text()
    )
    layer = config["layers"][SENTINEL2_LAYER]
    config["layers"] = {SENTINEL2_LAYER: layer}
    bands = layer["band_sets"][0]["bands"]
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "config.json").write_text(json.dumps(config))
    dataset = Dataset(UPath(ds_path))
    start = datetime(year, 1, 1, tzinfo=UTC)
    windows = []
    for i, t in enumerate(tiles):
        z = store.zone(t.zone)
        a, _, c, _, e, f = z.transform
        col0, row0 = round(c / a) + t.x0, round(f / e) + t.y0
        window = Window(
            storage=dataset.storage,
            # The pipeline's materialize args only apply to this group.
            group=PREDICTION_GROUP,
            name=str(i),
            projection=Projection(CRS.from_string(z.crs), a, e),
            bounds=(col0, row0, col0 + TILE, row0 + TILE),
            time_range=(start, start + timedelta(days=365)),
            data_factory=dataset.window_data_storage_factory,
        )
        window.save()
        windows.append(window)
    logger.info("materializing Sentinel-2 for %d tiles", len(windows))
    # The workers' pools are sized for a GPU node; cap them for a shared CPU one.
    stages = {
        name: replace(
            stage,
            apply_windows_args=replace(stage.apply_windows_args, workers=workers),
        )
        for name in ("prepare_args", "ingest_args", "materialize_args")
        if (stage := getattr(MATERIALIZE_PIPELINE_ARGS, name))
    }
    materialize_dataset(
        UPath(ds_path),
        materialize_pipeline_args=replace(MATERIALIZE_PIPELINE_ARGS, **stages),
    )

    out: list[np.ndarray | None] = []
    nodata = {"pixels": 0, "nodata_with_imagery": 0, "data_without_imagery": 0}
    for window, t in zip(windows, tiles):
        assert t.uniform_yx is not None and t.valid_bits is not None
        months = []
        has_imagery = np.zeros((TILE, TILE), dtype=bool)
        for layer_name, group_idx in window.list_completed_layers():
            if layer_name != SENTINEL2_LAYER:
                continue
            arr = window.data.read_raster(
                SENTINEL2_LAYER,
                bands,
                GeotiffRasterFormat(),
                bounds=window.bounds,
                group_idx=group_idx,
                resampling=Resampling.nearest,
            ).get_chw_array()
            has_imagery |= (arr != 0).any(0)
            months.append(arr[:, t.uniform_yx[:, 0], t.uniform_yx[:, 1]].T)
        valid = (
            np.unpackbits(t.valid_bits)[: TILE * TILE].reshape(TILE, TILE).astype(bool)
        )
        nodata["pixels"] += TILE * TILE
        nodata["nodata_with_imagery"] += int((~valid & has_imagery).sum())
        nodata["data_without_imagery"] += int((valid & ~has_imagery).sum())
        if not months:
            out.append(None)
            continue
        v = np.stack(months).astype(np.float32)
        # A month with every band zero is a mosaic gap, not a dark pixel.
        v[(v == 0).all(2)] = np.nan
        with np.errstate(all="ignore"):
            out.append(np.nanmedian(v, 0) / 10000)
    return bands, out, nodata


# ---------------------------------------------------------------- analysis


def cross_tile_duplicates(x: np.ndarray, tile_id: np.ndarray) -> tuple[float, int]:
    """Highest cosine and count of near-identical pairs between different tiles."""
    u = unit(x)
    best, count = -1.0, 0
    for s in range(0, len(u), 2048):
        sims = u[s : s + 2048] @ u.T
        sims[tile_id[s : s + 2048, None] == tile_id[None, :]] = -1
        best = max(best, float(sims.max()))
        count += int((sims >= DUPLICATE_COSINE).sum())
    return best, count // 2


def evaluate_gates(flat: dict) -> dict:
    """Check each gate whose value was measured."""
    gates = {}
    for key, (op, bound) in {**GATES, **S2_GATES}.items():
        if key not in flat:
            continue
        value = flat[key]
        ok = (
            value == bound
            if op == "=="
            else value <= bound
            if op == "<="
            else value >= bound
        )
        gates[key] = {"value": value, "op": op, "bound": bound, "pass": bool(ok)}
    return gates


def run(args: argparse.Namespace) -> dict:
    """Run every stage and return the results document."""
    storage_options = args.storage_options
    if storage_options is None:
        storage_options = (
            DEFAULT_STORAGE_OPTIONS if args.store_path == DEFAULT_STORE else {}
        )
    store = Store(
        path=args.store_path,
        root=zarr.open_group(
            args.store_path, mode="r", storage_options=storage_options or None
        ),
    )
    probe_zone = store.zone("utm31")
    years = probe_zone.years
    t_main = years.index(args.year)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    logger.info("listing %d shards", args.year)
    shards = list_shards(args.store_path, t_main, storage_options)
    logger.info("%d shards written for %d", len(shards), args.year)
    picks = rng.sample(shards, min(args.tiles, len(shards)))
    with ThreadPoolExecutor(args.workers) as ex:
        tiles = list(
            ex.map(
                lambda s: process_tile(
                    store, s, t_main, args.uniform_per_tile, args.strat_per_class
                ),
                picks,
            )
        )
    logger.info("read %d tiles", len(tiles))

    # Stage 1.
    digests = [t.digest for t in tiles if t.digest]
    with_data = [t for t in tiles if t.uniform_q is not None]
    uq = np.concatenate([t.uniform_q for t in with_data])
    ull = np.concatenate([t.uniform_lonlat for t in with_data])
    ulabel = np.concatenate([t.uniform_label for t in with_data])
    utile = np.repeat(
        np.arange(len(with_data)),
        [len(t.uniform_q) for t in with_data if t.uniform_q is not None],
    )
    x = dequantize(uq)
    best, dup_pairs = cross_tile_duplicates(x, utile)
    stage1 = {
        "tiles_read": len(tiles),
        "tiles_without_data": len(tiles) - len(with_data),
        "pixels": len(x),
        "partial_nodata_pixels": sum(t.partial_nodata for t in tiles),
        "all_zero_pixels": sum(t.all_zero for t in tiles),
        "saturated_fraction": sum(t.saturated for t in tiles)
        / max(sum(t.values for t in tiles), 1),
        "max_cross_tile_cosine": best,
        "cross_tile_pairs_ge_0_99999": dup_pairs,
        "duplicate_tiles": len(digests) - len(set(digests)),
    }

    # Stage 2.
    stage2 = {"128": geometry(x, np_rng), "64": geometry(x[:, :PREFIX_DIMS], np_rng)}

    # Years: tiles written in every year.
    logger.info("listing %d shards for the year check", years[0])
    every_year = set(list_shards(args.store_path, 0, storage_options)) & set(shards)
    year_picks = rng.sample(sorted(every_year), min(args.year_tiles, len(every_year)))
    with ThreadPoolExecutor(args.workers) as ex:
        year_results = list(ex.map(lambda s: year_tile(store, s, 256), year_picks))
    year_digests = [d for r in year_results for d in r["digests"]]
    pair_stats = {}
    for name in sorted({k for r in year_results for k in r["pairs"]}):
        got = [r["pairs"][name] for r in year_results if name in r["pairs"]]
        cos = np.concatenate([g[0] for g in got])
        a = np.concatenate([g[1] for g in got])
        b = np.concatenate([g[2] for g in got])
        null = (a * b[np_rng.permutation(len(b))]).sum(1)
        pair_stats[name] = {
            "pixels": len(cos),
            "median": float(np.median(cos)),
            "p5": float(np.percentile(cos, 5)),
            "null_median": float(np.median(null)),
        }
    years_out = {
        "tiles": len(year_picks),
        "tiles_eligible": len(every_year),
        "pairs": pair_stats,
        "year_cosine_median_min": min(p["median"] for p in pair_stats.values()),
        "duplicate_tiles_across_years": len(year_digests) - len(set(year_digests)),
    }

    # Stage 3: land cover.
    labeled = [t for t in tiles if t.strat_q is not None]
    sq = np.concatenate([t.strat_q for t in labeled])
    sy_ = np.concatenate([t.strat_label for t in labeled])
    sll = np.concatenate([t.strat_lonlat for t in labeled])
    stile = np.repeat(
        np.arange(len(labeled)),
        [len(t.strat_q) for t in labeled if t.strat_q is not None],
    )
    # Cap common classes so the dominant ones do not set the score.
    keep = []
    for code in WORLDCOVER_CLASSES:
        idx = np.nonzero(sy_ == code)[0]
        keep.extend(np_rng.choice(idx, min(len(idx), args.class_cap), replace=False))
    keep = np.sort(np.array(keep))
    codes = [c for c in WORLDCOVER_CLASSES if (sy_[keep] == c).sum() >= FOLDS]
    keep = keep[np.isin(sy_[keep], codes)]
    sx_ = dequantize(sq[keep])
    cls = np.searchsorted(codes, sy_[keep])
    tile_folds = np_rng.permutation(len(labeled)) % FOLDS
    folds = tile_folds[stile[keep]]

    def grouped_knn(feats: np.ndarray) -> dict:
        pred = np.empty(len(cls), dtype=np.int64)
        for f in range(FOLDS):
            tr, te = folds != f, folds == f
            pred[te] = knn_predict(feats[tr], cls[tr], feats[te], len(codes))
        return classification_metrics(cls, pred, len(codes))

    knn128 = grouped_knn(sx_)
    knn64 = grouped_knn(sx_[:, :PREFIX_DIMS])
    sil_idx = np_rng.choice(len(sx_), min(len(sx_), 4000), replace=False)
    land_cover = {
        "classes": [WORLDCOVER_CLASSES[c] for c in codes],
        "class_counts": [int((cls == i).sum()) for i in range(len(codes))],
        "pixels": len(cls),
        "tiles": len(labeled),
        "knn_k": KNN_K,
        "folds": FOLDS,
        "128": knn128,
        "64": {k: v for k, v in knn64.items() if k != "confusion"},
        "silhouette": silhouette_cosine(sx_[sil_idx], cls[sil_idx]),
        "knn_accuracy": knn128["accuracy"],
    }

    # Stage 3: physical variables on the uniform sample.
    logger.info("sampling elevation and climate")
    elev = sample_tiled(
        ull[:, 0],
        ull[:, 1],
        dem_tile,
        lambda t: DEM_URL.format(tile=t),
        np.nan,
        args.workers,
    )
    targets = {"elevation": elev}
    for name, path in worldclim_paths(Path(args.cache_dir)).items():
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(ull)], dtype=np.float32)
            vals[vals == src.nodata] = np.nan
            vals[vals < -1e30] = np.nan
        targets[name] = np.log1p(vals) if name == "precipitation" else vals
    ufolds = np_rng.permutation(len(with_data))[utile] % FOLDS
    # Gate on land, matching the article's land-only golden set; with water is reported
    # alongside. Both are global probes, so between-region spread carries much of R^2.
    land = (ulabel != 0) & (ulabel != 80)
    physical: dict[str, Any] = {}
    for name, y in targets.items():
        finite = np.isfinite(y)
        ok = finite & land
        physical[name] = {
            "pixels": int(ok.sum()),
            "r2_128": ridge_oof_r2(x[ok], y[ok], ufolds[ok]),
            "r2_64": ridge_oof_r2(x[ok, :PREFIX_DIMS], y[ok], ufolds[ok]),
            "r2_128_with_water": ridge_oof_r2(x[finite], y[finite], ufolds[finite]),
        }
    physical["elevation_r2"] = physical["elevation"]["r2_128"]

    # Stage 3: reconstruction, reported without a gate since LGND's bound has no units.
    reconstruction: dict[str, Any] | None = None
    if args.s2_tiles:
        s2_tiles = with_data[: args.s2_tiles]
        bands, medians, nodata = sentinel2_medians(
            store,
            s2_tiles,
            args.year,
            Path(args.cache_dir) / f"s2_{args.year}",
            args.workers,
        )
        got = [i for i, m in enumerate(medians) if m is not None]
        if not got:
            raise ValueError("no Sentinel-2 imagery was materialized for any tile")
        # The uniform sample is laid out tile by tile, in with_data order.
        offsets = np.cumsum(
            [0] + [len(t.uniform_q) for t in with_data if t.uniform_q is not None]
        )
        idx = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in got])
        y = np.concatenate([medians[i] for i in got])
        ok = np.isfinite(y).all(1)
        idx, y = idx[ok], y[ok]
        per_band = {
            b: {
                "r2_128": ridge_oof_r2(x[idx], y[:, j], ufolds[idx]),
                "r2_64": ridge_oof_r2(x[idx, :PREFIX_DIMS], y[:, j], ufolds[idx]),
            }
            for j, b in enumerate(bands)
        }
        reconstruction = {
            "tiles": len(s2_tiles),
            "tiles_with_imagery": len(got),
            "nodata": nodata,
            "pixels": len(idx),
            "bands": per_band,
            "mean_r2_128": float(np.mean([v["r2_128"] for v in per_band.values()])),
            "mean_r2_64": float(np.mean([v["r2_64"] for v in per_band.values()])),
        }

    # Stage 4: leave one region out.
    regions = region_of(sll[keep, 0], sll[keep, 1])
    per_region = {}
    for reg in sorted(set(regions)):
        te = regions == reg
        if te.sum() < 50:
            continue
        pred = knn_predict(sx_[~te], cls[~te], sx_[te], len(codes))
        m = classification_metrics(cls[te], pred, len(codes))
        per_region[reg] = {
            "pixels": int(te.sum()),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
        }
    held_out = float(np.mean([r["accuracy"] for r in per_region.values()]))
    transfer = {
        "random_split_accuracy": knn128["accuracy"],
        "held_out_mean_accuracy": held_out,
        "transfer_ratio": held_out / knn128["accuracy"],
        "regions": per_region,
    }

    flat: dict[str, Any] = {
        **stage1,
        **stage2["128"],
        **years_out,
        **land_cover,
        **physical,
        **transfer,
    }
    if reconstruction is not None:
        flat.update(
            reconstruction["nodata"],
            reconstruction_r2=reconstruction["mean_r2_128"],
        )
    gates = evaluate_gates(flat)

    root_attrs = dict(store.root.attrs)
    commit = subprocess.run(  # nosec: fixed argv
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "meta": {
            "store": args.store_path,
            "model": root_attrs.get("geoemb:model"),
            "build_version": root_attrs.get("geoemb:build_version"),
            "year": args.year,
            "seed": args.seed,
            "shards_available": len(shards),
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks_commit": commit,
        },
        "gates": gates,
        "stage1": stage1,
        "stage2": stage2,
        "years": years_out,
        "land_cover": land_cover,
        "physical": physical,
        "reconstruction": reconstruction,
        "transfer": transfer,
    }


def main() -> None:
    """Parse arguments, run the checks, write the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store_path", default=DEFAULT_STORE)
    parser.add_argument(
        "--storage_options",
        type=json.loads,
        default=None,
        help="fsspec options as JSON; defaults to anonymous access for the default store.",
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--tiles", type=int, default=1500)
    parser.add_argument("--year_tiles", type=int, default=80)
    parser.add_argument("--uniform_per_tile", type=int, default=32)
    parser.add_argument("--strat_per_class", type=int, default=24)
    parser.add_argument("--class_cap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--s2_tiles",
        type=int,
        default=0,
        help="Tiles to probe Sentinel-2 reconstruction on; 0 skips it (needs olmoearth_run).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--cache_dir", default=str(Path.home() / ".cache" / "rslp_sanity")
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Remote COGs: skip directory listings on open, and never wait forever on a read.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
    results = run(args)
    Path(args.output).write_text(json.dumps(results, indent=1, default=float))
    failed = [k for k, g in results["gates"].items() if not g["pass"]]
    logger.info("%d gates, %d failed: %s", len(results["gates"]), len(failed), failed)


if __name__ == "__main__":
    main()
