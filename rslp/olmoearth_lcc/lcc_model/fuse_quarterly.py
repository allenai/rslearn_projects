"""Fuse yearly LCC runs into a quarterly change-event series for the LCC Viewer lab.

Each LCC run reports, per pixel, at most one change (the most recent one) in the 30
months before its end date. Running the model once a year and fusing the runs gives a
per-pixel series of events, one value per calendar quarter.

Inputs are the web-mercator viewer COG directories of the runs, one per end date,
written beforehand from the ``*_summary.tif`` prediction outputs with
allenai/olmoearth_lcc_viewer ``scripts/make_cogs.py`` (9-band summary COGs named
``{EPSG}_{col}_{row}.tif`` plus ``index.json``; ``make_overview.py`` adds the low-zoom
overview of each run). make_cogs.py puts a tile on the same mercator grid in every
run, and nearest warping only picks pixels, so fusing the COGs pixel by pixel equals
fusing the UTM rasters and then warping.

Per pixel, each run contributes at most one event: (quarter of ts_post_month, event
code, src/dst land cover, binary score), kept when a class head reports change and the
binary score is >= --thr. The quarter is capped at the run's end-date quarter. Events
from different runs form one cluster (one physical event) when their quarters are
within one quarter of each other, or when the later run's change interval
(ts_pre_month, ts_post_month] overlaps the cluster's running intersection and its code
is already in the cluster (so a change seen by 3+ yearly runs stays one event). A
later report whose last unchanged image is after the earlier report's first changed
image is a genuine repeat and stays separate. For each cluster:

- quarter: if it has 2+ members whose intervals intersect, the quarter of the earliest
  changed image (min ts_post_month); otherwise the quarter of the best-dated member
  (event age 2-3 quarters before the run's end date is most reliable, see
  _rank_of_age).
- code, src, dst: score-weighted majority over the members.
- against: runs whose window covers that quarter but that report no event, or only an
  earlier one (a run reporting a later event is uninformative, since the model only
  reports the most recent change).
- accepted if n >= 2 and n / (n + against) >= --min_support, or if n == 1, against ==
  0 and score >= --thr_single (quarters only one run sees).

Land cover starts at the src of the first accepted event (else the majority src over
the runs) and takes the dst of each accepted event from its quarter on.

Outputs, in --out_dir:

- ``{EPSG}_{col}_{row}.tif``: uint8 COGs with 3 * NQ + 6 bands, NQ = number of
  quarters: event code per quarter (0 nodata, 1 none, 2.. see EVENT_CATEGORY_NAMES),
  land cover after each quarter's events, confidence per quarter (support x mean score
  for an accepted event, 255 - 255 * support for a rejected candidate in a quarter
  without an event, else 255), then QUARTERLY_SUMMARY_BANDS. Written band-interleaved
  so a browser reads only the bands it shows.
- ``index.json`` and a low-zoom ``overview.tif``, in the same layout as make_cogs.py
  and make_overview.py.

It also writes the lab's ``manifest.json`` (--manifest, default next to --out_dir)
listing the fused layer and the yearly runs. Layer paths are relative when the layer
is under the manifest's directory, else public https URLs for gs:// paths.

Usage::

    python -m rslp.olmoearth_lcc.lcc_model.fuse_quarterly \
        --run gs://bucket/lcc/end_20231020=2023-10-20 \
        --run gs://bucket/lcc/end_20241020=2024-10-20 \
        --run gs://bucket/lcc/end_20251020=2025-10-20 \
        --out_dir gs://bucket/lcc/quarterly
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import multiprocessing
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.shutil
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rslearn.utils.fsspec import open_rasterio_upath_reader
from upath import UPath

from .postprocess import (
    LC_CLASS_NAMES,
    NONE_CATEGORY_IDX,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
    SUMMARY_BANDS,
    SUMMARY_BINARY_CHANGE_BAND,
    SUMMARY_DST_CLASS_BAND,
    SUMMARY_POST_CLASS_BAND,
    SUMMARY_PRE_CLASS_BAND,
    SUMMARY_SRC_CLASS_BAND,
    SUMMARY_TS_POST_MONTH_BAND,
    SUMMARY_TS_PRE_MONTH_BAND,
)
from .timestamp_encoding import TIMESTAMP_EPOCH

# Event codes: the POST head's categories, then the PRE head's real categories
# (code = offset + pre class), used when the POST head says none (e.g. deforestation
# or wildfire without new land use).
PRE_EVENT_CODE_OFFSET = len(POST_CHANGE_CATEGORY_NAMES) - 2
EVENT_CATEGORY_NAMES = POST_CHANGE_CATEGORY_NAMES + PRE_CHANGE_CATEGORY_NAMES[2:]

# Per-pixel summary bands after the per-quarter bands. first/last_event_q are 1-based
# quarter positions (0 = no event); lc_start/lc_end are the land cover of the first and
# last quarter.
QUARTERLY_SUMMARY_BANDS = [
    "first_event_q",
    "last_event_q",
    "n_events",
    "last_event_code",
    "lc_start",
    "lc_end",
]

# Same palettes as allenai/olmoearth_lcc_viewer app/viewer/categories.py.
PRE_CHANGE_COLORS = {
    "deforestation": "#8b4513",
    "urban_erosion": "#708090",
    "wetland_loss": "#20b2aa",
    "water_contract": "#4682b4",
    "removed_crop_structure": "#daa520",
}
# Pre-head categories that are really same-change (land use unchanged).
SAME_CHANGE_COLORS = {
    "agricultural_activity": "#b8860b",
    "wildfire": "#e25822",
    "ice_motion": "#7fdbff",
    "flooding": "#001f3f",
}
POST_CHANGE_COLORS = {
    "vegetation_growth": "#2ecc40",
    "new_building": "#ff4136",
    "new_road": "#555555",
    "new_infrastructure": "#ff851b",
    "new_crop_field": "#9acd32",
    "new_aquafarm": "#00bcd4",
    "site_clearing": "#d2b48c",
    "water_expand": "#0074d9",
    "mining": "#b10dc9",
    "new_crop_structure": "#ffdc00",
    "selective_logging": "#3d9970",
    "landslide": "#85144b",
    "settlement": "#f012be",
}
LC_COLORS = [
    "#000000",
    "#b4b4b4",
    "#8b0000",
    "#f096ff",
    "#d1a33d",
    "#ffff4c",
    "#fae6a0",
    "#ffbb22",
    "#f0f0f0",
    "#006400",
    "#fa0000",
    "#0064c8",
    "#0096a0",
]

# Quarters between run reports that still count as the same event.
MERGE_GAP_QUARTERS = 1
# Stand-in quarter/month for "no event" and "no pre date".
SENTINEL = 999
# Rows fused at a time.
BLOCK_ROWS = 512

# Web-mercator tile grid, as in make_cogs.py / make_overview.py.
WEB_MERCATOR_HALF_CIRCUMFERENCE = math.pi * 6378137.0
OVERVIEW_MAX_ZOOM = 7
OVERVIEW_COG_NAME = "overview.tif"
INDEX_NAME = "index.json"
COG_OPTIONS = dict(
    COMPRESS="DEFLATE",
    BLOCKSIZE=256,
    OVERVIEW_RESAMPLING="NEAREST",
    BIGTIFF="IF_SAFER",
    INTERLEAVE="BAND",
    NUM_THREADS=2,
)


@dataclass(frozen=True)
class FusionParams:
    """Thresholds of the fusion (see the module docstring)."""

    thr: int = 128
    thr_single: int = 192
    min_support: float = 0.5
    lookback_months: int = 30


@dataclass(frozen=True)
class Run:
    """One yearly LCC run: its viewer COG directory and end date."""

    path: UPath
    end_date: date

    @property
    def end_quarter(self) -> int:
        """Quarter of the end date, counted from the first quarter of TIMESTAMP_EPOCH."""
        return quarter_index(self.end_date)


def quarter_index(d: date) -> int:
    """Quarters since the first quarter of TIMESTAMP_EPOCH (summary month 1)."""
    return (d.year - TIMESTAMP_EPOCH.year) * 4 + (d.month - 1) // 3


def quarter_name(q: int) -> str:
    """Name a quarter index, e.g. "2022Q1"."""
    return f"{TIMESTAMP_EPOCH.year + q // 4}Q{q % 4 + 1}"


def quarter_start(q: int) -> date:
    """First day of a quarter index."""
    return date(TIMESTAMP_EPOCH.year + q // 4, (q % 4) * 3 + 1, 1)


def _rank_of_age(age: npt.NDArray) -> npt.NDArray:
    """Dating reliability (lower = better) from event age in quarters before the end date.

    Measured on pairs of overlapping runs: events 2-3 quarters before the end date are
    dated most consistently, the newest and oldest ones least.
    """
    r = np.full(age.shape, 4, np.int16)
    r[(age == 1) | (age == 4)] = 1
    r[age == 0] = 2
    r[(age == 5) | (age == 6)] = 3
    r[(age == 2) | (age == 3)] = 0
    return r


def _weighted_mode(vals: npt.NDArray, w: npt.NDArray, mem: npt.NDArray) -> npt.NDArray:
    """Weighted majority of vals (K, h, w) over members mem; ties go to the first run."""
    tot = np.zeros(vals.shape, np.float32)
    for k in range(vals.shape[0]):
        tot += np.where(mem & (vals == vals[k : k + 1]), w[k : k + 1], 0)
    tot = np.where(mem, tot, -1)
    best = tot.argmax(0)[None]
    return np.take_along_axis(vals, best, 0)[0]


def run_events(
    stack: npt.NDArray, end_quarters: npt.NDArray, params: FusionParams
) -> dict[str, npt.NDArray]:
    """Per-run events from a (K, 9, h, w) int16 stack of summary bands."""
    binc = stack[:, SUMMARY_BINARY_CHANGE_BAND]
    pre = stack[:, SUMMARY_PRE_CLASS_BAND]
    post = stack[:, SUMMARY_POST_CLASS_BAND]
    ts_pre = stack[:, SUMMARY_TS_PRE_MONTH_BAND]
    ts_post = stack[:, SUMMARY_TS_POST_MONTH_BAND]
    valid = pre > 0
    ev = (
        valid
        & ((pre != NONE_CATEGORY_IDX) | (post != NONE_CATEGORY_IDX))
        & (binc >= params.thr)
        & (ts_post > 0)
    )
    q = np.minimum((ts_post - 1) // 3, end_quarters[:, None, None])
    code = np.where(
        post >= 2,
        post,
        np.where(pre >= 2, PRE_EVENT_CODE_OFFSET + pre, NONE_CATEGORY_IDX),
    )
    return dict(
        valid=valid,
        ev=ev,
        q=np.where(ev, q, SENTINEL).astype(np.int16),
        code=code.astype(np.int16),
        src=stack[:, SUMMARY_SRC_CLASS_BAND],
        dst=stack[:, SUMMARY_DST_CLASS_BAND],
        score=binc,
        pre_m=np.where(ts_pre > 0, ts_pre, -SENTINEL).astype(np.int16),
        post_m=ts_post.astype(np.int16),
    )


def cluster(
    events: dict[str, npt.NDArray], end_quarters: npt.NDArray, params: FusionParams
) -> list[dict[str, npt.NDArray]]:
    """Group the per-run events of each pixel into clusters and accept or reject them.

    Returns one dict of (h, w) arrays per cluster slot (at most K).
    """
    num_runs = events["q"].shape[0]
    lookback = params.lookback_months // 3
    order = np.argsort(events["q"], axis=0, kind="stable")

    def take(x: npt.NDArray) -> npt.NDArray:
        """Reorder per-run arrays by event quarter."""
        return np.take_along_axis(x, order, 0)

    qs, evs, codes = take(events["q"]), take(events["ev"]), take(events["code"])
    srcs, dsts = take(events["src"]), take(events["dst"])
    pres, posts = take(events["pre_m"]), take(events["post_m"])
    scores = take(events["score"]).astype(np.float32)
    run_idx = take(
        np.broadcast_to(
            np.arange(num_runs, dtype=np.int16)[:, None, None], events["q"].shape
        )
    )

    cid = np.zeros(qs.shape, np.int16)
    # Running min post month and bit set of codes of the current cluster.
    hi = posts[0].copy()
    cmask = np.left_shift(np.int32(1), codes[0].astype(np.int32))
    for k in range(1, num_runs):
        new = qs[k] - qs[k - 1] > MERGE_GAP_QUARTERS
        has_code = (np.right_shift(cmask, codes[k].astype(np.int32)) & 1) == 1
        overlap = evs[k] & evs[k - 1] & (pres[k] > 0) & (pres[k] < hi) & has_code
        new &= ~overlap
        cid[k] = cid[k - 1] + new
        bit = np.left_shift(np.int32(1), codes[k].astype(np.int32))
        cmask = np.where(new, bit, cmask | bit)
        hi = np.where(new, posts[k], np.minimum(hi, posts[k]))
    cid_by_run = np.empty_like(cid)
    np.put_along_axis(cid_by_run, order, cid, 0)

    key = _rank_of_age(end_quarters[run_idx] - qs).astype(np.float32) * 1000 - scores
    out = []
    for j in range(num_runs):
        mem = (cid == j) & evs
        n = mem.sum(0)
        has = n > 0
        if not has.any():
            continue
        best = np.where(mem, key, np.inf).argmin(0)[None]
        qj = np.take_along_axis(qs, best, 0)[0]
        lo_m = np.where(mem, pres, -SENTINEL).max(0)
        hi_m = np.where(mem, posts, SENTINEL).min(0)
        qmin = np.where(mem, qs, SENTINEL).min(0)
        qj = np.where((n >= 2) & (lo_m < hi_m), qmin, qj).astype(np.int16)
        smax = np.where(mem, scores, 0).max(0)
        smean = np.where(mem, scores, 0).sum(0) / np.maximum(n, 1)
        against = np.zeros(n.shape, np.int16)
        for r in range(num_runs):
            covers = (
                events["valid"][r]
                & (qj >= end_quarters[r] - lookback)
                & (qj <= end_quarters[r])
            )
            in_j = events["ev"][r] & (cid_by_run[r] == j)
            against += covers & ~in_j & (~events["ev"][r] | (events["q"][r] < qj))
        support = n / np.maximum(n + against, 1)
        acc = has & (
            ((n >= 2) & (support >= params.min_support))
            | ((n == 1) & (against == 0) & (smax >= params.thr_single))
        )
        out.append(
            dict(
                has=has,
                acc=acc,
                q=qj,
                code=_weighted_mode(codes, scores, mem),
                src=_weighted_mode(srcs, scores, mem),
                dst=_weighted_mode(dsts, scores, mem),
                support=support,
                conf=np.clip(np.round(support * smean), 1, 255).astype(np.uint8),
            )
        )
    return out


def compose(
    events: dict[str, npt.NDArray],
    clusters: list[dict[str, npt.NDArray]],
    first_quarter: int,
    num_quarters: int,
) -> npt.NDArray:
    """Write the (3 * NQ + 6, h, w) uint8 quarterly bands of one block."""
    shape = events["q"].shape[1:]
    any_valid = events["valid"].any(0)
    lc_init = _weighted_mode(
        events["src"], np.ones(events["src"].shape, np.float32), events["valid"]
    ).astype(np.int16)
    first_set = np.zeros(shape, bool)
    for c in clusters:
        lc_init = np.where(c["acc"] & ~first_set, c["src"], lc_init)
        first_set |= c["acc"]
    lc_init = np.where(any_valid, lc_init, 0)

    event = np.where(any_valid, NONE_CATEGORY_IDX, 0).astype(np.uint8)
    event = event[None].repeat(num_quarters, 0)
    conf = np.where(any_valid, 255, 0).astype(np.uint8)[None].repeat(num_quarters, 0)
    lc = np.empty((num_quarters,) + shape, np.uint8)
    for t in range(num_quarters):
        g = first_quarter + t
        cur = lc_init.copy()
        for c in clusters:
            cur = np.where(c["acc"] & (c["q"] <= g), c["dst"], cur)
            m = c["acc"] & (c["q"] == g)
            event[t] = np.where(m, c["code"], event[t])
            conf[t] = np.where(m, c["conf"], conf[t])
            rejected = (
                c["has"] & ~c["acc"] & (c["q"] == g) & (event[t] == NONE_CATEGORY_IDX)
            )
            rej_conf = np.clip(np.round(255 * (1 - c["support"])), 1, 255)
            conf[t] = np.where(
                rejected, np.minimum(conf[t], rej_conf.astype(np.uint8)), conf[t]
            )
        lc[t] = np.where(any_valid, cur, 0)

    is_ev = event >= 2
    tq = np.arange(1, num_quarters + 1, dtype=np.uint8)[:, None, None]
    first = np.where(is_ev, tq, 255).min(0)
    first = np.where(first == 255, 0, first).astype(np.uint8)
    last = np.where(is_ev, tq, 0).max(0).astype(np.uint8)
    last_code = np.take_along_axis(
        event, np.maximum(last.astype(np.int64) - 1, 0)[None], 0
    )[0]
    last_code = np.where(
        last > 0, last_code, np.where(any_valid, NONE_CATEGORY_IDX, 0)
    ).astype(np.uint8)
    summary = np.stack(
        [first, last, is_ev.sum(0).astype(np.uint8), last_code, lc[0], lc[-1]]
    )
    return np.concatenate([event, lc, conf, summary])


def fuse_block(
    stack: npt.NDArray,
    end_quarters: npt.NDArray,
    first_quarter: int,
    num_quarters: int,
    params: FusionParams,
) -> npt.NDArray:
    """Fuse a (K, 9, h, w) stack of summary bands into (3 * NQ + 6, h, w) uint8 bands.

    Runs must be ordered by end date.
    """
    events = run_events(stack.astype(np.int16), end_quarters, params)
    clusters = cluster(events, end_quarters, params)
    return compose(events, clusters, first_quarter, num_quarters)


def band_descriptions(quarters: list[str]) -> list[str]:
    """Band names of the fused COGs."""
    return (
        [f"event_{q}" for q in quarters]
        + [f"lc_{q}" for q in quarters]
        + [f"conf_{q}" for q in quarters]
        + QUARTERLY_SUMMARY_BANDS
    )


def mercator_resolution(zoom: int) -> float:
    """Projected meters per pixel of 256px web-mercator tiles at a zoom level."""
    return 2 * WEB_MERCATOR_HALF_CIRCUMFERENCE / (256 * 2**zoom)


def _upload(local_path: Path, path: UPath) -> None:
    """Copy a finished local file to path (atomically for local destinations)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.protocol in ("", "file"):
        tmp = Path(path.path).with_name(f".tmp_{path.name}")
        shutil.copyfile(local_path, tmp)
        tmp.replace(path.path)
    else:
        with local_path.open("rb") as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst, 16 << 20)


def _write_cog(
    path: UPath,
    width: int,
    height: int,
    count: int,
    crs: Any,
    transform: Any,
    blocks: Any,
    descriptions: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Write (row offset, array) blocks to a band-interleaved COG at path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw = Path(tmp_dir) / "raw.tif"
        profile = dict(
            driver="GTiff",
            width=width,
            height=height,
            count=count,
            dtype="uint8",
            crs=crs,
            transform=transform,
            nodata=0,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            interleave="band",
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(raw, "w", **profile) as dst:
            for row, data in blocks:
                dst.write(data, window=((row, row + data.shape[1]), (0, width)))
            for i, d in enumerate(descriptions or []):
                dst.set_band_description(i + 1, d)
            dst.update_tags(**(tags or {}))
        cog = Path(tmp_dir) / "cog.tif"
        rasterio.shutil.copy(raw, cog, driver="COG", **COG_OPTIONS)
        _upload(cog, path)


def _index_entry(name: str, bounds: Any, res: float) -> dict[str, Any]:
    """An index.json entry as written by make_cogs.py."""
    return {
        "cog": name,
        "bounds3857": list(bounds),
        "max_zoom": round(math.log2(mercator_resolution(0) / res)),
    }


def fuse_tile(
    name: str,
    runs: list[Run],
    out_dir: UPath,
    first_quarter: int,
    num_quarters: int,
    params: FusionParams,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fuse one COG tile across the runs and return its index.json entry.

    A run without this tile counts as no prediction there.
    """
    out_path = out_dir / name
    if out_path.exists() and not overwrite:
        with open_rasterio_upath_reader(out_path) as src:
            return _index_entry(name, src.bounds, src.transform.a)

    grids: list[tuple[int, int, Any, Any] | None] = []
    for run in runs:
        path = run.path / name
        if not path.exists():
            grids.append(None)
            continue
        with open_rasterio_upath_reader(path) as src:
            if src.count != len(SUMMARY_BANDS):
                raise ValueError(f"{path} has {src.count} bands, expected 9")
            grids.append((src.width, src.height, src.transform, src.crs))
    present = [g for g in grids if g is not None]
    if any(g != present[0] for g in present):
        raise ValueError(f"{name} is not on the same grid in every run")
    width, height, transform, crs = present[0]
    end_quarters = np.array([r.end_quarter for r in runs], np.int16)

    def blocks() -> Any:
        """Yield (row offset, fused bands) per block of rows."""
        with contextlib.ExitStack() as stack_ctx:
            srcs = [
                None
                if grid is None
                else stack_ctx.enter_context(
                    open_rasterio_upath_reader(run.path / name)
                )
                for run, grid in zip(runs, grids)
            ]
            for row in range(0, height, BLOCK_ROWS):
                rows = min(BLOCK_ROWS, height - row)
                stack = np.zeros((len(runs), len(SUMMARY_BANDS), rows, width), np.uint8)
                for k, src in enumerate(srcs):
                    if src is not None:
                        stack[k] = src.read(window=((row, row + rows), (0, width)))
                yield (
                    row,
                    fuse_block(
                        stack, end_quarters, first_quarter, num_quarters, params
                    ),
                )

    quarters = [quarter_name(first_quarter + t) for t in range(num_quarters)]
    tags = dict(
        event_codes=json.dumps(EVENT_CATEGORY_NAMES),
        lc_codes=json.dumps(LC_CLASS_NAMES),
        quarters=json.dumps(quarters),
        run_end_dates=json.dumps([r.end_date.isoformat() for r in runs]),
        params=json.dumps(asdict(params)),
    )
    _write_cog(
        out_path,
        width,
        height,
        3 * num_quarters + len(QUARTERLY_SUMMARY_BANDS),
        crs,
        transform,
        blocks(),
        band_descriptions(quarters),
        tags,
    )
    bounds = rasterio.transform.array_bounds(height, width, transform)
    return _index_entry(name, bounds, transform.a)


def write_overview(
    out_dir: UPath, entries: list[dict[str, Any]], valid_band: int = 0
) -> dict[str, Any]:
    """Write the low-zoom overview COG of the tiles and return its index.json entry.

    As in make_overview.py: a single raster at OVERVIEW_MAX_ZOOM over all tiles,
    sampled with nearest resampling, where the first tile with a prediction at a pixel
    (valid_band > 0) wins.
    """
    res = mercator_resolution(OVERVIEW_MAX_ZOOM)
    half = WEB_MERCATOR_HALF_CIRCUMFERENCE
    world = 256 * 2**OVERVIEW_MAX_ZOOM
    col0 = max(math.floor((min(e["bounds3857"][0] for e in entries) + half) / res), 0)
    col1 = min(
        math.ceil((max(e["bounds3857"][2] for e in entries) + half) / res), world
    )
    row0 = max(math.floor((half - max(e["bounds3857"][3] for e in entries)) / res), 0)
    row1 = min(
        math.ceil((half - min(e["bounds3857"][1] for e in entries)) / res), world
    )
    west, north = col0 * res - half, half - row0 * res
    width, height = col1 - col0, row1 - row0
    acc: npt.NDArray | None = None
    descriptions: list[str] = []
    tags: dict[str, str] = {}
    for e in entries:
        with open_rasterio_upath_reader(out_dir / e["cog"]) as src:
            if acc is None:
                acc = np.zeros((src.count, height, width), np.uint8)
                descriptions = [d or "" for d in src.descriptions]
                tags = src.tags()
            ixmin, iymin = (
                max(west, src.bounds.left),
                max(north - height * res, src.bounds.bottom),
            )
            ixmax, iymax = (
                min(west + width * res, src.bounds.right),
                min(north, src.bounds.top),
            )
            if ixmin >= ixmax or iymin >= iymax:
                continue
            c0 = math.floor((ixmin - west) / res + 0.5)
            c1 = min(max(math.floor((ixmax - west) / res + 0.5), c0 + 1), width)
            r0 = math.floor((north - iymax) / res + 0.5)
            r1 = min(max(math.floor((north - iymin) / res + 0.5), r0 + 1), height)
            data = src.read(
                window=rasterio.windows.from_bounds(
                    ixmin, iymin, ixmax, iymax, transform=src.transform
                ),
                out_shape=(src.count, r1 - r0, c1 - c0),
                resampling=Resampling.nearest,
            )
        dst = acc[:, r0:r1, c0:c1]
        paint = (data[valid_band] > 0) & (dst[valid_band] == 0)
        dst[:, paint] = data[:, paint]
    assert acc is not None
    transform = from_origin(west, north, res, res)
    _write_cog(
        out_dir / OVERVIEW_COG_NAME,
        width,
        height,
        acc.shape[0],
        "EPSG:3857",
        transform,
        [(0, acc)],
        descriptions,
        tags,
    )
    entry = _index_entry(
        OVERVIEW_COG_NAME,
        rasterio.transform.array_bounds(height, width, transform),
        res,
    )
    entry["overview"] = True
    return entry


def _layer_path(layer_dir: UPath, manifest_dir: UPath) -> str:
    """Layer path as the lab resolves it against the manifest's URL."""
    layer, base = str(layer_dir).rstrip("/"), str(manifest_dir).rstrip("/")
    if layer.startswith(base + "/"):
        return layer[len(base) + 1 :] + "/"
    if layer.startswith("gs://"):
        return "https://storage.googleapis.com/" + layer[len("gs://") :] + "/"
    if layer.startswith(("http://", "https://")):
        return layer + "/"
    raise ValueError(
        f"{layer_dir} is not under the manifest's directory {manifest_dir} and has no "
        "public URL"
    )


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """Web-mercator meters to rounded (lon, lat) degrees."""
    r = 6378137.0
    lon = math.degrees(x / r)
    lat = math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2)
    return round(lon, 5), round(lat, 5)


def build_manifest(
    runs: list[Run],
    out_dir: UPath,
    manifest_dir: UPath,
    entries: list[dict[str, Any]],
    first_quarter: int,
    num_quarters: int,
    params: FusionParams,
) -> dict[str, Any]:
    """The LCC Viewer lab manifest: the fused layer, then the runs by end date."""
    tiles = [e["bounds3857"] for e in entries if not e.get("overview")]
    west, south = _lonlat(min(b[0] for b in tiles), min(b[1] for b in tiles))
    east, north = _lonlat(max(b[2] for b in tiles), max(b[3] for b in tiles))
    event_classes = []
    for code, name in enumerate(EVENT_CATEGORY_NAMES):
        color = (
            POST_CHANGE_COLORS.get(name)
            or PRE_CHANGE_COLORS.get(name)
            or SAME_CHANGE_COLORS.get(name)
        )
        if color is None:
            continue
        if code < len(POST_CHANGE_CATEGORY_NAMES):
            group = "post"
        else:
            group = "same" if name in SAME_CHANGE_COLORS else "pre"
        event_classes.append(dict(idx=code, name=name, color=color, group=group))
    lc_classes = [
        dict(idx=i, name=LC_CLASS_NAMES[i], color=LC_COLORS[i])
        for i in range(1, len(LC_CLASS_NAMES))
    ]
    quarters = [quarter_name(first_quarter + t) for t in range(num_quarters)]

    description = (
        f"Fused from the {len(runs)} yearly runs below. Per quarter: event (code of the "
        "change accepted in that quarter, 1 = none), land cover after that quarter's "
        "events, and confidence (support x mean score). Reports from different runs of "
        "the same change are merged into one event."
    )
    # The last quarter is the latest end date's; it may end early.
    last_end = max(r.end_date for r in runs)
    last_q = first_quarter + num_quarters - 1
    if last_end < quarter_start(last_q + 1) - timedelta(days=1):
        start = quarter_start(last_q)
        description += (
            f" {quarters[-1].replace('Q', ' Q')} covers {start:%b} 1 to "
            f"{last_end:%b} {last_end.day} only."
        )
    layers: list[dict[str, Any]] = [
        dict(
            id=out_dir.name,
            kind="quarterly",
            path=_layer_path(out_dir, manifest_dir),
            label="Fused quarterly series",
            description=description,
            quarters=quarters,
            bands=dict(
                event=0,
                lc=num_quarters,
                conf=2 * num_quarters,
                summary=3 * num_quarters,
            ),
            summary_bands=QUARTERLY_SUMMARY_BANDS,
            none_code=NONE_CATEGORY_IDX,
            event_classes=event_classes,
            lc_classes=lc_classes,
        )
    ]
    for run in sorted(runs, key=lambda r: r.end_date):
        layers.append(
            dict(
                id=run.path.name,
                kind="summary",
                path=_layer_path(run.path, manifest_dir),
                end_date=run.end_date.isoformat(),
                label=f"Yearly run ending {run.end_date.isoformat()}",
                description=(
                    f"One LCC run over the {params.lookback_months} months "
                    "before the end date."
                ),
            )
        )
    return dict(
        version=1,
        bounds=[west, south, east, north],
        lc_classes=lc_classes,
        layers=layers,
    )


def _read_index(path: UPath) -> list[dict[str, Any]]:
    """Read a layer directory's index.json."""
    with (path / INDEX_NAME).open() as f:
        return json.load(f)


def _write_json(path: UPath, obj: Any) -> None:
    """Write a JSON file (local or remote)."""
    with path.open("w") as f:
        json.dump(obj, f, indent=1)


def fuse_runs(
    runs: list[Run],
    out_dir: UPath,
    manifest_path: UPath,
    params: FusionParams | None = None,
    first_quarter: int | None = None,
    workers: int = 8,
    overwrite: bool = False,
    tiles: list[str] | None = None,
) -> None:
    """Fuse the runs' COG tiles and write index.json, the overview and the manifest.

    Args:
        runs: the yearly runs, in any order.
        out_dir: directory for the fused COGs, index.json and overview.
        manifest_path: where to write the lab's manifest.json.
        params: fusion thresholds (default: FusionParams()).
        first_quarter: quarter index of the first output quarter. Default: 3 quarters
            before the earliest end date's quarter (so the first run's last year).
        workers: tiles fused in parallel.
        overwrite: re-fuse tiles whose COG already exists.
        tiles: only fuse these tile COG names (default: every tile of any run).
    """
    params = params or FusionParams()
    runs = sorted(runs, key=lambda r: r.end_date)
    if len({r.end_date for r in runs}) != len(runs):
        raise ValueError("two runs have the same end date")
    if first_quarter is None:
        first_quarter = runs[0].end_quarter - 3
    num_quarters = runs[-1].end_quarter - first_quarter + 1
    if not 1 <= num_quarters <= 254:
        raise ValueError(f"{num_quarters} output quarters, expected 1 to 254")

    names = sorted(
        {
            e["cog"]
            for run in runs
            for e in _read_index(run.path)
            if not e.get("overview")
        }
    )
    if tiles is not None:
        names = [n for n in names if n in tiles]
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = functools.partial(
        fuse_tile,
        runs=runs,
        out_dir=out_dir,
        first_quarter=first_quarter,
        num_quarters=num_quarters,
        params=params,
        overwrite=overwrite,
    )
    entries = []
    with multiprocessing.Pool(min(workers, len(names))) as pool:
        for i, entry in enumerate(pool.imap_unordered(fn, names)):
            entries.append(entry)
            print(f"[{i + 1}/{len(names)}] {entry['cog']}", flush=True)
    entries.sort(key=lambda e: e["cog"])
    entries.append(write_overview(out_dir, entries))
    _write_json(out_dir / INDEX_NAME, entries)
    manifest = build_manifest(
        runs,
        out_dir,
        manifest_path.parent,
        entries,
        first_quarter,
        num_quarters,
        params,
    )
    _write_json(manifest_path, manifest)
    print(
        f"wrote {len(names)} tiles, overview and index.json to {out_dir}; "
        f"manifest {manifest_path}"
    )


def _parse_run(value: str) -> Run:
    """Parse a --run DIR=YYYY-MM-DD argument."""
    path, sep, end = value.rpartition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"expected DIR=YYYY-MM-DD, got {value!r}")
    return Run(UPath(path), date.fromisoformat(end))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Fuse yearly LCC viewer COGs into a quarterly event series."
    )
    parser.add_argument(
        "--run",
        type=_parse_run,
        action="append",
        required=True,
        help="DIR=YYYY-MM-DD: a run's viewer COG directory (from make_cogs.py) and "
        "its end date. Repeat for each run.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for the fused COGs, index.json and overview.tif.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Where to write the lab's manifest.json (default: next to --out_dir).",
    )
    parser.add_argument(
        "--first_quarter",
        default=None,
        help="First output quarter, e.g. 2022Q1 (default: 3 quarters before the "
        "earliest end date's quarter). The last is the latest end date's quarter.",
    )
    defaults = FusionParams()
    parser.add_argument(
        "--thr",
        type=int,
        default=defaults.thr,
        help="Minimum binary change score (0-255) for a run's report to count.",
    )
    parser.add_argument(
        "--thr_single",
        type=int,
        default=defaults.thr_single,
        help="Minimum score to accept an event that only one run reports.",
    )
    parser.add_argument(
        "--min_support",
        type=float,
        default=defaults.min_support,
        help="Minimum fraction of covering runs that must report a multi-run event.",
    )
    parser.add_argument(
        "--lookback_months",
        type=int,
        default=defaults.lookback_months,
        help="Months before its end date that each run covers.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Tiles in parallel.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-fuse tiles whose COG already exists in --out_dir.",
    )
    args = parser.parse_args()

    first_quarter = None
    if args.first_quarter is not None:
        year, _, q = args.first_quarter.upper().partition("Q")
        first_quarter = (int(year) - TIMESTAMP_EPOCH.year) * 4 + int(q) - 1
    out_dir = UPath(args.out_dir)
    manifest = (
        UPath(args.manifest) if args.manifest else out_dir.parent / "manifest.json"
    )
    fuse_runs(
        runs=args.run,
        out_dir=out_dir,
        manifest_path=manifest,
        params=FusionParams(
            thr=args.thr,
            thr_single=args.thr_single,
            min_support=args.min_support,
            lookback_months=args.lookback_months,
        ),
        first_quarter=first_quarter,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
