"""Public status page for a predict run: a Leaflet map with one layer per year.

Built from the completion markers alone, so it needs no Beaker access and shows the
same thing whoever runs it. The supervisor calls `publish_status` periodically; it can
also be run by hand.

Each year is a transparent Web Mercator PNG overlay, one block per filled polygon,
under a page that holds the stats inline. Everything is uploaded no-store, since the
names never change between builds.
"""

import io
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pyproj
from fsspec.core import url_to_fs
from PIL import Image, ImageDraw
from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from rslp.large_scale_embeddings.predict_pipeline import (
    PATCH_SIZE,
    RESOLUTION,
    get_marker_fname,
)
from rslp.large_scale_embeddings.write_jobs import TILE_SIZE, enumerate_blocks
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Overlay width and height in Web Mercator pixels. At 8192 an 8192 px block (82 km)
# is about 17 px at the equator, and the PNGs stay a few hundred KB each.
MAP_PIXELS = 8192
# Web Mercator's latitude limit, where the square world image ends.
MERCATOR_MAX_LAT = 85.0511
KM2_PER_CROP = PATCH_SIZE * PATCH_SIZE * RESOLUTION * RESOLUTION / 1e6
MARKER_READ_THREADS = 32
# Red for the oldest year through violet for the newest. Years are spread across it
# by position, so nine years take one stop each.
SPECTRUM = [
    "#c62828",
    "#e0701f",
    "#d9a219",
    "#8fb230",
    "#35a853",
    "#1b9aa6",
    "#2878b8",
    "#4a4fb0",
    "#7b3fa0",
]
COVERAGE_COLOR = "#bfb498"
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")
# Neither the CDN nor a browser may keep a copy: every build overwrites the same names,
# and a cached copy would show a stale page with no sign that it is stale.
CACHE_CONTROL = "no-store"


class TruncatedListingError(RuntimeError):
    """A marker listing came back shorter than one already seen."""


def year_colors(years: list[int]) -> dict[int, str]:
    """Pick a spectrum color per year, oldest red and newest violet.

    Args:
        years: the run's years.

    Returns:
        a hex color per year.
    """
    ordered = sorted(years)
    steps = max(len(ordered) - 1, 1)
    return {
        year: SPECTRUM[round(i * (len(SPECTRUM) - 1) / steps)]
        for i, year in enumerate(ordered)
    }


def read_markers(
    completed_path: str, cache_path: UPath
) -> dict[str, tuple[float | None, int]]:
    """Read every marker in a directory, reusing what an earlier call already read.

    Markers are written once and never change, so only new names are fetched.

    Args:
        completed_path: the marker directory.
        cache_path: local JSON file holding the markers already read.

    Returns:
        (gpu_seconds, crops written) per marker name. gpu_seconds is None for a
        marker written before markers recorded it.

    Raises:
        TruncatedListingError: if a marker read before is missing from the listing.
            Markers are never deleted during a run, so this means the listing was
            cut short, and publishing it would show progress going backwards.
    """
    cached: dict[str, tuple[float | None, int]] = {}
    if cache_path.exists():
        cached = {
            name: (seconds, crops)
            for name, (seconds, crops) in json.loads(cache_path.read_text()).items()
        }

    completed = UPath(completed_path)
    names = {p.name for p in completed.iterdir()} if completed.exists() else set()
    lost = len(cached.keys() - names)
    if lost:
        raise TruncatedListingError(
            f"{lost} marker(s) read earlier are missing from {completed_path}"
        )

    def read_one(name: str) -> tuple[str, tuple[float | None, int]]:
        marker = json.loads((completed / name).read_text())
        return name, (marker.get("gpu_seconds"), len(marker.get("written", [])))

    new = sorted(names - cached.keys())
    if new:
        with ThreadPoolExecutor(MARKER_READ_THREADS) as pool:
            cached.update(pool.map(read_one, new))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached))
    logger.info("%s: %d markers (%d new)", completed_path, len(cached), len(new))
    return cached


def _mercator_px(lon: list[float], lat: list[float]) -> list[tuple[float, float]]:
    """Project WGS84 points to pixel coordinates on the square world image."""
    points = []
    for x, y in zip(lon, lat):
        y = max(-MERCATOR_MAX_LAT, min(MERCATOR_MAX_LAT, y))
        merc = math.log(math.tan(math.pi / 4 + math.radians(y) / 2))
        points.append(
            ((x + 180) / 360 * MAP_PIXELS, (1 - merc / math.pi) / 2 * MAP_PIXELS)
        )
    return points


def render_layer(blocks: list[tuple[Projection, PixelBounds]], color: str) -> bytes:
    """Draw blocks as filled polygons on a transparent Web Mercator PNG.

    Args:
        blocks: the blocks to draw.
        color: their fill, as a hex color.

    Returns:
        the PNG bytes.
    """
    image = Image.new("P", (MAP_PIXELS, MAP_PIXELS), 0)
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    image.putpalette([0, 0, 0] + rgb)
    draw = ImageDraw.Draw(image)

    by_crs: dict[str, list[tuple[Projection, PixelBounds]]] = defaultdict(list)
    for projection, bounds in blocks:
        by_crs[str(projection.crs)].append((projection, bounds))
    for crs, crs_blocks in by_crs.items():
        to_wgs84 = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        for projection, (x0, y0, x1, y1) in crs_blocks:
            xs = [x * projection.x_resolution for x in (x0, x1, x1, x0)]
            ys = [y * projection.y_resolution for y in (y0, y0, y1, y1)]
            lon, lat = to_wgs84.transform(xs, ys)
            lon = list(lon)
            # A block straddling the antimeridian is drawn twice, once each side,
            # rather than as a polygon spanning the whole world.
            shifts = [0.0]
            if max(lon) - min(lon) > 180:
                lon = [x + 360 if x < 0 else x for x in lon]
                shifts = [0.0, -MAP_PIXELS]
            for shift in shifts:
                points = [(px + shift, py) for px, py in _mercator_px(lon, list(lat))]
                draw.polygon(points, fill=1)

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True, transparency=0)
    return buf.getvalue()


def _upload(url: str, data: bytes, content_type: str) -> None:
    """Write one object uncached, since its name never changes."""
    if "://" not in url:
        # A local path has no cache in front of it and no content type to set.
        path = UPath(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    fs, path = url_to_fs(url)
    fs.pipe_file(
        path,
        data,
        content_type=content_type,
        fixed_key_metadata={"cache_control": CACHE_CONTROL},
    )


def publish_status(
    status_path: str,
    years: list[int],
    completed_path_template: str,
    title: str,
    cache_dir: str,
    job_size: int = TILE_SIZE,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    enumeration_cache_dir: str | None = None,
    workers: dict[str, Any] | None = None,
) -> None:
    """Rebuild the status page from the markers and upload it.

    Args:
        status_path: directory to publish index.html and the layer PNGs into.
        years: the run's years, one map layer each.
        completed_path_template: marker directory containing ``{year}``.
        title: the page heading.
        cache_dir: local directory for markers already read.
        job_size: the run's block size, as given to the supervisor.
        epsg_code: the run's zone restriction, if any.
        wgs84_bounds: the run's bounding box restriction, if any.
        geojson_fname: the run's footprint restriction, if any.
        enumeration_cache_dir: the supervisor's enumeration cache, reused here.
        workers: worker counts from the supervisor's latest cycle, or None to leave
            them off the page. Keys: working, allocated ([priority, count] pairs),
            spare, outside, waiting.
    """
    blocks = enumerate_blocks(
        job_size=job_size,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        geojson_fname=geojson_fname,
        enumeration_cache_dir=enumeration_cache_dir,
    )
    colors = year_colors(years)
    status = UPath(status_path)
    gpu_seconds = 0.0
    timed_crops = 0
    layers = []
    pngs: dict[str, bytes] = {}

    for year in sorted(years):
        completed_path = completed_path_template.format(year=year)
        cache_key = completed_path.replace("://", "_").replace("/", "_")
        markers = read_markers(completed_path, UPath(cache_dir) / f"{cache_key}.json")
        # Only markers for blocks the run still covers count, so the percentage
        # compares like with like if the coverage area has shrunk.
        done = []
        for projection, bounds in blocks:
            name = get_marker_fname(completed_path, projection, bounds).name
            if name not in markers:
                continue
            done.append((projection, bounds))
            seconds, crops = markers[name]
            if seconds is not None:
                gpu_seconds += seconds
                timed_crops += crops
        fname = f"{year}.png"
        pngs[fname] = render_layer(done, colors[year])
        layers.append(
            {
                "year": year,
                "color": colors[year],
                "done": len(done),
                "total": len(blocks),
                "file": fname,
            }
        )

    pngs["coverage.png"] = render_layer(blocks, COVERAGE_COLOR)
    gpu_hours = gpu_seconds / 3600
    now = datetime.now(DISPLAY_TZ)
    data = {
        "title": title,
        "updated": now.strftime("%a %b %-d, %-I:%M %p %Z"),
        "version": int(now.timestamp()),
        "gpu_hours": round(gpu_hours),
        "km2_per_gpu_hour": round(timed_crops * KM2_PER_CROP / gpu_hours)
        if gpu_hours
        else None,
        "coverage": {"color": COVERAGE_COLOR, "file": "coverage.png"},
        "layers": layers,
        "workers": workers,
    }
    for fname, png in pngs.items():
        _upload(str(status / fname), png, "image/png")
    html = PAGE_TEMPLATE.replace("__TITLE__", title).replace(
        "__DATA__", json.dumps(data)
    )
    _upload(str(status / "index.html"), html.encode(), "text/html; charset=utf-8")
    logger.info(
        "published status to %s: %s",
        status_path,
        {layer["year"]: layer["done"] for layer in layers},
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root{--paper:#f4f2ed;--panel:#fff;--ink:#191d1b;--muted:#6b746e;--rule:#d6d1c4;}
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --paper:#13161a;--panel:#1b2024;--ink:#eaefec;--muted:#98a29b;--rule:#2b3237;}}
  :root[data-theme="dark"]{--paper:#13161a;--panel:#1b2024;--ink:#eaefec;
    --muted:#98a29b;--rule:#2b3237;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);
    font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;}
  main{max-width:1240px;margin:0 auto;padding:20px 16px;display:flex;
    flex-direction:column;gap:14px;}
  header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
    flex-wrap:wrap;border-bottom:2px solid var(--ink);padding-bottom:8px;}
  h1{font-size:clamp(19px,2.4vw,26px);margin:0;letter-spacing:-.01em;}
  .updated{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;}
  .stats{display:flex;flex-wrap:wrap;gap:8px;}
  .stat{flex:1 1 120px;background:var(--panel);border:1px solid var(--rule);
    padding:10px 14px;}
  .stat .n{display:block;font-size:clamp(19px,2.4vw,27px);font-weight:600;
    font-variant-numeric:tabular-nums;letter-spacing:-.02em;}
  .stat .l{display:block;font-size:12px;color:var(--muted);}
  .stat .sw{display:inline-block;width:10px;height:10px;border-radius:2px;
    margin-right:6px;vertical-align:baseline;}
  .fleet{margin:-4px 0 0;font-size:12px;color:var(--muted);}
  #map{height:min(70vh,640px);border:1px solid var(--rule);background:var(--panel);}
  .leaflet-control-layers-overlays label{font-variant-numeric:tabular-nums;}
  .leaflet-control-layers-overlays .sw{display:inline-block;width:10px;height:10px;
    border-radius:2px;margin:0 6px 0 2px;vertical-align:middle;}
  .leaflet-image-layer{image-rendering:pixelated;}
</style>
</head>
<body>
<main>
  <header><h1 id="title"></h1><span class="updated" id="updated"></span></header>
  <div class="stats" id="stats"></div>
  <p class="fleet" id="fleet" hidden></p>
  <div id="map"></div>
</main>
<script>
const DATA = __DATA__;
const fmt = n => n.toLocaleString("en-US");
document.getElementById("title").textContent = DATA.title;
document.getElementById("updated").textContent = "updated " + DATA.updated;

const stats = document.getElementById("stats");
function stat(value, label, color) {
  const cell = document.createElement("div");
  cell.className = "stat";
  const n = document.createElement("span");
  n.className = "n";
  n.textContent = value;
  if (color) n.style.color = color;
  const l = document.createElement("span");
  l.className = "l";
  if (color) {
    const sw = document.createElement("i");
    sw.className = "sw";
    sw.style.background = color;
    l.appendChild(sw);
  }
  l.appendChild(document.createTextNode(label));
  cell.append(n, l);
  stats.appendChild(cell);
}
function pct(layer) {
  const value = 100 * layer.done / layer.total;
  return value > 0 && value < 0.1 ? "<0.1%" : value.toFixed(1) + "%";
}
stat(fmt(DATA.gpu_hours), "GPU-hours spent");
stat(DATA.km2_per_gpu_hour == null ? "\\u2013" : fmt(DATA.km2_per_gpu_hour),
     "km\\u00b2 per GPU-hour");
const workers = DATA.workers;
if (workers) {
  stat(fmt(workers.working), "GPUs working");
  const allocated = workers.allocated.filter(([, n]) => n > 0)
    .map(([priority, n]) => fmt(n) + " " + priority);
  const parts = [];
  if (allocated.length) parts.push(allocated.join(" and ") + " in our allocation");
  if (workers.spare) parts.push(fmt(workers.spare) + " on spare capacity");
  if (workers.outside) parts.push(fmt(workers.outside) + " outside Beaker");
  let line = "At last check: " + (parts.length ? parts.join(", ") : "none running");
  if (workers.waiting) line += "; " + fmt(workers.waiting) + " waiting for a slot";
  const fleetLine = document.getElementById("fleet");
  fleetLine.textContent = line + ".";
  fleetLine.hidden = false;
}
const started = DATA.layers.filter(layer => layer.done > 0);
for (const layer of [...started].reverse()) {
  stat(pct(layer), layer.year + " complete", layer.color);
}

// The view and layer choices survive the periodic reload, per tab. Storage can be
// unavailable (private windows, blocked site data), so every access is guarded.
const VIEW_KEY = "status-view";
let saved = null;
try { saved = JSON.parse(sessionStorage.getItem(VIEW_KEY)); } catch (e) {}

const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const map = L.map("map", {worldCopyJump: true, minZoom: 1, maxZoom: 9})
  .setView(saved ? saved.center : [20, 0], saved ? saved.zoom : 2);
L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/" +
    (dark ? "World_Dark_Gray_Base" : "World_Light_Gray_Base") +
    "/MapServer/tile/{z}/{y}/{x}",
  {attribution: "Tiles &copy; Esri", maxNativeZoom: 16}
).addTo(map);
const world = [[-85.0511, -180], [85.0511, 180]];
const bust = "?v=" + DATA.version;
// Fixed z-order, oldest year on top, so toggling a layer does not restack it.
const overlay = (file, opacity, zIndex) =>
  L.imageOverlay(file + bust, world, {opacity: opacity, zIndex: zIndex, interactive: false});
const swatch = (color, text) =>
  '<i class="sw" style="background:' + color + '"></i>' + text;

const overlays = {};
const byKey = {coverage: overlay(DATA.coverage.file, 0.45, 1)};
overlays[swatch(DATA.coverage.color, "coverage area")] = byKey.coverage;
// Listed newest first. DATA.layers runs oldest first, so the oldest gets the top z.
DATA.layers.forEach((layer, i) => { layer.z = 1 + DATA.layers.length - i; });
for (const layer of [...DATA.layers].reverse()) {
  const label = layer.done > 0 ? layer.year + " (" + pct(layer) + ")" : String(layer.year);
  const lyr = overlay(layer.file, 0.85, layer.z);
  byKey[layer.year] = lyr;
  overlays[swatch(layer.color, label)] = lyr;
}
// By default the coverage area and every year with work are on.
const shown = saved ? new Set(saved.shown) : new Set(
  ["coverage", ...DATA.layers.filter(l => l.done > 0).map(l => String(l.year))]);
for (const [key, lyr] of Object.entries(byKey)) {
  if (shown.has(String(key))) lyr.addTo(map);
}
L.control.layers(null, overlays, {collapsed: innerWidth < 700}).addTo(map);

// Reload every five minutes to pick up the latest build. A hidden tab waits until
// it is shown again rather than reloading in the background.
const RELOAD_MS = 5 * 60 * 1000;
let due = false;
function reload() {
  try {
    sessionStorage.setItem(VIEW_KEY, JSON.stringify({
      center: map.getCenter(),
      zoom: map.getZoom(),
      shown: Object.keys(byKey).filter(key => map.hasLayer(byKey[key])),
    }));
  } catch (e) {}
  location.reload();
}
setInterval(() => { if (document.hidden) due = true; else reload(); }, RELOAD_MS);
document.addEventListener("visibilitychange", () => {
  if (due && !document.hidden) reload();
});
</script>
</body>
</html>
"""
