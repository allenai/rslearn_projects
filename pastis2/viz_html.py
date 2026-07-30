"""Build a single self-contained interactive HTML QA viewer for the PASTIS2 samples.

6 region tabs at the top; clicking one shows N samples for that region. Each sample shows
the full 12-month Sentinel-2 time series (RGB) + one Sentinel-1 (vv) + the label mask.
Each tile is embedded as its OWN full-resolution PNG (base64) and laid out large via CSS
(horizontally scrollable strip; masks rendered crisp), so quality is higher than a single
squashed composite. Read-only; uses the expanded (Coffee/Cacao) class map.
Output: viz_qa/pastis2_qa.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

import viz_qa_samples as V  # rasterize_label (expanded map), window_geo, territory_of, read_s1, NAMES, RGB_IDX

REGIONS = ["metropole", "guadeloupe", "martinique", "guyane", "reunion", "mayotte"]
LABELS = {"metropole": "Metropole", "guadeloupe": "Guadeloupe", "martinique": "Martinique",
          "guyane": "Guyane", "reunion": "Réunion", "mayotte": "Mayotte"}

palette = np.vstack([
    plt.cm.tab20(np.linspace(0, 1, 20)),
    plt.cm.tab20b(np.linspace(0, 1, 20)),
    plt.cm.tab20c(np.linspace(0, 1, 20)),
])
COLORS = np.vstack([[0.85, 0.85, 0.85, 1], palette[1:V.NCLS]])[:V.NCLS]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(np.arange(-0.5, V.NCLS + 0.5, 1), CMAP.N)


def _uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def rgb_uri(arr: np.ndarray | None) -> str:
    buf = io.BytesIO()
    if arr is None:
        arr = np.full((128, 128, 3), 0.15, np.float32)
    plt.imsave(buf, np.clip(arr, 0, 1), format="png")
    return _uri(buf.getvalue())


def gray_uri(arr: np.ndarray | None) -> str:
    buf = io.BytesIO()
    if arr is None:
        plt.imsave(buf, np.full((128, 128), 0.15), cmap="gray", vmin=0, vmax=1, format="png")
    else:
        plt.imsave(buf, np.clip(arr, 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
    return _uri(buf.getvalue())


def mask_uri(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    plt.imsave(buf, CMAP(NORM(mask)), format="png")
    return _uri(buf.getvalue())


def s2_series(wdir: Path):
    out = []
    for i in range(12):
        layer = "sentinel2" if i == 0 else f"sentinel2.{i}"
        tifs = glob.glob(str(wdir / "layers" / layer / "*" / "geotiff.tif"))
        if not tifs:
            out.append(None); continue
        with rasterio.open(tifs[0]) as src:
            arr = src.read().astype(np.float32)
        out.append(np.clip(arr[V.RGB_IDX].transpose(1, 2, 0) / 3000.0, 0, 1))
    return out


def sample_html(wdir: Path, terr: str, mask: np.ndarray, posfrac: float) -> str:
    s2 = s2_series(wdir)
    s1 = V.read_s1(wdir)
    tiles = [f'<figure><img class="img mask" src="{mask_uri(mask)}"><figcaption>label</figcaption></figure>']
    for i in range(12):
        tiles.append(f'<figure><img class="img" src="{rgb_uri(s2[i])}"><figcaption>M{i+1}</figcaption></figure>')
    tiles.append(f'<figure><img class="img" src="{gray_uri(s1)}"><figcaption>S1 vv</figcaption></figure>')
    name = Path(wdir).name.split("_2018")[0]
    return (
        f'<div class="sample"><div class="cap">{LABELS[terr]} &middot; {name} '
        f'&middot; positive {posfrac*100:.0f}%</div><div class="strip">{"".join(tiles)}</div></div>'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-region", type=int, default=20)
    ap.add_argument("--out", default=str(V.HERE / "viz_qa" / "pastis2_qa.html"))
    args = ap.parse_args()

    by: dict[str, list] = {r: [] for r in REGIONS}
    for wdir in sorted(glob.glob(str(V.DS / "windows" / V.GROUP / "*"))):
        wdir = Path(wdir)
        if not (wdir / "metadata.json").exists():
            continue
        if not glob.glob(str(wdir / "layers" / "sentinel2*" / "*" / "geotiff.tif")):
            continue
        meta = json.load(open(wdir / "metadata.json"))
        _, _, lon, lat = V.window_geo(meta)
        terr = V.territory_of(lon, lat)
        if terr in by:
            by[terr].append((wdir, meta))

    sections = {}
    for terr in REGIONS:
        scored = []
        for wdir, meta in by[terr]:
            mask = V.rasterize_label(meta, terr)
            posfrac = float(((mask >= 1) & (mask <= V.ORIG_MAX)).mean())
            scored.append((posfrac, wdir, meta, mask))
        scored.sort(key=lambda x: -x[0])
        sections[terr] = "\n".join(
            sample_html(wdir, terr, mask, posfrac)
            for posfrac, wdir, meta, mask in scored[: args.per_region]
        )
        print(f"{terr}: {len(by[terr])} windows, embedded {min(len(scored), args.per_region)}")

    legend = "".join(
        f'<span class="sw"><i style="background:rgb({int(COLORS[c][0]*255)},{int(COLORS[c][1]*255)},{int(COLORS[c][2]*255)})"></i>{c} {V.NAMES.get(c, "?")}</span>'
        for c in range(V.NCLS)
    )
    btns = "".join(f'<button onclick="show(\'{r}\')" id="b_{r}">{LABELS[r]}</button>' for r in REGIONS)
    region_divs = "".join(
        f'<div id="r_{r}" class="region" style="display:{"block" if i == 0 else "none"}">'
        f'<h2>{LABELS[r]}</h2>{sections[r]}</div>'
        for i, r in enumerate(REGIONS)
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PASTIS2 QA</title>
<style>
body{{font-family:sans-serif;margin:12px;background:#fafafa}}
.tabs button{{font-size:15px;padding:8px 14px;margin:2px;cursor:pointer;border:1px solid #ccc;background:#eee;border-radius:6px}}
.tabs button.active{{background:#2b6cb0;color:#fff;border-color:#2b6cb0}}
.legend{{margin:10px 0;font-size:11px;line-height:1.9}}
.sw{{display:inline-block;margin-right:10px;white-space:nowrap}}
.sw i{{display:inline-block;width:12px;height:12px;margin-right:3px;vertical-align:middle;border:1px solid #999}}
.sample{{margin:14px 0;padding:6px;background:#fff;border:1px solid #e2e2e2;border-radius:6px}}
.cap{{font-size:12px;color:#333;margin-bottom:4px;font-weight:bold}}
.strip{{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px}}
.strip figure{{margin:0;flex:0 0 auto;text-align:center}}
.img{{width:200px;height:200px;image-rendering:auto;border:1px solid #ddd}}
.mask{{image-rendering:pixelated}}
.strip figcaption{{font-size:11px;color:#666}}
h2{{color:#2b6cb0}}
</style></head><body>
<h1>PASTIS2 QA — 12-month S2 series · S1 · label mask</h1>
<div class="tabs">{btns}</div>
<div class="legend"><b>classes:</b><br>{legend}</div>
{region_divs}
<script>
function show(r){{
  document.querySelectorAll('.region').forEach(d=>d.style.display='none');
  document.getElementById('r_'+r).style.display='block';
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  document.getElementById('b_'+r).classList.add('active');
}}
document.getElementById('b_{REGIONS[0]}').classList.add('active');
</script></body></html>"""
    out = Path(args.out)
    out.write_text(html)
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
