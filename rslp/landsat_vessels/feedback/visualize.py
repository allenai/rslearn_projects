"""Render a candidates-grid PNG of feedback false positives.

Same visual style as ``scripts/run_recent_inference.py`` (a grid of pan-sharpened RGB
crops with a coloured border and per-cell labels), but the crops are rendered from the
materialised feedback windows rather than from a live pipeline run -- Skylight's own
``image_chips`` are transient and get cleaned up, so they cannot be relied on.

Each crop is built exactly as the production pipeline builds its detection crops
(``predict_pipeline._write_detection_crop``): B2/B3/B4/B8 remapped ``(DN-5000)*255/12000``
to 8-bit, then a Brovey pan-sharpen ``band*B8/mean(B2,B3,B4)`` using the 15 m B8. Borders
are red because every window here is a human-confirmed false positive.

Only numpy/rasterio/PIL are imported (no torch), so it runs without the heavy model stack.

Usage:
    python -m rslp.landsat_vessels.feedback.visualize \
        --csv   /weka/dfive-default/yawenz/landsat/feedback_20260911/feedback_20260911.csv \
        --group feedback_20260911 \
        --out   /weka/dfive-default/yawenz/landsat/feedback_20260911/figures/fp_grid.png
"""

import argparse
import csv as _csv
import os
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling

from rslp.landsat_vessels.feedback import config

# Band order stored in the classifier dataset's 10-band "landsat" raster set.
TEN_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
COLOR_FP = (220, 50, 50)  # red: false positive (should have been rejected)


def _font(size: int) -> Any:
    """Load a DejaVu TrueType font at the given size, or PIL's default."""
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _remap(raw: np.ndarray) -> np.ndarray:
    """The pipeline's DN->8-bit remap: (DN-5000)*255/12000, clipped."""
    return np.clip((raw.astype(np.float32) - 5000.0) * 255.0 / 12000.0, 0, 255)


def render_crop(window_dir: str, size: int) -> Image.Image | None:
    """Pan-sharpened true-colour crop for one feedback window, or None if unreadable.

    Reads B8 (15 m) and B2/B3/B4 (30 m, upsampled to B8's grid with nearest neighbour),
    then applies the same remap + Brovey pan-sharpen as the production pipeline.
    """
    b8_path = os.path.join(window_dir, "layers", "landsat", "B8", "geotiff.tif")
    ten_path = os.path.join(
        window_dir, "layers", "landsat", "_".join(TEN_BANDS), "geotiff.tif"
    )
    if not (os.path.exists(b8_path) and os.path.exists(ten_path)):
        return None
    with rasterio.open(b8_path) as src:
        b8 = _remap(src.read(1))
    h, w = b8.shape
    rgb_bands = {}
    with rasterio.open(ten_path) as src:
        for band in ("B2", "B3", "B4"):
            arr = src.read(
                TEN_BANDS.index(band) + 1,
                out_shape=(h, w),
                resampling=Resampling.nearest,
            )
            rgb_bands[band] = _remap(arr)
    total = np.clip((rgb_bands["B2"] + rgb_bands["B3"] + rgb_bands["B4"]) / 3.0, 1, 255)
    sharp = {
        band: np.clip(rgb_bands[band] * b8 / total, 0, 255).astype(np.uint8)
        for band in ("B2", "B3", "B4")
    }
    rgb = np.stack([sharp["B4"], sharp["B3"], sharp["B2"]], axis=2)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


def main() -> None:
    """Render the false-positive candidates grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, help="feedback CSV (for det score/order)"
    )
    parser.add_argument(
        "--group", required=True, help="window group, e.g. feedback_20260911"
    )
    parser.add_argument("--dataset-root", default=str(config.DATASET_ROOT))
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--cell", type=int, default=150)
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(_csv.DictReader(f))

    # Group by scene, then by detector score desc, so same-scene FPs sit together.
    def sort_key(r: dict) -> tuple:
        """Order rows by scene, then by detector score descending."""
        return (r["scene_id"], -float(r.get("score") or 0.0))

    rows.sort(key=sort_key)
    windows_root = os.path.join(args.dataset_root, "windows", args.group)
    print(f"rendering {len(rows)} FP crops from {windows_root}")

    pad, label_h, legend_h = 6, 34, 42
    cell, cols = args.cell, args.cols
    n = max(len(rows), 1)
    grid_rows = (n + cols - 1) // cols
    cw = cell + 2 * pad
    ch = cell + label_h + 2 * pad
    grid = Image.new("RGB", (cols * cw, legend_h + grid_rows * ch), (25, 25, 25))
    draw = ImageDraw.Draw(grid)
    font, small, legend_font = _font(15), _font(13), _font(16)

    versions = sorted(
        {r.get("model_version", "") for r in rows if r.get("model_version")}
    )
    header = (
        f"{len(rows)} false positives (red = human-labelled BAD)  "
        f"model: {', '.join(versions) or 'n/a'}  group: {args.group}"
    )
    draw.rectangle([8, 12, 26, 30], fill=COLOR_FP)
    draw.text((32, 12), header, fill=(230, 230, 230), font=legend_font)

    rendered = 0
    for i, r in enumerate(rows):
        gx = (i % cols) * cw
        gy = legend_h + (i // cols) * ch
        window_dir = os.path.join(windows_root, r["event_id"])
        im = render_crop(window_dir, cell)
        if im is None:
            im = Image.new("RGB", (cell, cell), (60, 60, 60))
        else:
            rendered += 1
        grid.paste(im, (gx + pad, gy + pad + label_h))
        draw.rectangle(
            [
                gx + pad - 2,
                gy + pad + label_h - 2,
                gx + pad + cell + 1,
                gy + pad + label_h + cell + 1,
            ],
            outline=COLOR_FP,
            width=3,
        )
        parts = r["scene_id"].split("_")
        pathrow, tier = parts[2], parts[-1]
        idx = r["event_id"].rsplit("_", 1)[-1]
        det = f"{float(r['score']):.2f}" if r.get("score") else "n/a"
        draw.text((gx + pad, gy + 2), f"#{i} det={det}", fill=COLOR_FP, font=font)
        draw.text(
            (gx + pad, gy + 18),
            f"{pathrow}_{tier} #{idx}",
            fill=(200, 200, 200),
            font=small,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid.save(args.out)
    print(f"rendered {rendered}/{len(rows)} crops (rest were unreadable/placeholder)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
