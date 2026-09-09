"""Shared imagery helpers for the iter4+ prompt experiments.

Two capabilities the earlier iterations lacked:

1. ``label_image_box`` renders an image exactly like ``prompt.label_image`` but draws a
   thin hollow box around the CENTER of the frame, so the VLM knows precisely which
   location to judge (the windows are centered on the point of interest, but the change
   of interest is often small relative to the 640 m frame).

2. ``select_clearest_per_year`` replaces the iter0-2 "image nearest to July 1" rule with
   a cloud-aware selection: of the ~12 monthly Sentinel-2 mosaics available per year, it
   scores how clear the CENTER of each is and keeps the clearest one or two (preferring
   the growing-season months when a clear one exists). This directly targets the most
   common iter1 false-negative mode, where the single per-year mosaic happened to be
   cloud-covered even though clear months existed.
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from .prompt import _LABEL_STRIP_HEIGHT, _MIN_DISPLAY_WIDTH, _get_font

# Fraction of the frame (per side) that the center box spans. The base window is 64 px
# (~640 m), so 0.34 -> ~22 px -> ~215 m box centered on the point of interest.
_BOX_FRAC = 0.34
_BOX_COLOR = (255, 0, 255)  # magenta, rare in natural imagery
_BOX_WIDTH = 2


def center_clarity(chw_array: npt.NDArray) -> float:
    """Heuristic clarity of the image CENTER in [0, 1] (higher = clearer ground view).

    Penalizes both cloud/haze/snow (bright, low-saturation, near-white pixels) and
    missing data (near-black pixels) in the central region. Pure heuristic on RGB.
    """
    arr = np.asarray(chw_array)[:3].astype(np.float32)
    _, h, w = arr.shape
    cy0, cy1 = int(h * 0.33), int(np.ceil(h * 0.67))
    cx0, cx1 = int(w * 0.33), int(np.ceil(w * 0.67))
    center = arr[:, cy0:cy1, cx0:cx1]  # (3, hh, ww)
    r, g, b = center[0], center[1], center[2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    # Cloud/snow/haze: bright and gray (low saturation).
    bright = mn > 175.0
    gray = (mx - mn) < 35.0
    cloudy = bright & gray
    # Missing / nodata: essentially black.
    black = mx < 12.0
    bad = cloudy | black
    return float(1.0 - bad.mean())


def select_clearest_per_year(metas, layer_name, k, load_array):
    """Up to ``k`` clearest Sentinel-2 images per year (preferring May-Sep).

    For each calendar year, score every image's center clarity and keep the ``k`` with
    the highest clarity. Growing-season months (May-September) get a small clarity bonus
    so that, when clear imagery exists in multiple seasons, the series stays seasonally
    consistent and easier to compare across years.
    """
    by_year: dict[int, list] = {}
    for m in metas:
        if m.layer_name != layer_name:
            continue
        by_year.setdefault(m.time_range[0].year, []).append(m)

    chosen = []
    for year in sorted(by_year):
        scored = []
        for m in by_year[year]:
            clarity = center_clarity(load_array(m))
            in_season = 5 <= m.time_range[0].month <= 9
            rank = clarity + (0.05 if in_season else 0.0)
            scored.append((rank, m.time_range[0], m))
        # Highest rank first; break ties toward earlier-in-year for stability.
        scored.sort(key=lambda t: (-t[0], t[1]))
        chosen.extend(m for _, _, m in scored[:k])
    chosen.sort(key=lambda m: m.time_range[0])
    return chosen


def label_image_box(chw_array: npt.NDArray, caption: str) -> bytes:
    """Like ``prompt.label_image`` but draw a hollow box around the center."""
    hwc = np.transpose(np.asarray(chw_array), (1, 2, 0))
    hwc = np.clip(hwc[:, :, :3], 0, 255).astype(np.uint8)
    image = Image.fromarray(hwc, "RGB")

    if image.width < _MIN_DISPLAY_WIDTH:
        scale = max(1, _MIN_DISPLAY_WIDTH // image.width)
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)

    draw = ImageDraw.Draw(image)
    w, hgt = image.width, image.height
    bw, bh = w * _BOX_FRAC, hgt * _BOX_FRAC
    x0, y0 = (w - bw) / 2.0, (hgt - bh) / 2.0
    x1, y1 = (w + bw) / 2.0, (hgt + bh) / 2.0
    draw.rectangle([x0, y0, x1, y1], outline=_BOX_COLOR, width=_BOX_WIDTH)

    out = Image.new("RGB", (image.width, image.height + _LABEL_STRIP_HEIGHT), (0, 0, 0))
    out.paste(image, (0, 0))
    od = ImageDraw.Draw(out)
    od.text((4, image.height + 4), caption, fill=(255, 255, 255), font=_get_font())

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


# --- iter8+: pinpoint the exact center pixel (point labels are ~10 m, one S2 pixel) ---
# The label describes only the single ~10 m pixel at the frame center, not the ~215 m box
# that ``label_image_box`` draws. These helpers draw a crosshair reticle with an OPEN
# center gap (so the target pixel stays visible) and optionally a magnified center crop.
_GAP_FRAC = 0.05  # half-size of the open gap at the center (fraction of frame)
_TICK_FRAC = 0.14  # length of each crosshair tick beyond the gap (fraction of frame)


def _draw_crosshair(image: "Image.Image") -> None:
    """Draw a magenta crosshair reticle pointing at the exact center, gap left open."""
    draw = ImageDraw.Draw(image)
    w, hgt = image.width, image.height
    cx, cy = w / 2.0, hgt / 2.0
    gap = min(w, hgt) * _GAP_FRAC
    tick = min(w, hgt) * _TICK_FRAC
    # Four ticks pointing inward toward the center (open in the middle).
    draw.line([cx, cy - gap - tick, cx, cy - gap], fill=_BOX_COLOR, width=_BOX_WIDTH)
    draw.line([cx, cy + gap, cx, cy + gap + tick], fill=_BOX_COLOR, width=_BOX_WIDTH)
    draw.line([cx - gap - tick, cy, cx - gap, cy], fill=_BOX_COLOR, width=_BOX_WIDTH)
    draw.line([cx + gap, cy, cx + gap + tick, cy], fill=_BOX_COLOR, width=_BOX_WIDTH)
    # Tiny hollow box marking the exact target pixel(s).
    draw.rectangle([cx - gap, cy - gap, cx + gap, cy + gap], outline=_BOX_COLOR, width=1)


def _to_rgb_image(chw_array: npt.NDArray) -> "Image.Image":
    hwc = np.transpose(np.asarray(chw_array), (1, 2, 0))
    hwc = np.clip(hwc[:, :, :3], 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, "RGB")


def _with_caption(image: "Image.Image", caption: str) -> bytes:
    out = Image.new("RGB", (image.width, image.height + _LABEL_STRIP_HEIGHT), (0, 0, 0))
    out.paste(image, (0, 0))
    od = ImageDraw.Draw(out)
    od.text((4, image.height + 4), caption, fill=(255, 255, 255), font=_get_font())
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def label_image_crosshair(chw_array: npt.NDArray, caption: str) -> bytes:
    """Render the frame with a crosshair reticle pinpointing the exact center pixel."""
    image = _to_rgb_image(chw_array)
    if image.width < _MIN_DISPLAY_WIDTH:
        scale = max(1, _MIN_DISPLAY_WIDTH // image.width)
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    _draw_crosshair(image)
    return _with_caption(image, caption)


def label_image_zoom_pair(
    chw_array: npt.NDArray, caption: str, crop_frac: float = 0.4
) -> bytes:
    """Two panels side by side: full-frame context (left) and a magnified center crop
    (right), each with a crosshair on the exact center. ``crop_frac`` is the fraction of
    the frame kept in the zoom panel (0.4 -> central ~256 m of the ~640 m frame)."""
    base = _to_rgb_image(chw_array)
    w0, h0 = base.width, base.height
    # Center crop in native resolution, then upscale to the display size.
    cw, ch = max(1, int(round(w0 * crop_frac))), max(1, int(round(h0 * crop_frac)))
    x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
    crop = base.crop((x0, y0, x0 + cw, y0 + ch))

    disp = max(_MIN_DISPLAY_WIDTH, w0)
    full = base.resize((disp, disp), Image.NEAREST)
    zoom = crop.resize((disp, disp), Image.NEAREST)
    _draw_crosshair(full)
    _draw_crosshair(zoom)

    gap = 8
    canvas = Image.new("RGB", (disp * 2 + gap, disp), (0, 0, 0))
    canvas.paste(full, (0, 0))
    canvas.paste(zoom, (disp + gap, 0))
    cap = f"{caption}  [left: full ~640 m context | right: zoom of exact center]"
    return _with_caption(canvas, cap)


def _draw_center_box(image: "Image.Image", box_frac: float) -> None:
    """Draw a small hollow magenta box of ``box_frac`` of the panel, centered."""
    draw = ImageDraw.Draw(image)
    w, hgt = image.width, image.height
    bw, bh = w * box_frac, hgt * box_frac
    draw.rectangle(
        [(w - bw) / 2.0, (hgt - bh) / 2.0, (w + bw) / 2.0, (hgt + bh) / 2.0],
        outline=_BOX_COLOR,
        width=_BOX_WIDTH,
    )


def label_image_zoom_box(
    chw_array: npt.NDArray,
    caption: str,
    crop_frac: float = 0.4,
    box_frac: float = 0.12,
) -> bytes:
    """Two panels: full-frame context (left) and a magnified center crop (right), each
    with a SMALL box (geographically ``box_frac`` of the full ~640 m frame) marking the
    labeled point neighborhood. ``crop_frac`` is the fraction of the frame kept in the
    zoom panel; the box is scaled so it covers the same ground area in both panels."""
    base = _to_rgb_image(chw_array)
    w0, h0 = base.width, base.height
    cw, ch = max(1, int(round(w0 * crop_frac))), max(1, int(round(h0 * crop_frac)))
    x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
    crop = base.crop((x0, y0, x0 + cw, y0 + ch))

    disp = max(_MIN_DISPLAY_WIDTH, w0)
    full = base.resize((disp, disp), Image.NEAREST)
    zoom = crop.resize((disp, disp), Image.NEAREST)
    _draw_center_box(full, box_frac)
    # Same ground area is a larger fraction of the zoomed panel.
    _draw_center_box(zoom, min(0.9, box_frac / crop_frac))

    gap = 8
    canvas = Image.new("RGB", (disp * 2 + gap, disp), (0, 0, 0))
    canvas.paste(full, (0, 0))
    canvas.paste(zoom, (disp + gap, 0))
    cap = f"{caption}  [left: full ~640 m context | right: zoom of the marked point]"
    return _with_caption(canvas, cap)
