"""Render the annotation views for the round-1 windows.

Reads each materialized 512 px @ 15 m window and writes the images the annotation app
serves, plus a compiled spectral-curve file. Nothing here touches S3: the window
rasters on weka are the only input, so this can be re-run freely to change the rendering.

Per window, four images:

- ``crop_fixed.png`` (128 px) — pan-sharpened RGB with the production stretch,
  ``clip((DN - 5000) * 255/12000)``, pan-sharpened exactly as
  ``predict_pipeline._write_detection_crop`` does. This is pixel-for-pixel what the
  detector's own crop looks like, so an annotator judging a production false positive
  sees what production saw.
- ``crop_auto.png`` (128 px) — same pipeline, but the DN window comes from the 2-98th
  percentile of this crop's own RGB. The fixed stretch is a DN window, not a
  reflectance one, so low-sun and dark-water crops sit at the bottom of it and read as
  near-black even where the bands separate cleanly. One shared range across R/G/B
  rather than three per-band ranges, so colour ratios survive the stretch.
- ``b8.png`` (128 px) — the panchromatic band alone at its native 15 m, auto-stretched.
  The sharpest single band, and the one where a small hull reads most clearly.
- ``zoom_auto.webp``, ``zoom_fixed.webp``, ``zoom_b8.webp`` (512 px) — the full 7.68 km
  context window in each of the same three renderings, so switching the view in the app
  switches both panels together. The auto stretch here is computed over the whole 512 px
  window rather than the crop, so the surrounding scene reads even when the crop is dark.

The detection marker and the crop-extent box are drawn by the app in CSS rather than
baked in, so no pixel of these images is anything but sensor data.

Usage:
    python render_annotation_assets.py [--workers 16] [--limit N] [--overwrite]
"""

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
import tqdm
from PIL import Image
from rasterio.enums import Resampling
from upath import UPath

DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
ROUND_DIR = UPath("/weka/dfive-default/yawenz/landsat/annotation_round1")
SPECTRAL_CACHE = UPath("/weka/dfive-default/yawenz/landsat/spectral/cache")

GROUP = "round1_20260803"
LAYER_NAME = "landsat"
MULTIBAND_DIR = "B1_B2_B3_B4_B5_B6_B7_B9_B10_B11"
MULTIBAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]

WINDOW_SIZE = 512
CROP_SIZE = 128

# Every file a finished window must have. A window missing any of them is re-rendered, so
# adding a view here is enough to make the next run backfill it.
EXPECTED_ASSETS = [
    "crop_auto.png",
    "crop_fixed.png",
    "b8.png",
    "zoom_auto.webp",
    "zoom_fixed.webp",
    "zoom_b8.webp",
]
# Superseded by zoom_auto.webp; removed on re-render so it stops taking up space.
LEGACY_ASSETS = ["zoom.webp"]

# The production stretch from predict_pipeline._write_detection_crop.
FIXED_DN_LO = 5000.0
FIXED_DN_SPAN = 12000.0
# Percentiles for the auto stretch, and the floor on the resulting span so that a crop
# of near-uniform water does not get amplified into pure noise.
AUTO_PERCENTILES = (2.0, 98.0)
MIN_AUTO_SPAN = 500.0

# Bands plotted on the reflectance curve, in wavelength order. B8 and B9 are sampled but
# left off: B8 is one wide passband spanning B2-B4 so it is not an independent spectral
# point, and B9 sits in a water-vapour absorption feature where surface targets read
# near zero. Matches spectral/build_artifact.py.
REFL_BANDS = [
    ("B1", 443),
    ("B2", 482),
    ("B3", 561),
    ("B4", 654),
    ("B5", 865),
    ("B6", 1609),
    ("B7", 2201),
]
THERMAL_BANDS = [("B10", 10895), ("B11", 12005)]
POINTS = ["center", "tl", "tr", "bl", "br"]


def scale_to_bytes(dn: np.ndarray, lo: float, span: float) -> np.ndarray:
    """Map DN to 0-255 over the window [lo, lo + span]."""
    return np.clip((dn.astype(np.float32) - lo) * 255.0 / span, 0, 255).astype(np.uint8)


def auto_range(dn: np.ndarray) -> tuple[float, float]:
    """A shared DN window from the 2-98th percentile of the valid pixels.

    Zero is Landsat's fill value, so off-swath corners are excluded rather than dragging
    the low end to 0 and flattening everything above it.
    """
    valid = dn[dn > 0]
    if valid.size < 16:
        return FIXED_DN_LO, FIXED_DN_SPAN
    lo, hi = np.percentile(valid, AUTO_PERCENTILES)
    return float(lo), max(float(hi - lo), MIN_AUTO_SPAN)


def pansharpen(rgb8: dict[str, np.ndarray], b8_8: np.ndarray) -> np.ndarray:
    """Pan-sharpen 8-bit R/G/B with the 8-bit panchromatic band.

    The production formula: scale each band so the three sum to B8, which carries the
    15 m detail the 30 m bands do not.
    """
    total = np.clip(
        (
            rgb8["B2"].astype(np.int32)
            + rgb8["B3"].astype(np.int32)
            + rgb8["B4"].astype(np.int32)
        )
        // 3,
        1,
        255,
    )
    sharp = {
        band: np.clip(rgb8[band].astype(np.int32) * b8_8 // total, 0, 255).astype(
            np.uint8
        )
        for band in ("B2", "B3", "B4")
    }
    return np.stack([sharp["B4"], sharp["B3"], sharp["B2"]], axis=2)


def centre_crop(array: np.ndarray, size: int) -> np.ndarray:
    """Take the centre size x size of an array whose first two axes are y, x."""
    start = (array.shape[0] - size) // 2
    return array[start : start + size, start : start + size]


def render_window(record: dict, overwrite: bool) -> dict:
    """Render one window's four images. Returns a status record."""
    name = record["window"]
    out_dir = ROUND_DIR / "assets" / name
    if not overwrite and all(
        (out_dir / expected).exists() for expected in EXPECTED_ASSETS
    ):
        return {"window": name, "status": "cached"}

    window_dir = DATASET_ROOT / "windows" / GROUP / name / "layers" / LAYER_NAME
    if not (window_dir / "completed").exists():
        return {"window": name, "status": "not_materialized"}

    # Read the 30 m bands resampled up to the window's 15 m grid with nearest
    # neighbour, the same resampling the production pipeline uses, so the crop is not
    # blurred by interpolation that the detector never saw.
    with rasterio.open(window_dir / MULTIBAND_DIR / "geotiff.tif") as src:
        multiband = src.read(
            out_shape=(src.count, WINDOW_SIZE, WINDOW_SIZE),
            resampling=Resampling.nearest,
        )
    with rasterio.open(window_dir / "B8" / "geotiff.tif") as src:
        b8 = src.read(1)

    dn = {band: multiband[i] for i, band in enumerate(MULTIBAND_ORDER)}
    dn["B8"] = b8

    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed-stretch crop: identical arithmetic to the production crop writer.
    crop_dn = {
        band: centre_crop(dn[band], CROP_SIZE) for band in ("B2", "B3", "B4", "B8")
    }
    fixed8 = {
        band: scale_to_bytes(crop_dn[band], FIXED_DN_LO, FIXED_DN_SPAN)
        for band in ("B2", "B3", "B4", "B8")
    }
    with (out_dir / "crop_fixed.png").open("wb") as f:
        Image.fromarray(pansharpen(fixed8, fixed8["B8"])).save(f, format="PNG")

    # Auto-stretch crop: one shared RGB range, and B8 on its own range since it is a
    # different (wider) passband and would otherwise wash out the sharpening.
    rgb_stack = np.concatenate([crop_dn[b].ravel() for b in ("B2", "B3", "B4")])
    rgb_lo, rgb_span = auto_range(rgb_stack)
    b8_lo, b8_span = auto_range(crop_dn["B8"])
    auto8 = {
        band: scale_to_bytes(crop_dn[band], rgb_lo, rgb_span)
        for band in ("B2", "B3", "B4")
    }
    auto_b8 = scale_to_bytes(crop_dn["B8"], b8_lo, b8_span)
    with (out_dir / "crop_auto.png").open("wb") as f:
        Image.fromarray(pansharpen(auto8, auto_b8)).save(f, format="PNG")
    with (out_dir / "b8.png").open("wb") as f:
        Image.fromarray(auto_b8).save(f, format="PNG")

    # Zoom-out, in all three renderings so the app can switch both panels together.
    # The auto range here is the whole window's, not the crop's, so the surrounding scene
    # reads even when the crop itself is dark.
    zoom_rgb_lo, zoom_rgb_span = auto_range(
        np.concatenate([dn[b].ravel() for b in ("B2", "B3", "B4")])
    )
    zoom_b8_lo, zoom_b8_span = auto_range(dn["B8"])
    zoom_auto_b8 = scale_to_bytes(dn["B8"], zoom_b8_lo, zoom_b8_span)
    zoom_views = {
        "zoom_auto.webp": pansharpen(
            {
                band: scale_to_bytes(dn[band], zoom_rgb_lo, zoom_rgb_span)
                for band in ("B2", "B3", "B4")
            },
            zoom_auto_b8,
        ),
        "zoom_fixed.webp": pansharpen(
            {
                band: scale_to_bytes(dn[band], FIXED_DN_LO, FIXED_DN_SPAN)
                for band in ("B2", "B3", "B4", "B8")
            },
            scale_to_bytes(dn["B8"], FIXED_DN_LO, FIXED_DN_SPAN),
        ),
        "zoom_b8.webp": zoom_auto_b8,
    }
    for filename, array in zoom_views.items():
        with (out_dir / filename).open("wb") as f:
            Image.fromarray(array).save(f, format="WEBP", quality=90, method=4)

    for legacy in LEGACY_ASSETS:
        (out_dir / legacy).unlink(missing_ok=True)

    valid_fraction = float((dn["B8"] > 0).mean())
    return {
        "window": name,
        "status": "rendered",
        "valid_fraction": round(valid_fraction, 4),
        "crop_centre_dn_b8": int(crop_dn["B8"][CROP_SIZE // 2, CROP_SIZE // 2]),
        "auto_rgb_range": [round(rgb_lo), round(rgb_lo + rgb_span)],
    }


def render_window_safe(args: tuple[dict, bool]) -> dict:
    """Render one window's annotation assets, returning a status dict (never raises)."""
    record, overwrite = args
    try:
        return render_window(record, overwrite)
    except Exception as exc:  # noqa: BLE001 - one bad window must not stop the batch
        return {
            "window": record["window"],
            "status": "error",
            "error": f"{exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def compile_spectra(records: list[dict]) -> dict:
    """Collect the per-detection spectral curves from the spectral pipeline's cache.

    The curves were computed once (TOA reflectance with each scene's MTL coefficients,
    at-sensor brightness temperature for the thermal bands) and validated then; this
    only reshapes them for the app, keyed by window name.
    """
    wanted: dict[str, list[str]] = {}
    for record in records:
        wanted.setdefault(record["scene_id"], []).append(record["window"])

    curves: dict[str, dict] = {}
    missing_scenes = []
    for scene_id, windows in wanted.items():
        cache_path = SPECTRAL_CACHE / f"{scene_id}.json"
        if not cache_path.exists():
            missing_scenes.append(scene_id)
            continue
        with cache_path.open() as f:
            scene = json.load(f)
        for window in windows:
            detection = scene["detections"].get(window)
            if detection is None:
                continue
            curves[window] = {
                "refl": {
                    band: [detection["refl"].get(band, {}).get(p) for p in POINTS]
                    for band, _ in REFL_BANDS
                },
                "bt": {
                    band: [detection["bt"].get(band, {}).get(p) for p in POINTS]
                    for band, _ in THERMAL_BANDS
                },
                "dn": {
                    band: [detection["dn"].get(band, {}).get(p) for p in POINTS]
                    for band, _ in REFL_BANDS + THERMAL_BANDS
                },
                "sun_elevation": scene["cal"].get("sun_elevation"),
                "substituted_product": scene.get("substituted_product"),
            }
    if missing_scenes:
        print(f"  spectral cache missing for {len(missing_scenes)} scenes")
    return {
        "points": POINTS,
        "refl_bands": [{"band": b, "nm": nm} for b, nm in REFL_BANDS],
        "thermal_bands": [{"band": b, "nm": nm} for b, nm in THERMAL_BANDS],
        "curves": curves,
    }


def main() -> None:
    """Render annotation assets (crops, zoom-outs, spectral curves) for all windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-render windows that already have assets",
    )
    parser.add_argument(
        "--skip_spectra", action="store_true", help="only render images"
    )
    args = parser.parse_args()

    with (ROUND_DIR / f"{GROUP}_index.json").open() as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} windows in index")

    statuses: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(render_window_safe, (record, args.overwrite))
            for record in records
        ]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="rendering"
        ):
            statuses.append(future.result())

    counts: dict[str, int] = {}
    for status in statuses:
        counts[status["status"]] = counts.get(status["status"], 0) + 1
    print("render status:", dict(sorted(counts.items())))
    for status in statuses:
        if status["status"] == "error":
            print(f"  ERROR {status['window']}: {status['error']}")

    with (ROUND_DIR / "render_status.json").open("w") as f:
        json.dump(statuses, f)

    if not args.skip_spectra:
        spectra = compile_spectra(records)
        with (ROUND_DIR / "spectra.json").open("w") as f:
            json.dump(spectra, f)
        print(f"spectral curves for {len(spectra['curves'])} of {len(records)} windows")


if __name__ == "__main__":
    main()
