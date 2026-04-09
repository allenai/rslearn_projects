r"""Tag rslearn windows by Sentinel-2 cloudiness (pre vs post landslide event).

Walks window directories under an rslearn dataset (or a single ``windows/<group>`` folder),
runs cloud screening on each materialized ``pre_sentinel2*`` / ``post_sentinel2*`` stack
(12-band GeoTIFFs under ``layers/``), and writes results into ``metadata.json`` ``options``.

A window is tagged ``is_cloudy`` if **every** pre-event Sentinel-2 image is cloudy **or**
**every** post-event image is cloudy (per-image fraction ≥ ``--cloud_fraction_threshold``).
If there is at least one non-cloudy pre **and** at least one non-cloudy post, the window is
not tagged cloudy. Empty pre or post stacks do not count as “all cloudy” on that side.

**Cloud methods**

- ``omni`` (default): `OmniCloudMask <https://github.com/DPIRD-DMA/OmniCloudMask>`_ on
  R/G/B8A (same idea as ``rslearn.dataset.omni_cloud_mask``). Requires ``omnicloudmask``
  and a one-time model download (Hugging Face) unless weights are already cached.
- ``simple``: fast spectral heuristic (no ML): bright visible + low NDVI, for air-gapped
  or smoke tests only.

Example::

    PYTHONPATH=/path/to/rslearn_projects:/path/to/rslearn \\
        python rslp/landslide/scripts/tag_cloudy.py \\
        --dataset /weka/.../data/landslide/sen12landslides/all_positives \\
        --group sen12_landslides

Or point at one group directory (no dataset root required)::

    python rslp/landslide/scripts/tag_cloudy.py \\
        --windows_group_dir /weka/.../all_positives/windows/sen12_landslides

By default, windows whose ``metadata.json`` already contains ``options.is_cloudy`` are skipped.
Use ``--force_rerun`` to recompute and overwrite for every window.

Parallelism: ``--num_workers`` uses a thread pool (Torch often releases the GIL during inference).
With ``--omni_device cuda`` or ``cuda:0``, prefer ``--num_workers 1`` to avoid GPU contention; for
CPU inference, try 4–8 workers and tune ``OMP_NUM_THREADS`` if needed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import tqdm
from rslearn.utils.fsspec import open_atomic
from upath import UPath

# Band order in materialized stacks (dataset band_sets).
_BANDS_12 = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
_IDX_B03 = _BANDS_12.index("B03")
_IDX_B04 = _BANDS_12.index("B04")
_IDX_B8A = _BANDS_12.index("B8A")


def _layer_sort_key(dirname: str) -> tuple[str, int]:
    """Sort pre_sentinel2, pre_sentinel2.1, pre_sentinel2.2, ..."""
    if "." in dirname:
        base, rest = dirname.split(".", 1)
        if rest.isdigit():
            return (base, int(rest))
    return (dirname, 0)


def discover_s2_geotiffs(window_dir: Path, layer_base: str) -> list[Path]:
    """Return geotiff paths for ``layer_base`` item groups, in time/group order."""
    layers = window_dir / "layers"
    if not layers.is_dir():
        return []
    found: list[tuple[tuple[str, int], Path]] = []
    pat = re.compile(rf"^{re.escape(layer_base)}(\.\d+)?$")
    for p in layers.iterdir():
        if not p.is_dir() or not pat.match(p.name):
            continue
        for sub in p.iterdir():
            if not sub.is_dir():
                continue
            tif = sub / "geotiff.tif"
            if tif.is_file():
                found.append((_layer_sort_key(p.name), tif))
                break
    found.sort(key=lambda x: x[0])
    return [tif for _, tif in found]


def _read_rgbnir_stack(tif_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (3, H, W) float32 R, G, B8A and valid mask (H, W)."""
    with rasterio.open(tif_path) as src:
        if src.count != len(_BANDS_12):
            raise ValueError(
                f"{tif_path}: expected {len(_BANDS_12)} bands, got {src.count}"
            )
        g = src.read(_IDX_B03 + 1).astype(np.float32)
        r = src.read(_IDX_B04 + 1).astype(np.float32)
        nir = src.read(_IDX_B8A + 1).astype(np.float32)
    arr = np.stack([r, g, nir], axis=0)
    valid = (r > 0) | (g > 0) | (nir > 0)
    return arr, valid


def cloud_fraction_simple(tif_path: Path) -> float:
    """Rough cloud proxy: bright visible + low NDVI among valid pixels."""
    with rasterio.open(tif_path) as src:
        if src.count != len(_BANDS_12):
            raise ValueError(
                f"{tif_path}: expected {len(_BANDS_12)} bands, got {src.count}"
            )
        b2 = src.read(_BANDS_12.index("B02") + 1).astype(np.float32)
        b4 = src.read(_IDX_B04 + 1).astype(np.float32)
        nir = src.read(_IDX_B8A + 1).astype(np.float32)
    eps = 1e-6
    # Reflectance 0–1 (Sentinel-2 L2A on PC is commonly scaled ×10000).
    b2n = np.clip(b2 / 10000.0, 0.0, 1.0)
    b4n = np.clip(b4 / 10000.0, 0.0, 1.0)
    nirn = np.clip(nir / 10000.0, 0.0, 1.0)
    ndvi = (nirn - b4n) / (nirn + b4n + eps)
    bright = (b2n + b4n) / 2.0
    valid = (b2 > 0) | (b4 > 0) | (nir > 0)
    cloudy = valid & (bright > 0.25) & (ndvi < 0.15)
    n = int(valid.sum())
    if n == 0:
        return 0.0
    return float(cloudy.sum() / n)


def omni_cloud_class_mask(
    tif_path: Path,
    patch_size: int,
    patch_overlap: int,
    inference_device: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OmniCloudMask on one 12-band Sentinel-2 stack.

    Returns:
        classes_hw: uint8 array (H, W), values 0=clear, 1=thick cloud, 2=thin cloud,
            3=cloud shadow (OmniCloudMask convention).
        valid_hw: bool mask of pixels with any R/G/B8A reflectance > 0.
    """
    from omnicloudmask import predict_from_array

    arr, valid = _read_rgbnir_stack(tif_path)
    arr = np.clip(arr / 10000.0, 0.0, None)
    _, h, w = arr.shape
    pad_h = max(0, 32 - h)
    pad_w = max(0, 32 - w)
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    pred = predict_from_array(
        input_array=arr,
        patch_size=max(patch_size, h + pad_h),
        patch_overlap=patch_overlap,
        batch_size=1,
        inference_device=inference_device,
        softmax_output=False,
    )
    if pred.ndim == 3:
        mask = pred[0, :h, :w]
    else:
        mask = pred[:h, :w]
    return np.asarray(mask, dtype=np.uint8), valid


def cloud_fraction_omni(
    tif_path: Path,
    patch_size: int,
    patch_overlap: int,
    inference_device: str | None,
) -> float:
    """OmniCloudMask: fraction of pixels that are not class 0 (clear)."""
    mask, valid = omni_cloud_class_mask(
        tif_path,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        inference_device=inference_device,
    )
    cloudy = (mask != 0) & valid
    n = int(valid.sum())
    if n == 0:
        return 0.0
    return float(cloudy.sum() / n)


def iter_window_paths(
    dataset: Path | None,
    windows_group_dir: Path | None,
    group: str | None,
) -> Iterator[Path]:
    """Yield window directory paths (same order as :func:`_resolve_window_roots`)."""
    if windows_group_dir is not None:
        root = windows_group_dir.resolve()
        if not root.is_dir():
            raise SystemExit(f"--windows_group_dir is not a directory: {root}")
        if (root / "layers").is_dir():
            yield root
            return
        for w in sorted(root.iterdir(), key=lambda p: p.name):
            if w.is_dir() and (w / "layers").is_dir():
                yield w.resolve()
        return
    if dataset is None:
        raise SystemExit("Provide --dataset or --windows_group_dir")
    ds = dataset.resolve()
    win_root = ds / "windows"
    if not win_root.is_dir():
        raise SystemExit(f"No windows/ under dataset: {ds}")
    groups = [group] if group else sorted(p.name for p in win_root.iterdir() if p.is_dir())
    for g in groups:
        gd = win_root / g
        if not gd.is_dir():
            continue
        for w in sorted(gd.iterdir(), key=lambda p: p.name):
            if w.is_dir() and (w / "layers").is_dir():
                yield w.resolve()


def _resolve_window_roots(
    dataset: Path | None,
    windows_group_dir: Path | None,
    group: str | None,
    max_windows: int | None,
) -> list[Path]:
    out: list[Path] = []
    for p in iter_window_paths(dataset, windows_group_dir, group):
        out.append(p)
        if max_windows is not None and len(out) >= max_windows:
            break
    return out


def window_is_cloudy_from_fractions(
    pre_fracs: list[float],
    post_fracs: list[float],
    frac_threshold: float,
) -> tuple[bool, bool, bool]:
    """Derive ``is_cloudy`` from per-stack cloud fractions.

    Returns:
        is_cloudy: True if all pre stacks are cloudy or all post stacks are cloudy.
        all_pre_cloudy: True only if there is at least one pre image and each is cloudy.
        all_post_cloudy: True only if there is at least one post image and each is cloudy.
    """
    all_pre_cloudy = len(pre_fracs) > 0 and all(
        f >= frac_threshold for f in pre_fracs
    )
    all_post_cloudy = len(post_fracs) > 0 and all(
        f >= frac_threshold for f in post_fracs
    )
    is_cloudy = all_pre_cloudy or all_post_cloudy
    return is_cloudy, all_pre_cloudy, all_post_cloudy


def has_existing_is_cloudy_tag(window_dir: Path) -> bool:
    """True if ``metadata.json`` already has ``options.is_cloudy`` (bool)."""
    meta_path = window_dir / "metadata.json"
    if not meta_path.is_file():
        return False
    with meta_path.open() as f:
        meta = json.load(f)
    opts = meta.get("options")
    if not isinstance(opts, dict):
        return False
    return "is_cloudy" in opts


def process_window(
    window_dir: Path,
    method: str,
    frac_threshold: float,
    omni_patch_size: int,
    omni_patch_overlap: int,
    omni_device: str | None,
    dry_run: bool,
) -> dict:
    """Run cloud screening on one window and write ``is_cloudy`` into ``metadata.json``.

    Returns a small dict with paths and counts for logging.
    """
    pre_paths = discover_s2_geotiffs(window_dir, "pre_sentinel2")
    post_paths = discover_s2_geotiffs(window_dir, "post_sentinel2")

    def score_one(p: Path) -> float:
        if method == "simple":
            return cloud_fraction_simple(p)
        return cloud_fraction_omni(
            p,
            patch_size=omni_patch_size,
            patch_overlap=omni_patch_overlap,
            inference_device=omni_device,
        )

    pre_fracs = [score_one(p) for p in pre_paths]
    post_fracs = [score_one(p) for p in post_paths]

    pre_cloudy = sum(1 for f in pre_fracs if f >= frac_threshold)
    post_cloudy = sum(1 for f in post_fracs if f >= frac_threshold)
    is_cloudy, all_pre_cloudy, all_post_cloudy = window_is_cloudy_from_fractions(
        pre_fracs, post_fracs, frac_threshold
    )

    screening = {
        "method": method,
        "cloud_fraction_threshold": frac_threshold,
        "pre_n_images": len(pre_fracs),
        "post_n_images": len(post_fracs),
        "pre_cloud_fractions": pre_fracs,
        "post_cloud_fractions": post_fracs,
        "pre_cloudy_count": pre_cloudy,
        "post_cloudy_count": post_cloudy,
        "all_pre_cloudy": all_pre_cloudy,
        "all_post_cloudy": all_post_cloudy,
    }

    meta_path = window_dir / "metadata.json"
    if meta_path.is_file():
        with meta_path.open() as f:
            meta = json.load(f)
        opts = meta.setdefault("options", {})
        opts["is_cloudy"] = is_cloudy
        opts["cloud_screening"] = screening
        if not dry_run:
            with open_atomic(UPath(meta_path), "w") as f:
                json.dump(meta, f)

    return {
        "window": str(window_dir),
        "is_cloudy": is_cloudy,
        "pre_cloudy": pre_cloudy,
        "post_cloudy": post_cloudy,
        "all_pre_cloudy": all_pre_cloudy,
        "all_post_cloudy": all_post_cloudy,
    }


def main() -> None:
    """Parse CLI args and tag windows under a dataset or a single group directory."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="rslearn dataset root (contains windows/).",
    )
    p.add_argument(
        "--windows_group_dir",
        type=Path,
        default=None,
        help="Path to windows/<group> (only that group's windows).",
    )
    p.add_argument(
        "--group",
        type=str,
        default=None,
        help="With --dataset, only this group name (e.g. sen12_landslides).",
    )
    p.add_argument(
        "--method",
        choices=("omni", "simple"),
        default="omni",
        help="omni=OmniCloudMask (default); simple=spectral heuristic.",
    )
    p.add_argument(
        "--cloud_fraction_threshold",
        type=float,
        default=0.20,
        help="Image is cloudy if cloud pixel fraction >= this (valid pixels only).",
    )
    p.add_argument(
        "--omni_patch_size",
        type=int,
        default=64,
        help="OmniCloudMask patch size (64 matches typical landslide chips; raise if larger).",
    )
    p.add_argument(
        "--omni_patch_overlap",
        type=int,
        default=0,
        help="OmniCloudMask patch overlap.",
    )
    p.add_argument(
        "--omni_device",
        type=str,
        default=None,
        help="Torch device for OmniCloudMask, e.g. cuda or cpu (default: library default).",
    )
    p.add_argument("--dry_run", action="store_true", help="Do not write metadata.json.")
    p.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Process at most this many windows (order: group, name).",
    )
    p.add_argument(
        "--force_rerun",
        action="store_true",
        help="Recompute every window even if options.is_cloudy is already set.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Concurrent windows (thread pool). Use 1 with CUDA; try >1 for CPU inference.",
    )
    args = p.parse_args()
    if args.num_workers < 1:
        raise SystemExit("--num_workers must be >= 1")

    windows = _resolve_window_roots(
        args.dataset,
        args.windows_group_dir,
        args.group,
        args.max_windows,
    )
    if not windows:
        raise SystemExit("No windows found.")

    if args.method == "omni":
        try:
            import omnicloudmask  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "method=omni requires omnicloudmask (e.g. pip install omnicloudmask)"
            ) from e

    to_process: list[Path] = []
    skipped_existing = 0
    for w in windows:
        if not args.force_rerun and has_existing_is_cloudy_tag(w):
            skipped_existing += 1
        else:
            to_process.append(w)

    summary_cloudy = 0

    def _run_one(w: Path) -> dict:
        return process_window(
            w,
            method=args.method,
            frac_threshold=args.cloud_fraction_threshold,
            omni_patch_size=args.omni_patch_size,
            omni_patch_overlap=args.omni_patch_overlap,
            omni_device=args.omni_device,
            dry_run=args.dry_run,
        )

    if args.num_workers <= 1:
        for w in tqdm.tqdm(to_process, desc="windows"):
            try:
                r = _run_one(w)
                if r["is_cloudy"]:
                    summary_cloudy += 1
            except Exception as ex:
                tqdm.tqdm.write(f"[skip] {w}: {ex}")
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            future_map = {ex.submit(_run_one, w): w for w in to_process}
            for fut in tqdm.tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="windows",
            ):
                w = future_map[fut]
                try:
                    r = fut.result()
                    if r["is_cloudy"]:
                        summary_cloudy += 1
                except Exception as ex:
                    tqdm.tqdm.write(f"[skip] {w}: {ex}")

    processed = len(to_process)
    tqdm.tqdm.write(
        f"Done. Processed {processed}/{len(windows)} windows "
        f"({skipped_existing} skipped, already had is_cloudy; use --force_rerun to redo). "
        f"{summary_cloudy} processed windows have is_cloudy=true. "
        f"num_workers={args.num_workers}. dry_run={args.dry_run}."
    )


if __name__ == "__main__":
    main()
