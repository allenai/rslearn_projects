"""Merge per-window GeoTIFFs into a single mosaic GeoTIFF.

Example:
    python mosaic_windows.py \
        --ds-path /weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250616 \
        --windows-group nandi_county_2018 \
        --layer-name prediction_v4_1 \
        --output /weka/dfive-default/rslearn-eai/artifacts/nandi_crop_type/mosaic_2018_v4_1.tif
"""

import argparse
import sys
from pathlib import Path

import rasterio
from rasterio.merge import merge


def mosaic_windows(
    ds_path: Path,
    windows_group: str,
    layer_name: str,
    output: Path,
    layer_subpath: str = "output/geotiff.tif",
) -> None:
    """Mosaic window GeoTIFFs into one file."""
    base_dir = ds_path / "windows" / windows_group
    if not base_dir.is_dir():
        print(f"[error] Windows group not found: {base_dir}", file=sys.stderr)
        raise SystemExit(1)

    # Collect per-window GeoTIFFs
    tif_files = []
    for window_dir in base_dir.iterdir():
        if not window_dir.is_dir():
            continue
        tif_path = window_dir / "layers" / layer_name / layer_subpath
        if tif_path.exists():
            tif_files.append(tif_path)

    print(
        f"Found {len(tif_files)} GeoTIFF files under {base_dir} for layer '{layer_name}'."
    )
    if not tif_files:
        print("[error] No GeoTIFFs found. Check layer name / subpath.", file=sys.stderr)
        raise SystemExit(2)

    srcs = [rasterio.open(str(fp)) for fp in tif_files]
    try:
        mosaic, out_transform = merge(srcs)
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **out_meta) as dst:
            dst.write(mosaic)
        print(f"Mosaic written to: {output}")
    finally:
        for s in srcs:
            s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mosaic window GeoTIFFs into one file."
    )
    parser.add_argument("--ds-path", required=True, type=Path)
    parser.add_argument("--windows-group", required=True)
    parser.add_argument("--layer-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer-subpath", default="output/geotiff.tif")
    args = parser.parse_args()

    mosaic_windows(
        ds_path=args.ds_path,
        windows_group=args.windows_group,
        layer_name=args.layer_name,
        output=args.output,
        layer_subpath=args.layer_subpath,
    )
