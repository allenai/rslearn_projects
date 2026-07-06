"""Dump the captioned PNGs that would be included in the Gemini prompt for a window.

Given a point set JSON (from ``add_points``), an image-database window name, and a
destination directory, this reproduces the exact set of captioned images that
``run_gemini`` would feed to Gemini for that window and writes them as PNG files.

This reuses the same imagery selection and captioning logic as the prompt builder, so
the output matches what the model actually sees (minus the text prompt itself).

Example:
    python -m rslp.change_finder_v2.vlm.category_tagger.dump_prompt_images \
        --points points.json \
        --window-name -122.33210_47.60620_2023 \
        --output ./dump/
"""

from __future__ import annotations

import argparse
import re

from upath import UPath

from rslp.change_finder_v2.vlm.image_db import image_database

from .run_gemini import _build_image_refs, _parse_date
from .schema import PointRecord, PointSet


def _find_record(point_set: PointSet, window_name: str) -> PointRecord:
    """Return the single point record matching the image-db window name."""
    matches = [r for r in point_set.points if r.window_name == window_name]
    if not matches:
        raise ValueError(f"no point with window_name {window_name!r} in the point set")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} points with window_name {window_name!r}; expected one"
        )
    return matches[0]


def _safe_name(label: str) -> str:
    """Turn an image caption into a filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", label).strip("_")


def dump_prompt_images(
    points: str,
    window_name: str,
    output: str,
    image_db_path: str | None = None,
    s2_layer: str = "sentinel2",
    highres_layer: str = "esri",
) -> None:
    """Write the prompt images for one window to a destination directory."""
    point_set = PointSet.load(points)
    record = _find_record(point_set, window_name)
    db_path = image_db_path or point_set.image_db_path

    if not record.pre_change or not record.post_change:
        raise ValueError(
            f"record {record.window_name} is missing pre_change/post_change dates"
        )

    images = image_database.list_available_images(
        db_path, record.lon, record.lat, record.year, group=point_set.group
    )
    refs, _ = _build_image_refs(
        images,
        s2_layer,
        highres_layer,
        _parse_date(record.pre_change),
        _parse_date(record.post_change),
    )

    if not refs:
        print(f"No materialized images found for {window_name}; nothing written.")
        return

    out_dir = UPath(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ref in enumerate(refs):
        fname = f"{i:02d}_{_safe_name(ref.label)}.png"
        out_path = out_dir / fname
        with out_path.open("wb") as f:
            f.write(ref.png_bytes)
        print(f"wrote {out_path}")

    print(f"Wrote {len(refs)} images to {out_dir}")


def main(args: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Dump the captioned PNGs that would be included in the Gemini prompt for "
            "a window."
        )
    )
    parser.add_argument(
        "--points", required=True, help="Point set JSON produced by add_points."
    )
    parser.add_argument(
        "--window-name",
        required=True,
        help="Image-database window name to dump (e.g. -122.33210_47.60620_2023).",
    )
    parser.add_argument(
        "--output", required=True, help="Destination directory for the PNGs."
    )
    parser.add_argument(
        "--image-db-path",
        default=None,
        help="Override the image database path stored in the point set.",
    )
    parser.add_argument(
        "--s2-layer", default="sentinel2", help="Sentinel-2 layer name."
    )
    parser.add_argument(
        "--highres-layer", default="esri", help="High-resolution layer name."
    )
    parsed = parser.parse_args(args=args)

    dump_prompt_images(
        points=parsed.points,
        window_name=parsed.window_name,
        output=parsed.output,
        image_db_path=parsed.image_db_path,
        s2_layer=parsed.s2_layer,
        highres_layer=parsed.highres_layer,
    )


if __name__ == "__main__":
    main()
