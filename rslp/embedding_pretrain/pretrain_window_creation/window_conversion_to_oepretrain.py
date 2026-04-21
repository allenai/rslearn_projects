"""Convert rslearn windows into an OlmoEarth-friendly rslearn dataset.

This phase-0 script copies window metadata only. It does not copy materialized layers or
items. The destination dataset can then be materialized and converted by the
``olmoearth_pretrain`` dataset-creation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import tqdm
from olmoearth_pretrain.dataset_creation.constants import (
    SAMPLE_ID_OPTION,
    USE_GRID_REFERENCE_OPTION,
)
from upath import UPath

logger = logging.getLogger(__name__)

WINDOW_DURATION = timedelta(days=14)

DEFAULT_THREAD_WORKERS = 64


@dataclass(frozen=True)
class SourceWindowMetadata:
    """Lightweight representation of a source window read directly from metadata.json."""

    group: str
    name: str
    projection: dict[str, Any]
    x_resolution: float
    bounds: tuple[int, int, int, int]
    time_range: tuple[datetime, datetime] | None
    options: dict[str, Any]


@dataclass(frozen=True)
class ConvertedWindow:
    """A converted rslearn window payload."""

    group: str
    name: str
    projection: dict[str, Any]
    bounds: tuple[int, int, int, int]
    time_range: tuple[datetime, datetime]
    options: dict[str, Any]


def format_resolution(value: float) -> str:
    """Format a resolution value for filenames."""
    if float(int(value)) == value:
        return str(int(value))
    return str(value)


def sanitize_component(value: str) -> str:
    """Sanitize a user-facing identifier for use in a path segment."""
    return value.replace("/", "_").replace("\\", "_")


def get_sample_id(src: SourceWindowMetadata, sample_id_option_key: str | None) -> str:
    """Get the sample identifier for a source window."""
    if sample_id_option_key:
        sample_id = src.options.get(sample_id_option_key)
        if sample_id is not None and sample_id != "":
            return sanitize_component(str(sample_id))
    return sanitize_component(src.name)


def get_source_year(src: SourceWindowMetadata) -> int:
    """Get the source year used to sample the new 14-day range."""
    if "year" in src.options:
        return int(src.options["year"])
    if src.time_range is None:
        raise ValueError(f"window {src.name} is missing time_range and year option")
    return src.time_range[0].year


def sample_time_range(
    src: SourceWindowMetadata, seed: int, current_date: datetime
) -> tuple[datetime, datetime]:
    """Sample a deterministic 14-day time range within the source year.

    A cutoff is derived as ``current_date - 8 months``. When the source year
    matches the cutoff year, the start date is sampled between Jan 1 and the
    cutoff instead of the full year.
    """
    year = get_source_year(src)
    start_of_year = datetime(year, 1, 1, tzinfo=UTC)
    end_sampling_date = datetime(year + 1, 1, 1, tzinfo=UTC)

    cutoff = current_date - timedelta(days=8 * 30)
    if year == cutoff.year:
        end_sampling_date = cutoff

    num_days = (end_sampling_date - start_of_year).days
    digest = hashlib.sha256(f"{seed}:{src.group}:{src.name}".encode()).hexdigest()
    offset_days = int(digest[:16], 16) % num_days
    start_time = start_of_year + timedelta(days=offset_days)
    end_time = start_time + WINDOW_DURATION
    return (start_time, end_time)


def build_window_name(
    src: SourceWindowMetadata,
    group_name: str | None,
    sample_id_option_key: str | None,
) -> str:
    """Build the destination window name."""
    resolution = format_resolution(abs(src.x_resolution))
    sample_id = get_sample_id(src, sample_id_option_key)
    prefix = sanitize_component(group_name) if group_name else None
    if prefix:
        return f"{prefix}_{resolution}_{sample_id}"
    return f"{resolution}_{sample_id}"


def window_matches_filter(
    options: dict[str, Any], split_key: str | None, split_value: str | None
) -> bool:
    """Check if a window matches the optional options filter."""
    if split_key is None:
        return True
    if split_key not in options:
        return False
    if split_value is None:
        return True
    return str(options[split_key]) == split_value


def load_selected_window_names(selection_file_path: str) -> list[str]:
    """Load source window names from a JSON list file."""
    with open(selection_file_path) as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError(
            f"Expected a JSON list in {selection_file_path!r}, got {type(names).__name__}"
        )
    return [str(v) for v in names]


def read_source_metadata(
    metadata_path: Path, group: str, name: str
) -> SourceWindowMetadata:
    """Read a source window's metadata.json directly, bypassing rslearn's Dataset."""
    with open(metadata_path) as f:
        md = json.load(f)

    if len(md["bounds"]) != 4:
        raise ValueError(
            f"expected bounds to have 4 elements but got {len(md['bounds'])}"
        )

    time_range = None
    if md.get("time_range"):
        time_range = (
            datetime.fromisoformat(md["time_range"][0]),
            datetime.fromisoformat(md["time_range"][1]),
        )

    projection = md["projection"]
    x_resolution = float(projection.get("x_resolution", projection.get("x_res", 0)))

    return SourceWindowMetadata(
        group=group,
        name=name,
        projection=projection,
        x_resolution=x_resolution,
        bounds=(md["bounds"][0], md["bounds"][1], md["bounds"][2], md["bounds"][3]),
        time_range=time_range,
        options=md.get("options", {}),
    )


def convert_source(
    src: SourceWindowMetadata,
    dst_group: str,
    group_name: str | None,
    sample_id_option_key: str | None,
    seed: int,
    current_date: datetime,
) -> ConvertedWindow:
    """Convert one source window metadata into destination metadata."""
    sample_id = get_sample_id(src, sample_id_option_key)
    name = build_window_name(src, group_name, sample_id_option_key)
    options = dict(src.options)
    options[USE_GRID_REFERENCE_OPTION] = False
    options[SAMPLE_ID_OPTION] = sample_id

    return ConvertedWindow(
        group=dst_group,
        name=name,
        projection=src.projection,
        bounds=src.bounds,
        time_range=sample_time_range(src, seed, current_date),
        options=options,
    )


def write_window(dst_root: UPath, converted_window: ConvertedWindow) -> None:
    """Write one converted window to disk."""
    window_root = dst_root / "windows" / converted_window.group / converted_window.name
    window_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "group": converted_window.group,
        "name": converted_window.name,
        "projection": converted_window.projection,
        "bounds": list(converted_window.bounds),
        "time_range": [
            converted_window.time_range[0].isoformat(),
            converted_window.time_range[1].isoformat(),
        ],
        "options": converted_window.options,
    }
    metadata_path = window_root / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f)


def _process_one_window(
    src_root: UPath,
    src_group: str,
    window_name: str,
    dst_root: UPath,
    dst_group: str,
    group_name: str | None,
    sample_id_option_key: str | None,
    split_key: str | None,
    split_value: str | None,
    seed: int,
    dry_run: bool,
    current_date: datetime,
) -> ConvertedWindow | None:
    """Read, convert, and write a single window. Returns None if filtered out or missing."""
    metadata_path = src_root / "windows" / src_group / window_name / "metadata.json"
    try:
        src = read_source_metadata(metadata_path, src_group, window_name)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("skipping %s: %s", window_name, exc)
        return None

    if not window_matches_filter(src.options, split_key, split_value):
        return None

    converted = convert_source(
        src=src,
        dst_group=dst_group,
        group_name=group_name,
        sample_id_option_key=sample_id_option_key,
        seed=seed,
        current_date=current_date,
    )

    if not dry_run:
        write_window(dst_root, converted)

    return converted


def _list_all_window_names(src_root: UPath, src_group: str) -> list[str]:
    """List every window name in a source group by iterating the directory."""
    group_dir = src_root / "windows" / src_group
    if not group_dir.exists():
        return []
    return [p.name for p in group_dir.iterdir() if not p.name.startswith(".")]


def convert_dataset(
    src_root: UPath,
    src_group: str,
    dst_root: UPath,
    dst_group: str,
    current_date: datetime,
    split_key: str | None = None,
    split_value: str | None = None,
    selection_file_path: str | None = None,
    sample_id_option_key: str | None = None,
    group_name: str | None = None,
    seed: int = 0,
    dry_run: bool = False,
    workers: int = DEFAULT_THREAD_WORKERS,
    show_progress: bool = False,
) -> list[ConvertedWindow]:
    """Convert all matching windows from a source rslearn dataset."""
    if selection_file_path is not None:
        logger.info("loading selection file %s ...", selection_file_path)
        window_names = load_selected_window_names(selection_file_path)
    else:
        logger.info("listing all windows in %s/%s ...", src_root, src_group)
        window_names = _list_all_window_names(src_root, src_group)

    logger.info("processing %d windows with %d threads ...", len(window_names), workers)

    converted_windows: list[ConvertedWindow] = []
    seen_names: dict[str, str] = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_one_window,
                src_root=src_root,
                src_group=src_group,
                window_name=wn,
                dst_root=dst_root,
                dst_group=dst_group,
                group_name=group_name,
                sample_id_option_key=sample_id_option_key,
                split_key=split_key,
                split_value=split_value,
                seed=seed,
                dry_run=dry_run,
                current_date=current_date,
            ): wn
            for wn in window_names
        }

        it: Any = as_completed(futures)
        if show_progress:
            it = tqdm.tqdm(it, total=len(futures), desc="Converting windows")

        for future in it:
            source_name = futures[future]
            try:
                converted = future.result()
            except Exception:
                logger.exception("failed to process %s", source_name)
                errors += 1
                continue

            if converted is None:
                continue

            previous_source = seen_names.get(converted.name)
            if previous_source is not None:
                raise ValueError(
                    "destination window name collision: "
                    f"{converted.name} maps from both {previous_source} and {source_name}"
                )
            seen_names[converted.name] = source_name
            converted_windows.append(converted)

    if errors:
        logger.warning("%d windows failed to process", errors)

    return converted_windows


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Convert rslearn windows into an OlmoEarth-friendly rslearn dataset.",
    )
    parser.add_argument(
        "--src-root", required=True, help="Source rslearn dataset root."
    )
    parser.add_argument(
        "--src-group",
        required=True,
        help="Source rslearn window group to convert.",
    )
    parser.add_argument("--dst-root", required=True, help="Destination dataset root.")
    parser.add_argument(
        "--dst-group",
        default="res_10",
        help="Destination rslearn group directory (default: res_10).",
    )
    parser.add_argument(
        "--split-key",
        default=None,
        help="Optional window.options key used to filter source windows.",
    )
    parser.add_argument(
        "--split-value",
        default=None,
        help="Optional string value for the split filter.",
    )
    parser.add_argument(
        "--selection-file-path",
        default=None,
        help="Optional JSON file containing a list of window names to select for conversion.",
    )
    parser.add_argument(
        "--sample-id-option-key",
        default=None,
        help="Optional window.options key used to source sample_id for the converted windows.",
    )
    parser.add_argument(
        "--group-name",
        default=None,
        help="Optional prefix used in destination window names: {group_name}_{resolution}_{sample_id}.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic sampling seed."
    )
    parser.add_argument(
        "--current-date",
        required=True,
        help=(
            "Reference date (YYYY-MM-DD) used to cap time-range sampling. "
            "A cutoff of current_date - 8 months is computed; windows whose source year "
            "matches the cutoff year have their sampled start date capped at the cutoff."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the converted windows without writing files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_THREAD_WORKERS,
        help=f"Number of I/O threads (default: {DEFAULT_THREAD_WORKERS}).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show a tqdm progress bar.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    current_date = datetime.strptime(args.current_date, "%Y-%m-%d").replace(tzinfo=UTC)

    converted_windows = convert_dataset(
        src_root=UPath(args.src_root),
        src_group=args.src_group,
        dst_root=UPath(args.dst_root),
        dst_group=args.dst_group,
        current_date=current_date,
        split_key=args.split_key,
        split_value=args.split_value,
        selection_file_path=args.selection_file_path,
        sample_id_option_key=args.sample_id_option_key,
        group_name=args.group_name,
        seed=args.seed,
        dry_run=args.dry_run,
        workers=args.workers,
        show_progress=args.show_progress,
    )
    print(f"converted {len(converted_windows)} windows")


if __name__ == "__main__":
    main()
