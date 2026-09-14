"""Create the 12-monthly-mosaic twin windows from the dense dataset's windows.

Mirrors every window of the dense `national_ds` group into `national_ds_monthly`
at the IDENTICAL projection + pixel bounds (same cells), but with the PASTIS
12-month time range (Sep <year-1> .. +360 d) that the monthly-mosaic layers in
config_monthly.json tile via time_offset 0d..330d. The dense window's
`options["split"]` is copied verbatim, so the two datasets share splits exactly.

Run (after assign_splits.py has tagged the dense windows):
  python make_monthly_windows.py \
      --src data/national_ds --dst data/national_ds_monthly --group rpg_2019_dec31
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from rslearn.dataset import Window
from rslearn.dataset.dataset import Dataset
from upath import UPath

# PASTIS series: 12 x 30-day monthly mosaics starting 1 Sep 2018 (matches
# pastis_rslearn_export.WINDOW_TIME_RANGE and the mo01..mo12 time_offsets).
WINDOW_START = datetime(2018, 9, 1, tzinfo=UTC)
WINDOW_TIME_RANGE = (WINDOW_START, WINDOW_START + timedelta(days=360))


def main() -> None:
    """Clone dense windows into the monthly dataset (same cells, monthly range)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dense dataset root (has splits)")
    ap.add_argument("--dst", required=True, help="monthly dataset root (config.json)")
    ap.add_argument("--group", required=True, help="window group to mirror")
    args = ap.parse_args()

    src = Dataset(UPath(args.src))
    dst = Dataset(UPath(args.dst))
    windows = src.load_windows(groups=[args.group])
    print(f"mirroring {len(windows)} windows -> {args.dst}")

    start, end = WINDOW_TIME_RANGE
    made = 0
    for w in windows:
        minx, miny, maxx, maxy = w.bounds
        name = f"{minx}_{miny}_{maxx}_{maxy}_{start.isoformat()}_{end.isoformat()}"
        Window(
            storage=dst.storage,
            group=args.group,
            name=name,
            projection=w.projection,
            bounds=w.bounds,
            time_range=WINDOW_TIME_RANGE,
            options={"split": w.options.get("split")},
        ).save()
        made += 1
        if made % 500 == 0:
            print(f"  {made}/{len(windows)}")
    print(f"created {made} monthly windows in {args.dst}/{args.group}")


if __name__ == "__main__":
    main()
