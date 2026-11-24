"""Shared epoch and day conversion for the LCC day-valued rasters.

Kept dependency-light (standard library only, no torch) so both the model/task
modules and ``output_bands`` can import the epoch and conversion
helpers without pulling in heavy dependencies. The model output encodes the
predicted pre/post change dates as two ``output_change`` bands holding integer
days since :data:`UNIX_EPOCH`, the same encoding as the label day rasters and the
GeoZarr change_dates.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Day values are stored in a uint16 raster band.
MAX_DAY_VALUE = 65535

# Epoch of the ts_pre_days / ts_post_days output bands, the per-point label day
# rasters (label_ts_start, label_ts_end, label_transition_date), and the GeoZarr
# change_dates.
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# Value of the label day rasters at pixels without a dated positive point.
DAYS_NODATA = 65535


def parse_date(s: str) -> datetime:
    """Parse an ISO date string (as in the annotation JSONs) to a UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def timestamps_to_days(
    timestamps: list[tuple[datetime, datetime]],
) -> list[int]:
    """Convert per-timestep (start, end) ranges to day-values since the epoch.

    Each timestep's day value is its center date's whole-day offset from
    :data:`UNIX_EPOCH`, clamped to the uint16 range.

    Args:
        timestamps: list of (start, end) datetime ranges, one per timestep.

    Returns:
        list of integer day-values, one per timestep.
    """
    days: list[int] = []
    for start, end in timestamps:
        center = start + (end - start) / 2
        value = (center - UNIX_EPOCH).days
        days.append(max(0, min(MAX_DAY_VALUE, value)))
    return days
