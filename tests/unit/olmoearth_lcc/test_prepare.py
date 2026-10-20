from datetime import datetime, timedelta, timezone

from rslp.olmoearth_lcc.lcc_model.prepare import (
    FREQUENT_BLOCK_DURATION,
    OPTION_MARGIN,
    QUARTERLY_LOOKBACK,
    QUARTERLY_PERIOD,
    _compute_frequent_block_ends,
    _compute_window_time_range,
)


def test_window_time_range_aligns_quarterly_with_prediction() -> None:
    """Option 0 of an entry anchored at T - OPTION_MARGIN matches prediction at T.

    Prediction uses frequent [T - 90d, T] and quarterly [T - 1890d, T - 90d]. The
    quarterly mosaics are built backwards from the end of the window time range, so
    the end must be a whole number of periods after T - 90d.
    """
    t = datetime(2024, 5, 17, tzinfo=timezone.utc)
    anchor = t - OPTION_MARGIN
    block_ends = _compute_frequent_block_ends(
        anchor, anchor, "some_window", datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    block_starts = [end - FREQUENT_BLOCK_DURATION for end in block_ends]

    # Option 0's frequent block is exactly the prediction frequent block.
    assert block_ends[0] == t
    assert block_starts[0] == t - timedelta(days=90)

    start, end = _compute_window_time_range(block_starts, block_ends)
    assert start == t - timedelta(days=90) - QUARTERLY_LOOKBACK
    assert end >= max(block_ends)
    assert end - max(block_ends) < QUARTERLY_PERIOD
    # Period boundaries (end - k * 90d) include T - 90d and T - 1890d.
    assert (end - (t - timedelta(days=90))) % QUARTERLY_PERIOD == timedelta(0)
    assert (end - start) % QUARTERLY_PERIOD == timedelta(0)
