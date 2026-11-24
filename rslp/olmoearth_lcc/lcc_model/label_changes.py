"""The first and second change of a positive point, and their date rules.

A positive point labels one change (its pre_change, first_date_change_noticeable,
post_change, pre/post land cover and change categories, and an optional stop_date),
and optionally a second change after it (the change2_* fields). Training uses the
second change for imagery ending at or after its first-noticeable date, and the
first change before that (see transforms.gate_change_targets).

Points labeled before the second change existed may instead carry the legacy
transition fields (transition_date, transition_category, transition_change_category,
transition_terminal_state): an intermediate stage between pre and post. They map to
two changes that train the same way: the first change ends at the transition stage,
and the second runs from the point's pre side to its post side, noticeable once the
post-change state is complete. ``legacy_transition_to_change2`` does this, for
prepare.py and for the Studio migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .timestamp_encoding import parse_date

# The second change's point fields: date fields, then land cover and change category
# fields, then the optional stop date and terminal state.
CHANGE2_PRE_CHANGE = "change2_pre_change"
CHANGE2_FIRST_NOTICEABLE = "change2_first_date_change_noticeable"
CHANGE2_POST_CHANGE = "change2_post_change"
CHANGE2_PRE_CATEGORY = "change2_pre_category"
CHANGE2_POST_CATEGORY = "change2_post_category"
CHANGE2_PRE_CHANGE_CATEGORY = "change2_pre_change_category"
CHANGE2_POST_CHANGE_CATEGORY = "change2_post_change_category"
CHANGE2_STOP_DATE = "change2_stop_date"
CHANGE2_TERMINAL_STATE = "change2_terminal_state"
CHANGE2_FIELDS = (
    CHANGE2_PRE_CHANGE,
    CHANGE2_FIRST_NOTICEABLE,
    CHANGE2_POST_CHANGE,
    CHANGE2_PRE_CATEGORY,
    CHANGE2_POST_CATEGORY,
    CHANGE2_PRE_CHANGE_CATEGORY,
    CHANGE2_POST_CHANGE_CATEGORY,
    CHANGE2_STOP_DATE,
    CHANGE2_TERMINAL_STATE,
)
CHANGE2_DATE_FIELDS = (
    CHANGE2_PRE_CHANGE,
    CHANGE2_FIRST_NOTICEABLE,
    CHANGE2_POST_CHANGE,
)
# A second change must have these (and a point using any change2 field has one).
CHANGE2_REQUIRED_FIELDS = CHANGE2_DATE_FIELDS + (
    CHANGE2_PRE_CATEGORY,
    CHANGE2_POST_CATEGORY,
)

# The first change's dates: a positive point is trained only with all three (and,
# with a second change, CHANGE2_DATE_FIELDS).
POSITIVE_POINT_DATE_FIELDS = (
    "pre_change",
    "post_change",
    "first_date_change_noticeable",
)

# Legacy transition fields (replaced by the second change on 2026-11-24).
TRANSITION_DATE = "transition_date"
TRANSITION_CATEGORY = "transition_category"
TRANSITION_CHANGE_CATEGORY = "transition_change_category"
TRANSITION_TERMINAL_STATE = "transition_terminal_state"
# Legacy fields that were never used: points carrying them are rejected.
UNSUPPORTED_LEGACY_FIELDS = (
    "transition_stop_date",
    "post_transition_pre_change",
    "post_transition_pre_change_category",
)
LEGACY_TRANSITION_FIELDS = (
    TRANSITION_DATE,
    TRANSITION_CATEGORY,
    TRANSITION_CHANGE_CATEGORY,
    TRANSITION_TERMINAL_STATE,
) + UNSUPPORTED_LEGACY_FIELDS

# The window-level stop date (an entry key and Studio task attribute): no training
# sample of the window ends after it.
WINDOW_STOP_DATE = "window_stop_date"


@dataclass
class Change:
    """One change of a positive point, as annotated (dates as YYYY-MM-DD strings).

    For the first change, the fields are the point's own (pre_change,
    first_date_change_noticeable, post_change, pre_category, post_category,
    pre_change_category, post_change_category, stop_date); for the second, the
    change2_* fields.
    """

    pre_change: str
    first_noticeable: str
    post_change: str
    pre_category: str
    post_category: str
    pre_change_category: str
    post_change_category: str
    stop_date: str

    def as_point_fields(self) -> dict[str, str]:
        """The change in the first change's point field names."""
        return {
            "pre_change": self.pre_change,
            "first_date_change_noticeable": self.first_noticeable,
            "post_change": self.post_change,
            "pre_category": self.pre_category,
            "post_category": self.post_category,
            "pre_change_category": self.pre_change_category,
            "post_change_category": self.post_change_category,
            "stop_date": self.stop_date,
        }


def has_legacy_transition(pt: dict[str, Any]) -> bool:
    """Whether a point carries any legacy transition field."""
    return any(pt.get(field) for field in LEGACY_TRANSITION_FIELDS)


def has_change2(pt: dict[str, Any]) -> bool:
    """Whether a point carries any second-change field."""
    return any(pt.get(field) for field in CHANGE2_FIELDS)


def legacy_transition_to_change2(pt: dict[str, Any]) -> dict[str, Any]:
    """Convert a point with legacy transition fields to first + second change.

    The first change keeps the point's pre side and first-noticeable date and ends
    at the transition stage (post_change = transition_date, post land cover and
    change category = the transition's, terminal state = the transition's). The
    second change keeps the point's pre side (pre_change, pre_category,
    pre_change_category), is noticeable at the point's post_change, and ends in the
    point's post side, stop date and terminal state. Fields the point doesn't have
    stay unset. Points without a transition are returned unchanged (a copy).

    Raises:
        ValueError: if the point uses a legacy field that was never supported here
            (transition_stop_date, post-transition fields) or already has change2
            fields.
    """
    out = dict(pt)
    if not has_legacy_transition(pt):
        return out
    unsupported = [field for field in UNSUPPORTED_LEGACY_FIELDS if pt.get(field)]
    if unsupported:
        raise ValueError(f"cannot convert legacy fields {unsupported} of point {pt}")
    if has_change2(pt):
        raise ValueError(f"point has both legacy transition and change2 fields: {pt}")
    for field in LEGACY_TRANSITION_FIELDS:
        out.pop(field, None)
    second = {
        CHANGE2_PRE_CHANGE: pt.get("pre_change"),
        CHANGE2_FIRST_NOTICEABLE: pt.get("post_change"),
        CHANGE2_POST_CHANGE: pt.get("post_change"),
        CHANGE2_PRE_CATEGORY: pt.get("pre_category"),
        CHANGE2_POST_CATEGORY: pt.get("post_category"),
        CHANGE2_PRE_CHANGE_CATEGORY: pt.get("pre_change_category"),
        CHANGE2_POST_CHANGE_CATEGORY: pt.get("post_change_category"),
        CHANGE2_STOP_DATE: pt.get("stop_date"),
        CHANGE2_TERMINAL_STATE: pt.get("terminal_state"),
    }
    first = {
        "post_change": pt.get(TRANSITION_DATE),
        "post_category": pt.get(TRANSITION_CATEGORY),
        "post_change_category": pt.get(TRANSITION_CHANGE_CATEGORY),
        "stop_date": None,
        "terminal_state": pt.get(TRANSITION_TERMINAL_STATE),
    }
    for field, value in {**first, **second}.items():
        if value:
            out[field] = value
        else:
            out.pop(field, None)
    return out


def missing_date_fields(pt: dict[str, Any]) -> list[str]:
    """The unset date fields a positive point needs to be trained.

    The first change's pre_change, post_change and first_date_change_noticeable,
    and, if the point has a second change (after converting legacy transition
    fields), the second change's three dates. Points missing any are incomplete.
    """
    pt = legacy_transition_to_change2(pt)
    fields = [*POSITIVE_POINT_DATE_FIELDS]
    if has_change2(pt):
        fields += CHANGE2_DATE_FIELDS
    return [field for field in fields if not pt.get(field)]


def point_changes(pt: dict[str, Any]) -> tuple[Change, Change | None]:
    """The first and (optional) second change of a positive point.

    Legacy transition fields are converted first (legacy_transition_to_change2).
    Unset fields are "".
    """
    pt = legacy_transition_to_change2(pt)
    first = Change(
        pre_change=pt.get("pre_change") or "",
        first_noticeable=pt.get("first_date_change_noticeable") or "",
        post_change=pt.get("post_change") or "",
        pre_category=pt.get("pre_category") or "",
        post_category=pt.get("post_category") or "",
        pre_change_category=pt.get("pre_change_category") or "",
        post_change_category=pt.get("post_change_category") or "",
        stop_date=pt.get("stop_date") or "",
    )
    if not has_change2(pt):
        return first, None
    second = Change(
        pre_change=pt.get(CHANGE2_PRE_CHANGE) or "",
        first_noticeable=pt.get(CHANGE2_FIRST_NOTICEABLE) or "",
        post_change=pt.get(CHANGE2_POST_CHANGE) or "",
        pre_category=pt.get(CHANGE2_PRE_CATEGORY) or "",
        post_category=pt.get(CHANGE2_POST_CATEGORY) or "",
        pre_change_category=pt.get(CHANGE2_PRE_CHANGE_CATEGORY) or "",
        post_change_category=pt.get(CHANGE2_POST_CHANGE_CATEGORY) or "",
        stop_date=pt.get(CHANGE2_STOP_DATE) or "",
    )
    return first, second


def _date(value: str) -> datetime | None:
    return parse_date(value) if value else None


def point_date_errors(pt: dict[str, Any]) -> list[str]:
    """Date-rule violations of a positive point (empty when valid).

    First change (checked when its three dates are set):
    pre_change < first_noticeable, pre_change < post_change, and stop_date (if set)
    after both first_noticeable and post_change. first_noticeable may be after
    post_change (unusual, allowed).

    Second change (when any change2 field is set): its three dates and pre/post land
    cover are required, change2 pre_change < change2 first_noticeable and <
    change2 post_change, change2 first_noticeable > the first change's
    first_noticeable, change2 stop_date (if set) after its first_noticeable and
    post_change, and the first change's stop_date (if set) before change2
    first_noticeable.
    """
    try:
        first, second = point_changes(pt)
    except ValueError as e:
        return [str(e)]
    errors: list[str] = []
    pre, fo, post = (
        _date(first.pre_change),
        _date(first.first_noticeable),
        _date(first.post_change),
    )
    stop = _date(first.stop_date)
    if pre and fo and post:
        if not pre < fo:
            errors.append("pre_change must be before first_date_change_noticeable")
        if not pre < post:
            errors.append("pre_change must be before post_change")
        if stop and not (stop > fo and stop > post):
            errors.append(
                "stop_date must be after first_date_change_noticeable and post_change"
            )
    if second is None:
        return errors
    missing = [
        field
        for field, value in (
            (CHANGE2_PRE_CHANGE, second.pre_change),
            (CHANGE2_FIRST_NOTICEABLE, second.first_noticeable),
            (CHANGE2_POST_CHANGE, second.post_change),
            (CHANGE2_PRE_CATEGORY, second.pre_category),
            (CHANGE2_POST_CATEGORY, second.post_category),
        )
        if not value
    ]
    if missing:
        errors.append(f"second change is missing {', '.join(missing)}")
        return errors
    pre2, fo2, post2 = (
        parse_date(second.pre_change),
        parse_date(second.first_noticeable),
        parse_date(second.post_change),
    )
    stop2 = _date(second.stop_date)
    if not pre2 < fo2:
        errors.append(f"{CHANGE2_PRE_CHANGE} must be before {CHANGE2_FIRST_NOTICEABLE}")
    if not pre2 < post2:
        errors.append(f"{CHANGE2_PRE_CHANGE} must be before {CHANGE2_POST_CHANGE}")
    if fo and not fo2 > fo:
        errors.append(
            f"{CHANGE2_FIRST_NOTICEABLE} must be after first_date_change_noticeable"
        )
    if stop2 and not (stop2 > fo2 and stop2 > post2):
        errors.append(
            f"{CHANGE2_STOP_DATE} must be after {CHANGE2_FIRST_NOTICEABLE} and "
            f"{CHANGE2_POST_CHANGE}"
        )
    if stop and not stop < fo2:
        errors.append(f"stop_date must be before {CHANGE2_FIRST_NOTICEABLE}")
    return errors
