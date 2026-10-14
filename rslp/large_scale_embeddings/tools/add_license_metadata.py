"""Backfill the license convention onto a store created before it existed.

`init_store` writes every group's metadata once, at creation, so a store built before
the license convention was adopted carries neither the registration nor the attribute
and nothing will add them later. This walks the existing groups and writes both.

Only group attributes change. No chunk is read or rewritten, so this costs one small
object per group regardless of how large the archive is.

Safe to run against a store workers are writing to: workers write chunks, never group
metadata, and each `zarr.json` is replaced by a single atomic object write, so a
concurrent reader sees either the old attributes or the new ones.

Idempotent: a group that already carries the convention and a matching attribute is
left alone, so a partial run can simply be repeated.

The store carries consolidated metadata, so readers take a group's attributes from the
root's copy rather than the group's own `zarr.json`. Updating the groups alone changes
nothing a reader can see; this re-consolidates afterwards, as `init_store` does.
"""

import zarr

from rslp.large_scale_embeddings.zarr_store import (
    DEFAULT_LICENSE_URL,
    LICENSE_CONVENTION,
    build_license_attrs,
    writable_store,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)


def _needs_update(attrs: dict, license_url: str) -> bool:
    """Whether this group is missing the registration or the attribute.

    Args:
        attrs: the group's current attributes.
        license_url: the licence URL the store should record.

    Returns:
        True if anything would change.
    """
    registered = any(
        convention.get("uuid") == LICENSE_CONVENTION["uuid"]
        for convention in attrs.get("zarr_conventions", [])
    )
    return not registered or attrs.get("license") != {"url": license_url}


def add_license_metadata(
    store_path: str,
    license_url: str = DEFAULT_LICENSE_URL,
    dry_run: bool = True,
    storage_options: dict | None = None,
) -> int:
    """Add the license convention and attribute to every group in a store.

    Args:
        store_path: the Zarr store to update.
        license_url: URL of the licence text to record.
        dry_run: report what would change without writing. Defaults to True, since the
            alternative is an unasked-for write to a production archive.
        storage_options: fsspec storage options for remote stores.

    Returns:
        the number of groups updated, or that would be updated under `dry_run`.
    """
    root = zarr.open_group(
        store=writable_store(store_path, storage_options),
        mode="r" if dry_run else "r+",
    )
    groups = [("", root)] + [(name, group) for name, group in root.groups()]
    license_attrs = build_license_attrs(license_url)
    updated = 0

    for name, group in groups:
        attrs = dict(group.attrs)
        if not _needs_update(attrs, license_url):
            continue
        updated += 1
        label = name or "(root)"
        if dry_run:
            logger.info("would update %s", label)
            continue
        conventions = [
            convention
            for convention in attrs.get("zarr_conventions", [])
            if convention.get("uuid") != LICENSE_CONVENTION["uuid"]
        ]
        conventions.append(LICENSE_CONVENTION)
        # update() rather than assignment: every other attribute on the group was
        # written at creation and must survive untouched.
        group.attrs.update({"zarr_conventions": conventions, **license_attrs})
        logger.info("updated %s", label)

    if updated and not dry_run:
        # Readers resolve attributes through the root's consolidated copy, so without
        # this the per-group writes above are invisible.
        zarr.consolidate_metadata(root.store)
        logger.info("re-consolidated the store metadata")

    logger.info(
        "%s %d of %d group(s)",
        "would update" if dry_run else "updated",
        updated,
        len(groups),
    )
    return updated
