"""Backfilling the license convention onto a store that predates it."""

from pathlib import Path

import zarr

from rslp.large_scale_embeddings import zarr_store as zs
from rslp.large_scale_embeddings.tools.add_license_metadata import add_license_metadata

LICENSE_URL = "https://example.invalid/LICENSE.txt"


def _store_without_license(tmp_path: Path) -> str:
    """A store as it looked before the convention existed."""
    store_path = str(tmp_path / "embeddings.zarr")
    zs.init_store(
        store_path=store_path,
        zone_numbers=[14, 15],
        years=[2025],
        model_url="https://example.invalid/model",
        source_data=["https://example.invalid/s2"],
        resolution=10,
        tile_size=2048,
        dimensions=128,
        shard_size=2048,
    )
    root = zarr.open_group(store_path, mode="r+")
    for group in [root, *(g for _, g in root.groups())]:
        attrs = dict(group.attrs)
        attrs.pop("license", None)
        attrs["zarr_conventions"] = [
            c
            for c in attrs["zarr_conventions"]
            if c["uuid"] != zs.LICENSE_CONVENTION["uuid"]
        ]
        group.attrs.put(attrs)
    # Match how the store is actually written: readers see the consolidated copy.
    zarr.consolidate_metadata(root.store)
    return store_path


def test_backfill_adds_the_convention_to_every_group(tmp_path: Path) -> None:
    """Root and every zone must come out carrying both halves."""
    store_path = _store_without_license(tmp_path)
    assert add_license_metadata(store_path, LICENSE_URL, dry_run=False) == 3

    root = zarr.open_group(store_path, mode="r")
    for group in [root, *(g for _, g in root.groups())]:
        names = [c["name"] for c in group.attrs["zarr_conventions"]]
        assert "license" in names
        assert group.attrs["license"] == {"url": LICENSE_URL}


def test_backfill_preserves_the_attributes_already_there(tmp_path: Path) -> None:
    """Everything written at creation must survive: this is a live archive."""
    store_path = _store_without_license(tmp_path)
    before = {
        name: dict(group.attrs)
        for name, group in zarr.open_group(store_path, mode="r").groups()
    }
    add_license_metadata(store_path, LICENSE_URL, dry_run=False)

    after = {
        name: dict(group.attrs)
        for name, group in zarr.open_group(store_path, mode="r").groups()
    }
    for name, old in before.items():
        for key, value in old.items():
            if key == "zarr_conventions":
                assert all(c in after[name][key] for c in value), f"{name}: {key} lost"
            else:
                assert after[name][key] == value, f"{name}: {key} changed"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """The default must not touch a production store."""
    store_path = _store_without_license(tmp_path)
    assert add_license_metadata(store_path, LICENSE_URL) == 3
    root = zarr.open_group(store_path, mode="r")
    assert "license" not in dict(root.attrs)


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """A repeated run must be a no-op, so a partial run can just be rerun."""
    store_path = _store_without_license(tmp_path)
    add_license_metadata(store_path, LICENSE_URL, dry_run=False)
    assert add_license_metadata(store_path, LICENSE_URL, dry_run=False) == 0


def test_backfill_is_visible_through_consolidated_metadata(tmp_path: Path) -> None:
    """The root's consolidated copy must carry the change, not just each group.

    Readers resolve a group's attributes through the root when consolidated metadata is
    present, which it is here. Writing the groups and not re-consolidating leaves every
    per-group write invisible while the tool reports success.
    """
    store_path = _store_without_license(tmp_path)
    add_license_metadata(store_path, LICENSE_URL, dry_run=False)

    consolidated = zarr.open_group(store_path, mode="r", use_consolidated=True)
    for _, group in consolidated.groups():
        assert group.attrs["license"] == {
            "url": LICENSE_URL
        }, "zone attributes did not reach the consolidated copy"
