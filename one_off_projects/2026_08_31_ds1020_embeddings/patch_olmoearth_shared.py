"""Make olmoearth_shared item-property parsing tolerate new API fields.

The OlmoEarth Datasets API now returns sun_elevation and sun_azimuth on
landsat-8-9-c2-l1 items. olmoearth_shared's BaseItemProperties sets
extra="forbid", so an image whose olmoearth_shared predates those fields raises
"2 validation errors for Item / Extra inputs are not permitted" for every
Landsat item, and rslearn burns its whole retry budget on each window before
giving up on the layer.

The proper fix is already on olmoearth_shared develop (a75a059 adds both fields
to LandsatC2ItemProperties). This flips extra="forbid" to extra="ignore" in the
installed item_properties module so a stale image keeps working without pulling
a newer olmoearth_shared into the environment: unknown properties are dropped,
which is fine here because the data source only needs assets, geometry and time.

Idempotent, and scoped to that one module.
"""

import pathlib

import olmoearth_shared.models.datasets.item_properties as item_properties

OLD = 'ConfigDict(extra="forbid")'
NEW = 'ConfigDict(extra="ignore")'


def main() -> None:
    path = pathlib.Path(item_properties.__file__)
    source = path.read_text()
    if "sun_elevation" in source:
        print(f"{path}: already has sun_elevation, no patch needed")
        return
    if OLD not in source:
        print(f"{path}: no {OLD} found, nothing patched")
        return
    path.write_text(source.replace(OLD, NEW))
    print(f"{path}: patched {source.count(OLD)} occurrence(s) of extra=forbid -> ignore")


if __name__ == "__main__":
    main()
