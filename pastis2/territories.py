"""French RPG territories to cover, with their native CRS.

"All French parcels" spans metropolitan France + Corsica (one delivery, Lambert-93)
plus the five overseas départements (DROM), each delivered separately in its own CRS.
Windows are still created with ``use_utm=True`` (rslearn picks the right UTM zone per
window), so this CRS is mainly needed to (a) know how to read/reproject each RPG file
and (b) sanity-check that every territory is actually ingested (islands not dropped).

Fill ``url`` per territory from IGN geoservices / data.gouv for the chosen year, or set
``local`` to an already-downloaded archive/dir. RPG product: RPG 2-0, "PARCELLES"
(parcel-level, with CODE_CULTU), for a year matching the S2 growing season (PASTIS used
RPG 2019 -> S2 Sep 2018-Nov 2019).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Territory:
    key: str          # short id used in paths
    name: str
    epsg: int         # native CRS of the RPG delivery
    url: str | None = None    # RPG archive download URL (fill per year)
    local: str | None = None  # or a pre-downloaded archive/dir path


# EPSG refs: metropole+Corsica RGF93/Lambert-93; Guadeloupe/Martinique RGAF09/UTM20N;
# Guyane RGFG95/UTM22N; Reunion RGR92/UTM40S; Mayotte RGM04/UTM38S.
# IGN geoplateforme RPG 2019 download URLs (resolved from the capabilities API).
_DL = "https://data.geopf.fr/telechargement/download/RPG"
_URL = {
    # National metropolitan delivery is a single GeoPackage (~3.4 GB, ~9M parcels).
    "metropole": f"{_DL}/RPG_2-0_GPKG_LAMB93_FR-2019/RPG_2-0_GPKG_LAMB93_FR-2019.7z",
    "guadeloupe": f"{_DL}/RPG_2-0_SHP_UTM20W84GUAD_D971-2019/RPG_2-0_SHP_UTM20W84GUAD_D971-2019.7z",
    "martinique": f"{_DL}/RPG_2-0_SHP_UTM20W84MART_D972-2019/RPG_2-0_SHP_UTM20W84MART_D972-2019.7z",
    "guyane": f"{_DL}/RPG_2-0_SHP_UTM22RGFG95_D973-2019/RPG_2-0_SHP_UTM22RGFG95_D973-2019.7z",
    "reunion": f"{_DL}/RPG_2-0_SHP_RGR92UTM40S_D974-2019/RPG_2-0_SHP_RGR92UTM40S_D974-2019.7z",
    "mayotte": f"{_DL}/RPG_2-0_SHP_RGM04UTM38S_D976-2019/RPG_2-0_SHP_RGM04UTM38S_D976-2019.7z",
}

TERRITORIES: list[Territory] = [
    Territory("metropole", "France metropolitaine + Corse", 2154, url=_URL["metropole"]),
    Territory("guadeloupe", "Guadeloupe", 5490, url=_URL["guadeloupe"]),
    Territory("martinique", "Martinique", 5490, url=_URL["martinique"]),
    Territory("guyane", "Guyane", 2972, url=_URL["guyane"]),
    Territory("reunion", "La Reunion", 2975, url=_URL["reunion"]),
    Territory("mayotte", "Mayotte", 4471, url=_URL["mayotte"]),
]
