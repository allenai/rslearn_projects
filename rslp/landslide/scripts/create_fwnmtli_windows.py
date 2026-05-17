r"""Create windows from the Far-Western Nepal Multi-Temporal Landslide Inventory (points).

The Zenodo / NIAID point inventory provides a ``Year`` field (no month or day). For each
record we assign a nominal ``event_date`` of July 1 of that year so that negative and
positive Sentinel-2 time windows are well defined, then keep only rows with
``event_date >= --min_event_date`` (default 2017-06-01) so coverage stays in the
Sentinel-2 era after May 2017.

Default ``--ds_path`` is ``sen12landslides/all_positives`` (windows under ``windows/fwn_mtli/``);
use the same root for ``rslearn dataset prepare``.

Window semantics match ``create_glc_landslide_windows.py`` and ICIMOD / Sen12 scripts.
Positive-window ``time_range`` for rasters starts at **00:00 UTC on the nominal event
calendar date** (July 1 of ``Year``) so GLC-style ``pre_sentinel2`` (e.g. ``-365d`` /
``365d`` in ``config.json``) queries strictly before that midnight.

Example:
python create_fwnmtli_windows.py \\
    --shapefile_path /weka/dfive-default/piperw/data/landslide/niaid/LandslideInventory_FarWesternNepal/LandslideInventory_FarWesternNepal_Points_Dated1992_2018.shp \\
    --ds_path data/landslide/sen12landslides/all_positives/ \\
    --sample_type positive
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import pandas as pd
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from shapely.strtree import STRtree
from upath import UPath

from rslp.utils.windows import calculate_bounds


def utc_midnight_same_calendar_day(dt: datetime) -> datetime:
    """Return 00:00:00 UTC on the same calendar date as ``dt`` (interpreted in UTC)."""
    dt = dt.astimezone(timezone.utc)
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


WINDOW_RESOLUTION = 10  # meters per pixel
WINDOW_SIZE_PIXELS = 64
LABEL_LAYER = "label"
DEFAULT_BUFFER_DISTANCE = 30.0
DEFAULT_FWN_SHP = os.path.join(
    "/weka/dfive-default/piperw/data/landslide/niaid",
    "LandslideInventory_FarWesternNepal",
    "LandslideInventory_FarWesternNepal_Points_Dated1992_2018.shp",
)
DEFAULT_MIN_EVENT_DATE = "2017-06-01"
DEFAULT_LANDSLIDE_DATASET_ROOT = os.path.join(
    "/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides",
    "all_positives",
)


class LandslideSpatialIndex:
    """Spatial index for catalog points intersecting a window polygon."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        """Initialize spatial index from a GeoDataFrame."""
        self.gdf = gdf.copy()
        self.tree = STRtree(self.gdf.geometry)
        print(f"Built spatial index with {len(self.gdf)} landslide points")

    def query_overlapping(
        self,
        window_geometry: shapely.Geometry,
        time_range: tuple[object, object] | None = None,
    ) -> list[dict]:
        """Query landslides overlapping with window geometry."""
        possible_matches_idx = self.tree.query(window_geometry)
        overlapping: list[dict] = []
        for idx in possible_matches_idx:
            if self.gdf.iloc[idx].geometry.intersects(window_geometry):
                row = self.gdf.iloc[idx]
                if time_range is not None:
                    event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
                    if pd.isna(event_date):
                        continue
                    event_date = pd.Timestamp(event_date)
                    if event_date.tzinfo is None:
                        event_date = event_date.tz_localize("UTC")
                    else:
                        event_date = event_date.tz_convert("UTC")
                    start_time, end_time = time_range
                    start_ts = pd.Timestamp(start_time)
                    end_ts = pd.Timestamp(end_time)
                    if start_ts.tzinfo is None:
                        start_ts = start_ts.tz_localize("UTC")
                    else:
                        start_ts = start_ts.tz_convert("UTC")
                    if end_ts.tzinfo is None:
                        end_ts = end_ts.tz_localize("UTC")
                    else:
                        end_ts = end_ts.tz_convert("UTC")
                    if not (start_ts <= event_date <= end_ts):
                        continue
                overlapping.append(
                    {
                        "id": str(row["id"]),
                        "geometry": row["geometry"],
                        "event_type": str(row.get("event_type", "unknown")),
                        "event_date": row.get("event_date"),
                    }
                )
        return overlapping


def create_window_pair(
    row_data: dict,
    dataset: Dataset,
    sample_type: str,
    spatial_index: LandslideSpatialIndex,
    buffer_distance: float = DEFAULT_BUFFER_DISTANCE,
    group: str = "fwn_mtli",
) -> None:
    """Create negative and/or positive windows for one Far-Western Nepal inventory point."""
    sample_id = str(row_data["id"])
    latitude, longitude = float(row_data["latitude"]), float(row_data["longitude"])
    event_date = row_data["event_date"]
    event_type = str(row_data["event_type"])
    location = str(row_data["location"])
    geometry = row_data["geometry"]

    sampling_date = (
        pd.to_datetime(event_date).to_pydatetime().replace(tzinfo=timezone.utc)
    )
    event_year = int(sampling_date.year)

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    window_size = WINDOW_SIZE_PIXELS
    max_extent = window_size * WINDOW_RESOLUTION
    bounds = calculate_bounds(dst_geometry, window_size)
    print(f"  Window size: {window_size} pixels (~{max_extent:.2f} m extent)")

    split = "val" if event_year >= 2021 else "train"

    min_x, min_y, max_x, max_y = bounds
    proj_min_x = min_x * dst_projection.x_resolution
    proj_min_y = min_y * dst_projection.y_resolution
    proj_max_x = max_x * dst_projection.x_resolution
    proj_max_y = max_y * dst_projection.y_resolution
    window_geom_projected = shapely.box(proj_min_x, proj_min_y, proj_max_x, proj_max_y)

    from pyproj import Transformer

    transformer = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    window_geom_wgs84_coords = shapely.ops.transform(
        transformer.transform, window_geom_projected
    )

    negative_start_time = sampling_date.replace(year=sampling_date.year - 1)
    negative_end_time = negative_start_time + timedelta(days=60)
    negative_overlapping = spatial_index.query_overlapping(
        window_geom_wgs84_coords, time_range=(negative_start_time, negative_end_time)
    )

    if sample_type == "positive":
        positive_start_time = sampling_date
        positive_end_time = sampling_date + timedelta(days=60)
        extended_start_time = sampling_date - timedelta(days=180)
        positive_overlapping = spatial_index.query_overlapping(
            window_geom_wgs84_coords,
            time_range=(extended_start_time, positive_start_time),
        )
        positive_overlapping = [
            ls for ls in positive_overlapping if ls["id"] != sample_id
        ]
        primary_landslide = {
            "id": sample_id,
            "geometry": geometry,
            "event_type": event_type,
            "event_date": event_date,
        }
        positive_overlapping.insert(0, primary_landslide)
        print(
            f"  Found {len(positive_overlapping)} catalog points in positive window "
            "(incl. primary)"
        )

    print(
        f"  Found {len(negative_overlapping)} catalog points in negative window "
        "(prior-year window)"
    )
    if len(negative_overlapping) > 0:
        print(
            f"  WARNING: Negative window has {len(negative_overlapping)} events; "
            "they will be labeled as landslide where applicable."
        )

    if sample_type in ("negative", "positive"):
        negative_window_name = (
            f"{sample_id}_negative_{latitude:.4f}_{longitude:.4f}_{event_year}"
        )
        print(f"Creating NEGATIVE window: {negative_window_name}")
        print(
            f"  Time range: {negative_start_time} to {negative_end_time} "
            "(1 year before nominal event, 60 days)"
        )

        negative_window = Window(
            storage=dataset.storage,
            group=group,
            name=negative_window_name,
            projection=dst_projection,
            bounds=bounds,
            time_range=(negative_start_time, negative_end_time),
            options={
                "split": split,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "event_date": event_date.isoformat()
                if hasattr(event_date, "isoformat")
                else str(event_date),
                "event_type": str(event_type),
                "location": str(location),
                "event_year": int(event_year),
                "window_size": int(window_size),
                "polygon_extent_m": float(max_extent),
                "window_type": "negative",
                "time_range_start": negative_start_time.isoformat(),
                "time_range_end": negative_end_time.isoformat(),
                "num_overlapping_landslides": int(len(negative_overlapping)),
                "buffer_distance_m": float(buffer_distance),
            },
        )
        negative_window.save()

        negative_features = create_labeled_features(
            negative_overlapping,
            negative_window,
            buffer_distance,
            dst_crs,
            sample_id,
            event_type,
            event_date,
        )
        negative_layer_dir = negative_window.get_layer_dir(LABEL_LAYER)
        GeojsonVectorFormat().encode_vector(negative_layer_dir, negative_features)
        negative_window.mark_layer_completed(LABEL_LAYER)

    if sample_type == "positive":
        positive_window_name = (
            f"{sample_id}_positive_{latitude:.4f}_{longitude:.4f}_{event_year}"
        )
        post_interval_start = utc_midnight_same_calendar_day(sampling_date)
        post_interval_end = post_interval_start + timedelta(days=60)
        print(f"Creating POSITIVE window: {positive_window_name}")
        print(
            f"  Raster time_range (S2 pre/post): {post_interval_start} to {post_interval_end} "
            "(UTC midnight of nominal event calendar date + 60d)"
        )
        print(
            f"  Catalog overlap query (labels): {positive_start_time} to {positive_end_time}"
        )

        positive_window = Window(
            storage=dataset.storage,
            group=group,
            name=positive_window_name,
            projection=dst_projection,
            bounds=bounds,
            time_range=(post_interval_start, post_interval_end),
            options={
                "split": split,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "event_date": event_date.isoformat()
                if hasattr(event_date, "isoformat")
                else str(event_date),
                "event_type": str(event_type),
                "location": str(location),
                "event_year": int(event_year),
                "window_size": int(window_size),
                "polygon_extent_m": float(max_extent),
                "window_type": "positive",
                "time_range_start": post_interval_start.isoformat(),
                "time_range_end": post_interval_end.isoformat(),
                "num_overlapping_landslides": int(len(positive_overlapping)),
                "buffer_distance_m": float(buffer_distance),
            },
        )
        positive_window.save()

        positive_features = create_labeled_features(
            positive_overlapping,
            positive_window,
            buffer_distance,
            dst_crs,
            sample_id,
            event_type,
            event_date,
        )
        landslide_features = [
            f for f in positive_features if f.properties.get("label") == "landslide"
        ]
        if len(landslide_features) == 0:
            raise ValueError(
                f"Positive window {positive_window_name} has no landslide features. "
                f"sample_id={sample_id}, overlaps={len(positive_overlapping)}"
            )

        positive_layer_dir = positive_window.get_layer_dir(LABEL_LAYER)
        GeojsonVectorFormat().encode_vector(positive_layer_dir, positive_features)
        positive_window.mark_layer_completed(LABEL_LAYER)

    print(f"✓ Created window(s) for sample {sample_id}\n")


def create_labeled_features(
    overlapping_landslides: list[dict],
    window: Window,
    buffer_distance: float,
    dst_crs: str,
    sample_id: str,
    event_type: str,
    event_date: object,
) -> list[Feature]:
    """Vector labels: background, optional no_data buffer union, then landslide disks."""
    from pyproj import Transformer

    features: list[Feature] = [
        Feature(
            window.get_geometry(),
            {
                "label": "no_landslide",
                "event_type": event_type,
                "event_date": str(event_date),
            },
        )
    ]

    if len(overlapping_landslides) == 0:
        return features

    transformer_to_proj = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)

    buffer_union = None
    for landslide in overlapping_landslides:
        geom_proj = shapely.ops.transform(
            transformer_to_proj.transform, landslide["geometry"]
        )
        geom_proj = (
            shapely.make_valid(geom_proj) if not geom_proj.is_valid else geom_proj
        )
        buffer_geom = geom_proj.buffer(buffer_distance)
        if buffer_union is None:
            buffer_union = buffer_geom
        else:
            buffer_union = buffer_union.union(buffer_geom)

    if buffer_union is not None:
        buffer_wgs84 = shapely.ops.transform(
            transformer_to_wgs84.transform, buffer_union
        )
        if not buffer_wgs84.is_empty:
            features.append(
                Feature(
                    STGeometry(WGS84_PROJECTION, buffer_wgs84, None),
                    {
                        "label": "no_data",
                        "buffer_distance_m": float(buffer_distance),
                    },
                )
            )

    min_pixel_area = WINDOW_RESOLUTION * WINDOW_RESOLUTION
    for landslide in overlapping_landslides:
        raw_geom = landslide["geometry"]
        geom_proj = shapely.ops.transform(transformer_to_proj.transform, raw_geom)
        if not geom_proj.is_valid:
            geom_proj = shapely.make_valid(geom_proj)
        if not geom_proj.is_valid:
            geom_proj = geom_proj.buffer(0)

        if isinstance(geom_proj, shapely.GeometryCollection) and not isinstance(
            geom_proj, shapely.Polygon | shapely.MultiPolygon
        ):
            polys = [
                g
                for g in geom_proj.geoms
                if isinstance(g, shapely.Polygon | shapely.MultiPolygon)
                and not g.is_empty
            ]
            geom_proj = shapely.unary_union(polys) if polys else shapely.Polygon()

        if geom_proj.is_empty or geom_proj.area < min_pixel_area:
            centroid = shapely.ops.transform(
                transformer_to_proj.transform,
                raw_geom.centroid if not raw_geom.is_empty else raw_geom,
            )
            geom_proj = centroid.buffer(WINDOW_RESOLUTION)
            print(
                f"    WARNING: Landslide {landslide['id']} had no/tiny polygon area; "
                f"buffered centroid to {WINDOW_RESOLUTION} m radius"
            )

        landslide_wgs84 = shapely.ops.transform(
            transformer_to_wgs84.transform, geom_proj
        )
        if landslide_wgs84.is_empty or not landslide_wgs84.is_valid:
            print(
                f"    WARNING: Landslide {landslide['id']} invalid after fix; skipping"
            )
            continue

        features.append(
            Feature(
                STGeometry(WGS84_PROJECTION, landslide_wgs84, None),
                {
                    "label": "landslide",
                    "landslide_id": str(landslide["id"]),
                    "event_type": str(landslide["event_type"]),
                    "event_date": str(landslide["event_date"]),
                    "is_primary": bool(landslide["id"] == sample_id),
                },
            )
        )

    print(
        f"    Created {len(features)} label features: "
        f"{sum(1 for f in features if f.properties['label']=='landslide')} landslides, "
        f"{sum(1 for f in features if f.properties['label']=='no_data')} buffer, "
        f"{sum(1 for f in features if f.properties['label']=='no_landslide')} background"
    )
    return features


def _nominal_event_date_from_year(year: int) -> pd.Timestamp:
    """Map inventory year to a mid-season UTC timestamp (see module docstring)."""
    return pd.Timestamp(year=year, month=7, day=1, tz="UTC")


def load_fwn_gdf(shapefile_path: UPath, min_event_date: str) -> gpd.GeoDataFrame:
    """Load Far-Western Nepal points, convert to WGS84, attach nominal event dates."""
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    if "Year" not in gdf.columns:
        raise ValueError("Shapefile missing required column 'Year'")
    if "OBJECTID" not in gdf.columns:
        raise ValueError("Shapefile missing required column 'OBJECTID'")

    gdf["Year"] = pd.to_numeric(gdf["Year"], errors="coerce")
    gdf = gdf.dropna(subset=["Year"])
    gdf["Year"] = gdf["Year"].astype(int)

    gdf["event_date"] = gdf["Year"].apply(_nominal_event_date_from_year)
    min_ts = pd.to_datetime(min_event_date, utc=True)
    gdf = gdf[gdf["event_date"] >= min_ts].copy()
    print(
        f"FWN points with nominal event_date >= {min_event_date} "
        f"(July 1 of Year): {len(gdf)}"
    )

    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    gdf["id"] = "fwn_" + gdf["OBJECTID"].astype(int).astype(str)
    gdf["event_type"] = "unknown"
    gdf["location"] = "far_western_nepal"
    return gdf


def create_windows_from_fwn(
    shapefile_path: UPath,
    ds_path: UPath,
    sample_type: str,
    max_samples: int | None = None,
    buffer_distance: float = DEFAULT_BUFFER_DISTANCE,
    group: str = "fwn_mtli",
    min_event_date: str = DEFAULT_MIN_EVENT_DATE,
) -> None:
    """Load Far-Western Nepal points and write rslearn windows under ``group``."""
    gdf = load_fwn_gdf(shapefile_path, min_event_date)
    if gdf.empty:
        print("No FWN rows left after filtering; nothing to do.")
        return

    spatial_index = LandslideSpatialIndex(gdf)

    if max_samples is not None and max_samples < len(gdf):
        gdf = gdf.sample(n=max_samples, random_state=42)
        print(f"Sampled {max_samples} events for window creation")

    rows_data = []
    for _, row in gdf.iterrows():
        rows_data.append(
            {
                "id": str(row["id"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "event_date": row["event_date"],
                "event_type": str(row.get("event_type", "unknown")),
                "location": str(row.get("location", "far_western_nepal")),
                "geometry": row["geometry"],
            }
        )

    dataset = Dataset(ds_path)
    jobs = [
        dict(
            row_data=row,
            dataset=dataset,
            sample_type=sample_type,
            spatial_index=spatial_index,
            buffer_distance=buffer_distance,
            group=group,
        )
        for row in rows_data
    ]

    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(p, create_window_pair, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(
        description=(
            "Create landslide detection windows from the Far-Western Nepal "
            "Multi-Temporal Landslide Inventory (points)"
        )
    )
    parser.add_argument(
        "--shapefile_path",
        type=str,
        default=DEFAULT_FWN_SHP,
        help=f"Path to the dated points shapefile (default: {DEFAULT_FWN_SHP})",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        default=DEFAULT_LANDSLIDE_DATASET_ROOT,
        help=(
            "rslearn dataset root (directory with config.json). "
            f"Default: {DEFAULT_LANDSLIDE_DATASET_ROOT} "
            "(windows written to windows/fwn_mtli/)."
        ),
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        required=True,
        choices=("positive", "negative"),
        help="'positive' = negative + positive windows; 'negative' = negatives only",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of events (default: all passing filters)",
    )
    parser.add_argument(
        "--buffer_distance",
        type=float,
        default=DEFAULT_BUFFER_DISTANCE,
        help=f"Buffer in meters for no_data around labeled points (default {DEFAULT_BUFFER_DISTANCE})",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="fwn_mtli",
        help="Dataset window group name (default: fwn_mtli)",
    )
    parser.add_argument(
        "--min_event_date",
        type=str,
        default=DEFAULT_MIN_EVENT_DATE,
        help=(
            "Only nominal event dates on or after this day (YYYY-MM-DD). "
            f"Default {DEFAULT_MIN_EVENT_DATE} keeps July-based years aligned with "
            "Sentinel-2 after May 2017."
        ),
    )
    args = parser.parse_args()

    create_windows_from_fwn(
        UPath(args.shapefile_path),
        UPath(args.ds_path),
        sample_type=args.sample_type,
        max_samples=args.max_samples,
        buffer_distance=args.buffer_distance,
        group=args.group,
        min_event_date=args.min_event_date,
    )
