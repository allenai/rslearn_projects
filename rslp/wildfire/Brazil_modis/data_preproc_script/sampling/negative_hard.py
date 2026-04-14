"""Sampling scheme for the hard-test split."""

import math
import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import hydra
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from shapely.geometry import Polygon
from tqdm import tqdm

from data_preproc_script.constants import (
    CONFIG_PATH,
    EE_CRS,
    KEY_COLUMNS_SAMP,
)
from data_preproc_script.utils import assign_val, create_logger

logger = create_logger("neg_sampling_hard", "logs/neg_sampling_hard")


def create_square(row: gpd.GeoSeries, side_length: float = 0.25) -> Polygon:
    """Build a square polygon centered on the point geometry in ``row``."""
    lon, lat = row.geometry.x, row.geometry.y
    return Polygon(
        [
            (lon - side_length / 2, lat - side_length / 2),
            (lon + side_length / 2, lat - side_length / 2),
            (lon + side_length / 2, lat + side_length / 2),
            (lon - side_length / 2, lat + side_length / 2),
        ]
    )


def grid_identifier(
    pos_grid_path: os.PathLike,
    grid_path: os.PathLike,
    fire_weather_path: os.PathLike,
    region_path: os.PathLike,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    neg_samples: gpd.GeoDataFrame | None = None,
    year: int = 2015,
    sampling_ratio: float = 2.0,
    seed: int = 42,
    bucket: list[float] = [0, 2, 4, 6, 10, 15, 20, 25, 30, 40, 100],
) -> gpd.GeoDataFrame:
    """Sample hard negatives for one year using FWI, month, and land-cover cues.

    Args:
        pos_grid_path (os.PathLike): Positive samples path
        grid_path (os.PathLike): All grid positions path
        fire_weather_path (os.PathLike): FWI data path
        region_path (os.PathLike): Canada region bounds path
        min_x (float): Minimum longitude of the study area.
        max_x (float): Maximum longitude of the study area.
        min_y (float): Minimum latitude of the study area.
        max_y (float): Maximum latitude of the study area.
        neg_samples (Optional[gpd.GeoDataFrame], optional): Already sampled negative grid. Defaults to None.
        year (int, optional): Current year for sampling. Defaults to 2015.
        sampling_ratio (float, optional): Ratio of negative to samples. Defaults to 2.0.
        seed (int, optional): Random seed for reproductibility. Defaults to 42.
        bucket (List[float], optional): FWI buckets to sample from. Defaults to [0, 2, 4, 6, 10, 15, 20, 25, 30, 40, 100].

    Returns:
        gpd.GeoDataFrame: Sampled negatives example.
    """
    # Compute Canada grid & Extract Canada bounds
    canada_grid_df = gpd.read_file(grid_path)
    canada_rect = Polygon(
        [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
    ).buffer(0.25)
    canada_bounds = gpd.read_file(region_path)
    canada_bounds = canada_bounds.to_crs(epsg=4326)

    # Load the positive and negative grid and filter up to the end of the year
    pos_sample = gpd.read_file(pos_grid_path)
    end_date = datetime(year, 12, 31)
    start_date = datetime(year, 1, 1)
    up_year_sample = pos_sample[pos_sample["start_date"] <= end_date]
    year_sample = up_year_sample[up_year_sample["start_date"] >= start_date]

    # Starting Region loop
    tot_sampled_list = []
    for region in year_sample["region"].unique():
        logger.info(f"Region {region} Sampling")
        canada_geo = canada_bounds[canada_bounds["PRENAME"] == region]

        canada_grid_df_region = canada_grid_df[
            canada_grid_df["geometry"].intersects(canada_geo.geometry.values[0])
        ]
        # ID filtering + Region -> Could be replace by spatial join
        filter_ids = up_year_sample[up_year_sample.region == region]["grid_id"].unique()
        year_sample_region = year_sample[year_sample.region == region]
        filter_canada_grid_df = canada_grid_df_region[
            ~canada_grid_df_region["id"].isin(filter_ids)
        ]

        logger.info(f"Number of locations {canada_grid_df_region.shape[0]}")
        logger.info(f"Number of locations with positive samples {len(filter_ids)}")
        logger.info(
            f"Number of positive samples for this year and region {len(year_sample_region)}"
        )

        if neg_samples is not None:
            filter_ids = neg_samples["grid_id"].unique()
            filter_canada_grid_df = filter_canada_grid_df[
                ~filter_canada_grid_df["id"].isin(filter_ids)
            ]
            logger.info(f"Number of locations with negative samples {len(filter_ids)}")

        p_avl = len(filter_canada_grid_df) / len(canada_grid_df_region) * 100
        logger.info(f"Number of locations available {filter_canada_grid_df.shape[0]}")
        logger.info(f"Percentage of available negative samples: {p_avl:.2f}%")

        # Build FWI GeoPandas and filter on Canada (LONG)
        fwi_data = xr.open_dataset(fire_weather_path)
        fwi_data = fwi_data.to_dataframe().reset_index()
        fwi_data = fwi_data.dropna(subset=["fwinx_mean"])
        fwi_data = gpd.GeoDataFrame(
            fwi_data,
            geometry=gpd.points_from_xy(fwi_data["longitude"], fwi_data["latitude"]),
            crs=EE_CRS,
        )
        fwi_data = fwi_data[
            fwi_data["geometry"].within(canada_rect)
        ]  # For Speed-up NaN values over the sea
        fwi_data["geometry"] = fwi_data.apply(create_square, axis=1)

        # Assign FWI to the negative samples
        year_sample_region = assign_val(
            year_sample_region,
            fwi_data,
            ["id", "start_date", "lc"],
            "fwinx_mean",
            temp_cols=["start_date", "valid_time"],
        )
        filter_canada_grid_df = assign_val(
            filter_canada_grid_df, fwi_data, ["id", "valid_time", "lc"], "fwinx_mean"
        )
        year_sample_region["month"] = year_sample_region["start_date"].dt.month
        year_sample_region["bucket"] = pd.cut(
            year_sample_region["fwinx_mean"], bucket, labels=False, include_lowest=True
        )
        filter_canada_grid_df["month"] = filter_canada_grid_df["valid_time"].dt.month
        filter_canada_grid_df["bucket"] = pd.cut(
            filter_canada_grid_df["fwinx_mean"],
            bucket,
            labels=False,
            include_lowest=True,
        )
        logger.info(
            f"Total number of samples before sampling {filter_canada_grid_df.shape[0]}"
        )

        tot_year_sample_list = []
        unique_comb = year_sample_region[["month", "lc", "bucket"]].drop_duplicates()
        for month, lc, bucket_val in tqdm(
            unique_comb.itertuples(index=False),
            desc="Sampling Negative Data",
            total=len(unique_comb),
        ):
            sample_canada_grid_df = filter_canada_grid_df[
                (filter_canada_grid_df["month"] == month)
                & (filter_canada_grid_df["lc"] == lc)
                & (filter_canada_grid_df["bucket"] == bucket_val)
            ]

            year_sample_region_month_lc = year_sample_region[
                (year_sample_region["month"] == month)
                & (year_sample_region["lc"] == lc)
                & (year_sample_region["bucket"] == bucket_val)
            ]
            n_samples = math.ceil(sampling_ratio * len(year_sample_region_month_lc))
            if n_samples > sample_canada_grid_df.shape[0]:
                logger.info(
                    f"Number of samples for month {month}, lc {lc}, and bucket {bucket_val} is {sample_canada_grid_df.shape[0]} with {n_samples} required"
                )
                missing_samples = n_samples - sample_canada_grid_df.shape[0]
                logger.warning(
                    f"Missing samples for month {month}, lc {lc}, and bucket {bucket_val} is {missing_samples}"
                )
                n_samples = sample_canada_grid_df.shape[0]
                month_canada_grid_df = filter_canada_grid_df[
                    (filter_canada_grid_df["month"] == month)
                ]

                # Sample the available samples
                sampled_df = sample_canada_grid_df.sample(
                    n=n_samples, random_state=seed
                ).reset_index(drop=True)

                # Sample the missing samples from the month
                mis_sampled_df = month_canada_grid_df.sample(
                    n=missing_samples, random_state=seed
                ).reset_index(drop=True)
                sampled_df = pd.concat([sampled_df, mis_sampled_df], axis=0)

            else:
                sampled_df = sample_canada_grid_df.sample(
                    n=n_samples, random_state=seed
                ).reset_index(drop=True)

            tot_year_sample_list.append(sampled_df)

        # Yearly Post-Processing
        tot_year_sample_df = pd.concat(tot_year_sample_list, axis=0)
        tot_year_sample_df = tot_year_sample_df.rename(
            columns={
                "valid_time": "start_date",
                "geometry_left": "geometry",
                "id": "grid_id",
            }
        )
        tot_year_sample_df = tot_year_sample_df[
            ["grid_id", "start_date", "fwinx_mean", "bucket", "geometry"]
        ]
        tot_year_sample_df = gpd.GeoDataFrame(
            tot_year_sample_df, geometry=tot_year_sample_df.geometry, crs=EE_CRS
        )
        tot_year_sample_df = pd.concat(
            [tot_year_sample_df, tot_year_sample_df.bounds], axis=1
        )
        tot_year_sample_df["center_x"] = tot_year_sample_df.centroid.x
        tot_year_sample_df["center_y"] = tot_year_sample_df.centroid.y
        logger.info(
            f"Total number of negative samples after sampling {tot_year_sample_df.shape[0]} compared to positive samples {len(year_sample_region)}"
        )

        # Add Region Mapping with relative area assignment
        canada_bounds_region = canada_bounds.copy()
        canada_bounds_region["geometry_can"] = canada_bounds_region["geometry"]
        tot_year_sample_df = gpd.sjoin(
            tot_year_sample_df,
            canada_bounds_region[["PRENAME", "geometry_can", "geometry"]],
            how="inner",
            predicate="intersects",
        )

        tot_year_sample_df["area"] = tot_year_sample_df.apply(
            lambda x: x["geometry"].intersection(x["geometry_can"]).area, axis=1
        )
        tot_year_sample_df = tot_year_sample_df.sort_values(
            by=["grid_id", "start_date", "area"], ascending=[True, True, False]
        )
        tot_year_sample_df = tot_year_sample_df.drop_duplicates(
            subset=KEY_COLUMNS_SAMP, keep="first"
        )
        tot_year_sample_df = tot_year_sample_df.drop(
            ["geometry_can", "area", "index_right"], axis=1
        )
        tot_year_sample_df.rename(columns={"PRENAME": "region"}, inplace=True)
        logger.info(
            f"Total number of negative samples after region mapping {tot_year_sample_df.shape[0]}"
        )

        tot_sampled_list.append(tot_year_sample_df)

    tot_sampled_df = pd.concat(tot_sampled_list, axis=0)
    return tot_sampled_df


@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="sampling_hard"
)
def sampling_wrapper(cfg: DictConfig) -> None:
    """Run hard-negative sampling and write the combined output to disk."""
    if cfg.neg_grid_path is not None:
        tot_neg_samples = gpd.read_file(cfg.neg_grid_path)
        tot_neg_samples = tot_neg_samples[
            tot_neg_samples["start_date"].dt.year < cfg.start_year
        ]
    else:
        tot_neg_samples = gpd.GeoDataFrame()

    for year in tqdm(
        range(cfg.start_year, cfg.end_year + 1), desc="Yearly Sampling Negative Data"
    ):
        logger.info(f"Sampling negative data for year {year}")

        fire_weather_path = (
            Path(cfg.fire_weather_path) / str(year) / f"fwi_dc_agg_{year}.nc"
        )

        if year == 2015:
            neg_samples = grid_identifier(
                pos_grid_path=cfg.pos_grid_path,
                grid_path=cfg.grid_path,
                fire_weather_path=fire_weather_path,
                region_path=cfg.region_path,
                min_x=cfg.bounds.min_x,
                max_x=cfg.bounds.max_x,
                min_y=cfg.bounds.min_y,
                max_y=cfg.bounds.max_y,
                year=year,
                sampling_ratio=cfg.sampling_ratio,
                bucket=cfg.bucket,
                seed=cfg.seed,
            )
        else:
            neg_samples = grid_identifier(
                pos_grid_path=cfg.pos_grid_path,
                grid_path=cfg.grid_path,
                fire_weather_path=fire_weather_path,
                region_path=cfg.region_path,
                min_x=cfg.bounds.min_x,
                max_x=cfg.bounds.max_x,
                min_y=cfg.bounds.min_y,
                max_y=cfg.bounds.max_y,
                year=year,
                sampling_ratio=cfg.sampling_ratio,
                bucket=cfg.bucket,
                seed=cfg.seed,
                neg_samples=tot_neg_samples,
            )

        tot_neg_samples = pd.concat([tot_neg_samples, neg_samples])

    pos_samples = gpd.read_file(cfg.pos_grid_path)
    max_pos_id = pos_samples["id"].max()
    tot_neg_samples["id"] = range(max_pos_id + 1, max_pos_id + len(tot_neg_samples) + 1)
    tot_neg_samples.to_file(cfg.output_path)


if __name__ == "__main__":
    sampling_wrapper()
