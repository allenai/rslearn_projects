# For ewds.climate:
```shell
export CDSAPI_URL="https://ewds.climate.copernicus.eu/api"
source ./.secrets.env  # sets CDSAPI_KEY
python -m script.cds_download
```


# Contents:
The raw file is a daily, global gridded NetCDF at 0.25° x 0.25° spatial resolution (grid: '0.25/0.25' in fwi.yaml). Its dimensions are:

valid_time — one entry per day (Jan 1 through Dec 31)
latitude / longitude — 0.25° spacing
Its data variables are the two requested fields from the config:

drtcode — Drought Code (a slow-responding fuel moisture index)
fwinx — Fire Weather Index (the composite danger rating)
So for a single year you get a 3D cube: ~365 days x ~720 latitudes x ~1440 longitudes, with two variables per cell.


# 2. How it is processed / aggregated
Immediately after download, if cfg.agg == "fwi_agg" (which it is), fwi_agg() in utils.py is called on the raw daily NetCDF. This does:

Build the 8-day temporal grid (line 182): build_temporal_grid(year, year, 8) produces dates like Jan 1, Jan 9, Jan 17, ... Dec 25 — the same grid used for positives in temporal_grid_agg.py.

Aggregate daily values into 8-day windows (lines 192-213): For each 8-day window [date, date+7], it computes mean, min, and max across the valid_time dimension. This produces variables like fwinx_mean, fwinx_min, fwinx_max, drtcode_mean, etc. Each window is tagged with its start date as the new valid_time coordinate.

Concatenate all windows back along the valid_time dimension (line 216).

Save two outputs (lines 233-240):

fwi_dc_agg_{year}.nc — the full aggregated NetCDF (this is what negative sampling reads)
fwi_dc_agg_{YYYYMMDD}.tif — one GeoTIFF per 8-day window (for visualization/debugging)
Result: fwi_dc_agg_{year}.nc has the same spatial grid (0.25° x 0.25°, global) but the time dimension is now ~46 steps per year (every 8 days) instead of 365 daily steps. Each step has fwinx_mean, fwinx_min, fwinx_max, drtcode_mean, etc.
