# LFMC

This covers fine-tuning and prediction workflows for the LFMC model.

## Prepare labels

This will prepare labels for CONUS west of 100°W:

```shell
python -m rslp.lfmc.prepare_labels_herbaceous_woody \
    --csv_path /tmp/lfmc-labels-conus-herbaceous-woody.csv \
    --preset conus \
    --bbox="-124.7844079,24.7433195,-100,49.3457868"
```

## Generating a prediction geometry

1. Download MTBS wildfire perimeter data:

    ```shell
    curl -o mtbs_perimeter_data.zip https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip
    ```

    ```shell
    unzip mtbs_perimeter_data.zip -d mtbs_perimeter_data
    ```

1. Output an example wildfire and add start/end times:

    Note that this step requires [GDAL](https://gdal.org/).

    ```shell
    pushd mtbs_perimeter_data; \
        ogr2ogr -f GeoJSON /dev/stdout \
            mtbs_perims_DD.shp \
            -where "Incid_Name LIKE 'BOOTLEG' AND Ig_Date >= '2021-01-01' AND Ig_Date <= '2021-12-31'" \
            | jq '.features[0].properties += {"oe_start_time": "2021-06-06T00:00:00+00:00", "oe_end_time": "2021-07-06T00:00:00+00:00"}' > prediction_request_geometry.geojson \
        ; popd
    ```
