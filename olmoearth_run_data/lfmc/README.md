# LFMC

This covers fine-tuning and prediction workflows for the LFMC model.

## Prepare labels

This will prepare labels for CONUS west of 100°W:

```shell
python -m rslp.lfmc.fuel_type.prepare_labels_herbaceous_woody \
    --output_dir $(pwd)/olmoearth_run_data/lfmc/ \
    --preset conus \
    --bbox="-124.7844079,24.7433195,-100,49.3457868"
```

## Generating a prediction geometry

1. Download MTBS wildfire perimeter data:

    ```shell
    curl -o $(pwd)/olmoearth_run_data/lfmc/mtbs_perimeter_data.zip \
        https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip
    ```

    ```shell
    pushd $(pwd)/olmoearth_run_data/lfmc; \
        unzip mtbs_perimeter_data.zip -d mtbs_perimeter_data \
        ; popd
    ```

1. Output an example wildfire and add start/end times:

    Note that this step requires [GDAL](https://gdal.org/).

    ```shell
    pushd $(pwd)/olmoearth_run_data/lfmc/mtbs_perimeter_data; \
        ogr2ogr -f GeoJSON /dev/stdout \
            mtbs_perims_DD.shp \
            -where "Incid_Name LIKE 'BOOTLEG' AND Ig_Date >= '2021-01-01' AND Ig_Date <= '2021-12-31'" \
            | jq '.features[0].properties += {"oe_start_time": "2021-06-06T00:00:00+00:00", "oe_end_time": "2021-07-06T00:00:00+00:00"}' > prediction_request_geometry.geojson \
        ; popd
    ```

1. Optional: filter to areas with known woody or herbaceous material:

    ```shell
    # For woody
    python -m rslp.lfmc.fuel_type.geojson_cog_intersect \
        --raster_files gs://rslearn-eai/artifacts/nlcd/Tree_2011_2024/rcmap_tree_2021.cog.tif gs://rslearn-eai/artifacts/nlcd/Shrub_2011_2024/rcmap_shrub_2021.tif \
        --geometry_file $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.geojson \
        --output $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.new.geojson \
        --simplify_tolerance 0.01 && \
    mv $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.new.geojson $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.geojson
    ```

    ```shell
    # For herbaceous
    python -m rslp.lfmc.fuel_type.geojson_cog_intersect \
        --raster_files gs://rslearn-eai/artifacts/nlcd/Tree_2011_2024/rcmap_herbaceous_2021.cog.tif \
        --geometry_file $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.geojson \
        --output $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.new.geojson \
        --simplify_tolerance 0.01 && \
    mv $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.new.geojson $(pwd)/olmoearth_run_data/lfmc/prediction_request_geometry.geojson
    ```
