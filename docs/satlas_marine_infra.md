Satlas Marine Infrastructure
----------------------------

The Satlas marine infrastructure model uses Sentinel-2 L1C scenes to predict the
locations of off-shore wind turbines and off-shore platforms. Note that off-shore
platforms is a catch-all category for human-made objects in the ocean that are not wind
turbines.

It inputs four mosaics of Sentinel-2 images, where each mosaic should be constructed
using Sentinel-2 scenes from a distinct 30-day period.

TODO: It is trained on a dataset consisting of 43,443 image patches (ranging from 300x300 to
1000x1000) with 37,145 ship labels.

TODO: insert example image


Inference
---------

First, download the model checkpoint to the `RSLP_PREFIX` directory.

    cd rslearn_projects
    mkdir -p project_data/projects/satlas_marine_infra/data_20241210_run_20241210_00/checkpoints/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/satlas_marine_infra/best.ckpt -O project_data/projects/satlas_marine_infra/data_20241210_run_20241210_00/checkpoints/last.ckpt

The Satlas prediction pipeline applies the model on a bounding box in a UTM projection
at 10 m/pixel. Given a longitude and latitude where you want to apply the model, you
can use the code below to identify a suitable bounding box:

    longitude = 120.148
    latitude = 24.007
    window_size = 4096

    import json
    import shapely
    from rslearn.const import WGS84_PROJECTION
    from rslearn.utils.geometry import STGeometry
    from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection

    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(longitude, latitude), None)
    dst_projection = get_utm_ups_projection(longitude, latitude, 10, -10)
    dst_geom = src_geom.to_projection(dst_projection)
    center_point = (
        int(dst_geom.shp.x) // 2048 * 2048,
        int(dst_geom.shp.y) // 2048 * 2048,
    )
    bounds = (
        center_point[0] - window_size // 2,
        center_point[1] - window_size // 2,
        center_point[0] + window_size // 2,
        center_point[1] + window_size // 2,
    )
    print(json.dumps(dst_projection.serialize()))
    print(json.dumps(bounds))

Run the prediction pipeline. The argument after the projection and bounds specifies the
time range, it should be a seven month range to give enough options to pick the four
30-day mosaics, note that the timestamps are ISO 8601 formatted.

    mkdir out_dir
    python -m rslp.main satlas predict MARINE_INFRA '{"crs": "EPSG:32651", "x_resolution": 10, "y_resolution": -10}' '[18432, -268288, 22528, -264192]' '["2024-01-01T00:00:00+00:00", "2024-08-01T00:00:00+00:00"]' out_dir/ scratch_dir/ --use_rtree_index false

You may need to delete the "scratch_dir" directory if it exists already. This is used
to store a temporary rslearn dataset for ingesting the Sentinel-2 input images.

This generates a GeoJSON in out_dir but it is in pixel coordinates. Convert to
longitude/latitude coordinates using this script (which can also be used to merge
multiple GeoJSONs produced by the prediction pipeline):

    mkdir merged_dir
    python -m rslp.main satlas merge_points MARINE_INFRA 2024-01 out_dir/ merged_dir/

Now you can open the GeoJSON to view predicted positions of marine infrastructure, e.g.
in qgis:

    qgis merged_dir/2024-01.geojson


Training
--------

First, download the training dataset:

    cd rslearn_projects
    mkdir -p project_data/datasets/satlas_marine_infra/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/satlas_marine_infra/satlas_marine_infra.tar -O project_data/datasets/satlas_marine_infra.tar
    tar xvf project_data/datasets/satlas_marine_infra.tar --directory project_data/datasets/satlas_marine_infra/

It is an rslearn dataset consisting of window folders like
`windows/label/2102272_1262592/`. Inside each window folder:

- `layers/sentinel2{.1,.2,.3}/` contains the four input Sentinel-2 mosaics.
- `layers/label/data.geojson` contains the positions of marine infrastructure. These
  are offset from the bounds of the window which are in `metadata.json`, so subtract
  the window's bounds to get pixel coordinates relative to the image.
- `layers/mask/mask/image.png` contains a mask specifying the valid portion of the
  window. The labels were originally annotated in WebMercator projection, but have been
  re-projected to UTM in this dataset; the transformation results in a non-rectangular
  extent, so the window corresponds to the rectangular bounds of that extent while the
  mask specifies the extent within those bounds. This is used in the mask step in the
  model configuration file `data/satlas_marine_infra/config.yaml` to black out the
  other parts of the input image.

Use the command below to train the model. Note that Weights & Biases is needed. You can
disable W&B with `--no_log true` but then it may be difficult to track the metrics.

    python -m rslp.rslearn_main model fit --config data/satlas_marine_infra/config.yaml --data.init_args.path project_data/datasets/satlas_marine_infra/

To visualize outputs on the validation set:

    mkdir vis
    python -m rslp.rslearn_main model test --config data/satlas_marine_infra/config.yaml --data.init_args.path project_data/datasets/satlas_marine_infra/ --model.init_args.visualize_dir=vis/
