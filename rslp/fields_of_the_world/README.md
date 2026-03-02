This project includes:

- Scripts to analyze the spatial and temporal distribution of
  [Fields of the World](https://source.coop/kerner-lab/fields-of-the-world)
- Scripts to convert it to rslearn dataset.
- Dataset and model configs.

## Paths

- Original dataset is copied from S3 to WEKA at `/weka/dfive-default/rslearn-eai/artifacts/fields_of_the_world/`.
- The converted rslearn dataset with the two original Sentinel-2 images is at `/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_orig/`. Note that these images are in WGS84 projection at 6 m/pixel at the equator. Only B02/B03/B04/B08 are available.
- And with eight Sentinel-2 obtained with rslearn at `/weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/`. There is one image for each 30-day period, and the overall time range is based on the per-country time ranges in the original dataset.

## Analysis

`make_country_geojsons.py` will create a per-country GeoJSON file containing a point
corresponding to the center of each labeled patch. These are available at
`/weka//dfive-default/rslearn-eai/datasets/fields_of_the_world/analysis/country_geojsons/`
and can be opened in qgis to see the spatial distribution.

## Convert to rslearn dataset

The conversion script outputs two datasets, one with the original two Sentinel-2 images
from the dataset (one captured early in season, one later in season) and one with
windows in UTM projection.

In both cases there are four classes:

- 0: background
- 1: field interior
- 2: field boundary
- 3: nodata (invalid)

These countries only have foreground labels, no background labels, they may not be
suitable for training:

- Kenya
- India
- Rwanda
- Brazil

To convert (the paths are hardcoded):

```
# Create dataset folders and copy dataset config files.
mkdir /weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_orig/
mkdir /weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/
cp data/fields_of_the_world/dataset_orig.json /weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_orig/config.json
cp data/fields_of_the_world/dataset_utm.json /weka/dfive-default/rslearn-eai/datasets/fields_of_the_world/rslearn_dataset_utm/config.json
# Run conversion script, which populates both datasets.
python -m rslp.fields_of_the_world.convert_to_rslearn_dataset
```

## Training

Train a model on the original Sentinel-2 images from the source dataset:

```
rslearn model fit --config data/fields_of_the_world/model_two_image.yaml
```

Train a model on our version of the images:

```
rslearn model fit --config data/fields_of_the_world/model_eight_image.yaml
```
