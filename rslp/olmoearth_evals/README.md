## Evaluate OlmoEarth on Partner Tasks

This is code for evaluation of OlmoEarth and baselines on partner tasks. The code
here ensures that all models are able to accept a consistent input and produce a
consistent output for detection, segmentation, classification, and regression tasks.

## Datasets

Datasets for partner tasks are stored as [rslearn datasets](https://github.com/allenai/rslearn).
The structure is roughly as follows:

```
config.json
windows/
  group1/
    window_name1/
      metadata.json
      items.json
      layers/
        sentinel2/
          B02_B03_B04_B08/
            geotiff.tif
          ...
        sentinel2.1/
          B02_B03_B04_B08/
            geotiff.tif
          ...
        label/
          data.geojson
    window_name2/
      ...
  group2/
    ...
```

For a complete description of the rslearn dataset storage format, see
https://github.com/allenai/rslearn/blob/master/docs/DatasetFormat.md.

The dataset is organized into window folders like `windows/group1/window_name1/`, and
each window corresponds to a training example. Each window specifies spatiotemporal
bounds, and includes the inputs and labels for that location and time.

Here, the `sentinel2` and `sentinel2.1` folders contain Sentinel-2 images for different
timesteps, while `label/data.geojson` contains a label in a vector file. The label
differs by task, but for example, for classification tasks, it would contain a single
GeoJSON feature with a property containing the ground truth category name. For
segmentation tasks, the label is instead a GeoTIFF.

The `metadata.json` specifies the spatial bounds and time range of the window, but also
contains an `olmoearth_evals_split` key that is used to indicate whether the window is
used for training, validation, or testing.

### Forest Loss Driver Classification

The forest loss driver dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI and Amazon Conservation.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/forest_loss_driver.tar

For more details, [see the documentation in olmoearth_projects](https://github.com/allenai/olmoearth_projects/blob/main/docs/forest_loss_driver.md).

### Live Fuel Moisture Content (LFMC) Mapping

The LFMC dataset is adapted by the Allen Institute for AI from
[Globe-LFMC-2.0](https://springernature.figshare.com/articles/dataset/Globe-LFMC-2_0/25413790?backTo=%2Fcollections%2FGlobe-LFMC_2_0_An_enhanced_and_updated_database_for_Live_Fuel_Moisture_Content_research_%2F6980418&file=45049786),
a dataset created by Marta Yebra et al. Both the original dataset, and the dataset
converted to rslearn format, are released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/lfmc.tar

For more details, [see here](https://github.com/allenai/olmoearth_projects/blob/main/docs/lfmc.md).

### Mangrove Classification

The mangrove dataset is converted to rslearn format from [Global Mangrove Watch v4 Reference Samples](https://zenodo.org/records/17394267).
The dataset is created by [Global Mangrove Watch](https://www.mangrovealliance.org/global-mangrove-watch)
and is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/mangrove.tar

For more details, [see here](https://github.com/allenai/olmoearth_projects/blob/main/docs/mangrove.md).

### Marine Infrastructure Detection

The marine infrastructure detection dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/marine_infra.tar

For more details, [see here](https://github.com/allenai/rslearn_projects/blob/master/docs/satlas_marine_infra.md).

### Landsat Vessel Detection

The Landsat vessel detection dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/landsat_vessels.tar

For more details, [see here](https://github.com/allenai/rslearn_projects/blob/master/docs/landsat_vessels.md).

### Sentinel-1 Vessel Detection

The Sentinel-1 vessel detection dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/sentinel1_vessels.tar

### Sentinel-2 Vessel Detection

The Sentinel-2 vessel detection dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/sentinel2_vessels.tar

For more details, [see here](https://github.com/allenai/rslearn_projects/blob/master/docs/sentinel2_vessels.md).

### Sentinel-2 Vessel Attribute Prediction

The Sentinel-2 vessel attribute prediction dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI. We use the same dataset to evaluate separately on vessel
length estimation and vessel type classification.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/sentinel2_vessel_attribute.tar

For more details, [see here](https://github.com/allenai/rslearn_projects/blob/master/docs/sentinel2_vessels.md).

### Solar Farm Segmentation

The solar farm dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
by the Allen Institute for AI.

- Download link: https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/solar_farm.tar

## Running Evaluations

Here is an example of launching training for a given model and task:

```
# Download and extract the marine infrastructure dataset.
wget https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/marine_infra.tar
tar xvf marine_infra.tar -C partner_datasets/marine_infra/
# Set up RSLP_PREFIX where checkpoints will be saved.
export RSLP_PREFIX=./project_data/
# Run training. EVAL_ADAPTER_MODEL_ID is used to indicate which model to train with.
export EVAL_ADAPTER_MODEL_ID=olmoearth_tiny
rslearn model fit --config data/olmoearth_evals/tasks/marine_infra_base.yaml --config data/olmoearth_evals/tasks/marine_infra_ts.yaml --config data/olmoearth_evals/models/$EVAL_ADAPTER_MODEL_ID.yaml --run_name marine_infra_ts_olmoearth_tiny --project_name olmoearth_evals --data.init_args.path=./partner_task/marine_infra/
```

During training, it will log the training and validation metrics to W&B, and save the
checkpoint with the best validation metrics (along with the most recent checkpoint).
Then to evaluate on the test set using the best checkpoint on val:

```
export EVAL_ADAPTER_MODEL_ID=olmoearth_tiny
rslearn model test --config data/olmoearth_evals/tasks/marine_infra_base.yaml --config data/olmoearth_evals/tasks/marine_infra_ts.yaml --config data/olmoearth_evals/models/$EVAL_ADAPTER_MODEL_ID.yaml --run_name marine_infra_ts_olmoearth_tiny --project_name olmoearth_evals --data.init_args.path=./partner_task/marine_infra/ --log_mode=yes --load_checkpoint_mode=best
```

Not all models support all modalities or multi-modality.

- AnySat: supports all tasks.
- Clay: supports Sentinel-1/Sentinel-2/Landsat, but only one modality at a time.
- Copernicus-FM: supports Sentinel-1 and Sentinel-2.
- CROMA: supports Sentinel-1 and Sentinel-2.
- DINOv3: supports Sentinel-2 and Landsat, but only one modality at a time.
- Galileo: supports Sentinel-1 and Sentinel-2.
- OlmoEarth: supports all tasks.
- Panopticon: supports all tasks.
- Presto: supports Sentinel-1 and Sentinel-2.
- Prithvi: supports Sentinel-2 and Landsat, although it's not designed for non-HLS Sentinel-2.
- SatlasPretrain: in this eval it only supports Sentinel-2.
- TerraMind: supports Sentinel-1 and Sentinel-2.

Here are the `--config` options for the available tasks, which can replace the marine infrastructure ones in the example above.
Note that some tasks use multiple configuration files, with a base shared configuration file followed by different configs for
training with Sentinel-2 time series vs Sentinel-1 + Sentinel-2 multi-modal time series; later configuration files override
earlier ones.

```
# Forest loss driver.
--config data/olmoearth_evals/tasks/forest_loss_driver.yaml
# LFMC, can be S2 or S2+S1.
--config data/olmoearth_evals/tasks/lfmc_base.yaml --config data/olmoearth_evals/tasks/lfmc_ts.yaml
--config data/olmoearth_evals/tasks/lfmc_base.yaml --config data/olmoearth_evals/tasks/lfmc_mm.yaml
# Mangrove, can be S2 or S2+S1.
--config data/olmoearth_evals/tasks/mangrove_base.yaml data/olmoearth_evals/tasks/mangrove_ts.yaml
--config data/olmoearth_evals/tasks/mangrove_base.yaml data/olmoearth_evals/tasks/mangrove_mm.yaml
# Marine infrastructure detection, can be S2 or S2+S1.
--config data/olmoearth_evals/tasks/marine_infra_base.yaml data/olmoearth_evals/tasks/marine_infra_ts.yaml
--config data/olmoearth_evals/tasks/marine_infra_base.yaml data/olmoearth_evals/tasks/marine_infra_mm.yaml
# Landsat vessel detection.
--config data/olmoearth_evals/tasks/landsat_vessels.yaml
# Sentinel-1 vessel detection.
--config data/olmoearth_evals/tasks/sentinel1_vessels.yaml
# Sentinel-2 vessel detection.
--config data/olmoearth_evals/tasks/sentinel2_vessels.yaml
# Sentinel-2 vessel length estimation.
--config data/olmoearth_evals/tasks/sentinel2_vessel_length.yaml
# Sentinel-2 vessel type prediction.
--config data/olmoearth_evals/tasks/sentinel2_vessel_type.yaml
# Solar form segmentation, can be S2 or S2+S1.
--config data/olmoearth_evals/tasks/solar_farm_base.yaml data/olmoearth_evals/tasks/solar_farm_ts.yaml
--config data/olmoearth_evals/tasks/solar_farm_base.yaml data/olmoearth_evals/tasks/solar_farm_mm.yaml
```
