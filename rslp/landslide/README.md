# Landslide Detection

This project aims to detect landslides with semantic segmentation, i.e. predict each pixel as `no_landslide` or `landslide`.

## Data

- Sen12Landslides: 75,0000+ landslide polygons from events occurring between 2016-2023
- ICIMOD: 792 landslide polygons in Nepal occurring between 2015-2016 (there are more events from 2001-2016)
- OSM Ski Resorts: OpenStreetMap polygons of ski resorts, used as false positives

### 1. Create Windows

Run the following commands to create windows...

for Sen12Landslides:
```
python -m rslp.landslide.scripts.create_sen12landslides_windows --shapefile_path /weka/dfive-default/piperw/data/landslide/sen12landslides/inventories.shp --ds_path /weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives/ --sample_type positive --group sen12_landslides
```

for ICIMOD:
```
python -m rslp.landslide.scripts.create_icimod_landslide_windows --shapefile_path /weka/dfive-default/piperw/data/landslide/icimod/data/14dist_ls.shp --ds_path /weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives/ --sample_type positive --group icimod
```

for OSM-ski, the .pbf files are huge, so extract a smaller region before creating windows:
```
osmium extract -b 6.0,45.5,10.5,47.5 \
  /path/to/europe-latest.osm.pbf \
  -o /path/to/alps-ski-trial.osm.pbf

&&

python rslp/landslide/scripts/create_osm_ski_windows.py \
  --pbf_path /weka/dfive-default/piperw/data/alps-ski-trial.osm.pbf\
  --ds_path /weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives \
  --group osm_ski_resorts_trial \
  --max_samples 20 \
  --sample_mode prefix \
  --areas_only \
  --num_workers 4
```

### 2. Prepare, Ingest and Materialize
```
ROOT=/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives
GROUP={sen12_landslide, icimod, osm_ski}

rslearn dataset prepare --root "$ROOT" --group "$GROUP"
rslearn dataset ingest --root "$ROOT" --group "$GROUP"
rslearn dataset materialize --root "$ROOT" --group "$GROUP"
```
The `ingest` step is only needed for the `srtm` layer.

### 3. Rasterize
The above steps create vector geojsons, but we need rasters for training.

Run the following script to generate label rasters for all windows in `all_positives/`:
```
python rslp/landslide/scripts/rasterize_label_vectors.py   --ds_path /weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives ```

### 4. Splits
I manually sampled 100 held out windows from Sen12Landslides to assure correctness and diversity.


## Finetune OlmoEarth
So far, I have just been finetuning in interactive session with 1-4 gpus. This requires installation of olmoearth_pretrain, rslearn, and rslearn_projects.

```
rslearn model fit --config data/landslide/model.yaml
```
