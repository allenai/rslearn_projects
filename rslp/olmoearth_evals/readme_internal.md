## olmoearth_evals

This is code for evaluation of OlmoEarth and baselines on downstream tasks. The code
here ensures that all models are able to accept a consistent input and produce a
consistent output for detection, segmentation, classification, and regression tasks.

Here is an example of launching a 2D grid of models and tasks:

```
python -m rslp.main olmoearth_evals launch --models='[croma,olmoearth,panopticon,presto,satlaspretrain,terramind]' --tasks='[pastis_uni,pastis_ts]' --prefix 20251006 --image_name favyen/rslphelios16
```

## Comparing Models

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

The encoder freeze schedule is selected at launch time via `--freeze=<name>`,
where `<name>` is one of the YAMLs in `data/olmoearth_evals/freezes/`:

- `freezefor1_lrfactor1` (unfreeze after 1 epoch)
- `freezefor10_lrfactor10` (unfreeze after 10 epochs, encoder LR /10)
- `freezefor20_lrfactor1` (default, unfreeze after 20 epochs)
- `freezefor20_lrfactor20` (unfreeze after 20 epochs, encoder LR /20)
- `frozen` (encoder stays frozen for the whole run)

```
# Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,dinov3,galileo,olmoearth,panopticon,presto,prithvi,satlaspretrain,terramind]' --tasks='[pastis_uni,pastis_ts,marine_infra_uni,marine_infra_ts,wind_turbine_uni,wind_turbine_ts,solar_farm_uni,solar_farm_ts,sentinel2_vessel_length,sentinel2_vessel_type,sentinel2_vessels,lfmc_uni,lfmc_ts,mangrove_uni,mangrove_ts,awf_ts,nandi_ts,ecosystem]' --prefix 20251007b --image_name favyen/rslphelios16 --project final_downstream_eval_train
# Sentinel-1 + Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,copernicusfm,croma,galileo,olmoearth,panopticon,presto,terramind]' --tasks='[pastis_mm,marine_infra_mm,wind_turbine_mm,solar_farm_mm,lfmc_mm,mangrove_mm,awf_mm,nandi_mm]' --prefix 20251007b --image_name favyen/rslphelios16 --project final_downstream_eval_train
# Sentinel-1 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,galileo,olmoearth,panopticon,presto,terramind]' --tasks='[sentinel1_vessels]' --prefix 20251007b --image_name favyen/rslphelios16 --project final_downstream_eval_train
# Landsat tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,dinov3,olmoearth,panopticon,prithvi]' --tasks='[landsat_vessels]' --prefix 20251007b --image_name favyen/rslphelios17 --project final_downstream_eval_train
```

Here are subset to run after dropping Presto (very slow) and not doing the unitemporal version of
tasks that have a multitemporal option. And dropping PASTIS (already covered in olmoearth_pretrain).

```
# Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,dinov3,galileo,olmoearth,panopticon,prithvi,satlaspretrain,terramind]' --tasks='[marine_infra_ts,wind_turbine_ts,solar_farm_ts,sentinel2_vessel_length,sentinel2_vessel_type,sentinel2_vessels,lfmc_ts,mangrove_ts,awf_ts,nandi_ts,ecosystem]' --prefix final --image_name favyen/rslphelios20 --project final_downstream_eval_train
# Sentinel-1 + Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,copernicusfm,croma,galileo,olmoearth,panopticon,terramind]' --tasks='[marine_infra_mm,wind_turbine_mm,solar_farm_mm,lfmc_mm,mangrove_mm,awf_mm,nandi_mm]' --prefix final --image_name favyen/rslphelios20 --project final_downstream_eval_train
# Sentinel-1 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,galileo,olmoearth,panopticon,terramind]' --tasks='[sentinel1_vessels]' --prefix final --image_name favyen/rslphelios18 --project final_downstream_eval_train
# Landsat tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,dinov3,olmoearth,panopticon,prithvi]' --tasks='[landsat_vessels]' --prefix final --image_name favyen/rslphelios18 --project final_downstream_eval_train
```

Use `--test=true` to run the test stage, e.g.:

```
python -m rslp.main olmoearth_evals launch --models='[satlaspretrain]' --tasks='[solar_farm_ts]' --prefix final --image_name favyen/rslphelios20 --project final_downstream_eval_train --test=true
```

Then get the metric, e.g.:

```
python -m rslp.olmoearth_evals.scripts.get_best_wandb_metric --project final_downstream_eval_train --run_regex 'final_landsat_vessels_.*' --runs_after '2025-10-20T00:00:00Z' --metric test_eval_task/F1 --mode max
```


The AEF "models" are a bit unique because they rely on separately downloaded AEF embeddings, but we keep them here for consistency.
```
# AEF
python -m rslp.main olmoearth_evals launch --models='[aef]' --tasks='[nandi_aef,lfmc_aef,awf_aef,ecosystem_aef]' --prefix final --image_name favyen/rslpomp20251022a --project final_downstream_eval_train
```

## Evaluating OlmoEarth Models

Use `--model_config` to override the checkpoint path. Use `--freeze` to pick a
freeze schedule (default `freezefor20_lrfactor1`). You can also override
arguments like the LR by passing it via `--extra_args`.
Here is an example with all three:

```bash
python -m rslp.main olmoearth_evals launch \
    --models='[olmoearth]' \
    --tasks='[pastis_uni]' \
    --prefix run_custom_checkpoint_lr0.001 \
    --image_name favyen/rslpomp \
    --project 2026_04_30_olmoearth_evals \
    --clusters='[ai2/jupiter,ai2/saturn,ai2/ceres]' \
    --priority urgent \
    --freeze=freezefor10_lrfactor10 \
    --extra_args='["--model.init_args.optimizer.init_args.lr", "0.001"]' \
    --model_config='{"checkpoint_path": "/weka/dfive-default/helios/checkpoints/yawenzzzz/single_bandset_no_s1_drop_random_time_dropout_0.2/step667200"}'
```

### Recommended Tasks

Some tasks are quite slow or have highly variable performance.
Here are recommended tasks for evaluating OlmoEarth models.

Fast tasks (an hour or two):

```bash
python -m rslp.main olmoearth_evals launch \
    --models='[olmoearth]' \
    --tasks='[pastis_uni,awf_ts,nandi_ts,ecosystem]' \
    --prefix my_eval \
    --image_name favyen/rslpomp \
    --project 2026_04_30_eval
```

Slower but still reasonable tasks (less than a day):

```bash
python -m rslp.main olmoearth_evals launch \
    --models='[olmoearth]' \
    --tasks='[pastis_ts,lfmc_ts,africa_crop_mask,canada_crops_coarse,descals,glance,lcmap_lu,us_trees,worldcover,worldcover_200_per_class]' \
    --prefix my_eval \
    --image_name favyen/rslpomp \
    --project 2026_04_30_eval
```
