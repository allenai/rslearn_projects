This is code for evaluation of OlmoEarth and baselines on downstream tasks. The code
here ensures that all models are able to accept a consistent input and produce a
consistent output for detection, segmentation, classification, and regression tasks.

Here is an example of launching a 2D grid of models and tasks:

```
python -m rslp.main olmoearth_evals launch --models='[croma,olmoearth,panopticon,presto,satlaspretrain,terramind]' --tasks='[pastis_uni,pastis_ts]' --prefix 20251006 --image_name favyen/rslphelios16
```

Not all models support all modalities or multi-modality.

- AnySat: supports all tasks.
- Clay: supports Sentinel-1/Sentinel-2/Landsat, but only one modality at a time.
- Copernicus-FM: supports Sentinel-1 and Sentinel-2.
- CROMA: supports Sentinel-1 and Sentinel-2.
- DINOv3: supports Sentinel-2 and Landsat, but only one modality at a time.
- Galileo: supports all tasks.
- OlmoEarth: supports all tasks.
- Panopticon: supports all tasks.
- Presto: supports Sentinel-1 and Sentinel-2.
- Prithvi: supports Sentinel-2 and Landsat, although it's not designed for non-HLS Sentinel-2.
- SatlasPretrain: in this eval it only supports Sentinel-2.
- TerraMind: supports Sentinel-1 and Sentinel-2.

```
# Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,dinov3,galileo,olmoearth,panopticon,presto,prithvi,satlaspretrain,terramind]' --tasks='[pastis_uni,pastis_ts,marine_infra_uni,marine_infra_ts,wind_turbine_uni,wind_turbine_ts,solar_farm_uni,solar_farm_ts,sentinel2_vessel_length,sentinel2_vessel_type,sentinel2_vessels,lfmc_uni,lfmc_ts,mangrove_uni,mangrove_ts,forest_loss_driver]' --prefix 20251007b --image_name favyen/rslphelios16
# Sentinel-1 + Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,copernicusfm,croma,galileo,olmoearth,panopticon,presto,terramind]' --tasks='[pastis_mm,marine_infra_mm,wind_turbine_mm,solar_farm_mm,lfmc_mm,mangrove_mm]' --prefix 20251007b --image_name favyen/rslphelios16
# Sentinel-1 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,galileo,olmoearth,panopticon,presto,terramind]' --tasks='[sentinel1_vessels]' --prefix 20251007b --image_name favyen/rslphelios16
# Landsat tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,dinov3,galileo,olmoearth,panopticon,prithvi]' --tasks='[landsat_vessels]' --prefix 20251007b --image_name favyen/rslphelios17
```

Here are subset to run after dropping Presto and not doing the unitemporal version of
tasks that have a multitemporal option. And dropping PASTIS.

```

# Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,dinov3,galileo,olmoearth,panopticon,prithvi,satlaspretrain,terramind]' --tasks='[marine_infra_ts,wind_turbine_ts,solar_farm_ts,sentinel2_vessel_length,sentinel2_vessel_type,sentinel2_vessels,lfmc_ts,mangrove_ts,forest_loss_driver]' --prefix final --image_name favyen/rslphelios18 --project final_downstream_eval_train
# Sentinel-1 + Sentinel-2 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,copernicusfm,croma,galileo,olmoearth,panopticon,terramind]' --tasks='[marine_infra_mm,wind_turbine_mm,solar_farm_mm,lfmc_mm,mangrove_mm]' --prefix final --image_name favyen/rslphelios18 --project final_downstream_eval_train
# Sentinel-1 tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,copernicusfm,croma,galileo,olmoearth,panopticon,terramind]' --tasks='[sentinel1_vessels]' --prefix final --image_name favyen/rslphelios18 --project final_downstream_eval_train
# Landsat tasks.
python -m rslp.main olmoearth_evals launch --models='[anysat,clay,dinov3,galileo,olmoearth,panopticon,prithvi]' --tasks='[landsat_vessels]' --prefix final --image_name favyen/rslphelios18 --project final_downstream_eval_train
```
