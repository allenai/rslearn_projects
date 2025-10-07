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
- OlmoEarth: supports all tasks.
- Panopticon: supports all tasks.
- Presto: supports Sentinel-1 and Sentinel-2.
- Prithvi: supports Sentinel-2 and Landsat, although it's not designed for non-HLS Sentinel-2.
- SatlasPretrain: in this eval it only supports Sentinel-2.
- TerraMind: supports Sentinel-1 and Sentinel-2.

```
# Sentinel-2 tasks.
python data/helios_v3/run.py --models='[anysat,clay,copernicusfm,croma,olmoearth,panopticon,presto,prithvi,satlaspretrain,terramind]' --tasks='[pastis_uni,pastis_ts,marine_infra_uni,marine_infra_ts,wind_turbine_uni,wind_turbine_ts,solar_farm_uni,solar_farm_ts,sentinel2_vessel_length,sentinel2_vessel_type,sentinel2_vessels]' --prefix 20251007b --image_name favyen/rslphelios16
# Sentinel-1 + Sentinel-2 tasks.
python data/helios_v3/run.py --models='[anysat,copernicusfm,croma,olmoearth,panopticon,presto,terramind]' --tasks='[pastis_mm,marine_infra_mm,wind_turbine_mm,solar_farm_mm]' --prefix 20251007b --image_name favyen/rslphelios16
# Sentinel-1 tasks.
python data/helios_v3/run.py --models='[anysat,clay,copernicusfm,croma,olmoearth,panopticon,presto,terramind]' --tasks='[sentinel1_vessels]' --prefix 20251007b --image_name favyen/rslphelios16
```
