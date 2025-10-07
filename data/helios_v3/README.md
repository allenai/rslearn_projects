These model and task configs are designed to work with rslp.olmoearth_evals, which
provides an adapter model to allow the inputs from all of the tasks to be consistent
with all the baseline models we want to evaluate.

Here is an example of launching a 2D grid of models and tasks:

```
python data/helios_v3/run.py --model croma olmoearth panopticon presto satlaspretrain terramind --task pastis_uni pastis_ts --prefix 20251006
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
python data/helios_v3/run.py --model anysat clay copernicusfm croma olmoearth panopticon presto prithvi satlaspretrain terramind --task pastis_uni pastis_ts marine_infra_uni marine_infra_ts wind_turbine_uni wind_turbine_ts solar_farm_uni solar_farm_ts sentinel2_vessel_length sentinel2_vessel_type sentinel2_vessels --prefix 20251007b
# Sentinel-1 + Sentinel-2 tasks.
python data/helios_v3/run.py --model anysat copernicusfm croma olmoearth panopticon presto terramind --task pastis_mm marine_infra_mm wind_turbine_mm solar_farm_mm --prefix 20251007b
# Sentinel-1 tasks.
python data/helios_v3/run.py --model anysat clay copernicusfm croma olmoearth panopticon presto terramind --task sentinel1_vessels --prefix 20251007b
```
