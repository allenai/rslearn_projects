Small Sentinel-1 and Landsat vessel type/length OlmoEarth eval tasks. This
mirrors the Sentinel-2 version in
`one_off_projects/2026_04_30_more_olmoearth_evals/convert_vessel_attribute.py`.

## Source datasets

Both are built by `rslp/vessel_attribute/create_windows.py`, so their `info`
vector layer already has the `type` and `length` properties:

- Sentinel-1: `/weka/dfive-default/rslearn-eai/datasets/sentinel1_vessel_attribute/dataset_v1/20260330/`
  (raster layer `sentinel1`, bands `vv`, `vh`)
- Landsat: `/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_attribute/dataset_v1/20260330/`
  (raster layer `landsat_allbands`, bands `B1`..`B11`)

## Conversion

`convert_vessel_attribute.py` keeps only windows with both `type` and `length`,
samples the union of 500 per vessel type and 500 per length bucket
([0,50), [50,75), …, [225,250), [250,+inf)), then splits into 50% train,
25% val, 25% test. Groups are discovered automatically from the source
`windows/` dir. All modality rasters are center-cropped from 128x128 to 64x64;
the `info` layer is copied as-is.

```bash
python one_off_projects/2026_07_01_s1_landsat_vessel/convert_vessel_attribute.py sentinel1
python one_off_projects/2026_07_01_s1_landsat_vessel/convert_vessel_attribute.py landsat
```

Outputs:
- `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/small_sentinel1_vessel_attribute/`
- `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/small_landsat_vessel_attribute/`

## Task configs

- `data/olmoearth_evals/tasks/small_sentinel1_vessel_type.yaml`
- `data/olmoearth_evals/tasks/small_sentinel1_vessel_length.yaml`
- `data/olmoearth_evals/tasks/small_landsat_vessel_type.yaml`
- `data/olmoearth_evals/tasks/small_landsat_vessel_length.yaml`

These are registered in `rslp/olmoearth_evals/launch.py` under the matching task
keys, e.g.:

```bash
python -m rslp.main olmoearth_evals launch ... --tasks small_sentinel1_vessel_type
```
