This code is for applying Sentinel-1 vessel detection pipeline on all scenes from April
2026 for Global Fishing Watch to compare the quality against their own pipeline.

After finding scene IDs:

```bash
python -m rslp.main sentinel1_vessels write_entries --queue_name favyen/sentinel1-vessels-predict --json_fname /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/scene_ids_descending.json --json_out_dir /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/json_outputs/ --geojson_out_dir /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/geojson_outputs/ --crop_out_dir /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/crop_outputs/
python -m rslp.main sentinel1_vessels write_entries --queue_name favyen/sentinel1-vessels-predict --json_fname /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/scene_ids_ascending.json --json_out_dir /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/json_outputs/ --geojson_out_dir /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/geojson_outputs/ --crop_out_dir /weka/dfive-default/rslearn-eai/projects/2026_07_16_gfw_sentinel1/crop_outputs/
python -m rslp.main common launch --image_name favyen/rslpomp20260716a --queue_name favyen/sentinel1-vessels-predict --num_workers 32 --cluster=[ai2/jupiter,ai2/ceres] --gpus 1 --shared_memory 256GiB --priority urgent --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' --extra_env_secrets '{"AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID", "COPERNICUS_USERNAME": "COPERNICUS_USERNAME", "COPERNICUS_PASSWORD": "COPERNICUS_PASSWORD"}'
```

olmoearth_pretrain is not necessary but in this case we use olmoearth_pretrain.Dockerfile
built with the instructions at `rslp/olmoearth_pretrain/README.md`.

Some scenes fail, e.g. due to no historical overlapping scene matches, and in that case
entire batch might fail; so after the first round, run it again with batch size 1 to
make sure as many scenes are processed as possible.
