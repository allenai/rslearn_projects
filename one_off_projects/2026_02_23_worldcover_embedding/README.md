These are model configuration files to qualitatively evaluate the effectiveness of
embeddings at capturing fine-grained details. It focuses on training models on the
WorldCover data (for Ai2-internal use, it is available on WEKA at
`/weka/dfive-default/rslearn-eai/datasets/worldcover/`).

- `config_worldcover_ps1.yaml`: patch_size=1, linear layer (768 -> 13).
- `config_worldcover_ps2.yaml`: patch_size=2, bilinear interpolation + linear layer (768 -> 13).
- `config_worldcover_ps4.yaml`: patch_size=4, bilinear interpolation + linear layer (768 -> 13).
- `config_worldcover_ps4_bicubic.yaml`: patch_size=4, bicubic interpolatoin + linear layer (768 -> 13).
- `config_worldcover_ps4_twolayer.yaml`: patch_size=4, linear layer (768 -> 768) + upsample features + linear layer (768 -> 13).
- `config_worldcover_ps4_fourlayer.yaml`: patch_size=4, like twolayer but four layers total (two before upsampling, two after).
- `config_worldcover_ps4_reshape.yaml`: patch_size=4, linear layer (768 -> 13x4x4), reshape to get 10 m/pixel output.
- `config_worldcover_ps4_finetune.yaml`: patch_size=4 with upsampling decoder (UNetDecoder), full fine-tuning.

Train the model:

```
python -m rslp.rslearn_main model fit --config one_off_projects/2026_02_23_worldcover_embedding/config_worldcover_ps4.yaml
```

Or train it via Beaker job:

```
python -m rslp.main common beaker_train --image_name favyen/rslpomp20260216a --cluster=[ai2/jupiter,ai2/ceres] --config_path one_off_projects/2026_02_23_worldcover_embedding/config_worldcover_ps4.yaml --weka_mounts+='{"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}'
```

Create dataset with one window in Seattle for qualitative evaluation, and get predictions:

```
export DATASET_PATH=./dataset
mkdir $DATASET_PATH
cp one_off_projects/2026_02_23_worldcover_embedding/config.json $DATASET_PATH/config.json
rslearn dataset add_windows --root $DATASET_PATH --group default --utm --resolution 10 --window_size 2048 --src_crs EPSG:4326 --box=-122.255,47.589,-122.255,47.589 --start 2025-01-01T00:00:00+00:00 --end 2026-01-01T00:00:00+00:00 --name seattle
python -m rslp.rslearn_main model predict --config one_off_projects/2026_02_23_worldcover_embedding/config_worldcover_ps4.yaml --data.init_args.path=$DATASET_PATH --load_best=true
```