# LCC exploration sweeps (2026-07-01)

Four independent sweeps for the change_finder_v2 LCC single-pass OlmoEarth model,
all logging to W&B project `2026_07_01b_lcc_exploration`. Focus metric:
`val_binary/auroc`. Each folder has a `base.yaml`; runs launch by merging
`base.yaml` + a per-run override via two `--config` flags. Only `optimizer/` has a
`generate_configs.py` (generated overrides are NOT committed); the other three use
hand-written per-run YAMLs.

All configs derive from
[`config_pass20_v1_2.yaml`](../config_pass20_v1_2.yaml) except `per_image/`,
which uses the new `PerImageChangeModel`. Dataset:
`/weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_dataset_20260701/`.

## Sweeps

### optimizer/ (33 runs)
LR / LLRD / freeze-schedule sweep.
- Primary (27) = `LayerDecayAdamW` + `SimpleFreeze`:
  - `unfreeze{none,5}` x `layer_decay_rate{1.0,0.9,0.8,0.65}` x `lr{5e-5,1e-4,3e-4}` = 24
  - `frozen` x `lr{5e-5,1e-4,3e-4}` = 3 (fixed `decay=1.0`; the encoder never trains, so decay is a no-op). `frozen` = `SimpleFreeze(unfreeze_at_epoch=10000)` = linear-probe.
  - `layer_decay_rate: 1.0` = uniform LR for the whole model.
- Factor family (6) = uniform `AdamW` + `FreezeUnfreeze`: `lr{1e-4,3e-4}` x `unfreeze_lr_factor{1,10,100}`, `unfreeze_at_epoch=5`.
- Regenerate: `python optimizer/generate_configs.py`.

### augmentation/ (3 runs)
Extra training augmentations, on the standard LLRD finetune.
- `aug_tokendrop`: encoder `token_drop_rate: 0.3`.
- `aug_gaussian`: `GaussianNoise(std=0.1)` on `sentinel2_l2a` (train only).
- `aug_all`: both.
- Note: `timestep_drop`/`RandomTimeDropping` are intentionally NOT used (they break the per-pixel start/end timestep-index targets). Spatial crop is not needed (128x128 windows are already random-cropped to 64 by `crop_size: 64`).

### model_size/ (4 runs)
OlmoEarth v1.2 size sweep with uniform `AdamW (lr 1e-4)`: `size_{nano,tiny,small,base}` (embedding 128/192/384/768).

### per_image/ (2 runs)
New `PerImageChangeModel`: OlmoEarth applied per image (no cross-time attention in the encoder), then a temporal positional embedding + 1 temporal self-attention layer over T, then a temporal decoder (`perimage_attn` = cross-attention pool, `perimage_rnn` = GRU) + the singlepass conv/timestamp heads. Uniform `AdamW`, `batch_size: 2` (B*T encoder passes are heavy).

## Launch

Clusters `ai2/jupiter` + `ai2/ceres`, `--priority urgent`, image
`favyen/rslpomp20260630b` (new model code ships via `upload_code`; no image
rebuild). `beaker_train` derives project/run from the FIRST config, so pass
`--project_id`/`--experiment_id` explicitly (experiment_id = the variant name).

```bash
ROOT=data/change_finder_v2/lcc_model/20260701_exploration

launch() {  # $1 = folder, remaining = per-run config filenames
    local folder="$1"; shift
    local D="$ROOT/$folder"
    for name in "$@"; do
        python -m rslp.main common beaker_train \
            --config_paths+="$D/base.yaml" --config_paths+="$D/$name.yaml" \
            --project_id 2026_07_01b_lcc_exploration --experiment_id "$name" \
            --cluster+=ai2/jupiter --cluster+=ai2/ceres \
            --image_name favyen/rslpomp20260701a --priority urgent --gpus 1 \
            '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}'
    done
}

# optimizer (generate first, then launch all)
python "$ROOT/optimizer/generate_configs.py"
launch optimizer $(cd "$ROOT/optimizer" && ls opt_*.yaml optf_*.yaml | sed 's/.yaml$//')

# augmentation
launch augmentation aug_tokendrop aug_gaussian aug_all

# model_size
launch model_size size_nano size_tiny size_small size_base

# per_image
launch per_image perimage_attn perimage_rnn
```

Total: optimizer 33 + augmentation 3 + model_size 4 + per_image 2 = 42 runs.

## Notes
- Requires `RSLP_PREFIX` set locally at launch (for `management_dir: ${RSLP_PREFIX}/projects`) and the `dfive-default` Weka mount.
- Commit + push before launching; Beaker pulls the code from the repo.
