# SimpleConv LCC sweep (2026-06-30)

A sweep of small, from-scratch conv models for the change_finder_v2 LCC task, as a
lightweight alternative to the heavy OlmoEarth encoder. The hypothesis: for binary
change + change-timing, a simple spectral-temporal model may be competitive; the
pretrained encoder is expected to keep an edge on the 13-class `src`/`dst` land
cover typing.

## Model

[`SimpleConvChangeModel`](../../../../rslp/change_finder_v2/lcc_model/model_simpleconv.py)
runs directly on the normalized 20-timestep `sentinel2_l2a` stack `(B, 12, 20, 64, 64)`
built by `SinglePassSampler` (no OlmoEarth pass):

- 2D or 3D conv backbone at full temporal/spatial resolution (stem + residual
  blocks, GroupNorm+ReLU). 2D shares weights across timesteps; 3D mixes over time.
- Learned temporal positional embedding, then an optional temporal self-attention
  stack over the T tokens (per pixel).
- Per-task per-pixel temporal head (one each for binary/src/dst):
  - `attn`: a learned query attention-pools over time, or
  - `rnn`: a bidirectional GRU over time, mean-pooled.
- Start/end change-boundary heads: per-timestep linear logits over T (same as the
  single-pass model, no upsampling since we are already at full resolution).

Everything else (sampler, `SinglePassMultiTask`, metrics, and the five losses) is
reused unchanged from
[`model_singlepass.py`](../../../../rslp/change_finder_v2/lcc_model/model_singlepass.py).

Optimizer is plain `AdamW` (lr 1e-3), monitored on `val_binary/auroc`.

## Sweep (45 runs)

- Base grid (36): `conv_type{2d,3d} x num_conv_layers{4,8,12} x embedding_dim{64,128,256} x head_type{attn,rnn}`.
- Self-attention variants (9): `conv_type=3d`, `head_type=attn`, `num_temporal_selfattn_layers=2`, over the same 9 layer/dim combos.

Run names: `simpleconv_{conv}_{layers}l_{dim}d_{head}` and, for the self-attn
variants, `simpleconv_3d_{layers}l_{dim}d_attn_sa2`.

## Config layout

- [`base.yaml`](base.yaml): the full config (data, task, optimizer, trainer).
- `generate_configs.py`: emits the 45 per-variant override YAMLs into this folder.
  Each override fully specifies the `model.init_args.model` block plus `run_name`,
  and sets `batch_size: 4` for the heavy 3D/256/`>=8`-layer cells (everything else
  uses `batch_size: 8`).

The generated override files are NOT committed. Regenerate them with:

```bash
python data/change_finder_v2/lcc_model/20260630_simpleconv/generate_configs.py
```

Each run merges two configs (later overrides earlier):

```bash
rslearn model fit \
  --config data/change_finder_v2/lcc_model/20260630_simpleconv/base.yaml \
  --config data/change_finder_v2/lcc_model/20260630_simpleconv/simpleconv_3d_8l_128d_attn.yaml
```

## Launch all 45 on Beaker

Clusters `ai2/jupiter` + `ai2/ceres`, `urgent` priority, image `favyen/rslpomp20260630b`.

```bash
D=data/change_finder_v2/lcc_model/20260630_simpleconv
python "$D/generate_configs.py"

for cfg in "$D"/simpleconv_*.yaml; do
    name=$(basename "$cfg" .yaml)
    python -m rslp.main common beaker_train \
        --config_paths+="$D/base.yaml" \
        --config_paths+="$cfg" \
        --project_id 2026_06_30b_lcc_simpleconv \
        --experiment_id "$name" \
        --cluster+=ai2/jupiter \
        --cluster+=ai2/ceres \
        --image_name favyen/rslpomp20260630b \
        --priority urgent \
        --gpus 1 \
        '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}'
done
```

`beaker_train` overrides `project_name`/`run_name` with `--project_id`/`--experiment_id`,
so W&B runs land under project `2026_06_30_lcc_simpleconv` with the variant name.

### Single run

```bash
D=data/change_finder_v2/lcc_model/20260630_simpleconv
python -m rslp.main common beaker_train \
    --config_paths+="$D/base.yaml" \
    --config_paths+="$D/simpleconv_3d_8l_128d_attn_sa2.yaml" \
    --project_id 2026_06_30_lcc_simpleconv \
    --experiment_id simpleconv_3d_8l_128d_attn_sa2 \
    --cluster+=ai2/jupiter --cluster+=ai2/ceres \
    --image_name favyen/rslpomp20260630b \
    --priority urgent --gpus 1 \
    '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}'
```

## Notes

- Requires `RSLP_PREFIX` set locally at launch (folded into the job for
  `management_dir: ${RSLP_PREFIX}/projects`) and the `dfive-default` Weka mount for
  the dataset + `lcc_annotations.json`.
- Dataset: `/weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_dataset_20260629/`.
- Remember to commit + push before launching; Beaker pulls the code from the repo.
