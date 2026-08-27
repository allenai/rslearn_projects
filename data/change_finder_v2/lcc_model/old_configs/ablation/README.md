# LCC temporal ablations

Two ablations driven off a single shared "ablation dataset":

- **Ablation A — temporal input decisions:** train + evaluate 5 model variants that
  differ in how many quarterly images they consume, all at a fixed frequent block
  (option index 2 = first-observable + 2 months).
- **Ablation B — time after change:** evaluate the existing **production** model on the
  same dataset, sweeping the frequent block offset from +0 to +7 months after the
  first-observable date.

Both read AUROC from the `BalancedBinaryMetric` logged as `val_binary/auroc` /
`test_binary/auroc`.

## Phase 0 — build the ablation dataset (run once)

The ablation dataset materializes 8 fixed frequent options (first-observable + `i`·30
days, `i = 0..7`) plus the quarterly stack, so it can serve both ablations.

```bash
source ~/local_ai2_env/bin/activate
cd /home/favyen/ai2-unison-data/rslearn_projects

python -m rslp.change_finder_v2.lcc_model.prepare_ablation \
    --v2-json-paths <path/to/v2_annotations.json> [<more.json> ...] \
    --ds-path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_ablation_dataset_20260621/ \
    --workers 32

# then materialize windows with rslearn as usual, e.g.
rslearn dataset prepare --root <ABLATION_DS_PATH> ...
rslearn dataset materialize --root <ABLATION_DS_PATH> ...
```

The dataset path is baked into the configs:
`/weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_ablation_dataset_20260621/`
(used for both `data.path` and `annotations_path`).

## Ablation A — temporal input decisions

Each variant has its own config. Train, then test the best checkpoint and read
`test_binary/auroc`.

| Config | quarterly | frequent | quarterly anchor | passes | num_timestamps |
|---|---|---|---|---|---|
| `ablation_a_bitemporal.yaml` | 1 | 1 | one_year_before | 1 | 2 |
| `ablation_a_4q.yaml` | 4 | 4 | recent | 1 | 8 |
| `ablation_a_8q.yaml` | 8 | 4 | recent | 1 | 12 |
| `ablation_a_12q.yaml` | 12 | 4 | recent | 2 (pass1=8) | 16 |
| `ablation_a_16q_default.yaml` | 16 | 4 | recent | 2 (pass1=10) | 20 |

All variants are fixed at frequent option index 2. The `16q_default` variant matches the
production architecture.

```bash
for cfg in ablation_a_bitemporal ablation_a_4q ablation_a_8q ablation_a_12q ablation_a_16q_default; do
    rslearn model fit  --config data/change_finder_v2/lcc_model/ablation/$cfg.yaml
    rslearn model test --config data/change_finder_v2/lcc_model/ablation/$cfg.yaml --ckpt_path best
done
```

Checkpoints land under project `2026_06_21_lcc_temporal_ablation`, run `ablation_<variant>`.

## Ablation B — time after change

Reuses the **existing production model** (no training). Evaluate it on the ablation
dataset val split at each frequent offset by sweeping `LCC_OPTION_INDEX` (0–7); the
config interpolates it into the sampler's `option_index`. Model management is disabled
in this config, so `--ckpt_path` points directly at the production checkpoint.

Production checkpoint: project `2026_06_16_lcc`, run `olmoearth_base_ps4_01`, under
`${RSLP_PREFIX}/projects`.

```bash
PROD_CKPT=<path to production checkpoint>

for i in 0 1 2 3 4 5 6 7; do
    LCC_OPTION_INDEX=$i rslearn model test \
        --config data/change_finder_v2/lcc_model/ablation/ablation_b_time_after_change.yaml \
        --ckpt_path "$PROD_CKPT"
done
```

Each run reports `test_binary/auroc` for that offset; `i` months after first-observable
corresponds to `i`·30 days.
