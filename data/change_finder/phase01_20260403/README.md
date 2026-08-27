# Change Finder Phase 01 Experiments (2026-04-03)

16 single-GPU experiments exploring architecture, encoder size, head depth,
training recipe, and data augmentation for the change finder self-supervised
change detection task.

All experiments use `project_name: change_finder_phase01_20260403`.

## Group A: Architecture

| Config | What changes | Key diff from baseline |
|--------|-------------|----------------------|
| `baseline.yaml` | Reference point | — |
| `mean_pool.yaml` | Mean spatial pooling instead of max | `pool_mode: mean` |
| `attn_pool.yaml` | Learned attention pooling (linear + softmax over spatial positions) | `pool_mode: attn` |
| `feat_diff.yaml` | Feature difference-product combination instead of concatenation | `combine_mode: diff` |
| `margin_loss.yaml` | Margin ranking loss instead of cross-entropy; shared scorer head replaces classifier | `loss_mode: margin` |

## Group B: Encoder size

| Config | What changes | Key diff from baseline |
|--------|-------------|----------------------|
| `nano_encoder.yaml` | OlmoEarth-v1-Nano (128-d) | `model_id: OLMOEARTH_V1_NANO`, `in_channels: 128`, `fc_channels: 128` |
| `tiny_encoder.yaml` | OlmoEarth-v1-Tiny (192-d) | `model_id: OLMOEARTH_V1_TINY`, `in_channels: 192`, `fc_channels: 192` |

## Group C: Head architecture

| Config | What changes | Key diff from baseline |
|--------|-------------|----------------------|
| `shallow_head.yaml` | No hidden FC layers (linear projection only) | `num_fc_layers: 0` |
| `deep_wide_head.yaml` | 3 FC layers with 1024 channels | `num_fc_layers: 3`, `fc_channels: 1024` |

## Group D: Training recipe

| Config | What changes | Key diff from baseline |
|--------|-------------|----------------------|
| `frozen_encoder.yaml` | Encoder stays frozen for entire training | `FreezeUnfreeze` callback removed |
| `early_unfreeze.yaml` | Unfreeze encoder at epoch 3 (baseline: 10) | `unfreeze_at_epoch: 3` |
| `cosine_lr.yaml` | Cosine annealing schedule instead of plateau | `CosineAnnealingScheduler(T_max=100, eta_min=1e-6)` |
| `higher_lr.yaml` | 5x higher learning rate | `lr: 5e-4` |

## Group E: Data / augmentation

| Config | What changes | Key diff from baseline |
|--------|-------------|----------------------|
| `crop128.yaml` | Full 128px tile (no random crop) | `crop_size: 128` |
| `flip_aug.yaml` | Random H/V flips on train inputs | `Flip` transform added to train |
| `time_drop.yaml` | Randomly drop 20% of timesteps during training | `RandomTimeDropping` transform added to train |

## Shared defaults

Unless overridden above, all experiments use:

- Encoder: OlmoEarth-v1-Base (768-d), frozen for 10 epochs then unfrozen at 10x lower LR
- Head: 2 FC layers (768\*2 -> 512 -> 2)
- Optimizer: AdamW, lr=1e-4
- Scheduler: ReduceLROnPlateau (factor=0.2, patience=2, cooldown=10)
- Data: batch\_size=8, crop\_size=64, num\_samples=16384/epoch
- Trainer: max\_epochs=100, checkpoint on val\_loss
