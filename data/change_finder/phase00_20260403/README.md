# Change Finder Phase 0: Gap and Symmetry Experiments (2026-04-03)

10 single-GPU experiments investigating why the original change finder model
reaches 90% accuracy. The hypothesis is that the model exploits temporal order
cues rather than learning true similarity.

## Key changes from the original config

1. **Symmetric triplet sampling**: For every triplet `(close, query, far)` with
   label 0, the reversed `(far, query, close)` with label 1 is also included.
   This prevents the model from learning anchor position bias.

2. **Gap parameter**: Controls the year difference between query and the far
   anchor. `gap=6` is equivalent to the original (y0 vs y6), while `gap=2`
   uses many intermediate year combinations. Smaller gaps have more training
   diversity but less expected change signal.

3. **All 7 years loaded**: Configs load sentinel2\_y0 through sentinel2\_y6
   (instead of only y0, y1, y5, y6).

Uses `ChangeFinderGapTransform` (in `rslp/change_finder/train.py`) instead of
the original `ChangeFinderTransform`.

## Experiments: 5 gaps x 2 losses

| Config | gap | loss | Triplet count (symmetric) |
|--------|-----|------|--------------------------|
| `gap2_ce` | 2 | cross-entropy | 20 |
| `gap2_margin` | 2 | margin ranking | 20 |
| `gap3_ce` | 3 | cross-entropy | 16 |
| `gap3_margin` | 3 | margin ranking | 16 |
| `gap4_ce` | 4 | cross-entropy | 12 |
| `gap4_margin` | 4 | margin ranking | 12 |
| `gap5_ce` | 5 | cross-entropy | 8 |
| `gap5_margin` | 5 | margin ranking | 8 |
| `gap6_ce` | 6 | cross-entropy | 4 |
| `gap6_margin` | 6 | margin ranking | 4 |

Triplet count = number of distinct (anchor1, query, anchor2, label) options
per window when all 7 years are present.

## Shared settings

All experiments use the baseline architecture and training recipe:

- Encoder: OlmoEarth-v1-Base (768-d), frozen 10 epochs then unfrozen at 10x lower LR
- Head: 2 FC layers (768\*2 -> 512 -> 2)
- Optimizer: AdamW, lr=1e-4
- Scheduler: ReduceLROnPlateau (factor=0.2, patience=2, cooldown=10)
- Data: batch\_size=8, crop\_size=64, num\_samples=16384/epoch
- Trainer: max\_epochs=100, checkpoint on val\_loss

## What to look for

- If accuracy drops significantly with symmetric sampling (especially at gap=6),
  the original model was exploiting anchor order.
- If accuracy is much higher at large gaps than small gaps, the model is
  detecting real long-term change rather than subtle processing artifacts.
- If small gaps still achieve decent accuracy, processing artifacts may be the
  dominant signal.
- Margin vs CE comparison shows whether the ranking formulation is a better fit
  for this similarity task.
