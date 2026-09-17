# Finetune Helios for Landsat vessel detection

- Detector: 161747 train, 23824 val

Helios
```
python -m rslp.main olmoearth_pretrain launch_finetune --olmoearth_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/finetune_detector.yaml --cluster+=ai2/saturn-cirrascale --project_name 2025_06_26_helios_finetuning --run_name v2_landsat_vessel_detection_helios_base_ps4_freeze_unfreeze --gpus 1
```

SwinB pretrained on ImageNet
```
python -m rslp.main common beaker_train --image_name favyen/rslphelios2 --config_paths+=data/helios/v2_landsat_vessels/finetune_detector_swinb.yaml --cluster+=ai2/saturn-cirrascale --project_id 2025_06_26_helios_finetuning --experiment_id v2_landsat_vessel_detection_swin_imagenet_ps4 '--weka_mounts+={"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}' --gpus 1
```

- Classifier: 1783 train, 535 val

```
python -m rslp.main olmoearth_pretrain launch_finetune --olmoearth_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_landsat_vessels/finetune_classifier.yaml --cluster+=ai2/ceres-cirrascale --project_name 2025_06_26_helios_finetuning --run_name v2_landsat_vessel_classification_helios_base_ps4_add_prob_threshold
```

## Classifier training results (2026-09-08)

Finetuned the OlmoEarth-base classifier on the val set (correct vs. incorrect). Metrics are for the `correct` (positive) class at each run's best `val_correct_f1` epoch. **Run d (crop 32 / patch 2) is the winner** and is used in the pipeline (`data/landsat_vessels/config_classifier_20260908d.yaml`).

Run d trains on `selected_copy` + `phase2a_completed` + `round1_20260803` — 3,487 labelled windows (1,525 correct / 1,962 incorrect) — and validates on `feedback_20260325` (441 windows: 149 correct / 292 incorrect). Runs b/c drop one training group each (see Recipe).

| Run | Recipe | Accuracy | Correct recall | Correct precision | Correct F1 |
|-----|--------|----------|----------------|-------------------|------------|
| **d** | crop 32 / patch 2 | **0.918** | 0.905 | 0.870 | **0.887** |
| f | d + frozen head-only | 0.899 | 0.872 | 0.860 | 0.866 |
| e | crop 16 / patch 1 | 0.899 | 0.899 | 0.821 | 0.858 |
| b | no `phase2a_completed` | 0.883 | 0.932 | 0.742 | 0.826 |
| a | baseline (crop 64 / patch 4) | 0.877 | 0.892 | 0.767 | 0.825 |
| c | no `round1_20260803` | 0.848 | 0.966 | 0.647 | 0.775 |

Notes:
- **Resolution/patch is the biggest lever:** crop 32 / patch 2 (d) beats the crop 64 / patch 4 baseline (a) by +0.062 F1; going finer (crop 16 / patch 1, e) regresses.
- **`round1_20260803` matters, `phase2a_completed` is roughly neutral:** dropping round1 (c) costs ~0.05 F1; dropping phase2a (b) is ~baseline.
- **Frozen head-only (f)** is only ~0.02 F1 below full finetuning while training far fewer params — a cheap, fast option.

### Classifier config files

- **`config_classifier_20260908d.yaml`** — current deployed classifier (OlmoEarth-base Run d), pointed to by `rslp/landsat_vessels/config.py` and downloaded by the `Dockerfile`. Deploy operating point: detector `score_threshold=0.7` + classifier `positive_class_threshold=0.99`.
- **`config_classifier.yaml`** — legacy classifier (Swin-based, `rslearn-landsat-recheck/phase123_20240919_01_copy`, threshold 0.85). Kept for reference only; no longer used by the pipeline.
- The dated `config_classifier_2026*.yaml` files are training experiments (see the table above); only Run d is deployed.

## Smoke test results (2026-09-08)

Full pipeline with the new classifier (`data/landsat_vessels/config_classifier_20260908d.yaml`, crop_size 32 / patch_size 2), evaluated at detector threshold 0.9 and classifier prob(correct) threshold 0.9. All 7 scenes pass. (Deployment uses det 0.7 / cls 0.99 — see the sweep below.)

| Scene | Description | Expected | Detections |
|-------|-------------|----------|------------|
| LC09_L1GT_129107 | Mostly ice | [0, 10] | 0 |
| LC09_L1TP_001090 | Mostly whitecaps | [0, 10] | 0 |
| LC09_L1TP_193021 | Some vessels | [20, 50] | 40 |
| LC09_L1TP_170084 | Some vessels | [20, 50] | 39 |
| LC09_L1TP_177081 | Mostly whitecaps | [0, 10] | 3 |
| LC09_L1TP_010012 | Islands + ice | [0, 10] | 2 |
| LC09_L1TP_193030 | Some vessels | [20, 100] | 85 |

Reproduce (writes per-scene overview + pan-sharpened detection crops to `--out_dir`):
```
python -m rslp.landsat_vessels.evaluation.visualize_smoke_predictions --classify_config data/landsat_vessels/config_classifier_20260908d.yaml --det_thr 0.9 --cls_thr 0.9 --out_dir /tmp/smoke_vis
```

## Threshold sweep (2026-09-08)

How the total detection count (summed over all 7 scenes) moves across detector threshold (rows) x classifier prob(correct) threshold (cols):

```
 det\cls  0.30  0.40  0.50  0.60  0.70  0.75  0.80  0.85  0.90  0.95
    0.50   236   235   231   230   227   226   223   223   222   217
    0.70   214   213   210   209   207   207   205   205   204   199
    0.90   174   173   172   172   171   171   170   170   169   165
    0.95   153   152   151   151   150   150   149   149   148   146
```

Takeaway for tuning: the **detector threshold is the main knob** (0.5->0.95 drops the total ~236->153), while the **classifier threshold barely matters** — Run d's probabilities cluster near 0/1, so it acts as a sharp on/off noise filter rather than a smooth dial.

**Deployed at det 0.7 / cls 0.99**: the low detector threshold surfaces confident vessels in the 0.7-0.9 band, and cls 0.99 keeps only high-confidence classifier calls as a sharp false-positive filter (safe precisely because the probabilities are near 0/1).

Reproduce (prints pass matrix + per-scene and total detection-count matrices; caches per-scene candidates in `--out_dir`):
```
python -m rslp.landsat_vessels.evaluation.smoke_test_sweep_2d --classify_config data/landsat_vessels/config_classifier_20260908d.yaml --det_floor 0.5 --out_dir /tmp/smoke_sweep_2d
```

## Recent-scene inference

Run the deployed pipeline over a sampled list of recent scenes and render a per-scene `candidates_grid.png` (green/red/orange/yellow by detector x classifier score) for visual QA:
```
python -m rslp.landsat_vessels.scripts.run_recent_inference --scene_csv <scenes.csv> --out_dir <out> --classify_config data/landsat_vessels/config_classifier_20260908d.yaml --det_thr 0.7 --cls_thr 0.99 --workers 16
```
