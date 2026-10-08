# Landsat vessel detection

The pipeline (`predict_pipeline.py`, served by `api_main.py`) runs a detector over a
Landsat scene, a classifier on each detection to remove false positives, and an attribute
model on the rest. Run everything from the repo root. Model card and training/evaluation
docs are in `ai2_docs/landsat_vessels/`.

## Classifier configs

In `data/landsat_vessels/`:

- `config_classifier_20260908.yaml`: the v1.0.0 model, currently in production (run
  `olmoearth_base_layerdecay_20260908d`), at detector 0.7 / classifier 0.99.
- `config_classifier_20260928.yaml`: the same recipe retrained with the Skylight feedback
  groups (`feedback_20260911`, `feedback_20260928`) added.
- `config_classifier.yaml`: the legacy Swin classifier, for reference.

## Sub-modules

- `annotations/`: the annotation app for labelling detector candidates, and the record
  of annotation round 1. `scripts/` prepares its input.
- `feedback/`: the loop that turns Skylight false-positive feedback into training data,
  retrains, and publishes.
- `evaluation/get_metrics.py`: detector precision/recall against ground truth.

The classifier dataset
(`/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624`)
has one group per label source: `selected_copy`, `phase2a_completed`, `phase3a_selected`
(the original annotations), `feedback_20260325` (round-0 feedback, used for validation),
`round1_20260803` (from `annotations/`), and `feedback_<date>` (from `feedback/`).

## v1.0.0 training results (2026-09-08)

Trained on `selected_copy` + `phase2a_completed` + `round1_20260803` (3,487 windows) and
validated on `feedback_20260325` (441 windows). Metrics are for the `correct` class at the
best `val_correct_f1` epoch; rows below the first are ablations.

| Recipe | Accuracy | Recall | Precision | F1 |
|--------|----------|--------|-----------|----|
| **crop 32 / patch 2 (deployed)** | **0.918** | 0.905 | 0.870 | **0.887** |
| same, frozen encoder | 0.899 | 0.872 | 0.860 | 0.866 |
| crop 16 / patch 1 | 0.899 | 0.899 | 0.821 | 0.858 |
| crop 64 / patch 4, no `phase2a_completed` | 0.883 | 0.932 | 0.742 | 0.826 |
| crop 64 / patch 4 | 0.877 | 0.892 | 0.767 | 0.825 |
| crop 64 / patch 4, no `round1_20260803` | 0.848 | 0.966 | 0.647 | 0.775 |

## Scenario checks (2026-09-08)

All 7 scenes pass at detector 0.9 / classifier 0.9 (scene list and expected ranges in
`ai2_docs/landsat_vessels/train_eval.md`):

| Scene | Description | Expected | Detections |
|-------|-------------|----------|------------|
| LC09_L1GT_129107 | Mostly ice | [0, 10] | 0 |
| LC09_L1TP_001090 | Mostly whitecaps | [0, 10] | 0 |
| LC09_L1TP_193021 | Some vessels | [20, 50] | 40 |
| LC09_L1TP_170084 | Some vessels | [20, 50] | 39 |
| LC09_L1TP_177081 | Mostly whitecaps | [0, 10] | 3 |
| LC09_L1TP_010012 | Islands + ice | [0, 10] | 2 |
| LC09_L1TP_193030 | Some vessels | [20, 100] | 85 |

## Finetune Helios for Landsat vessel detection

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
