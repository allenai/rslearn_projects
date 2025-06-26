# Global Mangrove Watch

For this project, there're two tasks: 1. Mangrove classification, i.e. predict each point as `Mangrove`, `Water`, or `Other`. An initial mask is applied to ensure training uses only points within mangrove regions, and all the training samples are from 2020, 2. Mangrove change detection (TODO: add more details).

## Task 1: Mangrove Classification

- Mangrove ground-truth data (about 5M): `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv`
- Mangrove reference data: `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_reference_points.csv`

We haven't used the reference data yet, but they're separate points validated by experts and also hold out for validation.

### 1. Create Windows

We sampled 100K points from the original 5M points. Run the following command to create windows:
```
python rslp/mangrove/create_windows_for_classification.py --csv_path /weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_v4_points.csv --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626 --group_name sample_100K --window_size 32
```

### 2. Prepare/Materialize Windows

Run the following commands to prepare and materialize windows:
```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626 --group sample_100K --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python rslp/scripts/beaker_launcher.py --project mangrove_classification --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626 --group sample_100K --image_name favyen/rslp --clusters ai2/ceres-cirrascale --num_jobs 10
```

## Task 2: Mangrove Loss Detection

From 2021 to April 2025, there were 575K true alerts and 94K false alerts in total. Each alert is a point representing 20x20m area on the ground (i.e., a Sentinel2 pixel where the 10m bands were resampled to the 20m bands). The baseline method is based on a scoring system that applies thresholding to the NDVI time series, as described in [paper](https://doi.org/10.3390/rs15082050).

As a starting point, we can randomly select equal number of true and false alerts to form a training set, and then finetune Helios model to classify if the input time-series (Sentinel1 + Sentinel2) indicating a Mangrove loss or not.

| Year | True Positives | False Positives | Precision  |
|------|----------------|-----------------|------------|
| 2025 | 34,026         | 224             | 99.34%     |
| 2024 | 186,881        | 7,169           | 96.31%     |
| 2023 | 165,912        | 67,522          | 71.08%     |
| 2022 | 106,648        | 3,574           | 96.76%     |
| 2021 | 82,151         | 15,840          | 83.83%     |

Also, we split train/val by years (188K samples in total), using 2021-2023 to train, and 2024-2025 to validate. The key metric here will be the accuracy and F1 score for TP cases.

```
python /weka/dfive-default/yawenz/rslearn_projects/rslp/mangrove/create_windows_for_loss_detection.py --true_positives_csv_path /weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_2021_202504_true_positives.csv --false_positives_csv_path /weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_2021_202504_false_positives.csv --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/loss_detection/20250626 --window_size 32
```

```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/mangrove/loss_detection/20250626 --group sample_188K_temporal_split --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python rslp/scripts/beaker_launcher.py --project mangrove_loss_detection --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/loss_detection/20250626 --group sample_188K_temporal_split --image_name favyen/rslp --clusters ai2/titan-cirrascale --num_jobs 15
```
