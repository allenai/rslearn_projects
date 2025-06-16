# Global Mangrove Watch

For this project, there're two tasks: 1. Mangrove classification, i.e. predict each point as `Mangrove`, `Water`, or `Other`. An initial mask is applied to ensure training uses only points within mangrove regions, and all the training samples are from 2020, 2. Mangrove change detection (TODO: add more details).

## Task 1: Mangrove Classification

- Mangrove ground-truth data (about 5M): `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv`
- Mangrove reference data: `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_reference_points.csv`

We haven't used the reference data yet, but they're separate points validated by experts and also hold out for validation.

### 1. Create Windows

We sampled 100K points from the original 5M points. Run the following command to create windows:
```
python rslp/mangrove/create_windows_for_classification.py --csv_path /weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250616 --group_name sample_100K --window_size 32
```

### 2. Prepare/Materialize Windows

Run the following commands to prepare and materialize windows:
```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250616 --group sample_100K --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

python rslp/scripts/beaker_launcher.py --project mangrove_classification --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250616 --group sample_100K --image_name favyen/rslp --clusters ai2/saturn-cirrascale --num_jobs 10
```

## Task 2: Mangrove Loss Detection
