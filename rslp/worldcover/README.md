# WorldCover

## Dataset Location

The WorldCover dataset is saved on WEKA:
`/weka/dfive-default/rslearn-eai/datasets/worldcover`

## Reference Data

WorldCover 16.5M labels CSV file:
`/weka/dfive-default/rslearn-eai/artifacts/WorldCover/final_reference_data.csv`

The CSV file contains 165K clusters, each is 10x10 pixels at 10 meters resolution.

## Create Windows

To create windows from the CSV file:

```bash
python create_windows.py \
  --csv_path /weka/dfive-default/rslearn-eai/artifacts/WorldCover/final_reference_data.csv \
  --ds_path /weka/dfive-default/rslearn-eai/datasets/worldcover \
  --group_name 20260109 \
  --window_size 53
```
