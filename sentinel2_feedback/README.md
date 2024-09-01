# Sentinel2 Feedback

This project trains a model to determine if a given Sentinel-2 chip is valid or not.

## Setup 

#### Dependencies

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r rslearn_projects/requirements.txt
```

#### Auth

The `retrieve_dataset.py` script expects `--token`, which can be [accessed like this](https://api-int.skylight.earth/docs/#introduction-item-0).

## Source data 

This model is trained on a dataset of Sentinel-2 chips that have been labeled as GOOD or BAD.
The feedback.csv file is an export from the Skylight In-App Feedback tool, filtered for
Sentinel-2 events. Each row in the feedback file has an event_id, a label, and a link to the
event in the Skylight app.

A good way to generate a set of events to be labeled is to use the EAI [sample-events script](https://github.com/VulcanSkylight/eai/blob/master/ais/data/sample_events/sample-events.py#L1-L1) ([readme](https://github.com/VulcanSkylight/eai/blob/master/ais/data/sample_events/README.md#L1-L1)).

## Dataset Pre-processing

The `retrieve_dataset.py` script fetches event metadata from the Skylight API to identify the chip URL, and downloads the chip locally.
It outputs a csv file with the event_id, label, and local path to the chip, which is input into `create_rslearn_data.py`.

```
rslearn_projects/sentinel2_feedback $> 
python retrieve_dataset.py --token $token --feedback_csv feedback.sample.csv --chips_dir chips --output_csv dataset.sample.csv
```

The `create_rslearn_data.py` script creates an rslearn dataset from the chips and labels.

```
python create_rslearn_data.py --dataset_csv dataset.sample.csv --out_dir rslearn_dataset
```
