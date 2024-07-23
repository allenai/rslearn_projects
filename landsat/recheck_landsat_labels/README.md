After Landsat model was put into Skylight integration in early July 2024, even after
raising confidence threshold from 0.5 to 0.7 there are still many false positives.

Joe did a quick check of 11 small vessels that I found, and he said 5 of them were
incorrect. So that makes it seem that the training data quality is a big issue.

The goal now is to validate the vessel labels in the training data.

Phase 1 - have Joe go through up to 1000 labels and annotate if they are correct or
not.

Phase 2 - then we can train a classification model on the new correct/incorrect
annotations, and try to see which other labels are most likely to be incorrect and
validate them too.


Phase 1
-------

- `phase1_get_1000.py`: this was used to acquire the 1000 labels for Joe to annotate.
  It produces a CSV file but we ended up using the webserver below instead of the CSV.
- `phase1_server.py`: then use this to create webserver for the annotation.
- `phase1_assign_split.py`: randomly assign split to only the examples that were
  actually labeled.

To train the model:

    gsutil -m rsync -r gs://satlas-explorer-data/rslearn_labels/landsat/2024-07-18-recheck-landsat-labels/dataset_v1/live/ /path/to/store/rslearn_landsat_data/
    python -m rslearn.main model fit --config ~/rslearn_projects/landsat/recheck_landsat_labels/phase1_config.yaml --data.init_args.root_dir /path/to/store/rslearn_landsat_data/

2024-07-23: The classification model doesn't perform so well and seems to be ineffective at
prioritizing which parts of the dataset should be explored, so probably we will combine
the 1000 annotations that Joe labeled with annotations over model predictions and then
retrain the classification model, but directly deploy it (to post-process the object
detections) rather than use it to fix the training data.
