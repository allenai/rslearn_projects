Forest Loss Driver Classification
---------------------------------

This project is a collaboration with Amazon Conservation Association to develop a model
that can classify what caused a forest loss event detected by the GLAD Sentinel-2
system (e.g. mining, agriculture, hurricane/wind, river shift, etc.).


Dataset Setup
-------------

There are two sources of labels:

- Initial labels that Nadia (the GIS specialist at ACA) provided.
- Labels we tried to annotate, which were based around GLAD forest loss events.

The dataset setup does not need to be repeated, but here are the steps:

1. Get labels from `gs://satlas-explorer-data/rslearn_labels/amazon_conservation/nadia/`.
2. Run `convert_from_nadia.py` to convert them to windows in a target rslearn dataset.
3. Oh and use `config_closetime2.json` for that dataset which includes Sentinel-2 and
   Planet Labs imagery.
4. Get files needed by `create_unlabeled_dataset.py` from
   https://console.cloud.google.com/storage/browser/earthenginepartners-hansen/S2alert
5. Run `create_unlabeled_dataset.py` to randomly sample GLAD forest loss alerts for our
   own annotation.

Anyway you can get the materialized dataset here:

    gs://satlas-explorer-data/rslearn_labels/rslearn_amazon_conservation_closetime.tar

The useful groups are:

- nadia2, nadia3: labels from Nadia.
- peru3_flagged_in_peru: labels we weren't sure about but Nadia has corrected them.
- peru_interesting: old labels that Favyen went through and confirmed (mostly for road
  and landslide categories which are more clear).
- brazil_interesting: same as above but in Brazil.
- peru2: this is only used for evaluation. There may be "labels" here but they were
  just derived from model output so that they could be viewed in the same annotation
  tool.

Each window has some layers:

- best_pre_X: images from before the forest loss event. `best_times.json` indicates the
  timestamp of these images (they don't appear in `items.json`).
- best_post_X: same but for after the forest loss event.
- label: the label. It has `new_label` property which indicates the label. `old_label`
  is used for various things like showing model output.
