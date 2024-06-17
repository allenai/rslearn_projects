Forest Loss Driver Classification
---------------------------------

This project is a collaboration with Amazon Conservation Association to develop a model
that can classify what caused a forest loss event detected by the GLAD Sentinel-2
system (e.g. mining, agriculture, hurricane/wind, river shift, etc.).

The model currently inputs 6 Sentinel-2 images: 3 from before the forest loss and 3
after the forest loss. In general, it can input any public domain images before/after
the forest loss event.

The categories are:
- mining
- agriculture
- airstrip
- road
- logging
- burned
- landslide
- hurricane
- river
- none

The last category indicates that, although the GLAD system detected forest loss, we
don't think there's really any forest loss there.

There are four fine-grained agriculture categories that some labels use, but we
currently treat them all as generic "agriculture" since it was difficult to label it:
- agriculture-generic
- agriculture-small
- agriculture-rice
- agriculture-mennonite
- coca

There is also a "flood" category but we currently merge that into "river".

Other labels include "unknown" and "unlabeled" which indicate that example should not
be used for training.


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

The duration of the window is the time range that the forest loss was thought to have
happened. So the pre images are taken before this time range while the post images are
taken after the time range. If you want to add more images, keep in mind that the
`config_closetime2.json` enforces a 150 day gap between the most recent pre image and
the start of the time range, so use the same gap to ensure maximum compatibility
between the existing labels and any additional images.


Other Scripts
-------------

- `reformat_images.py`: reformat from rslearn dataset to a format compatibile with the
  old multisat codebase.
-
