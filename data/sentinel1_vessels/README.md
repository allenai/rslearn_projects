These are the config files for Sentinel-1 vessel detection.

The training data has separate groups for ascending orbit direction versus descending
orbit direction. This is because the model only performs well with historical images,
and the historical images are aligned with the current image when they share the same
orbit direction (otherwise, terrain correction is needed, which is expensive).
Starting with dataset version 20261103, the historical images also share the target's
relative orbit (same track), so they have the same viewing geometry as well.


config.json
-----------

This is the configuration for the training data.

Note that the orbit_direction attribute of the sentinel1_historical layer needs to be
set differently for the ascending groups vs the descending groups. See
`one_off_projects/convert_satlas_webmercator_to_rslearn/sentinel1_vessel/README.md` for
details.

This is also used for prediction when scene ID is provided and data is fetched from AWS
bucket. In this case the prediction pipeline populates the items.json so the orbit
direction in the config is unmodified (but has no effect).


config_predict_local_files.json
-------------------------------

This is for prediction when the user provides the vv/vh files directly. The user must
ensure the historical images are the same orbit direction as the target image.


Dataset Versions
----------------

- 20250521: this is generated from the siv.sqlite3 in sentinel-vessel-detections.
- 20250602: this is generated from `gs://satlas-explorer-data/siv-annotations/sentinel1.tar`.
  It corresponds to the first subset of annotations which are included in 20250521. These
  annotations may be higher quality but we seem to get better performance from 20250521, so
  this dataset should not be used.
- 20261103: 20250521 with same-track historical images and cleaned-up labels. The current
  model (`config.yaml`) was retrained on it with
  `20261103_static_labels/config_1hist_randhist.yaml`; see the README there for results.
  - `sentinel1_historical` has up to 4 item groups from the target's relative orbit (same
    track, not just same orbit direction), one random pass per 30-day period from 5 to 155
    days before the target, most recent first. rslearn's prepare can't express this, so
    run `python -m rslp.main sentinel1_vessels prepare --ds_path ...` after preparing the
    `sentinel1` layer.
  - Labels on static objects (platforms, turbines, islands) and other wrong labels were
    removed: 61,655 -> 52,577 labels. Their windows stay as hard negatives. See
    `one_off_projects/2026_11_03_s1_static_labels/README.md`.
