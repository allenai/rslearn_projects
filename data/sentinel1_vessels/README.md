These are the config files for Sentinel-1 vessel detection.

The training data has separate groups for ascending orbit direction versus descending
orbit direction. This is because the model only performs well with historical images,
and the historical images are aligned with the current image when they share the same
orbit direction (otherwise, terrain correction is needed, which is expensive).


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
