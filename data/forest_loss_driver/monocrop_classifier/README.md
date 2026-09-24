# Monocrop classifier configs

`config.json` is the rslearn dataset config and `model_classify_pool.yaml` is the
model config that worked best. See `rslp/forest_loss_driver/monocrop_classifier/README.md`
for how to create the dataset, train, and predict.

## Experiment history

Several other configs were tried before settling on `model_classify_pool.yaml`.
They were removed from the repository; this section records what was tried so
the choices in the remaining config make sense.

### Label hierarchy

The Studio annotations use eight raw classes: nodata, mennonites_nonsoybean,
mennonites_soybean, oil_palm, other_agriculture, pastures, rice, and soybean.
We compared:

- all eight raw classes;
- Bolivia-only windows (`groups: [bolivia]`) with all eight classes;
- Bolivia-only, soybean-only windows (a `class_group: soy` window tag selected
  mennonites_soybean and soybean) remapped to a 3-class nodata /
  mennonites_soybean / soybean problem;
- all countries with soybean merged into the mennonites_soybean slot (7
  classes).

The merged 7-class label set over all countries was the one carried forward.
`create_dataset.py` now bakes this merge into the
`label_vector` layer (`class_name` is the merged class, `raw_class_name` keeps
the Studio label), so the model config needs no `class_id_mapping`.

### Encoder fine-tuning

With OlmoEarth-v1.2-Base as the encoder we compared a fully frozen encoder
(`FreezeUnfreeze` on `model.encoder.0` with plain `AdamW`) against layer-wise
learning-rate decay (`LayerDecayAdamW`, `layer_decay_rate: 0.8`,
`num_layers: 12`). LLRD fine-tuning was better and is used in
`model_classify_pool.yaml`.

### Task formulation

The first models were segmentation models (`UNetDecoder` +
`SegmentationTask`) trained on a rasterized annotation polygon, with custom
per-window metrics that reduced each 128x128 window to a single majority-vote
prediction. Since every window is a single event with a single class, we
switched to window classification (`ClassificationTask` on the `label_vector`
layer), and compared three ways to get one feature vector per window:

- `FeatureCenterCrop` to `[1, 1]`, i.e. classify from the center patch token
  only: noticeably worse;
- masked pooling over only the patches inside the event polygon (a binary
  footprint raster was written for this): about the same as pooling over all
  patches;
- `PoolingDecoder` max-pooling over the whole feature map: as good as masked
  pooling and simpler.

`model_classify_pool.yaml` uses the last option. The classification model
reports `val_accuracy` (micro-averaged, so it is the fraction of windows
predicted correctly) and a confusion matrix, which replaced the per-window
metrics of the segmentation models.
