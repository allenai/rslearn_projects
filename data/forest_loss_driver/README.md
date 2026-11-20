This file summarizes the different dataset and model configuration files here. For
details on how the dataset was created, see `rslp/forest_loss_driver/README.md`.


Dataset Versions
----------------

- 20240912: original version of dataset.
- 20250424: fix point labels so that they are contained within the window. This is both
  for compatibility with ES Studio and because new version of rslearn VectorFormat only
  loads features within the window bounds.
- 20250429: remove Planet images and replace Sentinel-2 L1C images with L2A images from
  Planetary Computer which are stored as GeoTIFFs.
- 20250514: like 20250424 (with Sentinel-2 L1C images + Planet images) but put the
  polygon in the GeoJSON instead of the point (so when we import to ES Studio it shows
  up nicely with the forest loss polygon) and update items.json to include the best_X
  layers (so that the timestamps appear for those layers in ES Studio).
- 20250605: keep the Planet images but get 6 pre and 6 post Sentinel-2 L2A, Sentinel-1,
  and Landsat images from Planetary Computer and AWS.
- 20260924_utm: rebuild the training dataset directly from the labels in OlmoEarth
  Studio (`rslp/forest_loss_driver/create_dataset.py`) instead of syncing labels into
  the old rslearn windows. Windows are 128x128 at 10 m/pixel in UTM rather than Web
  Mercator, and only the Sentinel-2 L2A layers are kept. Adds the "Validatetest"
  labels. See `20260924_utm/README.md`.
- 20261120_hierarchy: same windows and dataset config as 20260924_utm, but the label
  feature also contains the set of allowed leaves of ACA's new three-level label
  hierarchy (`leaf_labels`), and the model predicts those 22 leaves using a loss that
  supports labels that only identify a non-leaf node. Adds the Validation 2026-09 and
  monocrop projects. See `20261120_hierarchy/README.md`.


Current Configuration
---------------------

`config.json` (dataset) and `config.yaml` (model) are the current default, copied from
`20260924_utm/`. The dataset config has the label layer plus the 4 least cloudy pre and
4 least cloudy post Sentinel-2 L2A images (matching what the model reads) via the
OlmoEarth Datasets data source in olmoearth_run. The model is OlmoEarth-v1.2-Base
fine-tuned with layer-wise learning rate decay. This is the model deployed in
`olmoearth_projects/olmoearth_run_data/forest_loss_driver/`.

`20261120_hierarchy/config.yaml` is the in-progress model on the new label hierarchy.
It is not the default yet because the deployed inference pipeline expects the ten flat
categories.

The older top-level configs (`config_ms.json`, `config_multimodal.json`,
`config_planet.json`, `config_studio_annotation.json`, and
`config_satlaspretrain_flip_oldmodel.yaml`) were removed; they can be recovered from
git history (they are present at commit `c69d5e64`). The subdirectories keep the
configs for earlier dataset and model versions.


Deployment Details
------------------

- 20260925: deploy the `20260924_utm/config.yaml` model (OlmoEarth-v1.2-Base, trained
  on the UTM dataset built from Studio labels including Validatetest). The inference
  pipeline now creates 10 m/pixel UTM windows to match.
- 20251219: the deployment is moved from rslearn_projects, where it was running in a
  Beaker job, onto the OlmoEarth platform, with the code to update forest-loss.allen.ai
  in `olmoearth_projects.projects.forest_loss_driver.deploy`. It still uses
  OlmoEarth-v1-FT-ForestLossDriver-Base, which corresponds to  `20251104/config.yaml`
  here.
- 20251104: deploy OlmoEarth-v1-FT-ForestLossDriver-Base on Brazil, Peru, and Colombia.
  The model uses Sentinel-2 L2A images from Microsoft Planetary Computer.
- 20240912: original deployment trained on Peru only, applying Satlas on Sentinel-2 L1C
  RGB PNGs.
