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


Dataset Configurations
----------------------

- config_ms.json: corresponds to dataset 20250429, it gets L2A images with all bands
  stored as GeoTIFF.
- config_studio_annotation.json: this is original config used for Brazil+Colombia
  dataset. It creates RGB GeoTIFFs that are good for uploading to and visualizing in
  Studio. It also gets Planet Labs RGB images.
- config_multimodal.json: this gets inputs that match what Helios can do, Sentinel-2 +
  Sentinel-1 + Landsat.


Deployment Details
------------------

- 20251219: the deployment is moved from rslearn_projects, where it was running in a
  Beaker job, onto the OlmoEarth platform, with the code to update forest-loss.allen.ai
  in `olmoearth_projects.projects.forest_loss_driver.deploy`. It still uses
  OlmoEarth-v1-FT-ForestLossDriver-Base, which corresponds to  `20251104/config.yaml`
  here.
- 20251104: deploy OlmoEarth-v1-FT-ForestLossDriver-Base on Brazil, Peru, and Colombia.
  The model uses Sentinel-2 L2A images from Microsoft Planetary Computer.
- 20240912: original deployment trained on Peru only, applying Satlas on Sentinel-2 L1C
  RGB PNGs.
