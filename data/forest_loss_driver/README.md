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

- config.json: current inference config that uses RGB PNGs.
- config_ms.json: corresponds to dataset 20250429, it gets L2A images with all bands
  stored as GeoTIFF.
