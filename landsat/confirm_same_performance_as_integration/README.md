2024-07-19 we want to make sure that the Landsat model has same performance when
executed directly from multisat as in integration.

So we find some detections in Skylight integration and use rslearn to get the Landsat
images.

    python -m rslearn.main dataset add_windows --root /data/favyenb/rslearn_landsat_inference_tmp/ --group default --box 120.2772,-36.1314,120.2772,-36.1314 --utm --src_crs EPSG:4326 --window_size 512 --start 2024-07-19T01:47:00Z --end 2024-07-19T01:48:00Z --resolution 15
