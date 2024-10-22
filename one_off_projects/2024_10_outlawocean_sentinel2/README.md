joe.galvin@theoutlawocean.com requests Sentinel-2 vessel detections in the AOI below
basically for all of Sentinel-2 history.

```
[[[59.9913,-11.25],[61.8333,-11.25],[61.8333,-8.3333],[59.9913,-8.3333],[59.9913,-11.25]]]
```

First get the Sentinel-2 scene IDs:

    python -m one_off_projects.2024_10_outlawocean_sentinel2.get_scene_ids --cache_path /tmp/rslearn_cache/ --out_fname one_off_projects/2024_10_outlawocean_sentinel2/scene_ids_try2.json

Build up-to-date Docker image and push to Beaker (first put rslearn in rslearn_projects directory)

    docker build -t 2024_10_outlawocean_sentinel2 .
    beaker image create 2024_10_outlawocean_sentinel2 --name 2024_10_outlawocean_sentinel2

Then run job launcher:

    python -m one_off_projects.2024_10_outlawocean_sentinel2.job_launcher --json_fname one_off_projects/2024_10_outlawocean_sentinel2/scene_ids_sentinel2.json --count 1

Remove `--count 1` flag to run all jobs.

It will write here:

    gs://rslearn-eai/projects/2024_10_outlawocean_sentinel2/vessel_detections/
