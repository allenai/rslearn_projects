Per-Pixel Land Cover Change Finder
==================================

`find_change.py` applies the per-pixel land-cover model to a ten-year Sentinel-2
dataset and, for each window, picks one qualifying land-cover decline (a class whose
probability stays above `--src_threshold` in the three years before a pivot year and
falls below `--dst_threshold` in the three years after). It writes one V2-compatible
annotation JSON file per window under `--output_dir`.

Optionally, `--src_category` restricts the declining (source) class to a specific
class name, and `--dst_category` requires the post-period argmax class to be a
specific class name (thresholds still apply to the source class probability only).

Example local usage:

    python -m rslp.change_finder_v2.per_pixel_land_cover.find_change \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
        --checkpoint_path /path/to/checkpoints/last.ckpt \
        --output_dir /path/to/output_jsons/ \
        --src_category tree \
        --dst_category urban/built-up


Launching on Beaker
-------------------

The script can be run as a Beaker job using the generic launcher in
`rslp.common.beaker_launcher` (registered as the `common beaker_launcher` workflow).
It assumes you have already built and pushed a Beaker image containing rslearn and
rslearn_projects (see `rslp/common/README.md`).

Since `find_change` is not a registered rslp workflow, use the launcher's `--command`
option to run the module directly. For example, to launch one job on the Jupiter
cluster with urgent priority and one GPU (with WEKA mounted so the dataset,
checkpoint, and output directory can live on `/weka`):

    python -m rslp.main common beaker_launcher \
        --image YOUR_BEAKER_IMAGE \
        --clusters '["ai2/jupiter-cirrascale-2"]' \
        --gpu_count 1 \
        --priority urgent \
        --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
        --command '["python", "-m", "rslp.change_finder_v2.per_pixel_land_cover.find_change", "--ds_path", "/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/", "--checkpoint_path", "/weka/dfive-default/rslearn-eai/projects/2026_04_05_worldcover_change/olmoearth_base_s2_center10_per_pixel_ps1_01/best.ckpt", "--output_dir", "/weka/dfive-default/path/to/output_jsons/", "--workers", "32"]'

Notes:

- The `BEAKER_ADDR`, `BEAKER_CONFIG`, and `BEAKER_TOKEN` environment variables must be
  configured for the launcher to work.
- Exactly one of `--clusters` or `--hostname` must be set.
- Multiple jobs can be launched against the same `--output_dir`: windows are processed
  in random order and windows with existing output JSON files are skipped, so jobs
  will not duplicate work.
