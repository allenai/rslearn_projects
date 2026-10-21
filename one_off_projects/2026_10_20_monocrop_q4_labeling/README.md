# Q4 monocrop labeling (2026-10)

Q4 monocrop annotation round for the ACA collaboration, set up like the Q4 forest loss
driver validation in `../2026_09_25_amazon_forest_loss_labeling`. The result is the Studio
project "Monocrop Validation 2026-10" (`9241501a-5fa6-49e1-a1aa-dafafe24d70b`, forest loss
driver org) with 1000 tasks.

Outputs are in
`/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/2026_10_20_monocrop_q4_labeling/`
(OUT below), and the monocrop prediction dataset is
`/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/monocrop_classifier/predict_20261021_q4/`.

## Steps

1. Extract GLAD events of at least 5 ha from 2023-01 through 2025-10 (so that every
   event has 12 months of post-event imagery) for BO/BR/CO/EC/PE, with unsmoothed
   (pixel-exact) polygons, then select at most 1000 per (country, month) with a 250 m
   minimum distance:

       python ../2026_09_25_amazon_forest_loss_labeling/select_events.py extract \
           --out_dir OUT/events_2023_2025 --start_month 2023-01 --end_month 2025-11 \
           --min_area_ha 5 --workers 14
       python ../2026_09_25_amazon_forest_loss_labeling/select_events.py select \
           --out_dir OUT/events_2023_2025

   The area is measured in the local UTM zone and the filter is applied before the
   per-tile-month cap. This gave 130,578 candidates and 58,639 selected events.

2. Run the forest loss driver model (utm) on Studio and merge the outputs:

       python ../2026_09_25_amazon_forest_loss_labeling/studio_jobs.py launch \
           --out_dir OUT/events_2023_2025 --models utm --job_name_prefix monocrop_q4_20261020
       # ... status / download, then:
       python ../2026_09_25_amazon_forest_loss_labeling/merge_predictions.py \
           --out_dir OUT/events_2023_2025 --job_name_prefix monocrop_q4_20261020

3. Export every labeled monocrop annotation (645) for the 1 km exclusion, and build the
   pool of agriculture events to run the monocrop classifier on (20,279; only Brazil
   hits the 6000 per-country cap):

       python export_labeled_annotations.py --out_dir OUT
       python build_monocrop_candidates.py --out_dir OUT --per_country 6000 \
           --source 2023_2025=OUT/events_2023_2025/events_with_new_model_predictions.geojson

4. Create, prepare and materialize the prediction dataset
   (`rslp.forest_loss_driver.monocrop_classifier.create_prediction_dataset`, then
   `rslearn dataset prepare/materialize`). The OlmoEarth Datasets Sentinel-2 source needs
   `OEDATASETS_API_URL`/`DATASETS_API_TOKEN`, and some assets are in requester-pays S3
   buckets that need AWS credentials, so materialization ran as Beaker jobs with those
   secrets (`rslp.main common launch_data_materialization_jobs`). Then predict on a
   Beaker GPU with the 20260914 checkpoint:

       python one_off_projects/2026_10_20_monocrop_q4_labeling/launch_monocrop_predict.py \
           --ds_path .../predict_20261021_q4/ --management_dir OUT/predict_mgmt

5. Select 1000 events for annotation (30 per country and predicted class, shortfalls
   filled from the same class in other countries, plus a recall sample for rice,
   oil_palm and mennonites_nonsoybean; tasks at least 1 km apart):

       python select_for_annotation.py --out_dir OUT --ds_path .../predict_20261021_q4

6. Create the Studio project (same annotation fields as the earlier monocrop projects,
   basemaps copied from Monocrop - Phase 2, 2560 m task boxes, shuffled task names, no
   prediction shown):

       python create_studio_project.py --out_dir OUT

   `OUT/studio_tasks.json` maps each task to its event_uid, and
   `OUT/events_for_annotation.geojson` has the predicted class, probabilities and
   sampling stratum of each event.
