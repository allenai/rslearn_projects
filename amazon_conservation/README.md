This contains code for the training the forest loss driver model, as well as
visualizing the outputs of the prediction pipeline.

The prediction pipeline itself has been integrated into `rslp` and some of this code
should be moved there later too.


Training and Prediction
-----------------------

Train the model:

    python -m rslp.rslearn_main model fit --config amazon_conservation/train/config_satlaspretrain_flip_oldmodel_unfreeze.yaml

Run the prediction data preparation pipeline, this populates an rslearn dataset with windows corresponding to recent forest loss events:

    python -m rslp.main forest_loss_driver predict --pred_config.workers 32 --pred_config.ds_root gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/

Replace YYYYMMDD with the date you are starting to run the pipeline on.
You can reduce the time range to predict using `--pred_config.days X` (the default is 365 to predict over the previous year).

Now obtain the images, this will take many hours:

    rslearn dataset prepare --root gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/ --workers 128 --batch-size 4
    rslearn dataset ingest --root gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/ --workers 64 --jobs-per-process 1 --no-use-initial-job
    rslearn dataset materialize --root gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/ --workers 64 --no-use-initial-job --batch-size 4

Run the second step of the pipeline which identifies the least cloudy images, these are the ones that the model will input:

    python -m rslp.main forest_loss_driver select_best_images --ds_path gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/

Now the model can be applied:

    python -m rslp.rslearn_main model predict --config data/forest_loss_driver/config_satlaspretrain_flip_oldmodel_unfreeze.yaml --data.init_args.path gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/


Visualization
-------------

After the prediction pipeline completes, we can visualize the forest loss events together with the predicted drivers.
First create a JSON file containing the windows that have outputs. Some windows will not have had enough non-cloudy Sentinel-2 images to compute outputs.

    python -m amazon_conservation.predict.find_windows_with_outputs --ds_path gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/ --out_fname good_windows.json

Collect all of the forest loss events with the predicted driver into a GeoJSON file:

    python -m amazon_conservation.predict.collect_outputs_to_geojson --ds_path gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/ --windows_fname good_windows.json --out_fname forest_loss_events.geojson

Use [tippecanoe](https://github.com/felt/tippecanoe) (requires installation from source) to create vector tiles of the GeoJSON suitable for loading in the web app:

    tippecanoe -zg -e out_tiles/ --drop-smallest-as-needed --no-tile-compression out.geojson
    # Copy to a public bucket, e.g.:
    aws s3 --endpoint-url https://b2bcf985082a37eaf385c532ee37928d.r2.cloudflarestorage.com sync out_tiles/ s3://satlas-explorer-data/rslearn-public/forest_loss_driver/YYYYMMDD/tiles/

You will need to change the `let pbfLayer = L.vectorGrid.protobuf` in `map.html` to match the path where the tiles can be read by web browser.
Now launch the webserver:

    cd amazon_conservation/predict
    python server.py gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_YYYYMMDD/ 8080 ../../good_windows.json
