This is for the Maldives ecosystem mapping collaboration with GEO Ecosystem Atlas.

It takes data annotated in Kili and exported to Google Cloud Storage, prepares an
rslearn dataset from those images and annotations, and then trains a model and supports
applying that model on large images.


Dataset Pre-processing
----------------------

On GCS, the dataset consists of big GeoTIFF files with paired semantic segmentation
annotations that are only labeled within one or more small bounding boxes in the image.

So the pre-processing will extract just those small crops and add them as labeled
images in the rslearn dataset.

It will also add the big images as unlabeled windows in a separate group in the same
rslearn dataset, for prediction.

To pre-process the data:

    python -m rslp.main maldives_ecosystem_mapping data --dp_config.workers 32

This yields an rslearn dataset with two groups: images will contain the full GeoTIFF
images, while crops will contain just the patches of the images that have annotations.

There will also be two more groups for Sentinel-2 training: images_sentinel2 and
crops_sentinel2. Ingesting the Sentinel-2 images can take quite a while; if you just
want Maxar, then you can skip the ingestion:

    python -m rslp.main maldives_ecosystem_mapping data --dp_config.workers 32 --dp_config.skip_ingest true

TODO: the config.json currently has an absolute path for cache directory. Need to
address https://github.com/allenai/rslearn/issues/31 before this will work correctly.
In the meantime, you can edit `config.json` manually or use `skip_ingest`.


Model Training
--------------

Train the model:

    PYTHONPATH=/path/to/rslearn:. python -m rslp.rslearn_main model fit --config data/maldives_ecosystem_mapping/config.yaml --autoresume=true
    PYTHONPATH=/path/to/rslearn:. python -m rslp.rslearn_main model fit --config data/maldives_ecosystem_mapping/config_sentinel2.yaml --autoresume=true

Get visualizations of validation crops:

    PYTHONPATH=/path/to/rslearn:. python -m rslp.rslearn_main model test --config data/maldives_ecosystem_mapping/config.yaml --autoresume=true --model.init_args.visualize_dir ~/vis/
    PYTHONPATH=/path/to/rslearn:. python -m rslp.rslearn_main model test --config data/maldives_ecosystem_mapping/config_sentinel2.yaml --autoresume=true --model.init_args.visualize_dir ~/vis/

Write predictions of the whole images:

    PYTHONPATH=/path/to/rslearn:. python -m rslp.rslearn_main model predict --config data/maldives_ecosystem_mapping/config.yaml --autoresume=true
    PYTHONPATH=/path/to/rslearn:. python -m rslp.rslearn_main model predict --config data/maldives_ecosystem_mapping/config_sentinel2.yaml --autoresume=true
