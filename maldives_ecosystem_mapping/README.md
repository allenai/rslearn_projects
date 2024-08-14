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

The dataset pre-processing is in preprocess/.

- retrieve_dataset.py: just copies the images and labels from GCS to local storage.
- create_rslearn_data.py: populates an rslearn dataset. The dataset will have two
  groups: images will contain the full GeoTIFF images, while crops will contain just the
  patches of the images that have annotations.

To pre-process the data:

    cd rslearn_projects/maldives_ecosystem_mapping/preprocess
    python retrieve_dataset.py --out_dir /data/favyenb/maldives_ecosystem_mapping_data/original/
    PYTHONPATH=/path/to/rslearn python create_rslearn_data.py --in_dir /data/favyenb/maldives_ecosystem_mapping_data/original/ --out_dir /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/

You may need to add missing classes to `create_rslearn_data.py`.
Also you should copy `rslearn_projects/maldives_ecosystem_mapping/config.json` to the `rslearn_dataset` directory.

Obtain Sentinel-2 images if desired:

    cd /path/to/rslearn
    python -m rslearn.main dataset prepare --root /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ --workers 32 --group crops_sentinel2
    python -m rslearn.main dataset prepare --root /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ --workers 32 --group images_sentinel2
    python -m rslearn.main dataset ingest --root /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ --workers 32 --group crops_sentinel2
    python -m rslearn.main dataset ingest --root /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ --workers 32 --group images_sentinel2
    python -m rslearn.main dataset materialize --root /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ --workers 32 --group crops_sentinel2
    python -m rslearn.main dataset materialize --root /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ --workers 32 --group images_sentinel2


Model Training
--------------

First assign crops to train/val as desired, the second argument is the number of validation images (others are training):

    cd rslearn_projects
    python maldives_ecosystem_mapping/train/assign_split.py /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ crops 4
    python maldives_ecosystem_mapping/train/assign_split.py /data/favyenb/maldives_ecosystem_mapping_data/rslearn_dataset/ crops_sentinel2 4

Then train the model:

    PYTHONPATH=/path/to/rslearn:. python -m rslp.main model fit --config maldives_ecosystem_mapping/train/config.yaml --autoresume=true
    PYTHONPATH=/path/to/rslearn:. python -m rslp.main model fit --config maldives_ecosystem_mapping/train/config_sentinel2.yaml --autoresume=true

Get visualizations of validation crops:

    PYTHONPATH=/path/to/rslearn:. python -m rslp.main model test --config maldives_ecosystem_mapping/train/config.yaml --autoresume=true --model.init_args.visualize_dir ~/vis/
    PYTHONPATH=/path/to/rslearn:. python -m rslp.main model test --config maldives_ecosystem_mapping/train/config_sentinel2.yaml --autoresume=true --model.init_args.visualize_dir ~/vis/

Write predictions of the whole images:

    PYTHONPATH=/path/to/rslearn:. python -m rslp.main model predict --config maldives_ecosystem_mapping/train/config.yaml --autoresume=true
    PYTHONPATH=/path/to/rslearn:. python -m rslp.main model predict --config maldives_ecosystem_mapping/train/config_sentinel2.yaml --autoresume=true
