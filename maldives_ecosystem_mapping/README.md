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

-


Model Training
--------------
