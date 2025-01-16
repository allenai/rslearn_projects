Landsat Vessel Detection
---------------------------

The Landsat vessel detection model detects vessels in Landsat 8 and 9 scenes.

A two-stage model is trained to detect vessels from Landsat scenes, consisting of a detector and a classifier. The detector is trained on a dataset consisting of 7,954 Landsat patches (ranging from 384x384 to 768x768) with 18,509 vessel labels. The classifier is trained on a dataset consisting of about 2,000 annotated detections (the input patch size is 64x64). See [our paper](https://arxiv.org/pdf/2312.03207) for more details about the model and dataset.

![Image showing a Landsat image with predicted positions of ships from the model overlayed.]()


Inference
---------

First, download the both the detector and classifier checkpoint to the `RSLP_PREFIX` directory.

    cd rslearn_projects
    mkdir -p project_data/projects/landsat_vessels/data_20240924_model_20240924_imagenet_patch512_flip_03/checkpoints/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/landsat_vessels/detector/best.ckpt -O project_data/projects/landsat_vessels/data_20240924_model_20240924_imagenet_patch512_flip_03/checkpoints/last.ckpt

    mkdir -p project_data/projects/rslearn-landsat-recheck/phase123_20240919_01_copy/checkpoints/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/landsat_vessels/classifer/best.ckpt -O project_data/projects/rslearn-landsat-recheck/phase123_20240919_01_copy/checkpoints/last.ckpt

The esasiest way to apply the model is using the prediction pipeline in `rslp/landsat_vessels/predict_pipeline.py`. It accepts a Landsat scene ID and automatically downloads the scene images from AWS.

    mkdir output_crops

TODO: add the command here.


Training
--------

First, download the training dataset for detector:

    cd rslearn_projects
    mkdir -p project_data/datasets/landsat_vessels/
    wget https://storage.googleapis.com/ai2-rslearn-projects-data/landsat_vessels/landsat_vessels_detector.tar -0
    tar xvf project_data/datasets/landsat_vessels_detector.tar --directory project_data/datasets/landsat_vessels/

It is an rslearn dataset consisting of window folders like

Use the command below to train the model. Note that Weights & Biases is needed.


Second, download the training dataset for classifier:

    cd rslearn_projects
    mkdir -p project_data/datasets/landsat_vessels/
    wget  -0
    tar xvf project_data/dataset/landsat_vessels_classifier.tar --directory project_data/datasets/landsat_vessels/

To visualize outputs on the
