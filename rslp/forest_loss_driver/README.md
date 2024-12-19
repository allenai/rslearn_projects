# Forest Loss Driver

## Overview
The Forest Loss Driver project aims to develop an automated global deforestation detection system to combat illegal deforestation activities. By leveraging satellite imagery and machine learning, this system:

- Detects and monitors forest loss events in near real-time
- Identifies the drivers/causes of deforestation (e.g. agriculture, mining, logging)
- Provides evidence through before/after satellite imagery
- Enables rapid response to illegal activities
- Supports conservation area managers in resource allocation
- Improves accountability and enforcement of forest protection laws

This technology is critical for preserving forests worldwide, protecting biodiversity, and mitigating climate change impacts. The system focuses particularly on monitoring protected areas and Indigenous territories where illegal deforestation remains a significant threat despite legal protections.

## Core Functionality

The system consists of two main pipeline components:

1. **Dataset Extraction Pipeline**
   - Processes forest loss alert GeoTIFFs to identify recent deforestation events
   - Collects satellite imagery (Sentinel-2, Planet) before and after each event
   - Filters for cloud-free images to ensure high quality data
   - Materializes an rslearn dataset with standardized windows around each event

2. **Model Prediction Pipeline**
   - Takes the prepared dataset and runs inference using trained models
   - Classifies the driver/cause of each deforestation event
   - Outputs predictions in GeoJSON format with confidence scores
   - Supports batch processing for large-scale inference

## Usage

### Environment Setup
Required environment variables:
- `RSLP_PREFIX`: GCS bucket prefix for model checkpoints \

Optional environment variables:
- `INDEX_CACHE_DIR`: Directory for caching image indices
- `TILE_STORE_ROOT_DIR`: Directory for tile storage cache
- `PL_API_KEY`: Planet API key (if using Planet imagery)

Otherwise, follow set up in [main readme](../../README.md)

### Pipeline Configuration

The current inference data configuration is stored in [data/forest_loss_driver/config.json](../../data/forest_loss_driver/config.json). This contains the bands and data sources the model needs to perform inference. It is essential this dataset configuration matches the configuration used to train the model.

The current pipeline configuration is stored in [forest_loss_driver_predict_pipeline_config.yaml](inference/config/forest_loss_driver_predict_pipeline_config.yaml) the default values can be found in this [config class](inference/config.py). This configuration points to the model configuration currently in use by the pipeline.
### Running the Pipeline

1. Extract dataset

2. Predict Forest Loss Driver Events


## Links to other specific kinds of docs and functionality

training doc \
deployment doc \
