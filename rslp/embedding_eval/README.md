## Embedding Evaluations

This module provides utilities to evaluate OlmoEarth embeddings in different settings
and against AlphaEarth embeddings.

Compared to rslearn (training with frozen encoder), the evaluation here is much faster
since we precompute the embeddings.

Compared to olmoearth_pretrain (kNN and linear probe evaluations), the datasets we test
on are more consistent here (window size at least 128x128 with label only in the center
of the window, and 12 Sentinel-2 images), allowing experimentation with settings like
overlap ratio and different input sizes.

## Datasets

Datasets should have a "sentinel2" layer with the Sentinel-2 L2A image time series.

The window options (in the window metadata) should have a key containing the label
category. For example, in the AlphaEarth evals, the key is "label", while for the AWF
and Nandi datasets, the key is "category". This label should correspond to the center
pixel of the window.

The dataset should be split into train and test in one of two ways:

1. Using groups. There should be a group named "train" and a group named "test".
2. Using another key in the window options. For example, Nandi and AWF use a key called
   "split". The value should be "train" or "test" (others are ignored).

### AlphaEarth Supplemental Evaluations

We test on several of the AlphaEarth supplemental evaluation datasets, which we
download from https://zenodo.org/records/16585402. For internal use, the WEKA paths
are:

- Raw format: `/weka/dfive-default/rslearn-eai/artifacts/deepmind_alphaearth_supplemental_evaluation_datasets/`
- Converted to rslearn format: `/weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/`

We test on these datasets (others may not have Sentinel-2 images materialized, or may
be regression tasks that we don't currently support):

- africa_crop_mask
- canada_crops_fine
- descals
- glance
- lcmap_lu
- us_trees

Below we document how the datasets are converted to rslearn format.

#### Convert to rslearn dataset format

The `create_datasets.py` converts them to rslearn dataset format. It has the WEKA paths
hardcoded, so simply run the script:

```
python -m rslp.embedding_eval.convert_alphaearth_supplemental_to_rslearn
```

Then the data needs to be materialized.

```
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/aster_ged/ --workers 128 --jobs-per-process 16 --retry-max-attempts 10 --retry-backoff-seconds 5 --disabled-layers landsat
rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/aster_ged/ --workers 128 --retry-max-attempts 10 --retry-backoff-seconds 5 --disabled-layers landsat --ignore-errors
```

### Other Datasets

Here are other datasets we can evaluate embeddings on:

- `/weka/dfive-default/rslearn-eai/datasets/awf/`
- `/weka/dfive-default/rslearn-eai/datasets/nandi/`

## Obtain AlphaEarth Embeddings

If you are setting up a new dataset for embedding evaluation, you can use `config.json`
which includes a "gse" layer that obtains AlphaEarth embeddings using the
`rslearn.data_sources.aws_google_satellite_embedding_v1.GoogleSatelliteEmbeddingV1`
data source.

Alternatively, you can copy just that "gse" layer into your existing dataset config
file.

## Explicitly Compute Embeddings and Evaluate

We can manually run a command to compute and cache embeddings, and another command to
evaluate the embeddings.

### Compute Embeddings

First, compute embeddings. The dataset must have a "sentinel2" layer with Sentinel-2
L2A image time series (the datasets mentioned above have this).

```bash
python -m rslp.embedding_eval.compute_olmoearth_embeddings \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/africa_crop_mask/ \
    --patch_size 1 \
    --model_id OlmoEarth-v1-Base \
    --input_size 32 \
    --embed_fname embeddings.h5
```

`embeddings.h5` will contain two datasets in the H5 file, "embeddings" with a
`(N, embed_dim)` tensor of embeddings, and "window_names" with a corresponding list of
group and window names.

You can specify an OlmoEarth checkpoint directory instead of the model ID:

```bash
python -m rslp.embedding_eval.compute_olmoearth_embeddings \
    --checkpoint_dir /weka/dfive-default/helios/checkpoints/favyen/favyen_decode_gse_worldcover_osm_srtm_titan/step370000 \
    # ...
```

By default, the images will be center cropped based on the `--input_size`, and we save
the embedding corresponding to the center patch. Center cropping means the label (which
we always assume corresponds to the center of the window) is in the center of the
input. We can pass `--label_position` to put the label in a different position in the
input, to e.g. test the impact of different overlap ratios and how the model performs
with less spatial context.

```bash
python -m rslp.embedding_eval.compute_olmoearth_embeddings \
    --patch_size 4 \
    --input_size 32 \
    # Have the script crop the window such that the center pixel of the window appears
    # at the bottom right of the crop.
    --label_position 31 31 \
    # ...
```

### Evaluate

Run an evaluation with kNN:

```bash
python -m rslp.embedding_eval.get_balanced_accuracy \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations/africa_crop_mask/ \
    # How many evaluation runs to average metrics over. If set > 1, then samples should
    # be set > 0, otherwise each run would use the same train set.
    --repeats 1 \
    # How many examples to sample per category on each run. 0 means to use all of the
    # training data. The --repeats and --samples option are mainly used for consistency
    # with AlphaEarth evaluation; for internal comparisons we can disable it.
    --samples 0 \
    # K for kNN evaluation method.
    --k 3 \
    # The filename containing the embeddings, or "gse" to load AlphaEarth embeddings
    # from a "gse" layer in the dataset. There are a couple other options too, see
    # --help for details.
    --embed_fname embeddings.h5 \
    # Either knn or linear_probe.
    --method knn \
    # The key in the window options containing the label category.
    --label_key label \
    # The key in the window options containing the split. "group" means the dataset has
    # "train" and "test" groups instead.
    --split_key group
```

The linear probe evaluation has a few different options, these are the defaults:

```bash
python -m rslp.embedding_eval.get_balanced_accuracy \
    --method linear_probe \
    # Learning rate for training the linear probe.
    --lr 0.001 \
    # Number of epochs to train for.
    --epochs 100 \
    # The batch size.
    --batch_size 32 \
    # ...
```

### Automated Evaluation

We can use `run_crop_experiments.py` to evaluate many settings together and create a
table.

There are example JSON files that configure the settings to evaluate on in
`rslp/embedding_eval/crop_experiment_configs/`. It will test each crop config combined
with method, while the patch size and model ID or checkpoint directory are fixed. It
will try to evaluate AlphaEarth embeddings as well.

```bash
python -m rslp.embedding_eval.run_crop_experiments --experiment_config rslp/embedding_eval/crop_experiment_configs/crop_experiment_results.json
```

The script is designed to run correctly when executed in parallel across multiple GPUs:
it will shuffle the experiments specified by the experiment config and iterate over
them, so different executions will process different experiments and skip over ones
that were previously completed based on the results JSON file.
