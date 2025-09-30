Google Satellite Embeddings
---------------------------

The dataset uses `config_gse.json` instead of `config.json` here which includes a layer
for Google Satellite Embeddings, to evaluate how well those embeddings perform on the
EuroSat task.

The layer expands the window duration to one year to ensure it matches with images in
the Google Earth Engine collection containing the embeddings. This is needed because,
although the images span a year, the current GEE data source in rslearn always sets the
time range to a point in time.

Here are the steps to materialize this data, which takes a few days (see
`rslp/common/README.md` for more details on the Beaker data materialization):

```
python -m rslp.rslearn_main dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/eurosat/rslearn_dataset/ --workers 64
python -m rslp.main common launch_data_materialization_jobs --image favyen/rslp_image --ds_path /weka/dfive-default/rslearn-eai/datasets/lfmc/20250626 --hosts+=jupiter-cs-aus-134.reviz.ai2.in --command '["rslearn", "dataset", "materialize", "--root", "/weka/dfive-default/rslearn-eai/datasets/lfmc/20250626", "--workers", "32", "--load-workers", "128", "--ignore-errors"]'
```
