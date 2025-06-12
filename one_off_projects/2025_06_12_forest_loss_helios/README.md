This is to train Helios model for forest loss driver classification by adapting the older config at
`one_off_projects/2025_04_29_helios_comparison/forest_loss_driver/helios_st_finetune.yaml` for new
version of Helios.

The updated config is at `one_off_projects/2025_06_12_forest_loss_helios/helios_wattn.yaml`.

Setup environment:

```
conda create -n rslearn python=3.12
conda activate rslearn
git clone https://github.com/allenai/rslearn
git clone --branch favyen/20250612-add-forest-loss-helios-model https://github.com/allenai/rslearn_projects
git clone --branch a0c13ba9a4ba0e4e0e8e8efc7acf03537d727806 git@github.com:allenai/helios.git
pip install ./rslearn[extra]
pip install ./helios
pip install ./rslearn_projects
```

Train the model (see `rslp/helios/README.md` for how the image is created):

```
python -m rslp.main common beaker_train --config_path one_off_projects/2025_06_12_forest_loss_helios/helios_wattn.yaml --image_name favyen/rslphelios2 --cluster+=ai2/jupiter-cirrascale-2 --weka_mounts+='{"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}'
```

Apply the model on the test set and produce visualizations. Note that this may have an error at the
end due to error writing confusion matrix since W&B is not enabled by default (unless
`--force_log=true` is passed), but the visualizations should still be produced.

```
ln -s /PATH/TO/HELIOS /opt/helios
mkdir ./vis
python -m rslp.rslearn_main model test --config one_off_projects/2025_06_12_forest_loss_helios/helios_wattn.yaml --model.init_args.visualize_dir=./vis --load_best=true
```

Build a new unlabeled dataset populated from GLAD forest loss alerts. The example here limits to
Peru with the 080W_20S_070W_10S.tif grid (see
https://console.cloud.google.com/storage/browser/earthenginepartners-hansen/S2alert), and limits
to five examples.

```
python -m rslp.main forest_loss_driver extract_alerts --ds_path /tmp/rslearn_dataset/ --extract_alerts_args.gcs_tiff_filenames ["080W_20S_070W_10S.tif"] --extract_alerts_args.countries ["PE"] --extract_alerts_args.tile_store_dir file:///tmp/tile_store/ --extract_alerts_args.index_cache_dir file:///tmp/index_cache_dir/ --extract_alerts_args.workers 32 --extract_alerts_args.max_number_of_events 5 --extract_alerts_args.days 90
cp data/forest_loss_driver/config_ms.json /tmp/rslearn_dataset/config.json
mkdir -p /tmp/rslearn_dataset/cache/planetary_computer
python -m rslp.rslearn_main dataset prepare --root /tmp/rslearn_dataset --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5
python -m rslp.rslearn_main dataset materialize --root /tmp/rslearn_dataset --workers 32 --retry-max-attempts 5 --retry-backoff-seconds 5 --no-use-initial-job
python -m rslp.main forest_loss_driver select_least_cloudy_images --ds_path /tmp/rslearn_dataset --select_least_cloudy_images_args.workers=32
```

Apply the model on this dataset:

```
python -m rslp.rslearn_main model predict --config one_off_projects/2025_06_12_forest_loss_helios/helios_wattn.yaml --data.init_args.path=/tmp/rslearn_dataset/ --load_best=true
```
