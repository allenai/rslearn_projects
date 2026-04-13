```shell
export RSLP_PREFIX=project_data/
python -m rslp.main common beaker_train \
    --image_name hadriens/rslpomp_260413_inst_embed \
    --cluster+=ai2/saturn \
    --experiment_id 's50ix24_instance_embedding' \
    --gpus 1 \
    --config_path rslp/embedding_pretrain/model.yaml \
    --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
    --priority urgent \
    --mode predict
```
