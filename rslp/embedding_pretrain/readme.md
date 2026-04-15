```shell
export RSLP_PREFIX="${MY_ROOT}/project_data"
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



# Compile embeddings to chunks
```shell
export OUTPUT_PATH="${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings"

python -m rslp.embedding_pretrain.compile_win_embeddings \
    --dataset "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/" \
    --group s50ix24 \
    --output-path ${OUTPUT_PATH} \
    --chunk-size 8192 \
    --crop-size 96
    # --test-mode \
```
