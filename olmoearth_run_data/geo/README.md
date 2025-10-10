# Global Ecosystem Watch #

export yamls from studio. do some cleaning maybe
- .json -> .geojson
- remove underscore from labels dict bands
- maybe rename es -> oe

    python -m rslp.main olmoearth_run prepare_labeled_windows --project_path /weka/dfive-default/joer/rslearn_projects/olmoearth_run_data/geo --scratch_path /weka/dfive-default/rslearn-eai/datasets/geo

    export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/geo/dataset_v2/dataset
    rslearn dataset prepare --root $DATASET_PATH --workers 64 --retry-max-attempts 8 --force
    rslearn dataset materialize --root $DATASET_PATH --workers 64 --retry-max-attempts 8


for local fine tuning

    export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/geo/dataset/
    python -m rslp.rslearn_main model fit --config data/helios/v2_geo_north_africa/finetune_6months.yaml

to launch beaker job

    python -m rslp.main helios launch_finetune --image_name favyen/rslphelios20 --config_paths+=data/helios/v2_geo_north_africa/finetune_6months.yaml --cluster+=ai2/jupiter --rslp_project helios_finetuning --experiment_id geo_north_africa_test2


    export EXTRA_FILES_PATH=/weka/dfive-default/helios/checkpoints
    export DATASET_PATH=/weka/dfive-default/joer/test/datasets/scratch_v0
    export NUM_WORKERS=32
    export TRAINER_DATA_PATH=/weka/dfive-default/joer/test/geo_north_africa
    export WANDB_PROJECT=helios_finetuning
    export WANDB_NAME=geo_north_africa_testlabel
    export WANDB_ENTITY=eai-ai2
    export GOOGLE_CLOUD_PROJECT=earthsystem-dev-c3po
    python -m rslp.main olmoearth_run olmoearth_run --config_path olmoearth_run_data/geo/ --scratch_path /weka/dfive-default/joer/test/datasets/scratch_v0/ --checkpoint_path /weka/dfive-default/joer/rslearn_projects/project_data/projects/helios_finetuning/geo_north_africa_test1/checkpoints/last.ckpt
