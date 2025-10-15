This module contains code to wrap OlmoEarth model for training in rslearn/rslearn_projects
along with launcher for OlmoEarth fine-tuning experiments.

## OlmoEarth Fine-tuning

Launch OlmoEarth fine-tuning experiments with a given checkpoint:

    python -m rslp.main olmoearth_pretrain launch_finetune --olmoearth_checkpoint_path /weka/dfive-default/olmoearth_pretrain/checkpoints/henryh/8latent_mim_random_patch_disc_new_exit_zero_lr_4e-05_base_shallow_decoder/step95000/ --patch_size 4 --encoder_embedding_size 768 --experiment_prefix apr07test --image_name favyen/rslpomp20251015 --tasks '["eurosat"]' --configs '["finetune"]'

The Weka `dfive-default` bucket will be automatically mounted at
`/weka/dfive-default/`. Leave `--tasks` out to run on all tasks (see `data/helios/` for
the list of model configuration files that will be used as templates for the
fine-tuning experiments). `--configs` is similarly optional.

If you need to create a new image, first create a copy of `rslearn_projects` repository
with subfolders `docker_build/rslearn` (containing https://github.com/allenai/rslearn) and
`docker_build/olmoearth_pretrain` (containing https://github.com/allenai/olmoearth_pretrain). Then run:

    docker build -t rslpomp -f olmoearth_pretrain.Dockerfile .
    beaker image create --name rslpomp rslpomp
