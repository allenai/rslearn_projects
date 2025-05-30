This module contains code to wrap Helios model for training in rslearn/rslearn_projects
along with launcher for Helios fine-tuning experiments.

## Helios Fine-tuning

Launch Helios fine-tuning experiments with a given checkpoint:

    python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/henryh/8latent_mim_random_patch_disc_new_exit_zero_lr_4e-05_base_shallow_decoder/step95000/ --patch_size 4 --encoder_embedding_size 768 --experiment_prefix apr07test --image_name favyen/rslphelios --tasks '["eurosat"]' --configs '["finetune"]'

The Weka `dfive-default` bucket will be automatically mounted at
`/weka/dfive-default/`. Leave `--tasks` out to run on all tasks (see `data/helios/` for
the list of model configuration files that will be used as templates for the
fine-tuning experiments). `--configs` is similarly optional.

If you need to create a new image, first create a copy of `rslearn_projects` repository
with subfolders `rslearn` (containing https://github.com/allenai/rslearn) and
`helios` (containing https://github.com/allenai/helios). Then run:

    docker build -t rslphelios -f helios.Dockerfile .
    beaker image create --name rslphelios rslphelios
