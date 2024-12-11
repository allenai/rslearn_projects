"""Model Predict Step for the Forest Loss Driver Inference Pipeline."""

from upath import UPath

from rslp.utils.rslearn import run_model_predict


# TODO: Add code to collect all outputs and return them
def forest_loss_driver_model_predict(
    model_cfg_fname: str, ds_path: str | UPath, model_data_load_workers: int
) -> None:
    """Run the model predict pipeline."""
    run_model_predict(
        model_cfg_fname,
        ds_path,
        # TODO: This a brittle hack to get the model load data workers be configurable upstream
        extra_args=[
            "--data.init_args.num_workers",
            str(model_data_load_workers),
        ],
    )
