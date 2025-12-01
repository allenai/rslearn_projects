"""Forest loss driver prediction pipeline."""

from upath import UPath

from rslp.log_utils import get_logger
from rslp.utils.rslearn import run_model_predict

MODEL_CFG_FNAME = "data/forest_loss_driver/20251104/config.yaml"

logger = get_logger(__name__)


def predict_pipeline(
    ds_path: UPath | str,
    extra_args: list[str] = [],
) -> None:
    """Apply the model on the specified dataset."""
    ds_path = UPath(ds_path) if not isinstance(ds_path, UPath) else ds_path
    logger.info(
        f"running model predict with config {MODEL_CFG_FNAME} on dataset {ds_path}"
    )
    run_model_predict(MODEL_CFG_FNAME, ds_path, extra_args=extra_args)
