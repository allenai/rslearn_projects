"""API for NISAR Vessel Detection."""

import tempfile
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from rslp.log_utils import get_logger
from rslp.nisar_vessels.config import (
    NISAR_HOST,
    NISAR_PORT,
    NISAR_SCORE_THRESHOLD,
)
from rslp.nisar_vessels.predict_pipeline import PredictionTask, predict_pipeline
from rslp.nisar_vessels.prom_metrics import TimerOperations, time_operation
from rslp.utils.mp import init_mp
from rslp.utils.prometheus import setup_prom_metrics
from rslp.vessels import VesselDetectionDict

logger = get_logger(__name__)

# Serializes GPU inference so a single worker only ever runs one prediction at a time.
_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for the NISAR Vessel Detection Service.

    Sets up the multiprocessing start method and preloads necessary modules.

    Args:
        app: FastAPI app instance.
    """
    logger.info("Initializing NISAR Vessel Detection Service")
    init_mp()
    yield
    logger.info("NISAR Vessel Detection Service shutdown.")


app = FastAPI(
    title="NISAR Vessel Detection API",
    description="API for detecting vessels in NISAR images.",
    version="0.0.1",
    lifespan=lifespan,
    docs_url="/docs",  # URL for Swagger UI
    redoc_url="/redoc",  # URL for ReDoc
)


class StatusEnum(StrEnum):
    """Enumeration for response status.

    Attributes:
        SUCCESS: Indicates a successful response.
        ERROR: Indicates an error occurred.
    """

    SUCCESS = "success"
    ERROR = "error"


class NisarResponse(BaseModel):
    """Response object for vessel detections.

    Attributes:
        status: whether the request succeeded.
        predictions: A list of vessel detections.
        error_message: Optional, error message if the request failed.
    """

    status: StatusEnum
    predictions: list[VesselDetectionDict]
    error_message: str | None = None


class NisarRequest(BaseModel):
    """Request object for vessel detections.

    Attributes:
        h5_path: path of the NISAR HDF5 granule to detect vessels in. The caller must
            have already placed the granule somewhere this service can read it; the
            service has no data source of its own to look a granule up with.
        scene_id: Optional; the granule name. Defaults to the filename of h5_path.
        crop_path: Optional; path to save the cropped images.
        scratch_path: Optional; scratch path to save the rslearn dataset.
        score_threshold: Optional; override the detector's score threshold for this
            request. Defaults to the NISAR_SCORE_THRESHOLD env var (0.7 if unset).
    """

    h5_path: str
    scene_id: str | None = None
    crop_path: str | None = None
    scratch_path: str | None = None
    score_threshold: float | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "description": "Minimal example",
                    "value": {
                        "h5_path": "/path/to/NISAR_L2_PR_GCOV_001_030_A_019_2000_DHDH_A_20260101T000312_20260101T000347_P01101_F_N_J_001.h5",
                    },
                },
                {
                    "description": "Example with crop output path",
                    "value": {
                        "h5_path": "/path/to/granule.h5",
                        "crop_path": "gs://path/to/write/crops",
                    },
                },
            ]
        }
    )


@app.get("/", summary="Home", description="Service status check endpoint.")
async def home() -> dict:
    """Service status check endpoint.

    Returns:
        dict: A simple message indicating that the service is running.
    """
    return {"message": "NISAR Detections App"}


@app.post(
    "/detections",
    response_model=NisarResponse,
    summary="Get Vessel Detections from NISAR",
    description="Returns vessel detections from NISAR imagery.",
)
def get_detections(info: NisarRequest) -> NisarResponse:
    """Returns vessel detections for a given request.

    Args:
        info: NisarRequest object containing the request data.

    Returns:
        NisarResponse: Response object with status and predictions.
    """
    if info.scratch_path:
        scratch_path = info.scratch_path
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_dir.name

    task = PredictionTask(
        h5_path=info.h5_path,
        scene_id=info.scene_id,
        crop_path=info.crop_path,
    )

    # 0.0 is a real threshold, so only fall back to the default when the field is unset.
    score_threshold = (
        info.score_threshold
        if info.score_threshold is not None
        else NISAR_SCORE_THRESHOLD
    )

    try:
        logger.info(f"Processing request for granule {info.h5_path}")
        with _inference_lock, time_operation(TimerOperations.TotalInferenceTime):
            vessel_detections = predict_pipeline(
                tasks=[task],
                score_threshold=score_threshold,
                scratch_path=scratch_path,
            )[0]
        return NisarResponse(
            status=StatusEnum.SUCCESS,
            predictions=[detection.to_dict() for detection in vessel_detections],
            error_message=None,
        )
    except ValueError as e:
        logger.exception("ValueError in prediction pipeline")
        return NisarResponse(
            status=StatusEnum.ERROR,
            predictions=[],
            error_message=f"ValueError in prediction pipeline: {e}",
        )
    except Exception as e:
        logger.exception("Unexpected error in prediction pipeline")
        return NisarResponse(
            status=StatusEnum.ERROR,
            predictions=[],
            error_message=f"Unexpected error in prediction pipeline: {e}",
        )


app.mount("/metrics", setup_prom_metrics())


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host=NISAR_HOST,
        port=NISAR_PORT,
        proxy_headers=True,
    )
