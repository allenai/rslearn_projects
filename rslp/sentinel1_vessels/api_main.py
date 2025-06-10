"""API for Sentinel-2 Vessel Detection."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from rslp.log_utils import get_logger
from rslp.sentinel1_vessels.predict_pipeline import (
    PredictionTask,
    Sentinel1Image,
    predict_pipeline,
)
from rslp.sentinel1_vessels.prom_metrics import TimerOperations, time_operation
from rslp.utils.mp import init_mp
from rslp.utils.prometheus import setup_prom_metrics
from rslp.vessels import VesselDetectionDict

# Load environment variables from the .env file
load_dotenv(override=True)
# Configurable host and port, overridable via environment variables
SENTINEL1_HOST = os.getenv("SENTINEL1_HOST", "0.0.0.0")
SENTINEL1_PORT = int(os.getenv("SENTINEL1_PORT", 5555))

# Set up the logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for the Sentinel-2 Vessel Detection Service.

    Sets up the multiprocessing start method and preloads necessary modules.

    Args:
        app: FastAPI app instance.
    """
    logger.info("Initializing Sentinel-2 Vessel Detection Service")
    init_mp()
    yield
    logger.info("Sentinel-2 Vessel Detection Service shutdown.")


app = FastAPI(
    title="Sentinel-2 Vessel Detection API",
    description="API for detecting vessels in Sentinel-2 images.",
    version="0.0.1",
    lifespan=lifespan,
    docs_url="/docs",  # URL for Swagger UI
    redoc_url="/redoc",  # URL for ReDoc
)


class StatusEnum(str, Enum):
    """Enumeration for response status.

    Attributes:
        SUCCESS: Indicates a successful response.
        ERROR: Indicates an error occurred.
    """

    SUCCESS = "success"
    ERROR = "error"


class Sentinel1Response(BaseModel):
    """Response object for vessel detections.

    Attributes:
        status: A list of status messages.
        predictions: A list of vessel detections.
        error_message: Optional, error message if the request failed.
    """

    status: StatusEnum
    predictions: list[VesselDetectionDict]
    error_message: str | None = None


class Sentinel1Request(BaseModel):
    """Request object for vessel detections.

    Attributes:
        scene_id: scene ID to process. The scene will be downloaded from AWS
            (credentials are required). One of scene ID or
            image/historical1/historical2 must be set.
        image: the vv and vh filename for the image to detect vessels in.
        historical1: the vv and vh filename for the first historical image. It must
            have the same orbit direction as the target image.
        historical2: the vv and vh filename for the second historical image. It must
            have the same orbit direction as the target image.
        crop_path: Optional; Path to save the cropped images.
        scratch_path: Optional; Scratch path to save the rslearn dataset.
    """

    scene_id: str | None = None
    image: Sentinel1Image | None = None
    historical1: Sentinel1Image | None = None
    historical2: Sentinel1Image | None = None
    crop_path: str | None = None
    scratch_path: str | None = None

    class Config:
        """Configuration for the Sentinel1Request model."""

        json_schema_extra = {
            "examples": [
                {
                    "description": "Example with scene_id",
                    "value": {
                        "scene_id": "S1C_IW_GRDH_1SDV_20250610T051814_20250610T051839_002716_00599D_6955.SAFE",
                    },
                },
                {
                    "description": "Example with scene_id and crop output path",
                    "value": {
                        "scene_id": "S1C_IW_GRDH_1SDV_20250610T051814_20250610T051839_002716_00599D_6955.SAFE",
                        "crop_path": "gs://path/to/write/crops",
                    },
                },
                {
                    "description": "Example with image_files",
                    "value": {
                        "image": {
                            "vh": "/path/to/image_vh.tif",
                            "vv": "/path/to/image_vv.tif",
                        },
                        "historical1": {
                            "vh": "/path/to/h1_vh.tif",
                            "vv": "/path/to/h1_vv.tif",
                        },
                        "historical2": {
                            "vh": "/path/to/h2_vh.tif",
                            "vv": "/path/to/h2_vv.tif",
                        },
                    },
                },
            ]
        }


@app.get("/", summary="Home", description="Service status check endpoint.")
async def home() -> dict:
    """Service status check endpoint.

    Returns:
        dict: A simple message indicating that the service is running.
    """
    return {"message": "Sentinel-1 Detections App"}


@app.post(
    "/detections",
    response_model=Sentinel1Response,
    summary="Get Vessel Detections from Sentinel-1",
    description="Returns vessel detections from Sentinel-1.",
)
async def get_detections(
    info: Sentinel1Request, response: Response
) -> Sentinel1Response:
    """Returns vessel detections for a given request.

    Args:
        info (Sentinel1Request): Sentinel1Request object containing the request data.
        response (Response): FastAPI Response object to manage the response state.

    Returns:
        Sentinel1Response: Response object with status and predictions.
    """
    if not info.scene_id or not (info.image and info.historical1 and info.historical2):
        logger.error(
            "Invalid request: Missing scene_id or image/historical1/historical2."
        )
        raise HTTPException(
            status_code=400,
            detail="scene_id or image/historical1/historical2 must be specified.",
        )

    # Get a scratch path to use.
    if info.scratch_path:
        scratch_path = info.scratch_path
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_dir.name

    # Create the PredictionTask to pass to prediction pipeline.
    task = PredictionTask(
        scene_id=info.scene_id,
        image=info.image,
        historical1=info.historical1,
        historical2=info.historical2,
        crop_path=info.crop_path,
    )

    try:
        logger.info("Processing request with input data.")
        with time_operation(TimerOperations.TotalInferenceTime):
            vessel_detections = predict_pipeline(
                tasks=[task],
                scratch_path=scratch_path,
            )[0]
        return Sentinel1Response(
            status=StatusEnum.SUCCESS,
            predictions=[detection.to_dict() for detection in vessel_detections],
            error_message=None,
        )
    except ValueError as e:
        logger.error(f"ValueError in prediction pipeline: {e}", exc_info=True)
        return Sentinel1Response(
            status=StatusEnum.ERROR,
            predictions=[],
            error_message=f"ValueError in prediction pipeline: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction pipeline: {e}", exc_info=True)
        return Sentinel1Response(
            status=StatusEnum.ERROR,
            predictions=[],
            error_message=f"Unexpected error in prediction pipeline: {e}",
        )


app.mount("/metrics", setup_prom_metrics())


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host=SENTINEL1_HOST,
        port=SENTINEL1_PORT,
        proxy_headers=True,
    )
