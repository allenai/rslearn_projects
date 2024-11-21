"""API for Landsat Vessel Detection."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from rslp.landsat_vessels.predict_pipeline import FormattedPrediction, predict_pipeline
from rslp.log_utils import get_logger
from rslp.utils.mp import init_mp

# Load environment variables from the .env file
load_dotenv()
# Configurable host and port, overridable via environment variables
LANDSAT_HOST = os.getenv("LANDSAT_HOST", "0.0.0.0")
LANDSAT_PORT = int(os.getenv("LANDSAT_PORT", 5555))

# Set up the logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for the Landsat Vessel Detection Service.

    Sets up the multiprocessing start method and preloads necessary modules.

    Args:
        app: FastAPI app instance.
    """
    logger.info("Initializing Landsat Vessel Detection Service")
    init_mp()
    yield
    logger.info("Landsat Vessel Detection Service shutdown.")


app = FastAPI(
    title="Landsat Vessel Detection API",
    description="API for detecting vessels in Landsat images.",
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


class LandsatResponse(BaseModel):
    """Response object for vessel detections.

    Attributes:
        status: A list of status messages.
        predictions: A list of formatted predictions.
    """

    status: list[StatusEnum]
    predictions: list[FormattedPrediction]


class LandsatRequest(BaseModel):
    """Request object for vessel detections.

    Attributes:
        scene_id: Optional; Scene ID to process. This queries scenes from AWS (only for T1 and T2 scenes).
        scene_zip_path: Optional; Path to a zipped scene file. This queries scenes from a downloaded zip file (local or on GCS).
        image_files: Optional; Dictionary of image files. This queries scenes from downloaded image files (local or on GCS).
        crop_path: Optional; Path to save the cropped images.
        scratch_path: Optional; Scratch path to save the rslearn dataset.
        json_path: Optional; Path to save the JSON output (the response object).

    Note:
        Only one of `scene_id`, `scene_zip_path`, or `image_files` needs to be provided.
    """

    scene_id: str | None = None
    scene_zip_path: str | None = None
    image_files: dict[str, str] | None = None
    crop_path: str | None = None
    scratch_path: str | None = None
    json_path: str | None = None

    class Config:
        """Configuration for the LandsatRequest model."""

        json_schema_extra = {
            "examples": [
                {
                    "description": "Example with scene_id",
                    "value": {
                        "scene_id": "LC08_L1TP_123032_20200716_20200722_01_T1",
                    },
                },
                {
                    "description": "Example with scene_id and paths",
                    "value": {
                        "scene_id": "LC08_L1TP_123032_20200716_20200722_01_T1",
                        "crop_path": "gs://path/to/crop",
                        "scratch_path": "gs://path/to/scratch",
                        "json_path": "gs://path/to/output.json",
                    },
                },
                {
                    "description": "Example with scene_zip_path",
                    "value": {
                        "scene_zip_path": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT.zip",
                    },
                },
                {
                    "description": "Example with image_files",
                    "value": {
                        "image_files": {
                            "B2": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B2.TIF",
                            "B3": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B3.TIF",
                            "B4": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B4.TIF",
                            "B5": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B5.TIF",
                            "B6": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B6.TIF",
                            "B7": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B7.TIF",
                            "B8": "gs://path/to/landsat_8_9/downloads/2024/10/30/LC08_L1GT_102011_20241030_20241030_02_RT_B8.TIF",
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
    return {"message": "Landsat Detections App"}


@app.post(
    "/detections",
    response_model=LandsatResponse,
    summary="Get Vessel Detections from Landsat",
    description="Returns vessel detections from Landsat.",
)
async def get_detections(info: LandsatRequest, response: Response) -> LandsatResponse:
    """Returns vessel detections for a given request.

    Args:
        info (LandsatRequest): LandsatRequest object containing the request data.
        response (Response): FastAPI Response object to manage the response state.

    Returns:
        LandsatResponse: Response object with status and predictions.
    """
    if not (info.scene_id or info.scene_zip_path or info.image_files):
        logger.error(
            "Invalid request: Missing scene_id, scene_zip_path, or image_files."
        )
        raise HTTPException(
            status_code=400,
            detail="scene_id, scene_zip_path, or image_files must be specified.",
        )
    try:
        logger.info("Processing request with input data.")
        json_data = predict_pipeline(
            scene_id=info.scene_id,
            scene_zip_path=info.scene_zip_path,
            image_files=info.image_files,
            json_path=info.json_path,
            scratch_path=info.scratch_path,
            crop_path=info.crop_path,
        )
        return LandsatResponse(status=[StatusEnum.SUCCESS], predictions=json_data)
    except ValueError as e:
        logger.error(f"ValueError in prediction pipeline: {e}", exc_info=True)
        return LandsatResponse(status=[StatusEnum.ERROR], predictions=[])
    except Exception as e:
        logger.error(f"Unexpected error in prediction pipeline: {e}", exc_info=True)
        return LandsatResponse(status=[StatusEnum.ERROR], predictions=[])


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host=LANDSAT_HOST,
        port=LANDSAT_PORT,
        proxy_headers=True,
    )
