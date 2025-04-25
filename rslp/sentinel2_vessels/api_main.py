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
from rslp.sentinel2_vessels.predict_pipeline import (
    ImageFile,
    PredictionTask,
    predict_pipeline,
)
from rslp.sentinel2_vessels.prom_metrics import time_operation, TimerOperations
from rslp.utils.mp import init_mp
from rslp.utils.prometheus import setup_prom_metrics
from rslp.vessels import VesselDetectionDict

# Load environment variables from the .env file
load_dotenv(override=True)
# Configurable host and port, overridable via environment variables
SENTINEL2_HOST = os.getenv("SENTINEL2_HOST", "0.0.0.0")
SENTINEL2_PORT = int(os.getenv("SENTINEL2_PORT", 5555))

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


class Sentinel2Response(BaseModel):
    """Response object for vessel detections.

    Attributes:
        status: A list of status messages.
        predictions: A list of vessel detections.
        error_message: Optional, error message if the request failed.
    """

    status: StatusEnum
    predictions: list[VesselDetectionDict]
    error_message: str | None = None


class Sentinel2Request(BaseModel):
    """Request object for vessel detections.

    Attributes:
        scene_id: scene ID to process. The scene will be downloaded from GCP. One of
            scene ID or image_files must be set.
        image_files: list of ImageFile image files. Currently we only use TCI,
            so it looks like ImageFile(bands=["R", "G", "B"], fname="...").
        crop_path: Optional; Path to save the cropped images.
        scratch_path: Optional; Scratch path to save the rslearn dataset.
    """

    scene_id: str | None = None
    image_files: list[ImageFile] | None = None
    crop_path: str | None = None
    scratch_path: str | None = None

    class Config:
        """Configuration for the Sentinel2Request model."""

        json_schema_extra = {
            "examples": [
                {
                    "description": "Example with scene_id",
                    "value": {
                        "scene_id": "S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425",
                    },
                },
                {
                    "description": "Example with scene_id and crop output path",
                    "value": {
                        "scene_id": "S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425",
                        "crop_path": "gs://path/to/write/crops",
                    },
                },
                {
                    "description": "Example with image_files",
                    "value": {
                        "image_files": [
                            {
                                "bands": ["R", "G", "B"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_TCI.jp2",
                            },
                            {
                                "bands": ["B01"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B01.jp2",
                            },
                            {
                                "bands": ["B02"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B02.jp2",
                            },
                            {
                                "bands": ["B03"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B03.jp2",
                            },
                            {
                                "bands": ["B04"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B04.jp2",
                            },
                            {
                                "bands": ["B05"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B05.jp2",
                            },
                            {
                                "bands": ["B06"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B06.jp2",
                            },
                            {
                                "bands": ["B07"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B07.jp2",
                            },
                            {
                                "bands": ["B08"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B08.jp2",
                            },
                            {
                                "bands": ["B8A"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B8A.jp2",
                            },
                            {
                                "bands": ["B09"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B09.jp2",
                            },
                            {
                                "bands": ["B10"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B10.jp2",
                            },
                            {
                                "bands": ["B11"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B11.jp2",
                            },
                            {
                                "bands": ["B12"],
                                "fname": "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_B12.jp2",
                            },
                        ],
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
    return {"message": "Sentinel-2 Detections App"}


@app.post(
    "/detections",
    response_model=Sentinel2Response,
    summary="Get Vessel Detections from Sentinel-2",
    description="Returns vessel detections from Sentinel-2.",
)
async def get_detections(
    info: Sentinel2Request, response: Response
) -> Sentinel2Response:
    """Returns vessel detections for a given request.

    Args:
        info (Sentinel2Request): Sentinel2Request object containing the request data.
        response (Response): FastAPI Response object to manage the response state.

    Returns:
        Sentinel2Response: Response object with status and predictions.
    """
    if not (info.scene_id or info.image_files):
        logger.error("Invalid request: Missing scene_id or image_files.")
        raise HTTPException(
            status_code=400,
            detail="scene_id or image_files must be specified.",
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
        image_files=info.image_files,
        crop_path=info.crop_path,
    )

    try:
        logger.info("Processing request with input data.")
        with time_operation(TimerOperations.TotalInferenceTime):
            vessel_detections = predict_pipeline(
                tasks=[task],
                scratch_path=scratch_path,
            )[0]
        return Sentinel2Response(
            status=StatusEnum.SUCCESS,
            predictions=[detection.to_dict() for detection in vessel_detections],
            error_message=None,
        )
    except ValueError as e:
        logger.error(f"ValueError in prediction pipeline: {e}", exc_info=True)
        return Sentinel2Response(
            status=StatusEnum.ERROR,
            predictions=[],
            error_message=f"ValueError in prediction pipeline: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction pipeline: {e}", exc_info=True)
        return Sentinel2Response(
            status=StatusEnum.ERROR,
            predictions=[],
            error_message=f"Unexpected error in prediction pipeline: {e}",
        )


app.mount("/metrics", setup_prom_metrics())


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host=SENTINEL2_HOST,
        port=SENTINEL2_PORT,
        proxy_headers=True,
    )
