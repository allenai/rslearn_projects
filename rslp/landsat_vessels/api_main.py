"""Landsat Vessel Detection Service."""

from __future__ import annotations

import multiprocessing
import os

import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel

from rslp.landsat_vessels.predict_pipeline import FormattedPrediction, predict_pipeline
from rslp.log_utils import get_logger

app = FastAPI()

# Set up the logger
logger = get_logger(__name__)

LANDSAT_HOST = "0.0.0.0"
LANDSAT_PORT = 5555


class LandsatResponse(BaseModel):
    """Response object for vessel detections."""

    status: list[str]
    predictions: list[FormattedPrediction]


class LandsatRequest(BaseModel):
    """Request object for vessel detections."""

    scene_id: str | None = None
    image_files: dict[str, str] | None = None
    crop_path: str | None = None
    scratch_path: str | None = None
    json_path: str | None = None


@app.on_event("startup")
async def rslp_init() -> None:
    """Landsat Vessel Service Initialization."""
    logger.info("Initializing")
    multiprocessing.set_start_method("forkserver", force=True)
    multiprocessing.set_forkserver_preload(
        [
            "rslp.utils.rslearn.materialize_dataset",
            "rslp.utils.rslearn.run_model_predict",
        ]
    )


@app.get("/")
async def home() -> dict:
    """Returns a simple message to indicate the service is running."""
    return {"message": "Landsat Detections App"}


@app.post("/detections", response_model=LandsatResponse)
async def get_detections(info: LandsatRequest, response: Response) -> LandsatResponse:
    """Returns vessel detections Response object for a given Request object."""
    # Ensure that either scene_id or image_files is specified.
    if info.scene_id is None and info.image_files is None:
        raise ValueError("Either scene_id or image_files must be specified.")

    try:
        if info.scene_id is not None:
            logger.info(f"Received request with scene_id: {info.scene_id}")
        elif info.image_files is not None:
            logger.info("Received request with image_files")
        json_data = predict_pipeline(
            crop_path=info.crop_path,
            scene_id=info.scene_id,
            image_files=info.image_files,
            scratch_path=info.scratch_path,
            json_path=info.json_path,
        )
        return LandsatResponse(
            status=["success"],
            predictions=[pred for pred in json_data],
        )
    except ValueError as e:
        logger.error(f"Value error during prediction pipeline: {e}")
        return LandsatResponse(status=["error"], predictions=[])
    except Exception as e:
        logger.error(f"Unexpected error during prediction pipeline: {e}")
        return LandsatResponse(status=["error"], predictions=[])


if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host=os.getenv("LANDSAT_HOST", default="0.0.0.0"),
        port=int(os.getenv("LANDSAT_PORT", default=5555)),
        proxy_headers=True,
    )
