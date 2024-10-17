"""Landsat Vessel Detection Service."""

from __future__ import annotations

import logging.config
import multiprocessing
import os

import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing_extensions import TypedDict

from rslp.landsat_vessels import predict_pipeline

app = FastAPI()
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"  # nosec B104
PORT = os.getenv("LANDSAT_PORT", default=5555)
# MODEL_VERSION = os.getenv("GIT_COMMIT_HASH", datetime.today())


class FormattedPrediction(TypedDict):
    """Formatted prediction for a single vessel detection."""

    # [{"longitude": 123.96480506005342, "latitude": -34.75794960371865, "score": 0.9195963740348816, "rgb_fname": "crops/0_rgb.png", "b8_fname": "crops/0_b8.png"}
    latitude: float
    longitude: float
    score: float
    rgb_fname: str
    b8_fname: str


class LandsatResponse(BaseModel):
    """Response object for vessel detections."""

    status: list[str]
    predictions: list[FormattedPrediction]


class LandsatRequest(BaseModel):
    """Request object for vessel detections."""

    scene_id: str | None = None
    image_files: dict[str, str] | None = None
    crop_path: str
    scratch_path: str
    json_path: str


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
    try:
        json_data = predict_pipeline(
            crop_path=info.crop_path,
            scene_id=info.scene_id,
            image_files=info.image_files,
            scratch_path=info.scratch_path,
            json_path=info.json_path,
        )
        return LandsatResponse(status=["success"], predictions=json_data)
    except Exception as e:
        logger.error(f"Error during prediction pipeline: {e}")
        return LandsatResponse(status=["error"], predictions=[])


if __name__ == "__main__":
    uvicorn.run("api_main:app", host=HOST, port=PORT, proxy_headers=True)
