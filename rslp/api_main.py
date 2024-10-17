"""VIIRS Vessel Detection Service."""

from __future__ import annotations

import logging.config
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

    # class Config:
    #     """example configuration for a request where files are stored in cloud"""

    #     schema_extra = {
    #         "example": {
    #             "input_dir": "input",
    #             "output_dir": "output",
    #             "filename": "VJ102DNB.A2022362.0154.021.2022362055600.nc",
    #             "geo_filename": "VJ103DNB.A2022362.0154.021.2022362052511.nc",
    #             "modraw_filename": "VJ102MOD.A2022362.0154.002.2022362115107.nc",
    #             "modgeo_filename": "VJ103MOD.A2022362.0154.002.2022362095104.nc",
    #         },
    # }


@app.on_event("startup")
async def rslp_init() -> None:
    """VIIRS Vessel Service Initialization."""
    logger.info("Initializing")


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

    #     ves_detections = all_detections["vessel_detections"]

    #     satellite_name = utils.get_provider_name(dnb_dataset)
    #     acquisition_time, end_time = utils.get_acquisition_time(dnb_dataset)
    #     chips_dict = utils.get_chips(image, ves_detections, dnb_dataset)
    #     if info.gcp_bucket is not None:
    #         chips_dict = utils.upload_image(
    #             info.gcp_bucket, chips_dict, info.output_dir, dnb_path
    #         )
    #     else:
    #         utils.save_chips_locally(
    #             chips_dict,
    #             destination_path=info.output_dir,
    #             chip_features=ves_detections,
    #         )

    #     average_moonlight = RoundedFloat(utils.get_average_moonlight(dnb_dataset), 2)

    #     frame_extents = utils.get_frame_extents(dnb_dataset)

    # predictions = utils.format_detections(chips_dict)
    # time_s = round(perf_counter() - start)
    # n_ves = len(chips_dict)
    # logger.info(
    #     f"In frame: {dnb_path}, vvd detected {n_ves} vessels in ({time_s} seconds)"
    # )
    # response.headers["n_detections"] = str(len(chips_dict))
    # response.headers["avg_moonlight"] = str(average_moonlight)
    # response.headers["lightning_count"] = str(all_detections["lightning_count"])
    # response.headers["gas_flare_count"] = str(all_detections["gas_flare_count"])
    # response.headers["inference_time"] = str("elapsed_time")


if __name__ == "__main__":
    uvicorn.run("api_main:app", host=HOST, port=PORT, proxy_headers=True)
