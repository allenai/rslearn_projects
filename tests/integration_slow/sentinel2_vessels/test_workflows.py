import json
import pathlib

from upath import UPath

from rslp.sentinel2_vessels.predict_pipeline import (
    ImageFile,
    PredictionTask,
    predict_pipeline,
)


def test_predict_pipeline_scene_with_vessels(tmp_path: pathlib.Path) -> None:
    """Verify that prediction detects vessels in a scene that has lots of vessels."""
    should_have_vessels_task = PredictionTask(
        scene_id="S2A_MSIL1C_20161130T110422_N0204_R094_T30UYD_20161130T110418",
        json_path=str(tmp_path / "1.json"),
        crop_path=str(tmp_path / "crops_1"),
    )
    tasks = [should_have_vessels_task]
    # TODO: Test S2B_MSIL1C_20200206T222749_N0209_R072_T01LAL_20200206T234349 too but
    # right now it doesn't work for scenes like that which cross 0 longitude.

    scratch_path = tmp_path / "scratch"
    scratch_path.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        UPath(task.crop_path).mkdir(parents=True, exist_ok=True)

    predict_pipeline(
        tasks=tasks,
        scratch_path=str(scratch_path),
    )

    with UPath(should_have_vessels_task.json_path).open() as f:
        # This is some scene off coast of UK which should have a bunch of vessels.
        vessels = json.load(f)
        assert len(vessels) > 0


def test_predict_pipeline_image_files(tmp_path: pathlib.Path) -> None:
    """Verify that prediction works when we pass individual image files."""
    # Reference the images on GCS.
    gcs_fname_template = "gs://gcp-public-data-sentinel-2/tiles/30/U/YD/S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425.SAFE/GRANULE/L1C_T30UYD_A016722_20180904T110820/IMG_DATA/T30UYD_20180904T110621_{band}.jp2"
    image_files = []
    for band in [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B8A",
    ]:
        image_files.append(
            ImageFile(
                bands=[band],
                fname=gcs_fname_template.format(band=band),
            )
        )

    tasks = [
        PredictionTask(
            image_files=image_files,
        )
    ]

    scratch_path = tmp_path / "scratch"
    scratch_path.mkdir(parents=True, exist_ok=True)

    predict_pipeline(
        tasks=tasks,
        scratch_path=str(scratch_path),
    )
