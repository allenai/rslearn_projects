"""Sample pipeline for running inference using EsPredictRunner."""

import logging
from pathlib import Path

from esrun.runner.local.predict_runner import EsPredictRunner

CONFIG_PATH = Path(__file__).parent
PREDICTION_REQUEST_PATH = CONFIG_PATH / "prediction_requests/test-request1.geojson"

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Main function to run the inference pipeline."""
    runner = EsPredictRunner(
        project_path=CONFIG_PATH,
        scratch_path=CONFIG_PATH / "scratch",
        prediction_request_geometry_path=PREDICTION_REQUEST_PATH,
    )
    partitions = runner.partition()
    for partition_id in partitions:
        runner.build_dataset(partition_id)
        runner.run_inference(partition_id)
        runner.postprocess(partition_id)

    runner.combine(partitions)


if __name__ == "__main__":
    main()
