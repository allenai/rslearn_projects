import logging
from pathlib import Path

from esrun.runner.local.predict_runner import EsPredictRunner

CONFIG_PATH = Path(__file__).parent
MODEL_CONFIG_PATH = CONFIG_PATH / 'model.yaml'
DATASET_CONFIG_PATH = CONFIG_PATH / 'dataset.json'
PARTITION_PATH = CONFIG_PATH / 'partition_strategies.yaml'
POSTPROCESS_PATH = CONFIG_PATH / 'postprocessing_strategies.yaml'
PREDICTION_REQUEST_PATH = CONFIG_PATH / 'prediction_requests/test-request1.geojson'

logging.basicConfig(level=logging.INFO)


def main():
    runner = EsPredictRunner(
        MODEL_CONFIG_PATH,
        DATASET_CONFIG_PATH,
        PARTITION_PATH,
        POSTPROCESS_PATH,
        PREDICTION_REQUEST_PATH,
        scratch_path=CONFIG_PATH / 'scratch'
    )
    partitions = runner.partition()
    for partition_id in partitions:
        runner.build_dataset(partition_id)
        runner.run_inference(partition_id)
        runner.postprocess(partition_id)

    runner.combine(partitions)


if __name__ == "__main__":
    main()
