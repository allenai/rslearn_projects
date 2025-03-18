"""
Definitions for prometheus metrics that are captured during inference to report on performance of the landsat
detection service.
"""

from enum import StrEnum

from prometheus_client import Histogram
from prometheus_client.context_managers import Timer

request_timer = Histogram(
    "landsat_rslearn_timer", "Timers for inference requests", ["operation"]
)


class TimerOperations(StrEnum):
    TotalInferenceTime = "TotalInferenceTime"
    SetupDataset = "SetupDataset"
    MaterializeDataset = "MaterializeDataset"
    RunModelPredict = "RunModelPredict"
    GetVesselDetections = "GetVesselDetections"
    RunClassifier = "RunClassifier"
    BuildPredictionsAndCrops = "BuildPredictionsAndCrops"


def time_operation(operation: TimerOperations) -> Timer:
    return request_timer.labels(operation=operation.value).time()
