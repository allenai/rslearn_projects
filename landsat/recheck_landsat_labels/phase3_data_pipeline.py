"""Data pipeline for landsat phase3_selected."""

from rslearn.dataset import Dataset
from rslearn.main import (
    IngestHandler,
    MaterializeHandler,
    PrepareHandler,
    apply_on_windows,
)
from upath import UPath

dst_path = "/home/yawenz/ml_detections/"
dst_path = UPath(dst_path)

dataset = Dataset(dst_path)
apply_on_windows(
    PrepareHandler(force=False),
    dataset,
    workers=16,
    group="phase3a_selected",
)
apply_on_windows(
    IngestHandler(),
    dataset,
    workers=16,
    use_initial_job=False,
    jobs_per_process=1,
    group="phase3a_selected",
)
apply_on_windows(
    MaterializeHandler(),
    dataset,
    workers=16,
    use_initial_job=False,
    group="phase3a_selected",
)
