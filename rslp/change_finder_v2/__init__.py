"""Change finder v2: point-based annotation format for land cover change."""

from .esri_baseline.build_stacks import build_stack as esri_build_stack
from .esri_baseline.build_stacks import build_stacks as esri_build_stacks
from .esri_baseline.compute_changes import (
    compute_all_changes as esri_compute_all_changes,
)
from .esri_baseline.compute_changes import (
    compute_changes as esri_compute_changes,
)
from .esri_baseline.compute_changes import (
    compute_changes_batch as esri_compute_changes_batch,
)
from .esri_baseline.write_jobs import write_jobs as esri_write_jobs
from .esri_baseline.write_jobs_changes import (
    write_jobs_changes as esri_write_jobs_changes,
)
from .lcc_model.predict_pipeline import predict_multi, predict_pipeline
from .lcc_model.write_jobs import write_jobs
from .lcc_model.write_jobs_random_2048 import write_jobs_random_2048

workflows = {
    "predict": predict_pipeline,
    "predict_multi": predict_multi,
    "write_jobs": write_jobs,
    "write_jobs_random_2048": write_jobs_random_2048,
    "esri_build_stack": esri_build_stack,
    "esri_build_stacks": esri_build_stacks,
    "esri_write_jobs": esri_write_jobs,
    "esri_compute_changes": esri_compute_changes,
    "esri_compute_all_changes": esri_compute_all_changes,
    "esri_compute_changes_batch": esri_compute_changes_batch,
    "esri_write_jobs_changes": esri_write_jobs_changes,
}
