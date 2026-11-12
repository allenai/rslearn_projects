"""Global quantized OlmoEarth embedding inference."""

from .predict_pipeline import materialize_pipeline, predict_pipeline
from .write_jobs import init_store, write_jobs

workflows = {
    "init_store": init_store,
    "materialize": materialize_pipeline,
    "predict": predict_pipeline,
    "write_jobs": write_jobs,
}
