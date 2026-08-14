"""Global quantized OlmoEarth embedding inference."""

from .predict_pipeline import predict_pipeline
from .write_jobs import init_store, write_jobs

workflows = {
    "init_store": init_store,
    "predict": predict_pipeline,
    "write_jobs": write_jobs,
}
