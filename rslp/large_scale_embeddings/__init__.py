"""Global quantized OlmoEarth embedding inference."""

from .predict_pipeline import predict_pipeline
from .write_jobs import write_jobs

workflows = {
    "predict": predict_pipeline,
    "write_jobs": write_jobs,
}
