"""Global quantized OlmoEarth embedding inference."""

from .predict_pipeline import predict_pipeline
from .supervise import launch_supervisor, supervise
from .write_jobs import init_store, write_jobs

workflows = {
    "init_store": init_store,
    "launch_supervisor": launch_supervisor,
    "predict": predict_pipeline,
    "supervise": supervise,
    "write_jobs": write_jobs,
}
