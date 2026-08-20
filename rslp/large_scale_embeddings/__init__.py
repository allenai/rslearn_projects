"""Global quantized OlmoEarth embedding inference.

The forward flow is three ordered steps:

1. ``predict`` writes int8 embeddings into the GeoZarr archive (``write_jobs`` or
   ``supervise`` enqueues the work).
2. ``fit_pca`` samples that archive and fits the global false-color basis, so the
   basis reflects exactly the data it will be applied to.
3. ``render_pca`` reads the embeddings back and writes the ``pca_rgb`` layer
   (``write_render_jobs`` enqueues the work). CPU only, no model.
"""

from .pca import fit_pca
from .predict_pipeline import predict_pipeline
from .render_pca import annotate_pca_store, render_pca_pipeline, write_render_jobs
from .supervise import launch_supervisor, supervise
from .write_jobs import init_store, write_jobs
from .zarr_store import init_pca_store

workflows = {
    "fit_pca": fit_pca,
    "init_pca_store": init_pca_store,
    "init_store": init_store,
    "launch_supervisor": launch_supervisor,
    "predict": predict_pipeline,
    "render_pca": render_pca_pipeline,
    "annotate_pca_store": annotate_pca_store,
    "write_render_jobs": write_render_jobs,
    "supervise": supervise,
    "write_jobs": write_jobs,
}
