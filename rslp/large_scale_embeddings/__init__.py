"""Global quantized OlmoEarth embedding inference.

The forward flow is three ordered steps:

1. ``predict`` writes int8 embeddings into the GeoZarr archive (``write_jobs`` or
   ``supervise`` enqueues the work).
2. ``fit_pca`` samples that archive and fits the global false-color basis, so the
   basis reflects exactly the data it will be applied to.
3. ``render_pca`` reads the embeddings back and writes the ``pca_rgb`` layer
   (``write_render_jobs`` enqueues the work). CPU only, no model.
4. ``render_web_pca`` warps that layer into a single web-mercator pyramid for display.
   The UTM pyramid keeps ``shard == one prediction window`` at every level, so object
   count for a view never falls however far you zoom out, and a view spanning two zones
   cannot be drawn at all. This stage fixes both. CPU only, no model.
"""

from .bench_chunking import build_variants, measure
from .full_run import launch_run_all, run_all
from .pca import fit_pca
from .predict_pipeline import predict_pipeline
from .render_pca import annotate_pca_store, render_pca_pipeline, write_render_jobs
from .render_web_pca import init_web_store, render_web_pca_pipeline
from .supervise import launch_supervisor, supervise
from .write_jobs import init_store, write_jobs
from .zarr_store import init_pca_store

workflows = {
    # Read-cost sweep over the two free chunk parameters. See bench_chunking.
    "bench_build_variants": build_variants,
    "bench_measure": measure,
    "fit_pca": fit_pca,
    "init_pca_store": init_pca_store,
    "init_store": init_store,
    "launch_run_all": launch_run_all,
    "launch_supervisor": launch_supervisor,
    "predict": predict_pipeline,
    "render_utm_pca": render_pca_pipeline,
    "render_web_pca": render_web_pca_pipeline,
    "init_web_store": init_web_store,
    "run_all": run_all,
    "annotate_pca_store": annotate_pca_store,
    "write_render_jobs": write_render_jobs,
    "supervise": supervise,
    "write_jobs": write_jobs,
}
