"""Diagnostics that sit outside the forward pipeline.

Nothing here runs as part of a production run: these measure or validate a store
after the fact. ``bench_chunking`` is registered as a workflow so it can run on
Beaker against real data; ``check_spatial_order`` runs standalone.
"""

from rslp.large_scale_embeddings.tools.bench_chunking import build_variants, measure

__all__ = ["build_variants", "measure"]
