"""Utility functions for using Prometheus Metrics inside RSLP APIs."""
import os
from typing import Any

import prometheus_client
from prometheus_client import make_asgi_app, multiprocess


def setup_prom_metrics() -> Any:
    """Create Prometheus asgi app for Fastapi."""
    multi_proc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multi_proc_dir:
        # If we're not using multiproc, then just use the default registry
        return make_asgi_app()

    # Otherwise setup prometheus multiproc mode.
    if os.path.isdir(multi_proc_dir):
        for multi_proc_file in os.scandir(multi_proc_dir):
            os.remove(multi_proc_file.path)
    else:
        os.makedirs(multi_proc_dir)

    # Create the multiproc collector, and set it up to be connected to fastapi
    registry = prometheus_client.CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=multi_proc_dir)
    return make_asgi_app(registry=registry)
