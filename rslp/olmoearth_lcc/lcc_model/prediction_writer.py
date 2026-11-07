"""Prediction writer that writes LCC window outputs into a GeoZarr store.

The standard RslearnWriter encodes each window's 62-band uint16 output as a GeoTIFF
in the main loop, which the prediction pipeline then had to read back. LCCTileWriter
instead merges each window's crops in memory and hands the merged array to a thread
pool (zstd and numpy release the GIL, and threads avoid pickling ~0.5 GB arrays)
that writes the window into the GeoZarr store (plus the probability arrays in full
stores) and computes its 2x reduction for the pyramid store.

The level-1 blocks are kept in ``results`` for the prediction pipeline to assemble
per tile. A semaphore caps the merged windows in flight to bound memory.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from lightning.pytorch import LightningModule, Trainer
from rslearn.dataset import Window
from rslearn.train.prediction_writer import (
    CropPredictionMerger,
    PendingCropOutput,
    RslearnWriter,
)
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection

from rslp.log_utils import get_logger

from . import geozarr
from .geozarr_pyramid import reduce_2x

logger = get_logger(__name__)

DEFAULT_WRITER_WORKERS = 4
DEFAULT_MAX_PENDING_WINDOWS = 4


@dataclass
class WindowResult:
    """The outputs of one window that the pipeline assembles per tile."""

    bounds: PixelBounds
    # The level-1 (change, change_dates) reduction, if pyramid is set.
    level1: tuple[np.ndarray, np.ndarray] | None


class LCCTileWriter(RslearnWriter):
    """Write LCC window outputs to a GeoZarr store."""

    def __init__(
        self,
        output_layer: str,
        zarr_store_path: str,
        pyramid: bool = False,
        workers: int = DEFAULT_WRITER_WORKERS,
        max_pending_windows: int = DEFAULT_MAX_PENDING_WINDOWS,
        selector: list[str] | None = None,
        merger: CropPredictionMerger | None = None,
    ):
        """Create a new LCCTileWriter.

        Args:
            output_layer: the output layer of the dataset config, which only
                provides the band dtype for merging crops (nothing is written to
                it).
            zarr_store_path: the GeoZarr store to write windows into.
            pyramid: whether to compute each window's level-1 pyramid reduction.
            workers: number of writer threads.
            max_pending_windows: maximum merged windows submitted but not yet
                processed.
            selector: keys to access the desired output in the output dict.
            merger: merger for the outputs of overlapping crops, e.g. a
                RasterMerger with the overlap_pixels of the predict config.
        """
        super().__init__(output_layer=output_layer, selector=selector, merger=merger)
        self.zarr_store_path = zarr_store_path
        self.pyramid = pyramid
        self.workers = workers
        self.max_pending_windows = max_pending_windows

        self.results: list[WindowResult] = []
        # Seconds that on_predict_end waited for the writer threads.
        self.write_wait = 0.0
        # Windows whose last crop was already processed. The all-crops dataset pads
        # dataloader workers by repeating their windows, so crops of these windows
        # may arrive again and must be ignored.
        self.finished_windows: set[str] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._semaphore = threading.Semaphore(max_pending_windows)
        self._futures: list[Future[WindowResult]] = []
        self._full_store = False

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ) -> None:
        """Initialize the merger and the writer threads.

        Args:
            trainer: the trainer.
            pl_module: the LightningModule.
            stage: the stage (fit/validate/test/predict).
        """
        super().setup(trainer, pl_module, stage)
        detail = geozarr.read_store_detail(self.zarr_store_path)
        if detail is None:
            raise ValueError(f"GeoZarr store {self.zarr_store_path} does not exist")
        self._full_store = detail == geozarr.DETAIL_FULL
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="lcc_writer"
            )

    def process_output(
        self,
        window: Window,
        crop_idx: int,
        num_crops: int,
        cur_bounds: PixelBounds,
        output: npt.NDArray | list[Feature],
    ) -> None:
        """Collect one crop output and submit the window once it is complete.

        Args:
            window: the window that the output pertains to.
            crop_idx: the index of this crop for the window.
            num_crops: the total number of crops to be processed for the window.
            cur_bounds: the bounds of the current crop.
            output: the output data.
        """
        if window.name in self.finished_windows:
            return
        self._raise_failed()

        self.pending_outputs.setdefault(window.name, []).append(
            PendingCropOutput(cur_bounds, output)
        )
        if crop_idx < num_crops - 1:
            return

        pending_output = self.pending_outputs.pop(window.name)
        self.finished_windows.add(window.name)
        assert self.layer_config is not None and self.merger is not None
        merged = self.merger.merge(window, pending_output, self.layer_config)

        assert self._executor is not None
        self._semaphore.acquire()
        future = self._executor.submit(
            self._process_window, window.name, window.projection, window.bounds, merged
        )
        future.add_done_callback(lambda _: self._semaphore.release())
        self._futures.append(future)

    def _process_window(
        self,
        name: str,
        projection: Projection,
        bounds: PixelBounds,
        arr: np.ndarray,
    ) -> WindowResult:
        """Write one merged window output and derive its tile-level results."""
        change, change_dates = geozarr.geozarr_window_arrays(arr)
        geozarr.write_window(
            self.zarr_store_path,
            projection,
            bounds,
            change,
            change_dates,
            prob_arrays=geozarr.prob_window_arrays(arr) if self._full_store else None,
        )
        level1 = reduce_2x(change, change_dates) if self.pyramid else None
        logger.debug("wrote window %s", name)
        return WindowResult(bounds=bounds, level1=level1)

    def _raise_failed(self) -> None:
        """Re-raise the error of the first failed window job, if any."""
        for future in self._futures:
            if future.done() and future.exception() is not None:
                raise RuntimeError("LCC window writer failed") from future.exception()

    def _shutdown_executor(self) -> None:
        """Cancel queued window jobs and wait for the running ones to finish."""
        self._futures = []
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Stop the writer threads when prediction fails.

        Lightning skips on_predict_end in that case, so without this the threads
        would outlive the failed tile and keep writing its queued windows.

        Args:
            trainer: the trainer.
            pl_module: the LightningModule.
            exception: the exception that stopped prediction.
        """
        self._shutdown_executor()

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Wait for all window jobs and collect their results.

        Args:
            trainer: the trainer.
            pl_module: the LightningModule.
        """
        start_time = time.monotonic()
        try:
            for future in self._futures:
                self.results.append(future.result())
        finally:
            self._shutdown_executor()
            self.write_wait = time.monotonic() - start_time
        logger.info("wrote %d windows to %s", len(self.results), self.zarr_store_path)
