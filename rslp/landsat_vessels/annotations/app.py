"""Annotation app for the Landsat vessel classifier re-annotation rounds.

Serves the pre-rendered window views (crop, zoom-out, panchromatic) and spectral curves
for a window group, and records one label per detection to an append-only JSONL on weka.

Append-only is the point: every keystroke is durable the moment it is made, a relabel is
a new record rather than a mutation, and the file survives the browser, the server and
the machine.

Each label is then written straight through into that window's rslearn ``label`` layer,
so the dataset is up to date as you annotate and no separate step is required to make the
labels real. The JSONL stays the source of truth: it is written and fsynced *before* the
window write is attempted, so a failing dataset write can never lose an annotation, and
``scripts/apply_round1_labels.py`` reconciles the whole log against the dataset (and
writes the labelled pool copies) whenever you want to be sure.

Usage:
    python -m uvicorn rslp.landsat_vessels.annotations.app:app --host 127.0.0.1 --port 8501

Environment:
    ANNOTATION_ROUND_DIR  round directory (default /weka/.../landsat/annotation_round1)
    ANNOTATION_GROUP      window group (default round1_20260803)
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rslearn.dataset import Dataset
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.landsat_vessels.annotations import writeback

logger = logging.getLogger(__name__)

DATASET_ROOT = UPath(
    os.environ.get(
        "ANNOTATION_DATASET_ROOT",
        "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624",
    )
)
ROUND_DIR = Path(
    os.environ.get(
        "ANNOTATION_ROUND_DIR", "/weka/dfive-default/yawenz/landsat/annotation_round1"
    )
)
GROUP = os.environ.get("ANNOTATION_GROUP", "round1_20260803")

ASSETS_DIR = ROUND_DIR / "assets"
LABELS_PATH = ROUND_DIR / "labels" / f"{GROUP}_labels.jsonl"
STATIC_DIR = Path(__file__).parent / "static"

# The classifier's two classes, plus the two non-answers. "unsure" is a real annotation
# outcome and must stay distinguishable from "not yet seen": a detection a human could
# not call is evidence about the data, and silently dropping those would make the
# remaining labels look cleaner than the imagery actually is.
LABELS = ["correct", "incorrect", "unsure", "skip"]
REASONS = ["ice", "cloud", "glint", "whitecap", "land", "wake", "other"]

# Metadata carried through to the UI, in display order.
DISPLAY_FIELDS = [
    "slice",
    "stratum",
    "tier",
    "split",
    "region",
    "classifier_label",
    "classifier_prob_correct",
    "detector_score",
    "distance_to_coast_m",
    "land_cover_class",
    "scene_cloud_cover",
    "ts",
    "scene_id",
    "latitude",
    "longitude",
    "selection_reason",
    "substituted_product",
]


class LabelRequest(BaseModel):
    """One annotation event."""

    window: str
    label: str
    reason: str | None = None
    note: str = ""
    annotator: str = "unknown"
    elapsed_ms: int | None = Field(
        default=None, description="time spent on this window"
    )


class ItemsRequest(BaseModel):
    """A page's worth of windows to fetch at once."""

    windows: list[str] = Field(max_length=200)


class Store:
    """Index, spectra and labels, loaded once and mutated under a lock."""

    def __init__(self) -> None:
        """Load the window index and initialize in-memory annotation state."""
        with (ROUND_DIR / f"{GROUP}_index.json").open() as f:
            records = json.load(f)
        self.order: list[str] = [r["window"] for r in records]
        self.records: dict[str, dict] = {r["window"]: r for r in records}

        spectra_path = ROUND_DIR / "spectra.json"
        if spectra_path.exists():
            with spectra_path.open() as f:
                self.spectra = json.load(f)
        else:
            self.spectra = {
                "points": [],
                "refl_bands": [],
                "thermal_bands": [],
                "curves": {},
            }

        # Windows whose imagery has finished rendering. Recomputed on demand so the app
        # can be started while the render is still running.
        self.rendered: set[str] = set()
        self.refresh_rendered()

        self.labels: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._load_labels()

        # Write-through into the rslearn dataset. Windows are loaded lazily and cached:
        # loading all 2,000 up front costs seconds of startup for something most sessions
        # touch a few hundred of.
        self.dataset = Dataset(DATASET_ROOT)
        self.vector_format = GeojsonVectorFormat()
        self.windows: dict[str, object] = {}
        self.sync_failures: dict[str, str] = {}

    def get_window(self, name: str) -> object:
        """The rslearn Window for a name, loaded on first use."""
        if name not in self.windows:
            loaded = self.dataset.load_windows(groups=[GROUP], names=[name])
            if not loaded:
                raise KeyError(f"window {name} not found in group {GROUP}")
            self.windows[name] = loaded[0]
        return self.windows[name]

    def sync_to_window(self, record: dict) -> None:
        """Apply one annotation event to the window's label layer.

        Raises on failure; the caller reports it rather than the annotator losing work,
        since the JSONL has already been written by this point.
        """
        name = record["window"]
        window = self.get_window(name)
        label = record.get("label")
        if label is None or label == "skip":
            # Cleared, or taken back to "come back to this": the dataset should not keep
            # a label the annotator no longer stands behind.
            writeback.clear_label(window)
            return
        properties = writeback.build_properties(record, self.records[name], GROUP)
        writeback.write_label(window, properties, self.vector_format)

    def refresh_rendered(self) -> int:
        """Rescan which windows have a rendered crop."""
        if not ASSETS_DIR.exists():
            self.rendered = set()
            return 0
        self.rendered = {
            path.name
            for path in ASSETS_DIR.iterdir()
            if (path / "crop_auto.png").exists()
        }
        return len(self.rendered)

    def _load_labels(self) -> None:
        """Replay the JSONL; the last record for a window wins."""
        if not LABELS_PATH.exists():
            return
        with LABELS_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                window = record.get("window")
                if window is None:
                    continue
                if record.get("label") is None:
                    self.labels.pop(window, None)
                else:
                    self.labels[window] = record

    def append(self, record: dict) -> None:
        """Append one event to the JSONL and update the in-memory view."""
        with self.lock:
            LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LABELS_PATH.open("a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if record.get("label") is None:
                self.labels.pop(record["window"], None)
            else:
                self.labels[record["window"]] = record

    def item(self, window: str) -> dict:
        """Return the rendered view payload for a single window."""
        record = self.records[window]
        return {
            "window": window,
            "index": self.order.index(window),
            "rendered": window in self.rendered,
            "meta": {field: record.get(field) for field in DISPLAY_FIELDS},
            "detection_id": record.get("detection_id"),
            "crop_view_size": record.get("crop_view_size"),
            "train_view_size": record.get("train_view_size"),
            "window_size": 512,
            "resolution": record.get("resolution"),
            "spectra": self.spectra["curves"].get(window),
            "label": self.labels.get(window),
        }


app = FastAPI(title="Landsat vessel annotation")
store = Store()

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/")
def index() -> FileResponse:
    """The single-page app."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/session")
def session() -> dict[str, Any]:
    """Everything the client needs to build the queue and the filter controls."""
    facets: dict[str, dict[str, int]] = {
        "slice": {},
        "stratum": {},
        "tier": {},
        "split": {},
    }
    for record in store.records.values():
        for facet in facets:
            value = str(record.get(facet))
            facets[facet][value] = facets[facet].get(value, 0) + 1

    queue = [
        {
            "window": window,
            "slice": store.records[window].get("slice"),
            "stratum": store.records[window].get("stratum"),
            "tier": store.records[window].get("tier"),
            "split": store.records[window].get("split"),
            "prob": store.records[window].get("classifier_prob_correct"),
            "rendered": window in store.rendered,
            "label": (store.labels.get(window) or {}).get("label"),
            "reason": (store.labels.get(window) or {}).get("reason"),
        }
        for window in store.order
    ]
    return {
        "group": GROUP,
        "total": len(store.order),
        "rendered": len(store.rendered),
        "labels": LABELS,
        "reasons": REASONS,
        # Band definitions for the curve axes; the samples themselves come per item.
        "points": store.spectra["points"],
        "refl_bands": store.spectra["refl_bands"],
        "thermal_bands": store.spectra["thermal_bands"],
        "facets": {k: dict(sorted(v.items())) for k, v in facets.items()},
        "queue": queue,
        "labels_path": str(LABELS_PATH),
        "dataset_root": str(DATASET_ROOT),
        "write_through": True,
        "sync_failures": len(store.sync_failures),
    }


@app.get("/api/item/{window}")
def item(window: str) -> dict[str, Any]:
    """Full detail for one window."""
    if window not in store.records:
        raise HTTPException(status_code=404, detail=f"unknown window {window}")
    return store.item(window)


@app.post("/api/items")
def items(request: ItemsRequest) -> dict[str, Any]:
    """Detail for a page of windows in one round trip.

    The gallery shows 20 at a time, each with its own spectral curve; fetching them
    one by one would be 20 requests per page turn.
    """
    unknown = [w for w in request.windows if w not in store.records]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown windows: {unknown[:5]}")
    return {"items": [store.item(window) for window in request.windows]}


@app.post("/api/label")
def label(request: LabelRequest) -> dict[str, Any]:
    """Record one annotation. Sending label='clear' removes the window's label."""
    if request.window not in store.records:
        raise HTTPException(status_code=404, detail=f"unknown window {request.window}")
    if request.label == "clear":
        record: dict[str, Any] = {
            "window": request.window,
            "label": None,
            "annotator": request.annotator,
            "ts": time.time(),
        }
    else:
        if request.label not in LABELS:
            raise HTTPException(status_code=400, detail=f"bad label {request.label}")
        if request.reason is not None and request.reason not in REASONS:
            raise HTTPException(status_code=400, detail=f"bad reason {request.reason}")
        record = {
            "window": request.window,
            "label": request.label,
            "reason": request.reason,
            "note": request.note,
            "annotator": request.annotator,
            "elapsed_ms": request.elapsed_ms,
            "ts": time.time(),
        }
    # Order matters: the log is durable before the dataset write is attempted, so a
    # dataset problem degrades to "run apply_round1_labels.py later", never to lost work.
    store.append(record)
    synced = True
    sync_error = None
    try:
        store.sync_to_window(record)
        store.sync_failures.pop(request.window, None)
    except Exception as exc:
        synced = False
        sync_error = f"{type(exc).__name__}: {exc}"
        store.sync_failures[request.window] = sync_error
        logger.exception("failed to write label layer for %s", request.window)
    return {
        "ok": True,
        "label": record,
        "synced": synced,
        "sync_error": sync_error,
        "sync_failures": len(store.sync_failures),
        "progress": progress(),
    }


@app.get("/api/progress")
def progress() -> dict[str, Any]:
    """Counts of what has been labelled, overall and per slice."""
    by_label: dict[str, int] = {}
    by_slice: dict[str, dict[str, int]] = {}
    by_reason: dict[str, int] = {}
    for window, record in store.labels.items():
        value = record.get("label")
        if value is None:
            continue
        by_label[value] = by_label.get(value, 0) + 1
        slice_name = str(store.records[window].get("slice"))
        by_slice.setdefault(slice_name, {})
        by_slice[slice_name][value] = by_slice[slice_name].get(value, 0) + 1
        reason = record.get("reason")
        if reason:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    return {
        "total": len(store.order),
        "labeled": sum(by_label.values()),
        "by_label": dict(sorted(by_label.items())),
        "by_slice": {k: dict(sorted(v.items())) for k, v in sorted(by_slice.items())},
        "by_reason": dict(sorted(by_reason.items())),
    }


@app.post("/api/resync")
def resync() -> dict[str, Any]:
    """Re-apply every label in the log to its window.

    The write-through path covers normal use; this is the repair button for labels made
    while the dataset was unwritable, and it is what the app's "resync" control calls.
    """
    written = 0
    failed: dict[str, str] = {}
    for window, record in list(store.labels.items()):
        try:
            store.sync_to_window(record)
            written += 1
        except Exception as exc:  # noqa: BLE001 - collected and reported per window
            failed[window] = f"{type(exc).__name__}: {exc}"
    store.sync_failures = failed
    return {"attempted": len(store.labels), "ok": written, "failed": failed}


@app.get("/api/label_layers")
def label_layers() -> dict[str, Any]:
    """How many windows in the group currently carry a label layer.

    The independent check on write-through: this counts what is actually on disk in the
    dataset, not what the app believes it wrote.
    """
    windows_dir = DATASET_ROOT / "windows" / GROUP
    on_disk = sum(
        1
        for window in windows_dir.iterdir()
        if (window / "layers" / writeback.LABEL_LAYER / "completed").exists()
    )
    return {
        "labeled_in_log": sum(1 for r in store.labels.values() if r.get("label")),
        "label_layers_on_disk": on_disk,
        "sync_failures": len(store.sync_failures),
    }


@app.post("/api/refresh_rendered")
def refresh_rendered() -> dict[str, Any]:
    """Rescan the assets directory, for use while a render is still running."""
    return {"rendered": store.refresh_rendered(), "total": len(store.order)}


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Health check: report the group name and number of rendered windows."""
    return JSONResponse({"ok": True, "group": GROUP, "rendered": len(store.rendered)})
