"""Tests for supervise's stage dispatch.

The two stages share the shallow-queue and worker-top-up loop; what differs is how
remaining work is enumerated and which workflow the queue entries name. These tests
cover the validation and the enumeration branch without touching Beaker.
"""

import importlib

import pytest

from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs

# The package re-exports the workflow functions under their module names, so
# `from rslp.large_scale_embeddings import supervise` yields the function. Import the
# module itself to reach its constants.
sup = importlib.import_module("rslp.large_scale_embeddings.supervise")


def _base_kwargs(**overrides: object) -> dict:
    kwargs = dict(
        inputs=EmbeddingInputs.S2,
        years=[2024],
        store_path="gs://bucket/s2.zarr",
        completed_path_template="gs://bucket/s2_{year}_completed/",
        queue_name="user/queue",
        checkpoint_path="/weka/ckpt",
        image_name="user/image",
        cluster=["ai2/cluster"],
        max_cycles=0,
    )
    kwargs.update(overrides)
    return kwargs


def test_stages_are_the_two_expected() -> None:
    assert sup.STAGES == (sup.STAGE_PREDICT, sup.STAGE_RENDER_PCA)
    assert sup.STAGE_PREDICT == "predict"
    assert sup.STAGE_RENDER_PCA == "render_pca"


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="stage must be one of"):
        sup.supervise(**_base_kwargs(stage="nonsense"))


def test_render_stage_requires_artifact_and_marker_path() -> None:
    with pytest.raises(ValueError, match="artifact_path, pca_completed_path"):
        sup.supervise(**_base_kwargs(stage=sup.STAGE_RENDER_PCA))

    with pytest.raises(ValueError, match="pca_completed_path"):
        sup.supervise(
            **_base_kwargs(
                stage=sup.STAGE_RENDER_PCA, artifact_path="gs://bucket/pca"
            )
        )

    with pytest.raises(ValueError, match="artifact_path"):
        sup.supervise(
            **_base_kwargs(
                stage=sup.STAGE_RENDER_PCA,
                pca_completed_path="gs://bucket/pca_completed/",
            )
        )


def test_render_stage_error_points_at_fit_pca() -> None:
    with pytest.raises(ValueError, match="fit_pca"):
        sup.supervise(**_base_kwargs(stage=sup.STAGE_RENDER_PCA))


def test_predict_stage_needs_no_pca_arguments() -> None:
    """max_cycles=0 exits before any Beaker call, so this only exercises validation."""
    sup.supervise(**_base_kwargs())


def test_run_cycle_render_stage_enumerates_from_source_markers(monkeypatch) -> None:
    """The render stage must call get_render_jobs, not the predict enumerator."""
    calls: dict[str, object] = {}

    def fake_get_render_jobs(**kwargs: object) -> list[list[str]]:
        calls.update(kwargs)
        return [["--source_marker", "a.json"], ["--source_marker", "b.json"]]

    monkeypatch.setattr(
        "rslp.large_scale_embeddings.render_pca.get_render_jobs", fake_get_render_jobs
    )

    enqueued: dict[str, object] = {}

    class FakeQueueApi:
        def get(self, name: str) -> object:
            return object()

        def list_entries(self, queue: object) -> list:
            return []

        def list_workers(self, queue: object) -> list:
            return []

    class FakeBeaker:
        queue = FakeQueueApi()

        def __enter__(self) -> "FakeBeaker":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        @classmethod
        def from_env(cls, **kwargs: object) -> "FakeBeaker":
            return cls()

    monkeypatch.setattr("beaker.Beaker", FakeBeaker)

    def fake_write_jobs(queue_name, project, workflow, batch) -> None:
        enqueued["workflow"] = workflow
        enqueued["count"] = len(batch)

    def fake_launch_workers(**kwargs: object) -> None:
        enqueued["launched"] = kwargs.get("num_workers")
        enqueued["gpus"] = kwargs.get("gpus")

    monkeypatch.setattr("rslp.common.worker.write_jobs", fake_write_jobs)
    monkeypatch.setattr("rslp.common.worker.launch_workers", fake_launch_workers)

    class Result:
        value = -1

    result = Result()
    sup._run_cycle(
        {
            "queue_name": "user/queue",
            "num_workers": 2,
            "stale_seconds": 900,
            "years": [2024, 2025],
            "stage": sup.STAGE_RENDER_PCA,
            "store_path": "gs://bucket/s2.zarr",
            "artifact_path": "gs://bucket/pca",
            "pca_completed_path": "gs://bucket/pca_completed/",
            "completed_path_template": "gs://bucket/s2_{year}_completed/",
            "patch_size": 1,
            "image_name": "user/image",
            "cluster": ["ai2/cluster"],
            "gpus": 0,
            "shared_memory": "64GiB",
            "priority": "normal",
            "weka_bucket": "b",
            "weka_mount_path": "/m",
            "datasets_api_url": "https://example.invalid",
            "datasets_token_secret": "SECRET",  # noqa: S106
        },
        result,
    )

    # Both years' marker directories are handed to the render enumerator.
    assert calls["source_completed_paths"] == [
        "gs://bucket/s2_2024_completed/",
        "gs://bucket/s2_2025_completed/",
    ]
    assert calls["completed_path"] == "gs://bucket/pca_completed/"
    assert calls["artifact_path"] == "gs://bucket/pca"
    # Entries name the render workflow, not predict.
    assert result.value == 2
    assert enqueued["workflow"] == sup.STAGE_RENDER_PCA
    assert enqueued["count"] == 2
    # The render stage asks for no GPUs.
    assert enqueued["gpus"] == 0
