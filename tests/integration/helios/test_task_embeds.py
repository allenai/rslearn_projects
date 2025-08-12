"""Testing task-conditioned Helios model."""

from pathlib import Path

import torch
import pytest

from rslp.helios.model import Helios, TaskConditionedHelios


@pytest.fixture
def patch_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent real file I/O: make Helios.load_model a no-op."""
    def fake_load_model(self, *args, **kwargs):
        # set a harmless placeholder; Helios.forward will be monkeypatched in tests that call it
        self.model = None
        return None

    monkeypatch.setattr(Helios, "load_model", fake_load_model, raising=True)


def _dict_embeds(tasks, dim, base=0.0):
    """Build dict: task -> 1D tensor[dim]."""
    return {t: (torch.arange(dim, dtype=torch.float32) + float(i) + base)
            for i, t in enumerate(tasks)}


def test_learned_init(tmp_path: Path, patch_load_model: None) -> None:
    """Learned task embeddings initialization from pretrained (dict) and indexing."""
    tasks = ["det", "seg", "depth"]
    dim = 4
    d = _dict_embeds(tasks, dim)
    ckpt = tmp_path / "embeds.pt"
    torch.save(d, ckpt)

    model = TaskConditionedHelios(
        checkpoint_path="unused",  # load_model is patched to no-op
        task_embed_opts={"dim": dim, "type": "learned", "tasks": tasks, "path": str(ckpt)},
    )

    # Expect weights stacked in the file’s (and thus self.tasks) order
    expect = torch.stack([d[t] for t in model.tasks], dim=0)
    w = model.task_embed_table.weight.detach().cpu()
    assert w.shape == (len(model.tasks), dim)
    assert torch.allclose(w, expect)

    # compute_task_embeds coerces dtype & device; indexing correct
    idx = torch.tensor([0.0, 2.0])  # float on purpose to test cast->long
    out = model.compute_task_embeds(idx)
    assert out.shape == (2, dim)
    assert torch.allclose(out.cpu(), expect[[0, 2]])


def test_precomputed(tmp_path: Path, patch_load_model: None) -> None:
    """Precomputed task embeddings freezing and indexing using dict format."""
    dim = 3
    d = {"a": torch.tensor([1.0, 2.0, 3.0]),
         "b": torch.tensor([4.0, 5.0, 6.0])}
    ckpt = tmp_path / "embeds.pt"
    torch.save(d, ckpt)

    # tasks/dim from file; values in task_embed_opts are ignored for these fields
    model = TaskConditionedHelios(
        checkpoint_path="unused",
        task_embed_opts={"dim": dim, "type": "precomputed", "tasks": ["x","y"], "path": str(ckpt)},
    )

    # Frozen params
    assert all(p.requires_grad is False for p in model.task_embed_table.parameters())

    # Indexing
    expect_full = torch.stack([d[t] for t in model.tasks], dim=0)
    idx = torch.tensor([1, 0, 1])
    out = model.compute_task_embeds(idx)
    assert out.shape == (3, dim)
    assert torch.allclose(out.cpu(), expect_full[[1, 0, 1]])

    # Even if frozen, weights can be zeroed with no_grad; verify effect
    with torch.no_grad():
        model.task_embed_table.weight.zero_()
    out2 = model.compute_task_embeds(idx)
    assert torch.all(out2 == 0)


def test_forward_restore(patch_load_model: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward sets and restores task embedding in forward_kwargs."""
    # Patch Helios.forward to assert the task_emb is present
    def fake_forward(self, inputs):
        assert "task_emb" in self.forward_kwargs
        return [torch.tensor(float(len(inputs)))]

    monkeypatch.setattr(Helios, "forward", fake_forward, raising=True)

    tasks = ["x", "y"]
    model = TaskConditionedHelios(
        checkpoint_path="unused",
        task_embed_opts={"dim": 2, "type": "learned", "tasks": tasks},
    )
    # Pre-set a different value to ensure it's restored after forward()
    model.forward_kwargs["task_emb"] = "SENTINEL"

    inputs = [{"dataset_source": "x"}, {"dataset_source": "x"}]
    out = model.forward(inputs)
    assert isinstance(out, list) and out and torch.is_tensor(out[0])
    # Ensure restore happened
    assert model.forward_kwargs.get("task_emb") == "SENTINEL"


def test_forward_matches(patch_load_model: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """If task embeddings are zeroed, TCH should match Helios output (same super())."""
    # Single stub for both: return batch-size tensor, ignore task_emb contents
    def simple_forward(self, inputs):
        return [torch.tensor(float(len(inputs)))]

    monkeypatch.setattr(Helios, "forward", simple_forward, raising=True)

    base = Helios(checkpoint_path="unused")
    tch = TaskConditionedHelios(
        checkpoint_path="unused",
        task_embed_opts={"dim": 4, "type": "learned", "tasks": ["x", "y"]},
    )
    with torch.no_grad():
        tch.task_embed_table.weight.zero_()

    inputs = [{"dataset_source": "x"}, {"dataset_source": "x"}, {"dataset_source": "x"}]
    assert torch.allclose(base.forward(inputs)[0], tch.forward(inputs)[0])


def test_invalid_opts(tmp_path: Path, patch_load_model: None) -> None:
    """Invalid task embed opts and malformed dict entries."""
    # Bad type
    with pytest.raises(ValueError):
        TaskConditionedHelios(
            checkpoint_path="unused",
            task_embed_opts={"dim": 4, "type": "nope", "tasks": ["a"]},
        )
    # Precomputed without path
    with pytest.raises(ValueError):
        TaskConditionedHelios(
            checkpoint_path="unused",
            task_embed_opts={"dim": 2, "type": "precomputed", "tasks": ["a", "b"]},
        )
    # Dict entry with wrong vector length (should raise during load_pretrained_embeds)
    tasks, dim = ["a", "b", "c"], 2
    bad = {"a": torch.randn(dim), "b": torch.randn(dim + 1), "c": torch.randn(dim)}
    ckpt = tmp_path / "bad.pt"
    torch.save(bad, ckpt)
    with pytest.raises(ValueError):
        TaskConditionedHelios(
            checkpoint_path="unused",
            task_embed_opts={"dim": dim, "type": "learned", "tasks": tasks, "path": str(ckpt)},
        )
