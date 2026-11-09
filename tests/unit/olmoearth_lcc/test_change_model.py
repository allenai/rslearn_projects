"""Tests for the two-pass LCC ChangeModel using a stub encoder."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import torch
import torch.nn as nn
from rslearn.models.component import TokenFeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage

from rslp.olmoearth_lcc.lcc_model.model import ChangeModel
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    LC_CLASS_NAMES,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from rslp.olmoearth_lcc.lcc_model.transforms import (
    EMBEDDING_KEY,
    EMBEDDING_NODATA,
    INPUT_KEY,
    TS_END_KEY,
    TS_START_KEY,
)

PATCH = 4
DIM = 16
T = 22
NUM_PASS1 = 11
CROP = 16


class StubEncoder(nn.Module):
    """Stand-in for OlmoEarth (token_pooling=False).

    Returns ``(B, DIM, H/P, W/P, T_in)`` tokens where every value encodes the
    sample's first pixel value (so we can identify which sample a token came
    from) plus the timestep index within the pass, so that the model's
    split/concat ordering can be verified.
    """

    patch_size = PATCH

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, context: ModelContext) -> TokenFeatureMaps:
        self.calls += 1
        feats = []
        for inp in context.inputs:
            image: RasterImage = inp[INPUT_KEY]
            c, t, h, w = image.image.shape
            sample_id = image.image[0, 0, 0, 0]
            steps = torch.arange(t, dtype=torch.float32)
            tokens = (sample_id + steps / 100).view(1, 1, 1, t)
            feats.append(tokens.expand(DIM, h // PATCH, w // PATCH, t))
        return TokenFeatureMaps([torch.stack(feats, dim=0)])


def _timestamps(n: int) -> list[tuple[datetime, datetime]]:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        (t0 + timedelta(days=30 * i), t0 + timedelta(days=30 * i + 1)) for i in range(n)
    ]


def _context(batch_size: int, history_years: int = 0) -> ModelContext:
    inputs = []
    for b in range(batch_size):
        image = torch.full((3, T, CROP, CROP), float(b + 1))
        inp: dict[str, Any] = {
            INPUT_KEY: RasterImage(image=image, timestamps=_timestamps(T))
        }
        if history_years:
            emb = torch.randint(-127, 128, (8, history_years, CROP, CROP)).float()
            # Sample 0: oldest year entirely nodata, one nodata pixel in the next.
            if b == 0:
                emb[:, 0] = EMBEDDING_NODATA
                emb[:, 1, 0, 0] = EMBEDDING_NODATA
            years = range(2020 - history_years, 2020)
            inp[EMBEDDING_KEY] = RasterImage(
                image=emb,
                timestamps=[
                    (
                        datetime(y, 1, 1, tzinfo=timezone.utc),
                        datetime(y + 1, 1, 1, tzinfo=timezone.utc),
                    )
                    for y in years
                ],
            )
        inputs.append(inp)
    return ModelContext(inputs=inputs, metadatas=[None] * batch_size)


def _model(**kwargs: Any) -> ChangeModel:
    defaults: dict[str, Any] = dict(
        num_timesteps=T,
        num_pass1=NUM_PASS1,
        embedding_dim=DIM,
        decoder_stages=[[(DIM, 3)], [(8, 3)], [(8, 3)]],
    )
    return ChangeModel(encoder=StubEncoder(), **{**defaults, **kwargs})


def _targets(batch_size: int) -> list[dict]:
    targets = []
    for _ in range(batch_size):
        binary = torch.ones(CROP, CROP, dtype=torch.long)
        binary[0, 0] = 2  # one change point
        valid = torch.ones(1, 1, CROP, CROP)
        ts_valid = torch.zeros(CROP, CROP)
        ts_valid[0, 0] = 1

        def seg(classes: torch.Tensor) -> dict:
            return {
                "classes": RasterImage(image=classes[None, None]),
                "valid": RasterImage(image=valid),
            }

        targets.append(
            {
                "binary": seg(binary),
                "src": seg(torch.ones(CROP, CROP, dtype=torch.long)),
                "dst": seg(torch.full((CROP, CROP), 2, dtype=torch.long)),
                "pre_change": seg(torch.ones(CROP, CROP, dtype=torch.long)),
                "post_change": seg(torch.ones(CROP, CROP, dtype=torch.long)),
                TS_START_KEY: {
                    "classes": RasterImage(image=torch.full((1, 1, CROP, CROP), 8)),
                    "valid": RasterImage(image=ts_valid[None, None]),
                },
                TS_END_KEY: {
                    "classes": RasterImage(image=torch.full((1, 1, CROP, CROP), 12)),
                    "valid": RasterImage(image=ts_valid[None, None]),
                },
            }
        )
    return targets


def test_two_pass_features_single_batched_call_and_ordering() -> None:
    """Both halves are encoded in one call and re-assembled per sample in order."""
    model = _model()
    context = _context(batch_size=3)
    feature = model._per_timestep_features(context)

    assert model.encoder.calls == 1
    assert feature.shape == (3, DIM, CROP // PATCH, CROP // PATCH, T)
    for b in range(3):
        sample_id = float(b + 1)
        # First half: pass1 timesteps 0..10 of this sample.
        expected1 = sample_id + torch.arange(NUM_PASS1) / 100
        # Second half: pass2 timesteps 0..10 of this sample (indices restart).
        expected2 = sample_id + torch.arange(T - NUM_PASS1) / 100
        torch.testing.assert_close(feature[b, 0, 0, 0, :NUM_PASS1], expected1)
        torch.testing.assert_close(feature[b, 0, 0, 0, NUM_PASS1:], expected2)


def test_single_pass_features() -> None:
    """num_pass1=None encodes the full stack in one call without a split."""
    model = _model(num_pass1=None)
    context = _context(batch_size=3)
    feature = model._per_timestep_features(context)

    assert model.encoder.calls == 1
    assert feature.shape == (3, DIM, CROP // PATCH, CROP // PATCH, T)
    for b in range(3):
        # The stub numbers steps within a pass, so a single pass runs 0..T-1
        # with no restart at NUM_PASS1.
        expected = float(b + 1) + torch.arange(T) / 100
        torch.testing.assert_close(feature[b, 0, 0, 0], expected)


@pytest.mark.parametrize("num_pass1", [NUM_PASS1, None])
def test_rejects_wrong_timestep_count(num_pass1: int | None) -> None:
    """A stack that does not match num_timesteps is an error in both modes."""
    model = _model(num_pass1=num_pass1)
    image = torch.zeros(3, T - 1, CROP, CROP)
    context = ModelContext(
        inputs=[{INPUT_KEY: RasterImage(image=image, timestamps=_timestamps(T - 1))}],
        metadatas=[None],
    )
    with pytest.raises(ValueError, match="timesteps"):
        model._per_timestep_features(context)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"binary_mode": "breakpoint"},
        {"binary_mode": "breakpoint", "season_embed": True},
        {"temporal_aggregation": "diff"},
        {
            "temporal_depth": 1,
            "temporal_heads": 2,
            "temporal_pos_enc": True,
            "temporal_aggregation": "attn",
        },
        # bpcat: pre/post change decoders read concat(before, after).
        {"binary_mode": "breakpoint", "change_category_input": "breakpoint"},
        # bpcat_temporal: breakpoint scan over temporally contextualized tokens.
        {
            "binary_mode": "breakpoint",
            "change_category_input": "breakpoint",
            "temporal_depth": 1,
            "temporal_heads": 2,
            "temporal_pos_enc": True,
        },
        # bpbin: breakpoint binary only; src/dst/pre/post on the attn-pooled feature.
        {
            "binary_mode": "breakpoint",
            "src_dst_input": "pooled",
            "temporal_depth": 1,
            "temporal_heads": 2,
            "temporal_pos_enc": True,
            "temporal_aggregation": "attn",
        },
        # bpcat_1pass: single encoder pass over the full stack.
        {
            "num_pass1": None,
            "binary_mode": "breakpoint",
            "change_category_input": "breakpoint",
        },
        # bpts: timestep heads from the breakpoint scan.
        {
            "binary_mode": "breakpoint",
            "change_category_input": "breakpoint",
            "ts_input": "breakpoint",
        },
        {
            "binary_mode": "breakpoint",
            "change_category_input": "breakpoint",
            "ts_input": "breakpoint",
            "temporal_depth": 1,
            "temporal_heads": 2,
            "temporal_time_enc": True,
        },
    ],
)
def test_forward_shapes_and_losses(kwargs: dict) -> None:
    """Forward produces full-resolution outputs and all losses in every mode."""
    model = _model(**kwargs)
    model.eval()
    batch_size = 2
    out = model(_context(batch_size), _targets(batch_size))

    assert set(out.loss_dict) == {
        "binary_cls",
        "src_cls",
        "dst_cls",
        "pre_change_cls",
        "post_change_cls",
        "start_ce",
        "end_ce",
    }
    for loss in out.loss_dict.values():
        assert torch.isfinite(loss)

    assert len(out.outputs) == batch_size
    o = out.outputs[0]
    assert o["binary"].shape == (3, CROP, CROP)
    assert o["src"].shape == (len(LC_CLASS_NAMES), CROP, CROP)
    assert o["dst"].shape == (len(LC_CLASS_NAMES), CROP, CROP)
    assert o["pre_change"].shape == (len(PRE_CHANGE_CATEGORY_NAMES), CROP, CROP)
    assert o["post_change"].shape == (len(POST_CHANGE_CATEGORY_NAMES), CROP, CROP)
    assert o[TS_START_KEY].shape == (T, CROP, CROP)
    assert o[TS_END_KEY].shape == (T, CROP, CROP)
    assert len(o["timestep_days"]) == T
    torch.testing.assert_close(
        o["binary"].sum(dim=0), torch.ones(CROP, CROP), atol=1e-5, rtol=0
    )


def test_invalid_options() -> None:
    """Option validation."""
    with pytest.raises(ValueError, match="num_pass1"):
        ChangeModel(
            encoder=StubEncoder(),
            num_timesteps=T,
            num_pass1=T,
            embedding_dim=DIM,
            decoder_stages=[[(DIM, 3)]],
        )
    with pytest.raises(ValueError, match="binary_mode"):
        _model(binary_mode="centered")
    with pytest.raises(ValueError, match="temporal_pos_enc"):
        _model(temporal_pos_enc=True)
    with pytest.raises(ValueError, match="src_dst_input"):
        _model(src_dst_input="concat")
    with pytest.raises(ValueError, match="change_category_input"):
        _model(change_category_input="concat")
    # Breakpoint features only exist in breakpoint mode.
    with pytest.raises(ValueError, match="src_dst_input"):
        _model(src_dst_input="breakpoint")
    with pytest.raises(ValueError, match="ts_input"):
        _model(ts_input="breakpoint")
    with pytest.raises(ValueError, match="ts_input"):
        _model(binary_mode="breakpoint", ts_input="split")
    with pytest.raises(ValueError, match="change_category_input"):
        _model(change_category_input="breakpoint")
    with pytest.raises(ValueError, match="loss_keys"):
        _model(loss_keys=["post_change"])
    with pytest.raises(ValueError, match="loss_keys"):
        _model(loss_keys=[])


def test_loss_keys_trains_only_selected_heads() -> None:
    """loss_keys drops the other losses, so their heads get no gradient."""
    model = _model(
        binary_mode="breakpoint",
        change_category_input="breakpoint",
        loss_keys=["pre_change_cls", "post_change_cls"],
    )
    out = model(_context(2), _targets(2))
    assert set(out.loss_dict) == {"pre_change_cls", "post_change_cls"}
    sum(out.loss_dict.values()).backward()
    assert all(p.grad is None for p in model.decoder_src.parameters())
    assert all(p.grad is None for p in model.start_head.parameters())
    assert all(p.grad is not None for p in model.decoder_post_change.parameters())


def test_feature_routing_options() -> None:
    """The routing options resolve and size the decoders as documented."""
    # Default: src/dst follow binary_mode; categories use the pooled feature.
    assert _model().src_dst_input == "pooled"
    assert _model(binary_mode="breakpoint").src_dst_input == "breakpoint"
    assert _model(binary_mode="breakpoint").change_category_input == "pooled"
    # Explicit decoupling of src/dst from the breakpoint.
    m = _model(binary_mode="breakpoint", src_dst_input="pooled")
    assert m.src_dst_input == "pooled"
    # concat(before, after) doubles the change-category decoder input width.
    m = _model(binary_mode="breakpoint", change_category_input="breakpoint")
    assert m.decoder_pre_change[0].in_channels == 2 * DIM
    assert m.decoder_post_change[0].in_channels == 2 * DIM
    assert m.decoder_src[0].in_channels == DIM
    assert _model().decoder_pre_change[0].in_channels == DIM


HISTORY = dict(
    num_pass1=None,
    binary_mode="breakpoint",
    change_category_input="breakpoint",
    history_years=3,
    history_input_dim=8,
    history_hidden=8,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(history_context=True),
        dict(temporal_depth=1, temporal_heads=2, temporal_time_enc=True),
        dict(
            history_memory=True,
            temporal_depth=1,
            temporal_heads=2,
            temporal_time_enc=True,
        ),
        dict(
            history_context=True,
            history_memory=True,
            temporal_depth=1,
            temporal_heads=2,
            temporal_time_enc=True,
        ),
        dict(
            history_memory=True,
            history_in_scan=True,
            temporal_depth=1,
            temporal_heads=2,
            temporal_time_enc=True,
        ),
        dict(
            history_memory=True,
            history_in_scan=True,
            temporal_depth=1,
            temporal_heads=2,
            temporal_time_enc=True,
            binary_mode="breakpoint",
            ts_input="breakpoint",
        ),
    ],
)
def test_history_forward(kwargs: dict) -> None:
    """Every annual-embedding mode runs, with finite losses, in train and eval."""
    uses_history = kwargs.get("history_context") or kwargs.get("history_memory")
    args = {**HISTORY, **kwargs}
    if not uses_history:
        args = {k: v for k, v in args.items() if not k.startswith("history")}
    k = args.get("history_years", 0)
    n_ts = T + (k if kwargs.get("history_in_scan") else 0)
    model = _model(**args)
    batch_size = 2
    for train in (True, False):
        model.train(train)
        targets = _targets(batch_size)
        out = model(_context(batch_size, k), targets)
        for loss in out.loss_dict.values():
            assert torch.isfinite(loss)
        o = out.outputs[0]
        assert o[TS_START_KEY].shape == (n_ts, CROP, CROP)
        assert len(o["timestep_days"]) == n_ts
        for value in o.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()


def test_history_all_nodata_uses_null_context() -> None:
    model = _model(**HISTORY, history_context=True)
    model.eval()
    context = _context(1, 3)
    context.inputs[0][EMBEDDING_KEY].image[:] = EMBEDDING_NODATA
    feat, valid = model._history_features(context, CROP // PATCH, CROP // PATCH)
    assert not valid.any()
    ctx = model._history_context_vector(feat, valid)
    expected = model.hist_null[None, :, None, None].expand_as(ctx)
    torch.testing.assert_close(ctx, expected)


def test_history_invalid_options() -> None:
    with pytest.raises(ValueError):
        _model(**{**HISTORY, "history_years": 0}, history_context=True)
    with pytest.raises(ValueError):
        _model(**HISTORY)  # history_years without a mode
    with pytest.raises(ValueError):
        _model(**HISTORY, history_memory=True)  # needs temporal_depth
    with pytest.raises(ValueError):
        _model(**HISTORY, history_context=True, history_in_scan=True)
    with pytest.raises(ValueError):
        _model(temporal_depth=1, temporal_heads=2, temporal_pos_enc=True, temporal_time_enc=True)


def test_breakpoint_ts_logits_index_mapping() -> None:
    """With zero offsets, ts_start index t is split t and ts_end index t is split t-1."""
    model = _model(binary_mode="breakpoint", ts_input="breakpoint")
    assert not hasattr(model, "start_head")
    nn.init.zeros_(model.ts_split_offset.weight)
    nn.init.zeros_(model.ts_split_offset.bias)
    B, H, W = 2, 3, 3
    feature = torch.randn(B, DIM, H, W, T)
    _, _, _, scores, hidden = model._breakpoint_features(feature, return_splits=True)
    assert scores.shape == (B, T - 1, H, W)
    assert hidden.shape[:2] == (B, T - 1)
    start, end = model._breakpoint_ts_logits(feature, scores, hidden)
    assert start.shape == end.shape == (B, T, H, W)
    torch.testing.assert_close(start[:, : T - 1], scores)
    torch.testing.assert_close(end[:, 1:], scores)
    last = model.ts_start_last(feature[..., -1].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    torch.testing.assert_close(start[:, T - 1 :], last)
