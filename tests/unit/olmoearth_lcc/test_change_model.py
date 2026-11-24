"""Tests for the LCC ChangeModel using a stub encoder."""

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
    INPUT_KEY,
    TS_END_KEY,
    TS_START_KEY,
)

PATCH = 4
DIM = 16
T = 22
CROP = 16


class StubEncoder(nn.Module):
    """Stand-in for OlmoEarth (token_pooling=False).

    Returns ``(B, DIM, H/P, W/P, T_in)`` tokens where every value encodes the
    sample's first pixel value (so we can identify which sample a token came
    from) plus the timestep index, so that the token ordering can be verified.
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


def _context(batch_size: int) -> ModelContext:
    inputs = []
    for b in range(batch_size):
        image = torch.full((3, T, CROP, CROP), float(b + 1))
        inputs.append({INPUT_KEY: RasterImage(image=image, timestamps=_timestamps(T))})
    return ModelContext(inputs=inputs, metadatas=[None] * batch_size)


def _model(**kwargs: Any) -> ChangeModel:
    defaults: dict[str, Any] = dict(
        num_timesteps=T,
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


def test_single_pass_features() -> None:
    """The full stack is encoded in one call, keeping each sample's tokens in order."""
    model = _model()
    context = _context(batch_size=3)
    feature = model._per_timestep_features(context)

    assert model.encoder.calls == 1
    assert feature.shape == (3, DIM, CROP // PATCH, CROP // PATCH, T)
    for b in range(3):
        expected = float(b + 1) + torch.arange(T) / 100
        torch.testing.assert_close(feature[b, 0, 0, 0], expected)


def test_rejects_wrong_timestep_count() -> None:
    """A stack that does not match num_timesteps is an error."""
    model = _model()
    image = torch.zeros(3, T - 1, CROP, CROP)
    context = ModelContext(
        inputs=[{INPUT_KEY: RasterImage(image=image, timestamps=_timestamps(T - 1))}],
        metadatas=[None],
    )
    with pytest.raises(ValueError, match="timesteps"):
        model._per_timestep_features(context)


def test_forward_shapes_and_losses() -> None:
    """Forward produces full-resolution outputs and all losses."""
    model = _model()
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
    with pytest.raises(ValueError, match="decoder_stages"):
        _model(decoder_stages=[[(DIM, 3)]])
    with pytest.raises(ValueError, match="num_classes_binary"):
        _model(num_classes_binary=2)


def test_decoder_input_widths() -> None:
    """src/dst read one aggregate; the change categories read concat(before, after)."""
    m = _model()
    assert m.decoder_src[0].in_channels == DIM
    assert m.decoder_dst[0].in_channels == DIM
    assert m.decoder_pre_change[0].in_channels == 2 * DIM
    assert m.decoder_post_change[0].in_channels == 2 * DIM
