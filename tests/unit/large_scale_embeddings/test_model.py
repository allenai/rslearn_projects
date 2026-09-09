"""Unit tests for rslp.large_scale_embeddings.model."""

import torch
from rslearn.models.component import FeatureMaps

from rslp.large_scale_embeddings.model import (
    NODATA_VALUE,
    QuantizedEmbeddingHead,
    dequantize_embeddings,
    quantize_embeddings,
)


def test_quantize_known_values() -> None:
    """Quantization matches hardcoded expected values."""
    values = torch.tensor([0.0, 1.0, -1.0, 0.25, -0.25, 2.0, -2.0])
    # sqrt(0.25) * 127.5 = 63.75 which rounds to 64; values beyond +/-1 saturate at
    # +/-127.
    expected = torch.tensor([0, 127, -127, 64, -64, 127, -127], dtype=torch.int8)
    assert torch.equal(quantize_embeddings(values), expected)


def test_quantize_never_produces_nodata() -> None:
    """The quantizer never emits the reserved nodata value (-128)."""
    values = torch.linspace(-100, 100, 100001)
    quantized = quantize_embeddings(values)
    assert quantized.dtype == torch.int8
    assert int(quantized.min()) >= -127
    assert int(quantized.max()) <= 127
    assert not (quantized == NODATA_VALUE).any()


def test_quantize_roundtrip() -> None:
    """Dequantizing recovers values in [-1, 1] within the quantization error bound."""
    values = torch.linspace(-1, 1, 10001)
    recovered = dequantize_embeddings(quantize_embeddings(values))
    # In the square-root domain, the rounding error is at most 0.5/127.5, plus the
    # clamp of 127.5 to 127 at the extremes. Mapping back through the square gives a
    # worst-case error of about 2 * (1/127.5) in the value domain.
    assert (recovered - values).abs().max() < 0.02


def test_head_quantizes_and_normalizes() -> None:
    """The head L2-normalizes per pixel then quantizes to int8."""
    # One sample with C=2, H=1, W=1 whose embedding vector is (3, 4) with norm 5.
    features = torch.tensor([[[[3.0]], [[4.0]]]])
    head = QuantizedEmbeddingHead(l2_normalize=True)
    output = head(FeatureMaps([features]), context=None)
    # Normalized vector is (0.6, 0.8); sqrt(0.6)*127.5 = 98.76 -> 99 and
    # sqrt(0.8)*127.5 = 114.03 -> 114.
    expected = torch.tensor([[[[99]], [[114]]]], dtype=torch.int8)
    assert output.outputs.dtype == torch.int8
    assert torch.equal(output.outputs, expected)


def test_head_without_normalization() -> None:
    """With l2_normalize disabled, the head quantizes the raw values."""
    features = torch.tensor([[[[0.25]], [[-1.0]]]])
    head = QuantizedEmbeddingHead(l2_normalize=False)
    output = head(FeatureMaps([features]), context=None)
    expected = torch.tensor([[[[64]], [[-127]]]], dtype=torch.int8)
    assert torch.equal(output.outputs, expected)
