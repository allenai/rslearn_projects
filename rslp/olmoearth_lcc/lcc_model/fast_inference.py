"""Compiled, cuDNN-attention inference for the LCC predict pipeline.

With fast_inference, the OlmoEarth encoder's transformer blocks are compiled with
torch.compile (via the rslearn wrapper's compile_model option) and attention uses the
cuDNN SDPA kernel (sdpa_backends). This matches eager outputs closely, runs at the
same speed as eager at batch size 8, and uses much less GPU memory (about 1.7 GB
instead of 16.5 GB). It has only been tested on H100 GPUs.

The options are passed as command-line overrides to rslearn model predict, so the
predict YAML config itself stays eager.
"""

import torch
import torch._inductor.config

from rslp.log_utils import get_logger

logger = get_logger(__name__)

# cuDNN SDPA together with torch.compile hits an inductor stride assert on torch 2.7.
MIN_TORCH_VERSION = "2.8"

ENCODER_ARG_PREFIX = "--model.init_args.model.init_args.encoder.init_args"
FAST_INFERENCE_ARGS = [
    f"{ENCODER_ARG_PREFIX}.compile_model=true",
    f"{ENCODER_ARG_PREFIX}.sdpa_backends=[cudnn]",
]


def prepare_fast_inference() -> list[str]:
    """Configure this process for compiled inference.

    This must be called in the same process as rslearn model predict, before the
    first compiled forward pass.

    Returns:
        the extra arguments to pass to rslearn model predict.
    """
    if torch.__version__ < MIN_TORCH_VERSION:
        raise RuntimeError(
            f"fast_inference requires torch>={MIN_TORCH_VERSION} but got "
            f"{torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("fast_inference requires a CUDA GPU")

    torch._inductor.config.max_autotune_gemm_backends = "ATEN"

    logger.info("using fast inference with args %s", FAST_INFERENCE_ARGS)
    return list(FAST_INFERENCE_ARGS)
