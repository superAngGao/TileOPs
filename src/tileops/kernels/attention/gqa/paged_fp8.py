"""Read-only paged FP8 storage with FP32 descales around 16-bit attention tiles."""

from typing import Optional

import torch

from tileops.kernels.attention.call_spec import ATTENTION_DTYPES, AttentionCall
from tileops.kernels.attention.gqa.paged import GQAPagedFwdKernel

__all__ = ["GQAPagedFP8Kernel"]


class GQAPagedFP8Kernel(GQAPagedFwdKernel):
    """Read FP8 cache values at each request's scale, with optional FP8 Q and RoPE.

    The shared split/unsplit scan multiplies 16-bit tiles of the stored values.
    Per-head scales commute with RoPE and are applied to FP32 scores/outputs, so
    scaling introduces no intermediate 16-bit rounding or overflow. The caller's
    page pool remains in FP8.
    """

    supported_archs: list[int] = [89, 90]

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        if call.cache_dtype != torch.float8_e4m3fn:
            return "requires an FP8 E4M3 cache"
        if call.dtype not in (*ATTENTION_DTYPES, torch.float8_e4m3fn):
            return "requires a 16-bit or FP8 E4M3 query"
        if (call.out_dtype or call.dtype) not in ATTENTION_DTYPES:
            return "requires a 16-bit output"
        if call.page_size <= 0:
            return "requires a positive page size"
        return call.tensor_core_dim_refusal
