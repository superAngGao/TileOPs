"""Construction contract for fixed-shape BSHD prefill kernels."""

from typing import Optional

import torch

from ..kernel_base import Kernel

__all__ = ["DensePrefillKernel"]


class DensePrefillKernel(Kernel):
    """Static configuration shared by native Dense prefill implementations."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        window_size_left: int = -1,
        window_size_right: int = -1,
        fuse_rope: bool = False,
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if is_causal and max_seqlen_q > max_seqlen_kv:
            raise ValueError("causal Dense prefill requires max_seqlen_q <= max_seqlen_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.fuse_rope = fuse_rope
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self._validate_spec()
        self._build_program()
        self.init_config(config, tune)

    def _validate_spec(self) -> None:
        """Reject static semantics this implementation cannot honour."""

    def _build_program(self) -> None:
        raise NotImplementedError

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
