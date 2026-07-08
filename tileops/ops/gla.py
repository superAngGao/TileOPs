from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.gla import GLABwdKernel, GLAFwdKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["GLABwdOp", "GLAFwdOp", "GLAPrefillFwdOp"]


class GLAFwdOp(Op):
    """GLA (Gated Linear Attention) forward operator.

    Chunked GLA forward: (q, k, v, g) -> (o, final_state).

    Layout: BTHD (batch, seq_len, heads, dim).

    Args:
        batch: Batch size.
        seq_len: Sequence length (must be divisible by chunk_size).
        heads: Number of attention heads.
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        scale: Query scale factor (default: dim_k**-0.5).
        output_final_state: Whether to return the final hidden state.
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        output_final_state: bool = False,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale
        self.output_final_state = output_final_state
        self.dtype = dtype

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        fwd_kernel_cls = self.kernel_map["GLAFwdKernel"]
        self.kernel = fwd_kernel_cls(
            batch, seq_len, heads, dim_k, dim_v, chunk_size,
            scale=scale,
            output_final_state=output_final_state,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLAFwdKernel": GLAFwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run GLA forward.

        Args:
            q: Query tensor [B, T, H, K].
            k: Key tensor [B, T, H, K].
            v: Value tensor [B, T, H, V].
            g: Log-space forget gates [B, T, H, K].
            initial_state: Optional initial hidden state [B, H, K, V].

        Returns:
            Tuple of (o, final_state). final_state is None if output_final_state=False.
        """
        return self.kernel(q, k, v, g, initial_state)


class GLAPrefillFwdOp(Op):
    """GLA inference prefill operator.

    Public serving contract: ``(q, k, v, g) -> (o, final_state)`` with a
    zero initial recurrent state. This keeps the manifest-facing entry free of
    optional initial-state and output-final-state switches.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        layout: str = "bthd",
    ) -> None:
        layout = self._normalize_layout(layout)
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale
        self.dtype = dtype
        self.layout = layout

        if seq_len % chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
            )

        self.dispatch_kernel(kernel_map)
        kernel_cls = self.kernel_map["GLAFwdKernel"]
        self.kernel = kernel_cls(
            batch,
            seq_len,
            heads,
            dim_k,
            dim_v,
            chunk_size,
            scale=scale,
            output_final_state=True,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLAFwdKernel": GLAFwdKernel,
        }

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        layout = layout.lower()
        if layout != "bthd":
            raise ValueError("GLAPrefillFwdOp currently supports layout='bthd' only")
        return layout

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape
        return {
            "o": (q_shape[0], q_shape[1], q_shape[2], v_shape[-1]),
            "final_state": (q_shape[0], q_shape[2], q_shape[-1], v_shape[-1]),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> None:
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
            if tensor.dtype != self.dtype:
                raise ValueError(f"{name}.dtype must be {self.dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> None:
        q_shape = (self.batch, self.seq_len, self.heads, self.dim_k)
        v_shape = (self.batch, self.seq_len, self.heads, self.dim_v)
        if tuple(q.shape) != q_shape:
            raise ValueError(f"q must have shape {q_shape}, got {tuple(q.shape)}")
        if tuple(k.shape) != q_shape:
            raise ValueError(f"k must have shape {q_shape}, got {tuple(k.shape)}")
        if tuple(v.shape) != v_shape:
            raise ValueError(f"v must have shape {v_shape}, got {tuple(v.shape)}")
        if tuple(g.shape) != q_shape:
            raise ValueError(f"g must have shape {q_shape}, got {tuple(g.shape)}")
        if not all(tensor.is_cuda for tensor in (q, k, v, g)):
            raise ValueError("q, k, v, and g must be CUDA tensors")

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import gla_prefill_fwd_roofline

        return gla_prefill_fwd_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        self._validate_dtypes(q, k, v, g)
        self._validate_shapes(q, k, v, g)
        o, final_state = self.kernel(q, k, v, g, None)
        return o, final_state.to(self.dtype)


class GLABwdOp(Op):
    """GLA (Gated Linear Attention) backward operator.

    Computes gradients (dq, dk, dv, dg) given output gradient do.

    Uses h_out saved from the forward pass (no recomputation needed).

    Layout: BTHD (batch, seq_len, heads, dim).

    Args:
        batch: Batch size.
        seq_len: Sequence length (must be divisible by chunk_size).
        heads: Number of attention heads.
        dim_k: Key/query dimension.
        dim_v: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        scale: Query scale factor (default: dim_k**-0.5).
        dtype: Data type for computation.
        kernel_map: Optional kernel overrides.
        tune: Whether to autotune kernels.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: float = -1.0,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale
        self.dtype = dtype

        assert seq_len % chunk_size == 0, (
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

        self.dispatch_kernel(kernel_map)

        bwd_kernel_cls = self.kernel_map["GLABwdKernel"]
        self.kernel = bwd_kernel_cls(
            batch, seq_len, heads, dim_k, dim_v, chunk_size,
            scale=scale,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "GLABwdKernel": GLABwdKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        h: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        has_initial_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run GLA backward.

        Args:
            q: Queries [B, T, H, K].
            k: Keys [B, T, H, K].
            v: Values [B, T, H, V].
            g: Log-space forget gates [B, T, H, K].
            h: Hidden states from forward [B, NT+1, H, K, V] (fp32).
            do: Output gradient [B, T, H, V].
            dht: Final-state gradient [B, H, K, V].
            has_initial_state: Whether initial_state was provided by the user.

        Returns:
            Tuple of (dq, dk, dv, dg).
        """
        return self.kernel(q, k, v, g, h, do, dht, has_initial_state)
