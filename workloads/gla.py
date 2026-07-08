import torch

from workloads.workload_base import WorkloadBase


class GLADecodeTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        scale: float = -1.0,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.scale = scale

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        gk = -torch.rand(B, H, DK, device="cuda", dtype=self.dtype)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, gk, state


class GLAPrefillFwdTest(WorkloadBase):
    """Inference prefill workload for GLA in BTHD layout."""

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int,
        dtype: torch.dtype,
        scale: float = -1.0,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.scale = scale
        self.shape = (batch, seq_len, heads, dim_k)

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, T, H, K, V = self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, T, H, K, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, T, H, K, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, T, H, V, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, T, H, K, device="cuda", dtype=self.dtype)
        return q, k, v, g
