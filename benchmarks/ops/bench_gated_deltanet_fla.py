"""Benchmark TileOPs vs FLA for Gated DeltaNet: fwd, bwd, decode."""
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gated_deltanet_bwd import GatedDeltaNetBwdTest
from tests.ops.test_gated_deltanet_decode import GatedDeltaNetDecodeTest
from tests.ops.test_gated_deltanet_fwd import GatedDeltaNetFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetDecodeOp, GatedDeltaNetFwdOp

# =============================================================================
# Forward: TileOPs vs FLA
# =============================================================================


class GatedDeltaNetFwdFLABenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 2.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (2 * DK + 2 * DV + 2) * elem


class GatedDeltaNetFwdFLAFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 1024, 4, 64, 64, 32, torch.float32, False),
            (2, 2048, 4, 64, 64, 32, torch.float32, False),
            (2, 4096, 4, 64, 64, 32, torch.float32, False),
            (2, 1024, 4, 64, 64, 32, torch.float16, False),
            (2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
        ]),
    ]


@GatedDeltaNetFwdFLAFixture
def test_gated_deltanet_fwd_vs_fla(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetFwdFLABenchmark(test)
    inputs = test.gen_inputs()

    op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gated_deltanet_fwd_vs_fla", locals(), result, tag="tileops")

    q, k, v, g, beta = inputs
    result_fla = bm.profile(lambda: chunk_gated_delta_rule(q, k, v, g, beta))
    BenchmarkReport.record("gated_deltanet_fwd_vs_fla", locals(), result_fla, tag="fla")


# =============================================================================
# Backward: TileOPs (bwd-only) vs FLA (fwd+bwd)
# =============================================================================


class GatedDeltaNetBwdFLABenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        return 4.0 * B * H * S * DK * DV

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, S, DK, DV = t.batch, t.heads, t.seq_len, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * S * (4 * DK + 3 * DV + 4) * elem


class GatedDeltaNetBwdFLAFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            (2, 1024, 4, 64, 64, 32, torch.float32, False),
            (2, 2048, 4, 64, 64, 32, torch.float32, False),
            (2, 4096, 4, 64, 64, 32, torch.float32, False),
            (2, 1024, 4, 64, 64, 32, torch.float16, False),
            (2, 2048, 4, 64, 64, 32, torch.float16, False),
            (2, 4096, 4, 64, 64, 32, torch.float16, False),
            (2, 1024, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 2048, 4, 64, 64, 32, torch.bfloat16, False),
            (2, 4096, 4, 64, 64, 32, torch.bfloat16, False),
        ]),
    ]


@GatedDeltaNetBwdFLAFixture
def test_gated_deltanet_bwd_vs_fla(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    test = GatedDeltaNetBwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    bm = GatedDeltaNetBwdFLABenchmark(test)
    inputs = test.gen_inputs()
    do, q, k, v, g, beta, S_fwd = inputs

    # TileOPs: bwd-only
    bwd_op = GatedDeltaNetBwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    result = bm.profile(bwd_op.forward, *inputs)
    BenchmarkReport.record("gated_deltanet_bwd_vs_fla", locals(), result, tag="tileops")

    # FLA: fwd+bwd via autograd (needs grad context, so profile manually)
    q_fla = q.detach().requires_grad_(True)
    k_fla = k.detach().requires_grad_(True)
    v_fla = v.detach().requires_grad_(True)
    g_fla = g.detach().requires_grad_(True)
    beta_fla = beta.detach().requires_grad_(True)

    def fla_fwd_bwd():
        o_fla, _ = chunk_gated_delta_rule(q_fla, k_fla, v_fla, g_fla, beta_fla)
        o_fla.backward(do, retain_graph=True)

    from tilelang.profiler import do_bench
    latency = do_bench(fla_fwd_bwd, warmup=100, rep=100, backend='cupti')
    if latency <= 0:
        latency = do_bench(fla_fwd_bwd, warmup=100, rep=100, backend='event')
    result_fla = {"latency_ms": latency}
    BenchmarkReport.record("gated_deltanet_bwd_vs_fla", locals(), result_fla, tag="fla")


# =============================================================================
# Decode: TileOPs vs FLA (fused_recurrent with T=1)
# =============================================================================


class GatedDeltaNetDecodeFLABenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        return 2.0 * B * H * (2 * DK * DV + DK * DV + DK)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        B, H, DK, DV = t.batch, t.heads, t.dim_k, t.dim_v
        elem = t.dtype.itemsize
        return B * H * (2 * DK + DV + 2 + 2 * DK * DV + DV) * elem


class GatedDeltaNetDecodeFLAFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype", [
            (1, 32, 128, 128, torch.float16),
            (1, 32, 128, 128, torch.bfloat16),
            (8, 32, 128, 128, torch.float16),
            (8, 32, 128, 128, torch.bfloat16),
            (32, 32, 128, 128, torch.float16),
            (32, 32, 128, 128, torch.bfloat16),
            (64, 32, 128, 128, torch.float16),
            (64, 32, 128, 128, torch.bfloat16),
        ]),
    ]


@GatedDeltaNetDecodeFLAFixture
def test_gated_deltanet_decode_vs_fla(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
) -> None:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    test = GatedDeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    bm = GatedDeltaNetDecodeFLABenchmark(test)
    inputs = test.gen_inputs()
    q, k, v, g, beta, state = inputs

    # TileOPs
    op = GatedDeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("gated_deltanet_decode_vs_fla", locals(), result, tag="tileops")

    # FLA: fused_recurrent with T=1
    q_fla = q.unsqueeze(2)
    k_fla = k.unsqueeze(2)
    v_fla = v.unsqueeze(2)
    g_fla = g.unsqueeze(2)
    beta_fla = beta.unsqueeze(2)
    state_fla = state.float()

    def fla_decode():
        return fused_recurrent_gated_delta_rule(
            q_fla, k_fla, v_fla, g_fla, beta=beta_fla,
            initial_state=state_fla, output_final_state=True,
        )

    result_fla = bm.profile(fla_decode)
    BenchmarkReport.record("gated_deltanet_decode_vs_fla", locals(), result_fla, tag="fla")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
