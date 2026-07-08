"""Benchmark: TileOps DeltaNet inference prefill."""

from typing import Any

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import DeltaNetPrefillFwdOp
from workloads.deltanet import DeltaNetPrefillFwdTest

_OP_NAME = "DeltaNetPrefillFwdOp"


def _deltanet_prefill_args(workload: dict[str, Any]) -> tuple[int, int, int, int, int, int, str]:
    layout = workload.get("layout", "bthd")
    if layout == "bthd":
        batch, seq_len, heads, dim_k = workload["q_shape"]
        _, v_seq_len, v_heads, dim_v = workload["v_shape"]
    else:
        batch, heads, seq_len, dim_k = workload["q_shape"]
        _, v_heads, v_seq_len, dim_v = workload["v_shape"]
    if v_seq_len != seq_len or v_heads != heads:
        raise ValueError("DeltaNet prefill q_shape and v_shape must share seq_len and heads")
    return batch, heads, seq_len, dim_k, dim_v, workload.get("chunk_size", 64), layout


_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), _deltanet_prefill_args, tune=False)


@pytest.mark.parametrize(
    "batch, heads, seq_len, dim_k, dim_v, chunk_size, layout, dtype, tune",
    _BENCH_PARAMS,
)
def test_deltanet_prefill_fwd_bench(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    layout: str,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = DeltaNetPrefillFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    q, k, v, beta = test.gen_inputs()
    if layout == "bthd":
        inputs = (
            q.permute(0, 2, 1, 3).contiguous(),
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
            beta.permute(0, 2, 1).contiguous(),
        )
    else:
        inputs = (q, k, v, beta)

    op = DeltaNetPrefillFwdOp(
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        layout=layout,
        tune=tune,
    )
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

