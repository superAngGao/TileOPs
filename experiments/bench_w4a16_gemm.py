"""Cold-cache benchmark for the local W4A16 feasibility kernels.

This script deliberately uses TileOps' production ``BenchmarkBase.profile``
path instead of ad-hoc CUDA-event timing.  That path flushes the full L2 before
every measured call, rotates cloned input addresses, and uses CUPTI to report
pure kernel time.

The benchmark remains local to ``experiments``: it does not declare a public
operator or manifest contract.
"""

from __future__ import annotations

import argparse
import sys
import types
from dataclasses import dataclass
from typing import Callable

import torch
from tilelang.quantize import interleave_weight

# The remote feasibility environment intentionally has only runtime
# dependencies.  benchmark_base imports pytest for manifest parametrization,
# but BenchmarkBase/bench_kernel do not use it.  Keep this standalone script
# runnable without changing the production benchmark implementation.
try:
    import pytest as _pytest  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pytest"] = types.ModuleType("pytest")

from benchmarks.benchmark_base import BenchmarkBase
from experiments.w4a16_gemm_feasibility import (
    quantize_weight_int4,
    w4a16_gemm_3wg_kernel,
    w4a16_gemv_kernel,
)
from tileops.ops import GemmOp


@dataclass(frozen=True)
class GemmWorkload:
    """Static metadata consumed by the experiment benchmark."""

    m: int
    n: int
    k: int
    dtype: torch.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.k)


class StaticGemmBenchmark(BenchmarkBase[GemmWorkload]):
    """Benchmark with an explicit logical-FLOP and per-implementation byte count."""

    def __init__(self, workload: GemmWorkload, memory_bytes: int) -> None:
        super().__init__(workload)
        self._memory_bytes = memory_bytes

    def calculate_flops(self) -> float:
        return float(2 * self.workload.m * self.workload.n * self.workload.k)

    def calculate_memory(self) -> float:
        return float(self._memory_bytes)


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _build_w4_kernel(
    workload: GemmWorkload,
    split_k_warps: int = 1,
    n_partition: int = 4,
) -> Callable:
    m, n, k = workload.shape
    if m == 1:
        return w4a16_gemv_kernel(m, n, k, "float16")(
            n_partition=n_partition,
            split_k_warps=split_k_warps,
        )
    return w4a16_gemm_3wg_kernel(m, n, k, "float16")(
        block_m=64,
        block_n=64,
        block_k=64,
        num_stages=2,
    )


def _format_result(tag: str, result: dict) -> str:
    return (
        f"{tag:14s} latency_ms={result['latency_ms']:.6f} "
        f"tflops={result['tflops']:.3f} "
        f"bandwidth_tbs={result['bandwidth_tbs']:.3f}"
    )


def run_workload(
    workload: GemmWorkload,
    gemv_configs: list[tuple[int, int]],
) -> None:
    if workload.dtype != torch.float16:
        raise ValueError("The current LOP3 W4A16 experiment supports float16 only.")

    m, n, k = workload.shape
    torch.manual_seed(0)
    activation = torch.randn((m, k), device="cuda", dtype=workload.dtype)
    source_weight = (
        torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.25
    )
    packed, scale, zero, dequantized = quantize_weight_int4(
        source_weight,
        quant_mode="affine",
    )
    packed = interleave_weight(
        packed,
        nbits=4,
        target_dtype="float16",
    ).view(torch.uint8)
    dense_weight = dequantized.to(workload.dtype)

    w4_kernels = [
        (
            split_k_warps,
            n_partition,
            _build_w4_kernel(
                workload,
                split_k_warps=split_k_warps,
                n_partition=n_partition,
            ),
        )
        for split_k_warps, n_partition in gemv_configs
    ]
    tileops_a16 = GemmOp(trans_a=False, trans_b=True)

    def torch_cublas(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs.T)

    expected = torch_cublas(activation, dense_weight)
    tileops_actual = tileops_a16(activation, dense_weight)
    tolerance = {"atol": 5e-2, "rtol": 3e-2}
    for _, _, w4_kernel in w4_kernels:
        w4_actual = w4_kernel(activation, packed, scale, zero)
        torch.testing.assert_close(w4_actual, expected, **tolerance)
    torch.testing.assert_close(tileops_actual, expected, **tolerance)
    torch.cuda.synchronize()

    output_bytes = m * n * workload.dtype.itemsize
    w4_memory = _tensor_bytes(activation, packed, scale, zero) + output_bytes
    a16_memory = _tensor_bytes(activation, dense_weight) + output_bytes

    w4_benchmark = StaticGemmBenchmark(workload, w4_memory)
    a16_benchmark = StaticGemmBenchmark(workload, a16_memory)

    w4_results = [
        (
            split_k_warps,
            n_partition,
            w4_benchmark.profile(
                w4_kernel,
                activation,
                packed,
                scale,
                zero,
            ),
        )
        for split_k_warps, n_partition, w4_kernel in w4_kernels
    ]
    tileops_result = a16_benchmark.profile(
        tileops_a16,
        activation,
        dense_weight,
    )
    cublas_result = a16_benchmark.profile(
        torch_cublas,
        activation,
        dense_weight,
    )

    print(f"shape=({m},{n},{k}) dtype=float16 cache=cold")
    for split_k_warps, n_partition, w4_result in w4_results:
        tag = f"w4-s{split_k_warps}-n{n_partition}"
        print(_format_result(tag, w4_result))
    print(_format_result("tileops-a16", tileops_result))
    print(_format_result("torch-cublas", cublas_result))
    for split_k_warps, n_partition, w4_result in w4_results:
        print(
            f"speedup s{split_k_warps}-n{n_partition}: "
            f"w4/tileops="
            f"{tileops_result['latency_ms'] / w4_result['latency_ms']:.3f}x "
            f"w4/cublas="
            f"{cublas_result['latency_ms'] / w4_result['latency_ms']:.3f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        metavar="M,N,K",
        help="Shape to benchmark; may be repeated. Defaults to four feasibility shapes.",
    )
    parser.add_argument(
        "--gemv-config",
        action="append",
        metavar="SPLIT_K_WARPS,N_PARTITION",
        help=(
            "M=1 GEMV launch configuration; may be repeated. "
            "Defaults to 1,4."
        ),
    )
    args = parser.parse_args()

    if args.gemv_config:
        gemv_configs = []
        for value in args.gemv_config:
            config = tuple(int(dim) for dim in value.split(","))
            if len(config) != 2:
                raise ValueError(
                    "--gemv-config expects SPLIT_K_WARPS,N_PARTITION, "
                    f"got {value!r}")
            gemv_configs.append(config)
    else:
        gemv_configs = [(1, 4)]

    if args.shape:
        shapes = []
        for value in args.shape:
            dims = tuple(int(dim) for dim in value.split(","))
            if len(dims) != 3:
                raise ValueError(f"--shape expects M,N,K, got {value!r}")
            shapes.append(dims)
    else:
        shapes = [
            (128, 2112, 7168),
            (128, 7168, 2048),
            (1, 7168, 2048),
            (1, 8192, 8192),
        ]

    for m, n, k in shapes:
        run_workload(
            GemmWorkload(m, n, k, torch.float16),
            gemv_configs,
        )


if __name__ == "__main__":
    main()
