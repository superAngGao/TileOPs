"""Cold-cache benchmark for register-dequant TMA ping-pong W4A16 GEMV."""

from __future__ import annotations

import argparse
import sys
import types

import torch
from tilelang.quantize import interleave_weight

try:
    import pytest as _pytest  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pytest"] = types.ModuleType("pytest")

from benchmarks.benchmark_base import bench_kernel
from experiments.w4a16_gemm_feasibility import (
    quantize_weight_int4,
    w4a16_gemv_kernel,
    w4a16_gemv_tma_kernel,
)


def run_shape(m: int, n: int, k: int) -> None:
    if m != 1:
        raise ValueError("The ping-pong experiment requires M=1")

    torch.manual_seed(0)
    activation = torch.randn((m, k), device="cuda", dtype=torch.float16)
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
    expected = activation @ dequantized.half().T

    variants = [
        (
            "direct-split8-n1-o4",
            w4a16_gemv_kernel(m, n, k, "float16")(
                n_partition=1,
                split_k_warps=8,
                outputs_per_warp=4,
            ),
        ),
        (
            "pingpong-3wg-split2-o16-s2",
            w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=2,
                outputs_per_warp=16,
                split_k_warps=2,
                consumer_warpgroups=2,
            ),
        ),
        (
            "pingpong-3wg-split2-o16-s3",
            w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=3,
                outputs_per_warp=16,
                split_k_warps=2,
                consumer_warpgroups=2,
            ),
        ),
        (
            "pingpong-5wg-split2-o8-s2",
            w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=2,
                outputs_per_warp=8,
                split_k_warps=2,
                consumer_warpgroups=4,
            ),
        ),
        (
            "pingpong-5wg-split2-o8-s3",
            w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=3,
                outputs_per_warp=8,
                split_k_warps=2,
                consumer_warpgroups=4,
            ),
        ),
        (
            "pingpong-5wg-split4-o16-s2",
            w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=2,
                outputs_per_warp=16,
                split_k_warps=4,
                consumer_warpgroups=4,
            ),
        ),
        (
            "pingpong-5wg-split4-o16-s3",
            w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=3,
                outputs_per_warp=16,
                split_k_warps=4,
                consumer_warpgroups=4,
            ),
        ),
    ]

    for tag, kernel in variants:
        actual = kernel(activation, packed, scale, zero)
        torch.testing.assert_close(
            actual,
            expected,
            atol=5e-2,
            rtol=3e-2,
        )
        source = kernel.get_kernel_source().lower()
        has_tma = "tma" in source or "cp.async.bulk.tensor" in source
        latency_ms = bench_kernel(
            kernel,
            args=(activation, packed, scale, zero),
        )
        print(
            f"shape=({m},{n},{k}) tag={tag} "
            f"tma={has_tma} latency_ms={latency_ms:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        metavar="M,N,K",
    )
    args = parser.parse_args()
    for value in args.shape:
        dims = tuple(int(dim) for dim in value.split(","))
        if len(dims) != 3:
            raise ValueError(f"--shape expects M,N,K, got {value!r}")
        run_shape(*dims)


if __name__ == "__main__":
    main()
