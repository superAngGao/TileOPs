"""Cold-cache comparison of scalar GEMV and asynchronous WGMMA for M=1.

This intentionally reuses the experimental explicit 2-WG and 3-WG W4A16
pipelines.  Both issue Hopper WGMMA asynchronously after LOP3 dequantization;
the experiment answers whether that compute pipeline can overcome the minimum
64-row WGMMA tile when only one logical row is requested.
"""

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
    w4a16_gemm_kernel,
    w4a16_gemm_2wg_kernel,
    w4a16_gemm_3wg_kernel,
    w4a16_decode_wgmma_nt_kernel,
    w4a16_decode_wgmma_rs_kernel,
    w4a16_gemv_kernel,
)


def run_shape(m: int, n: int, k: int) -> None:
    if m != 1:
        raise ValueError("This experiment is intentionally scoped to M=1")

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
    expected = torch.matmul(activation, dequantized.half().T)

    split_k_warps = 1 if k <= 2048 else (2 if k <= 4096 else 4)
    n_partition = 4 if split_k_warps == 1 else (2 if k != 8192 else 1)
    variants = [
        (
            f"gemv-s{split_k_warps}-n{n_partition}",
            w4a16_gemv_kernel(m, n, k, "float16")(
                n_partition=n_partition,
                split_k_warps=split_k_warps,
            ),
        ),
        (
            "mma-m16-n64-k64",
            w4a16_gemm_kernel(
                m,
                n,
                k,
                "float16",
                decode_mode="lop3",
                force_mma=True,
            )(
                block_m=16,
                block_n=64,
                block_k=64,
                num_stages=2,
                threads=128,
            ),
        ),
        (
            "mma-m16-n128-k64",
            w4a16_gemm_kernel(
                m,
                n,
                k,
                "float16",
                decode_mode="lop3",
                force_mma=True,
            )(
                block_m=16,
                block_n=128,
                block_k=64,
                num_stages=2,
                threads=256,
            ),
        ),
        (
            "wgmma-nt-3wg-k128-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=2,
                wgmma_depth=0,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k128-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=2,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k64-s2-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=64,
                num_stages=2,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k128-s3-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=3,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k128-s4-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=4,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k256-s2-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=256,
                num_stages=2,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k256-s3-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=256,
                num_stages=3,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "tma-wgmma-nt-3wg-k512-s2-depth0",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=512,
                num_stages=2,
                wgmma_depth=0,
                tma_activation=True,
            ),
        ),
        (
            "wgmma-nt-3wg-k128-depth1",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=2,
                wgmma_depth=1,
            ),
        ),
        (
            "wgmma-nt-3wg-k128-depth2",
            w4a16_decode_wgmma_nt_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=3,
                wgmma_depth=2,
            ),
        ),
        (
            "wgmma-rs-2wg-k128-depth0",
            w4a16_decode_wgmma_rs_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=2,
                wgmma_depth=0,
            ),
        ),
        (
            "wgmma-rs-2wg-k128-depth1",
            w4a16_decode_wgmma_rs_kernel(m, n, k, "float16")(
                block_out=64,
                block_k=128,
                num_stages=2,
                wgmma_depth=1,
            ),
        ),
        (
            "wgmma-2wg-k64",
            w4a16_gemm_2wg_kernel(m, n, k, "float16")(
                block_m=64,
                block_n=64,
                block_k=64,
                num_stages=2,
            ),
        ),
        (
            "wgmma-2wg-k128",
            w4a16_gemm_2wg_kernel(m, n, k, "float16")(
                block_m=64,
                block_n=64,
                block_k=128,
                num_stages=2,
            ),
        ),
        (
            "wgmma-3wg-k64",
            w4a16_gemm_3wg_kernel(m, n, k, "float16")(
                block_m=64,
                block_n=64,
                block_k=64,
                num_stages=2,
            ),
        ),
        (
            "wgmma-3wg-k128",
            w4a16_gemm_3wg_kernel(m, n, k, "float16")(
                block_m=64,
                block_n=64,
                block_k=128,
                num_stages=2,
            ),
        ),
    ]

    tolerance = {"atol": 5e-2, "rtol": 3e-2}
    for tag, kernel in variants:
        actual = kernel(activation, packed, scale, zero)
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.cuda.synchronize()
        source = kernel.get_kernel_source().lower()
        if tag.startswith("mma-") or tag.startswith("wgmma-rs-"):
            lowering_evidence = [
                line.strip()
                for line in source.splitlines()
                if "mma" in line or "gemm" in line or "warpgroup" in line
            ]
            print(
                f"lowering tag={tag}: "
                + " | ".join(lowering_evidence[:6])
            )
        instruction = (
            "wgmma"
            if "wgmma" in source
            else (
                "mma"
                if (
                    "mma.sync" in source
                    or "mma_sync<" in source
                    or "gemm_mma" in source
                )
                else "scalar"
            )
        )
        has_cp_async = "cp.async" in source or "cp_async" in source
        has_tma = "tma" in source or "cp.async.bulk.tensor" in source
        latency_ms = bench_kernel(
            kernel,
            args=(activation, packed, scale, zero),
        )
        print(
            f"shape=({m},{n},{k}) tag={tag} instruction={instruction} "
            f"cp_async={has_cp_async} tma={has_tma} "
            f"latency_ms={latency_ms:.6f}"
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
