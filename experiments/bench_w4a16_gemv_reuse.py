"""Cold-cache ablation for activation reuse across W4A16 output columns."""

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
    w4a16_splitk_reduce_kernel,
)


def run_shape(
    m: int,
    n: int,
    k: int,
    configs: list[tuple[int, int, int]] | None,
) -> None:
    if m != 1:
        raise ValueError("The activation-reuse experiment requires M=1")

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

    if configs is None:
        split_k_warps = 1 if k <= 2048 else (2 if k <= 4096 else 4)
        n_partition = 4 if split_k_warps == 1 else (2 if k != 8192 else 1)
        shape_configs = [
            (split_k_warps, n_partition, outputs_per_warp)
            for outputs_per_warp in (1, 2, 4, 8)
        ]
    else:
        shape_configs = configs

    for split_k_warps, n_partition, outputs_per_warp in shape_configs:
        kernel = w4a16_gemv_kernel(m, n, k, "float16")(
            n_partition=n_partition,
            split_k_warps=split_k_warps,
            outputs_per_warp=outputs_per_warp,
        )
        actual = kernel(activation, packed, scale, zero)
        torch.testing.assert_close(
            actual,
            expected,
            atol=5e-2,
            rtol=3e-2,
        )
        torch.cuda.synchronize()
        latency_ms = bench_kernel(
            kernel,
            args=(activation, packed, scale, zero),
        )
        ctas = (n + n_partition * outputs_per_warp - 1) // (
            n_partition * outputs_per_warp
        )
        print(
            f"shape=({m},{n},{k}) split={split_k_warps} "
            f"n_partition={n_partition} "
            f"outputs_per_warp={outputs_per_warp} ctas={ctas} "
            f"latency_ms={latency_ms:.6f}"
        )

    if k >= 8192:
        direct_n64 = w4a16_gemv_kernel(m, n, k, "float16")(
            n_partition=8,
            split_k_warps=1,
            outputs_per_warp=8,
        )
        direct_actual = direct_n64(activation, packed, scale, zero)
        torch.testing.assert_close(
            direct_actual, expected, atol=5e-2, rtol=3e-2)
        direct_ms = bench_kernel(
            direct_n64,
            args=(activation, packed, scale, zero),
        )
        print(
            f"shape=({m},{n},{k}) tag=direct-n64-k256 "
            f"latency_ms={direct_ms:.6f}"
        )

        for num_stages in (2, 3, 4):
            tma_kernel = w4a16_gemv_tma_kernel(m, n, k, "float16")(
                block_n=64,
                block_k=256,
                num_stages=num_stages,
                outputs_per_warp=8,
                split_k_warps=1,
            )
            tma_actual = tma_kernel(activation, packed, scale, zero)
            torch.testing.assert_close(
                tma_actual, expected, atol=5e-2, rtol=3e-2)
            source = tma_kernel.get_kernel_source().lower()
            if "tma" not in source and "cp.async.bulk.tensor" not in source:
                raise RuntimeError("TMA experiment did not lower to TMA")
            tma_ms = bench_kernel(
                tma_kernel,
                args=(activation, packed, scale, zero),
            )
            print(
                f"shape=({m},{n},{k}) tag=tma-n64-k256-s{num_stages} "
                f"latency_ms={tma_ms:.6f}"
            )

        for block_n, outputs_per_warp in ((16, 2), (32, 4)):
            for num_stages in (2, 3, 4):
                tma_kernel = w4a16_gemv_tma_kernel(
                    m, n, k, "float16"
                )(
                    block_n=block_n,
                    block_k=256,
                    num_stages=num_stages,
                    outputs_per_warp=outputs_per_warp,
                    split_k_warps=1,
                )
                tma_actual = tma_kernel(
                    activation, packed, scale, zero)
                torch.testing.assert_close(
                    tma_actual, expected, atol=5e-2, rtol=3e-2)
                tma_ms = bench_kernel(
                    tma_kernel,
                    args=(activation, packed, scale, zero),
                )
                print(
                    f"shape=({m},{n},{k}) "
                    f"tag=tma-n{block_n}-k256-s{num_stages} "
                    f"latency_ms={tma_ms:.6f}"
                )

        for split_k_warps in (2, 4, 8):
            outputs_per_warp = 8 * split_k_warps
            stage_configs = (2,) if split_k_warps == 8 else (2, 3, 4)
            for num_stages in stage_configs:
                tma_split_n64 = w4a16_gemv_tma_kernel(
                    m, n, k, "float16"
                )(
                    block_n=64,
                    block_k=256,
                    num_stages=num_stages,
                    outputs_per_warp=outputs_per_warp,
                    split_k_warps=split_k_warps,
                )
                split_n64_actual = tma_split_n64(
                    activation, packed, scale, zero)
                torch.testing.assert_close(
                    split_n64_actual, expected, atol=5e-2, rtol=3e-2)
                split_n64_ms = bench_kernel(
                    tma_split_n64,
                    args=(activation, packed, scale, zero),
                )
                print(
                    f"shape=({m},{n},{k}) "
                    f"tag=tma-split{split_k_warps}-n64-"
                    f"o{outputs_per_warp}-"
                    f"k{256 * split_k_warps}-s{num_stages} "
                    f"latency_ms={split_n64_ms:.6f}"
                )

        for split_k_warps, outputs_per_warp in ((2, 8), (4, 16)):
            for num_stages in (2, 3):
                five_wg_kernel = w4a16_gemv_tma_kernel(
                    m, n, k, "float16"
                )(
                    block_n=64,
                    block_k=256,
                    num_stages=num_stages,
                    outputs_per_warp=outputs_per_warp,
                    split_k_warps=split_k_warps,
                    consumer_warpgroups=4,
                )
                five_wg_actual = five_wg_kernel(
                    activation, packed, scale, zero)
                torch.testing.assert_close(
                    five_wg_actual, expected, atol=5e-2, rtol=3e-2)
                five_wg_ms = bench_kernel(
                    five_wg_kernel,
                    args=(activation, packed, scale, zero),
                )
                print(
                    f"shape=({m},{n},{k}) "
                    f"tag=tma-5wg-split{split_k_warps}-n64-"
                    f"o{outputs_per_warp}-"
                    f"k{256 * split_k_warps}-s{num_stages} "
                    f"latency_ms={five_wg_ms:.6f}"
                )

        for global_split_k in (2, 4):
            split_producer = w4a16_gemv_tma_kernel(
                m, n, k, "float16"
            )(
                block_n=64,
                block_k=256,
                num_stages=3,
                outputs_per_warp=8,
                split_k_warps=1,
                global_split_k=global_split_k,
            )
            split_reducer = w4a16_splitk_reduce_kernel(
                m, n, global_split_k, "float16"
            )()

            def split_pipeline(
                activation_arg,
                packed_arg,
                scale_arg,
                zero_arg,
                producer=split_producer,
                reducer=split_reducer,
            ):
                return reducer(
                    producer(
                        activation_arg,
                        packed_arg,
                        scale_arg,
                        zero_arg,
                    )
                )

            split_actual = split_pipeline(
                activation, packed, scale, zero)
            torch.testing.assert_close(
                split_actual, expected, atol=5e-2, rtol=3e-2)
            partials = split_producer(
                activation, packed, scale, zero)
            producer_ms = bench_kernel(
                split_producer,
                args=(activation, packed, scale, zero),
            )
            reducer_ms = bench_kernel(
                split_reducer,
                args=(partials,),
            )
            split_ms = bench_kernel(
                split_pipeline,
                args=(activation, packed, scale, zero),
            )
            print(
                f"shape=({m},{n},{k}) "
                f"tag=tma-global-split{global_split_k}-"
                "local1-n64-k256-s3 "
                f"producer_ms={producer_ms:.6f} "
                f"reducer_ms={reducer_ms:.6f} "
                f"latency_ms={split_ms:.6f}"
            )

        for split_k_warps, outputs_per_warp, num_stages in (
            (1, 8, 3),
            (2, 16, 3),
            (2, 16, 4),
        ):
            weight_tma_kernel = w4a16_gemv_tma_kernel(
                m, n, k, "float16"
            )(
                block_n=64,
                block_k=256,
                num_stages=num_stages,
                outputs_per_warp=outputs_per_warp,
                split_k_warps=split_k_warps,
                tma_activation=False,
            )
            weight_tma_actual = weight_tma_kernel(
                activation, packed, scale, zero)
            torch.testing.assert_close(
                weight_tma_actual, expected, atol=5e-2, rtol=3e-2)
            weight_tma_ms = bench_kernel(
                weight_tma_kernel,
                args=(activation, packed, scale, zero),
            )
            print(
                f"shape=({m},{n},{k}) "
                f"tag=weight-tma-split{split_k_warps}-n64-"
                f"o{outputs_per_warp}-"
                f"k{256 * split_k_warps}-s{num_stages} "
                f"latency_ms={weight_tma_ms:.6f}"
            )

        for split_k_warps in (2, 4, 8):
            block_n = 64 // split_k_warps
            tma_split_kernel = w4a16_gemv_tma_kernel(
                m, n, k, "float16"
            )(
                block_n=block_n,
                block_k=256,
                num_stages=3,
                outputs_per_warp=8,
                split_k_warps=split_k_warps,
            )
            split_actual = tma_split_kernel(
                activation, packed, scale, zero)
            torch.testing.assert_close(
                split_actual, expected, atol=5e-2, rtol=3e-2)
            split_ms = bench_kernel(
                tma_split_kernel,
                args=(activation, packed, scale, zero),
            )
            print(
                f"shape=({m},{n},{k}) "
                f"tag=tma-split{split_k_warps}-"
                f"n{block_n}-k{256 * split_k_warps}-s3 "
                f"latency_ms={split_ms:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        required=True,
        metavar="M,N,K",
    )
    parser.add_argument(
        "--config",
        action="append",
        metavar="SPLIT_K_WARPS,N_PARTITION,OUTPUTS_PER_WARP",
    )
    args = parser.parse_args()
    configs = None
    if args.config:
        configs = []
        for value in args.config:
            config = tuple(int(dim) for dim in value.split(","))
            if len(config) != 3:
                raise ValueError(
                    "--config expects "
                    "SPLIT_K_WARPS,N_PARTITION,OUTPUTS_PER_WARP"
                )
            configs.append(config)
    for value in args.shape:
        dims = tuple(int(dim) for dim in value.split(","))
        if len(dims) != 3:
            raise ValueError(f"--shape expects M,N,K, got {value!r}")
        run_shape(*dims, configs=configs)


if __name__ == "__main__":
    main()
