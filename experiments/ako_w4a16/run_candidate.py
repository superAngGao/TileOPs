"""Run one W4A16 AKO candidate through correctness and cold-cache timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
    w4a16_decode_wgmma_rs_kernel,
    w4a16_gemv_kernel,
    w4a16_gemv_tma_kernel,
    w4a16_splitk_reduce_kernel,
)


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(dim) for dim in value.split(","))
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return shape


def pack_mma_sync_weight(
    packed: torch.Tensor,
    block_n: int = 64,
    block_k: int = 256,
) -> torch.Tensor:
    """Offline W4 reorder matching mma.m16n8k16 B-fragment ownership."""
    n, packed_k = packed.shape
    k = packed_k * 2
    if n % block_n or k % block_k:
        raise ValueError("mma-sync packing requires N64/K256 alignment")
    n_blocks = n // block_n
    k_blocks = k // block_k
    output_tiles = block_n // 8
    logical = packed.view(
        n_blocks,
        output_tiles,
        8,
        k_blocks,
        block_k // 16,
        8,
    )
    fragment = torch.empty(
        (
            n_blocks,
            k_blocks,
            output_tiles,
            block_k // 16,
            32,
            2,
        ),
        device=packed.device,
        dtype=torch.uint8,
    )
    lanes = torch.arange(32, device=packed.device)
    output_in_tile = lanes // 4
    byte_in_half = lanes % 4
    for output_tile in range(output_tiles):
        for k_tile in range(block_k // 16):
            fragment[
                :,
                :,
                output_tile,
                k_tile,
                :,
                0,
            ] = logical[
                :,
                output_tile,
                output_in_tile,
                :,
                k_tile,
                byte_in_half,
            ].permute(1, 2, 0)
            fragment[
                :,
                :,
                output_tile,
                k_tile,
                :,
                1,
            ] = logical[
                :,
                output_tile,
                output_in_tile,
                :,
                k_tile,
                byte_in_half + 4,
            ].permute(1, 2, 0)
    return fragment.reshape(
        n_blocks,
        k_blocks,
        block_n // 64,
        32,
        64 * block_k // (2 * 32),
    ).contiguous()


def pack_mma_transposed_weight(
    packed: torch.Tensor,
    block_n: int = 128,
    block_k: int = 256,
) -> torch.Tensor:
    """Pack W4 directly in row-major mma.m16n8k16 A-fragment order."""
    n, packed_k = packed.shape
    k = packed_k * 2
    if block_n % 64 or n % block_n or k % block_k:
        raise ValueError(
            "transposed MMA packing requires N64/K256 alignment")
    n_blocks = n // block_n
    k_blocks = k // block_k
    source = packed.view(
        n_blocks,
        block_n,
        k_blocks,
        block_k // 2,
    ).permute(0, 2, 1, 3)
    fragment = torch.empty(
        (
            n_blocks,
            k_blocks,
            block_n // 16,
            block_k // 16,
            32,
            4,
        ),
        device=packed.device,
        dtype=torch.uint8,
    )
    lanes = torch.arange(32, device=packed.device)
    row0 = lanes // 4
    pair0 = lanes % 4
    for n_tile in range(block_n // 16):
        output_base = n_tile * 16
        for k_tile in range(block_k // 16):
            byte_base = k_tile * 8
            fragment[:, :, n_tile, k_tile, :, 0] = source[
                :, :, output_base + row0, byte_base + pair0
            ]
            fragment[:, :, n_tile, k_tile, :, 1] = source[
                :, :, output_base + row0 + 8, byte_base + pair0
            ]
            fragment[:, :, n_tile, k_tile, :, 2] = source[
                :, :, output_base + row0, byte_base + pair0 + 4
            ]
            fragment[:, :, n_tile, k_tile, :, 3] = source[
                :, :, output_base + row0 + 8, byte_base + pair0 + 4
            ]
    return fragment.view(
        n_blocks,
        k_blocks,
        block_n // 64,
        32,
        4 * block_k // 16 * 4,
    ).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument(
        "--family",
        choices=("direct", "tma", "wgmma-rs", "mma-sync"),
        default="tma",
    )
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--outputs-per-warp", type=int, default=16)
    parser.add_argument("--split-k-warps", type=int, default=4)
    parser.add_argument("--consumer-warpgroups", type=int, default=4)
    parser.add_argument("--global-split-k", type=int, default=1)
    parser.add_argument("--reduce-threads", type=int, default=256)
    parser.add_argument("--no-tma-activation", action="store_true")
    parser.add_argument("--no-tma-weight", action="store_true")
    parser.add_argument("--group-post-scale", action="store_true")
    parser.add_argument("--group-reduce-post-scale", action="store_true")
    parser.add_argument("--group-metadata-fp16", action="store_true")
    parser.add_argument("--raw-multiply-fp16", action="store_true")
    parser.add_argument("--raw-fp16-values", type=int, default=0)
    parser.add_argument("--half2-accum-pairs", type=int, default=0)
    parser.add_argument("--cache-activation-fp32", action="store_true")
    parser.add_argument("--producer-activation-sum", action="store_true")
    parser.add_argument("--wgmma-depth", type=int, default=0)
    parser.add_argument("--post-scale-wgmma", action="store_true")
    parser.add_argument("--mma-batch", type=int, default=1)
    parser.add_argument("--mma-pingpong", action="store_true")
    parser.add_argument("--mma-direct-activation", action="store_true")
    parser.add_argument("--mma-n-reuse", action="store_true")
    parser.add_argument("--mma-transpose", action="store_true")
    parser.add_argument("--mma-schedule-id", type=int, default=0)
    parser.add_argument("--scalar-schedule-id", type=int, default=-1)
    parser.add_argument("--consumer-max-nreg", type=int, default=112)
    parser.add_argument("--benchmark-incorrect", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    m, n, k = args.shape
    if m != 1:
        raise ValueError("The AKO campaign is scoped to M=1")

    config = {
        "family": args.family,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "num_stages": args.num_stages,
        "outputs_per_warp": args.outputs_per_warp,
        "split_k_warps": args.split_k_warps,
        "consumer_warpgroups": args.consumer_warpgroups,
        "global_split_k": args.global_split_k,
        "reduce_threads": args.reduce_threads,
        "tma_activation": not args.no_tma_activation,
        "tma_weight": not args.no_tma_weight,
        "group_post_scale": args.group_post_scale,
        "group_reduce_post_scale": args.group_reduce_post_scale,
        "group_metadata_fp16": args.group_metadata_fp16,
        "raw_multiply_fp16": args.raw_multiply_fp16,
        "raw_fp16_values": args.raw_fp16_values,
        "half2_accum_pairs": args.half2_accum_pairs,
        "cache_activation_fp32": args.cache_activation_fp32,
        "producer_activation_sum": args.producer_activation_sum,
        "wgmma_depth": args.wgmma_depth,
        "post_scale_wgmma": args.post_scale_wgmma,
        "mma_batch": args.mma_batch,
        "mma_pingpong": args.mma_pingpong,
        "mma_direct_activation": args.mma_direct_activation,
        "mma_n_reuse": args.mma_n_reuse,
        "mma_transpose": args.mma_transpose,
        "mma_schedule_id": args.mma_schedule_id,
        "scalar_schedule_id": args.scalar_schedule_id,
        "consumer_max_nreg": args.consumer_max_nreg,
        "weight_layout": (
            "mma-sync-transposed-fragment"
            if args.mma_transpose
            else "mma-sync-fragment"
            if args.family == "mma-sync"
            else "lop3-interleaved"
        ),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()

    torch.manual_seed(0)
    activation = torch.randn((m, k), device="cuda", dtype=torch.float16)
    source_weight = (
        torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.25
    )
    packed, scale, zero, dequantized = quantize_weight_int4(
        source_weight,
        quant_mode="affine",
    )
    if args.family == "mma-sync":
        if args.mma_transpose:
            packed = pack_mma_transposed_weight(
                packed,
                block_n=args.block_n,
                block_k=args.block_k,
            )
        else:
            packed = pack_mma_sync_weight(
                packed,
                block_n=args.block_n,
                block_k=args.block_k,
            )
    else:
        packed = interleave_weight(
            packed,
            nbits=4,
            target_dtype="float16",
        ).view(torch.uint8)
    expected = activation @ dequantized.half().T

    compile_started = time.perf_counter()
    if args.family == "direct":
        kernel = w4a16_gemv_kernel(m, n, k, "float16")(
            n_partition=1,
            split_k_warps=args.split_k_warps,
            outputs_per_warp=args.outputs_per_warp,
        )
    elif args.family == "wgmma-rs":
        kernel = w4a16_decode_wgmma_rs_kernel(
            m, n, k, "float16"
        )(
            block_out=args.block_n,
            block_k=128,
            num_stages=args.num_stages,
            wgmma_depth=args.wgmma_depth,
            post_scale_wgmma=args.post_scale_wgmma,
        )
    else:
        producer = w4a16_gemv_tma_kernel(m, n, k, "float16")(
            block_n=args.block_n,
            block_k=args.block_k,
            num_stages=args.num_stages,
            outputs_per_warp=args.outputs_per_warp,
            split_k_warps=args.split_k_warps,
            consumer_warpgroups=args.consumer_warpgroups,
            global_split_k=args.global_split_k,
            tma_activation=not args.no_tma_activation,
            tma_weight=not args.no_tma_weight,
            group_post_scale=args.group_post_scale,
            group_reduce_post_scale=args.group_reduce_post_scale,
            group_metadata_fp16=args.group_metadata_fp16,
            raw_multiply_fp16=args.raw_multiply_fp16,
            raw_fp16_values=args.raw_fp16_values,
            half2_accum_pairs=args.half2_accum_pairs,
            cache_activation_fp32=args.cache_activation_fp32,
            producer_activation_sum=args.producer_activation_sum,
            mma_sync=args.family == "mma-sync",
            mma_batch=args.mma_batch,
            mma_pingpong=args.mma_pingpong,
            mma_direct_activation=args.mma_direct_activation,
            mma_n_reuse=args.mma_n_reuse,
            mma_transpose=args.mma_transpose,
            mma_schedule_id=args.mma_schedule_id,
            scalar_schedule_id=args.scalar_schedule_id,
            consumer_max_nreg=args.consumer_max_nreg,
        )
        if args.global_split_k == 1:
            kernel = producer
            source_kernels = (producer,)
        else:
            reducer = w4a16_splitk_reduce_kernel(
                m, n, args.global_split_k, "float16"
            )(threads=args.reduce_threads)

            def kernel(
                activation_arg,
                packed_arg,
                scale_arg,
                zero_arg,
            ):
                return reducer(
                    producer(
                        activation_arg,
                        packed_arg,
                        scale_arg,
                        zero_arg,
                    )
                )

            source_kernels = (producer, reducer)
    if args.family in ("direct", "wgmma-rs"):
        if args.global_split_k != 1:
            raise ValueError(
                "global split-K is implemented only for tma/mma-sync")
        source_kernels = (kernel,)
    actual = kernel(activation, packed, scale, zero)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started

    correct = True
    try:
        torch.testing.assert_close(
            actual, expected, atol=5e-2, rtol=3e-2)
    except AssertionError:
        if not args.benchmark_incorrect:
            raise
        correct = False
    difference = (actual.float() - expected.float()).abs()
    source = "\n".join(
        compiled.get_kernel_source() for compiled in source_kernels)
    source_hash = hashlib.sha256(source.encode()).hexdigest()

    if args.full:
        bench_args = {}
        protocol = "confirmation"
    else:
        bench_args = {
            "n_warmup": 3,
            "n_repeat": 10,
            "n_trials": 1,
        }
        protocol = "screening"
    latency_ms = bench_kernel(
        kernel,
        args=(activation, packed, scale, zero),
        **bench_args,
    )

    result = {
        "candidate_id": args.candidate_id,
        "shape": [m, n, k],
        "config": config,
        "config_hash": config_hash,
        "source_hash": source_hash,
        "compile_seconds": compile_seconds,
        "correct": correct,
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "protocol": protocol,
        "latency_ms": latency_ms,
    }
    print("AKO_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
