"""Single-launch Nsight Compute target for the best W4A16 ping-pong kernel."""

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

from experiments.w4a16_gemm_feasibility import (
    quantize_weight_int4,
    w4a16_gemv_tma_kernel,
)
from experiments.ako_w4a16.run_candidate import (
    pack_mma_sync_weight,
    pack_mma_transposed_weight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--group-post-scale", action="store_true")
    parser.add_argument("--group-reduce-post-scale", action="store_true")
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--outputs-per-warp", type=int, default=16)
    parser.add_argument("--split-k-warps", type=int, default=4)
    parser.add_argument("--consumer-warpgroups", type=int, default=4)
    parser.add_argument("--no-tma-activation", action="store_true")
    parser.add_argument("--no-tma-weight", action="store_true")
    parser.add_argument("--group-metadata-fp16", action="store_true")
    parser.add_argument("--raw-multiply-fp16", action="store_true")
    parser.add_argument("--raw-fp16-values", type=int, default=0)
    parser.add_argument("--half2-accum-pairs", type=int, default=0)
    parser.add_argument("--cache-activation-fp32", action="store_true")
    parser.add_argument("--producer-activation-sum", action="store_true")
    parser.add_argument("--mma-sync", action="store_true")
    parser.add_argument("--mma-batch", type=int, default=1)
    parser.add_argument("--mma-pingpong", action="store_true")
    parser.add_argument("--mma-direct-activation", action="store_true")
    parser.add_argument("--mma-n-reuse", action="store_true")
    parser.add_argument("--mma-transpose", action="store_true")
    parser.add_argument("--mma-schedule-id", type=int, default=0)
    parser.add_argument("--scalar-schedule-id", type=int, default=-1)
    parser.add_argument("--consumer-max-nreg", type=int, default=112)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    activation = torch.randn(
        (1, args.k), device="cuda", dtype=torch.float16)
    source_weight = (
        torch.randn(
            (args.n, args.k),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.25
    )
    packed, scale, zero, dequantized = quantize_weight_int4(
        source_weight,
        quant_mode="affine",
    )
    if args.mma_sync:
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

    kernel = w4a16_gemv_tma_kernel(
        1, args.n, args.k, "float16"
    )(
        block_n=args.block_n,
        block_k=args.block_k,
        num_stages=args.num_stages,
        outputs_per_warp=args.outputs_per_warp,
        split_k_warps=args.split_k_warps,
        consumer_warpgroups=args.consumer_warpgroups,
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
        mma_sync=args.mma_sync,
        mma_batch=args.mma_batch,
        mma_pingpong=args.mma_pingpong,
        mma_direct_activation=args.mma_direct_activation,
        mma_n_reuse=args.mma_n_reuse,
        mma_transpose=args.mma_transpose,
        mma_schedule_id=args.mma_schedule_id,
        scalar_schedule_id=args.scalar_schedule_id,
        consumer_max_nreg=args.consumer_max_nreg,
    )

    actual = kernel(activation, packed, scale, zero)
    if not args.skip_correctness:
        torch.testing.assert_close(
            actual, expected, atol=5e-2, rtol=3e-2)
    for _ in range(3):
        kernel(activation, packed, scale, zero)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("tileops_pingpong")
    kernel(activation, packed, scale, zero)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
