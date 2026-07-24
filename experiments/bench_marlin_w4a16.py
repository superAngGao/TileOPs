"""Cold-cache vLLM Marlin W4A16 microbenchmark.

The timed callable launches only ``marlin_gemm``.  Inputs already use Marlin's
physical layout, so repacking and quantization are deliberately outside the
measurement.  Timing comes from TileOps' production ``bench_kernel`` helper.
"""

from __future__ import annotations

import argparse
import sys
import types

import torch

try:
    import pytest as _pytest  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pytest"] = types.ModuleType("pytest")

from benchmarks.benchmark_base import bench_kernel
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.scalar_type import scalar_types


GROUP_SIZE = 128


def run_shape(m: int, n: int, k: int, use_fp32_reduce: bool) -> None:
    if k % 16 or k % GROUP_SIZE or n % 64:
        raise ValueError("Marlin test requires K % 128 == 0 and N % 64 == 0")

    torch.manual_seed(0)
    device = torch.device("cuda")
    activation = torch.randn((m, k), dtype=torch.float16, device=device)

    # Marlin UINT4 physical layouts after AWQ repacking:
    #   qweight: 16x64 tiles, eight nibbles per int32
    #   scales:  one FP16 value per (group, output column)
    #   zeros:   eight UINT4 zero-points per int32
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // 16, n * 2),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand(
        (k // GROUP_SIZE, n),
        dtype=torch.float16,
        device=device,
    )
    zeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // GROUP_SIZE, n // 8),
        dtype=torch.int32,
        device=device,
    )
    workspace = marlin_make_workspace_new(device)

    def marlin(
        a: torch.Tensor,
        packed: torch.Tensor,
        weight_scales: torch.Tensor,
        weight_zeros: torch.Tensor,
        locks: torch.Tensor,
    ) -> torch.Tensor:
        return ops.marlin_gemm(
            a=a,
            c=None,
            b_q_weight=packed,
            b_bias=None,
            b_scales=weight_scales,
            a_scales=None,
            global_scale=None,
            b_zeros=weight_zeros,
            g_idx=None,
            perm=None,
            workspace=locks,
            b_q_type=scalar_types.uint4,
            size_m=m,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    actual = marlin(activation, qweight, scales, zeros, workspace)
    if actual.shape != (m, n) or not torch.isfinite(actual).all():
        raise RuntimeError("Marlin smoke check failed")
    torch.cuda.synchronize()

    latency_ms = bench_kernel(
        marlin,
        args=(activation, qweight, scales, zeros, workspace),
    )
    print(
        f"shape=({m},{n},{k}) dtype=float16 cache=cold "
        f"reduce={'fp32' if use_fp32_reduce else 'fp16'} "
        f"marlin_ms={latency_ms:.6f}"
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
        "--reduce",
        choices=("fp32", "fp16", "both"),
        default="fp32",
    )
    args = parser.parse_args()
    for value in args.shape:
        dims = tuple(int(dim) for dim in value.split(","))
        if len(dims) != 3:
            raise ValueError(f"--shape expects M,N,K, got {value!r}")
        modes = ("fp32", "fp16") if args.reduce == "both" else (args.reduce,)
        for mode in modes:
            run_shape(*dims, use_fp32_reduce=(mode == "fp32"))


if __name__ == "__main__":
    main()
