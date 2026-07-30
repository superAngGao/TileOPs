#!/usr/bin/env python3
"""Compile, warm up, and launch one FP8 GQA call for profiler collection."""

from __future__ import annotations

import argparse

import torch

from benchmarks.ops.attention.bench_gqa_fp8 import (
    _fa3_gqa_fp8_fwd,
    _make_inputs,
    _manifest_cases,
)
from tileops.ops import GroupedQueryAttentionPrefillFwdOp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--implementation", choices=("tileops", "fa3"), default="tileops")
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    matches = [case for case in _manifest_cases() if args.case in case.label]
    if len(matches) != 1:
        labels = ", ".join(case.label for case in matches)
        raise SystemExit(f"Expected exactly one case, found {len(matches)}: {labels}")
    case = matches[0]
    inputs = _make_inputs(case)
    if args.implementation == "tileops":
        op = GroupedQueryAttentionPrefillFwdOp(
            batch=case.batch,
            heads=case.heads,
            heads_kv=case.heads_kv,
            dim=case.dim,
            max_seqlen_q=case.seq_len,
            max_seqlen_kv=case.seq_len,
            is_causal=False,
            dtype=case.out_dtype,
            backend="fp8",
            validate_uniform_cu_seqlens=case.validate_uniform_cu_seqlens,
        )
    else:
        op = _fa3_gqa_fp8_fwd(case)
        if op is None:
            raise SystemExit("FA3 is not available in this runtime")

    for _ in range(args.warmup):
        op(*inputs)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"{args.implementation}_fp8_gqa_profile")
    op(*inputs)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
