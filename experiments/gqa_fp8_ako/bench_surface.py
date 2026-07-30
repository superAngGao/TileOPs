#!/usr/bin/env python3
"""Stable A/B runner for the FP8 GQA AKO ladder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.benchmark_base import bench_kernel
from benchmarks.ops.attention.bench_gqa_fp8 import (
    _fa3_gqa_fp8_fwd,
    _make_inputs,
    _manifest_cases,
)
from tileops.ops import GroupedQueryAttentionPrefillFwdOp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _measure(fn, inputs, args: argparse.Namespace) -> float:
    return bench_kernel(
        fn,
        args=inputs,
        n_warmup=args.warmup,
        n_repeat=args.repeat,
        n_trials=args.trials,
    )


def main() -> None:
    args = _parse_args()
    cases = _manifest_cases()
    if args.case:
        cases = [case for case in cases if any(token in case.label for token in args.case)]
    if not cases:
        raise SystemExit("No benchmark cases matched")

    records = []
    for case in cases:
        case_record = asdict(case)
        case_record["out_dtype"] = str(case.out_dtype)
        inputs = _make_inputs(case)
        q, k, v, cu_q, cu_kv, q_descale, k_descale, v_descale = inputs

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
        implementations = [
            ("tileops_canonical_2d_descale", op, inputs),
        ]
        fa3_fn = _fa3_gqa_fp8_fwd(case)
        if fa3_fn is not None:
            implementations.append(("fa3", fa3_fn, inputs))

        for implementation, fn, fn_inputs in implementations:
            fn(*fn_inputs)
            torch.cuda.synchronize()
            latency_ms = _measure(fn, fn_inputs, args)
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "round": 1,
                "case": case_record,
                "implementation": implementation,
                "latency_ms": latency_ms,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "trials": args.trials,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            }
            records.append(record)
            print(
                f"{case.label:58s} {implementation:30s} "
                f"{latency_ms:9.6f} ms",
                flush=True,
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
