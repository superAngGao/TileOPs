"""Run resumable, source-distinct scalar pipeline schedule rounds.

Each schedule ID is a compile-time rewrite of operation traversal and shared
ring handoff.  Launch/tile parameters stay frozen and therefore are not
counted as rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from tilelang.quantize import interleave_weight

try:
    import pytest as _pytest  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pytest"] = types.ModuleType("pytest")

from benchmarks.benchmark_base import bench_kernel
from experiments.w4a16_gemm_feasibility import (
    quantize_weight_int4,
    w4a16_gemv_tma_kernel,
)


SHAPE = (1, 8192, 81920)
GROUP_SIZE = 128
H200_PEAK_GBPS = 4800.0


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def schedule_descriptor(schedule_id: int) -> dict[str, Any]:
    output_modes = (
        "forward",
        "reverse",
        "even-odd",
        "odd-even",
        "half-xor",
        "bit-reverse",
        "gray",
        "reverse-gray",
    )
    value_modes = output_modes
    output_mode = schedule_id % 8
    value_mode = (schedule_id // 8) % 8
    decode_reverse = (schedule_id // 64) % 2
    release_policy = (schedule_id // 128) % 3
    release_names = (
        "after-register-capture-before-activation-sum",
        "after-activation-sum-before-decode",
        "after-decode-and-compute",
    )
    return {
        "load_pipeline": "one-producer-wg-staged-ring",
        "buffer_handoff": release_names[release_policy],
        "activation_path": (
            "cooperative-global-to-shared_then-register-"
            f"{value_modes[value_mode]}-fp32-use"
        ),
        "weight_path": (
            "tma-packed-w4-to-shared_then-register-"
            f"{output_modes[output_mode]}-output-consumption"
        ),
        "metadata_path": (
            "tma-scale-plus-cooperative-zero_to-shared_then-register"
        ),
        "decode_schedule": (
            f"{output_modes[output_mode]}-outputs_"
            f"{'reverse' if decode_reverse else 'forward'}-lop3-chunks_"
            f"{value_modes[value_mode]}-value-fma"
        ),
        "compute_instruction": "scalar-fp32-fma",
        "accumulator_path": (
            f"eight-output-register-array-{output_modes[output_mode]}"
        ),
        "reduction_path": "warp-shuffle-then-shared-split-k-reduction",
        "output_schedule": "one-cta-n64_split-k2",
        "weight_layout": "tilelang-lop3-interleaved-row-major",
    }


def logical_bytes() -> int:
    m, n, k = SHAPE
    ctas = n // 64
    return (
        n * k // 2
        + n * (k // GROUP_SIZE) * 4
        + n * (k // GROUP_SIZE)
        + ctas * m * k * 2
        + m * n * 2
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=300)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--start-schedule", type=int, default=0)
    parser.add_argument(
        "--records-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    rounds_path = args.records_dir / "shape4_structural_rounds.jsonl"
    trials_path = args.records_dir / "shape4_structural_trials.jsonl"
    attempts_path = args.records_dir / "shape4_structural_attempts.jsonl"
    rounds = load_jsonl(rounds_path)
    existing_signatures = {
        record["structure_signature"] for record in rounds
    }
    existing_source_hashes = {
        record["winner"]["result"]["source_hash"]
        for record in rounds
        if "winner" in record
    }
    existing_schedule_ids = {
        record["schedule_id"]
        for record in rounds
        if "schedule_id" in record
    }

    torch.manual_seed(0)
    m, n, k = SHAPE
    activation = torch.randn((m, k), device="cuda", dtype=torch.float16)
    source_weight = (
        torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.25
    )
    packed, scale, zero, dequantized = quantize_weight_int4(
        source_weight, quant_mode="affine")
    packed = interleave_weight(
        packed, nbits=4, target_dtype="float16").view(torch.uint8)
    expected = activation @ dequantized.half().T
    del source_weight, dequantized
    torch.cuda.empty_cache()

    producer = w4a16_gemv_tma_kernel(m, n, k, "float16")
    bytes_per_launch = logical_bytes()
    new_valid = 0

    for schedule_id in range(args.start_schedule, 384):
        if len(rounds) >= args.target_total:
            break
        if args.max_new is not None and new_valid >= args.max_new:
            break
        if schedule_id in existing_schedule_ids:
            continue
        structure = schedule_descriptor(schedule_id)
        signature = hashlib.sha256(canonical(structure).encode()).hexdigest()
        if signature in existing_signatures:
            continue

        round_number = len(rounds) + 1
        round_id = (
            f"shape4-sr{round_number:03d}-scalar-schedule-"
            f"{schedule_id:03d}"
        )
        parameters = {
            "block_n": 64,
            "block_k": 512,
            "num_stages": 3,
            "outputs_per_warp": 8,
            "split_k_warps": 2,
            "consumer_warpgroups": 4,
            "consumer_max_nreg": 112,
        }
        started = time.perf_counter()
        try:
            kernel = producer(
                **parameters,
                tma_activation=False,
                tma_weight=True,
                group_post_scale=True,
                scalar_schedule_id=schedule_id,
            )
            actual = kernel(activation, packed, scale, zero)
            torch.cuda.synchronize()
            torch.testing.assert_close(
                actual, expected, atol=5e-2, rtol=3e-2)
            difference = (actual.float() - expected.float()).abs()
            source = kernel.get_kernel_source()
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            if source_hash in existing_source_hashes:
                raise ValueError("duplicate generated source hash")
            latency_ms = bench_kernel(
                kernel,
                args=(activation, packed, scale, zero),
                n_warmup=3,
                n_repeat=10,
                n_trials=1,
            )
        except Exception as error:  # noqa: BLE001 - campaign must resume
            attempt = {
                "round_id": round_id,
                "schedule_id": schedule_id,
                "counted_round": False,
                "reason": type(error).__name__,
                "detail": str(error)[-4000:],
                "structure_signature": signature,
            }
            append_jsonl(attempts_path, attempt)
            print("AKO_STRUCTURAL_ATTEMPT=" + canonical(attempt), flush=True)
            continue

        effective_gbps = bytes_per_launch / latency_ms / 1e6
        config = {
            **parameters,
            "family": "tma",
            "tma_activation": False,
            "tma_weight": True,
            "group_post_scale": True,
            "scalar_schedule_id": schedule_id,
        }
        result = {
            "candidate_id": round_id,
            "shape": list(SHAPE),
            "config": config,
            "config_hash": hashlib.sha256(
                canonical(config).encode()).hexdigest(),
            "source_hash": source_hash,
            "correct": True,
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "protocol": "screening",
            "latency_ms": latency_ms,
            "logical_bytes": bytes_per_launch,
            "logical_effective_gbps": effective_gbps,
            "logical_pct_h200_peak": (
                effective_gbps / H200_PEAK_GBPS * 100.0
            ),
        }
        trial = {
            "round_id": round_id,
            "schedule_id": schedule_id,
            "structure_signature": signature,
            "trial_index": 1,
            "parameters": parameters,
            "result": result,
        }
        parent_id = (
            rounds[-1]["round_id"]
            if rounds
            else "shape4-baseline-scalar-tma-ring"
        )
        record = {
            "round_number": round_number,
            "round_id": round_id,
            "parent_id": parent_id,
            "schedule_id": schedule_id,
            "hypothesis": (
                "A compile-time reordering of independent output, decode, "
                "and K-value work plus shared-slot release timing can change "
                "instruction overlap without changing arithmetic precision."
            ),
            "transform": (
                f"Apply scalar schedule {schedule_id}: "
                f"{structure['decode_schedule']}; release "
                f"{structure['buffer_handoff']}."
            ),
            "structure": structure,
            "structure_signature": signature,
            "trial_count": 1,
            "successful_trial_count": 1,
            "winner": trial,
            "elapsed_seconds": time.perf_counter() - started,
        }
        append_jsonl(trials_path, trial)
        append_jsonl(rounds_path, record)
        rounds.append(record)
        existing_signatures.add(signature)
        existing_source_hashes.add(source_hash)
        existing_schedule_ids.add(schedule_id)
        new_valid += 1
        print("AKO_STRUCTURAL_ROUND=" + canonical(record), flush=True)

    summary = {
        "completed_total": len(rounds),
        "new_valid": new_valid,
        "next_schedule": (
            max(existing_schedule_ids) + 1
            if existing_schedule_ids else args.start_schedule
        ),
        "target_total": args.target_total,
    }
    print("AKO_SCHEDULE_CAMPAIGN=" + canonical(summary), flush=True)
    if len(rounds) < args.target_total and (
        args.max_new is None or new_valid < args.max_new
    ):
        raise RuntimeError(
            "schedule space exhausted before target structural rounds")


if __name__ == "__main__":
    main()
