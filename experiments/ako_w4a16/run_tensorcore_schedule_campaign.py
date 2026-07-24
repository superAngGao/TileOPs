"""Run resumable transposed mma.sync structural schedule rounds.

Each schedule changes compile-time fragment traversal, register predecode
depth, K-group traversal, or group-scale handoff.  Launch parameters remain
frozen and are not counted as rounds.
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

try:
    import pytest as _pytest  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pytest"] = types.ModuleType("pytest")

from benchmarks.benchmark_base import bench_kernel
from experiments.ako_w4a16.run_candidate import (
    pack_mma_transposed_weight,
)
from experiments.w4a16_gemm_feasibility import (
    quantize_weight_int4,
    w4a16_gemv_tma_kernel,
)


SHAPE = (1, 8192, 81920)
GROUP_SIZE = 128
H200_PEAK_GBPS = 4800.0
PARAMETERS = {
    "block_n": 64,
    "block_k": 256,
    "num_stages": 3,
    "outputs_per_warp": 16,
    "split_k_warps": 4,
    "consumer_warpgroups": 4,
    "global_split_k": 1,
    "consumer_max_nreg": 96,
}


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
    sub_modes = (
        "forward",
        "reverse",
        "even-odd",
        "odd-even",
        "half-xor",
        "bit-reverse",
        "gray",
        "reverse-gray",
    )
    sub_mode = schedule_id & 7
    batch_log2 = (schedule_id >> 3) & 3
    batch = 1 << batch_log2
    reverse_group = bool((schedule_id >> 5) & 1)
    defer_scale = bool((schedule_id >> 6) & 1)
    return {
        "load_pipeline": "producer-wg-tma-w4-three-stage-ring",
        "weight_layout": (
            "offline-row-major-m16n8k16-A-fragment_N64-box"
        ),
        "activation_path": (
            "cooperative-global-to-shared_then-mma-B-col0-fragment"
        ),
        "weight_path": (
            "tma-packed-w4-to-shared_then-lop3-direct-A-fragment"
        ),
        "mma_mapping": (
            "W[N16,K16]-as-A_times-padded-activation[K16,8]-as-B"
        ),
        "fragment_schedule": (
            f"{sub_modes[sub_mode]}-K16_predecode-{batch}-fragments"
        ),
        "group_schedule": (
            "reverse-K128-groups" if reverse_group
            else "forward-K128-groups"
        ),
        "scale_handoff": (
            "buffer-two-group-accumulators-then-scale"
            if defer_scale
            else "scale-each-group-before-next-group"
        ),
        "compute_instruction": "mma.sync.m16n8k16.f32.f16.f16",
        "accumulator_path": "temporary-FP32-K128_then-final-FP32",
        "reduction_path": "shared-CTA-local-splitK4-reduction",
        "output_schedule": "one-CTA-N64_four-N16-output-warps",
    }


def logical_bytes() -> int:
    m, n, k = SHAPE
    ctas = n // PARAMETERS["block_n"]
    return (
        n * k // 2
        + n * (k // GROUP_SIZE) * 4
        + n * (k // GROUP_SIZE)
        + ctas * m * k * 2
        + m * n * 2
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=100)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--start-schedule", type=int, default=0)
    parser.add_argument(
        "--records-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    rounds_path = args.records_dir / "shape4_tensorcore_rounds.jsonl"
    trials_path = args.records_dir / "shape4_tensorcore_trials.jsonl"
    attempts_path = args.records_dir / "shape4_tensorcore_attempts.jsonl"
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
    packed = pack_mma_transposed_weight(
        packed,
        block_n=PARAMETERS["block_n"],
        block_k=PARAMETERS["block_k"],
    )
    expected = activation @ dequantized.half().T
    del source_weight, dequantized
    torch.cuda.empty_cache()

    producer = w4a16_gemv_tma_kernel(m, n, k, "float16")
    bytes_per_launch = logical_bytes()
    new_valid = 0

    for schedule_id in range(args.start_schedule, 128):
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
            f"shape4-tc{round_number:03d}-mma-schedule-"
            f"{schedule_id:03d}"
        )
        started = time.perf_counter()
        try:
            kernel = producer(
                **PARAMETERS,
                tma_activation=False,
                tma_weight=True,
                group_post_scale=True,
                mma_sync=True,
                mma_transpose=True,
                mma_schedule_id=schedule_id,
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
        except Exception as error:  # noqa: BLE001 - resumable campaign
            attempt = {
                "round_id": round_id,
                "schedule_id": schedule_id,
                "counted_round": False,
                "reason": type(error).__name__,
                "detail": str(error)[-4000:],
                "structure_signature": signature,
            }
            append_jsonl(attempts_path, attempt)
            print("AKO_TENSORCORE_ATTEMPT=" + canonical(attempt), flush=True)
            continue

        effective_gbps = bytes_per_launch / latency_ms / 1e6
        config = {
            **PARAMETERS,
            "family": "mma-sync",
            "mma_transpose": True,
            "mma_schedule_id": schedule_id,
            "tma_activation": False,
            "tma_weight": True,
            "group_post_scale": True,
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
            "parameters": PARAMETERS,
            "result": result,
        }
        parent_id = (
            rounds[-1]["round_id"]
            if rounds
            else "shape4-transposed-mma-baseline"
        )
        record = {
            "round_number": round_number,
            "round_id": round_id,
            "parent_id": parent_id,
            "schedule_id": schedule_id,
            "hypothesis": (
                "Compile-time fragment traversal, predecode distance, "
                "K128 group order, and scale handoff can overlap LOP3 with "
                "mma.sync while retaining FP32 group accumulation."
            ),
            "transform": (
                f"Apply transposed MMA schedule {schedule_id}: "
                f"{structure['fragment_schedule']}, "
                f"{structure['group_schedule']}, "
                f"{structure['scale_handoff']}."
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
        print("AKO_TENSORCORE_ROUND=" + canonical(record), flush=True)

    summary = {
        "completed_total": len(rounds),
        "new_valid": new_valid,
        "next_schedule": (
            max(existing_schedule_ids) + 1
            if existing_schedule_ids else args.start_schedule
        ),
        "target_total": args.target_total,
    }
    print("AKO_TENSORCORE_CAMPAIGN=" + canonical(summary), flush=True)
    if len(rounds) < args.target_total and (
        args.max_new is None or new_valid < args.max_new
    ):
        raise RuntimeError(
            "schedule space exhausted before target structural rounds")


if __name__ == "__main__":
    main()
