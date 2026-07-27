"""Run configurable AKO sweeps for Gated DeltaNet backward.

The runner launches each candidate in a fresh Python subprocess. That is slower
than in-process autotuning, but it keeps TileLang compile/runtime failures from
poisoning the whole search loop and leaves an auditable JSONL trail.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _REPO_ROOT / "experiments/gated_deltanet_bwd_prefill_style/bench_current_bwd.py"


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--layout", default="bthd")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--target-total", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--num-stages", default="1,2,3")
    parser.add_argument("--threads", default="128,256")
    parser.add_argument("--parallel-threads", default="128,256")
    parser.add_argument("--recurrence-threads", default="128,256")
    parser.add_argument("--recurrence-block-v", default="16,32,64")
    parser.add_argument("--recurrence-split-carry", default="0,1,2")
    parser.add_argument("--stage-breakdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        default="experiments/gated_deltanet_bwd_prefill_style/results/config_ako_d128.jsonl",
    )
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-known-failures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def _candidate_space(args: argparse.Namespace) -> list[dict[str, int]]:
    fields = {
        "num_stages": _csv_ints(args.num_stages),
        "threads": _csv_ints(args.threads),
        "parallel_threads": _csv_ints(args.parallel_threads),
        "recurrence_threads": _csv_ints(args.recurrence_threads),
        "recurrence_block_v": _csv_ints(args.recurrence_block_v),
        "recurrence_split_carry": _csv_ints(args.recurrence_split_carry),
    }
    candidates = [
        dict(zip(fields.keys(), values, strict=True))
        for values in itertools.product(*fields.values())
    ]
    candidates = [
        candidate
        for candidate in candidates
        if args.dim_v % candidate["recurrence_block_v"] == 0
    ]
    if args.skip_known_failures:
        candidates = [
            candidate
            for candidate in candidates
            if not (
                candidate["recurrence_block_v"] == 16
                and candidate["recurrence_threads"] >= 256
            )
        ]
    return candidates


def _bench_command(args: argparse.Namespace, candidate: dict[str, int]) -> list[str]:
    cmd = [
        sys.executable,
        str(_BENCH),
        "--batch",
        str(args.batch),
        "--heads",
        str(args.heads),
        "--seq-len",
        str(args.seq_len),
        "--dim-k",
        str(args.dim_k),
        "--dim-v",
        str(args.dim_v),
        "--chunk-size",
        str(args.chunk_size),
        "--dtype",
        args.dtype,
        "--layout",
        args.layout,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--trials",
        str(args.trials),
        "--num-stages",
        str(candidate["num_stages"]),
        "--threads",
        str(candidate["threads"]),
        "--parallel-threads",
        str(candidate["parallel_threads"]),
        "--recurrence-threads",
        str(candidate["recurrence_threads"]),
        "--recurrence-block-v",
        str(candidate["recurrence_block_v"]),
    ]
    cmd.extend(["--recurrence-split-carry-mode", str(candidate["recurrence_split_carry"])])
    if args.stage_breakdown:
        cmd.append("--stage-breakdown")
    return cmd


def _parse_bench_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _run_one(args: argparse.Namespace, round_id: int, candidate: dict[str, int]) -> dict[str, Any]:
    started = time.time()
    cmd = _bench_command(args, candidate)
    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "round": round_id,
        "candidate": candidate,
        "cmd": cmd,
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "status": "timeout",
                "elapsed_sec": time.time() - started,
                "stdout_tail": (exc.stdout or "")[-4000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
            }
        )
        return record

    bench = _parse_bench_json(proc.stdout)
    record.update(
        {
            "status": "pass" if proc.returncode == 0 and bench is not None else "fail",
            "returncode": proc.returncode,
            "elapsed_sec": time.time() - started,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    )
    if bench is not None:
        record["bench"] = bench
        record["latency_ms"] = bench.get("legacy_bhsd_full_bwd_ms")
        record["stage_breakdown_ms"] = bench.get("stage_breakdown_ms")
    return record


def _load_records(output: Path) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    records = []
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"status": "bad_json", "raw_line": line})
    return records


def _record_latency(record: dict[str, Any]) -> float | None:
    latency = record.get("latency_ms")
    if isinstance(latency, (int, float)):
        return float(latency)
    bench = record.get("bench")
    if isinstance(bench, dict):
        latency = bench.get("legacy_bhsd_full_bwd_ms")
        if isinstance(latency, (int, float)):
            return float(latency)
    return None


def _summarize(records: list[dict[str, Any]], candidates: list[dict[str, int]]) -> str:
    pass_records = [
        record
        for record in records
        if record.get("status") == "pass" and _record_latency(record) is not None
    ]
    fail_count = sum(1 for record in records if record.get("status") == "fail")
    timeout_count = sum(1 for record in records if record.get("status") == "timeout")
    bad_count = sum(1 for record in records if record.get("status") == "bad_json")
    lines = [
        f"records={len(records)} pass={len(pass_records)} fail={fail_count} "
        f"timeout={timeout_count} bad_json={bad_count} candidate_space={len(candidates)}"
    ]
    for record in sorted(pass_records, key=lambda item: _record_latency(item) or float("inf"))[:5]:
        latency = _record_latency(record)
        lines.append(
            f"best round={record.get('round')} latency_ms={latency:.6f} "
            f"candidate={record.get('candidate')}"
        )
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    labels: dict[str, Any] = {}
    for record in pass_records:
        latency = _record_latency(record)
        candidate = record.get("candidate")
        if latency is None or not isinstance(candidate, dict):
            continue
        key = json.dumps(candidate, sort_keys=True)
        labels[key] = candidate
        grouped[key].append(latency)
    if grouped:
        lines.append("candidate aggregates:")
    for key, latencies in sorted(grouped.items(), key=lambda item: sorted(item[1])[len(item[1]) // 2])[:5]:
        ordered = sorted(latencies)
        median = ordered[len(ordered) // 2]
        lines.append(
            f"candidate count={len(ordered)} best_ms={ordered[0]:.6f} "
            f"median_ms={median:.6f} candidate={labels[key]}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidates = _candidate_space(args)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    output = (_REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_records(output) if args.resume or args.summarize_only else []

    if args.summarize_only:
        print(_summarize(existing, candidates))
        return

    start_round = len(existing) + 1 if args.resume else 1
    rounds = args.rounds
    if args.target_total > 0:
        rounds = max(0, args.target_total - len(existing))
    if rounds <= 0:
        print(_summarize(existing, candidates))
        return

    with open(output, "a", encoding="utf-8") as f:
        for offset in range(rounds):
            round_id = start_round + offset
            candidate = candidates[(round_id - 1) % len(candidates)]
            record = _run_one(args, round_id, candidate)
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            latency = record.get("latency_ms")
            suffix = f" latency_ms={latency:.6f}" if isinstance(latency, float) else ""
            print(f"round={round_id} status={record['status']} candidate={candidate}{suffix}")

    print(_summarize(_load_records(output), candidates))


if __name__ == "__main__":
    main()
