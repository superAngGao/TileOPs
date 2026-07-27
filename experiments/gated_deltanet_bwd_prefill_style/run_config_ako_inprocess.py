"""Fast in-process AKO sweep for known-valid Gated DeltaNet backward configs.

Use ``run_config_ako.py`` first when exploring unknown candidates: it launches
fresh subprocesses and records lowering/runtime failures safely. Once a
candidate family is known-valid, this script reuses one Python process and
TileLang's compile cache to collect repeated timing samples more quickly.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.gated_deltanet_bwd_prefill_style.bench_current_bwd import (  # noqa: E402
    BenchConfig,
    _bench_full_bwd,
    _cuda_env,
    _default_recurrence_block_v,
    _default_recurrence_split_carry,
    _make_inputs,
    _time_layout_convert,
    _to_bhsd,
)


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
    parser.add_argument("--num-stages", default="1,2")
    parser.add_argument("--threads", default="128,256")
    parser.add_argument("--parallel-threads", default="256")
    parser.add_argument("--recurrence-threads", default="128,256")
    parser.add_argument("--recurrence-block-v", default="16,32,64")
    parser.add_argument("--recurrence-split-carry", default="0,1,2")
    parser.add_argument("--skip-known-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--output",
        default="experiments/gated_deltanet_bwd_prefill_style/results/config_ako_d128_inprocess_300.jsonl",
    )
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


def _load_records(output: Path) -> list[dict[str, Any]]:
    if not output.exists():
        return []
    records = []
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _base_config(args: argparse.Namespace) -> BenchConfig:
    default_split_carry = _default_recurrence_split_carry(args.dim_v, args.chunk_size)
    default_threads = 128 if default_split_carry else (256 if args.chunk_size >= 64 else 128)
    default_parallel_threads = 256 if default_split_carry else default_threads
    return BenchConfig(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        dim_k=args.dim_k,
        dim_v=args.dim_v,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        layout="bhsd" if args.layout == "bhtd" else args.layout,
        warmup=args.warmup,
        repeat=args.repeat,
        trials=args.trials,
        stage_breakdown=False,
        num_stages=2,
        threads=default_threads,
        parallel_threads=default_parallel_threads,
        recurrence_threads=128 if default_split_carry else default_threads,
        recurrence_block_v=_default_recurrence_block_v(args.dim_v, args.chunk_size),
        recurrence_split_carry=default_split_carry,
    )


def _config_for(base: BenchConfig, candidate: dict[str, int]) -> BenchConfig:
    cfg = BenchConfig(**asdict(base))
    cfg.num_stages = candidate["num_stages"]
    cfg.threads = candidate["threads"]
    cfg.parallel_threads = candidate["parallel_threads"]
    cfg.recurrence_threads = candidate["recurrence_threads"]
    cfg.recurrence_block_v = candidate["recurrence_block_v"]
    cfg.recurrence_split_carry = candidate["recurrence_split_carry"]
    return cfg


def _summarize(records: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    labels: dict[str, Any] = {}
    passes = 0
    for record in records:
        if record.get("status") != "pass":
            continue
        latency = record.get("latency_ms")
        candidate = record.get("candidate")
        if not isinstance(latency, (int, float)) or not isinstance(candidate, dict):
            continue
        passes += 1
        key = json.dumps(candidate, sort_keys=True)
        labels[key] = candidate
        grouped[key].append(float(latency))
    lines = [f"records={len(records)} pass={passes} fail={len(records) - passes}"]
    for key, values in sorted(grouped.items(), key=lambda item: sorted(item[1])[len(item[1]) // 2])[:5]:
        ordered = sorted(values)
        lines.append(
            f"candidate count={len(ordered)} best_ms={ordered[0]:.6f} "
            f"median_ms={ordered[len(ordered)//2]:.6f} candidate={labels[key]}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidates = _candidate_space(args)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    output = (_REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_records(output) if args.resume else []

    if args.summarize_only:
        print(_summarize(existing))
        return

    rounds = args.rounds
    if args.target_total > 0:
        rounds = max(0, args.target_total - len(existing))
    if rounds <= 0:
        print(_summarize(existing))
        return

    base = _base_config(args)
    inputs = _make_inputs(base)
    layout_ms = _time_layout_convert(base, inputs)
    inputs_bhsd = _to_bhsd(base, *inputs)
    env = _cuda_env()
    start_round = len(existing) + 1 if args.resume else 1

    with open(output, "a", encoding="utf-8") as f:
        for offset in range(rounds):
            round_id = start_round + offset
            candidate = candidates[(round_id - 1) % len(candidates)]
            cfg = _config_for(base, candidate)
            started = time.time()
            record: dict[str, Any] = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "round": round_id,
                "candidate": candidate,
                "config": asdict(cfg),
                "env": env,
                "bthd_to_bhsd_layout_convert_ms": layout_ms,
                "mode": "inprocess_known_valid",
            }
            try:
                latency = _bench_full_bwd(cfg, inputs_bhsd)
                record.update({"status": "pass", "latency_ms": latency})
            except Exception as exc:  # noqa: BLE001 - failure is search data.
                record.update(
                    {
                        "status": "fail",
                        "error": repr(exc),
                        "traceback_tail": traceback.format_exc()[-4000:],
                    }
                )
            record["elapsed_sec"] = time.time() - started
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            suffix = ""
            if isinstance(record.get("latency_ms"), float):
                suffix = f" latency_ms={record['latency_ms']:.6f}"
            print(f"round={round_id} status={record['status']} candidate={candidate}{suffix}")

    print(_summarize(_load_records(output)))


if __name__ == "__main__":
    main()
