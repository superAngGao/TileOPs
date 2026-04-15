#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

from timeline_probe_registry import MILESTONES, PROBES, milestone_dict, probe_dict


ROOT = Path("/home/ga/TileOPs")
DATA_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "data"


def parse_region_cycles_table(stdout: str) -> dict:
    cycles = {}
    samples = None
    total = None
    delayed_rescale = None
    for line in stdout.splitlines():
        region_match = re.match(r"\s+(wgmma_issue|wait<1> \(QK\)|softmax|wait<0> \(PV res\)|TOTAL)\s+(\d+)\s+([0-9.]+)%", line)
        if region_match:
            label = region_match.group(1)
            value = float(region_match.group(2))
            if label == "TOTAL":
                total = value
            else:
                cycles[label] = value
            continue
        delayed_match = re.search(r"Delayed rescale \(outside TOTAL\):\s+([0-9.]+)\s+cycles", line)
        if delayed_match:
            delayed_rescale = float(delayed_match.group(1))
            continue
        sample_match = re.search(r"Samples:\s+(\d+)", line)
        if sample_match:
            samples = int(sample_match.group(1))
    if not cycles:
        raise ValueError("failed to parse region cycle table")
    parsed = {
        "cycles": cycles,
        "total_cycles": total,
        "samples": samples,
    }
    if delayed_rescale is not None:
        parsed["delayed_rescale_outside_total"] = delayed_rescale
    return parsed


def parse_split_barrier_summary(stdout: str) -> dict:
    patterns = {
        "barrier_wait_k_full": r"barrier_wait\(k_full\):\s+([0-9.]+)\s+cycles",
        "scheduler_sync": r"scheduler_sync:\s+([0-9.]+)\s+cycles",
        "sum_cycles": r"sum:\s+([0-9.]+)\s+cycles",
        "clear_acc_s_inferred": r"clear\(acc_s\).*?([0-9.\-]+)\s+cycles",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    if not parsed:
        raise ValueError("failed to parse split barrier summary")
    return parsed


def parse_tail_split_summary(stdout: str) -> dict:
    patterns = {
        "v_empty_handoff": r"v_empty_handoff:\s+([0-9.]+)\s+cycles",
        "delayed_rescale": r"delayed_rescale:\s+([0-9.]+)\s+cycles",
        "acc_s_cast_copy": r"acc_s_cast_copy:\s+([0-9.]+)\s+cycles",
        "sum_cycles": r"sum:\s+([0-9.]+)\s+cycles",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    if not parsed:
        raise ValueError("failed to parse tail split summary")
    return parsed


def parse_total_cycle_summary(stdout: str) -> dict:
    patterns = {
        "steady_iter_total": r"steady_iter_total:\s+([0-9.]+)\s+cycles",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            parsed[key] = float(match.group(1))
    sample_match = re.search(r"Samples:\s+(\d+)", stdout)
    if sample_match:
        parsed["samples"] = int(sample_match.group(1))
    if not parsed:
        raise ValueError("failed to parse total cycle summary")
    return parsed


def parse_named_cycle_summary(stdout: str) -> dict:
    parsed = {}
    for line in stdout.splitlines():
        match = re.match(r"\s*([a-zA-Z0-9_<>]+):\s+([0-9.]+)\s+cycles", line)
        if match:
            parsed[match.group(1)] = float(match.group(2))
            continue
        sample_match = re.search(r"Samples:\s+(\d+)", line)
        if sample_match:
            parsed["samples"] = int(sample_match.group(1))
    if not parsed:
        raise ValueError("failed to parse named cycle summary")
    return parsed


PARSERS = {
    "region_cycles_table": parse_region_cycles_table,
    "split_barrier_summary": parse_split_barrier_summary,
    "tail_split_summary": parse_tail_split_summary,
    "total_cycle_summary": parse_total_cycle_summary,
    "named_cycle_summary": parse_named_cycle_summary,
}


def resolve_probes(args: argparse.Namespace) -> list[str]:
    if args.probes:
        return args.probes
    if args.milestone is None:
        raise ValueError("provide --milestone or explicit --probes")
    probes = [probe_id for probe_id, probe in PROBES.items() if args.milestone in probe.milestones]
    if not probes:
        raise ValueError(f"no probes registered for milestone {args.milestone}")
    return probes


def run_probe(probe_id: str, env: dict[str, str]) -> dict:
    probe = PROBES[probe_id]
    result = {
        "probe_id": probe_id,
        "label": probe.label,
        "probe_kind": probe.probe_kind,
        "status": probe.status,
        "description": probe.description,
        "milestones": list(probe.milestones),
    }
    if probe.status != "implemented":
        if probe.notes:
            result["notes"] = probe.notes
        return result
    if probe.script_path is None or probe.parser is None:
        raise ValueError(f"implemented probe {probe_id} missing script_path/parser")

    cmd = [sys.executable, probe.script_path]
    if probe.script_args:
        cmd.extend(probe.script_args)
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    result["command"] = cmd
    result["returncode"] = proc.returncode
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    if proc.returncode != 0:
        result["status"] = "error"
        return result

    parser_fn = PARSERS[probe.parser]
    result["parsed"] = parser_fn(proc.stdout)
    result["status"] = "ok"
    return result


def build_output(args: argparse.Namespace, probe_ids: list[str], results: list[dict]) -> dict:
    env_subset = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "TILELANG_CLEANUP_TEMP_FILES": os.environ.get("TILELANG_CLEANUP_TEMP_FILES"),
        "V2P_NUM_SMS": os.environ.get("V2P_NUM_SMS"),
    }
    payload = {
        "study": "ws_kernel_evolution",
        "kind": "timeline_probe_run",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "milestone": args.milestone,
        "milestone_spec": milestone_dict().get(args.milestone) if args.milestone else None,
        "requested_probes": probe_ids,
        "probe_registry": {probe_id: probe_dict()[probe_id] for probe_id in probe_ids},
        "environment": env_subset,
        "results": results,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-milestones", action="store_true")
    parser.add_argument("--list-probes", action="store_true")
    parser.add_argument("--milestone", choices=sorted(MILESTONES))
    parser.add_argument("--probes", nargs="+", choices=sorted(PROBES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.list_milestones:
        print(json.dumps(milestone_dict(), indent=2))
        return
    if args.list_probes:
        print(json.dumps(probe_dict(), indent=2))
        return

    probe_ids = resolve_probes(args)
    env = os.environ.copy()
    env.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")
    results = []
    for probe_id in probe_ids:
        print(f"Running {probe_id}...")
        results.append(run_probe(probe_id, env))

    payload = build_output(args, probe_ids, results)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        milestone = args.milestone or "ad_hoc"
        args.output = DATA_DIR / f"{stamp}_timeline_probes_{milestone}.json"
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
