"""Evaluate one AKO pipeline structure with an internal parameter search.

A structural round is deliberately distinct from a parameter trial.  This
driver writes one round record only after one or more parameter trials for a
declared pipeline/dataflow structure compile, pass correctness, and finish the
TileOps cold-cache screening benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STRUCTURAL_FIELDS = (
    "load_pipeline",
    "buffer_handoff",
    "activation_path",
    "weight_path",
    "metadata_path",
    "decode_schedule",
    "compute_instruction",
    "accumulator_path",
    "reduction_path",
    "output_schedule",
    "weight_layout",
)

PARAMETER_FIELDS = (
    "block_n",
    "block_k",
    "num_stages",
    "outputs_per_warp",
    "split_k_warps",
    "consumer_warpgroups",
    "consumer_max_nreg",
    "global_split_k",
    "reduce_threads",
)


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


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(dim) for dim in value.split(","))
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return shape


def parse_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("AKO_RESULT="):
            return json.loads(line.removeprefix("AKO_RESULT="))
    raise ValueError("runner did not emit AKO_RESULT")


def validate_spec(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    missing = [
        field
        for field in (
            "round_id",
            "parent_id",
            "hypothesis",
            "transform",
            "structure",
            "runner_flags",
        )
        if field not in spec
    ]
    if missing:
        raise ValueError(f"structural spec is missing {missing}")
    structure = spec["structure"]
    missing_structure = [
        field for field in STRUCTURAL_FIELDS if field not in structure
    ]
    if missing_structure:
        raise ValueError(
            f"structure is missing semantic fields {missing_structure}")
    illegal_parameters = sorted(set(structure) & set(PARAMETER_FIELDS))
    if illegal_parameters:
        raise ValueError(
            "parameter-only fields cannot be part of a structural signature: "
            f"{illegal_parameters}"
        )
    signature = hashlib.sha256(canonical(structure).encode()).hexdigest()
    return signature, structure


def parameter_trials(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    unknown = sorted(set(grid) - set(PARAMETER_FIELDS))
    if unknown:
        raise ValueError(f"unknown parameter-grid fields: {unknown}")
    fields = sorted(grid)
    return [
        dict(zip(fields, values, strict=True))
        for values in itertools.product(*(grid[field] for field in fields))
    ]


def runner_command(
    round_id: str,
    shape: tuple[int, int, int],
    runner_flags: list[str],
    trial_index: int,
    parameters: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.ako_w4a16.run_candidate",
        "--candidate-id",
        f"{round_id}-t{trial_index:03d}",
        "--shape",
        ",".join(str(dim) for dim in shape),
        *runner_flags,
    ]
    for field, value in parameters.items():
        flag = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--shape", type=parse_shape, default=(1, 8192, 81920))
    parser.add_argument("--records-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    signature, structure = validate_spec(spec)
    records_dir = args.records_dir
    records_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = records_dir / "shape4_structural_rounds.jsonl"
    trials_path = records_dir / "shape4_structural_trials.jsonl"
    attempts_path = records_dir / "shape4_structural_attempts.jsonl"
    baseline_path = records_dir / "shape4_structural_baseline.json"

    existing_rounds = load_jsonl(rounds_path)
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.exists()
        else None
    )
    if any(row["round_id"] == spec["round_id"] for row in existing_rounds):
        raise ValueError(f"duplicate round_id {spec['round_id']}")
    known_structures = existing_rounds + ([baseline] if baseline else [])
    if any(row["structure_signature"] == signature for row in known_structures):
        raise ValueError(
            "this pipeline/dataflow structure already consumed a round")
    if known_structures:
        known_ids = {row["round_id"] for row in known_structures}
        if spec["parent_id"] not in known_ids:
            raise ValueError(
                f"unknown parent_id {spec['parent_id']}; expected the frozen "
                "baseline or a counted round")
        previous = next(
            row for row in known_structures
            if row["round_id"] == spec["parent_id"]
        )
        changed = {
            field: {
                "from": previous["structure"].get(field),
                "to": structure.get(field),
            }
            for field in STRUCTURAL_FIELDS
            if previous["structure"].get(field) != structure.get(field)
        }
        if not changed:
            raise ValueError(
                "declared transform changes no pipeline/dataflow field")
    elif spec["parent_id"] is not None:
        raise ValueError(
            "without a baseline, the first structural round needs parent_id=null")
    else:
        changed = {field: {"from": None, "to": structure[field]}
                   for field in STRUCTURAL_FIELDS}

    if "parameter_trials" in spec:
        trials = spec["parameter_trials"]
        for trial in trials:
            unknown = sorted(set(trial) - set(PARAMETER_FIELDS))
            if unknown:
                raise ValueError(f"unknown parameter-trial fields: {unknown}")
    elif "parameter_grid" in spec:
        trials = parameter_trials(spec["parameter_grid"])
    else:
        raise ValueError("spec needs parameter_trials or parameter_grid")
    if not trials:
        raise ValueError("parameter search produced no trials")
    successes: list[dict[str, Any]] = []
    started = time.time()

    for trial_index, parameters in enumerate(trials, start=1):
        command = runner_command(
            spec["round_id"],
            args.shape,
            spec["runner_flags"],
            trial_index,
            parameters,
        )
        launch = {
            "round_id": spec["round_id"],
            "structure_signature": signature,
            "trial_index": trial_index,
            "parameters": parameters,
            "command": command,
        }
        print("AKO_STRUCTURAL_LAUNCH=" + canonical(launch), flush=True)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            attempt = {
                **launch,
                "counted_round": False,
                "reason": "timeout",
            }
            append_jsonl(attempts_path, attempt)
            print("AKO_STRUCTURAL_ATTEMPT=" + canonical(attempt), flush=True)
            continue

        if completed.returncode != 0:
            attempt = {
                **launch,
                "counted_round": False,
                "reason": "runner_failed",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:],
            }
            append_jsonl(attempts_path, attempt)
            print("AKO_STRUCTURAL_ATTEMPT=" + canonical(attempt), flush=True)
            continue

        try:
            result = parse_result(completed.stdout)
        except ValueError as error:
            attempt = {
                **launch,
                "counted_round": False,
                "reason": str(error),
            }
            append_jsonl(attempts_path, attempt)
            print("AKO_STRUCTURAL_ATTEMPT=" + canonical(attempt), flush=True)
            continue
        trial = {
            **launch,
            "result": result,
        }
        append_jsonl(trials_path, trial)
        successes.append(trial)
        print("AKO_STRUCTURAL_TRIAL=" + canonical(trial), flush=True)

    if not successes:
        raise RuntimeError(
            "no parameter trial compiled, passed correctness, and benchmarked; "
            "the attempted structure does not consume a round"
        )

    winner = min(successes, key=lambda trial: trial["result"]["latency_ms"])
    round_record = {
        "round_number": len(existing_rounds) + 1,
        "round_id": spec["round_id"],
        "parent_id": spec["parent_id"],
        "hypothesis": spec["hypothesis"],
        "transform": spec["transform"],
        "changed_structural_fields": changed,
        "structure": structure,
        "structure_signature": signature,
        "trial_count": len(trials),
        "successful_trial_count": len(successes),
        "winner": winner,
        "elapsed_seconds": time.time() - started,
    }
    append_jsonl(rounds_path, round_record)
    print("AKO_STRUCTURAL_ROUND=" + canonical(round_record), flush=True)


if __name__ == "__main__":
    main()
