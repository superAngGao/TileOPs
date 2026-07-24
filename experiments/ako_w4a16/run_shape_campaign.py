"""Run a resumable pool of valid AKO candidates for one decode shape."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(dim) for dim in value.split(","))
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return shape


def candidate_pool() -> list[dict[str, int | bool]]:
    configs: list[dict[str, int | bool]] = []

    def add_family(
        *,
        block_k: int,
        split_k_warps: int,
        outputs_per_warp: int,
        consumer_warpgroups: int,
        stages: tuple[int, ...],
        nregs: tuple[int, ...],
        tma_activation_values: tuple[bool, ...],
    ) -> None:
        for num_stages in stages:
            for consumer_max_nreg in nregs:
                for tma_activation in tma_activation_values:
                    configs.append(
                        {
                            "block_k": block_k,
                            "split_k_warps": split_k_warps,
                            "outputs_per_warp": outputs_per_warp,
                            "consumer_warpgroups": consumer_warpgroups,
                            "num_stages": num_stages,
                            "consumer_max_nreg": consumer_max_nreg,
                            "tma_activation": tma_activation,
                            "group_metadata_fp16": False,
                            "cache_activation_fp32": False,
                        }
                    )

    add_family(
        block_k=256,
        split_k_warps=4,
        outputs_per_warp=16,
        consumer_warpgroups=4,
        stages=(2, 3, 4),
        nregs=(96, 104, 112),
        tma_activation_values=(True, False),
    )
    add_family(
        block_k=256,
        split_k_warps=2,
        outputs_per_warp=8,
        consumer_warpgroups=4,
        stages=(2, 3, 4),
        nregs=(96, 104, 112),
        tma_activation_values=(True, False),
    )
    add_family(
        block_k=512,
        split_k_warps=2,
        outputs_per_warp=8,
        consumer_warpgroups=4,
        stages=(2, 3, 4),
        nregs=(96, 104, 112),
        tma_activation_values=(False,),
    )
    add_family(
        block_k=1024,
        split_k_warps=1,
        outputs_per_warp=4,
        consumer_warpgroups=4,
        stages=(2, 3, 4),
        nregs=(96, 104, 112),
        tma_activation_values=(False,),
    )
    add_family(
        block_k=256,
        split_k_warps=8,
        outputs_per_warp=32,
        consumer_warpgroups=4,
        stages=(2,),
        nregs=(96, 104, 112),
        tma_activation_values=(True, False),
    )
    add_family(
        block_k=256,
        split_k_warps=2,
        outputs_per_warp=16,
        consumer_warpgroups=2,
        stages=(2, 3, 4),
        nregs=(160, 192, 224, 232),
        tma_activation_values=(True, False),
    )
    add_family(
        block_k=256,
        split_k_warps=1,
        outputs_per_warp=8,
        consumer_warpgroups=2,
        stages=(2, 3, 4),
        nregs=(160, 192, 224, 232),
        tma_activation_values=(True, False),
    )

    # Precision/register-reuse variants are appended after the structural
    # sweep and serve as deterministic backfill if a structural candidate
    # fails to compile for a particular K.
    base = list(configs[:36])
    for config in base:
        variant = dict(config)
        variant["group_metadata_fp16"] = True
        configs.append(variant)
    for config in base:
        variant = dict(config)
        variant["cache_activation_fp32"] = True
        configs.append(variant)

    unique: list[dict[str, int | bool]] = []
    seen: set[str] = set()
    for config in configs:
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--shape-index", type=int, required=True)
    parser.add_argument("--target-valid", type=int, default=75)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    runner = Path(__file__).with_name("run_candidate.py")
    shape_arg = ",".join(str(dim) for dim in args.shape)
    valid_round = args.start_round
    attempts = 0

    for config in candidate_pool()[args.offset:]:
        if valid_round >= args.start_round + args.target_valid:
            break
        candidate_id = (
            f"s{args.shape_index}-r{valid_round:03d}-"
            f"k{config['block_k']}-"
            f"{config['consumer_warpgroups']}wg-"
            f"split{config['split_k_warps']}-"
            f"o{config['outputs_per_warp']}-"
            f"s{config['num_stages']}-"
            f"n{config['consumer_max_nreg']}-"
            f"{'tmaa' if config['tma_activation'] else 'copya'}"
            f"{'-metah' if config['group_metadata_fp16'] else ''}"
            f"{'-accr' if config['cache_activation_fp32'] else ''}"
        )
        command = [
            sys.executable,
            str(runner),
            "--candidate-id",
            candidate_id,
            "--shape",
            shape_arg,
            "--family",
            "tma",
            "--block-k",
            str(config["block_k"]),
            "--num-stages",
            str(config["num_stages"]),
            "--outputs-per-warp",
            str(config["outputs_per_warp"]),
            "--split-k-warps",
            str(config["split_k_warps"]),
            "--consumer-warpgroups",
            str(config["consumer_warpgroups"]),
            "--consumer-max-nreg",
            str(config["consumer_max_nreg"]),
            "--group-post-scale",
        ]
        if not config["tma_activation"]:
            command.append("--no-tma-activation")
        if config["group_metadata_fp16"]:
            command.append("--group-metadata-fp16")
        if config["cache_activation_fp32"]:
            command.append("--cache-activation-fp32")

        attempts += 1
        print(
            "AKO_LAUNCH="
            + json.dumps(
                {
                    "attempt": attempts,
                    "candidate_id": candidate_id,
                    "config": config,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired:
            print(
                "AKO_ATTEMPT="
                + json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "counted_round": False,
                        "reason": "timeout",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        if completed.returncode == 0 and "AKO_RESULT=" in completed.stdout:
            valid_round += 1
        else:
            print(
                "AKO_ATTEMPT="
                + json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "counted_round": False,
                        "returncode": completed.returncode,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    completed_valid = valid_round - args.start_round
    print(
        "AKO_CAMPAIGN="
        + json.dumps(
            {
                "attempts": attempts,
                "completed_valid": completed_valid,
                "shape": args.shape,
                "shape_index": args.shape_index,
                "target_valid": args.target_valid,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if completed_valid != args.target_valid:
        raise RuntimeError(
            f"candidate pool exhausted at {completed_valid} valid rounds")


if __name__ == "__main__":
    main()
