"""Launch a deterministic first-phase config grid for one AKO shape."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(dim) for dim in value.split(","))
    if len(shape) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--start-round", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    runner = Path(__file__).with_name("run_candidate.py")
    configs: list[dict[str, int | bool]] = []
    for consumer_warpgroups, split_k_warps in (
        (4, 2),
        (4, 4),
        (2, 2),
        (4, 1),
        (2, 1),
        (4, 8),
        (2, 4),
    ):
        output_warps = consumer_warpgroups * 4 // split_k_warps
        outputs_per_warp = 64 // output_warps
        for num_stages, tma_activation in itertools.product(
            (2, 3, 4),
            (True, False),
        ):
            config = {
                "consumer_warpgroups": consumer_warpgroups,
                "split_k_warps": split_k_warps,
                "outputs_per_warp": outputs_per_warp,
                "num_stages": num_stages,
                "tma_activation": tma_activation,
            }
            # Round 1 already measured this exact configuration.
            if config == {
                "consumer_warpgroups": 4,
                "split_k_warps": 4,
                "outputs_per_warp": 16,
                "num_stages": 2,
                "tma_activation": True,
            }:
                continue
            configs.append(config)

    configs = configs[args.offset :]
    if args.limit is not None:
        configs = configs[: args.limit]

    shape_arg = ",".join(str(dim) for dim in args.shape)
    for offset, config in enumerate(configs):
        round_number = args.start_round + offset
        candidate_id = (
            f"s1-r{round_number:03d}-"
            f"{config['consumer_warpgroups']}wg-"
            f"split{config['split_k_warps']}-"
            f"o{config['outputs_per_warp']}-"
            f"s{config['num_stages']}-"
            f"{'tma-a' if config['tma_activation'] else 'copy-a'}"
        )
        command = [
            sys.executable,
            str(runner),
            "--candidate-id",
            candidate_id,
            "--shape",
            shape_arg,
            "--consumer-warpgroups",
            str(config["consumer_warpgroups"]),
            "--split-k-warps",
            str(config["split_k_warps"]),
            "--outputs-per-warp",
            str(config["outputs_per_warp"]),
            "--num-stages",
            str(config["num_stages"]),
            "--group-post-scale",
        ]
        if not config["tma_activation"]:
            command.append("--no-tma-activation")
        print(
            "AKO_LAUNCH="
            + json.dumps(
                {"candidate_id": candidate_id, "config": config},
                sort_keys=True,
            ),
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            print(
                "AKO_ATTEMPT="
                + json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "returncode": completed.returncode,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
