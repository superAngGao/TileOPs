"""SASS scoreboard counter for the FP8 GQA boundary demo.

The tool is intentionally text-only: feed it `nvdisasm` output and it reports
the WGMMA/scoreboard shape used in the TileLang-vs-FA3 discussion.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


QGMMA_RE = re.compile(r"\bQGMMA(?:\.[A-Za-z0-9]+)*\.(64x\d+x32)\b")


@dataclass(frozen=True)
class SassScoreboardSummary:
    qgmma: int
    qgmma_m64n224k32: int
    qgmma_m64n128k32: int
    qgmma_m64n32k32: int
    warpgroup_depbar: int
    warpgroup_arrive: int
    ldl: int
    stl: int
    qgmma_runs: tuple[int, ...]
    max_qgmma_run: int

    @property
    def depbar_per_qgmma(self) -> float:
        return self.warpgroup_depbar / self.qgmma if self.qgmma else 0.0

    @property
    def arrive_per_qgmma(self) -> float:
        return self.warpgroup_arrive / self.qgmma if self.qgmma else 0.0

    @property
    def looks_per_qgmma_scoreboard(self) -> bool:
        return self.qgmma > 0 and self.warpgroup_depbar >= self.qgmma * 0.8


def summarize_sass(text: str) -> SassScoreboardSummary:
    runs: list[int] = []
    current_run = 0
    qgmma_shapes: list[str] = []

    for line in text.splitlines():
        match = QGMMA_RE.search(line)
        if match:
            current_run += 1
            qgmma_shapes.append(match.group(1))
            continue
        if current_run:
            runs.append(current_run)
            current_run = 0
    if current_run:
        runs.append(current_run)

    return SassScoreboardSummary(
        qgmma=len(qgmma_shapes),
        qgmma_m64n224k32=qgmma_shapes.count("64x224x32"),
        qgmma_m64n128k32=qgmma_shapes.count("64x128x32"),
        qgmma_m64n32k32=qgmma_shapes.count("64x32x32"),
        warpgroup_depbar=len(re.findall(r"\bWARPGROUP\.DEPBAR\b", text)),
        warpgroup_arrive=len(re.findall(r"\bWARPGROUP\.ARRIVE\b", text)),
        ldl=len(re.findall(r"\bLDL\b", text)),
        stl=len(re.findall(r"\bSTL\b", text)),
        qgmma_runs=tuple(runs),
        max_qgmma_run=max(runs, default=0),
    )


def render_summary(summary: SassScoreboardSummary) -> str:
    return "\n".join(
        [
            f"QGMMA={summary.qgmma}",
            f"QGMMA.64x224x32={summary.qgmma_m64n224k32}",
            f"QGMMA.64x128x32={summary.qgmma_m64n128k32}",
            f"QGMMA.64x32x32={summary.qgmma_m64n32k32}",
            f"WARPGROUP.DEPBAR={summary.warpgroup_depbar}",
            f"WARPGROUP.ARRIVE={summary.warpgroup_arrive}",
            f"LDL={summary.ldl}",
            f"STL={summary.stl}",
            f"QGMMA runs={list(summary.qgmma_runs)}",
            f"max QGMMA run={summary.max_qgmma_run}",
            f"DEPBAR/QGMMA={summary.depbar_per_qgmma:.3f}",
            f"ARRIVE/QGMMA={summary.arrive_per_qgmma:.3f}",
            f"per-QGMMA-like={summary.looks_per_qgmma_scoreboard}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", type=Path, help="Path to nvdisasm text output.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    args = parser.parse_args()

    summary = summarize_sass(args.sass.read_text())
    if args.json:
        payload = {
            **asdict(summary),
            "depbar_per_qgmma": summary.depbar_per_qgmma,
            "arrive_per_qgmma": summary.arrive_per_qgmma,
            "looks_per_qgmma_scoreboard": summary.looks_per_qgmma_scoreboard,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_summary(summary))


if __name__ == "__main__":
    main()
