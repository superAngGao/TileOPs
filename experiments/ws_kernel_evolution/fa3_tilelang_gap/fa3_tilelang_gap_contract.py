"""Executable contract for the FP8 GQA FA3-vs-TileLang gap discussion.

This module deliberately does not build a kernel.  It records the desired FA3
steady-state order and the TileLang capabilities that are supported, missing, or
still uncertain.  Keeping this as code gives later reports and slides a single
source of truth without modifying production kernels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    MISSING = "missing"
    UNCERTAIN = "uncertain"


class CapabilityDisposition(str, Enum):
    RESOLVED_IN_TILELANG = "resolved_in_tilelang"
    BLOCKED_BY_TILELANG_API = "blocked_by_tilelang_api"
    NEEDS_AB_PROBE = "needs_ab_probe"
    NEEDS_CORRECTNESS_KERNEL = "needs_correctness_kernel"


@dataclass(frozen=True)
class SteadyStateStep:
    order: int
    name: str
    intent: str
    fa3_contract: str


@dataclass(frozen=True)
class TileLangCapability:
    name: str
    desired: str
    observed: str
    status: CapabilityStatus
    disposition: CapabilityDisposition
    requested_api: str


FA3_STEADY_STATE: tuple[SteadyStateStep, ...] = (
    SteadyStateStep(
        0,
        "prepare_prev_p",
        "Carry P[n-1] in a persistent RS-A register fragment.",
        "tOrP is prepared by the previous iteration and remains in the mainloop scope.",
    ),
    SteadyStateStep(
        1,
        "issue_qk",
        "Issue QK[n] as a grouped WGMMA batch without waiting.",
        "flash::gemm<zero_init=true, wg_wait=-1>(..., tSrS)",
    ),
    SteadyStateStep(
        2,
        "issue_pv",
        "Issue PV[n-1] from already-prepared tOrP without waiting.",
        "flash::gemm<zero_init=false, wg_wait=-1>(..., tOrP, ..., tOrO)",
    ),
    SteadyStateStep(
        3,
        "wait_qk_only",
        "Wait for older QK while allowing newer PV to remain outstanding.",
        "warpgroup_wait<1>()",
    ),
    SteadyStateStep(
        4,
        "softmax_current_qk",
        "Read tSrS safely and run mask/softmax for QK[n].",
        "scoremod, mask, max_get_scale, online_softmax(tSrS)",
    ),
    SteadyStateStep(
        5,
        "wait_pv",
        "Wait for PV[n-1] before consuming/updating tOrO.",
        "warpgroup_wait<0>()",
    ),
    SteadyStateStep(
        6,
        "prepare_next_p",
        "Convert tSrS into the persistent tOrP for the next iteration.",
        "permute_Cregs_fp8 + convert_layout_acc_Aregs + convert_type_out",
    ),
)


TILELANG_CAPABILITIES: tuple[TileLangCapability, ...] = (
    TileLangCapability(
        "persistent_torp_fragment",
        "Allocate tOrP outside the loop and keep it alive across iterations.",
        "Likely expressible with loop-external T.alloc_fragment, but not yet verified by a TileLang correctness kernel.",
        CapabilityStatus.UNCERTAIN,
        CapabilityDisposition.NEEDS_CORRECTNESS_KERNEL,
        "A minimal TileLang smoke kernel proving cross-iteration fragment lifetime.",
    ),
    TileLangCapability(
        "qk_acc_to_pv_areg_fragment",
        "Convert QK accumulator C-layout fp32 registers into PV RS-A fp8 registers.",
        "No high-level fragment reinterpret/permute primitive is currently known.",
        CapabilityStatus.MISSING,
        CapabilityDisposition.BLOCKED_BY_TILELANG_API,
        "T.reinterpret_fragment or T.convert_fragment_layout.",
    ),
    TileLangCapability(
        "full_context_grouped_overlap",
        "Keep QK/PV grouped WGMMA scoreboard shape through the full TileLang context.",
        "Source/PTX can express the schedule, but SASS can degrade to per-QGMMA DEPBAR.",
        CapabilityStatus.UNCERTAIN,
        CapabilityDisposition.NEEDS_AB_PROBE,
        "Documented FA3-style overlap pattern or T.gemm_overlap.",
    ),
    TileLangCapability(
        "tail_fence_ab_control",
        "A/B test operand-fence placement around grouped WGMMA issue.",
        "Tail fence is a hypothesis, not a proven root cause.",
        CapabilityStatus.UNCERTAIN,
        CapabilityDisposition.NEEDS_AB_PROBE,
        "Optional no_tail_fence or manual atom-level controls.",
    ),
    TileLangCapability(
        "exact_cute_smem_layout",
        "Make TMA destination exactly match FA3 SmemLayoutK/SmemLayoutVtMma.",
        "Swizzle flag can match, but elementwise CUTE layout equivalence is not guaranteed.",
        CapabilityStatus.MISSING,
        CapabilityDisposition.BLOCKED_BY_TILELANG_API,
        "CUTE-layout-compatible TMA destination API or layout mapping tool.",
    ),
)


def validate_steady_state(steps: Iterable[SteadyStateStep] = FA3_STEADY_STATE) -> None:
    """Assert the FIFO order needed for FA3's wait_group<1> contract.

    The critical invariant is QK before PV before wait<1>.  Reversing QK and PV
    would make wait<1> preserve QK as the outstanding group, so reading tSrS for
    softmax would be unsafe.
    """

    positions = {step.name: step.order for step in steps}
    required = [
        "prepare_prev_p",
        "issue_qk",
        "issue_pv",
        "wait_qk_only",
        "softmax_current_qk",
        "wait_pv",
        "prepare_next_p",
    ]
    missing = [name for name in required if name not in positions]
    if missing:
        raise AssertionError(f"missing steady-state steps: {missing}")

    if not (
        positions["issue_qk"]
        < positions["issue_pv"]
        < positions["wait_qk_only"]
        < positions["softmax_current_qk"]
        < positions["wait_pv"]
        < positions["prepare_next_p"]
    ):
        raise AssertionError(
            "FA3 overlap order must be QK -> PV -> wait<1> -> softmax -> wait<0> -> prepare P"
        )


def render_capability_table(
    capabilities: Iterable[TileLangCapability] = TILELANG_CAPABILITIES,
) -> str:
    lines = [
        "| Capability | Desired | Observed | Status | Disposition | Requested API |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in capabilities:
        lines.append(
            f"| {item.name} | {item.desired} | {item.observed} | "
            f"{item.status.value} | {item.disposition.value} | {item.requested_api} |"
        )
    return "\n".join(lines)


def resolved_capabilities(
    capabilities: Iterable[TileLangCapability] = TILELANG_CAPABILITIES,
) -> tuple[TileLangCapability, ...]:
    return tuple(
        item
        for item in capabilities
        if item.disposition == CapabilityDisposition.RESOLVED_IN_TILELANG
    )


def blocked_capabilities(
    capabilities: Iterable[TileLangCapability] = TILELANG_CAPABILITIES,
) -> tuple[TileLangCapability, ...]:
    return tuple(
        item
        for item in capabilities
        if item.disposition == CapabilityDisposition.BLOCKED_BY_TILELANG_API
    )


def uncertain_capabilities(
    capabilities: Iterable[TileLangCapability] = TILELANG_CAPABILITIES,
) -> tuple[TileLangCapability, ...]:
    return tuple(
        item
        for item in capabilities
        if item.disposition
        in (
            CapabilityDisposition.NEEDS_AB_PROBE,
            CapabilityDisposition.NEEDS_CORRECTNESS_KERNEL,
        )
    )


def render_steady_state(steps: Iterable[SteadyStateStep] = FA3_STEADY_STATE) -> str:
    validate_steady_state(steps)
    lines = ["Desired FA3 steady state:"]
    for step in sorted(steps, key=lambda item: item.order):
        lines.append(f"{step.order}. {step.name}: {step.intent}")
        lines.append(f"   FA3 contract: {step.fa3_contract}")
    return "\n".join(lines)


def render_discussion_brief() -> str:
    """Render a concise issue/discussion body from the contract data."""

    validate_steady_state()
    lines = [
        "# FP8 GQA FA3-style overlap gaps in TileLang",
        "",
        "We can express a FA3-like source schedule, but the full register, WGMMA,",
        "and shared-memory contract is not yet TileLang-equivalent to FA3.",
        "",
        "## Desired steady state",
        "",
        "```text",
    ]
    for step in sorted(FA3_STEADY_STATE, key=lambda item: item.order):
        lines.append(f"{step.order}. {step.name} - {step.intent}")
    lines.extend(
        [
            "```",
            "",
            "Critical FIFO invariant: QK[n] must issue before PV[n-1], so",
            "`wait_group<1>` waits for QK while leaving PV outstanding.",
            "",
            "## Capability status",
            "",
            render_capability_table(),
            "",
            "## Requests",
            "",
        ]
    )
    for item in blocked_capabilities():
        lines.append(f"- {item.requested_api} for `{item.name}`.")
    for item in uncertain_capabilities():
        lines.append(f"- A documented A/B path for `{item.name}`: {item.requested_api}")
    return "\n".join(lines)


def as_json() -> str:
    payload = {
        "steady_state": [asdict(step) for step in FA3_STEADY_STATE],
        "capabilities": [asdict(item) for item in TILELANG_CAPABILITIES],
        "resolved": [item.name for item in resolved_capabilities()],
        "blocked": [item.name for item in blocked_capabilities()],
        "uncertain": [item.name for item in uncertain_capabilities()],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("table", "steady-state", "discussion", "json"),
        default="table",
        help="Output format.",
    )
    args = parser.parse_args()

    validate_steady_state()
    if args.format == "table":
        print(render_capability_table())
    elif args.format == "steady-state":
        print(render_steady_state())
    elif args.format == "discussion":
        print(render_discussion_brief())
    elif args.format == "json":
        print(as_json())


if __name__ == "__main__":
    main()
