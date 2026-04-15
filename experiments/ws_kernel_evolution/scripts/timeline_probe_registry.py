from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path("/home/ga/TileOPs")


@dataclass(frozen=True)
class MilestoneSpec:
    milestone_id: str
    label: str
    status: str
    description: str
    repo_root: str | None = None
    kernel_entry: str | None = None
    default_shape: str | None = None
    causal: bool | None = None
    block_m: int | None = None
    block_n: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    label: str
    status: str
    probe_kind: str
    script_path: str | None
    script_args: tuple[str, ...] | None
    parser: str | None
    description: str
    milestones: tuple[str, ...]
    notes: str | None = None


MILESTONES: dict[str, MilestoneSpec] = {
    "pre_pr871": MilestoneSpec(
        milestone_id="pre_pr871",
        label="Pre-PR-871 WGMMA",
        status="partial",
        description="Hopper GQA WGMMA-pipelined kernel before warp-specialized producer/consumer pipeline was introduced.",
        repo_root=str(ROOT),
        kernel_entry="tileops.kernels.flash_attn.fwd:GqaFwdWgmmaPipelinedKernel",
        default_shape="B=4 S=4096 H=64 Hkv=8 D=128",
        causal=True,
        block_m=128,
        block_n=128,
        notes="Use this as the pre-WS milestone. Performance harness exists and a coarse tile-timing probe is available; fine-grained in-loop timing currently hits a TileLang/TVM WgmmaSyncRewriter crash.",
    ),
    "pr871_base": MilestoneSpec(
        milestone_id="pr871_base",
        label="PR 871 Base",
        status="planned",
        description="Persistent causal kernel from PR 871 before our locality reorder.",
        repo_root="/tmp/tileops-pr871-base",
        kernel_entry="tileops.kernels.flash_attn.fwd:GqaFwdWsPersistentKernel",
        default_shape="B=4 S=4096 H=64 Hkv=8 D=128",
        causal=True,
        block_m=128,
        block_n=128,
        notes="Performance harness exists; timeline probes still need milestone-specific instrumented scripts.",
    ),
    "pr871_reorder": MilestoneSpec(
        milestone_id="pr871_reorder",
        label="PR 871 + Reorder",
        status="planned",
        description="PR 871 persistent kernel with KV-head-friendly persistent ordering.",
        repo_root="/tmp/tileops-pr871-reorder",
        kernel_entry="tileops.kernels.flash_attn.fwd:GqaFwdWsPersistentKernel",
        default_shape="B=4 S=4096 H=64 Hkv=8 D=128",
        causal=True,
        block_m=128,
        block_n=128,
        notes="Needs instrumented probes for delayed handoff and steady-state split timings.",
    ),
    "anchor_causal": MilestoneSpec(
        milestone_id="anchor_causal",
        label="Anchor Causal",
        status="planned",
        description="Anchor causal kernel with delayed rescale, anchored waits, and L2-friendly persistent ordering.",
        repo_root=str(ROOT),
        kernel_entry="_test_ws_fa3_v2_persistent_anchor_causal.py:build_fa3_v2_persistent_causal",
        default_shape="B=4 S=4096 H=64 Hkv=8 D=128",
        causal=True,
        block_m=128,
        block_n=128,
        notes="Requires dedicated anchor-aware probes because wait semantics differ from plain wait_wgmma.",
    ),
    "ws_current_bn176": MilestoneSpec(
        milestone_id="ws_current_bn176",
        label="Current WS BN176 Reference",
        status="implemented",
        description="Current non-causal BN176 WS reference used for existing timeline analysis and split-barrier measurements.",
        repo_root=str(ROOT),
        kernel_entry="_bench_clock_bn176.py instrumented kernel",
        default_shape="B=4 S=4096 H=64 Hkv=8 D=128",
        causal=False,
        block_m=128,
        block_n=176,
        notes="This is a framework bootstrap node, not one of the four blog milestones.",
    ),
    "ws_current_bn128": MilestoneSpec(
        milestone_id="ws_current_bn128",
        label="Current WS BN128 Reference",
        status="implemented",
        description="Current non-causal BN128 WS reference used by earlier intra-WG measurements.",
        repo_root=str(ROOT),
        kernel_entry="_bench_clock_intra_wg.py instrumented kernel",
        default_shape="B=4 S=4096 H=64 Hkv=8 D=128",
        causal=False,
        block_m=128,
        block_n=128,
    ),
}


PROBES: dict[str, ProbeSpec] = {
    "pre_pr871_coarse_total": ProbeSpec(
        probe_id="pre_pr871_coarse_total",
        label="Pre-PR871 coarse tile timing",
        status="implemented",
        probe_kind="coarse_total",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pre_pr871_wgmma_total_cycle.py"),
        script_args=None,
        parser="named_cycle_summary",
        description="Coarse timing for the pre-WS WGMMA-pipelined kernel: software-pipelined loop body total and epilogue total on the last causal tile.",
        milestones=("pre_pr871",),
        notes="This milestone does not share the WS producer/consumer schedule vocabulary. Fine-grained in-loop clocking currently crashes in WgmmaSyncRewriter, so we start from coarse boundaries.",
    ),
    "current_bn176_steady_state_regions": ProbeSpec(
        probe_id="current_bn176_steady_state_regions",
        label="BN176 steady-state region cycles",
        status="implemented",
        probe_kind="steady_state_regions",
        script_path=str(ROOT / "_bench_clock_bn176.py"),
        script_args=None,
        parser="region_cycles_table",
        description="Measures WG1 steady-state region cycles: wgmma_issue, wait<1>, softmax, wait<0>.",
        milestones=("ws_current_bn176",),
    ),
    "current_bn176_split_barrier": ProbeSpec(
        probe_id="current_bn176_split_barrier",
        label="BN176 split barrier/scheduler",
        status="implemented",
        probe_kind="barrier_split",
        script_path=str(ROOT / "_bench_split_barrier.py"),
        script_args=None,
        parser="split_barrier_summary",
        description="Splits pre-QK rendezvous into barrier_wait(k_full), scheduler_sync, and inferred clear(acc_s).",
        milestones=("ws_current_bn176",),
    ),
    "current_bn128_steady_state_regions": ProbeSpec(
        probe_id="current_bn128_steady_state_regions",
        label="BN128 steady-state region cycles",
        status="implemented",
        probe_kind="steady_state_regions",
        script_path=str(ROOT / "_bench_clock_intra_wg.py"),
        script_args=None,
        parser="region_cycles_table",
        description="Measures WG1 steady-state regions on the BN128 reference schedule.",
        milestones=("ws_current_bn128",),
    ),
    "pr871_base_steady_state_regions": ProbeSpec(
        probe_id="pr871_base_steady_state_regions",
        label="PR 871 steady-state region cycles",
        status="implemented",
        probe_kind="steady_state_regions",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_base_steady_state.py"),
        script_args=("--variant", "base"),
        parser="region_cycles_table",
        description="PR 871-specific clock probe for QK/PV/softmax overlap in the persistent causal kernel.",
        milestones=("pr871_base",),
        notes="Derived from the PR 871 base persistent kernel schedule with WG1 steady-state timestamps.",
    ),
    "pr871_base_scheduler_split": ProbeSpec(
        probe_id="pr871_base_scheduler_split",
        label="PR 871 front-end split",
        status="implemented",
        probe_kind="scheduler_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_frontend_split.py"),
        script_args=("--variant", "base"),
        parser="split_barrier_summary",
        description="PR 871 front-end split into barrier_wait(k_full), scheduler sync, and clear(acc_s).",
        milestones=("pr871_base",),
        notes="Measures WG1 front-end directly before QK issue.",
    ),
    "pr871_base_tail_split": ProbeSpec(
        probe_id="pr871_base_tail_split",
        label="PR 871 tail split",
        status="implemented",
        probe_kind="tail_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_tail_split.py"),
        script_args=("--variant", "base"),
        parser="tail_split_summary",
        description="PR 871 tail split after wait<0>: v_empty handoff and acc_s cast copy.",
        milestones=("pr871_base",),
        notes="Measures WG1 steady-state tail without perturbing the main steady-state region probe.",
    ),
    "pr871_base_core_split": ProbeSpec(
        probe_id="pr871_base_core_split",
        label="PR 871 detailed core split",
        status="implemented",
        probe_kind="core_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_core_split.py"),
        script_args=("--variant", "base"),
        parser="named_cycle_summary",
        description="Detailed PR 871 WG1 steady-state core split for QK issue, rescale, wait(v_full), PV issue, post-wait<1> preamble, softmax core, and wait<0> fence.",
        milestones=("pr871_base",),
        notes="Introduced to remove ambiguity in the old merged 'issue' and 'softmax' interval labels.",
    ),
    "pr871_base_total_cycle": ProbeSpec(
        probe_id="pr871_base_total_cycle",
        label="PR 871 steady-state total cycle",
        status="implemented",
        probe_kind="total_cycle",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_total_cycle.py"),
        script_args=("--variant", "base"),
        parser="total_cycle_summary",
        description="Directly measured WG1 steady-state total cycle count for PR 871 base.",
        milestones=("pr871_base",),
        notes="Measures from pre-k_full wait to post-tail copy for n_idx > 0.",
    ),
    "pr871_reorder_steady_state_regions": ProbeSpec(
        probe_id="pr871_reorder_steady_state_regions",
        label="PR 871 reorder steady-state region cycles",
        status="implemented",
        probe_kind="steady_state_regions",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_base_steady_state.py"),
        script_args=("--variant", "reorder"),
        parser="region_cycles_table",
        description="Reorder-specific clock probe for steady-state overlap using the same instrumented harness as PR 871 base.",
        milestones=("pr871_reorder",),
        notes="Shares the same instrumented harness as PR 871 base, differing only in persistent tile ordering.",
    ),
    "pr871_reorder_scheduler_split": ProbeSpec(
        probe_id="pr871_reorder_scheduler_split",
        label="PR 871 reorder front-end split",
        status="implemented",
        probe_kind="scheduler_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_frontend_split.py"),
        script_args=("--variant", "reorder"),
        parser="split_barrier_summary",
        description="PR 871 reorder front-end split into barrier_wait(k_full), scheduler sync, and clear(acc_s).",
        milestones=("pr871_reorder",),
        notes="Same probe as PR 871 base with reordered persistent tile traversal.",
    ),
    "pr871_reorder_tail_split": ProbeSpec(
        probe_id="pr871_reorder_tail_split",
        label="PR 871 reorder tail split",
        status="implemented",
        probe_kind="tail_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_tail_split.py"),
        script_args=("--variant", "reorder"),
        parser="tail_split_summary",
        description="PR 871 reorder tail split after wait<0>: v_empty handoff and acc_s cast copy.",
        milestones=("pr871_reorder",),
        notes="Same tail probe as PR 871 base with reordered persistent tile traversal.",
    ),
    "pr871_reorder_core_split": ProbeSpec(
        probe_id="pr871_reorder_core_split",
        label="PR 871 reorder detailed core split",
        status="implemented",
        probe_kind="core_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_core_split.py"),
        script_args=("--variant", "reorder"),
        parser="named_cycle_summary",
        description="Detailed reorder WG1 steady-state core split using the same instrumented harness as PR 871 base.",
        milestones=("pr871_reorder",),
        notes="Lets us compare QK/PV launch, preamble, and softmax-core pieces without the old merged labels.",
    ),
    "pr871_reorder_total_cycle": ProbeSpec(
        probe_id="pr871_reorder_total_cycle",
        label="PR 871 reorder steady-state total cycle",
        status="implemented",
        probe_kind="total_cycle",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_pr871_total_cycle.py"),
        script_args=("--variant", "reorder"),
        parser="total_cycle_summary",
        description="Directly measured WG1 steady-state total cycle count for PR 871 reorder.",
        milestones=("pr871_reorder",),
        notes="Measures from pre-k_full wait to post-tail copy for n_idx > 0.",
    ),
    "anchor_causal_steady_state_regions": ProbeSpec(
        probe_id="anchor_causal_steady_state_regions",
        label="Anchor causal steady-state region cycles",
        status="implemented",
        probe_kind="steady_state_regions",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_anchor_causal_steady_state.py"),
        script_args=None,
        parser="region_cycles_table",
        description="Anchor-aware clock probe for delayed rescale and wait_wgmma_anchor<1/0>.",
        milestones=("anchor_causal",),
        notes="Measures WG1 steady-state with anchor-specific wait placement; delayed rescale is outside the measured region table.",
    ),
    "anchor_causal_scheduler_split": ProbeSpec(
        probe_id="anchor_causal_scheduler_split",
        label="Anchor causal scheduler split",
        status="implemented",
        probe_kind="scheduler_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_anchor_causal_scheduler_split.py"),
        script_args=None,
        parser="split_barrier_summary",
        description="Split probe for named-barrier handoff cost in the anchor kernel.",
        milestones=("anchor_causal",),
        notes="Reports barrier_wait(k_full), named-barrier scheduler sync, and zero explicit clear(acc_s) front-end cost.",
    ),
    "anchor_causal_tail_split": ProbeSpec(
        probe_id="anchor_causal_tail_split",
        label="Anchor causal tail split",
        status="implemented",
        probe_kind="tail_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_anchor_causal_tail_split.py"),
        script_args=None,
        parser="tail_split_summary",
        description="Split probe for the post-wait<0> tail in the anchor kernel: v_empty handoff, delayed rescale, and acc_s cast copy.",
        milestones=("anchor_causal",),
        notes="Measures WG1 steady-state tail after wait_wgmma_anchor<0> without perturbing the main steady-state region probe.",
    ),
    "anchor_causal_core_split": ProbeSpec(
        probe_id="anchor_causal_core_split",
        label="Anchor causal detailed core split",
        status="implemented",
        probe_kind="core_split",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_anchor_causal_core_split.py"),
        script_args=None,
        parser="named_cycle_summary",
        description="Detailed anchor WG1 steady-state core split with explicit QK issue, wait(v_full), PV issue, post-wait<1> preamble, softmax core, and wait<0> fence.",
        milestones=("anchor_causal",),
        notes="Uses rescale_before_pv=0 to make the delayed-rescale design visible without mixing it back into the core window.",
    ),
    "anchor_causal_total_cycle": ProbeSpec(
        probe_id="anchor_causal_total_cycle",
        label="Anchor causal steady-state total cycle",
        status="implemented",
        probe_kind="total_cycle",
        script_path=str(ROOT / "experiments/ws_kernel_evolution/scripts/bench_timeline_anchor_causal_total_cycle.py"),
        script_args=None,
        parser="total_cycle_summary",
        description="Directly measured WG1 steady-state total cycle count for anchor causal.",
        milestones=("anchor_causal",),
        notes="Measures from pre-k_full wait to post-delayed-rescale tail for n_idx > 0.",
    ),
}


def milestone_dict() -> dict[str, dict]:
    return {key: asdict(value) for key, value in MILESTONES.items()}


def probe_dict() -> dict[str, dict]:
    return {key: asdict(value) for key, value in PROBES.items()}
