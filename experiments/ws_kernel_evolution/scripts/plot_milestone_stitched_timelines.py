#!/usr/bin/env python3
"""Plot measured milestone timelines with an additional NCU tensor-pipe panel.

This figure intentionally separates the measured regions into four panels:

- front-end split
- detailed steady-state core split
- tail split
- Tensor Pipe Utilization (NCU)

The goal is to preserve intuition while avoiding the misleading impression that
these independently measured pieces add up to a real per-iteration runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path("/home/ga/TileOPs")
DATA_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "data"
FIG_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "figures"
OUT = FIG_DIR / "milestone_stitched_timelines.png"


COLORS = {
    "front_wait": "#F4A261",
    "front_sched": "#E9C46A",
    "front_clear": "#D62828",
    "core_qk_issue": "#457B9D",
    "core_rescale": "#A8DADC",
    "core_wait_v": "#BDBDBD",
    "core_pv_issue": "#1D3557",
    "core_wait1": "#8D99AE",
    "core_preamble": "#F28482",
    "core_softmax": "#E76F51",
    "core_wait0": "#6C757D",
    "tail_handoff": "#2A9D8F",
    "tail_copy": "#6C757D",
    "tail_rescale": "#7B2CBF",
}


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def get_probe(data: dict, probe_id: str) -> dict:
    for result in data["results"]:
        if result["probe_id"] == probe_id:
            return result["parsed"]
    raise KeyError(probe_id)


def build_rows() -> list[dict]:
    base = load_json("20260415_timeline_probes_pr871_base.json")
    reorder = load_json("20260415_timeline_probes_pr871_reorder.json")
    anchor = load_json("20260415_timeline_probes_anchor_causal.json")
    ncu = load_json("20260415_ncu_tensor_pipe_milestones_4k.json")

    ncu_map = {row["variant"]: row for row in ncu["results"]}

    return [
        {
            "label": "PR871 Base",
            "measured_total": get_probe(base, "pr871_base_total_cycle")["steady_iter_total"],
            "tensor_pct": ncu_map["pr871_base"]["ncu"]["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"],
            "front": [
                ("wait(k_full)", get_probe(base, "pr871_base_scheduler_split")["barrier_wait_k_full"], COLORS["front_wait"]),
                ("scheduler", get_probe(base, "pr871_base_scheduler_split")["scheduler_sync"], COLORS["front_sched"]),
                ("clear", get_probe(base, "pr871_base_scheduler_split")["clear_acc_s_inferred"], COLORS["front_clear"]),
            ],
            "core": [
                ("QK", get_probe(base, "pr871_base_core_split")["qk_issue"], COLORS["core_qk_issue"]),
                ("rescale", get_probe(base, "pr871_base_core_split")["rescale_before_pv"], COLORS["core_rescale"]),
                ("wait(v)", get_probe(base, "pr871_base_core_split")["wait_v_full"], COLORS["core_wait_v"]),
                ("PV", get_probe(base, "pr871_base_core_split")["pv_issue"], COLORS["core_pv_issue"]),
                ("wait1", get_probe(base, "pr871_base_core_split")["wait1_fence"], COLORS["core_wait1"]),
                ("preamble", get_probe(base, "pr871_base_core_split")["post_wait1_preamble"], COLORS["core_preamble"]),
                ("softmax", get_probe(base, "pr871_base_core_split")["softmax_core"], COLORS["core_softmax"]),
                ("wait0", get_probe(base, "pr871_base_core_split")["wait0_fence"], COLORS["core_wait0"]),
            ],
            "tail": [
                ("v_empty", get_probe(base, "pr871_base_tail_split")["v_empty_handoff"], COLORS["tail_handoff"]),
                ("acc_s copy", get_probe(base, "pr871_base_tail_split")["acc_s_cast_copy"], COLORS["tail_copy"]),
            ],
        },
        {
            "label": "PR871 Reorder",
            "measured_total": get_probe(reorder, "pr871_reorder_total_cycle")["steady_iter_total"],
            "tensor_pct": ncu_map["pr871_reorder"]["ncu"]["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"],
            "front": [
                ("wait(k_full)", get_probe(reorder, "pr871_reorder_scheduler_split")["barrier_wait_k_full"], COLORS["front_wait"]),
                ("scheduler", get_probe(reorder, "pr871_reorder_scheduler_split")["scheduler_sync"], COLORS["front_sched"]),
                ("clear", get_probe(reorder, "pr871_reorder_scheduler_split")["clear_acc_s_inferred"], COLORS["front_clear"]),
            ],
            "core": [
                ("QK", get_probe(reorder, "pr871_reorder_core_split")["qk_issue"], COLORS["core_qk_issue"]),
                ("rescale", get_probe(reorder, "pr871_reorder_core_split")["rescale_before_pv"], COLORS["core_rescale"]),
                ("wait(v)", get_probe(reorder, "pr871_reorder_core_split")["wait_v_full"], COLORS["core_wait_v"]),
                ("PV", get_probe(reorder, "pr871_reorder_core_split")["pv_issue"], COLORS["core_pv_issue"]),
                ("wait1", get_probe(reorder, "pr871_reorder_core_split")["wait1_fence"], COLORS["core_wait1"]),
                ("preamble", get_probe(reorder, "pr871_reorder_core_split")["post_wait1_preamble"], COLORS["core_preamble"]),
                ("softmax", get_probe(reorder, "pr871_reorder_core_split")["softmax_core"], COLORS["core_softmax"]),
                ("wait0", get_probe(reorder, "pr871_reorder_core_split")["wait0_fence"], COLORS["core_wait0"]),
            ],
            "tail": [
                ("v_empty", get_probe(reorder, "pr871_reorder_tail_split")["v_empty_handoff"], COLORS["tail_handoff"]),
                ("acc_s copy", get_probe(reorder, "pr871_reorder_tail_split")["acc_s_cast_copy"], COLORS["tail_copy"]),
            ],
        },
        {
            "label": "Anchor Causal",
            "measured_total": get_probe(anchor, "anchor_causal_total_cycle")["steady_iter_total"],
            "tensor_pct": ncu_map["anchor_causal"]["ncu"]["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"],
            "front": [
                ("wait(k_full)", get_probe(anchor, "anchor_causal_scheduler_split")["barrier_wait_k_full"], COLORS["front_wait"]),
                ("scheduler", get_probe(anchor, "anchor_causal_scheduler_split")["scheduler_sync"], COLORS["front_sched"]),
            ],
            "core": [
                ("QK", get_probe(anchor, "anchor_causal_core_split")["qk_issue"], COLORS["core_qk_issue"]),
                ("rescale", get_probe(anchor, "anchor_causal_core_split")["rescale_before_pv"], COLORS["core_rescale"]),
                ("wait(v)", get_probe(anchor, "anchor_causal_core_split")["wait_v_full"], COLORS["core_wait_v"]),
                ("PV", get_probe(anchor, "anchor_causal_core_split")["pv_issue"], COLORS["core_pv_issue"]),
                ("wait1", get_probe(anchor, "anchor_causal_core_split")["wait1_fence"], COLORS["core_wait1"]),
                ("preamble", get_probe(anchor, "anchor_causal_core_split")["post_wait1_preamble"], COLORS["core_preamble"]),
                ("softmax", get_probe(anchor, "anchor_causal_core_split")["softmax_core"], COLORS["core_softmax"]),
                ("wait0", get_probe(anchor, "anchor_causal_core_split")["wait0_fence"], COLORS["core_wait0"]),
            ],
            "tail": [
                ("v_empty", get_probe(anchor, "anchor_causal_tail_split")["v_empty_handoff"], COLORS["tail_handoff"]),
                ("delayed rescale", get_probe(anchor, "anchor_causal_tail_split")["delayed_rescale"], COLORS["tail_rescale"]),
                ("acc_s copy", get_probe(anchor, "anchor_causal_tail_split")["acc_s_cast_copy"], COLORS["tail_copy"]),
            ],
        },
    ]


def draw_panel(ax, rows: list[dict], key: str, title: str) -> None:
    y_positions = [2.0, 1.0, 0.0]
    max_sum = max(sum(width for _, width, _ in row[key]) for row in rows)

    for y, row in zip(y_positions, rows):
        cursor = 0.0
        for label, width, color in row[key]:
            ax.barh(y, width, left=cursor, height=0.6, color=color, edgecolor="white", linewidth=1.0)
            if width >= 70:
                ax.text(
                    cursor + width / 2,
                    y,
                    f"{label}\n{width:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if color not in {COLORS["front_wait"], COLORS["front_sched"], COLORS["core_rescale"], COLORS["core_wait_v"], COLORS["core_wait1"], COLORS["tail_handoff"]} else "#1F2937",
                    fontweight="bold",
                    linespacing=0.9,
                )
            cursor += width

        ax.text(
            min(cursor + 14, max_sum + 10),
            y,
            f"sum {cursor:.0f}",
            ha="left",
            va="center",
            fontsize=9,
            color="#111827",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max_sum + 110)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{row['label']}\nmeasured total {row['measured_total']:.0f}" for row in rows],
        fontsize=10,
        fontweight="bold",
    )


def draw_tensor_panel(ax, rows: list[dict], title: str) -> None:
    y_positions = [2.0, 1.0, 0.0]
    xmax = 100.0

    for y, row in zip(y_positions, rows):
        pct = row["tensor_pct"]
        ax.barh(y, pct, height=0.6, color=COLORS["core_pv_issue"], edgecolor="white", linewidth=1.0)
        ax.barh(y, xmax - pct, left=pct, height=0.6, color="#E5E7EB", edgecolor="white", linewidth=1.0)
        ax.text(
            min(pct / 2, max(pct - 6, 5)),
            y,
            f"{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        ax.text(
            pct + 2.0,
            y,
            "same GMMA work",
            ha="left",
            va="center",
            fontsize=9,
            color="#4B5563",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{row['label']}\nmeasured total {row['measured_total']:.0f}" for row in rows],
        fontsize=10,
        fontweight="bold",
    )


def main() -> None:
    rows = build_rows()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(27.0, 7.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1.55, 0.85, 0.9]},
    )

    draw_panel(axes[0], rows, "front", "Front-End Split")
    draw_panel(axes[1], rows, "core", "Detailed Core Split")
    draw_panel(axes[2], rows, "tail", "Tail Split")
    draw_tensor_panel(axes[3], rows, "Tensor Pipe Utilization (NCU)")

    axes[0].set_xlabel("Cycles")
    axes[1].set_xlabel("Cycles")
    axes[2].set_xlabel("Cycles")
    axes[3].set_xlabel("% of peak elapsed")

    axes[0].tick_params(axis="y", labelleft=True)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    axes[3].tick_params(axis="y", labelleft=False)

    fig.suptitle("WS Kernel Evolution: Measured Timeline Pieces", fontsize=16, fontweight="bold", y=0.96)

    legend_items = [
        Patch(facecolor=COLORS["front_wait"], label="wait(k_full)"),
        Patch(facecolor=COLORS["front_sched"], label="scheduler handoff"),
        Patch(facecolor=COLORS["front_clear"], label="explicit clear(acc_s)"),
        Patch(facecolor=COLORS["core_qk_issue"], label="QK issue"),
        Patch(facecolor=COLORS["core_rescale"], label="rescale before PV"),
        Patch(facecolor=COLORS["core_wait_v"], label="wait(v_full)"),
        Patch(facecolor=COLORS["core_pv_issue"], label="PV issue"),
        Patch(facecolor=COLORS["core_wait1"], label="wait<1> + acc_s fence"),
        Patch(facecolor=COLORS["core_preamble"], label="k_empty + last-mask preamble"),
        Patch(facecolor=COLORS["core_softmax"], label="softmax core"),
        Patch(facecolor=COLORS["core_wait0"], label="wait<0> + acc_o fence"),
        Patch(facecolor=COLORS["tail_handoff"], label="v_empty handoff"),
        Patch(facecolor=COLORS["tail_copy"], label="acc_s cast copy"),
        Patch(facecolor=COLORS["tail_rescale"], label="delayed rescale"),
    ]
    fig.legend(
        handles=legend_items,
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=9,
    )

    fig.text(
        0.01,
        0.095,
        "These panels are shown separately on purpose. Front-end, detailed core, and tail come from different probes and should not be summed as a runtime.",
        fontsize=9,
        color="#374151",
    )
    fig.text(
        0.01,
        0.065,
        "The old coarse 'issue' / 'softmax' labels were ambiguous. This updated figure uses the new core split to separate QK issue, rescale, wait(v), PV issue, preamble, and softmax core.",
        fontsize=9,
        color="#4B5563",
    )
    fig.text(
        0.01,
        0.035,
        "The NCU panel uses sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed on the canonical 4k causal shape. "
        "GMMA work is essentially unchanged across these WS milestones, but tensor-pipe utilization rises.",
        fontsize=9,
        color="#4B5563",
    )
    fig.text(
        0.01,
        0.008,
        "All values here are measured under B=4, S=4096, H=64, Hkv=8, D=128, causal on GPU 1. "
        "The per-row 'measured total' is an independent total-cycle probe.",
        fontsize=9,
        color="#6B7280",
    )

    fig.tight_layout(rect=[0, 0.16, 1, 0.93])
    fig.savefig(OUT, dpi=220)
    print(OUT)


if __name__ == "__main__":
    main()
