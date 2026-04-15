#!/usr/bin/env python3
"""Plot a discussion-ready WS two-WG schematic for the PR871 base milestone."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path("/home/ga/TileOPs")
DATA_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "data"
FIG_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "figures"
OUT = FIG_DIR / "ws_two_wg_schematic.png"


COLORS = {
    "producer": "#F4A261",
    "wait": "#E9C46A",
    "qk": "#457B9D",
    "rescale": "#A8DADC",
    "pv": "#1D3557",
    "softmax": "#E76F51",
    "handoff": "#2A9D8F",
    "clear": "#D62828",
    "tensor": "#264653",
}


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def get_metrics() -> tuple[float, float, float, float, float]:
    probe = load_json("20260415_timeline_probes_pr871_base.json")
    parsed = {row["probe_id"]: row["parsed"] for row in probe["results"]}

    ncu = load_json("20260415_ncu_tensor_pipe_milestones_4k.json")
    base = next(row for row in ncu["results"] if row["variant"] == "pr871_base")

    total = parsed["pr871_base_total_cycle"]["steady_iter_total"]
    front = parsed["pr871_base_scheduler_split"]["sum_cycles"]
    core = parsed["pr871_base_steady_state_regions"]["total_cycles"]
    tail = parsed["pr871_base_tail_split"]["sum_cycles"]
    tensor_pct = base["ncu"]["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]
    return total, front, core, tail, tensor_pct


def draw_box(ax, x: float, y: float, w: float, h: float, color: str, label: str, text_color: str = "white") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        edgecolor="white",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=text_color,
        linespacing=0.92,
    )


def arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = "#6B7280", style: str = "->") -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=1.3))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    total, front, core, tail, tensor_pct = get_metrics()

    fig, ax = plt.subplots(figsize=(17.0, 8.2))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(-0.2, 4.8)
    ax.axis("off")

    ax.text(0.2, 4.45, "Warp-Specialized Persistent Pipeline", fontsize=18, fontweight="bold", color="#111827")
    ax.text(
        0.2,
        4.18,
        "PR871 base. Discussion-ready schematic for B=4, S=4096, H=64, Hkv=8, D=128, causal",
        fontsize=10,
        color="#4B5563",
    )

    lane_y = {
        "tensor": 3.45,
        "producer": 2.65,
        "wg1": 1.8,
        "wg2": 0.95,
    }
    lane_labels = {
        "tensor": "Tensor Core\nutilization",
        "producer": "WG0 /\nproducer",
        "wg1": "WG1 /\nconsumer",
        "wg2": "WG2 /\nconsumer",
    }
    for key, y in lane_y.items():
        ax.text(0.45, y + 0.12, lane_labels[key], ha="center", va="center", fontsize=11, fontweight="bold", color="#1F2937")
        ax.hlines(y, xmin=1.0, xmax=16.9, colors="#D1D5DB", linewidth=1.4)

    h = 0.34

    # Producer lane
    draw_box(ax, 1.4, lane_y["producer"] - h / 2, 1.25, h, COLORS["producer"], "K[n+1]\nTMA")
    draw_box(ax, 5.75, lane_y["producer"] - h / 2, 1.25, h, COLORS["producer"], "V[n]\nTMA")
    draw_box(ax, 9.85, lane_y["producer"] - h / 2, 1.25, h, COLORS["producer"], "K[n+2]\nTMA")
    draw_box(ax, 14.05, lane_y["producer"] - h / 2, 1.25, h, COLORS["producer"], "V[n+1]\nTMA")

    # WG1 lane
    draw_box(ax, 1.45, lane_y["wg1"] - h / 2, 1.10, h, COLORS["wait"], "wait(k)\n+sched", text_color="#1F2937")
    draw_box(ax, 2.65, lane_y["wg1"] - h / 2, 0.75, h, COLORS["clear"], "clear")
    draw_box(ax, 3.55, lane_y["wg1"] - h / 2, 1.10, h, COLORS["qk"], "QK")
    draw_box(ax, 4.75, lane_y["wg1"] - h / 2, 0.95, h, COLORS["rescale"], "rescale", text_color="#1F2937")
    draw_box(ax, 5.90, lane_y["wg1"] - h / 2, 0.75, h, COLORS["wait"], "wait(v)", text_color="#1F2937")
    draw_box(ax, 6.80, lane_y["wg1"] - h / 2, 1.10, h, COLORS["pv"], "PV")
    draw_box(ax, 8.00, lane_y["wg1"] - h / 2, 0.70, h, COLORS["wait"], "wait1", text_color="#1F2937")
    draw_box(ax, 8.82, lane_y["wg1"] - h / 2, 1.40, h, COLORS["softmax"], "softmax")
    draw_box(ax, 10.35, lane_y["wg1"] - h / 2, 0.68, h, COLORS["wait"], "wait0", text_color="#1F2937")
    draw_box(ax, 11.15, lane_y["wg1"] - h / 2, 1.15, h, COLORS["handoff"], "k_empty /\nv_empty")

    # WG2 lane
    draw_box(ax, 7.10, lane_y["wg2"] - h / 2, 1.05, h, COLORS["wait"], "release\nfrom WG1", text_color="#1F2937")
    draw_box(ax, 8.28, lane_y["wg2"] - h / 2, 0.75, h, COLORS["clear"], "clear")
    draw_box(ax, 9.15, lane_y["wg2"] - h / 2, 1.10, h, COLORS["qk"], "QK")
    draw_box(ax, 10.35, lane_y["wg2"] - h / 2, 0.95, h, COLORS["rescale"], "rescale", text_color="#1F2937")
    draw_box(ax, 11.45, lane_y["wg2"] - h / 2, 0.75, h, COLORS["wait"], "wait(v)", text_color="#1F2937")
    draw_box(ax, 12.35, lane_y["wg2"] - h / 2, 1.10, h, COLORS["pv"], "PV")
    draw_box(ax, 13.55, lane_y["wg2"] - h / 2, 0.70, h, COLORS["wait"], "wait1", text_color="#1F2937")
    draw_box(ax, 14.38, lane_y["wg2"] - h / 2, 1.40, h, COLORS["softmax"], "softmax")
    draw_box(ax, 15.90, lane_y["wg2"] - h / 2, 0.68, h, COLORS["wait"], "wait0", text_color="#1F2937")

    # Tensor lane: PR871-base-style ordering
    tc_h = 0.24
    draw_box(ax, 3.48, lane_y["tensor"] - tc_h / 2, 1.65, tc_h, COLORS["tensor"], "WG1 QK active")
    draw_box(ax, 6.73, lane_y["tensor"] - tc_h / 2, 1.70, tc_h, COLORS["tensor"], "WG1 PV active")
    draw_box(ax, 9.05, lane_y["tensor"] - tc_h / 2, 1.65, tc_h, COLORS["tensor"], "WG2 QK active")
    draw_box(ax, 12.28, lane_y["tensor"] - tc_h / 2, 1.70, tc_h, COLORS["tensor"], "WG2 PV active")

    # Handoffs/arrows
    arrow(ax, 7.9, lane_y["wg1"] + 0.33, 7.6, lane_y["wg2"] + 0.18, color=COLORS["handoff"])
    ax.text(7.25, 1.48, "named / scheduler\nhandoff", ha="center", va="top", fontsize=8.5, color=COLORS["handoff"], fontweight="bold")
    arrow(ax, 12.25, lane_y["wg1"] + 0.10, 14.0, lane_y["producer"] - 0.05, color=COLORS["handoff"])
    ax.text(13.15, 2.02, "buffer return enables\nnext producer wave", ha="center", va="bottom", fontsize=8.5, color=COLORS["handoff"], fontweight="bold")

    # Callouts
    ax.text(
        0.25,
        0.25,
        f"Measured timing anchors\nfront-end: {front:.0f} cycles\nsteady-state core: {core:.0f} cycles\ntail: {tail:.0f} cycles\nmeasured total: {total:.0f} cycles",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#111827",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F9FAFB", edgecolor="#D1D5DB"),
    )
    ax.text(
        5.95,
        0.28,
        f"Nsight Compute\nTensor pipe utilization: {tensor_pct:.1f}% of peak elapsed\nGMMA work: unchanged vs reorder/anchor",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#111827",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F9FAFB", edgecolor="#D1D5DB"),
    )
    ax.text(
        11.55,
        0.30,
        "Discussion figure: lane ordering and overlap are structural.\n"
        "Active Tensor Core windows are drawn longer than issue boxes,\n"
        "and WG2 starts only after WG1's scheduler release.",
        ha="left",
        va="bottom",
        fontsize=9.3,
        color="#6B7280",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=220)
    print(OUT)


if __name__ == "__main__":
    main()
