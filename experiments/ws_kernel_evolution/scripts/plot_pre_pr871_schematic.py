#!/usr/bin/env python3
"""Plot a discussion-ready schematic for the pre-PR871 milestone."""

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
OUT = FIG_DIR / "pre_pr871_schematic.png"


COLORS = {
    "k_copy": "#F4A261",
    "qk": "#457B9D",
    "softmax": "#E76F51",
    "cast_copy": "#6C757D",
    "rescale": "#A8DADC",
    "v_copy": "#2A9D8F",
    "pv": "#1D3557",
    "epilogue": "#7B2CBF",
    "tensor_lane": "#264653",
}


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text())


def get_pre_metrics() -> tuple[float, float, float, float]:
    coarse = load_json("20260415_timeline_probes_pre_pr871.json")
    coarse_parsed = coarse["results"][0]["parsed"]
    loop_body = coarse_parsed["loop_body_total"]
    epilogue = coarse_parsed["epilogue_total"]

    ncu = load_json("20260415_ncu_tensor_pipe_milestones_4k.json")
    pre = next(r for r in ncu["results"] if r["variant"] == "pre_pr871_wgmma")
    tensor_pct = pre["ncu"]["metrics"]["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]
    kernel_ns = pre["ncu"]["metrics"]["gpu__time_duration.sum"]
    return loop_body, epilogue, tensor_pct, kernel_ns


def draw_box(ax, x: float, y: float, w: float, h: float, color: str, label: str, text_color: str = "white") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
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
        fontsize=9,
        fontweight="bold",
        color=text_color,
        linespacing=0.9,
    )


def draw_iteration(ax, x0: float, y_pipeline: float, y_tc: float, title: str) -> None:
    h = 0.38
    gap = 0.07
    widths = {
        "k_copy": 0.70,
        "qk": 1.00,
        "softmax": 1.10,
        "cast": 0.72,
        "rescale": 0.72,
        "v_copy": 0.70,
        "pv": 1.00,
    }
    x = x0
    draw_box(ax, x, y_pipeline, widths["k_copy"], h, COLORS["k_copy"], "K copy", text_color="#1F2937")
    x += widths["k_copy"] + gap
    qk_x = x
    draw_box(ax, x, y_pipeline, widths["qk"], h, COLORS["qk"], "QK mask\n+ issue")
    x += widths["qk"] + gap
    draw_box(ax, x, y_pipeline, widths["softmax"], h, COLORS["softmax"], "online\nsoftmax")
    x += widths["softmax"] + gap
    draw_box(ax, x, y_pipeline, widths["cast"], h, COLORS["cast_copy"], "acc_s\ncopy")
    x += widths["cast"] + gap
    draw_box(ax, x, y_pipeline, widths["rescale"], h, COLORS["rescale"], "rescale", text_color="#1F2937")
    x += widths["rescale"] + gap
    draw_box(ax, x, y_pipeline, widths["v_copy"], h, COLORS["v_copy"], "V copy", text_color="#1F2937")
    x += widths["v_copy"] + gap
    pv_x = x
    draw_box(ax, x, y_pipeline, widths["pv"], h, COLORS["pv"], "PV issue")

    ax.text(x0 + 3.1, y_pipeline + 0.56, title, ha="center", va="bottom", fontsize=9, color="#374151", fontweight="bold")

    tc_h = 0.26
    qk_active_x = qk_x - 0.04
    qk_active_w = widths["qk"] + 0.34
    pv_active_x = pv_x - 0.04
    pv_active_w = widths["pv"] + 0.34
    draw_box(ax, qk_active_x, y_tc, qk_active_w, tc_h, COLORS["tensor_lane"], "QK active")
    draw_box(ax, pv_active_x, y_tc, pv_active_w, tc_h, COLORS["tensor_lane"], "PV active")
    ax.annotate(
        "",
        xy=(pv_active_x - 0.04, y_tc + tc_h / 2),
        xytext=(qk_active_x + qk_active_w + 0.04, y_tc + tc_h / 2),
        arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=1.2),
    )
    ax.text(
        (qk_active_x + qk_active_w + pv_active_x) / 2,
        y_tc + tc_h / 2 + 0.16,
        "non-TC work between\nQK and PV",
        ha="center",
        va="center",
        fontsize=8,
        color="#6B7280",
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    loop_body, epilogue, tensor_pct, kernel_ns = get_pre_metrics()

    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(-0.1, 3.3)
    ax.axis("off")

    ax.text(0.2, 3.05, "Single-CTA WGMMA Pipeline", fontsize=18, fontweight="bold", color="#111827")
    ax.text(
        0.2,
        2.82,
        "Pre-WS baseline. Representative last causal tile for B=4, S=4096, H=64, Hkv=8, D=128, block_m=128, block_n=128",
        fontsize=10,
        color="#4B5563",
    )

    ax.text(0.35, 2.25, "Tensor Core\nutilization", ha="center", va="center", fontsize=11, fontweight="bold", color="#1F2937")
    ax.text(0.35, 1.45, "CTA /\nsoftware pipeline", ha="center", va="center", fontsize=11, fontweight="bold", color="#1F2937")

    ax.hlines([2.22, 1.42], xmin=0.9, xmax=15.7, colors="#D1D5DB", linewidth=1.4)

    draw_iteration(ax, 1.15, 1.23, 2.09, "warmup / steady-state iteration k_idx")
    draw_iteration(ax, 8.65, 1.23, 2.09, "next iteration k_idx+1")

    draw_box(ax, 13.05, 0.46, 2.25, 0.42, COLORS["epilogue"], "epilogue:\nnormalize O, store O/LSE")
    ax.annotate(
        "",
        xy=(13.02, 0.67),
        xytext=(12.05, 1.05),
        arrowprops=dict(arrowstyle="->", color="#7B2CBF", lw=1.4),
    )

    ax.text(
        0.3,
        0.75,
        f"Measured coarse timing\nloop body total: {loop_body:.0f} cycles\nepilogue total: {epilogue:.1f} cycles",
        ha="left",
        va="center",
        fontsize=11,
        color="#111827",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F9FAFB", edgecolor="#D1D5DB"),
    )

    ax.text(
        4.55,
        0.55,
        f"Nsight Compute\nkernel time: {kernel_ns/1e6:.3f} ms\nTensor pipe utilization: {tensor_pct:.1f}% of peak elapsed",
        ha="left",
        va="center",
        fontsize=11,
        color="#111827",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F9FAFB", edgecolor="#D1D5DB"),
    )

    ax.text(
        0.2,
        0.12,
        "Discussion figure: inner loop stage widths are structural, not cycle-accurate. "
        "QK/PV active windows are intentionally drawn longer than issue boxes.",
        fontsize=9.5,
        color="#6B7280",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=220)
    print(OUT)


if __name__ == "__main__":
    main()
