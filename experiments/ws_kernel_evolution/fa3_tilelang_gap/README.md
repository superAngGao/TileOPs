# FP8 GQA FA3-vs-TileLang Gap Report

This folder is an isolated discussion artifact.  It does not modify production
attention kernels.  The goal is to keep the current FP8 GQA/FA3 intent, observed
SASS behavior, and suspected TileLang API gaps in one clean place for later
slides, issue filing, and team discussion.

## Files

- `fp8_gqa_fa3_tilelang_gap_report_cn.md`
  Human-readable technical report.
- `fa3_tilelang_gap_contract.py`
  Code-form contract/spec for the desired FA3-style steady state and the current
  TileLang capability gaps.
- `test_fa3_tilelang_gap_contract.py`
  Lightweight tests that guard the FIFO order and capability classification.
- `boundary_demo_kernel.py`
  Minimal harness around the streaming-P plain-wait overlap kernel selected as
  the boundary-demo candidate.
- `test_boundary_demo_kernel.py`
  Correctness gate comparing the boundary-demo candidate against the serial
  BN224 TMA-V baseline.
- `sass_scoreboard.py`
  Text-only counter for `nvdisasm` output.  It reports QGMMA shape counts,
  WARPGROUP scoreboard counts, and contiguous QGMMA runs.
- `test_sass_scoreboard.py`
  Unit tests for grouped versus per-QGMMA SASS examples.

## Commands

From the repository root:

```bash
python3 experiments/ws_kernel_evolution/fa3_tilelang_gap/fa3_tilelang_gap_contract.py
python3 experiments/ws_kernel_evolution/fa3_tilelang_gap/fa3_tilelang_gap_contract.py --format steady-state
python3 experiments/ws_kernel_evolution/fa3_tilelang_gap/fa3_tilelang_gap_contract.py --format discussion
python3 experiments/ws_kernel_evolution/fa3_tilelang_gap/fa3_tilelang_gap_contract.py --format json
python3 -m pytest experiments/ws_kernel_evolution/fa3_tilelang_gap/test_fa3_tilelang_gap_contract.py
python3 -m pytest experiments/ws_kernel_evolution/fa3_tilelang_gap/test_boundary_demo_kernel.py
python3 -m pytest experiments/ws_kernel_evolution/fa3_tilelang_gap/test_sass_scoreboard.py
python3 experiments/ws_kernel_evolution/fa3_tilelang_gap/sass_scoreboard.py path/to/kernel.sass
```

## Code-Level Resolution

This folder only resolves the pieces that can be cleanly solved without changing
production kernels:

- The correct FA3 FIFO order is executable and test-guarded.
- No TileLang kernel capability is marked resolved until a correctness kernel
  proves it.
- Persistent `tOrP` fragment lifetime is treated as a hypothesis requiring a
  minimal TileLang smoke kernel.
- Structural blockers remain marked as blockers instead of being hidden behind
  experimental helper code.
- Tail-fence behavior is kept as an A/B hypothesis, not a claimed root cause.

## Kernel Correctness Policy

This directory currently wraps the existing delayed-accumulate TileLang kernel
as the boundary-demo candidate.  The candidate must pass
`test_boundary_demo_kernel.py` against the serial BN224 TMA-V baseline before it
can be used in a TileLang community discussion.

## Why This Exists

The FP8 GQA experiments can write a FA3-like source schedule, but the generated
SASS often loses FA3's grouped WGMMA scoreboard shape.  The main uncertainty is
not the attention algorithm itself.  It is whether TileLang can express the same
register-fragment, WGMMA grouping, and CUTE shared-memory contracts that FA3's
C++ mainloop gives to ptxas.
