# RETIRED: tilelang_common_h_descv.patch

**This patch is no longer needed.** See ablation results in
[superAngGao/TileOPs-report-static#9 comment 4197515908](https://github.com/superAngGao/TileOPs-report-static/issues/9#issuecomment-4197515908).

After the per-WG self-contained body restructure in `_test_ws_fa3_v2.py`,
each `GmmaDescriptor` local has a live range bounded by a single basic
block (one WG's `with T.ws(N):` body). ptxas can register-promote them
via standard SSA analysis, even with the original by-reference
`initialize_wgmma_descriptor(...)` signature.

Empirically:
- with Option 1 patch:    stack 0, spill 0, 521.7 TFLOPS
- without Option 1 patch: stack 0, spill 0, 521.8 TFLOPS

The patch file is left here as historical record. Do NOT apply.
