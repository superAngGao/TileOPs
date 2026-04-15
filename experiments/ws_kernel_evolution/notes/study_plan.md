# Study Plan

## Story Spine

1. Problem setup and measurement protocol
2. Pre-PR-871 state
3. PR 871
4. PR 871 + persistent reorder
5. Anchor causal
6. Failed directions and lessons learned

## Priority Evidence To Collect

- Throughput tables for the four milestone nodes
- One accurate steady-state timeline per milestone node
- Barrier / wait / clear / softmax / PV timing breakdown where possible
- Short correctness checks for every milestone
- FA3 comparison on the same shapes and hardware

## Failed Directions Worth Summarizing

- early WG2 release
- explicit two-stage `QK/QK/PV/PV`
- `ScaleOut::Zero` / zero-clear
- toolchain / TileLang-version constraints
- measurement pitfalls such as cross-job GPU interference

## Immediate Next Steps

1. Add a benchmark harness for the four milestone nodes.
2. Store the validated 4096/8192/16384 causal comparison data here.
3. Generate timeline figures with raw cycle annotations rather than hand-only labels.

## Current Milestone Naming

- `pre_pr871`: `GqaFwdWgmmaPipelinedKernel`
- `pr871_base`: persistent WS kernel from PR 871
- `pr871_reorder`: PR 871 + persistent KV-head-friendly ordering
- `anchor_causal`: anchored persistent causal kernel
