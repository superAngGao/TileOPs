# W4A16 AKO campaign

This directory contains the completed four-shape parameter campaign and the
resumable structural campaign for shape ``(1, 8192, 81920)``.

Each shape has a budget of 75 valid rounds.  A valid round is a unique
candidate that compiles, passes correctness, and completes the cold-cache
TileOps benchmark.  Compile and correctness failures are recorded as attempts
but do not consume the round budget.

The early-stop condition is:

1. the incumbent passes the full `bench_kernel` protocol;
2. Nsight Compute reports
   `dram__throughput.avg.pct_of_peak_sustained_elapsed >= 80`;
3. the profiled kernel is the same source/config hash as the incumbent.

Candidate selection is local to one shape.  Results from different shapes are
never combined into a weighted score.

The completed four-shape result is recorded in `campaign_summary.json`.
`shape2.log`, `shape3.log`, and `shape4.log` contain the complete stdout for
their 75 valid rounds.  Shape 1 was developed interactively; its early rounds
are split between `rounds.jsonl`, `attempts.jsonl`, and the final summary.

## Shape-4 structural campaign

The 300-round follow-up uses a stricter definition: a round changes the
kernel's pipeline or dataflow.  Block sizes, stage counts, warp counts,
register limits, and split-K values are parameter trials *inside* a structural
round and never increment its round number by themselves.

The prior 75-point winner is frozen as a non-counting round zero in
`shape4_structural_baseline.json`.  Each new structure is declared under
`structures/`.  Run one with:

```
python experiments/ako_w4a16/run_structural_round.py \
  --spec experiments/ako_w4a16/structures/round001_fragment_mma_sync.json
```

The driver rejects duplicate structural signatures and a child structure that
does not change any declared structural field relative to its parent.  A
structure consumes a round only if at least one internal trial compiles,
passes correctness, and completes the official cold-cache `bench_kernel`
screening protocol.  Records are separated into structural rounds, successful
parameter trials, and non-counting attempts.

The campaign completed all 300 effective structural rounds.  All structural
signatures and generated source hashes are unique, and every counted round
passed correctness.  The confirmed winner is scalar schedule 175 at
0.1361794 ms.  Nsight Compute reports 43.36% DRAM throughput, 78.06% SM
throughput, and 83.08% issue-active, so the 80% DRAM early-stop threshold was
not reached.  The machine-readable result and comparisons are in
`campaign_summary.json`; the full per-round records are in
`shape4_structural_rounds.jsonl`.

## Shape-4 transposed Tensor Core campaign

The follow-up Tensor Core campaign completed 100 effective structural rounds.
It maps 16 real output channels to the M dimension of
`mma.sync.m16n8k16`, pads only the batch/N dimension, and decodes the offline
fragment-aware W4 layout directly from shared memory into the MMA A
register fragment.  FP32 temporary accumulators are retained per K128 group
before affine scale application.

All 100 structural signatures and generated source hashes are unique, and
all counted rounds passed correctness.  The confirmed winner is schedule 92
at 0.16867074 ms.  It is 14.46% faster than the earlier group-post-scale MMA
prototype, but remains 23.86% slower than scalar schedule 175 and 24.35%
slower than Marlin.

Nsight Compute reports 35.23% DRAM throughput, 77.45% SM throughput, 81.86%
issue-active, and 11.84% Tensor-pipe activity.  The low Tensor utilization
with high issue activity identifies fragment decode/preparation and group
handoff as the remaining bottleneck.  The full result is in
`shape4_tensorcore_summary.json`; per-round records are in
`shape4_tensorcore_rounds.jsonl`, and non-counting development failures are
in `shape4_tensorcore_development_attempts.jsonl`.
