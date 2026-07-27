# Gated DeltaNet Backward Prefill-Style Experiments

This directory is the scratchpad for the Gated DeltaNet backward optimization
line. The target is a high-performance backward kernel that uses the same
structural idea as optimized prefill: shorten the cross-chunk recurrence with
chunk / segment affine summaries, while keeping local work GEMM-shaped.

Initial scope:

1. Measure the current backward pipeline and stage breakdown.
2. Separate BHSD legacy cost from BTHD wrapper/layout cost.
3. Prototype the reverse affine-scan formulation for `dh` propagation.
4. Only then replace the current `dh_recurrence_bwd` implementation.

The design plan is in:

```text
docs/design/gated-deltanet-backward-prefill-style-plan-cn.md
```

## Baseline timing

Example:

```bash
python experiments/gated_deltanet_bwd_prefill_style/bench_current_bwd.py \
  --batch 1 --heads 16 --seq-len 8192 --dim-k 128 --dim-v 128 \
  --chunk-size 64 --dtype fp16 --layout bthd --stage-breakdown
```

The `layout=bthd` mode generates BTHD inputs and times the explicit wrapper
conversion into the current BHSD-only backward path. It is a baseline for
layout overhead, not a BTHD-native optimized kernel.

## AKO config search

The first AKO loop searches low-risk launch and V-tiling choices around the
current split backward implementation:

```bash
python experiments/gated_deltanet_bwd_prefill_style/run_config_ako.py \
  --target-total 300 \
  --batch 1 --heads 16 --seq-len 4096 --dim-k 128 --dim-v 128 \
  --chunk-size 64 --dtype fp16 --layout bthd \
  --warmup 1 --repeat 3 --trials 1 --no-stage-breakdown \
  --skip-known-failures \
  --output experiments/gated_deltanet_bwd_prefill_style/results/config_ako_d128.jsonl
```

Each candidate runs in a fresh Python process and appends one JSON object. The
record includes the candidate config, pass/fail status, full backward latency,
and stage timing when enabled. This is intentionally conservative: failed
TileLang lowers or runtime errors should become search data rather than killing
the whole experiment. The recommended loop is to run broad sweeps with
`--no-stage-breakdown`, then rerun the best few candidates with
`--stage-breakdown --warmup 5 --repeat 20 --trials 3`.
Use `--target-total 300` with the default `--resume` behavior to continue an
interrupted run until the JSONL file contains 300 records.
`--skip-known-failures` removes candidate families that previous records already
showed to be deterministic lowering/runtime failures.

Early local smoke results:

- `B=1,S=64,H=1,DK=DV=64,chunk=32,fp16,BHSD`: full backward runs.
- `B=1,S=4096,H=16,DK=DV=64,chunk=64,fp16,BTHD-wrapper`: full backward
  `0.530 ms`, with `dh_recurrence_bwd` `0.420 ms`.
- `B=1,S=4096,H=16,DK=DV=128,chunk=64,fp16,BTHD-wrapper`: after adding
  V-tiling to `dh_recurrence_bwd`, full backward runs at `1.082 ms`; stage
  timing reports `dh_recurrence_bwd` `0.530 ms` and partial reduction
  `0.032 ms`.

The original d128 failure was part of the design signal: the optimized backward
path needs V-dimension tiling or an equivalent state-blocking scheme before the
reverse affine scan can target Qwen-style `DK=DV=128` workloads. The current
patch removes that feasibility blocker; the cross-chunk reverse scan remains
the next performance step.

First AKO config-search notes for
`B=1,S=4096,H=16,DK=DV=128,chunk=64,fp16,BTHD-wrapper`:

- `recurrence_block_v=64,num_stages=3` failed with dynamic shared memory
  `240160` bytes.
- A 48-record repeated sweep over the 24-candidate low-risk launch space
  produced 40 passing rows and 8 deterministic TileLang layout-rewrite
  failures. The failing family was `block_v=16,recurrence_threads=256`.
- A follow-up 60-record sweep over the 20 valid candidates used
  `--skip-known-failures` and produced 60 passing rows, with three samples per
  valid candidate.
- A fast in-process 300-record sweep over the same 20 valid candidates produced
  300 passing rows. Each candidate has 15 samples in that file:
  `results/config_ako_d128_inprocess_300.jsonl`.
- Short 3/5-repeat smoke found best-looking candidates at `1.043890 ms` and
  `0.825387 ms`, but more stable `warmup=5,repeat=20,trials=3` reruns showed
  those were noisy enough that they should not drive the default.
- Repeated 3-repeat valid-sweep medians also favored
  `block_v=32,stages=2,threads=256,recurrence_threads=256`: three rows with
  best `0.878140 ms` and median `0.982867 ms`.
- The 300-record in-process median ranking also favored the same family:
  `block_v=32,stages=2,recurrence_threads=256`. It slightly preferred
  `threads=128` over `threads=256` in short-repeat median
  (`1.138653 ms` vs `1.148665 ms`), but stable reruns showed the difference is
  negligible.
- Stable timing favored the default `recurrence_block_v=32`:
  - `block_v=32,stages=2,threads=256`: full `1.186958 ms`;
    `dh_recurrence_bwd` `0.587935 ms`, partial reduction `0.041128 ms`.
  - `block_v=32,stages=2,threads=128`: full `1.185371 ms`;
    `dh_recurrence_bwd` `0.586381 ms`, partial reduction `0.041131 ms`.
  - `block_v=64,stages=2,threads=256`: full `1.294315 ms`;
    `dh_recurrence_bwd` `0.691640 ms`, partial reduction `0.022379 ms`.
  - `block_v=16,stages=2,threads=256,recurrence_threads=128`: full
    `1.207783 ms`; `dh_recurrence_bwd` `0.563610 ms`, partial reduction
    `0.083284 ms`.

The current default keeps `recurrence_block_v=32` for `DV=128`: it is small
enough to pass shared-memory limits and large enough to avoid excessive
cross-tile reduction overhead.

## Reverse affine scan prototype

The cross-chunk `dh` recurrence in the current kernel can be written as:

```text
X[i] = G[i] + alpha[i] * X[i + 1]
```

where `G[i]` is the chunk-local adjoint contribution and `alpha[i]` is the
chunk boundary decay factor. The reference prototype verifies that a segment
can summarize this as:

```text
X[left] = A_segment * X[right] + B_segment
```

and compose summaries associatively before running short local reverse
recurrences inside each segment.

Example checks:

```bash
python experiments/gated_deltanet_bwd_prefill_style/prototype_reverse_affine_scan.py \
  --chunks 64 --dim-k 128 --dim-v 128 --segment-chunks 8
```

Observed local checks:

- `chunks=16,DK=DV=8,segment=4,fp32`: max abs `5.96e-08`.
- `chunks=64,DK=DV=128,segment=8,fp32`: max abs `1.19e-07`.
- `chunks=64,DK=DV=128,segment=16,fp64`: max abs `2.22e-16`.

This is a correctness prototype for the future reverse-scan kernel; it does not
yet replace `dh_recurrence_bwd` in the hot path.

## Real-intermediate reverse scan gate

`prototype_reverse_affine_real.py` runs the current forward and `bwd_parallel`
kernels, then recomputes the `dh_recurrence_bwd` correction outputs from a
Torch segmented reverse affine scan over the real `dh_local`, `v_new`, `S`, `k`,
and `g_cum` intermediates.

Example:

```bash
python experiments/gated_deltanet_bwd_prefill_style/prototype_reverse_affine_real.py \
  --batch 1 --heads 2 --seq-len 128 --dim-k 128 --dim-v 128 \
  --chunk-size 64 --segment-chunks 2 --dtype fp16
```

Observed checks:

- `B=1,H=1,S=128,DK=DV=64,chunk=64,segment=1,fp16`: pass;
  `dk_corr` max abs `5.38e-04`, `du_corr` max abs `3.81e-06`,
  `dg_corr` max abs `8.34e-07`.
- `B=1,H=2,S=128,DK=DV=128,chunk=64,segment=2,fp16`: pass;
  `dk_corr` max abs `6.84e-03`, `du_corr` max abs `7.63e-06`,
  `dg_corr` max abs `1.91e-06`.

This gate fixes the future kernel boundary: a reverse-scan kernel needs to
produce the same per-chunk successor carry that the current sequential
`dh_recurrence_bwd` uses before computing `dk_corr`, `du_corr`, and `dg_corr`.

## Split-carry and segment-carry TileLang boundary

The first TileLang hot-path step splits the old `dh_recurrence_bwd` kernel into
two candidate stages behind `recurrence_split_carry=1`:

1. `_dh_carry_after_scan_tl`: reverse scan over chunks to materialize each
   chunk's successor-side `dh` carry.
2. `_dh_correction_from_carry_tl`: chunk-parallel correction GEMMs using the
   materialized carry.

This is not yet the final segment reverse-scan implementation. It keeps the
carry producer as a sequential reverse scan, but it moves the heavy correction
work off that serial loop and fixes the kernel boundary for the next AKO round.

The next candidate, `recurrence_split_carry=2`, turns that boundary into a
segment-carry path:

1. `_dh_segment_summary_tl`: summarize each short segment as
   `X[left] = B_segment + A_segment * X[right]`.
2. `_dh_segment_boundary_scan_tl`: reverse scan the segment summaries.
3. `_dh_segment_local_carry_tl`: expand each segment boundary carry into
   per-chunk successor carries.
4. `_dh_correction_from_carry_tl`: reuse the chunk-parallel correction consumer.

This shrinks the producer-side dependency from all chunks to segment summaries,
then keeps only short within-segment reverse loops. For `dim_v=128,chunk=64`,
the operator default now enables this segment-carry path with:

```text
threads=128, parallel_threads=256, recurrence_threads=128,
recurrence_block_v=64, recurrence_split_carry=2
```

Correctness checks:

- `B=1,H=1,S=128,DK=DV=64,chunk=64,fp16`: split path matches default backward
  output exactly on `dq/dk/dv/dg/dbeta`.
- `B=1,H=2,S=128,DK=DV=128,chunk=64,fp16,recurrence_block_v=32`: split path
  matches default backward output exactly on `dq/dk/dv/dg/dbeta`.
- `pytest -q tests/ops/test_gated_deltanet_chunkwise_bwd.py -m smoke -k split_carry --tb=short`:
  `2 passed`; both `recurrence_split_carry=1` and `2` match the monolithic
  baseline on `dq/dk/dv/dg/dbeta`.
- `pytest -q tests/ops/test_gated_deltanet_chunkwise_bwd.py -m full -k '128-2-128-128-64' --tb=short`:
  `1 passed`.

Stable local timing for `B=1,S=4096,H=16,DK=DV=128,chunk=64,fp16,BTHD-wrapper`,
`warmup=5,repeat=20,trials=3`, H200:

| path | full backward | dh recurrence / carry work | correction work | reduction |
| --- | ---: | ---: | ---: | ---: |
| default recurrence | `1.146564 ms` | `dh_recurrence_bwd 0.587648 ms` | included in recurrence | `0.039275 ms` |
| split carry | `0.842685 ms` | `dh_carry_after_scan 0.088928 ms` | `dh_correction_from_carry 0.186823 ms` | `0.039422 ms` |
| split carry, block_v=64 default | `0.781947 ms` | `dh_carry_after_scan 0.118798 ms` | `dh_correction_from_carry 0.129728 ms` | `0.021413 ms` |
| segment carry, block_v=64 default | `0.705536 ms` | summary `0.016203 ms` + boundary scan `0.008784 ms` + local carry `0.023086 ms` | `dh_correction_from_carry 0.099558 ms` | `0.020293 ms` |

The split-inclusive 300-round in-process AKO sweep completed with `300 pass`
and `0 fail`. The best median candidate was:

```text
num_stages=2, threads=128, parallel_threads=256,
recurrence_threads=128, recurrence_block_v=64, recurrence_split_carry=1
```

It appeared in seven short-repeat samples with best `0.701216 ms` and median
`0.706207 ms`; the split-carry table row above reports the same candidate under
the more stable `warmup=5,repeat=20,trials=3` stage-breakdown timing. The later
segment-carry candidate improves the stable full backward timing to
`0.705536 ms` and is now the d128 default.

A second 300-round sweep including `recurrence_split_carry=2` also completed
with `300 pass` and `0 fail`. The top median families were all segment-carry
with `recurrence_block_v=64`; the best short-repeat median was:

```text
num_stages=2, threads=128, parallel_threads=256,
recurrence_threads=256, recurrence_block_v=64, recurrence_split_carry=2
best=0.612958 ms, median=0.671614 ms
```

However, its stable stage-breakdown rerun measured `0.768857 ms`, slower than
the `recurrence_threads=128` segment-carry default at `0.705536 ms`. The default
therefore keeps `recurrence_threads=128`; this is a useful example of why
short-repeat AKO rankings still need a formal timing gate.

Result file:

```text
experiments/gated_deltanet_bwd_prefill_style/results/split_carry_d128_stage_20260724.jsonl
experiments/gated_deltanet_bwd_prefill_style/results/split_carry_d128_best_stage_20260724.jsonl
experiments/gated_deltanet_bwd_prefill_style/results/segment_carry_d128_stage_20260724.jsonl
experiments/gated_deltanet_bwd_prefill_style/results/segment_carry_d128_rt256_stage_20260724.jsonl
experiments/gated_deltanet_bwd_prefill_style/results/config_ako_d128_split_carry_inprocess_300.jsonl
experiments/gated_deltanet_bwd_prefill_style/results/config_ako_d128_segment_carry_inprocess_300.jsonl
```
