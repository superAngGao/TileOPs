# Timeline Measurement Framework

This note defines how we collect cycle-level evidence for the milestone
timeline figures in the WS kernel evolution study.

## Goal

Every timeline figure in the final write-up should be traceable to:

- one milestone node
- one or more structured probe outputs
- one plotting script
- one note explaining which arrows/segments come from measurement and which
  come from code-order inference

## Milestones

Target narrative nodes:

1. `pre_pr871`
2. `pr871_base`
3. `pr871_reorder`
4. `anchor_causal`

Bootstrap reference nodes already supported by existing probes:

- `ws_current_bn176`
- `ws_current_bn128`

## Probe Families

### 1. Steady-state region cycles

Purpose:

- partition one steady-state consumer iteration into large regions
- quantify overlap between QK, PV, softmax, and wait tails

Typical outputs:

- `wgmma_issue`
- `wait<1>`
- `softmax`
- `wait<0>`

### 2. Barrier / scheduler split

Purpose:

- split the pre-QK rendezvous into sub-costs
- identify how much of a long front-end block is true data wait vs scheduler
  handoff vs clear/reset work

Typical outputs:

- `barrier_wait(k_full)`
- scheduler handoff cost
- inferred `clear(acc_s)` or equivalent reset cost

### 3. Timeline metadata

Purpose:

- record shape, causal flag, block sizes, GPU, env vars, and kernel source
- keep figure generation reproducible

## Measurement Rules

- Run on `GPU 1` unless explicitly overridden.
- Run serially; do not overlap experiments with other GPU jobs.
- Keep the same benchmark shape across milestone comparisons unless the kernel
  is invalid for that shape.
- If a kernel has structural constraints, record them as validity notes rather
  than hiding the missing point.

## Interpretation Rules

- A measured cycle segment may be plotted directly.
- A dependency arrow may be inferred from code order, but it must be labeled as
  inferred in accompanying notes if no direct cycle measurement exists.
- If a plotted duration is inferred instead of measured, the figure should make
  that visually clear or the note should say so explicitly.

## Current State

Implemented:

- BN176 steady-state region cycles
- BN176 barrier/scheduler split
- BN128 steady-state region cycles

Planned:

- PR 871 base steady-state probe
- PR 871 reorder steady-state probe
- anchor causal steady-state probe
- anchor scheduler split probe

## Framework Entry Point

Use:

```bash
python experiments/ws_kernel_evolution/scripts/run_timeline_probes.py --list-milestones
python experiments/ws_kernel_evolution/scripts/run_timeline_probes.py --list-probes
python experiments/ws_kernel_evolution/scripts/run_timeline_probes.py \
  --milestone ws_current_bn176 \
  --output experiments/ws_kernel_evolution/data/<file>.json
```

The registry and runner live in:

- `scripts/timeline_probe_registry.py`
- `scripts/run_timeline_probes.py`
