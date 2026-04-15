# WS Two-WG Schematic

This note defines the discussion-ready schematic for the first warp-specialized
milestone.

Preferred reader-facing title:

- `Warp-Specialized Persistent Pipeline`

Recommended baseline variant:

- `pr871_base`

## Purpose

This figure complements the `Single-CTA WGMMA Pipeline` schematic.

The single-CTA figure answers:

- what did the pre-WS baseline look like?

This WS figure answers:

- how do the two consumer warp groups cooperate?
- where does the producer sit?
- how is Tensor Core work interleaved across WG1 and WG2?

## Lanes

The schematic should use four lanes:

- `Tensor Core utilization`
- `WG0 / producer`
- `WG1 / consumer`
- `WG2 / consumer`

## Structural Story To Show

For `pr871_base`, the discussion-ready ordering should emphasize:

1. producer loads `K[n+1]` / `V[n]`
2. WG1 starts first:
   - `wait(k)`
   - scheduler sync
   - `clear`
   - `QK`
   - `rescale`
   - `wait(v)`
   - `PV`
   - `wait1`
   - `softmax`
   - `wait0`
   - buffer return
3. WG2 is released later by WG1 and then runs the same local sequence
4. Tensor Core windows appear in the rough order:
   - `WG1 QK`
   - `WG1 PV`
   - `WG2 QK`
   - `WG2 PV`

This is intentionally a structural picture, not a cycle-accurate lane chart.

## Measured Anchors To Show

For the canonical causal shape:

- `front-end = 573 cycles`
- `steady-state core = 2113 cycles`
- `tail = 134 cycles`
- `measured total = 2915 cycles`
- `Tensor pipe utilization = 60.0% of peak elapsed`

These values come from:

- [20260415_timeline_probes_pr871_base.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_timeline_probes_pr871_base.json)
- [20260415_ncu_tensor_pipe_milestones_4k.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_ncu_tensor_pipe_milestones_4k.json)

## Drawing Rules

1. Show active Tensor Core windows longer than local issue boxes.
2. Keep WG2 visibly delayed relative to WG1.
3. Show producer activity as TMA waves, not as direct compute work.
4. Use utilization as a callout, not as the width of the Tensor lane.
