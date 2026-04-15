# Single-CTA WGMMA Timeline Drawing Spec

This note defines how the `pre_pr871` milestone should be visualized.

For discussion and blog-facing figures, the preferred display name is:

- `Single-CTA WGMMA Pipeline`

`pre_pr871` should stay as the internal experiment/milestone id only.

## Why It Needs A Different Template

The `pre_pr871` milestone is **not** a warp-specialized producer/consumer kernel.

It is:

- `GqaFwdWgmmaPipelinedKernel`
- one CTA
- one software-pipelined loop over K/V tiles
- overlap across loop iterations managed by `T.Pipelined(...)`

So it should **not** be drawn with:

- producer WG lane
- consumer WG1 lane
- consumer WG2 lane
- named-barrier handoff arrows

Those are appropriate only for:

- `pr871_base`
- `pr871_reorder`
- `anchor_causal`

## Recommended Visual Form

The recommended figure for this milestone is a **single-CTA software-pipeline diagram**.

The minimal form should have:

- one CTA lane labeled `CTA / software pipeline`
- one `Tensor Core` lane showing where QK/PV GEMMs are active
- one repeated loop-body box labeled by iteration `k_idx`
- one epilogue box after the loop

Inside each loop-body box, use the source-order stages:

1. `K copy`
2. `QK mask + issue`
3. `online softmax update`
4. `acc_s -> acc_s_cast copy`
5. `rescale(acc_o)`
6. `V copy`
7. `PV issue`

Then show:

8. `epilogue`
   - normalize `acc_o /= logsum`
   - store `O`
   - write `LSE`

## Why The Tensor Core Lane Matters

Adding a `Tensor Core` lane is strongly recommended because it makes the
advantage of WS kernels much easier to see.

Without that extra lane, a single-CTA diagram mostly shows source order:

- copy
- QK
- softmax
- PV

But the real comparison we care about is:

- how continuously Tensor Cores stay busy
- how much non-TC work sits between QK and PV windows
- how much of the softmax/update work can or cannot overlap with GEMM issue

That is exactly where the WS milestones become visually stronger:

- `pre_pr871`:
  - one CTA
  - software pipeline
  - Tensor Core utilization is tied to that one CTA's local loop structure
- `pr871_base / reorder / anchor`:
  - producer/consumer WS schedule
  - multiple lanes
  - better opportunity to keep Tensor Cores fed while other work proceeds

So yes: a `Tensor Core` lane is one of the best ways to make the WS advantage
visible in the final figures.

## How To Draw The Tensor Core Lane

For `pre_pr871`, the `Tensor Core` lane should be shown as structural, not yet
cycle-accurate:

- mark a `QK` active window
- later mark a `PV` active window
- make those active windows visibly longer than the corresponding `QK issue` /
  `PV issue` boxes on the CTA lane
- keep the gap between them visible
- repeat this pattern across loop iterations

This is enough to communicate the important point:

- in the pre-WS kernel, Tensor Core work is gated by one CTA's local pipeline
- in the WS kernels, Tensor Core work can be fed by a more explicit
  producer/consumer schedule

This distinction matters because:

- `issue` is only the launch / instruction-emission side
- `active` covers the longer period where Tensor Core work is actually in flight

The lane itself should still be drawn structurally, not as a numerically scaled
hardware percentage.

More precise wording for the lane:

- `Tensor Core active windows`
- `Tensor Core issue windows`

More precise wording for nearby profiler-backed callouts:

- `Tensor Core utilization = xx%`
- `Tensor pipe utilization = xx% of peak elapsed`

Less precise wording to avoid:

- using a numeric utilization percentage as if it were the width of the lane

## What Can Be Measured Today

Current validated coarse timing source:

- [20260415_timeline_probes_pre_pr871.json](/home/ga/TileOPs/experiments/ws_kernel_evolution/data/20260415_timeline_probes_pre_pr871.json)

Current coarse numbers for the representative last causal tile:

- `loop_body_total = 174390.0 cycles`
- `epilogue_total = 787.6 cycles`

These came from:

- [bench_timeline_pre_pr871_wgmma_total_cycle.py](/home/ga/TileOPs/experiments/ws_kernel_evolution/scripts/bench_timeline_pre_pr871_wgmma_total_cycle.py)

Measurement setup:

- `B=4, S=4096, H=64, Hkv=8, D=128, causal`
- `block_m=128`
- `block_n=128`
- representative tile:
  - `batch=0`
  - `head=0`
  - last causal `M` tile

## What Should Be Treated As Structural, Not Measured

At the moment, the following sub-stages should be shown as **structural** rather
than cycle-accurate widths:

- `K copy`
- `QK mask + issue`
- `online softmax update`
- `acc_s_cast copy`
- `rescale(acc_o)`
- `V copy`
- `PV issue`

They are valid execution stages, but their individual cycle widths are not yet
reliably measurable in this environment.

## Current Toolchain Limitation

Fine-grained clock instrumentation *inside* the pre-PR871 `T.Pipelined + T.gemm`
loop currently crashes the TileLang/TVM `WgmmaSyncRewriter` pass.

So until that limitation is resolved:

- use coarse measured totals
- use stage ordering as structural information
- do not assign numeric widths to the inner loop sub-stages

## Drawing Rules

Use these rules for the blog and discussion figures:

1. Mark the figure as `single-CTA software pipeline`, not `warp-specialized`.
2. Use a `not to scale` note for inner loop sub-stages unless we later obtain
   stable fine-grained timings.
3. Show `loop_body_total` and `epilogue_total` as the only cycle-accurate
   labels for now.
4. Add a separate `Tensor Core` lane to show QK/PV active windows.
5. Keep `pre_pr871` visually separate from the WS milestones.
6. Do not place `pre_pr871` on a `WG0 / WG1 / WG2` lane chart.

## Practical Recommendation

For the final blog, the cleanest combination is probably:

- `pre_pr871`:
  - one standalone schematic figure
  - single-CTA software-pipeline loop
  - plus one `Tensor Core` utilization callout lane
  - coarse measured totals only
- `pr871_base / reorder / anchor`:
  - one shared WS-style timeline family
  - multi-lane handoff diagrams
  - measured front/core/tail overlays where available
