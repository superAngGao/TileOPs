# Round 027 S3584 FP16 Source Profile

Contract:

- shape: `B=1, S=3584, H=64, Hkv=8, D=128`;
- GPU: H200 at 1500 MHz;
- TileOps implementation: accepted Round 022;
- comparison: FA3 FP8 forward;
- profiler: Nsight Compute 2025.2.1.

| Metric | TileOps | FA3 |
| --- | ---: | ---: |
| Duration | 659.040 us | 533.664 us |
| Dynamic instructions | 202.45 M | 183.21 M |
| Tensor-pipe active | 40.06% | 49.10% |
| Eligible warps / scheduler | 0.59 | 0.65 |
| No eligible warp | 58.69% | 54.26% |
| L1 hit rate | 63.80% | 99.36% |
| L2 hit rate | 97.94% | 93.04% |
| Long-scoreboard samples | 10,995 | 6,337 |

Both kernels launch the same `132 x 384` grid and use 168 registers per
thread. FA3 selects `Tile128x224x128`, two mainloop stages, register-source PV,
and intra-warpgroup overlap; GQA packing is disabled for this shape.

The largest TileOps source-correlated stall is the producer's `k_empty` wait:
after transposing V and publishing `v_full`, the producer has caught the
consumer and cannot reuse the next K buffer. It contributes 5,766
long-scoreboard samples. Producer waits for the two V TMA loads contribute
another 2,673 samples.

The softmax arithmetic and Tensor Core instruction counts are nearly equal.
The largest instruction-count discrepancy is 6.34 M additional `PRMT`
instructions in TileOps. The accepted helper converts 56 floats to FP8x2 and
then executes 28 `PRMT 0x5410` instructions to form the 28 FP8x4 register-A
operands before each PV issue. FA3 merges the second FP8x2 conversion directly
into the 32-bit destination. Across both TileOps consumer warpgroups, removing
that extra pack operation would eliminate about 6.42 M dynamic instructions.

