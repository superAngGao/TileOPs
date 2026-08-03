# FP8 GQA AKO

This directory records the optimization ladder for the FP8 Tensor Core GQA
prefill path.

## Goal

Bring the TileOps FP8 GQA prefill implementation to the FA3 performance
neighborhood while:

- preserving the canonical TileOps op and manifest contracts;
- keeping correctness, benchmark, lowering, and attribution gates explicit;
- using TileLang primitives wherever their performance is within measurement
  noise of the best validated implementation;
- retaining low-level helpers only where the current TileLang/lowering contract
  cannot express the required Hopper fragment or pipeline behavior efficiently.

## Fixed Scope

- GPU: NVIDIA H200, GPU 4, SM clock fixed at 1500 MHz
- TileOps base: `8e939500584e5ddb2083b9652a7adca6fff3b2ee`
- Torch: `2.10.0+cu129`
- TileLang: `0.1.11+cu129.git65dbc983`
- Q/K/V: `torch.float8_e4m3fn`
- output: FP16 or BF16
- head dimension: 128
- dense uniform non-causal prefill
- descale ABI: FP32 `[batch, heads_kv]`

The initial performance surface is the manifest-owned set:

- `S=896, H=32, Hkv=8`
- `S=1792, H=32, Hkv=8`
- `S=3584, H=64, Hkv=8`
- `S=7168, H=64, Hkv=8`

Feature expansion such as causal attention, arbitrary sequence tails, varlen
batches, decode, or paged FP8 KV cache is outside the performance-search loop.

## Gates

Every candidate must pass:

1. correctness against the dequantized FP32 reference, including output, LSE,
   and non-finite checks;
2. a fixed H200 benchmark against the current TileOps baseline and FA3;
3. lowering inspection for register use, spills, WGMMA grouping, and
   `WARPGROUP.DEPBAR` / `WARPGROUP.ARRIVE` behavior;
4. attribution: wrapper, local schedule, compiler-contract probe, and
   integrated-kernel rows remain separate.

## Runtime Images

- TileOps release contract:
  `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10`
- Same Torch/TileLang stack with the prebuilt FA3 baseline:
  `ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10-dev`

No package is installed or upgraded inside either image.

## Current Accepted Candidate

Round 069 defers the online-softmax row-sum quad reduction until finalization
for schedules shorter than 32 K/V tiles. It keeps lane-local partial sums
through the tile loop, then uses TileLang `T.shfl_xor` primitives to combine
them once per output row. The 32-tile S7168 shape retains Round 061's per-tile
reduction because the faster consumer exposed a repeated-launch liveness wall
in the existing producer/V-buffer protocol.

The official runner test file passes all ten cases, including direct output and
LSE checks plus explicit S3584/S7168 H64 liveness coverage. On the fixed H200
surface, the hybrid dispatch reaches 83.6-91.5% of FA3 throughput. It improves
Round 061 by 5.8-7.9% on the first three FP16 shapes and stays within noise at
S7168:

| Shape | Round 061 | Round 069 | Improvement | FA3 | R069 / FA3 throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 FP16 | `0.033290 ms` | `0.031362 ms` | `5.79%` | `0.028694 ms` | `91.49%` |
| S1792 FP16 | `0.105113 ms` | `0.096846 ms` | `7.86%` | `0.086032 ms` | `88.83%` |
| S3584 FP16 | `0.643138 ms` | `0.602913 ms` | `6.25%` | `0.536506 ms` | `88.99%` |
| S7168 FP16 | `2.446207 ms` | `2.437024 ms` | `0.38%` | `2.043584 ms` | `83.86%` |

Raw latency and NCU evidence is archived in `results/round069/`.
