# FP8 GQA AKO Log

## Status

- Maximum rounds: 300
- Selected production candidate: current main baseline
- Best validated candidate: current main baseline
- Current structural question: whether the latest TileLang/lowering stack can
  preserve FA3-style grouped QK/PV overlap without fragment-layout conversion,
  register spilling, or conservative scoreboard serialization.

## Round 000: Contract Freeze

**Hypothesis**

A clean, fixed contract is required before interpreting any optimization as a
kernel improvement.

**Action**

- cloned current upstream `main`;
- created branch `perf/gqa-fp8-ako-fa3-level`;
- fixed GPU 4 and the runtime images listed in `README.md`;
- separated TileOps-only validation from the FA3 A/B environment;
- retained the previous dirty worktree as historical evidence only.

**Gate result**

Passed:

- official-runner correctness: `8 passed`;
- stable benchmark: `warmup=5`, `repeat=20`, `trials=3`;
- same-process TileOps/FA3 comparison on GPU 4.

## Round 001: Clean Main Baseline

**Hypothesis**

The public `[B, Hkv]` descale wrapper may explain part of the observed gap, but
the serial QK/softmax/PV mainloop is expected to remain the dominant wall.

**Action**

Measured:

- canonical TileOps op with 2D descales;
- the same TileOps kernel with scales pre-expanded outside the timed region;
- FA3 with the same FP8 Q/K/V and descales.

**Result**

| Shape | TileOps canonical | TileOps pre-expanded | FA3 | Canonical / FA3 | Descale expansion |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.050490 ms | 0.041937 ms | 0.028675 ms | 1.761x | 20.4% |
| S896 H32/Hkv8 BF16 | 0.050643 ms | 0.042058 ms | 0.028728 ms | 1.763x | 20.4% |
| S1792 H32/Hkv8 FP16 | 0.140337 ms | 0.131501 ms | 0.086254 ms | 1.627x | 6.7% |
| S1792 H32/Hkv8 BF16 | 0.139363 ms | 0.130789 ms | 0.086067 ms | 1.619x | 6.6% |
| S3584 H64/Hkv8 FP16 | 0.812658 ms | 0.803397 ms | 0.534705 ms | 1.520x | 1.2% |
| S3584 H64/Hkv8 BF16 | 0.812694 ms | 0.805893 ms | 0.529788 ms | 1.534x | 0.8% |
| S7168 H64/Hkv8 FP16 | 3.054555 ms | 3.045751 ms | 2.040679 ms | 1.497x | 0.3% |
| S7168 H64/Hkv8 BF16 | 3.060255 ms | 3.047775 ms | 2.033469 ms | 1.505x | 0.4% |

**Decision**

Direct 2D descale consumption is selected as a low-risk short-sequence
improvement. It is not treated as the structural answer: long-sequence
performance still requires about a 1.5x improvement to reach FA3.

## Round 002: Direct 2D Descale Consumption

**Hypothesis**

The kernel can consume the public FA3 `[B, Hkv]` descale ABI directly. Removing
the repeated host-side expansion should recover the pre-expanded baseline
without changing the mainloop.

**Action**

- changed the TileLang kernel tensor contract from internal 3D scale tensors to
  direct 2D descales;
- mapped each query head to its owning KV head inside the persistent task;
- removed `repeat_interleave`, `expand`, and `contiguous` from kernel dispatch;
- removed the undocumented direct-kernel 3D scale compatibility path.

**Gate result**

- ruff: pass;
- official-runner correctness: `8 passed`;
- no CUDA-events fallback accepted in benchmark rows.

| Shape | Round 001 | Round 002 | Change | FA3 | Round 002 / FA3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S896 H32/Hkv8 FP16 | 0.050490 ms | 0.042019 ms | -16.8% | 0.028622 ms | 1.468x |
| S3584 H64/Hkv8 FP16 | 0.812658 ms | 0.803116 ms | -1.2% | 0.534503 ms | 1.503x |
| S7168 H64/Hkv8 FP16 | 3.054555 ms | 3.034873 ms | -0.6% | 2.044088 ms | 1.485x |

**Decision**

Accepted. This closes the wrapper allocation gap and makes the public ABI the
native kernel ABI. The long-sequence structural wall is unchanged.
