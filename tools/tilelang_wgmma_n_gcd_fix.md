# TileLang wgmma N gcd fix — UPSTREAM CANDIDATE

## Status

ACTIVE — applied to conda env `env_tilelang_20260119`. Required for any
TileLang gemm where the desired wgmma N is not a divisor of 256 (e.g.
block_n = 144 / 176 / 240, all of which are valid Hopper wgmma N values
but get split by the conservative `gcd(., 256)` policy).

## Bug

`tilelang/intrinsics/wgmma_macro_generator.py::_initialize_wgmma_prefix`
picks `inst_n = gcd(warp_col_tiles, 256)`. This is overly conservative —
Hopper wgmma fp16/bf16 supports any N in {8, 16, ..., 256} step 8, so
`inst_n = warp_col_tiles` is valid as long as warp_col_tiles is a multiple
of 8 in [8, 256]. The `gcd` policy was probably written assuming N has to
be a power of 2 sub-multiple of 256.

### Symptom

Lowered wgmma instruction count blows up for block_n values that don't
divide 256:

| warp_col_tiles | gcd(., 256) | wgmma per QK call | factor vs cute single-atom |
| --- | --- | --- | --- |
| 128 | 128 | 8 (1 N tile × 8 K) | 1× ✓ |
| 192 | 64 | 24 (3 N tiles × 8 K) | 3× |
| **176** | **16** | **88 (11 N tiles × 8 K)** | **11×** |
| 144 | 16 | 72 | 9× |
| 240 | 16 | 120 | 15× |

cute's `ss_op_selector` directly picks `wgmma.m64nNk16` for arbitrary valid
N — this is what FA3 uses to get single-atom 176.

### Performance impact (GQA fwd, B=4 H=64 Hkv=8 D=128 fp16, H200 locked)

| block_n | unpatched | patched | Δ |
| --- | --- | --- | --- |
| 128 (S=4224) | 553.6 / 554.6 | 553.6 / 554.6 | 0 (already optimal) |
| 176 (S=4224) | 420.3 (-24.3%) | 584.6 / 605.3 (+5.6%/+9.1%) | **+39% / +44%** |
| 192 (S=4224) | 543.7 (-2.0%) | 562.0 / 589.9 (+1.5%/+6.4%) | +3.4% / +8.5% |

## Fix

See `tilelang_wgmma_n_gcd_fix.patch` in this directory. One-liner: when
`warp_col_tiles` is itself a valid wgmma N (multiple of 8 in [8, 256]),
use it directly instead of taking `gcd(., 256)`.

## Caveats

- The wgmma N must also be valid for the chosen swizzle mode. For our test
  shapes (N ∈ {176, 192}, fp16, K-major B operand) this works without
  changing swizzle. Other shapes / dtypes may need additional swizzle
  consideration — the assertion at the original site already catches
  invalid N values.
- block_n = 192 hits a register spill cliff (96 fp32/thread for acc_s
  pushes total live regs over the 240 setmaxnreg quota), so 192 is slower
  than 176 even with the patch. 176 is the FA3-aligned sweet spot.

## Upstream PR plan

1. File a TileLang issue with the symptom table above.
2. PR with the one-liner fix + a test case at warp_col_tiles ∈ {144, 176,
   240} that the patched path picks the single-atom wgmma.
3. Optionally extend the fix to also try larger atoms when warp_col_tiles
   is a multiple of (e.g.) 256 (decomposing 256 into 1×256 instead of 2×128).

## Reproducing

See `_dump_blockn_wgmma.py` (counts wgmma_ss / wgmma_rs templates per
block_n) and `_bench_blockn_sweep.py` (bench all three block_n on a single
GPU). The patched conda env was used for all `_test_ws_fa3_v2_threadbind`
and `_test_ws_fa3_v2_tb_mbarrier` benches in issue #9 follow-ups.
