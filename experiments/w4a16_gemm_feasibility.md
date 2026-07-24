# W4A16 GEMM feasibility on H200

Date: 2026-07-23

## Environment

- GPU: NVIDIA H200 NVL, SM90
- PyTorch: 2.9.1+cu128
- TileLang: 0.1.12
- TileOps base commit: `76f35fd`
- Experiment: `experiments/w4a16_gemm_feasibility.py`
- Benchmark: `experiments/bench_w4a16_gemm.py`
- Marlin benchmark: `experiments/bench_marlin_w4a16.py`

The Marlin comparison uses a separate environment with vLLM 0.25.1 and
PyTorch 2.11.0+cu130 on the same H200.

This is a local feasibility experiment, not a public TileOps op or manifest
contract.

## Physical and numerical contract under test

- Activation: FP16 or BF16 `[M, K]`
- Logical weight: group-wise INT4 `[N, K]`
- Packed storage: `uint8[N, K / 2]`
- Nibble order: low nibble is even K, high nibble is odd K
- Group size: 128
- Scale: FP32 `[N, K / 128]`
- Zero point: UINT8 `[N, K / 128]`
- Dequantization: `w_a16 = (q_u4 - zero) * scale`
- Output: activation dtype `[M, N]`
- Accumulation: FP32

The LOP3 path additionally applies TileLang's 32-bit-word weight interleave.
This proves symmetric and affine group-wise arithmetic, but it is not yet an
exact AWQ or GPTQ checkpoint layout.

## Lowering evidence

The GEMM paths generate FP16/BF16 Tensor Core instructions after dequantization,
for example:

```cpp
tl::wgmma_ss<
    tl::DataType::kFloat16,
    tl::DataType::kFloat16,
    tl::DataType::kFloat32,
    64, 64, 16, false, false, 1, 1
>(...);
```

There is no native INT4-by-FP16 WGMMA. Packed INT4 is loaded, decoded and
scaled into an FP16 shared-memory tile before WGMMA.

## Pipelines tested

### Automatic warp specialization

Source uses `T.Pipelined` and `T.gemm`. TileLang lowers it to a 256-thread
producer/consumer kernel:

- WG0: activation cp.async, packed-weight TMA, scale/zero cp.async
- WG1: LOP3 dequantization followed by WGMMA

Dequantization and WGMMA remain serial within WG1.

### Explicit two-warpgroup pipeline

- WG0: load and wait, then LOP3 dequantize the next tile
- WG1: WGMMA the current tile
- Explicit `load_full`, `ready`, and `empty` barriers
- Generic-to-async proxy fence before exposing the dequantized tile

This overlaps load+dequant with WGMMA, but makes WG0 a long combined stage.

### Explicit three-warpgroup pipeline

- WG0: load A, packed W, scale and zero
- WG1: LOP3 decode, subtract zero, multiply scale, write FP16 shared tile
- WG2: WGMMA
- Explicit `load_full -> ready -> empty` barrier chain
- Generic-to-async proxy fence between WG1 and WG2

This realizes:

```text
load[k+2]  ||  dequant[k+1]  ||  WGMMA[k]
```

Omitting `T.fence_proxy_async()` before WG1 signals `ready` caused about 4.2%
incorrect elements. With the fence, three consecutive correctness launches
pass.

## Benchmark protocol

Cross-implementation results use TileOps' production
`BenchmarkBase.profile()` path:

- `bench_kernel()` flushes the full device L2 before every invocation;
- tensor inputs rotate through three cloned address sets when their combined
  clone pool is at most 1 GiB;
- timing is pure device-kernel time collected through CUPTI, with the
  infrastructure's CUDA-event fallback if CUPTI is unavailable;
- each result uses 10 cold-cache warmups and 3 trials of 50 repetitions, then
  reports the median trial mean;
- W4A16, `GemmOp`, and `torch.matmul` receive the same activation and the same
  dequantized logical weight;
- all three implementations use FP16 inputs and FP32 accumulation;
- logical throughput is `2*M*N*K / latency`;
- bandwidth counts activation, weight/quantization metadata, and output bytes
  for the implementation being measured.

The H200 NVL reports 60 MiB of L2. The full-L2 flush is material here: the
two GEMV FP16 weights occupy 28 MiB and 128 MiB, while their packed W4 plus
FP32 scale and UINT8 zero-point storage occupies 7.55 MiB and 34.5 MiB.

Run the benchmark from the repository root with:

```bash
python -m experiments.bench_w4a16_gemm
```

GPU clocks were not locked because the H200 is a shared remote resource. A
second run reproduced the speedup ratios within 1.7%.

Marlin is timed through the same `bench_kernel()` helper. Its timed callable
launches only `marlin_gemm`; quantization and physical-layout repacking are
outside the measurement. The benchmark supplies already-repacked random
UINT4 weights, group-128 FP16 scales, and packed zero points. A separate
AWQ-style group-128 correctness test established that Marlin and this
experiment implement the same affine UINT4 arithmetic, although their
physical packed layouts differ.

## Cold-cache results

All reported W4A16 results use affine UINT4, FP16 activation/output, FP32
accumulation, group size 128, and LOP3 decode. GEMM uses the explicit 3-WG
`64x64x64`, two-stage configuration. GEMV uses the register-decode kernel.

| Shape `(M,N,K)` | Implementation | Latency (ms) | TFLOPS | Logical bandwidth (TB/s) | W4 speedup |
|---|---|---:|---:|---:|---:|
| `(128,2112,7168)` | W4A16 | 0.066430 | 58.340 | 0.159 | 1.000x |
|  | TileOps A16 | 0.061485 | 63.032 | 0.531 | **0.926x** |
|  | torch/cuBLAS A16 | 0.018025 | 215.010 | 1.812 | **0.271x** |
| `(128,7168,2048)` | W4A16 | 0.037043 | 101.452 | 0.277 | 1.000x |
|  | TileOps A16 | 0.023698 | 158.583 | 1.338 | **0.640x** |
|  | torch/cuBLAS A16 | 0.014081 | 266.897 | 2.253 | **0.380x** |
| `(1,7168,2048)` | W4A16 | 0.007460 | 3.936 | 1.063 | 1.000x |
|  | TileOps A16 | 0.017483 | 1.679 | 1.680 | **2.344x** |
|  | torch/cuBLAS A16 | 0.013638 | 2.153 | 2.154 | **1.828x** |
| `(1,8192,8192)` | W4A16 | 0.027279 | 4.920 | 1.327 | 1.000x |
|  | TileOps A16 | 0.062373 | 2.152 | 2.152 | **2.287x** |
|  | torch/cuBLAS A16 | 0.048420 | 2.772 | 2.773 | **1.775x** |

`W4 speedup` is `baseline latency / W4 latency`; values above 1 mean W4 is
faster. Under the cold-cache protocol, W4A16 wins clearly for M=1 decode but
does not beat either A16 baseline for M=128.

## Decode split-K experiment

The original GEMV assigns one warp to an output column. That warp walks the
entire K dimension serially in 256-element chunks. The experimental
`split_k_warps` mode assigns 2, 4, 8, or 16 warps in one CTA to the same
column:

1. each warp processes a strided K partition with the unchanged LOP3 decode;
2. each warp performs its existing 32-lane shuffle reduction;
3. lane zero writes one FP32 partial to shared memory;
4. one warp sums the partials and writes the FP16 output.

This is CTA-local split-K. It adds no global workspace or atomic operation and
therefore isolates whether the serial K dependency is limiting the original
kernel.

All rows below use `M=1`, `N=8192`, affine UINT4 group-128 weights, and the
same official cold-cache CUPTI protocol. `Single warp` is the original
`split=1,n_partition=4` launch. The best configuration was selected from
`split={1,2,4,8,16}` and several `n_partition` values.

| K | Single warp (ms) | Best CTA split-K | Best (ms) | Split-K gain | Marlin (ms) | Best TileOps / Marlin |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.008124 | `split=1,n=4` | 0.008124 | 1.000x | 0.009736 | **0.834x** |
| 4096 | 0.014748 | `split=2,n=2` | 0.013771 | 1.071x | 0.013454 | 1.024x |
| 8192 | 0.027390 | `split=4,n=1` | 0.024499 | 1.118x | 0.021714 | 1.128x |
| 16384 | 0.051289 | `split=4,n=2` | 0.046742 | 1.097x | 0.037830 | 1.236x |

The last column is latency ratio, so values below one favor TileOps. At
K=2048 the low-overhead single-warp kernel is 1.20x faster than Marlin.
The crossover is near K=4096. CTA-local split-K recovers 7-12% for longer K,
but does not reach Marlin: Marlin is 1.13x faster at K=8192 and 1.24x faster
at K=16384.

More split is not monotonically better. At K=8192, `split=8,n=1` takes
0.025968 ms and `split=16,n=1` takes 0.029987 ms, both slower than the
four-way result. At K=16384, `split=8,n=2` regresses to 0.084444 ms. Barrier
cost, larger CTAs, and reduced scheduling flexibility overtake the shorter
per-warp dependency chain.

The experiment confirms that serial K processing explains part of the
long-K gap. It does not explain all of it. Marlin additionally uses a tiled
Tensor Core main loop, a fixed-SM tile scheduler, and cross-CTA reduction;
the current TileOps GEMV remains a scalar-FMA register-decode kernel.

## Marlin long-K ablations

Source inspection of vLLM 0.25.1 Marlin shows four relevant mechanisms for
small-M, long-K execution:

1. It launches a fixed grid of `sms` CTAs and stripes the `(N tile, K tile)`
   plane evenly across them. On the 132-SM H200, the selected M=1
   configuration is the first valid small-batch candidate: `K128 x N128`,
   256 threads, and four shared-memory stages.
2. One CTA reuses each activation K tile across roughly 128 output columns.
   Packed INT4 weight tiles move through shared memory, but dequantized FP16
   weights go directly to MMA register fragments.
3. The main loop combines a four-stage `cp.async` shared pipeline with
   two-way register buffering. Its loop order explicitly interleaves LOP3
   dequantization with independent `mma.sync.m16n8k16` accumulator updates.
4. If a stripe splits one N slice across CTAs, partials are reduced through a
   small L2-resident output or FP32 workspace guarded by GPU-scope locks.

Two experiments isolate which ideas matter first for TileOps.

First, switching Marlin between its FP32 workspace reduction and direct FP16
output/L2 reduction barely changes cold-cache latency:

| K | Marlin FP32 reduce | Marlin FP16 reduce | Difference |
|---:|---:|---:|---:|
| 2048 | 0.009726 ms | 0.009796 ms | FP32 0.7% faster |
| 8192 | 0.021674 ms | 0.021782 ms | FP32 0.5% faster |
| 16384 | 0.037838 ms | 0.037955 ms | FP32 0.3% faster |

The reduction implementation is therefore not the primary long-K advantage.
The striped split is useful for balancing work across SMs, but its final
reduction cost is already negligible.

Second, the TileOps scalar GEMV was extended so one warp can retain one
activation vector chunk and accumulate 2, 4, or 8 output columns. This reduces
activation reloads and creates independent accumulators without changing the
INT4 arithmetic. A joint sweep over split-K warps, CTA warps, and outputs per
warp produced:

| K | Previous TileOps | Best reuse config `(split,n,outputs)` | Best TileOps | Marlin | TileOps / Marlin |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0.024459 ms | `(8,1,4)` | **0.023317 ms** | 0.021674 ms | 1.076x |
| 16384 | 0.046949 ms | `(8,1,4)` | **0.042472 ms** | 0.037838 ms | 1.122x |

At K=2048, increasing outputs per warp regresses from 0.008037 ms to at least
0.008679 ms, so this must remain a long-K dispatch specialization. At K=8192
the reuse sweep reduces the Marlin gap from about 13% to 8%; at K=16384 it
reduces the gap from about 24% to 12%.

This result prioritizes the remaining work:

1. preserve N tiling and activation reuse in the Tensor Core design;
2. make the packed layout feed each MMA/WGMMA register fragment without
   redundant cross-lane LOP3 decoding;
3. interleave multiple independent accumulator updates with decode and
   prefetch, using double register buffers and a deeper shared pipeline;
4. add fixed-SM stripe scheduling and cross-CTA split-K after a CTA computes a
   sufficiently wide N tile;
5. treat the exact global reduction representation as a later tuning choice.

## TMA N64 and CTA-local split-K experiment

A strict direct-load versus TMA comparison was added for the long-K scalar
path. Both variants compute an `N64 x K256` CTA tile and retain eight output
accumulators per consumer warp. The TMA version uses one producer warpgroup
and two consumer warpgroups:

- the producer copies one activation tile and 64 packed-weight rows into a
  shared-memory ring with TMA, while loading scale/zero metadata once per tile;
- consumers reuse each activation register vector across eight output rows,
  perform LOP3 decode from packed shared memory, and accumulate with FP32 FMA;
- ready/empty mbarriers permit TMA of the next stage to overlap current decode
  and compute.

All variants passed the affine UINT4 correctness check and the generated
source was required to contain TMA lowering. Official cold-cache results:

| K | Direct N64 | TMA S2 | TMA S3 | TMA S4 | Best existing split/reuse GEMV |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0.037598 ms | 0.028615 ms | **0.026758 ms** | 0.028557 ms | 0.023308 ms |
| 16384 | 0.071023 ms | 0.053728 ms | **0.050067 ms** | 0.053984 ms | 0.042461 ms |

At identical CTA geometry, three-stage TMA reduces latency by 28.8% at K=8192
and 29.5% at K=16384. Its minimum-byte effective bandwidth rises from
0.963 to 1.353 TB/s at K=8192 and from 1.019 to 1.446 TB/s at K=16384.
This validates TMA as a useful component of a tiled long-K pipeline.

CTA-local split-K was then added without changing the global output or
requiring a workspace. The useful mapping keeps the CTA at N64: splitting K by
two reduces the number of output warps from eight to four and raises each
warp's independent outputs from 8 to 16. Thus the weight and activation reuse
of the N64 tile is preserved while the serial K dependency is halved. The
producer transfers two contiguous K256 partitions per pipeline step; consumer
lane-zero values are reduced through shared memory after the K loop.

| K | N64 split1 | N64 split2 | N64 split4 | Best gain | Best direct GEMV | Marlin |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 0.026621 ms | **0.024305 ms** (S4) | 0.024624 ms | 8.7% | 0.023317 ms | 0.021674 ms |
| 16384 | 0.050468 ms | 0.044293 ms | **0.043610 ms** (S3) | 13.6% | 0.042472 ms | 0.037838 ms |

All configurations passed the affine UINT4 correctness check. Minimum-byte
effective bandwidth for the best split is 1.49 TB/s at K=8192 and 1.66 TB/s
at K=16384, about 31% and 35% of the H200's nominal 4.8 TB/s. Local split-K
closes most of the original TMA gap: it remains 4.2% and 2.7% behind the best
direct TileOps GEMV, and 12% and 15% behind Marlin.

The negative ablations explain the remaining design constraints:

- shrinking the TMA N tile increases activation duplication and loses badly:
  at K=8192 the best N32 and N16 results are 0.035965 and 0.054987 ms versus
  0.026621 ms for split1 N64;
- at K=8192, keeping N64 while moving to split4/output32 gives 0.024624 ms,
  slightly behind split2, while split8/output64 regresses to 0.044438 ms;
  split8 S4 additionally requests 301056 bytes of dynamic shared memory and
  cannot launch;
- shrinking N with K split is worse still: split2/N32, split4/N16, and
  split8/N8 measure 0.029251, 0.033276, and 0.040024 ms at K=8192;
- replacing the small activation TMA with producer LDG-to-STS is not cheaper.
  It regresses split1 from 0.026621 to 0.030621 ms and split2 from 0.024305
  to at least 0.025063 ms;
- merging all split partitions into one packed-weight TMA is neutral for
  split2 and fails TileLang's default shared-layout bijection at split4, so
  the implementation retains one regular TMA tile per partition.

The current TMA candidate therefore keeps N64 and selects its local K tile by
shape: `K512/split2/output16` at K=8192 and `K1024/split4/output32` at
K=16384. Three or four stages should also be dispatched by shape. The next
substantial experiment should preserve this N reuse while changing CTA
scheduling or packed-fragment decode; split8 is not promising.

## Register-dequant TMA ping-pong

The scalar TMA pipeline was changed from holding a shared slot through the
entire decode/FMA loop to true early-release ping-pong:

1. the producer TMA-loads packed W4, activation, and scale into one ring slot;
2. consumers copy every packed fragment and its metadata into registers;
3. consumers immediately arrive on that slot's `empty` barrier;
4. the producer refills the released slot while consumers perform LOP3,
   dequantization, and FP32 accumulation entirely from registers.

The best mapping uses one producer warpgroup, four consumer warpgroups, N64,
four-way CTA-local K partitioning, 16 outputs per consumer warp, and two
shared slots. Official cold-cache results:

| Shape | Best direct GEMV | Double buffer | Triple buffer | Best TMA vs direct |
|---|---:|---:|---:|---:|
| `(1,8192,8192)` | **0.023286 ms** | 0.023812 ms | 0.024250 ms | 2.3% slower |
| `(1,8192,16384)` | 0.042680 ms | **0.041552 ms** | 0.042300 ms | 2.6% faster |

Triple buffering regresses by about 1.8% on both shapes. Early release already
lets two slots cover TMA with register-only consumer work. A third slot raises
dynamic shared memory from roughly 76 KB to 113 KB and changes the cheap
two-slot stage/phase arithmetic into modulo-three indexing, without shortening
the compute-limited consumer stage. The dispatch therefore retains double
buffering; triple buffering remains a benchmark ablation.

The scale tile was also changed from strided per-thread global loads to one
two-dimensional TMA transfer. The narrow UINT8 zero tile is not TMA-capable in
the current TileLang lowering and uses cooperative `T.copy`. This reduces
Nsight Compute's excessive global sectors from 884736/1056768 to
98304/139264, an 89% reduction in absolute excess sectors, with neutral
end-to-end latency.

Nsight Compute on `(1,8192,16384)` shows why double buffering alone does not
saturate HBM:

| Metric | Value |
|---|---:|
| DRAM throughput | 30.5% / 1.41 TB/s |
| Compute throughput | 63.7% |
| Issue slots busy | 70.6% |
| Registers per thread | 96 |
| Achieved occupancy | 30.5% |
| Grid | 128 CTAs on 132 SMs |

The kernel is consumer-compute limited after ping-pong overlap: both the FP16
dequantization and FP32 FMA pipelines are about 63% utilized, while DRAM is
only about 30%. The next optimization must keep the W4 tile register-resident
and reduce consumer instructions, preferably by feeding a non-redundant
MMA/WGMMA register fragment. Re-materializing an FP16 weight tile in shared
memory would undo the bandwidth benefit.

## M=1 Tensor Core instruction experiment

The Hopper path must distinguish three separate choices:

- Warp-level MMA: forcing `T.gemm` through TileLang's `tl.disable_wgmma` pass
  produces `tl::mma_sync<...,16,8,16,...>`. Its global-to-shared transfers use
  `cp.async`, but the matrix instruction is warp-synchronous.
- SS-WGMMA: both operands are FP16 shared-memory tiles. The transposed
  formulation computes `C.T[64,8] = W[64,K] @ padded_A.T[K,8]`, mapping all 64
  WGMMA rows to real output channels and padding only seven batch columns.
- RS-WGMMA: packed INT4 is loaded to shared memory and decoded into WGMMA's
  register A fragment; the padded activation is the shared B operand. This
  removes the materialized FP16 weight tile and one warpgroup.

Both WGMMA forms lower through TileLang's WGMMA support and issue Hopper's
asynchronous matrix multiply. `depth=0` waits after each committed K tile;
`depth=1/2` keeps that many WGMMA groups pending across K iterations. SS-WGMMA
still overlaps its independent loader, dequantizer, and compute warpgroups
even at depth zero. RS-WGMMA depth one additionally tries to overlap the
compute WG's next register decode with the preceding WGMMA.

All variants passed the affine UINT4 correctness check. Measurements use the
official cold-cache CUPTI protocol:

| Shape | split-K GEMV | SS-WGMMA transposed d0 | SS d1 | SS d2 | RS d0 | RS d1 | M16 MMA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(1,8192,2048)` | 0.00808 | **0.01620** | 0.02466 | 0.02468 | 0.02190 | 0.02950 | 0.02790 |
| `(1,8192,8192)` | 0.02451 | **0.05716** | 0.09675 | 0.09766 | 0.07981 | 0.11460 | 0.11072 |

Latencies are milliseconds. Transposing the SS-WGMMA mapping improves the
previous `A[64,K] @ W.T[K,64]` three-WG result by 20-21%, but is still
2.0-2.3x slower than the scalar split-K GEMV.

Keeping WGMMA groups pending is not beneficial in this pipeline. Relative to
SS depth zero, depth one is 1.52x slower at K=2048 and 1.69x slower at K=8192;
depth two does not recover the loss. The accumulator has a serial K
dependency, while the three independent warpgroups already provide
load/dequant/compute overlap. Delaying shared-stage release instead constrains
the two-slot producer ring.

The RS prototype proves that TileLang can feed a dequantized register fragment
to WGMMA correctly. It is not yet an optimized data path: the LOP3 intrinsic
produces eight adjacent FP16 values in one lane, whereas the RS-WGMMA fragment
spreads each group over four lanes. The prototype redundantly decodes the
group in those four owner lanes, so it loses to SS depth zero. A competitive RS
version needs a quad-lane exchange or a packed layout/decode intrinsic designed
for the WGMMA register-fragment layout.

The current result therefore supports an asynchronous WGMMA implementation
for the Hopper Tensor Core experiment, but not its use as the M=1 dispatch
winner yet. The scalar split-K GEMV remains the measured decode path while the
RS fragment decode is optimized.

## Preliminary tuning results

The following measurements predate the official benchmark adapter. They use
manual CUDA-event timing without an L2 flush and are retained only to compare
W4 pipeline variants. They must not be used for cross-implementation claims.

The `A16 baseline` column below is pre-dequantized `torch.matmul`, not TileOps
`GemmOp`.

### Model-shaped GEMM

| Shape `(M,N,K)` | Pipeline | Stages | Latency (ms) | Throughput | A16 baseline | Relative |
|---|---:|---:|---:|---:|---:|---:|
| `(128,2112,7168)` | automatic WS | 2 | 0.055893 | 69.339 TFLOPS | 199.671 TFLOPS | 34.7% |
| `(128,2112,7168)` | explicit 2 WG | 2 | 0.086614 | 44.745 TFLOPS | 188.631 TFLOPS | 23.7% |
| `(128,2112,7168)` | explicit 3 WG | 2 | 0.049099 | 78.933 TFLOPS | 202.543 TFLOPS | 39.0% |
| `(128,7168,2048)` | explicit 2 WG | 2 | 0.052818 | 71.152 TFLOPS | 234.412 TFLOPS | 30.4% |
| `(128,7168,2048)` | explicit 3 WG | 2 | 0.032429 | 115.888 TFLOPS | 230.230 TFLOPS | 50.3% |

The automatic-WS path was incorrect for `(128,7168,2048)` with about 5.5%
mismatched elements, so no automatic-WS performance number is accepted for
that shape.

For `(128,2112,7168)`, the explicit three-WG stage sweep was:

| Stages | Throughput |
|---:|---:|
| 2 | 78.5-78.9 TFLOPS |
| 3 | 71.9 TFLOPS |
| 4 | 68.9 TFLOPS |

Larger `block_m` or `block_n` also regressed. The current best tested
configuration is `64x64x64`, two ring slots.

### Decode GEMV

A separate LOP3 register-decode + warp-reduction mode was tested for `M=1`.

| Shape `(M,N,K)` | W4A16 | Pre-dequantized FP16 | Speedup |
|---|---:|---:|---:|
| `(1,8192,8192)` | 0.027458 ms | 0.044024 ms | 1.60x |

This old hot-cache comparison is superseded by the cold-cache table above.
The GEMV mode does not use WGMMA.

## Conclusions

1. W4A16 is implementable in TileLang on H200.
2. Both symmetric and affine group-wise INT4 arithmetic are correct.
3. TileLang lowers the dequantized tiles to native FP16/BF16 WGMMA.
4. Packed layout and LOP3 decoding are required for useful performance.
5. The explicit two-WG design is correct but not competitive in the tested
   model shapes.
6. The explicit three-WG design is the best GEMM candidate tested so far.
7. The cold-cache benchmark validates a separate GEMV dispatch for M=1:
   W4A16 is 2.29-2.34x faster than TileOps A16 and 1.78-1.83x faster than
   torch/cuBLAS for the two tested decode shapes.
8. At M=128 the current W4A16 GEMM reaches 64-93% of TileOps A16 and 27-38%
   of torch/cuBLAS, so GEMM optimization remains an upstream gate.
9. CTA-local split-K is useful from approximately K=4096 and recovers up to
   11.8% in the tested decode sweep, validating the long-K diagnosis.
10. Split-K alone is insufficient to match Marlin. A practical experimental
    dispatch is single-warp for K=2048, two-way split for K=4096, and four-way
    split for K>=8192; closing the remaining long-K gap requires a tiled
    Tensor Core and/or cross-CTA design.

## Remaining gates before an upstream issue or manifest

- Confirm the Marlin crossover and dispatch thresholds on more N/K model
  shapes, then compare against a second W4A16 library such as CUTLASS or
  BitBLAS.
- Define exact AWQ and GPTQ physical layouts, including packed zero points,
  permutation/interleave and possible `g_idx`.
- Add BF16 fast decode; TileLang 0.1.12's LOP3 helper exposes FP16 output only.
- Resolve or avoid the automatic-WS wide-N correctness failure.
- Add bias/epilogue and shape/alignment dispatch rules.
- Test more decode, mid-M and prefill model shapes.
- Move the experiment into TileOps op/test/benchmark structure only after the
  physical contract is reviewed.
