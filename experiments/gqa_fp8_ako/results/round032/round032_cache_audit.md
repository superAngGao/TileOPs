# Round 032 TileLang Cache Audit

## Why This Audit Was Needed

Rounds 028 and 029 changed only the C++ helper injected into generated CUDA.
The shared TileLang cache did not reliably include that helper content in its
cache key, so a candidate could reuse a cubin compiled from an older helper.
This made the earlier shared-cache lowering comparison untrustworthy.

Round 032 rebuilt each S896 candidate with a unique empty directory mounted at
`/ci-cache/tilelang`. The source, runtime image, shape, GPU, and NCU command
were otherwise fixed.

## Fresh-Cache Results

Shape: `B=1, S=896, H=32, Hkv=8, D=128`, FP16 output.

| Candidate | Duration | Executed instructions | No eligible warp | Eligible warps / scheduler | PRMT | F2FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 022 baseline | 34.11 us | 7,068,440 | 64.71% | 0.48 | 412,160 | 458,752 |
| Round 028 CUTLASS converter | 36.51 us | 7,064,890 | 66.58% | 0.47 | 412,160 | 458,752 |
| Round 029 inline PTX merge | 36.74 us | 7,079,181 | 66.60% | 0.47 | 412,160 | 458,752 |

The old shared-cache profile that had been treated as the baseline measured
38.30 us, 7.82 M executed instructions, and 469,504 PRMT instructions. Those
figures do not describe the fresh Round 022 compiler artifact.

## Decisions

- All future candidate compiles use a unique empty TileLang cache.
- Round 028 remains rejected. A fresh-cache S3584 run also reproduced its
  long-loop liveness failure, and its short-shape lowering did not reduce PRMT.
- Round 029 is rejected because its fresh artifact did not reduce PRMT or F2FP
  and regressed S896 by 7.7%. The earlier shared-cache S3584 liveness
  observation is not used as evidence.
- Round 022 remains the accepted implementation.

The additional PRMT in the Round 027 TileOps-versus-FA3 profile remains a real
whole-kernel difference, but these two local pack rewrites do not reproduce
FA3's merged conversion lowering. The next experiment must target schedule or
typed-fragment structure and must be compiled from an isolated cache.
