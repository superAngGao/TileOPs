## Manifest status

![ops](https://img.shields.io/badge/ops-176-blue) ![implemented](https://img.shields.io/badge/implemented-163%20%2F%20176%20%2893%25%29-brightgreen) ![spec--only](https://img.shields.io/badge/spec--only-13-orange)

### Per-family coverage

| Family | Implemented | Spec-only | Total | Progress | Workloads |
| --- | ---: | ---: | ---: | --- | ---: |
| `attention` | 14 | 0 | 14 | `██████████` 100% | 78 |
| `attention_indexing` | 2 | 0 | 2 | `██████████` 100% | 5 |
| `bmm` | 2 | 0 | 2 | `██████████` 100% | 14 |
| `convolution` | 3 | 0 | 3 | `██████████` 100% | 45 |
| `elementwise` | 69 | 0 | 69 | `██████████` 100% | 146 |
| `gemm` | 4 | 0 | 4 | `██████████` 100% | 37 |
| `linear_attention` | 5 | 6 | 11 | `█████░░░░░` 45% | 43 |
| `mamba` | 0 | 7 | 7 | `░░░░░░░░░░` 0% | 29 |
| `moe` | 7 | 0 | 7 | `██████████` 100% | 58 |
| `normalization` | 10 | 0 | 10 | `██████████` 100% | 50 |
| `pool` | 12 | 0 | 12 | `██████████` 100% | 36 |
| `position_encoding` | 6 | 0 | 6 | `██████████` 100% | 13 |
| `quantization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `reduction` | 19 | 0 | 19 | `██████████` 100% | 62 |
| `regularization` | 1 | 0 | 1 | `██████████` 100% | 3 |
| `scan` | 2 | 0 | 2 | `██████████` 100% | 4 |
| `sequence_modeling` | 5 | 0 | 5 | `██████████` 100% | 15 |
| `spectral` | 1 | 0 | 1 | `██████████` 100% | 3 |

### Spec coverage

| Field | Coverage |
| --- | ---: |
| `ref_api` | 176 / 176 (100%) |
| `roofline` (func or flops+bytes) | 176 / 176 (100%) |
| `source.kernel_map` | 169 / 176 (96%) |
| `source.bench_manifest_driven` | 163 / 176 (93%) |

**Workloads:** 644 total — 3.77 per implemented op.

### Conformance gaps

- Implemented ops without `kernel_map`: **0**
- Implemented ops without `roofline`: **0**
- Implemented ops without `source.bench_manifest_driven`: **0**
- Implemented ops with fewer than two workloads: **0**

<details><summary>Spec-only ops (13)</summary>

| | | |
| --- | --- | --- |
| `CBProducerOp` | `DaCumsumFwdOp` | `DeltaNetBwdOp` |
| `DeltaNetFwdOp` | `GLABwdOp` | `GLAFwdOp` |
| `GatedDeltaNetBwdOp` | `GatedDeltaNetFwdOp` | `Mamba2FwdOp` |
| `SSDChunkScanFwdOp` | `SSDChunkStateFwdOp` | `SSDDecodeOp` |
| `SSDStatePassingFwdOp` |  |  |

</details>
