# Round 069 Evidence

This directory archives the accepted deferred row-sum reduction candidate.

- `s*.jsonl`: CUPTI latency rows for TileOps and FA3, collected with warmup 5,
  repeat 20, three trials, L2 flush, and median-trial-mean reporting.
- `ncu_s3584_full.csv`: full-kernel S3584 FP16 NCU metrics.
- `ncu_s3584_source.csv`: source-level S3584 FP16 NCU metrics used for the
  opcode and stall attribution in the AKO log.

Environment: NVIDIA H200 GPU 4 at 1500 MHz, Torch `2.10.0+cu129`, CUDA 12.9,
and the unmodified
`ghcr.io/tile-ai/tileops-runner:65dbc98-torch2.10-dev` image.

The official-runner `tests/ops/attention/test_gqa_fp8.py` suite passed all
eight cases before these rows were accepted.
