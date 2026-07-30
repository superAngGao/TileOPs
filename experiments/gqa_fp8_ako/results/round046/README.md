# Round 046 Partial Diagnostic

Round 046 removed the softmax-scale shared-memory round trip and improved the
S896 and S3584 cases, but the required S1792/H32 FP16 case failed to complete
in two independent empty TileLang caches.

The JSONL files in this directory are retained only as evidence for the
short-shape performance result. They are not a complete benchmark surface and
must not be used as an accepted-candidate result.
