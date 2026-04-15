# WS Kernel Evolution Study

This directory collects scripts, raw measurements, figures, and notes for the
blog-quality reconstruction of the WS FA3 kernel optimization path.

Current milestone nodes:

1. Pre-PR-871 baseline
2. PR 871
3. PR 871 + persistent reorder
4. Anchor causal

We also keep failed or weak directions here so the final write-up can include
the dead ends, not just the winners.

## Layout

- `scripts/`: reproducible benchmark or timeline-generation entrypoints
- `data/`: raw benchmark outputs, timing tables, environment metadata
- `figures/`: generated plots and timeline images used in notes/blog drafts
- `notes/`: experiment plans, interpretation, and per-stage conclusions

## Measurement Rules

- Prefer locked clocks and a fixed GPU (`CUDA_VISIBLE_DEVICES=1` in our recent runs).
- Record exact shape, causal flag, block sizes, warmup/repetition counts.
- Keep raw samples, not just medians.
- Keep environment metadata with each dataset:
  - TileLang version / commit
  - TileOPs commit or worktree path
  - GPU model and visible device
  - any non-default env vars
- If a run is known to be contaminated, keep it only if clearly marked
  `invalid`; otherwise discard it.

## Naming

- Data files: `YYYYMMDD_<topic>_<variant>.json`
- Figures: `YYYYMMDD_<topic>_<variant>.png`
- Notes: `YYYYMMDD_<topic>.md`

The goal is to make every figure in the eventual blog traceable back to a raw
dataset in `data/` and a reproducible script in `scripts/`.
