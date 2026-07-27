"""Baseline timing for the current Gated DeltaNet backward pipeline.

This script intentionally benchmarks the current BHSD backward implementation.
When ``--layout bthd`` is used, inputs are generated in BTHD form and converted
to BHSD before calling the legacy kernel. That measures wrapper/layout overhead;
it is not a BTHD-native optimized path.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_base import bench_kernel
from tileops.kernels.gated_deltanet.compute_w_u_bwd import compute_w_u_bwd_full_tl
from tileops.kernels.gated_deltanet.fused_prepare_compute_w_u import (
    fused_prepare_compute_w_u_tl,
)
from tileops.kernels.gated_deltanet.gated_deltanet_bwd import (
    GatedDeltaNetBwdKernel,
    _bwd_parallel_tl,
    _compute_dw_correction_tl,
    _dh_carry_after_scan_tl,
    _dh_correction_from_carry_tl,
    _dh_recurrence_bwd_tl,
    _dh_segment_boundary_scan_tl,
    _dh_segment_local_carry_tl,
    _dh_segment_summary_tl,
    _merge_bwd_outputs_tl,
    _reduce_dh_recurrence_partials_tl,
)
from tileops.kernels.gated_deltanet.gated_deltanet_fwd import _chunk_local_cumsum
from tileops.ops import GatedDeltaNetFwdOp

_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@dataclass
class BenchConfig:
    batch: int
    heads: int
    seq_len: int
    dim_k: int
    dim_v: int
    chunk_size: int
    dtype: str
    layout: str
    warmup: int
    repeat: int
    trials: int
    stage_breakdown: bool
    num_stages: int
    threads: int
    parallel_threads: int
    recurrence_threads: int
    recurrence_block_v: int
    recurrence_split_carry: int


def _cuda_env() -> dict[str, Any]:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "device": torch.cuda.get_device_name(device),
        "sm_count": props.multi_processor_count,
        "l2_cache_size": props.L2_cache_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _make_inputs(cfg: BenchConfig) -> tuple[torch.Tensor, ...]:
    dtype = _DTYPES[cfg.dtype]
    B, H, S, DK, DV = cfg.batch, cfg.heads, cfg.seq_len, cfg.dim_k, cfg.dim_v
    torch.manual_seed(42)
    if cfg.layout == "bthd":
        q = torch.randn(B, S, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, S, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, S, H, DV, device="cuda", dtype=dtype) * 0.1
        g = -torch.rand(B, S, H, device="cuda", dtype=dtype)
        beta = torch.rand(B, S, H, device="cuda", dtype=dtype) * 0.5
        do = torch.randn(B, S, H, DV, device="cuda", dtype=dtype) * 0.1
        return q, k, v, g, beta, do

    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    return q, k, v, g, beta, do


def _to_bhsd(
    cfg: BenchConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if cfg.layout != "bthd":
        return q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), do.contiguous()
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
        do.permute(0, 2, 1, 3).contiguous(),
    )


def _time_layout_convert(cfg: BenchConfig, inputs: tuple[torch.Tensor, ...]) -> float:
    if cfg.layout != "bthd":
        return 0.0

    def convert():
        return _to_bhsd(cfg, *inputs)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(cfg.warmup):
        convert()
    torch.cuda.synchronize()
    samples = []
    for _ in range(cfg.trials):
        start.record()
        for _ in range(cfg.repeat):
            convert()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / cfg.repeat)
    return float(torch.tensor(samples).median().item())


def _bench_full_bwd(cfg: BenchConfig, inputs_bhsd: tuple[torch.Tensor, ...]) -> float:
    q, k, v, g, beta, do = inputs_bhsd
    dtype = _DTYPES[cfg.dtype]
    fwd = GatedDeltaNetFwdOp(chunk_size=cfg.chunk_size)
    _o, S, _Aw, _Au = fwd.forward(q, k, v, g, beta)
    op = GatedDeltaNetBwdKernel(
        cfg.batch,
        cfg.heads,
        cfg.seq_len,
        cfg.chunk_size,
        cfg.dim_k,
        cfg.dim_v,
        GatedDeltaNetBwdKernel.dtype_to_str(dtype),
        config=_kernel_config(cfg),
    )
    return bench_kernel(
        op.forward,
        (do, q, k, v, g, beta, S),
        n_warmup=cfg.warmup,
        n_repeat=cfg.repeat,
        n_trials=cfg.trials,
    )


def _bench_stage_breakdown(cfg: BenchConfig, inputs_bhsd: tuple[torch.Tensor, ...]) -> dict[str, float]:
    q, k, v, g, beta, do = inputs_bhsd
    B, H, S, DK, DV, BC = (
        cfg.batch,
        cfg.heads,
        cfg.seq_len,
        cfg.dim_k,
        cfg.dim_v,
        cfg.chunk_size,
    )
    dtype_str = "float16" if _DTYPES[cfg.dtype] == torch.float16 else (
        "bfloat16" if _DTYPES[cfg.dtype] == torch.bfloat16 else "float32"
    )

    fwd = GatedDeltaNetFwdOp(chunk_size=BC)
    _o, S_buf, _Aw_ref, _Au_ref = fwd.forward(q, k, v, g, beta)
    g_cum = _chunk_local_cumsum(g.float(), BC).to(g.dtype)

    fused_fn = fused_prepare_compute_w_u_tl(
        B, H, S, BC, DK, DV, dtype_str, write_duplicate_A=False,
    )(cfg.num_stages, cfg.threads)
    bwd_parallel_fn = _bwd_parallel_tl(B, H, S, BC, DK, DV, dtype_str)(
        cfg.parallel_threads
    )
    reduce_dh_partials_fn = _reduce_dh_recurrence_partials_tl(
        B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
    )(cfg.recurrence_threads)
    merge_bwd_outputs_fn = _merge_bwd_outputs_tl(
        B, H, S, BC, DK, dtype_str,
    )(cfg.threads)
    wu_bwd_fn = compute_w_u_bwd_full_tl(B, H, S, BC, DK, DV, dtype_str)(
        cfg.num_stages, cfg.threads
    )
    dw_correction_fn = _compute_dw_correction_tl(
        B, H, S, BC, DK, DV, dtype_str,
    )(cfg.threads)

    Aw, Au, w, u = fused_fn(k, v, g_cum, beta)
    dq, dk_partial, dg_partial, dw, du_partial, v_new, dh_local = bwd_parallel_fn(
        do, q, k, g_cum, w, u, S_buf
    )
    if cfg.recurrence_split_carry == 0:
        dh_fn = _dh_recurrence_bwd_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.num_stages, cfg.recurrence_threads)
        dk_corr_partial, du_corr, dg_corr_partial = dh_fn(g_cum, k, v_new, S_buf, dh_local)
        dh_stage_inputs = (g_cum, k, v_new, S_buf, dh_local)
    elif cfg.recurrence_split_carry == 1:
        dh_carry_after_scan_fn = _dh_carry_after_scan_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_correction_from_carry_fn = _dh_correction_from_carry_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.recurrence_threads)
        dh_carry_after = dh_carry_after_scan_fn(g_cum, dh_local)
        dk_corr_partial, du_corr, dg_corr_partial = dh_correction_from_carry_fn(
            g_cum, k, v_new, S_buf, dh_carry_after
        )
        dh_stage_inputs = (g_cum, dh_local, dh_carry_after)
    else:
        segment_chunks = _segment_chunks_for(S, BC)
        dh_segment_summary_fn = _dh_segment_summary_tl(
            B, H, S, BC, DK, DV, dtype_str,
            block_v=cfg.recurrence_block_v,
            segment_chunks=segment_chunks,
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_segment_boundary_scan_fn = _dh_segment_boundary_scan_tl(
            B, H, S, BC, DK, DV, dtype_str,
            block_v=cfg.recurrence_block_v,
            segment_chunks=segment_chunks,
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_segment_local_carry_fn = _dh_segment_local_carry_tl(
            B, H, S, BC, DK, DV, dtype_str,
            block_v=cfg.recurrence_block_v,
            segment_chunks=segment_chunks,
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_correction_from_carry_fn = _dh_correction_from_carry_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.recurrence_threads)
        segment_alpha, segment_local = dh_segment_summary_fn(g_cum, dh_local)
        segment_carry_after = dh_segment_boundary_scan_fn(segment_alpha, segment_local)
        dh_carry_after = dh_segment_local_carry_fn(g_cum, dh_local, segment_carry_after)
        dk_corr_partial, du_corr, dg_corr_partial = dh_correction_from_carry_fn(
            g_cum, k, v_new, S_buf, dh_carry_after
        )
        dh_stage_inputs = (
            g_cum,
            dh_local,
            segment_alpha,
            segment_local,
            segment_carry_after,
            dh_carry_after,
        )
    dk_corr, dg_corr = reduce_dh_partials_fn(dk_corr_partial, dg_corr_partial)
    dw_corr = dw_correction_fn(du_corr, S_buf, g_cum)
    dk_prepare, _dv, _dbeta, dg_prepare = wu_bwd_fn(
        dw, dw_corr, du_partial, du_corr, Aw, k, v, g_cum, beta,
    )
    result = {
        "fused_prepare_compute_w_u_ms": bench_kernel(
            fused_fn, (k, v, g_cum, beta), cfg.warmup, cfg.repeat, cfg.trials
        ),
        "bwd_parallel_ms": bench_kernel(
            bwd_parallel_fn, (do, q, k, g_cum, w, u, S_buf), cfg.warmup, cfg.repeat, cfg.trials
        ),
        "reduce_dh_recurrence_partials_ms": bench_kernel(
            reduce_dh_partials_fn,
            (dk_corr_partial, dg_corr_partial),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        ),
        "compute_dw_correction_ms": bench_kernel(
            dw_correction_fn,
            (du_corr, S_buf, g_cum),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        ),
        "compute_w_u_bwd_full_ms": bench_kernel(
            wu_bwd_fn,
            (dw, dw_corr, du_partial, du_corr, Aw, k, v, g_cum, beta),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        ),
        "merge_bwd_outputs_ms": bench_kernel(
            merge_bwd_outputs_fn,
            (dk_partial, dk_corr, dk_prepare, dg_partial, dg_corr, dg_prepare),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        ),
    }
    if cfg.recurrence_split_carry == 0:
        dh_fn = _dh_recurrence_bwd_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.num_stages, cfg.recurrence_threads)
        result["dh_recurrence_bwd_ms"] = bench_kernel(
            dh_fn, dh_stage_inputs, cfg.warmup, cfg.repeat, cfg.trials
        )
    elif cfg.recurrence_split_carry == 1:
        dh_carry_after_scan_fn = _dh_carry_after_scan_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_correction_from_carry_fn = _dh_correction_from_carry_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.recurrence_threads)
        result["dh_carry_after_scan_ms"] = bench_kernel(
            dh_carry_after_scan_fn,
            dh_stage_inputs[:2],
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        )
        result["dh_correction_from_carry_ms"] = bench_kernel(
            dh_correction_from_carry_fn,
            (g_cum, k, v_new, S_buf, dh_stage_inputs[2]),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        )
    else:
        segment_chunks = _segment_chunks_for(S, BC)
        dh_segment_summary_fn = _dh_segment_summary_tl(
            B, H, S, BC, DK, DV, dtype_str,
            block_v=cfg.recurrence_block_v,
            segment_chunks=segment_chunks,
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_segment_boundary_scan_fn = _dh_segment_boundary_scan_tl(
            B, H, S, BC, DK, DV, dtype_str,
            block_v=cfg.recurrence_block_v,
            segment_chunks=segment_chunks,
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_segment_local_carry_fn = _dh_segment_local_carry_tl(
            B, H, S, BC, DK, DV, dtype_str,
            block_v=cfg.recurrence_block_v,
            segment_chunks=segment_chunks,
        )(cfg.num_stages, cfg.recurrence_threads)
        dh_correction_from_carry_fn = _dh_correction_from_carry_tl(
            B, H, S, BC, DK, DV, dtype_str, block_v=cfg.recurrence_block_v
        )(cfg.recurrence_threads)
        result["dh_segment_summary_ms"] = bench_kernel(
            dh_segment_summary_fn,
            dh_stage_inputs[:2],
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        )
        result["dh_segment_boundary_scan_ms"] = bench_kernel(
            dh_segment_boundary_scan_fn,
            (dh_stage_inputs[2], dh_stage_inputs[3]),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        )
        result["dh_segment_local_carry_ms"] = bench_kernel(
            dh_segment_local_carry_fn,
            (g_cum, dh_local, dh_stage_inputs[4]),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        )
        result["dh_correction_from_carry_ms"] = bench_kernel(
            dh_correction_from_carry_fn,
            (g_cum, k, v_new, S_buf, dh_stage_inputs[5]),
            cfg.warmup,
            cfg.repeat,
            cfg.trials,
        )
    return result


def _default_recurrence_split_carry(dim_v: int, chunk_size: int) -> int:
    return 2 if chunk_size >= 64 and dim_v > 64 and dim_v % 64 == 0 else 0


def _default_recurrence_block_v(dim_v: int, chunk_size: int) -> int:
    if _default_recurrence_split_carry(dim_v, chunk_size) != 0:
        return 64
    return 32 if dim_v > 64 and dim_v % 32 == 0 else 0


def _kernel_config(cfg: BenchConfig) -> dict[str, int]:
    return {
        "num_stages": cfg.num_stages,
        "threads": cfg.threads,
        "parallel_threads": cfg.parallel_threads,
        "recurrence_threads": cfg.recurrence_threads,
        "recurrence_block_v": cfg.recurrence_block_v,
        "recurrence_split_carry": cfg.recurrence_split_carry,
    }


def _segment_chunks_for(seq_len: int, chunk_size: int) -> int:
    num_chunks = seq_len // chunk_size
    if num_chunks % 8 == 0:
        return 8
    if num_chunks % 4 == 0:
        return 4
    if num_chunks % 2 == 0:
        return 2
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="fp16")
    parser.add_argument("--layout", choices=("bhsd", "bhtd", "bthd"), default="bthd")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--stage-breakdown", action="store_true")
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--parallel-threads", type=int, default=0)
    parser.add_argument("--recurrence-threads", type=int, default=0)
    parser.add_argument("--recurrence-block-v", type=int, default=-1)
    parser.add_argument("--recurrence-split-carry", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--recurrence-split-carry-mode",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Explicit split-carry mode. Overrides the boolean split-carry flag.",
    )
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = "bhsd" if args.layout == "bhtd" else args.layout
    default_split_carry = _default_recurrence_split_carry(args.dim_v, args.chunk_size)
    default_threads = 128 if default_split_carry else (256 if args.chunk_size >= 64 else 128)
    threads = args.threads or default_threads
    default_parallel_threads = 256 if default_split_carry else threads
    parallel_threads = args.parallel_threads or default_parallel_threads
    recurrence_threads = args.recurrence_threads or (128 if default_split_carry else threads)
    recurrence_block_v = (
        _default_recurrence_block_v(args.dim_v, args.chunk_size)
        if args.recurrence_block_v < 0
        else args.recurrence_block_v
    )
    if args.recurrence_split_carry_mode is not None:
        recurrence_split_carry = args.recurrence_split_carry_mode
    else:
        recurrence_split_carry = (
            default_split_carry
            if args.recurrence_split_carry is None
            else (1 if args.recurrence_split_carry else 0)
        )
    cfg = BenchConfig(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        dim_k=args.dim_k,
        dim_v=args.dim_v,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        layout=layout,
        warmup=args.warmup,
        repeat=args.repeat,
        trials=args.trials,
        stage_breakdown=args.stage_breakdown,
        num_stages=args.num_stages,
        threads=threads,
        parallel_threads=parallel_threads,
        recurrence_threads=recurrence_threads,
        recurrence_block_v=recurrence_block_v,
        recurrence_split_carry=recurrence_split_carry,
    )
    if cfg.seq_len % cfg.chunk_size != 0:
        raise ValueError("seq_len must be divisible by chunk_size")

    inputs = _make_inputs(cfg)
    layout_ms = _time_layout_convert(cfg, inputs)
    inputs_bhsd = _to_bhsd(cfg, *inputs)
    full_ms = _bench_full_bwd(cfg, inputs_bhsd)
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(cfg),
        "env": _cuda_env(),
        "legacy_bhsd_full_bwd_ms": full_ms,
        "bthd_to_bhsd_layout_convert_ms": layout_ms,
        "note": (
            "BTHD mode measures wrapper conversion into the current BHSD-only "
            "backward path; it is not a native optimized BTHD kernel."
        ),
    }
    if cfg.stage_breakdown:
        result["stage_breakdown_ms"] = _bench_stage_breakdown(cfg, inputs_bhsd)
    line = json.dumps(result, sort_keys=True)
    print(line)
    if args.output:
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
