"""Validate reverse affine scan on real Gated DeltaNet backward intermediates.

This script runs the existing forward + ``bwd_parallel`` kernels, then replaces
the sequential ``dh_recurrence_bwd`` carry with a Torch reference segmented
reverse affine scan:

    carry_after[i] = dH entering chunk i from successor chunks
    carry_before[i] = dh_local[i] + alpha[i] * carry_after[i]

The correction formulas are then recomputed from ``carry_after`` and compared
against the current TileLang ``dh_recurrence_bwd`` + partial reduction outputs.
It is a correctness gate for the future reverse-scan kernel boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.gated_deltanet_bwd_prefill_style.bench_current_bwd import (  # noqa: E402
    BenchConfig,
    _DTYPES,
    _default_recurrence_block_v,
    _make_inputs,
    _to_bhsd,
)
from tileops.kernels.gated_deltanet.fused_prepare_compute_w_u import (  # noqa: E402
    fused_prepare_compute_w_u_tl,
)
from tileops.kernels.gated_deltanet.gated_deltanet_bwd import (  # noqa: E402
    _bwd_parallel_tl,
    _dh_recurrence_bwd_tl,
    _reduce_dh_recurrence_partials_tl,
)
from tileops.kernels.gated_deltanet.gated_deltanet_fwd import (  # noqa: E402
    _LOG2E,
    _chunk_local_cumsum,
)
from tileops.ops import GatedDeltaNetFwdOp  # noqa: E402


def _dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"unsupported dtype: {dtype}")


def _reverse_segment_carry_after(
    alpha: torch.Tensor,
    local: torch.Tensor,
    segment_chunks: int,
) -> torch.Tensor:
    """Return carry entering each chunk from its successor side.

    ``alpha`` has shape [B,H,N]. ``local`` has shape [B,H,N,DK,DV].
    """
    B, H, N = alpha.shape
    carry_after = torch.empty_like(local)
    segments: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []

    for start in range(0, N, segment_chunks):
        end = min(N, start + segment_chunks)
        A = torch.ones(B, H, device=local.device, dtype=torch.float32)
        Bsum = torch.zeros(B, H, local.shape[-2], local.shape[-1], device=local.device, dtype=torch.float32)
        for idx in range(end - 1, start - 1, -1):
            Bsum = local[:, :, idx].float() + alpha[:, :, idx].float()[..., None, None] * Bsum
            A = alpha[:, :, idx].float() * A
        segments.append((start, end, A, Bsum))

    suffix_A = torch.ones(B, H, device=local.device, dtype=torch.float32)
    suffix_B = torch.zeros(B, H, local.shape[-2], local.shape[-1], device=local.device, dtype=torch.float32)
    suffix: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _start, _end, A, Bsum in reversed(segments):
        suffix.append((suffix_A, suffix_B))
        suffix_B = Bsum + A[..., None, None] * suffix_B
        suffix_A = A * suffix_A
    suffix.reverse()

    for (start, end, _A, _Bsum), (_suffix_A, suffix_B_after) in zip(segments, suffix, strict=True):
        carry = suffix_B_after
        for idx in range(end - 1, start - 1, -1):
            carry_after[:, :, idx] = carry.to(local.dtype)
            carry = local[:, :, idx].float() + alpha[:, :, idx].float()[..., None, None] * carry
    return carry_after


def _torch_corrections(
    g: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    S: torch.Tensor,
    dh_local: torch.Tensor,
    chunk_size: int,
    segment_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, SL, DK = k.shape
    DV = v_new.shape[-1]
    num_chunks = SL // chunk_size
    g_chunks = g.reshape(B, H, num_chunks, chunk_size)
    k_chunks = k.reshape(B, H, num_chunks, chunk_size, DK)
    v_chunks = v_new.reshape(B, H, num_chunks, chunk_size, DV)
    alpha = torch.exp2(g_chunks[:, :, :, -1].float() * _LOG2E)
    carry_after = _reverse_segment_carry_after(alpha, dh_local, segment_chunks)

    dk_corr = torch.empty(B, H, SL, DK, device=k.device, dtype=k.dtype)
    du_corr = torch.empty(B, H, SL, DV, device=v_new.device, dtype=v_new.dtype)
    dg_corr = torch.empty(B, H, SL, device=g.device, dtype=g.dtype)

    for cid in range(num_chunks):
        g_c = g_chunks[:, :, cid].float()
        g_last = g_c[:, :, -1]
        dh_buf = carry_after[:, :, cid].float()
        k_c = k_chunks[:, :, cid].float()
        v_c = v_chunks[:, :, cid].float()
        scale = torch.exp2((g_last[..., None] - g_c) * _LOG2E)
        k_scaled = k_c * scale[..., None]

        du = torch.einsum("bhnk,bhkv->bhnv", k_scaled, dh_buf)
        dP = torch.einsum("bhnv,bhkv->bhnk", v_c, dh_buf)
        dk = dP * scale[..., None]
        d_g_pos = (dP * k_scaled).sum(dim=-1)
        dg = -d_g_pos

        h_c = S[:, :, cid].float()
        d_g_last_1 = (dh_buf * h_c).sum(dim=(-1, -2)) * alpha[:, :, cid]
        d_g_last_2 = d_g_pos.sum(dim=-1)
        dg[:, :, -1] = dg[:, :, -1] + d_g_last_1 + d_g_last_2

        start = cid * chunk_size
        end = start + chunk_size
        dk_corr[:, :, start:end] = dk.to(k.dtype)
        du_corr[:, :, start:end] = du.to(v_new.dtype)
        dg_corr[:, :, start:end] = dg.to(g.dtype)

    return dk_corr, du_corr, dg_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--segment-chunks", type=int, default=2)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--layout", default="bhsd")
    parser.add_argument("--recurrence-block-v", type=int, default=-1)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = _DTYPES[args.dtype]
    block_v = (
        _default_recurrence_block_v(args.dim_v, args.chunk_size)
        if args.recurrence_block_v < 0
        else args.recurrence_block_v
    )
    cfg = BenchConfig(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        dim_k=args.dim_k,
        dim_v=args.dim_v,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        layout=args.layout,
        warmup=1,
        repeat=1,
        trials=1,
        stage_breakdown=False,
        num_stages=2,
        threads=256,
        parallel_threads=256,
        recurrence_threads=256,
        recurrence_block_v=block_v,
        recurrence_split_carry=0,
    )
    inputs = _make_inputs(cfg)
    q, k, v, g, beta, do = _to_bhsd(cfg, *inputs)
    B, H, S_LEN, DK, DV, BC = (
        cfg.batch,
        cfg.heads,
        cfg.seq_len,
        cfg.dim_k,
        cfg.dim_v,
        cfg.chunk_size,
    )
    dtype_str = _dtype_str(dtype)

    fwd = GatedDeltaNetFwdOp(B, H, S_LEN, DK, DV, BC, dtype)
    _o, S_buf, _Aw_ref, _Au_ref = fwd.forward(q, k, v, g, beta)
    g_cum = _chunk_local_cumsum(g.float(), BC).to(g.dtype)

    fused_fn = fused_prepare_compute_w_u_tl(B, H, S_LEN, BC, DK, DV, dtype_str)(2, 256)
    bwd_parallel_fn = _bwd_parallel_tl(B, H, S_LEN, BC, DK, DV, dtype_str)(256)
    dh_fn = _dh_recurrence_bwd_tl(B, H, S_LEN, BC, DK, DV, dtype_str, block_v=block_v)(2, 256)
    reduce_fn = _reduce_dh_recurrence_partials_tl(B, H, S_LEN, BC, DK, DV, dtype_str, block_v=block_v)(256)

    Aw, Au, w, u = fused_fn(k, v, g_cum, beta)
    _dq, _dk_partial, _dg_partial, _dw, _du_partial, v_new, dh_local = bwd_parallel_fn(
        do, q, k, g_cum, w, u, S_buf
    )
    dk_part, du_ref, dg_part = dh_fn(g_cum, k, v_new, S_buf, dh_local)
    dk_ref, dg_ref = reduce_fn(dk_part, dg_part)
    dk_scan, du_scan, dg_scan = _torch_corrections(
        g_cum, k, v_new, S_buf, dh_local, BC, args.segment_chunks
    )

    metrics: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shape": {
            "batch": B,
            "heads": H,
            "seq_len": S_LEN,
            "dim_k": DK,
            "dim_v": DV,
            "chunk_size": BC,
            "segment_chunks": args.segment_chunks,
            "dtype": args.dtype,
            "block_v": block_v,
        },
        "status": "pass",
    }
    for name, ref, scan in (
        ("dk_corr", dk_ref, dk_scan),
        ("du_corr", du_ref, du_scan),
        ("dg_corr", dg_ref, dg_scan),
    ):
        diff = (ref.float() - scan.float()).abs()
        ok = torch.allclose(ref.float(), scan.float(), atol=args.atol, rtol=args.rtol)
        metrics[name] = {
            "allclose": bool(ok),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }
        if not ok:
            metrics["status"] = "fail"
    print(json.dumps(metrics, sort_keys=True))
    if metrics["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
