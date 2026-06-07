"""Boundary-demo kernel harness for the FP8 GQA FA3-vs-TileLang gap.

This file intentionally wraps an existing TileLang kernel instead of introducing
another production kernel class.  The goal is to provide a clean, reproducible
entry point for the boundary example:

* FA3-style FP8 grouping with TileOps' expanded scale contract.
* Candidate kernel numerically matches the serial TMA-V baseline.
* Candidate is the known direct-overlap / streaming-P / plain-wait probe used to
  expose grouped-WGMMA scoreboard degradation in full TileLang context.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tileops.kernels.attention.gqa_fwd_fp8 import (
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
)


class CorrectnessResult(NamedTuple):
    out_max_abs: float
    out_mean_abs: float
    lse_max_abs: float
    lse_mean_abs: float


@dataclass(frozen=True)
class BoundaryDemoConfig:
    batch: int = 1
    seq_len: int = 896
    heads: int = 2
    heads_kv: int = 1
    dim: int = 128
    out_dtype: torch.dtype = torch.float16
    atol_out: float = 1e-3
    atol_lse: float = 1e-5
    rtol_out: float = 1e-3
    rtol_lse: float = 1e-5

    def __post_init__(self) -> None:
        if self.seq_len % 896 != 0:
            raise ValueError(
                "Boundary demo compares a BN224 overlap candidate with the TMA-V "
                "baseline, so seq_len must be divisible by lcm(224, 128) == 896."
            )
        if self.dim != 128:
            raise ValueError("Boundary demo follows the FA3 hdim128 FP8 path.")
        if self.heads % self.heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv.")


def has_hopper_fp8() -> bool:
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9


def quantize_q_fa3_gqa_scale(
    q: torch.Tensor, heads_kv: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize Q with FA3 grouping and return TileOps' expanded scale layout."""

    batch, seq_len, heads, _dim = q.shape
    groups = heads // heads_kv
    q_grouped = q.reshape(batch, -1, heads_kv, groups, q.shape[-1])
    descale = q_grouped.abs().amax(dim=(1, 3, 4)).clamp_min(1e-6) / 448.0
    q_fp8 = torch.clamp(
        q_grouped / descale[:, None, :, None, None],
        -448.0,
        448.0,
    ).to(torch.float8_e4m3fn)
    q_head_scale = descale.repeat_interleave(groups, dim=1)
    scale_blocks = seq_len // 128
    q_scale = q_head_scale[:, :, None].expand(batch, heads, scale_blocks)
    return q_fp8.reshape_as(q).contiguous(), q_scale.float().contiguous()


def quantize_kv_fa3_scale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K/V with FA3 per-KV-head scaling and expand for TileOps kernels."""

    batch, seq_len, heads, _dim = x.shape
    descale = x.abs().amax(dim=(1, 3)).clamp_min(1e-6) / 448.0
    x_fp8 = torch.clamp(
        x / descale[:, None, :, None],
        -448.0,
        448.0,
    ).to(torch.float8_e4m3fn)
    scale_blocks = seq_len // 128
    scale = descale[:, :, None].expand(batch, heads, scale_blocks)
    return x_fp8.contiguous(), scale.float().contiguous()


def make_boundary_inputs(
    cfg: BoundaryDemoConfig, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = (
        torch.randn(
            cfg.batch,
            cfg.seq_len,
            cfg.heads,
            cfg.dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    )
    k = (
        torch.randn(
            cfg.batch,
            cfg.seq_len,
            cfg.heads_kv,
            cfg.dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    )
    v = (
        torch.randn(
            cfg.batch,
            cfg.seq_len,
            cfg.heads_kv,
            cfg.dim,
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    )
    q_fp8, q_scale = quantize_q_fa3_gqa_scale(q, cfg.heads_kv)
    k_fp8, k_scale = quantize_kv_fa3_scale(k)
    v_fp8, v_scale = quantize_kv_fa3_scale(v)
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


def build_baseline(cfg: BoundaryDemoConfig) -> GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel:
    return GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
        cfg.batch,
        cfg.heads,
        cfg.heads_kv,
        cfg.seq_len,
        cfg.dim,
        cfg.out_dtype,
    )


def build_candidate(
    cfg: BoundaryDemoConfig,
) -> GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel:
    return GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel(
        cfg.batch,
        cfg.heads,
        cfg.heads_kv,
        cfg.seq_len,
        cfg.dim,
        cfg.out_dtype,
    )


def run_correctness(cfg: BoundaryDemoConfig = BoundaryDemoConfig()) -> CorrectnessResult:
    if not has_hopper_fp8():
        raise RuntimeError("Boundary demo correctness requires Hopper CUDA + torch fp8.")

    inputs = make_boundary_inputs(cfg)
    baseline = build_baseline(cfg)
    candidate = build_candidate(cfg)

    out_base, lse_base = baseline(*inputs)
    out_candidate, lse_candidate = candidate(*inputs)
    torch.cuda.synchronize()

    if not torch.isfinite(out_candidate.float()).all():
        raise AssertionError("candidate output contains non-finite values")
    if not torch.isfinite(lse_candidate.float()).all():
        raise AssertionError("candidate LSE contains non-finite values")

    torch.testing.assert_close(
        out_candidate.float(),
        out_base.float(),
        atol=cfg.atol_out,
        rtol=cfg.rtol_out,
    )
    torch.testing.assert_close(
        lse_candidate.float(),
        lse_base.float(),
        atol=cfg.atol_lse,
        rtol=cfg.rtol_lse,
    )

    out_abs = (out_candidate.float() - out_base.float()).abs()
    lse_abs = (lse_candidate.float() - lse_base.float()).abs()
    return CorrectnessResult(
        out_max_abs=float(out_abs.max().item()),
        out_mean_abs=float(out_abs.mean().item()),
        lse_max_abs=float(lse_abs.max().item()),
        lse_mean_abs=float(lse_abs.mean().item()),
    )


def main() -> None:
    result = run_correctness()
    print(f"boundary_demo_correctness=PASS {result._asdict()}")


if __name__ == "__main__":
    main()
