from typing import Optional

import pytest
import torch

from tileops.kernels.attention import (
    GQAPrefillFwdWsPersistentCausalKernel,
)
from tileops.ops import GroupedQueryAttentionDenseFwdOp
from tileops.perf.formulas import gqa_dense_fwd_roofline
from tileops.utils import is_h200


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    groups = q.shape[2] // k.shape[2]
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scores = q_bhsd @ k_bhsd.transpose(-2, -1)
    scores *= q.shape[-1] ** -0.5 if sm_scale is None else sm_scale
    if softcap is not None and softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    seq_len_q, seq_len_kv = q.shape[1], k.shape[1]
    q_pos = torch.arange(seq_len_q, device=q.device)[:, None] + seq_len_kv - seq_len_q
    k_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
    scores.masked_fill_(~(k_pos <= q_pos).view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v_bhsd).transpose(1, 2).to(q.dtype).contiguous()


def _inputs(
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch, seq_len_q, heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, 128, device="cuda", dtype=dtype)
    return q, k, torch.randn_like(k)


@pytest.mark.smoke
def test_gqa_dense_roofline_counts_rectangular_causal_attention() -> None:
    flops, nbytes = gqa_dense_fwd_roofline(
        q_shape=(1, 2, 4, 8),
        kv_shape=(1, 4, 2, 8),
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        is_causal=True,
    )
    assert flops == 4 * 1 * 4 * 7 * 8
    assert nbytes == (64 + 2 * 64 + 64) * 2


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, sm_scale, softcap, atol, rtol",
    [
        pytest.param(torch.float16, None, None, 5e-3, 1e-5, id="fp16"),
        pytest.param(torch.bfloat16, 0.125, None, 8e-2, 1e-2, id="bf16-scale"),
        pytest.param(torch.float16, None, 2.0, 5e-3, 1e-5, id="fp16-softcap"),
    ],
)
def test_gqa_dense_h200_main_kernel_matches_reference(
    dtype: torch.dtype,
    sm_scale: Optional[float],
    softcap: Optional[float],
    atol: float,
    rtol: float,
) -> None:
    if not is_h200():
        pytest.skip("Dense warp-specialized prefill is currently supported on H200")
    q, k, v = _inputs(1, 96, 150, 8, 2, dtype)
    op = GroupedQueryAttentionDenseFwdOp(sm_scale=sm_scale, softcap=softcap)
    output = op(q, k, v)
    torch.testing.assert_close(
        output,
        _reference(q, k, v, sm_scale=sm_scale, softcap=softcap),
        atol=atol,
        rtol=rtol,
    )
    assert isinstance(next(iter(op.iter_kernels())), GQAPrefillFwdWsPersistentCausalKernel)


@pytest.mark.smoke
def test_gqa_dense_h200_main_kernel_covers_square_attention() -> None:
    if not is_h200():
        pytest.skip("Dense warp-specialized prefill is currently supported on H200")
    q, k, v = _inputs(4, 512, 512, 32, 8, torch.float16)
    op = GroupedQueryAttentionDenseFwdOp()
    output = op(q, k, v)
    torch.testing.assert_close(output, _reference(q, k, v), atol=5e-3, rtol=1e-5)
    assert isinstance(next(iter(op.iter_kernels())), GQAPrefillFwdWsPersistentCausalKernel)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_kwargs, dim",
    [
        pytest.param({"is_causal": False}, 128, id="noncausal"),
        pytest.param({}, 64, id="head-dim-64"),
        pytest.param({"window_size_left": 32}, 128, id="sliding-window"),
    ],
)
def test_gqa_dense_rejects_unimplemented_regions(op_kwargs: dict, dim: int) -> None:
    if not is_h200():
        pytest.skip("H200 support-region rejection is device-specific")
    q = torch.randn(1, 64, 8, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 96, 2, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="no implementation serves this call"):
        GroupedQueryAttentionDenseFwdOp(**op_kwargs)(q, k, v)
