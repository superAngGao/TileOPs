import pytest
import torch

from tests.ops.test_deltanet_fwd import _get_tolerances
from tests.test_base import FixtureBase
from tileops.ops import DeltaNetFwdOp, DeltaNetPrefillFwdOp
from workloads.deltanet import DeltaNetPrefillFwdTest


class DeltaNetPrefillFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, layout", [
            pytest.param(1, 64, 2, 64, 64, 64, torch.float32, "bthd", marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 64, torch.float16, "bthd", marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 64, torch.bfloat16, "bthd", marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 64, torch.float16, "bhtd", marks=pytest.mark.smoke),
        ]),
    ]


@DeltaNetPrefillFwdFixture
def test_deltanet_prefill_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    layout: str,
) -> None:
    torch.manual_seed(42)
    test = DeltaNetPrefillFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    q_bhtd, k_bhtd, v_bhtd, beta_bhtd = test.gen_inputs()

    fwd_op = DeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    ref_o_bhtd, ref_states, *_ = fwd_op(q_bhtd, k_bhtd, v_bhtd, beta_bhtd)
    ref_final_state = ref_states[:, :, -1].to(dtype)

    prefill_op = DeltaNetPrefillFwdOp(
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        layout=layout,
    )
    if layout == "bthd":
        q = q_bhtd.permute(0, 2, 1, 3).contiguous()
        k = k_bhtd.permute(0, 2, 1, 3).contiguous()
        v = v_bhtd.permute(0, 2, 1, 3).contiguous()
        beta = beta_bhtd.permute(0, 2, 1).contiguous()
        out_o, out_state = prefill_op(q, k, v, beta)
        out_o_bhtd = out_o.permute(0, 2, 1, 3).contiguous()
    else:
        out_o_bhtd, out_state = prefill_op(q_bhtd, k_bhtd, v_bhtd, beta_bhtd)

    assert len((out_o_bhtd, out_state)) == 2
    torch.testing.assert_close(out_o_bhtd, ref_o_bhtd, **_get_tolerances(dtype))
    torch.testing.assert_close(out_state, ref_final_state, **_get_tolerances(dtype))

