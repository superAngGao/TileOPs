import pytest
import torch

from tests.ops.gla_test_utils import get_tolerances
from tests.test_base import FixtureBase
from tileops.ops import GLAFwdOp, GLAPrefillFwdOp
from workloads.gla import GLAPrefillFwdTest


class GLAPrefillFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype", [
            pytest.param(1, 64, 2, 64, 64, 64, torch.float32, marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 64, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 64, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@GLAPrefillFwdFixture
def test_gla_prefill_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    scale = dim_k ** -0.5
    test = GLAPrefillFwdTest(
        batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, scale=scale
    )
    inputs = test.gen_inputs()

    ref_op = GLAFwdOp(
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size,
        scale=scale,
        output_final_state=True,
        dtype=dtype,
    )
    ref_o, ref_final_state = ref_op(*inputs)

    prefill_op = GLAPrefillFwdOp(
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size,
        scale=scale,
        dtype=dtype,
    )
    out_o, out_state = prefill_op(*inputs)

    assert len((out_o, out_state)) == 2
    tols = get_tolerances(dtype)
    torch.testing.assert_close(out_o, ref_o, **tols)
    torch.testing.assert_close(out_state, ref_final_state.to(dtype), **tols)

