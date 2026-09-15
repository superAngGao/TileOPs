import pytest
import torch

from tileops.backend import TensorSpec, registry
from tileops.ops import GatedDeltaNetFwdOp

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


def test_gated_deltanet_contract_reaches_target_builder() -> None:
    calls = []

    def build_kernel(*inputs, **params):
        calls.append((inputs, params))

        def kernel(q, k, v, g, beta, initial_state, A_log, dt_bias):
            del k, g, beta, initial_state, A_log, dt_bias
            batch, seq_len, _heads, dim_k = q.shape
            value_heads, dim_v = v.shape[2:]
            return (
                torch.empty(batch, seq_len, value_heads, dim_v, dtype=q.dtype),
                torch.empty(batch, value_heads, dim_v, dim_k, dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("GatedDeltaNetFwdOp", "gdn_test", build_kernel)

    batch, seq_len, heads, value_heads, dim_k, dim_v = 1, 7, 2, 4, 8, 6
    q = torch.randn(batch, seq_len, heads, dim_k, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, value_heads, dim_v, dtype=torch.float16)
    g = torch.randn(batch, seq_len, value_heads, dtype=torch.float16)
    beta = torch.randn_like(g)
    initial_state = torch.randn(batch, value_heads, dim_v, dim_k, dtype=torch.float32)
    A_log = torch.randn(value_heads, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, dtype=torch.float32)

    op = GatedDeltaNetFwdOp(
        scale=0.125,
        use_qk_l2norm=True,
        use_gate_activation=True,
        use_beta_sigmoid=True,
        allow_neg_eigval=True,
        target="gdn_test",
    )
    o, final_state = op(q, k, v, g, beta, initial_state, A_log, dt_bias)

    assert o.shape == (batch, seq_len, value_heads, dim_v)
    assert final_state.shape == (batch, value_heads, dim_v, dim_k)
    assert calls == [
        (
            tuple(
                TensorSpec.of(tensor)
                for tensor in (q, k, v, g, beta, initial_state, A_log, dt_bias)
            ),
            {
                "scale": 0.125,
                "use_qk_l2norm": True,
                "use_gate_activation": True,
                "use_beta_sigmoid": True,
                "allow_neg_eigval": True,
            },
        )
    ]


def test_gated_deltanet_rejects_invalid_optional_inputs() -> None:
    q = torch.empty(1, 1, 2, 8, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty(1, 1, 2, 6, dtype=torch.float16)
    g = torch.empty(1, 1, 2, dtype=torch.float16)
    beta = torch.empty_like(g)

    with pytest.raises(ValueError, match="requires A_log and dt_bias"):
        GatedDeltaNetFwdOp(use_gate_activation=True).forward(q, k, v, g, beta)

    bad_state = torch.empty(1, 2, 8, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\[B, HV, V, K\]"):
        GatedDeltaNetFwdOp().forward(q, k, v, g, beta, bad_state)
