"""Report per-gradient accuracy for the Gated DeltaNet backward kernel."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tileops.ops import GatedDeltaNetBwdOp, GatedDeltaNetFwdOp


def _reference_forward(q, k, v, g_raw, beta, chunk_size):
    batch, heads, seq_len, dim_k = q.shape
    dim_v = v.shape[-1]
    num_chunks = seq_len // chunk_size
    g_cum = g_raw.float().reshape(
        batch, heads, num_chunks, chunk_size,
    ).cumsum(-1).reshape(batch, heads, seq_len)
    state = q.new_zeros(batch, heads, dim_k, dim_v)
    outputs = []
    eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(
        chunk_size, chunk_size, device=q.device, dtype=torch.float32,
    ))
    for chunk in range(num_chunks):
        sl = slice(chunk * chunk_size, (chunk + 1) * chunk_size)
        q_chunk = q[:, :, sl, :].float()
        k_chunk = k[:, :, sl, :].float()
        v_chunk = v[:, :, sl, :].float()
        g_chunk = g_cum[:, :, sl]
        beta_chunk = beta[:, :, sl].float()
        gram = torch.einsum("bhik,bhjk->bhij", k_chunk, k_chunk)
        gamma = torch.exp(g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(-2))
        matrix = beta_chunk.unsqueeze(-1) * gamma * gram
        inverse = torch.linalg.inv(eye + torch.tril(matrix, diagonal=-1))
        w = inverse @ (k_chunk * beta_chunk.unsqueeze(-1))
        u = inverse @ (v_chunk * beta_chunk.unsqueeze(-1))
        g_last = g_chunk[:, :, -1:]
        v_new = u - (w * torch.exp(g_chunk + g_last).unsqueeze(-1)) @ state
        output_state = (q_chunk @ state) * torch.exp(g_chunk).unsqueeze(-1)
        attention = (q_chunk @ k_chunk.transpose(-2, -1)) * gamma * mask
        outputs.append(output_state + attention @ v_new)
        k_scaled = k_chunk * torch.exp(g_last - g_chunk).unsqueeze(-1)
        state = (
            state * torch.exp(g_last).unsqueeze(-1)
            + k_scaled.transpose(-2, -1) @ v_new
        )
    return torch.cat(outputs, dim=2)


def _reference_backward(do, q, k, v, g, beta, chunk_size):
    inputs = [
        tensor.float().detach().requires_grad_(True)
        for tensor in (q, k, v, g, beta)
    ]
    output = _reference_forward(*inputs, chunk_size)
    return torch.autograd.grad((output * do.float()).sum(), inputs)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    expected_norm = torch.linalg.vector_norm(expected_f)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "l2_relative": float(
            (torch.linalg.vector_norm(diff) / expected_norm.clamp_min(1e-12)).item()
        ),
        "cosine": float(torch.nn.functional.cosine_similarity(
            actual_f.flatten(), expected_f.flatten(), dim=0,
        ).item()),
        "reference_max_abs": float(expected_f.abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shape = (1, args.heads, args.seq_len, args.dim)
    dtype = torch.float16
    q = torch.randn(*shape, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(*shape, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(*shape, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(
        1, args.heads, args.seq_len, device="cuda", dtype=dtype,
    )
    beta = torch.rand(
        1, args.heads, args.seq_len, device="cuda", dtype=dtype,
    ) * 0.5
    do = torch.randn(*shape, device="cuda", dtype=dtype) * 0.1

    fwd = GatedDeltaNetFwdOp(chunk_size=args.chunk_size)
    _output, states, _aw, _au = fwd.forward(q, k, v, g, beta)
    actual = GatedDeltaNetBwdOp(chunk_size=args.chunk_size).forward(
        do, q, k, v, g, beta, states,
    )
    expected = _reference_backward(do, q, k, v, g, beta, args.chunk_size)

    result = {
        "shape": {
            "batch": 1,
            "heads": args.heads,
            "seq_len": args.seq_len,
            "dim_k": args.dim,
            "dim_v": args.dim,
            "chunk_size": args.chunk_size,
            "dtype": "float16",
            "seed": args.seed,
        },
        "metrics": {
            name: _metrics(actual_grad, expected_grad)
            for name, actual_grad, expected_grad in zip(
                ("dq", "dk", "dv", "dg", "dbeta"),
                actual,
                expected,
                strict=True,
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
