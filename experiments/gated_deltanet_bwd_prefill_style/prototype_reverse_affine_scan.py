"""Reference reverse affine scan for Gated DeltaNet backward dh propagation.

Current ``dh_recurrence_bwd`` carries a matrix-shaped adjoint state backward
over chunks. In the existing implementation, the cross-chunk recurrence has the
form

    X[i] = G[i] + alpha[i] * X[i + 1]

where ``G[i]`` is the chunk-local gradient contribution and ``alpha[i]`` is the
chunk boundary decay factor. This script checks that the recurrence can be
summarized per segment as

    X[left] = A_segment * X[right] + B_segment

and composed with an associative affine combine. It is a math/contract
prototype for the future reverse-scan kernel, not a performance benchmark.
"""

from __future__ import annotations

import argparse

import torch


def sequential_reverse(alpha: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    num_chunks = alpha.numel()
    out = torch.empty_like(local)
    carry = torch.zeros_like(local[0])
    for idx in range(num_chunks - 1, -1, -1):
        carry = local[idx] + alpha[idx] * carry
        out[idx] = carry
    return out


def summarize_segment(
    alpha: torch.Tensor,
    local: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.ones((), dtype=local.dtype, device=local.device)
    B = torch.zeros_like(local[0])
    for idx in range(end - 1, start - 1, -1):
        B = local[idx] + alpha[idx] * B
        A = alpha[idx] * A
    return A, B


def compose(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    A_left, B_left = left
    A_right, B_right = right
    return A_left * A_right, B_left + A_left * B_right


def segment_reverse_scan(
    alpha: torch.Tensor,
    local: torch.Tensor,
    segment_chunks: int,
) -> torch.Tensor:
    num_chunks = alpha.numel()
    segments = []
    for start in range(0, num_chunks, segment_chunks):
        end = min(num_chunks, start + segment_chunks)
        segments.append((start, end, summarize_segment(alpha, local, start, end)))

    suffix = []
    carry = (
        torch.ones((), dtype=local.dtype, device=local.device),
        torch.zeros_like(local[0]),
    )
    for _start, _end, summary in reversed(segments):
        suffix.append(carry)
        carry = compose(summary, carry)
    suffix.reverse()

    out = torch.empty_like(local)
    for (start, end, _summary), (_A_after, B_after) in zip(segments, suffix, strict=True):
        carry = B_after
        for idx in range(end - 1, start - 1, -1):
            carry = local[idx] + alpha[idx] * carry
            out[idx] = carry
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--dim-k", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--segment-chunks", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.float64
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    # Keep alpha in a realistic decay-like range but away from denormal corners.
    alpha = torch.rand(args.chunks, generator=generator, dtype=dtype) * 0.4 + 0.6
    local = torch.randn(args.chunks, args.dim_k, args.dim_v, generator=generator, dtype=dtype) * 0.1

    ref = sequential_reverse(alpha, local)
    segmented = segment_reverse_scan(alpha, local, args.segment_chunks)
    max_abs = (ref - segmented).abs().max().item()
    torch.testing.assert_close(ref, segmented, atol=1e-5 if dtype == torch.float32 else 1e-10, rtol=1e-5)
    print(
        {
            "chunks": args.chunks,
            "dim_k": args.dim_k,
            "dim_v": args.dim_v,
            "segment_chunks": args.segment_chunks,
            "dtype": args.dtype,
            "max_abs": max_abs,
            "status": "pass",
        }
    )


if __name__ == "__main__":
    main()
