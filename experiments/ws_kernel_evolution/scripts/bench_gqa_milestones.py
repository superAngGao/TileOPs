#!/usr/bin/env python3
"""Benchmark milestone kernels on production-aligned GQA forward shapes.

The shape list mirrors the forward production-oriented configurations from
TileOPs main's `benchmarks/ops/attention/bench_gqa.py` as of 2026-04-15.
For this WS-kernel study we default to the inference/prefill subset because it
matches the causal forward kernels under discussion more closely.

Design note:
- The parent process orchestrates runs and writes JSON.
- Each benchmark case runs in a fresh subprocess (`--worker`) so imports from
  different TileOPs worktrees do not pollute one another.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch


ROOT = Path("/home/ga/TileOPs")
DATA_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "data"
DEFAULT_BASE_REPO = Path("/tmp/tileops-pr871-base")
DEFAULT_REORDER_REPO = Path("/tmp/tileops-pr871-reorder")
ANCHOR_SCRIPT = ROOT / "_test_ws_fa3_v2_persistent_anchor_causal.py"


@dataclass(frozen=True)
class ShapeConfig:
    shape_id: str
    batch: int
    seq_len: int
    heads: int
    heads_kv: int
    dim: int
    causal: bool
    dtype: str


PRODUCTION_PREFILL_SHAPES = [
    ShapeConfig("llama8b-1k", 1, 1024, 32, 8, 128, True, "float16"),
    ShapeConfig("llama8b-4k", 1, 4096, 32, 8, 128, True, "float16"),
    ShapeConfig("llama8b-8k", 1, 8192, 32, 8, 128, True, "float16"),
    ShapeConfig("llama8b-32k", 1, 32768, 32, 8, 128, True, "float16"),
    ShapeConfig("llama8b-128k", 1, 131072, 32, 8, 128, True, "float16"),
    ShapeConfig("llama70b-4k", 1, 4096, 64, 8, 128, True, "float16"),
    ShapeConfig("llama405b-4k", 1, 4096, 128, 8, 128, True, "float16"),
]

TRAINING_SHAPES = [
    ShapeConfig("train-8b-4k", 2, 4096, 32, 8, 128, True, "bfloat16"),
    ShapeConfig("train-8b-8k", 1, 8192, 32, 8, 128, True, "bfloat16"),
    ShapeConfig("train-70b-4k", 1, 4096, 64, 8, 128, True, "bfloat16"),
    ShapeConfig("train-405b-4k", 1, 4096, 128, 8, 128, True, "bfloat16"),
    ShapeConfig("sft-8b", 2, 2048, 32, 8, 128, True, "bfloat16"),
]


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _shape_map(include_training: bool) -> dict[str, ShapeConfig]:
    shapes = list(PRODUCTION_PREFILL_SHAPES)
    if include_training:
        shapes.extend(TRAINING_SHAPES)
    return {shape.shape_id: shape for shape in shapes}


def _compute_tflops(shape: ShapeConfig, median_ms: float) -> float:
    flops_per_matmul = 2.0 * shape.batch * shape.heads * shape.seq_len * shape.seq_len * shape.dim
    flops = flops_per_matmul * 2.0
    if shape.causal:
        flops /= 2.0
    return flops / (median_ms / 1e3) / 1e12


def _build_repo_op(repo: Path, shape: ShapeConfig, tune: bool):
    sys.path.insert(0, str(repo))
    from tileops.ops import GroupQueryAttentionFwdOp  # noqa: PLC0415

    dtype = _dtype_from_name(shape.dtype)
    op = GroupQueryAttentionFwdOp(
        shape.batch,
        shape.heads,
        shape.heads_kv,
        shape.seq_len,
        shape.dim,
        shape.causal,
        dtype,
        tune=tune,
    )
    return op


def _build_pre_pr871_wgmma(shape: ShapeConfig):
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from tileops.kernels.flash_attn.fwd import GqaFwdWgmmaPipelinedKernel  # noqa: PLC0415

    dtype = _dtype_from_name(shape.dtype)
    return GqaFwdWgmmaPipelinedKernel(
        shape.batch,
        shape.heads,
        shape.heads_kv,
        shape.seq_len,
        shape.dim,
        shape.causal,
        dtype,
        tune=False,
    )


def _load_module_from_path(module_name: str, path: Path):
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_anchor_causal(shape: ShapeConfig):
    if not shape.causal:
        raise ValueError("anchor_causal only supports causal=True")
    if shape.dtype != "float16":
        raise ValueError("anchor_causal currently only supports float16")
    m_blocks = math.ceil(shape.seq_len / 128)
    if m_blocks % 2 != 0:
        raise ValueError(f"anchor_causal requires even M_blocks, got {m_blocks}")
    num_sms = int(os.environ.get("V2P_NUM_SMS", "132"))
    total_pairs = shape.batch * shape.heads_kv * (m_blocks // 2) * (shape.heads // shape.heads_kv)
    if total_pairs < num_sms:
        raise ValueError(
            f"anchor_causal requires total_pairs >= NUM_SMS, got {total_pairs} < {num_sms}"
        )
    module = _load_module_from_path("anchor_causal_module", ANCHOR_SCRIPT)
    return module.build_fa3_v2_persistent_causal(
        shape.batch,
        shape.seq_len,
        shape.heads,
        shape.heads_kv,
        shape.dim,
        block_m=128,
        block_n=128,
    )(block_m=128, block_n=128)


def _build_fa3(shape: ShapeConfig):
    from flash_attn_interface import flash_attn_func  # noqa: PLC0415

    def run(q, k, v):
        out = flash_attn_func(q, k, v, causal=shape.causal)
        return out[0] if isinstance(out, tuple) else out

    return run


def _prepare_inputs(shape: ShapeConfig):
    dtype = _dtype_from_name(shape.dtype)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    q = torch.randn(
        shape.batch, shape.seq_len, shape.heads, shape.dim, device="cuda", dtype=dtype
    )
    k = torch.randn(
        shape.batch, shape.seq_len, shape.heads_kv, shape.dim, device="cuda", dtype=dtype
    )
    v = torch.randn(
        shape.batch, shape.seq_len, shape.heads_kv, shape.dim, device="cuda", dtype=dtype
    )
    return q, k, v


def _bench_callable(fn, q, k, v, warmup: int, reps: int):
    for _ in range(warmup):
        out = fn(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize()

    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    median_ms = statistics.median(samples)
    return median_ms, samples


def _worker_run(args: argparse.Namespace) -> None:
    shape = _shape_map(args.include_training)[args.shape_id]
    q, k, v = _prepare_inputs(shape)
    result: dict[str, Any] = {
        "variant": args.variant,
        "shape": asdict(shape),
        "warmup": args.warmup,
        "reps": args.reps,
        "status": "ok",
    }

    try:
        if args.variant == "pre_pr871_wgmma":
            op = _build_pre_pr871_wgmma(shape)

            def fn(q, k, v):
                return op(q, k, v)

            kernel_name = type(op).__name__
        elif args.variant == "pr871_base":
            op = _build_repo_op(Path(args.base_repo), shape, args.tune)

            def fn(q, k, v):
                return op(q, k, v)

            kernel_name = type(op.kernel).__name__
        elif args.variant == "pr871_reorder":
            op = _build_repo_op(Path(args.reorder_repo), shape, args.tune)

            def fn(q, k, v):
                return op(q, k, v)

            kernel_name = type(op.kernel).__name__
        elif args.variant == "current_repo":
            op = _build_repo_op(ROOT, shape, args.tune)

            def fn(q, k, v):
                return op(q, k, v)

            kernel_name = type(op.kernel).__name__
        elif args.variant == "anchor_causal":
            fn = _build_anchor_causal(shape)
            kernel_name = "anchor_causal"
        elif args.variant == "fa3":
            fn = _build_fa3(shape)
            kernel_name = "fa3"
        else:
            raise ValueError(f"unknown variant {args.variant}")

        median_ms, samples = _bench_callable(fn, q, k, v, args.warmup, args.reps)
        result["median_ms"] = median_ms
        result["samples_ms"] = samples
        result["tflops"] = _compute_tflops(shape, median_ms)
        result["kernel_name"] = kernel_name
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result))


def _environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    gpu_name = None
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": gpu_count,
        "gpu_name_visible0": gpu_name,
        "tilelang_cleanup_temp_files": os.environ.get("TILELANG_CLEANUP_TEMP_FILES"),
        "v2p_num_sms": os.environ.get("V2P_NUM_SMS"),
        "base_repo": str(args.base_repo),
        "reorder_repo": str(args.reorder_repo),
    }


def _parent_run(args: argparse.Namespace) -> None:
    shape_map = _shape_map(args.include_training)
    selected_shape_ids = args.shapes or list(shape_map)
    results = []

    for shape_id in selected_shape_ids:
        if shape_id not in shape_map:
            raise ValueError(f"unknown shape_id {shape_id}")
        for variant in args.variants:
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--variant",
                variant,
                "--shape-id",
                shape_id,
                "--warmup",
                str(args.warmup),
                "--reps",
                str(args.reps),
                "--base-repo",
                str(args.base_repo),
                "--reorder-repo",
                str(args.reorder_repo),
            ]
            if args.include_training:
                cmd.append("--include-training")
            if args.tune:
                cmd.append("--tune")

            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            lines = [line for line in proc.stdout.splitlines() if line.strip()]
            if not lines:
                result = {
                    "variant": variant,
                    "shape": asdict(shape_map[shape_id]),
                    "status": "error",
                    "error": f"worker produced no JSON output (rc={proc.returncode})",
                    "stderr": proc.stderr.strip(),
                }
            else:
                result = json.loads(lines[-1])
                if proc.returncode != 0 and result.get("status") == "ok":
                    result["status"] = "error"
                    result["error"] = f"worker exited with rc={proc.returncode}"
                    result["stderr"] = proc.stderr.strip()
            results.append(result)
            status = result["status"]
            if status == "ok":
                print(
                    f"{shape_id:<14} {variant:<14} "
                    f"{result['median_ms']:>9.4f} ms  {result['tflops']:>7.1f} TF"
                )
            else:
                print(f"{shape_id:<14} {variant:<14} ERROR  {result.get('error', 'unknown')}")

    payload = {
        "study": "ws_kernel_evolution",
        "source": "bench_gqa_milestones.py",
        "shape_set": "production_prefill" if not args.include_training else "production_prefill_plus_training",
        "variants": args.variants,
        "environment": _environment_metadata(args),
        "results": results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = DATA_DIR / f"{stamp}_gqa_milestones.json"
    else:
        args.output = Path(args.output)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pre_pr871_wgmma", "pr871_base", "pr871_reorder", "anchor_causal", "fa3"],
        choices=[
            "current_repo",
            "pre_pr871_wgmma",
            "pr871_base",
            "pr871_reorder",
            "anchor_causal",
            "fa3",
        ],
    )
    parser.add_argument("--shapes", nargs="*", help="Subset of shape ids to run")
    parser.add_argument("--include-training", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--base-repo", type=Path, default=DEFAULT_BASE_REPO)
    parser.add_argument("--reorder-repo", type=Path, default=DEFAULT_REORDER_REPO)
    parser.add_argument("--output", type=Path)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--variant",
        choices=[
            "current_repo",
            "pre_pr871_wgmma",
            "pr871_base",
            "pr871_reorder",
            "anchor_causal",
            "fa3",
        ],
    )
    parser.add_argument("--shape-id")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        if args.variant is None or args.shape_id is None:
            raise ValueError("--worker requires --variant and --shape-id")
        _worker_run(args)
    else:
        _parent_run(args)


if __name__ == "__main__":
    main()
