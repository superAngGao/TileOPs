#!/usr/bin/env python3
"""Profile tensor-pipe activity for key milestone kernels with Nsight Compute.

This script uses a two-level design:

- parent mode: invokes `ncu` serially for each milestone variant and parses CSV
- worker mode: builds the requested kernel, warms it up, then brackets one
  profiled invocation with `cudaProfilerStart/Stop`

The initial target is the canonical causal comparison shape:

- `B=4, S=4096, H=64, Hkv=8, D=128, causal=True`
"""

from __future__ import annotations

import argparse
import csv
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
TMP_NCU_DIR = ROOT / ".tmp" / "ncu"
DEFAULT_BASE_REPO = Path("/tmp/tileops-pr871-base")
DEFAULT_REORDER_REPO = Path("/tmp/tileops-pr871-reorder")
ANCHOR_SCRIPT = ROOT / "_test_ws_fa3_v2_persistent_anchor_causal.py"
ENV_PY = Path("/home/ga/anaconda3/envs/env_tilelang_20260119/bin/python")
NCU_BIN = Path("/usr/local/cuda/bin/ncu")

METRICS = [
    "gpu__time_duration.sum",
    "sm__pipe_tensor_cycles_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor_op_gmma",
    "sm__sass_inst_executed_op_shared_gmma",
]


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


PROFILE_SHAPES = {
    "canonical_4k": ShapeConfig(
        "canonical_4k", batch=4, seq_len=4096, heads=64, heads_kv=8, dim=128, causal=True, dtype="float16"
    )
}


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _build_repo_op(repo: Path, shape: ShapeConfig, tune: bool):
    sys.path.insert(0, str(repo))
    from tileops.ops import GroupQueryAttentionFwdOp  # noqa: PLC0415

    dtype = _dtype_from_name(shape.dtype)
    return GroupQueryAttentionFwdOp(
        shape.batch,
        shape.heads,
        shape.heads_kv,
        shape.seq_len,
        shape.dim,
        shape.causal,
        dtype,
        tune=tune,
    )


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
    import importlib.util

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


def _prepare_inputs(shape: ShapeConfig):
    dtype = _dtype_from_name(shape.dtype)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    q = torch.randn(shape.batch, shape.seq_len, shape.heads, shape.dim, device="cuda", dtype=dtype)
    k = torch.randn(shape.batch, shape.seq_len, shape.heads_kv, shape.dim, device="cuda", dtype=dtype)
    v = torch.randn(shape.batch, shape.seq_len, shape.heads_kv, shape.dim, device="cuda", dtype=dtype)
    return q, k, v


def _worker_run(args: argparse.Namespace) -> None:
    shape = PROFILE_SHAPES[args.shape_id]
    q, k, v = _prepare_inputs(shape)

    if args.variant == "pre_pr871_wgmma":
        op = _build_pre_pr871_wgmma(shape)

        def fn(q, k, v):
            return op(q, k, v)

        kernel_name = type(op).__name__
    elif args.variant == "pr871_base":
        op = _build_repo_op(Path(args.base_repo), shape, tune=False)

        def fn(q, k, v):
            return op(q, k, v)

        kernel_name = type(op.kernel).__name__
    elif args.variant == "pr871_reorder":
        op = _build_repo_op(Path(args.reorder_repo), shape, tune=False)

        def fn(q, k, v):
            return op(q, k, v)

        kernel_name = type(op.kernel).__name__
    elif args.variant == "anchor_causal":
        fn = _build_anchor_causal(shape)
        kernel_name = "anchor_causal"
    else:
        raise ValueError(f"unknown variant {args.variant}")

    for _ in range(args.warmup):
        out = fn(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize()

    ms_samples = []
    for _ in range(args.reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(q, k, v)
        if isinstance(out, tuple):
            out = out[0]
        end.record()
        end.synchronize()
        ms_samples.append(float(start.elapsed_time(end)))
    median_ms = statistics.median(ms_samples)

    cudart = torch.cuda.cudart()
    torch.cuda.synchronize()
    cudart.cudaProfilerStart()
    out = fn(q, k, v)
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()

    print(
        json.dumps(
            {
                "variant": args.variant,
                "shape": asdict(shape),
                "kernel_name": kernel_name,
                "median_ms": median_ms,
                "samples_ms": ms_samples,
            }
        )
    )


def _parse_ncu_csv(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith('"')]
    if not lines:
        raise ValueError("no CSV rows found in ncu output")

    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows:
        raise ValueError("failed to parse ncu CSV rows")

    kernel_names = sorted({row["Kernel Name"] for row in rows if row.get("Kernel Name")})
    metrics: dict[str, float] = {}

    desired_suffixes = {
        "gpu__time_duration.sum",
        "sm__pipe_tensor_cycles_active.sum",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_pipe_tensor_op_gmma.sum",
        "sm__sass_inst_executed_op_shared_gmma.sum",
    }
    for row in rows:
        metric_name = row["Metric Name"]
        if metric_name in desired_suffixes:
            try:
                metrics[metric_name] = float(row["Metric Value"])
            except ValueError:
                pass

    return {"kernel_names": kernel_names, "metrics": metrics}


def _parent_run(args: argparse.Namespace) -> None:
    TMP_NCU_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for variant in args.variants:
        cmd = [
            str(NCU_BIN),
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--csv",
            "--metrics",
            ",".join(METRICS),
            str(ENV_PY),
            __file__,
            "--worker",
            "--variant",
            variant,
            "--shape-id",
            args.shape_id,
            "--warmup",
            str(args.warmup),
            "--reps",
            str(args.reps),
            "--base-repo",
            str(args.base_repo),
            "--reorder-repo",
            str(args.reorder_repo),
        ]
        env = os.environ.copy()
        env["TMPDIR"] = str(TMP_NCU_DIR)
        env["HOME"] = str(TMP_NCU_DIR)

        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        stdout = proc.stdout
        stderr = proc.stderr

        worker_lines = [line for line in stdout.splitlines() if line.startswith("{") and line.endswith("}")]
        worker_payload = json.loads(worker_lines[-1]) if worker_lines else {}
        ncu_payload = _parse_ncu_csv(stdout) if proc.returncode == 0 else {}

        result = {
            "variant": variant,
            "shape_id": args.shape_id,
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "worker": worker_payload,
            "ncu": ncu_payload,
            "stderr": stderr,
        }
        results.append(result)

        if result["status"] == "ok":
            m = result["ncu"]["metrics"]
            print(
                f"{variant:<16} "
                f"time_ns={m.get('gpu__time_duration.sum', float('nan')):.0f} "
                f"tensor_pct={m.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed', float('nan')):.1f}% "
                f"tensor_cycles={m.get('sm__pipe_tensor_cycles_active.sum', float('nan')):.0f} "
                f"gmma_inst={m.get('sm__inst_executed_pipe_tensor_op_gmma.sum', float('nan')):.0f}"
            )
        else:
            print(f"{variant:<16} ERROR rc={proc.returncode}")

    payload = {
        "study": "ws_kernel_evolution",
        "kind": "ncu_tensor_pipe_profile",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shape": asdict(PROFILE_SHAPES[args.shape_id]),
        "variants": args.variants,
        "metrics": METRICS,
        "environment": {
            "python": sys.version,
            "python_executable": str(ENV_PY),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name_visible0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ncu": subprocess.run([str(NCU_BIN), "--version"], capture_output=True, text=True).stdout.strip(),
            "tmpdir": str(TMP_NCU_DIR),
            "base_repo": str(args.base_repo),
            "reorder_repo": str(args.reorder_repo),
        },
        "results": results,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = args.output or DATA_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ncu_tensor_pipe_milestones.json"
    Path(output).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pre_pr871_wgmma", "pr871_base", "pr871_reorder", "anchor_causal"],
        choices=["pre_pr871_wgmma", "pr871_base", "pr871_reorder", "anchor_causal"],
    )
    parser.add_argument("--shape-id", default="canonical_4k", choices=sorted(PROFILE_SHAPES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--base-repo", type=Path, default=DEFAULT_BASE_REPO)
    parser.add_argument("--reorder-repo", type=Path, default=DEFAULT_REORDER_REPO)
    parser.add_argument("--output", type=Path)

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=["pre_pr871_wgmma", "pr871_base", "pr871_reorder", "anchor_causal"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        if args.variant is None:
            raise ValueError("--worker requires --variant")
        _worker_run(args)
    else:
        _parent_run(args)


if __name__ == "__main__":
    main()
