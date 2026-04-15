#!/usr/bin/env python3
"""Dump CUDA / ptxas / SASS for reorder vs delayed-rescale-only kernels.

This compares the two *experimental* causal kernels used in the timeline study:

- reorder core split
- reorder + delayed_rescale_only core split

Both are compiled through TileLang, then the generated CUDA is recompiled with
nvcc `-Xptxas -v` to recover resource usage and a cubin for SASS dumping.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/ga/TileOPs")
OUT_DIR = ROOT / "experiments" / "ws_kernel_evolution" / "data" / "codegen_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_PY = Path("/home/ga/anaconda3/envs/env_tilelang_20260119/bin/python")
if sys.executable != str(ENV_PY):
    raise RuntimeError(f"run with {ENV_PY}, got {sys.executable}")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import tilelang  # noqa: E402

from experiments.ws_kernel_evolution.scripts.bench_timeline_pr871_core_split import (  # noqa: E402
    build_clock_kernel as build_reorder_core_split,
)
from experiments.ws_kernel_evolution.scripts.bench_timeline_pr871_reorder_delayed_rescale_core_split import (  # noqa: E402
    build_clock_kernel as build_reorder_delayed_core_split,
)


def get_cuda_source(compiled_kernel) -> str:
    if hasattr(compiled_kernel, "get_kernel_source"):
        return compiled_kernel.get_kernel_source()
    if hasattr(compiled_kernel, "kernel_source"):
        return compiled_kernel.kernel_source
    raise AttributeError("compiled kernel has no kernel source accessor")


def compile_variant(tag: str, build_fn, kwargs: dict) -> Path:
    kernel_builder = build_fn(**kwargs)
    compiled = kernel_builder(block_m=128, block_n=128)
    cuda_src = get_cuda_source(compiled)
    cu_path = OUT_DIR / f"{tag}.cu"
    cu_path.write_text(cuda_src)
    print(f"[{tag}] wrote CUDA to {cu_path}")
    return cu_path


def nvcc_compile(cu_path: Path) -> tuple[Path, str]:
    tilelang_dir = Path(tilelang.__file__).resolve().parent
    cubin_path = cu_path.with_suffix(".cubin")
    cmd = [
        "/usr/local/cuda/bin/nvcc",
        "-cubin",
        "-O3",
        "-lineinfo",
        "-arch=sm_90a",
        "-std=c++17",
        f"-I{tilelang_dir / 'src'}",
        f"-I{tilelang_dir / '3rdparty/cutlass/include'}",
        "-DENABLE_BF16",
        "--use_fast_math",
        "-Xptxas",
        "-v",
        "-o",
        str(cubin_path),
        str(cu_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    ptxas_text = (proc.stdout or "") + (proc.stderr or "")
    ptxas_path = cu_path.with_suffix(".ptxas.txt")
    ptxas_path.write_text(ptxas_text)
    print(f"[{cu_path.stem}] wrote ptxas log to {ptxas_path}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvcc failed for {cu_path.name} with code {proc.returncode}; "
            f"see {ptxas_path}"
        )
    return cubin_path, ptxas_text


def dump_sass(cubin_path: Path) -> Path:
    sass_path = cubin_path.with_suffix(".sass")
    proc = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    sass_path.write_text(proc.stdout)
    print(f"[{cubin_path.stem}] wrote SASS to {sass_path}")
    return sass_path


def main() -> None:
    common_kwargs = dict(batch=4, seq_len=4096, heads=64, heads_kv=8, dim=128)
    reorder_cu = compile_variant(
        "reorder_core_split_codegen",
        build_reorder_core_split,
        dict(common_kwargs, variant="reorder"),
    )
    delayed_cu = compile_variant(
        "reorder_delayed_rescale_core_split_codegen",
        build_reorder_delayed_core_split,
        common_kwargs,
    )

    for cu_path in (reorder_cu, delayed_cu):
        cubin_path, _ = nvcc_compile(cu_path)
        dump_sass(cubin_path)

    print(f"Artifacts written under {OUT_DIR}")


if __name__ == "__main__":
    main()
