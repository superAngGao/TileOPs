"""TileLang CTA shell that calls an FA3/CUTLASS device helper.

This is intentionally a probe, not a production kernel.  The default path keeps
the TileLang PrimFunc responsible for launching the CTA grid.  Inside that
TileLang kernel we create TMA descriptors and use ``T.call_extern`` to enter a
CUDA device helper that invokes the FA3/CUTLASS body.
"""

import argparse
import hashlib
import math
import os
import re
import subprocess
from pathlib import Path

import torch
import tilelang
from tilelang import _ffi_api
import tilelang.language as T
from tilelang.engine.callback import register_c_postproc
from tilelang.language.kernel import _load_cuda_source, _normalize_cluster_dims, _normalize_threads
from tvm import tir
from tvm.script.ir_builder.tir import evaluate as T_evaluate

from benchmarks.benchmark_base import bench_kernel
from tilelang.contrib.nvcc import get_nvcc_compiler
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH


ROOT = Path(__file__).resolve().parent
FA3_INC = ROOT / ".github" / "runner" / "vendor" / "flash-attention" / "hopper"
TMP = ROOT / ".tmp"
LAUNCHER_NVCC_FLAGS = [
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ftemplate-backtrace-limit=0",
    "--use_fast_math",
    "--resource-usage",
    "-O3",
    "-DNDEBUG",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-shared",
    "-Xcompiler",
    "-fPIC",
    "-lineinfo",
    "-arch=sm_90a",
]
TMA_DTYPE_UINT8 = 0
TMA_INTERLEAVE_NONE = 0
TMA_SWIZZLE_128B = 3
TMA_L2_PROMOTION_128B = 2
TMA_OOB_FILL_NONE = 0
EVICT_NORMAL = 0


def _cuda_source_kernel_with_smem(
    *blocks: int,
    threads: int,
    source_code_or_path: str,
    entry_name: str,
    dynamic_smem_bytes: int,
    cluster_dims=None,
) -> None:
    source = _load_cuda_source(source_code_or_path)
    attrs = {
        "code_block_source": source,
        "code_block_entry_name": entry_name,
    }
    cluster_dims = _normalize_cluster_dims(cluster_dims)
    if cluster_dims is not None:
        attrs["cluster_dims"] = cluster_dims

    with _ffi_api.KernelLaunch(blocks, _normalize_threads(threads, is_cpu=False), attrs):
        if dynamic_smem_bytes > 0:
            smem_marker = T.alloc_shared((dynamic_smem_bytes,), "uint8")
            T.evaluate(smem_marker[0])
        T_evaluate(tir.call_extern("int32", entry_name))


def _quantize_kv_fa3_descale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    descale = x.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x / descale[:, None, :, None], -448.0, 448.0)
    return x_fp8.to(torch.float8_e4m3fn).contiguous(), descale.float().contiguous()


def _quantize_q_fa3_gqa_descale(
    x: torch.Tensor,
    heads_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    group_size = heads // heads_kv
    x_grouped = x.reshape(batch, seq_len, heads_kv, group_size, dim)
    descale = x_grouped.abs().amax(dim=(1, 3, 4)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x_grouped / descale[:, None, :, None, None], -448.0, 448.0)
    return x_fp8.to(torch.float8_e4m3fn).reshape(batch, seq_len, heads, dim).contiguous(), descale.float().contiguous()


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def _fa3_should_pack_gqa(seq_len: int, heads: int, heads_kv: int, block_m: int = 128) -> bool:
    if heads == heads_kv:
        return False
    group = heads // heads_kv
    nopack_eff = float(seq_len) / float(_round_up(seq_len, block_m))
    pack_eff = float(seq_len * group) / float(_round_up(seq_len * group, block_m))
    return nopack_eff < 0.9 * pack_eff


def _parse_pack_gqa(value: str, seq_len: int, heads: int, heads_kv: int) -> bool:
    if value == "auto":
        return _fa3_should_pack_gqa(seq_len, heads, heads_kv)
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"unknown pack_gqa mode: {value}")


def _reference_attention(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softcap: float = 0.0,
) -> torch.Tensor:
    batch, seq_len, heads, dim = q_fp8.shape
    heads_kv = k_fp8.shape[2]
    group = heads // heads_kv
    head_to_kv = torch.arange(heads, device=q_fp8.device) // group
    q = q_fp8.float() * q_descale[:, None, head_to_kv, None]
    k = k_fp8.float() * k_descale[:, None, :, None]
    v = v_fp8.float() * v_descale[:, None, :, None]
    k_for_q = k[:, :, head_to_kv, :]
    v_for_q = v[:, :, head_to_kv, :]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k_for_q) * (1.0 / math.sqrt(dim))
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v_for_q).to(torch.bfloat16)


def _launcher_source(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    pack_gqa: bool,
    softcap: float,
) -> str:
    if dim != 128:
        raise ValueError("this probe only targets the FA3 FP8 hdim128 mode")
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    scale = 1.0 / math.sqrt(dim)
    pack_gqa_literal = "true" if pack_gqa else "false"
    softcap_literal = "true" if softcap > 0.0 else "false"
    return f"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#include "cutlass/numeric_types.h"
#include "flash.h"
#include "flash_fwd_launch_template.h"

extern "C" int tileops_fa3_shaped_host_launch(
    void* q,
    void* k,
    void* v,
    float* q_descale,
    float* k_descale,
    float* v_descale,
    void* output,
    float* lse) {{
  constexpr int B = {batch};
  constexpr int S = {seq_len};
  constexpr int H = {heads};
  constexpr int HKV = {heads_kv};
  constexpr int D = {dim};
  constexpr bool PackGQA = {pack_gqa_literal};

  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {{
    std::fprintf(stderr, "cudaGetDevice failed: %s\\n", cudaGetErrorString(err));
    return -1;
  }}
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) {{
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\\n", cudaGetErrorString(err));
    return -1;
  }}

  Flash_fwd_params params{{}};
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.o_ptr = output;
  params.softmax_lse_ptr = lse;

  params.q_batch_stride = int64_t(S * H * D);
  params.k_batch_stride = int64_t(S * HKV * D);
  params.v_batch_stride = int64_t(S * HKV * D);
  params.o_batch_stride = int64_t(S * H * D);
  params.q_row_stride = int64_t(H * D);
  params.k_row_stride = int64_t(HKV * D);
  params.v_row_stride = int64_t(HKV * D);
  params.o_row_stride = int64_t(H * D);
  params.q_head_stride = D;
  params.k_head_stride = D;
  params.v_head_stride = D;
  params.o_head_stride = D;
  params.v_dim_stride = 1;

  params.q_descale_ptr = q_descale;
  params.k_descale_ptr = k_descale;
  params.v_descale_ptr = v_descale;
  params.q_descale_batch_stride = HKV;
  params.k_descale_batch_stride = HKV;
  params.v_descale_batch_stride = HKV;
  params.q_descale_head_stride = 1;
  params.k_descale_head_stride = 1;
  params.v_descale_head_stride = 1;

  params.b = B;
  params.b_k = B;
  params.h = H;
  params.h_k = HKV;
  params.seqlen_q = S;
  params.seqlen_k = S;
  params.seqlen_q_rounded = ((S + 127) / 128) * 128;
  params.seqlen_k_rounded = ((S + 127) / 128) * 128;
  params.d = D;
  params.d_rounded = D;
  params.dv = D;
  params.dv_rounded = D;
  params.total_q = B * S;
  params.total_k = B * S;
  params.total_knew = 0;

  params.scale_softmax = {scale:.17g}f;
  params.softcap = static_cast<float>({softcap:.17g});
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.window_size_left = S - 1;
  params.window_size_right = S - 1;
  params.attention_chunk = 0;
  params.is_bf16 = false;
  params.is_fp32 = false;
  params.is_e4m3 = true;
  params.is_causal = false;
  params.is_local = false;
  params.is_rotary_interleaved = false;
  params.num_splits = 1;
  params.pack_gqa = PackGQA;
  params.arch = props.major * 10 + props.minor;
  params.num_sm = props.multiProcessorCount;
  params.page_table = nullptr;
  params.page_size = 1;
  params.num_pages = 0;
  params.pagedkv_tma = false;
  params.skip_scheduler_metadata_computation = false;

  if (params.arch != 90) {{
    std::fprintf(stderr, "this probe expects sm90, got sm%d\\n", params.arch);
    return -1;
  }}

  run_flash_fwd<90, 128, 128, 1,
                cutlass::float_e4m3_t, cutlass::bfloat16_t,
                false, false, {softcap_literal}, false, false, false, false,
                PackGQA, false, false>(params, cudaStream_t{{0}});
  err = cudaGetLastError();
  if (err != cudaSuccess) {{
    std::fprintf(stderr, "run_flash_fwd failed: %s\\n", cudaGetErrorString(err));
    return -1;
  }}
  return 0;
}}
"""


def _compile_launcher(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    pack_gqa: bool,
    softcap: float,
    force: bool = False,
) -> Path:
    TMP.mkdir(exist_ok=True)
    source = _launcher_source(batch, seq_len, heads, heads_kv, dim, pack_gqa, softcap)
    cmd = [
        get_nvcc_compiler(),
        *LAUNCHER_NVCC_FLAGS,
        f"-I{FA3_INC}",
        f"-I{CUTLASS_INCLUDE_DIR}",
        f"-I{TILELANG_TEMPLATE_PATH}",
        "{cu_path}",
        "-lcuda",
        "-lcudart",
        "-o",
        "{so_path}",
    ]
    digest_input = source + "\n// nvcc:\n" + "\n".join(cmd)
    digest = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
    cu_path = TMP / f"tileops_fa3_shaped_launcher_{digest}.cu"
    so_path = TMP / f"tileops_fa3_shaped_launcher_{digest}.so"
    if so_path.exists() and not force:
        return so_path
    cu_path.write_text(source)
    cmd = [str(cu_path) if arg == "{cu_path}" else str(so_path) if arg == "{so_path}" else arg
           for arg in cmd]
    subprocess.run(cmd, check=True)
    return so_path


def _register_host_postproc(launcher_so: Path) -> None:
    launcher_so_str = str(launcher_so)
    packed_call_re = re.compile(
        r'  if \(tileops_fa3_shaped_shell_packed == NULL\) \{.*?'
        r'  if \(TVMFFIFunctionCall\(tileops_fa3_shaped_shell_packed, \(TVMFFIAny\*\) stack_ffi_any, \d+, &result_\d+\) != 0\) \{\n'
        r'    return -1;\n'
        r'  \}\n',
        re.DOTALL,
    )

    helper = f"""
#include <dlfcn.h>
typedef int (*tileops_fa3_shaped_host_launch_fn_t)(void*, void*, void*, float*, float*, float*, void*, float*);
static void* tileops_fa3_shaped_launcher_handle = NULL;
static tileops_fa3_shaped_host_launch_fn_t tileops_fa3_shaped_host_launch_fn = NULL;
static int tileops_fa3_load_shaped_launcher(void) {{
  if (tileops_fa3_shaped_host_launch_fn != NULL) {{
    return 0;
  }}
  tileops_fa3_shaped_launcher_handle = dlopen("{launcher_so_str}", RTLD_NOW | RTLD_LOCAL);
  if (tileops_fa3_shaped_launcher_handle == NULL) {{
    const char* err = dlerror();
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", err ? err : "dlopen tileops_fa3_shaped_launcher failed");
    return -1;
  }}
  tileops_fa3_shaped_host_launch_fn = (tileops_fa3_shaped_host_launch_fn_t)dlsym(
      tileops_fa3_shaped_launcher_handle, "tileops_fa3_shaped_host_launch");
  if (tileops_fa3_shaped_host_launch_fn == NULL) {{
    const char* err = dlerror();
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", err ? err : "dlsym tileops_fa3_shaped_host_launch failed");
    return -1;
  }}
  return 0;
}}
"""

    replacement = """
  if (tileops_fa3_load_shaped_launcher() != 0) {
    return -1;
  }
  if (tileops_fa3_shaped_host_launch_fn(q, k, v,
                                        (float*)q_descale, (float*)k_descale, (float*)v_descale,
                                        output, (float*)lse) != 0) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tileops_fa3_shaped_host_launch failed");
    return -1;
  }
"""

    def postproc(code, target):
        if "tileops_fa3_shaped_shell_packed" not in code:
            return code
        code = code.replace("#include <stdbool.h>\n", "#include <stdbool.h>\n" + helper)
        code, count = packed_call_re.subn(replacement, code, count=1)
        if count != 1:
            TMP.mkdir(exist_ok=True)
            (TMP / "tileops_fa3_shaped_postproc_failed.c").write_text(code)
            raise RuntimeError("failed to replace TileLang packed CUDA call with FA3 shaped launcher")
        return code

    register_c_postproc(postproc, override=True)


def _dummy_cuda_source() -> str:
    return """
extern "C" __global__ void tileops_fa3_shaped_shell(
    void* k, float* k_descale, float* lse, void* output,
    void* q, float* q_descale, void* v, float* v_descale) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    lse[0] = 0.0f;
  }
}
"""


def _source(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, stages: int, query_smem: bool) -> str:
    if dim != 128:
        raise ValueError("probe only supports dim=128")
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    group = heads // heads_kv
    num_m_blocks = (seq_len * group + 127) // 128
    return f"""
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <new>

#include "flash.h"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"
#include "flash_fwd_kernel_sm90.h"

__device__ int tileops_fa3_identity_page_table[{batch}];

extern "C" __global__ void __launch_bounds__(384, 1)
tileops_fa3_shaped_shell(
    void* k,
    float* k_descale,
    float* lse,
    void* output,
    void* q,
    float* q_descale,
    void* v,
    float* v_descale) {{
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementOut = cutlass::bfloat16_t;
  using TileShape_MNK = Shape<Int<128>, Int<224>, Int<128>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
      {stages}, ClusterShape, TileShape_MNK, 128, Element, float, cutlass::arch::Sm90,
      false, false, false, false,
      true,   // PagedKVNonTMA: dense K/V is exposed as one page per batch.
      false, false,
      true,   // MmaPV_is_RS, required for FP8.
      true,   // IntraWGOverlap.
      true,   // PackGQA.
      false,
      false>; // V_colmajor=false -> producer transpose_V path.
  using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
      typename CollectiveMainloop::TileShape_MNK_PV,
      ClusterShape, ElementOut, cutlass::arch::Sm90,
      CollectiveMainloop::NumMmaThreads,
      false, true, false, CollectiveMainloop::Transpose_V>;
  using Scheduler = flash::SingleTileScheduler<false, false, true, 128>;
  using AttnKernel = flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>;

  extern __shared__ __align__(1024) char smem_buf[];

  constexpr int B = {batch};
  constexpr int S = {seq_len};
  constexpr int H = {heads};
  constexpr int HKV = {heads_kv};
  constexpr int D = {dim};
  constexpr int GROUP = {group};
  constexpr int NUM_M_BLOCKS = {num_m_blocks};

  if constexpr ({str(query_smem).lower()}) {{
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {{
      lse[0] = float(AttnKernel::SharedStorageSize);
      lse[1] = float(sizeof(typename AttnKernel::SharedStorage));
    }}
    return;
  }}

  if (threadIdx.x == 0) {{
    tileops_fa3_identity_page_table[blockIdx.z] = int(blockIdx.z);
  }}
  __threadfence();
  __syncthreads();

  cutlass::FastDivmod attention_chunk_divmod(1);
  attention_chunk_divmod.divisor = 0;
  cutlass::FastDivmod page_size_divmod(S);
  cutlass::FastDivmod blocks_per_page_divmod((S + 223) / 224);

  typename CollectiveMainloop::Params mainloop_params{{
      static_cast<Element const*>(q),
      make_shape(make_shape(GROUP, S), D, HKV, B),
      make_stride(make_stride(int64_t(D), int64_t(H * D)), _1{{}}, int64_t(GROUP * D), int64_t(S * H * D)),
      make_shape(make_shape(GROUP, S), D, HKV, B),
      make_stride(make_stride(int64_t(D), int64_t(H * D)), _1{{}}, int64_t(GROUP * D), int64_t(S * H * D)),
      static_cast<Element*>(k),
      make_shape(S, D, HKV, B),
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(S * HKV * D)),
      static_cast<Element*>(v),
      D,
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(S * HKV * D)),
      static_cast<Element const*>(nullptr),
      make_shape(0, D, HKV, B),
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(0)),
      static_cast<Element const*>(nullptr),
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(0)),
      static_cast<Element const*>(nullptr),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D)),
      make_shape(S, D, H, B),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D)),
      static_cast<Element const*>(nullptr),
      make_shape(S, 0),
      make_stride(int64_t(0), _1{{}}),
      static_cast<Element const*>(nullptr),
      make_stride(int64_t(0), _1{{}}),
      false,
      tileops_fa3_identity_page_table,
      make_shape(B, 1),
      make_stride(int64_t(1), _1{{}}),
      page_size_divmod,
      blocks_per_page_divmod,
      cutlass::FastDivmod(GROUP),
      typename CollectiveMainloop::TMA_Q{{}},
      typename CollectiveMainloop::TMA_K{{}},
      typename CollectiveMainloop::TMA_V{{}},
      typename CollectiveMainloop::TMA_K{{}},
      typename CollectiveMainloop::TMA_V{{}},
      typename CollectiveMainloop::TMA_Qv{{}},
      float(0.08838834764831845f * M_LOG2E),
      q_descale,
      k_descale,
      v_descale,
      make_stride(int64_t(HKV), int64_t(1)),
      make_stride(int64_t(HKV), int64_t(1)),
      make_stride(int64_t(HKV), int64_t(1)),
      0.0f,
      -1,
      -1,
      attention_chunk_divmod,
      1,
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr)}};

  typename CollectiveEpilogue::Params epilogue_params{{
      static_cast<ElementOut*>(output),
      make_shape(S, D, H, B, 1),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D), int64_t(0)),
      make_shape(make_shape(GROUP, S), D, HKV, B, 1),
      make_stride(make_stride(int64_t(D), int64_t(H * D)), _1{{}}, int64_t(GROUP * D), int64_t(S * H * D), int64_t(0)),
      static_cast<float*>(nullptr),
      make_stride(int64_t(0), _1{{}}, int64_t(0), int64_t(0), int64_t(0)),
      make_stride(make_stride(int64_t(0), int64_t(0)), _1{{}}, int64_t(0), int64_t(0), int64_t(0)),
      lse,
      make_stride(_1{{}}, int64_t(S), int64_t(H * S), int64_t(0)),
      make_shape(make_shape(GROUP, S), HKV, B, 1),
      make_stride(make_stride(int64_t(S), _1{{}}), int64_t(GROUP * S), int64_t(H * S), int64_t(0)),
      static_cast<float*>(nullptr),
      make_stride(_1{{}}, int64_t(S), int64_t(H * S), int64_t(H * S * B)),
      make_stride(make_stride(int64_t(S), _1{{}}), int64_t(GROUP * S), int64_t(H * S), int64_t(H * S * B)),
      cutlass::FastDivmod(GROUP),
      typename CollectiveEpilogue::TMA_O{{}},
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr)}};

  typename Scheduler::Params scheduler_params{{
      NUM_M_BLOCKS, HKV, B, 1, GROUP, S,
      cutlass::FastDivmod(GROUP),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr)}};

  AttnKernel kernel;
  int const params_idx = (int(blockIdx.z) * H + int(blockIdx.y)) * NUM_M_BLOCKS + int(blockIdx.x);
  void* params_raw = static_cast<void*>(
      tileops_fa3_call_extern_detail::params_storage[params_idx].bytes);
  if (threadIdx.x == 0) {{
    ::new (params_raw) typename AttnKernel::Params{{
        mainloop_params,
        epilogue_params,
        cutlass::KernelHardwareInfo{{0, 132}},
        scheduler_params}};
    __threadfence();
  }}
  __syncthreads();
  auto const* params_global = reinterpret_cast<typename AttnKernel::Params const*>(params_raw);
  kernel(*params_global, smem_buf);
}}
"""


def _extern_helper_header_source(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    stages: int,
    query_smem: bool,
    static_persistent_call_extern: bool,
) -> str:
    if dim != 128:
        raise ValueError("probe only supports dim=128")
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    group = heads // heads_kv
    num_m_blocks = (seq_len + 127) // 128
    scheduler_type = (
        "flash::StaticPersistentTileScheduler<false>"
        if static_persistent_call_extern
        else "flash::SingleTileScheduler<false, false, false, 128>"
    )
    params_storage_count = "NUM_SMS" if static_persistent_call_extern else "NUM_M_BLOCKS * H * B"
    scheduler_params_init = (
        """{
      NUM_M_BLOCKS * H * B,
      cutlass::FastDivmod(NUM_M_BLOCKS),
      cutlass::FastDivmod(H),
      cutlass::FastDivmod(1)}"""
        if static_persistent_call_extern
        else """{
      NUM_M_BLOCKS, H, B, 1, GROUP, S,
      cutlass::FastDivmod(1),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr)}"""
    )
    params_idx_from_block = "int const params_idx = int(blockIdx.x);"
    if not static_persistent_call_extern:
        params_idx_from_block = (
            "int const params_idx = "
            "(int(blockIdx.z) * H + int(blockIdx.y)) * NUM_M_BLOCKS + int(blockIdx.x);"
        )
    run_prepared_work_check = (
        """tile_m_tl != int(blockIdx.x) ||
      bidh_tl != int(blockIdx.y) ||
      bidb_tl != int(blockIdx.z) ||
      tile_m_tl < 0 ||
      tile_m_tl >= NUM_SMS ||
      bidh_tl != 0 ||
      bidb_tl != 0"""
        if static_persistent_call_extern
        else """tile_m_tl != int(blockIdx.x) ||
      bidh_tl != int(blockIdx.y) ||
      bidb_tl != int(blockIdx.z) ||
      tile_m_tl < 0 ||
      tile_m_tl >= NUM_M_BLOCKS ||
      bidh_tl < 0 ||
      bidh_tl >= H ||
      bidb_tl < 0 ||
      bidb_tl >= B"""
    )
    run_prepared_params_idx = (
        "int const params_idx = tile_m_tl;"
        if static_persistent_call_extern
        else "int const params_idx = (bidb_tl * H + bidh_tl) * NUM_M_BLOCKS + tile_m_tl;"
    )
    return f"""
#pragma once

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "flash.h"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"
#include "flash_fwd_kernel_sm90.h"

__device__ int tileops_fa3_call_extern_identity_page_table[{batch}];

namespace tileops_fa3_call_extern_detail {{
using namespace cute;
using Element = cutlass::float_e4m3_t;
using ElementOut = cutlass::bfloat16_t;
using TileShape_MNK = Shape<Int<128>, Int<224>, Int<128>>;
using ClusterShape = Shape<_1, _1, _1>;
using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
    {stages}, ClusterShape, TileShape_MNK, 128, Element, float, cutlass::arch::Sm90,
    false, false, false, false,
    false,
    false, false,
    true,
    true,
    false,
    false,
    false>;
using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
    typename CollectiveMainloop::TileShape_MNK_PV,
    ClusterShape, ElementOut, cutlass::arch::Sm90,
    CollectiveMainloop::NumMmaThreads,
    false, false, false, CollectiveMainloop::Transpose_V>;
using Scheduler = {scheduler_type};
using AttnKernel = flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>;

constexpr int B = {batch};
constexpr int S = {seq_len};
constexpr int H = {heads};
constexpr int HKV = {heads_kv};
constexpr int D = {dim};
constexpr int GROUP = {group};
constexpr int NUM_M_BLOCKS = {num_m_blocks};
constexpr int NUM_SMS = 132;

struct alignas(64) ParamsStorage {{
  unsigned char bytes[sizeof(typename AttnKernel::Params)];
}};

__device__ ParamsStorage params_storage[{params_storage_count}];
}}  // namespace tileops_fa3_call_extern_detail

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_prepare_params(
    const CUtensorMap& q_desc,
    const CUtensorMap& k_desc,
    const CUtensorMap& v_desc,
    const CUtensorMap& output_desc,
    const void* q,
    const void* k,
    const void* v,
    const float* q_descale,
    const float* k_descale,
    const float* v_descale,
    const void* output,
    const float* lse,
    int tx_tl,
    int lane_id_tl,
    int warp_id_tl,
    int warpgroup_id_tl,
    int warpgroup_lane_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    void* smem_raw) {{
  (void)q_desc;
  (void)k_desc;
  (void)v_desc;
  (void)output_desc;

  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using ElementOut = cutlass::bfloat16_t;
  using TileShape_MNK = Shape<Int<128>, Int<224>, Int<128>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<
      {stages}, ClusterShape, TileShape_MNK, 128, Element, float, cutlass::arch::Sm90,
      false, false, false, false,
      false,
      false, false,
      true,
      true,
      false,
      false,
      false>;
  using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
      typename CollectiveMainloop::TileShape_MNK_PV,
      ClusterShape, ElementOut, cutlass::arch::Sm90,
      CollectiveMainloop::NumMmaThreads,
      false, false, false, CollectiveMainloop::Transpose_V>;
  using Scheduler = {scheduler_type};
  using AttnKernel = flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>;
  using AttnParams = typename AttnKernel::Params;

  (void)smem_raw;
  float* lse_mut = const_cast<float*>(lse);

  constexpr int B = {batch};
  constexpr int S = {seq_len};
  constexpr int H = {heads};
  constexpr int HKV = {heads_kv};
  constexpr int D = {dim};
  constexpr int GROUP = {group};
  constexpr int NUM_M_BLOCKS = {num_m_blocks};
  constexpr int NUM_SMS = 132;

  int const tx_cuda = int(threadIdx.x);
  bool const binding_mismatch =
      tx_tl != tx_cuda ||
      lane_id_tl != tx_cuda % 32 ||
      warp_id_tl != tx_cuda / 32 ||
      warpgroup_id_tl != tx_cuda / 128 ||
      warpgroup_lane_tl != tx_cuda % 128;
  if (binding_mismatch) {{
    asm volatile("trap;");
  }}
  bool const work_coord_mismatch =
      tile_m_tl != int(blockIdx.x) ||
      bidh_tl != int(blockIdx.y) ||
      bidb_tl != int(blockIdx.z) ||
      bidh_kv_tl != int(blockIdx.y) / GROUP ||
      bidh_tl < 0 ||
      bidh_tl >= H ||
      bidh_kv_tl < 0 ||
      bidh_kv_tl >= HKV;
  if (work_coord_mismatch) {{
    asm volatile("trap;");
  }}

  if constexpr ({str(query_smem).lower()}) {{
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {{
      lse_mut[0] = float(AttnKernel::SharedStorageSize);
      lse_mut[1] = float(sizeof(typename AttnKernel::SharedStorage));
    }}
    return;
  }}

  if (threadIdx.x == 0) {{
    tileops_fa3_call_extern_identity_page_table[blockIdx.z] = int(blockIdx.z);
  }}
  __threadfence();
  __syncthreads();

  cutlass::FastDivmod attention_chunk_divmod(1);
  attention_chunk_divmod.divisor = 0;
  cutlass::FastDivmod page_size_divmod(1);
  cutlass::FastDivmod blocks_per_page_divmod(1);

  typename CollectiveMainloop::TMA_Q tma_load_q{{}};
  typename CollectiveMainloop::TMA_K tma_load_k{{}};
  typename CollectiveMainloop::TMA_V tma_load_v{{}};
  using TmaQTraits = typename CollectiveMainloop::TMA_Q::Traits;
  using TmaKTraits = typename CollectiveMainloop::TMA_K::Traits;
  using TmaVTraits = typename CollectiveMainloop::TMA_V::Traits;
  static_cast<TmaQTraits&>(tma_load_q).tma_desc_ =
      *reinterpret_cast<cute::TmaDescriptor const*>(&q_desc);
  static_cast<TmaKTraits&>(tma_load_k).tma_desc_ =
      *reinterpret_cast<cute::TmaDescriptor const*>(&k_desc);
  static_cast<TmaVTraits&>(tma_load_v).tma_desc_ =
      *reinterpret_cast<cute::TmaDescriptor const*>(&v_desc);
  static_cast<TmaQTraits&>(tma_load_q).aux_params_.g_stride_ =
      make_stride(E<1>{{}}, E<0>{{}}, E<2>{{}}, E<3>{{}});
  static_cast<TmaKTraits&>(tma_load_k).aux_params_.g_stride_ =
      make_stride(E<1>{{}}, E<0>{{}}, E<2>{{}}, E<3>{{}});
  static_cast<TmaVTraits&>(tma_load_v).aux_params_.g_stride_ =
      make_stride(E<0>{{}}, E<1>{{}}, E<2>{{}}, E<3>{{}});

  typename CollectiveEpilogue::TMA_O tma_store_o{{}};
  using TmaOTraits = typename CollectiveEpilogue::TMA_O::Traits;
  static_cast<TmaOTraits&>(tma_store_o).tma_desc_ =
      *reinterpret_cast<cute::TmaDescriptor const*>(&output_desc);
  static_cast<TmaOTraits&>(tma_store_o).aux_params_.g_stride_ =
      make_stride(E<1>{{}}, E<0>{{}}, E<2>{{}}, E<3>{{}}, E<4>{{}});

  typename CollectiveMainloop::Params mainloop_params{{
      static_cast<Element const*>(q),
      make_shape(S, D, H, B),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D)),
      make_shape(S, D, H, B),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D)),
      const_cast<Element*>(static_cast<Element const*>(k)),
      make_shape(S, D, HKV, B),
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(S * HKV * D)),
      const_cast<Element*>(static_cast<Element const*>(v)),
      D,
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(S * HKV * D)),
      static_cast<Element const*>(nullptr),
      make_shape(0, D, HKV, B),
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(0)),
      static_cast<Element const*>(nullptr),
      make_stride(int64_t(HKV * D), _1{{}}, int64_t(D), int64_t(0)),
      static_cast<Element const*>(nullptr),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D)),
      make_shape(S, D, H, B),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D)),
      static_cast<Element const*>(nullptr),
      make_shape(S, 0),
      make_stride(int64_t(0), _1{{}}),
      static_cast<Element const*>(nullptr),
      make_stride(int64_t(0), _1{{}}),
      false,
      static_cast<int const*>(nullptr),
      make_shape(B, 0),
      make_stride(int64_t(0), _1{{}}),
      page_size_divmod,
      blocks_per_page_divmod,
      cutlass::FastDivmod(GROUP),
      tma_load_q,
      tma_load_k,
      tma_load_v,
      tma_load_k,
      tma_load_v,
      typename CollectiveMainloop::TMA_Qv{{}},
      float(0.08838834764831845f * M_LOG2E),
      q_descale,
      k_descale,
      v_descale,
      make_stride(int64_t(HKV), int64_t(1)),
      make_stride(int64_t(HKV), int64_t(1)),
      make_stride(int64_t(HKV), int64_t(1)),
      0.0f,
      -1,
      -1,
      attention_chunk_divmod,
      1,
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr)}};

  typename CollectiveEpilogue::Params epilogue_params{{
      const_cast<ElementOut*>(static_cast<ElementOut const*>(output)),
      make_shape(S, D, H, B, 1),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D), int64_t(0)),
      make_shape(S, D, H, B, 1),
      make_stride(int64_t(H * D), _1{{}}, int64_t(D), int64_t(S * H * D), int64_t(0)),
      static_cast<float*>(nullptr),
      make_stride(int64_t(0), _1{{}}, int64_t(0), int64_t(0), int64_t(0)),
      make_stride(int64_t(0), _1{{}}, int64_t(0), int64_t(0), int64_t(0)),
      lse_mut,
      make_stride(_1{{}}, int64_t(S), int64_t(H * S), int64_t(0)),
      make_shape(S, H, B, 1),
      make_stride(_1{{}}, int64_t(S), int64_t(H * S), int64_t(0)),
      static_cast<float*>(nullptr),
      make_stride(_1{{}}, int64_t(S), int64_t(H * S), int64_t(H * S * B)),
      make_stride(_1{{}}, int64_t(S), int64_t(H * S), int64_t(H * S * B)),
      cutlass::FastDivmod(1),
      tma_store_o,
      static_cast<int const*>(nullptr),
      static_cast<int const*>(nullptr)}};

  typename Scheduler::Params scheduler_params{scheduler_params_init};

  {params_idx_from_block}
  void* params_raw = static_cast<void*>(
      tileops_fa3_call_extern_detail::params_storage[params_idx].bytes);
  if (threadIdx.x == 0) {{
    ::new (params_raw) AttnParams{{
        mainloop_params,
        epilogue_params,
        cutlass::KernelHardwareInfo{{0, 132}},
        scheduler_params}};
    __threadfence();
  }}
  __syncthreads();
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_run_prepared(
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    void* smem_raw) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  bool const work_coord_mismatch =
      {run_prepared_work_check};
  if (work_coord_mismatch) {{
    asm volatile("trap;");
  }}

  char* smem_buf = reinterpret_cast<char*>(smem_raw);
  {run_prepared_params_idx}
  void* params_raw = static_cast<void*>(params_storage[params_idx].bytes);
  auto const* params_global = reinterpret_cast<AttnKernel::Params const*>(params_raw);
  AttnKernel kernel;
  kernel(*params_global, smem_buf);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_role_probe(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw);

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_prepare_runtime(
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    void* smem_raw) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  int const params_idx = (bidb_tl * H + bidh_tl) * NUM_M_BLOCKS + tile_m_tl;
  auto const* params = reinterpret_cast<AttnKernel::Params const*>(params_storage[params_idx].bytes);
  char* smem_buf = reinterpret_cast<char*>(smem_raw);
  typename AttnKernel::SharedStorage& shared_storage =
      *reinterpret_cast<typename AttnKernel::SharedStorage*>(smem_buf);

  static constexpr int NumMmaThreads =
      AttnKernel::NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  using ClusterShape = typename AttnKernel::ClusterShape;
  using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
  using MainloopPipelineKVNew = typename CollectiveMainloop::MainloopPipelineKVNew;
  using PipelineParamsK = typename MainloopPipelineK::Params;
  using PipelineParamsV = typename MainloopPipelineV::Params;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();
  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
  int const warp_group_idx = cutlass::canonical_warp_group_idx();

  if (warp_idx == 0 && lane_predicate) {{
    CollectiveMainloop::prefetch_tma_descriptors(params->mainloop);
    CollectiveEpilogue::prefetch_tma_descriptors(params->epilogue);
    shared_storage.pipelines.barrier_Q.init(
        AttnKernel::Use_TMA_Q ? 1 : AttnKernel::NumProducerThreads);
    if constexpr (AttnKernel::HasQv) {{
      shared_storage.pipelines.barrier_Qv.init(
          AttnKernel::Use_TMA_Q ? 1 : AttnKernel::NumProducerThreads);
    }}
    shared_storage.pipelines.barrier_O.init(
        size(ClusterShape{{}}) *
        (AttnKernel::Use_TMA_O ? 1 : NumMmaThreads));
  }}

  PipelineParamsK pipeline_params_k;
  pipeline_params_k.role = warp_group_idx == 0
      ? MainloopPipelineK::ThreadCategory::Producer
      : MainloopPipelineK::ThreadCategory::Consumer;
  pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  pipeline_params_k.is_leader = warp_group_thread_idx == 0;
  pipeline_params_k.num_consumers =
      !AttnKernel::LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;

  static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
  PipelineParamsVt pipeline_params_vt = pipeline_params_k;
  if constexpr (!AttnKernel::SameHeadDim) {{
    pipeline_params_vt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    if constexpr (AttnKernel::LargeHeadDimV) {{ pipeline_params_vt.num_consumers = NumMmaThreads; }}
  }}

  PipelineParamsV pipeline_params_v;
  pipeline_params_v.role = warp_group_idx == 0
      ? MainloopPipelineV::ThreadCategory::Producer
      : MainloopPipelineV::ThreadCategory::Consumer;
  pipeline_params_v.producer_arv_count = AttnKernel::NumProducerThreads;
  pipeline_params_v.consumer_arv_count = NumMmaThreads;

  MainloopPipelineK pipeline_k(
      shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{{}});
  MainloopPipelineV pipeline_v(
      shared_storage.pipelines.pipeline_v, pipeline_params_v);
  pipeline_params_vt.num_consumers = AttnKernel::NumProducerThreads;
  MainloopPipelineVt pipeline_vt(
      shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{{}});

  if constexpr (size(ClusterShape{{}}) > 1) {{
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  }} else {{
    __syncthreads();
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_run_role(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  int const tx_cuda = int(threadIdx.x);
  int const role_cuda = tx_cuda < 128 ? 0 : (tx_cuda < 256 ? 1 : 2);
  bool const role_mismatch =
      role_tl != role_cuda ||
      tx_tl != tx_cuda ||
      warpgroup_id_tl != tx_cuda / 128 ||
      group_tl <= 0 ||
      bidh_kv_tl != bidh_tl / group_tl;
  if (role_mismatch) {{
    asm volatile("trap;");
  }}

  static constexpr int NumMmaThreads =
      AttnKernel::NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int MmaThreadOffset =
      AttnKernel::NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockM = get<0>(typename AttnKernel::TileShape_MNK_PV{{}});

  using ClusterShape = typename AttnKernel::ClusterShape;
  using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
  using MainloopPipelineKVNew = typename CollectiveMainloop::MainloopPipelineKVNew;
  using PipelineState = typename CollectiveMainloop::PipelineState;
  using PipelineParamsK = typename MainloopPipelineK::Params;
  using PipelineParamsV = typename MainloopPipelineV::Params;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

  int const params_idx = (bidb_tl * H + bidh_tl) * NUM_M_BLOCKS + tile_m_tl;
  auto const* params = reinterpret_cast<AttnKernel::Params const*>(params_storage[params_idx].bytes);
  char* smem_buf = reinterpret_cast<char*>(smem_raw);
  typename AttnKernel::SharedStorage& shared_storage =
      *reinterpret_cast<typename AttnKernel::SharedStorage*>(smem_buf);

  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
  int const warp_group_idx = cutlass::canonical_warp_group_idx();

  PipelineParamsK pipeline_params_k;
  pipeline_params_k.role = warp_group_idx == 0
      ? MainloopPipelineK::ThreadCategory::Producer
      : MainloopPipelineK::ThreadCategory::Consumer;
  pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  pipeline_params_k.is_leader = warp_group_thread_idx == 0;
  pipeline_params_k.num_consumers =
      !AttnKernel::LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;

  static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
  PipelineParamsVt pipeline_params_vt = pipeline_params_k;
  if constexpr (!AttnKernel::SameHeadDim) {{
    pipeline_params_vt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    if constexpr (AttnKernel::LargeHeadDimV) {{ pipeline_params_vt.num_consumers = NumMmaThreads; }}
  }}

  PipelineParamsV pipeline_params_v;
  pipeline_params_v.role = warp_group_idx == 0
      ? MainloopPipelineV::ThreadCategory::Producer
      : MainloopPipelineV::ThreadCategory::Consumer;
  pipeline_params_v.producer_arv_count = AttnKernel::NumProducerThreads;
  pipeline_params_v.consumer_arv_count = NumMmaThreads;

  MainloopPipelineK pipeline_k(
      shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{{}}, cute::false_type{{}});
  MainloopPipelineV pipeline_v(
      shared_storage.pipelines.pipeline_v, pipeline_params_v, cute::false_type{{}});
  pipeline_params_vt.num_consumers = AttnKernel::NumProducerThreads;
  MainloopPipelineVt pipeline_vt(
      shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{{}}, cute::false_type{{}});

  CollectiveMainloop mainloop;
  CollectiveEpilogue epilogue;
  Scheduler scheduler(reinterpret_cast<typename Scheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

  if (role_tl == 0) {{
    cutlass::arch::warpgroup_reg_dealloc<AttnKernel::LoadRegisterRequirement>();
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
    int work_idx = 0;
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    static constexpr bool SingleProducerWarp =
        AttnKernel::NumProducerThreads == cutlass::NumThreadsPerWarp;
    if constexpr (SingleProducerWarp) {{
      if (warp_idx_in_warpgroup != 0) {{ return; }}
    }}
    if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) {{ scheduler.init_consumer(); }}

    cutlass::arch::wait_on_dependent_grids();
    auto block_coord = cute::make_tuple(
        int32_t(tile_m_tl), int32_t(bidh_tl), int32_t(bidb_tl), int32_t(0));
    typename AttnKernel::SeqlenInfo_t seqlen_info{{
        get<2>(block_coord),
        get<0>(params->mainloop.shape_Q),
        !params->mainloop.ptr_pagetable ? size<0>(params->mainloop.shape_K)
                                        : size<0>(params->mainloop.shape_K) * size<1>(params->mainloop.shape_pagetable),
        get<0>(params->mainloop.shape_K_new),
        params->mainloop.cu_seqlens_q, params->mainloop.cu_seqlens_k,
        params->mainloop.cu_seqlens_k_new, params->mainloop.seqused_q,
        params->mainloop.seqused_k, params->mainloop.leftpad_k,
        params->mainloop.seqlens_rotary}};
    auto scheduler_prefetch = []() {{}};
    mainloop.load(
        params->mainloop, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write,
        shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
    mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, shared_storage, work_idx);
  }} else {{
    cutlass::arch::warpgroup_reg_alloc<AttnKernel::MmaRegisterRequirement>();
    typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
    PipelineState smem_pipe_read;
    scheduler.init_consumer();
    mainloop.mma_init();
    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params->scheduler);
         work_tile_info.is_valid(params->scheduler);) {{
      auto block_coord = work_tile_info.get_block_coord(params->scheduler);
      int const bidb = get<2>(block_coord);
      typename AttnKernel::SeqlenInfo_t seqlen_info{{
          bidb,
          get<0>(params->mainloop.shape_Q),
          !params->mainloop.ptr_pagetable ? size<0>(params->mainloop.shape_K)
                                          : size<0>(params->mainloop.shape_K) * size<1>(params->mainloop.shape_pagetable),
          get<0>(params->mainloop.shape_K_new),
          params->mainloop.cu_seqlens_q, params->mainloop.cu_seqlens_k,
          params->mainloop.cu_seqlens_k_new, params->mainloop.seqused_q,
          params->mainloop.seqused_k, params->mainloop.leftpad_k,
          params->mainloop.seqlens_rotary}};
      float softmax_scale_log2 = params->mainloop.softmax_scale_log2;
      if constexpr (AttnKernel::Is_FP8 && !AttnKernel::Has_softcap) {{
        int const bidh = get<1>(block_coord);
        int const bidh_kv = !AttnKernel::PackGQA
            ? params->mainloop.qhead_per_khead_divmod.divide(bidh)
            : bidh;
        float const q_descale = params->mainloop.ptr_q_descale == nullptr ? 1.0f
            : params->mainloop.ptr_q_descale[
                bidb * get<0>(params->mainloop.stride_q_descale) +
                bidh_kv * get<1>(params->mainloop.stride_q_descale)];
        float const k_descale = params->mainloop.ptr_k_descale == nullptr ? 1.0f
            : params->mainloop.ptr_k_descale[
                bidb * get<0>(params->mainloop.stride_k_descale) +
                bidh_kv * get<1>(params->mainloop.stride_k_descale)];
        softmax_scale_log2 *= q_descale * k_descale;
      }}
      flash::Softmax<!AttnKernel::LargeHeadDimV ? 2 * (2 * kBlockM / NumMmaThreads) : 2,
                     !AttnKernel::Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
      Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(typename AttnKernel::TileShape_MNK_PV{{}}));
      bool tile_valid;
      if constexpr (!AttnKernel::LargeHeadDimV) {{
        tile_valid = mainloop.mma(
            params->mainloop, pipeline_k, pipeline_v, smem_pipe_read, tOrO, softmax,
            threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
      }} else {{
        if (warp_group_idx == 1) {{
          tile_valid = mainloop.mma(
              params->mainloop, pipeline_k, pipeline_v, smem_pipe_read, tOrO, softmax,
              threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
        }} else {{
          tile_valid = mainloop.mma_pv(
              params->mainloop, pipeline_v, smem_pipe_read, tOrO, softmax,
              threadIdx.x - MmaThreadOffset, seqlen_info, block_coord, shared_storage);
        }}
      }}
      work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(
          params->scheduler, work_tile_info);
      if constexpr (AttnKernel::Split && AttnKernel::Varlen) {{
        if (!work_tile_info.is_valid(params->scheduler)) {{
          cutlass::arch::launch_dependent_grids();
        }}
      }}
      if (tile_valid) {{
        epilogue.store(
            params->epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
            threadIdx.x - MmaThreadOffset, block_coord);
      }} else {{
        epilogue.store_zero(params->epilogue, threadIdx.x - MmaThreadOffset, block_coord);
      }}
    }}
    epilogue.store_tail();
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_run_producer(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int producer_warp_tl,
    int producer_lane_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  if (role_tl != 0) {{
    asm volatile("trap;");
  }}
  int const tx_cuda = int(threadIdx.x);
  bool const producer_gating_mismatch =
      tx_tl != tx_cuda ||
      tx_cuda >= 128 ||
      warpgroup_id_tl != tx_cuda / 128 ||
      producer_warp_tl != (tx_cuda / 32) % 4 ||
      producer_lane_tl != tx_cuda % 32;
  if (producer_gating_mismatch) {{
    asm volatile("trap;");
  }}
  tileops_fa3_shaped_run_role(
      0, tx_tl, warpgroup_id_tl, tile_m_tl, bidh_tl, bidb_tl,
      bidh_kv_tl, group_tl, smem_raw);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_producer_enter(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int producer_warp_tl,
    int producer_lane_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  using namespace tileops_fa3_call_extern_detail;

  (void)tile_m_tl;
  (void)bidb_tl;
  (void)smem_raw;
  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  int const tx_cuda = int(threadIdx.x);
  bool const mismatch =
      role_tl != 0 ||
      tx_tl != tx_cuda ||
      tx_cuda >= 128 ||
      warpgroup_id_tl != tx_cuda / 128 ||
      producer_warp_tl != (tx_cuda / 32) % 4 ||
      producer_lane_tl != tx_cuda % 32 ||
      group_tl <= 0 ||
      bidh_kv_tl != bidh_tl / group_tl;
  if (mismatch) {{
    asm volatile("trap;");
  }}
  cutlass::arch::warpgroup_reg_dealloc<AttnKernel::LoadRegisterRequirement>();
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_producer_reg_dealloc(
    int tx_tl,
    int warpgroup_id_tl,
    int producer_warp_tl,
    int producer_lane_tl) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  int const tx_cuda = int(threadIdx.x);
  bool const mismatch =
      tx_tl != tx_cuda ||
      tx_cuda >= 128 ||
      warpgroup_id_tl != tx_cuda / 128 ||
      producer_warp_tl != (tx_cuda / 32) % 4 ||
      producer_lane_tl != tx_cuda % 32;
  if (mismatch) {{
    asm volatile("trap;");
  }}
  cutlass::arch::warpgroup_reg_dealloc<AttnKernel::LoadRegisterRequirement>();
}}

__device__ __forceinline__ int tileops_fa3_shaped_swizzle_128b_index(int row, int col) {{
  int const row_phase = row & 7;
  int const row_tile = row >> 3;
  int const vec_col = col >> 4;
  int const vec = col & 15;
  return row_tile * 1024 + ((row_phase * 8 + (vec_col ^ row_phase)) * 16) + vec;
}}

__device__ __forceinline__ int tileops_fa3_shaped_swizzle_128b_index_contiguous(
    int row,
    int col,
    int continuous) {{
  int const row_phase = row & 7;
  int const row_tile = row >> 3;
  int const vec_col_tile = (col >> 4) >> 3;
  int const vec_col_phase = (col >> 4) & 7;
  int const vec = col & 15;
  return row_tile * 8 * continuous
      + vec_col_tile * 1024
      + ((row_phase * 8 + (vec_col_phase ^ row_phase)) * 16)
      + vec;
}}

extern "C" __device__ __forceinline__ int tileops_fa3_shaped_smem_q_offset_bytes() {{
  using namespace tileops_fa3_call_extern_detail;
  return int(reinterpret_cast<char*>(
      reinterpret_cast<AttnKernel::SharedStorage*>(0)->tensors.mainloop.smem_q.data()) -
      reinterpret_cast<char*>(0));
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_q_hlir_tma_probe(
    const void* q_shadow_raw,
    const void* q_raw,
    int tx_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128) {{
    asm volatile("trap;");
  }}

  auto const* q_shadow = static_cast<unsigned char const*>(q_shadow_raw);
  auto const* q = static_cast<unsigned char const*>(q_raw);
  int const rows[5] = {{0, 0, 7, 8, 127}};
  int const cols[5] = {{0, D - 1, 16, 0, D - 1}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const row = rows[i];
    int const col = cols[i];
    int const global_row = tile_m_tl * 128 + row;
    int const global_offset = ((bidb_tl * S + global_row) * H + bidh_tl) * D + col;
    int const smem_offset = tileops_fa3_shaped_swizzle_128b_index(row, col);
    mismatch = mismatch || (q_shadow[smem_offset] != q[global_offset]);
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary(
    const void* q_stage_raw,
    const void* q_raw,
    int tx_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || (tx_tl != 128 && tx_tl != 256)) {{
    asm volatile("trap;");
  }}

  auto* q_stage = const_cast<Element*>(static_cast<Element const*>(q_stage_raw));
  int const mma_thread_idx = tx_tl - cutlass::NumThreadsPerWarpGroup;
  static constexpr int MmaWarpGroups =
      size(typename CollectiveMainloop::TiledMmaPV{{}}) / cutlass::NumThreadsPerWarpGroup;
  Layout warp_group_thread_layout = make_layout(
      make_shape(Int<MmaWarpGroups>{{}}),
      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{{}}));
  int const warp_group_idx = mma_thread_idx / cutlass::NumThreadsPerWarpGroup;

  Tensor sQ = make_tensor(make_smem_ptr(q_stage), typename CollectiveMainloop::SmemLayoutQ{{}});
  typename CollectiveMainloop::TiledMmaQK tiled_mma_qk;
  auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
  auto tSrQ = wg_mma_qk.partition_fragment_A(sQ);
  (void)tSrQ;

  auto const* q_stage_bytes = static_cast<unsigned char const*>(q_stage_raw);
  auto const* q = static_cast<unsigned char const*>(q_raw);
  int const rows[5] = {{0, 0, 7, 8, 127}};
  int const cols[5] = {{0, D - 1, 16, 0, D - 1}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const row = rows[i];
    int const col = cols[i];
    int const global_row = tile_m_tl * 128 + row;
    int const global_offset = ((bidb_tl * S + global_row) * H + bidh_tl) * D + col;
    int const smem_offset = tileops_fa3_shaped_swizzle_128b_index(row, col);
    mismatch = mismatch || (q_stage_bytes[smem_offset] != q[global_offset]);
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_q_fa3_stage_probe(
    const void* smem_raw,
    const void* q_raw,
    int tx_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128) {{
    asm volatile("trap;");
  }}

  auto const* smem = static_cast<unsigned char const*>(smem_raw);
  auto const* q = static_cast<unsigned char const*>(q_raw);
  int const smem_q_offset = tileops_fa3_shaped_smem_q_offset_bytes();
  int const rows[5] = {{0, 0, 7, 8, 127}};
  int const cols[5] = {{0, D - 1, 16, 0, D - 1}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const row = rows[i];
    int const col = cols[i];
    int const global_row = tile_m_tl * 128 + row;
    int const global_offset = ((bidb_tl * S + global_row) * H + bidh_tl) * D + col;
    int const smem_offset = smem_q_offset + tileops_fa3_shaped_swizzle_128b_index(row, col);
    mismatch = mismatch || (smem[smem_offset] != q[global_offset]);
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_k_hlir_tma_probe(
    const void* k_stage_raw,
    const void* k_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128) {{
    asm volatile("trap;");
  }}

  auto const* k_stage = static_cast<unsigned char const*>(k_stage_raw);
  auto const* k = static_cast<unsigned char const*>(k_raw);
  int const rows[5] = {{0, 0, 7, 8, 223}};
  int const cols[5] = {{0, D - 1, 16, 0, D - 1}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const row = rows[i];
    int const col = cols[i];
    int const global_row = tile_n_tl * 224 + row;
    int const global_offset = ((bidb_tl * S + global_row) * HKV + bidh_kv_tl) * D + col;
    int const smem_offset = tileops_fa3_shaped_swizzle_128b_index(row, col);
    mismatch = mismatch || (k_stage[smem_offset] != k[global_offset]);
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_v_hlir_tma_probe(
    const void* v_stage_raw,
    const void* v_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128) {{
    asm volatile("trap;");
  }}

  auto const* v_stage = static_cast<unsigned char const*>(v_stage_raw);
  auto const* v = static_cast<unsigned char const*>(v_raw);
  int const rows[5] = {{0, 0, 7, 8, 223}};
  int const cols[5] = {{0, D - 1, 16, 0, D - 1}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const row = rows[i];
    int const col = cols[i];
    int const global_row = tile_n_tl * 224 + row;
    int const global_offset = ((bidb_tl * S + global_row) * HKV + bidh_kv_tl) * D + col;
    int const smem_offset = tileops_fa3_shaped_swizzle_128b_index(row, col);
    mismatch = mismatch || (v_stage[smem_offset] != v[global_offset]);
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_kv_hlir_stage_boundary(
    int stage_kind_tl,
    const void* stage_raw,
    const void* global_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128 ||
      stage_kind_tl < 0 || stage_kind_tl > 1) {{
    asm volatile("trap;");
  }}

  auto const* stage = static_cast<unsigned char const*>(stage_raw);
  auto const* global = static_cast<unsigned char const*>(global_raw);
  int const rows[5] = {{0, 0, 7, 8, 223}};
  int const cols[5] = {{0, D - 1, 16, 0, D - 1}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const row = rows[i];
    int const col = cols[i];
    int const global_row = tile_n_tl * 224 + row;
    if (global_row < S) {{
      int const global_offset = ((bidb_tl * S + global_row) * HKV + bidh_kv_tl) * D + col;
      int const smem_offset = tileops_fa3_shaped_swizzle_128b_index(row, col);
      mismatch = mismatch || (stage[smem_offset] != global[global_offset]);
    }}
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_vt_hlir_stage_boundary(
    const void* vt_stage_raw,
    const void* v_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128) {{
    asm volatile("trap;");
  }}

  auto* vt_stage = const_cast<Element*>(static_cast<Element const*>(vt_stage_raw));
  auto const* v = static_cast<unsigned char const*>(v_raw);
  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(vt_stage), typename CollectiveMainloop::SmemLayoutVt{{}}));
  int const ds[5] = {{0, 0, 7, 16, D - 1}};
  int const ns[5] = {{0, 223, 16, 0, 111}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const d = ds[i];
    int const n = ns[i];
    int const global_row = tile_n_tl * 224 + n;
    if (global_row < S) {{
      int const global_offset = ((bidb_tl * S + global_row) * HKV + bidh_kv_tl) * D + d;
      Element actual = sVt(d, n, 0);
      unsigned char const actual_byte = reinterpret_cast<unsigned char const*>(&actual)[0];
      mismatch = mismatch || (actual_byte != v[global_offset]);
    }}
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_vt_to_v_boundary(
    void* vt_stage_raw,
    void* v_stage_raw,
    int tx_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl < 0 || tx_tl >= cutlass::NumThreadsPerWarpGroup) {{
    asm volatile("trap;");
  }}

  auto* vt_stage = static_cast<Element*>(vt_stage_raw);
  auto* v_stage = static_cast<Element*>(v_stage_raw);
  Tensor sVt = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(vt_stage), typename CollectiveMainloop::SmemLayoutVt{{}}));
  Tensor sV = cute::as_position_independent_swizzle_tensor(
      make_tensor(make_smem_ptr(v_stage), typename CollectiveMainloop::SmemLayoutVtMma{{}}));

  typename CollectiveMainloop::S2RTiledCopyVt s2r_tiled_copy_vt;
  typename CollectiveMainloop::R2STiledCopyV r2s_tiled_copy_v;
  auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(tx_tl);
  auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(tx_tl);
  Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(
      flat_divide(sVt, typename CollectiveMainloop::LDSM_divide_shape{{}}));
  Tensor tTranssV_ = r2s_thr_copy_v.partition_D(
      flat_divide(sV, typename CollectiveMainloop::STSM_divide_shape{{}}));
  CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
  CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
  CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
  CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
  CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
  CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
  static constexpr int Transpose_ILP =
      (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
  Tensor tTranssVt = logical_divide(
      group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
      Shape<Underscore, Int<Transpose_ILP>>{{}});
  Tensor tTranssV = logical_divide(
      group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
      Shape<Underscore, Int<Transpose_ILP>>{{}});

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {{
    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{{}}), _0{{}}));
    static_assert(size<0>(tTransrV) == 16);
    Tensor tTransrV_64 = recast<uint2>(tTransrV);
    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), _0{{}}), tTransrV);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size(tTransrV_64); ++j) {{
      uint32_t upper = tTransrV_64[j].x;
      uint32_t lower = tTransrV_64[j].y;
      tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
      tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
    }}
    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), _0{{}}));
  }}
  cutlass::arch::fence_view_async_shared();
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_vt_to_v_inplace_boundary(
    void* vt_v_stage_raw,
    int tx_tl) {{
  tileops_fa3_shaped_vt_to_v_boundary(vt_v_stage_raw, vt_v_stage_raw, tx_tl);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_v_mma_layout_boundary(
    const void* v_stage_raw,
    int tx_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x)) {{
    asm volatile("trap;");
  }}

  auto* v_stage = const_cast<Element*>(static_cast<Element const*>(v_stage_raw));
  Tensor sV = make_tensor(make_smem_ptr(v_stage), typename CollectiveMainloop::SmemLayoutVtMma{{}});
  typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
  auto wg_mma_pv = tiled_mma_pv.get_slice(_0{{}});
  auto tOrV = wg_mma_pv.partition_fragment_B(sV);
  (void)tOrV;
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_v_mma_hlir_stage_boundary(
    const void* v_stage_raw,
    const void* v_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128) {{
    asm volatile("trap;");
  }}

  auto* v_stage = const_cast<Element*>(static_cast<Element const*>(v_stage_raw));
  auto const* v = static_cast<unsigned char const*>(v_raw);
  Tensor sV = make_tensor(make_smem_ptr(v_stage), typename CollectiveMainloop::SmemLayoutVtMma{{}});
  typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
  auto wg_mma_pv = tiled_mma_pv.get_slice(_0{{}});
  auto tOrV = wg_mma_pv.partition_fragment_B(sV);
  (void)tOrV;

  int const ds[5] = {{0, 0, 7, 16, D - 1}};
  int const ns[5] = {{0, 223, 16, 0, 111}};
  bool mismatch = false;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 5; ++i) {{
    int const d = ds[i];
    int const n = ns[i];
    int const global_row = tile_n_tl * 224 + n;
    if (global_row < S) {{
      int const global_offset = ((bidb_tl * S + global_row) * HKV + bidh_kv_tl) * D + d;
      Element actual = sV(d, n, 0);
      unsigned char const actual_byte = reinterpret_cast<unsigned char const*>(&actual)[0];
      mismatch = mismatch || (actual_byte != v[global_offset]);
    }}
  }}
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_pv_correctness_boundary(
    const void* p_raw,
    const void* v_stage_raw,
    void* o_stage_raw,
    void* out_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) ||
      tx_tl < 0 || tx_tl >= int(size(typename CollectiveMainloop::TiledMmaPV{{}})) ||
      tile_n_tl < 0 || tile_n_tl >= (S / 224) ||
      bidh_kv_tl < 0 || bidh_kv_tl >= HKV ||
      bidb_tl < 0 || bidb_tl >= B) {{
    asm volatile("trap;");
  }}

  auto const* p = static_cast<float const*>(p_raw);
  auto* v_stage = const_cast<Element*>(static_cast<Element const*>(v_stage_raw));
  auto* o_stage = static_cast<ElementOut*>(o_stage_raw);
  auto* out = static_cast<ElementOut*>(out_raw);
  (void)o_stage;

  int const row_base = (tx_tl >> 7) * 64;
  int const p_offset =
      (((bidb_tl * (S / 224) + tile_n_tl) * HKV + bidh_kv_tl) * 128 + row_base) * 224;
  fp8_e4_t p_frag[112];
  float acc_o[64];
  float ones[64];

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 64; ++i) {{
    acc_o[i] = 0.0f;
    ones[i] = 1.0f;
  }}

  tl::fp8_pack_p_logical_fa3_raw_64x128x224(
      const_cast<float*>(p + p_offset), 224, p_frag);
  tl::fp8_pv_cute_grouped_begin_accumulate_from_p_frag_fa3_raw_64x128x224(
      p_frag,
      reinterpret_cast<fp8_e4_t*>(v_stage),
      8,
      acc_o);
  tl::fp8_fa3_raw_acc_permute_to_canonical_64x128(acc_o);

  int const out_base =
      (((bidb_tl * (S / 224) + tile_n_tl) * HKV + bidh_kv_tl) * 128 + row_base) * D;
  tl::fp8_fa3_raw_acc_store_global_64x128(acc_o, ones, 4, out + out_base, D);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_check_kv_hlir_consumer_tensor_boundary(
    int stage_kind_tl,
    const void* stage_raw,
    const void* global_raw,
    int tx_tl,
    int tile_n_tl,
    int bidh_kv_tl,
    int bidb_tl) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}
  if (tx_tl != int(threadIdx.x) || tx_tl != 128 ||
      stage_kind_tl < 0 || stage_kind_tl > 1) {{
    asm volatile("trap;");
  }}

  auto* stage = const_cast<Element*>(static_cast<Element const*>(stage_raw));
  int const mma_thread_idx = tx_tl - cutlass::NumThreadsPerWarpGroup;
  static constexpr int MmaWarpGroups =
      size(typename CollectiveMainloop::TiledMmaPV{{}}) / cutlass::NumThreadsPerWarpGroup;
  Layout warp_group_thread_layout = make_layout(
      make_shape(Int<MmaWarpGroups>{{}}),
      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{{}}));
  int const warp_group_idx = mma_thread_idx / cutlass::NumThreadsPerWarpGroup;

  if (stage_kind_tl == 0) {{
    Tensor sK = make_tensor(make_smem_ptr(stage), typename CollectiveMainloop::SmemLayoutK{{}});
    typename CollectiveMainloop::TiledMmaQK tiled_mma_qk;
    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
    auto tSrK = wg_mma_qk.partition_fragment_B(sK);
    (void)tSrK;
  }} else {{
    Tensor sV = make_tensor(make_smem_ptr(stage), typename CollectiveMainloop::SmemLayoutVtMma{{}});
    typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
    auto tOrV = wg_mma_pv.partition_fragment_B(sV);
    (void)tOrV;
  }}

  tileops_fa3_shaped_check_kv_hlir_stage_boundary(
      stage_kind_tl, stage_raw, global_raw, tx_tl, tile_n_tl, bidh_kv_tl, bidb_tl);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_producer_load_one_tile(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int producer_warp_tl,
    int producer_lane_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  using namespace cute;
  using namespace tileops_fa3_call_extern_detail;

  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  int const tx_cuda = int(threadIdx.x);
  bool const mismatch =
      role_tl != 0 ||
      tx_tl != tx_cuda ||
      tx_cuda >= 128 ||
      warpgroup_id_tl != tx_cuda / 128 ||
      producer_warp_tl != (tx_cuda / 32) % 4 ||
      producer_lane_tl != tx_cuda % 32 ||
      group_tl <= 0 ||
      bidh_kv_tl != bidh_tl / group_tl;
  if (mismatch) {{
    asm volatile("trap;");
  }}

  static constexpr int NumMmaThreads =
      AttnKernel::NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  using ClusterShape = typename AttnKernel::ClusterShape;
  using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
  using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
  using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
  using PipelineState = typename CollectiveMainloop::PipelineState;
  using PipelineParamsK = typename MainloopPipelineK::Params;
  using PipelineParamsV = typename MainloopPipelineV::Params;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

  int const params_idx = (bidb_tl * H + bidh_tl) * NUM_M_BLOCKS + tile_m_tl;
  auto const* params = reinterpret_cast<AttnKernel::Params const*>(params_storage[params_idx].bytes);
  char* smem_buf = reinterpret_cast<char*>(smem_raw);
  typename AttnKernel::SharedStorage& shared_storage =
      *reinterpret_cast<typename AttnKernel::SharedStorage*>(smem_buf);

  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
  PipelineParamsK pipeline_params_k;
  pipeline_params_k.role = MainloopPipelineK::ThreadCategory::Producer;
  pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  pipeline_params_k.is_leader = warp_group_thread_idx == 0;
  pipeline_params_k.num_consumers =
      !AttnKernel::LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;

  static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
  PipelineParamsVt pipeline_params_vt = pipeline_params_k;
  if constexpr (!AttnKernel::SameHeadDim) {{
    pipeline_params_vt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    if constexpr (AttnKernel::LargeHeadDimV) {{ pipeline_params_vt.num_consumers = NumMmaThreads; }}
  }}

  PipelineParamsV pipeline_params_v;
  pipeline_params_v.role = MainloopPipelineV::ThreadCategory::Producer;
  pipeline_params_v.producer_arv_count = AttnKernel::NumProducerThreads;
  pipeline_params_v.consumer_arv_count = NumMmaThreads;

  MainloopPipelineK pipeline_k(
      shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{{}}, cute::false_type{{}});
  MainloopPipelineV pipeline_v(
      shared_storage.pipelines.pipeline_v, pipeline_params_v, cute::false_type{{}});
  pipeline_params_vt.num_consumers = AttnKernel::NumProducerThreads;
  MainloopPipelineVt pipeline_vt(
      shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{{}}, cute::false_type{{}});

  CollectiveMainloop mainloop;
  PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
  int work_idx = 0;
  cutlass::arch::wait_on_dependent_grids();
  auto block_coord = cute::make_tuple(
      int32_t(tile_m_tl), int32_t(bidh_tl), int32_t(bidb_tl), int32_t(0));
  typename AttnKernel::SeqlenInfo_t seqlen_info{{
      get<2>(block_coord),
      get<0>(params->mainloop.shape_Q),
      !params->mainloop.ptr_pagetable ? size<0>(params->mainloop.shape_K)
                                      : size<0>(params->mainloop.shape_K) * size<1>(params->mainloop.shape_pagetable),
      get<0>(params->mainloop.shape_K_new),
      params->mainloop.cu_seqlens_q, params->mainloop.cu_seqlens_k,
      params->mainloop.cu_seqlens_k_new, params->mainloop.seqused_q,
      params->mainloop.seqused_k, params->mainloop.leftpad_k,
      params->mainloop.seqlens_rotary}};
  auto scheduler_prefetch = []() {{}};
  mainloop.load(
      params->mainloop, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write,
      shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
  mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, shared_storage, work_idx);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_producer_load_tail(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int producer_warp_tl,
    int producer_lane_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  (void)tile_m_tl;
  (void)bidb_tl;
  (void)smem_raw;
  if constexpr ({str(query_smem).lower()}) {{
    return;
  }}

  int const tx_cuda = int(threadIdx.x);
  bool const mismatch =
      role_tl != 0 ||
      tx_tl != tx_cuda ||
      tx_cuda >= 128 ||
      warpgroup_id_tl != tx_cuda / 128 ||
      producer_warp_tl != (tx_cuda / 32) % 4 ||
      producer_lane_tl != tx_cuda % 32 ||
      group_tl <= 0 ||
      bidh_kv_tl != bidh_tl / group_tl;
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_run_consumer_wg1(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  if (role_tl != 1) {{
    asm volatile("trap;");
  }}
  tileops_fa3_shaped_run_role(
      1, tx_tl, warpgroup_id_tl, tile_m_tl, bidh_tl, bidb_tl,
      bidh_kv_tl, group_tl, smem_raw);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_run_consumer_wg2(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  if (role_tl != 2) {{
    asm volatile("trap;");
  }}
  tileops_fa3_shaped_run_role(
      2, tx_tl, warpgroup_id_tl, tile_m_tl, bidh_tl, bidb_tl,
      bidh_kv_tl, group_tl, smem_raw);
}}

extern "C" __device__ __forceinline__ void tileops_fa3_shaped_role_probe(
    int role_tl,
    int tx_tl,
    int warpgroup_id_tl,
    int tile_m_tl,
    int bidh_tl,
    int bidb_tl,
    int bidh_kv_tl,
    int group_tl,
    void* smem_raw) {{
  (void)smem_raw;
  int const tx_cuda = int(threadIdx.x);
  int const role_cuda = tx_cuda < 128 ? 0 : (tx_cuda < 256 ? 1 : 2);
  bool const mismatch =
      role_tl != role_cuda ||
      tx_tl != tx_cuda ||
      warpgroup_id_tl != tx_cuda / 128 ||
      tile_m_tl != int(blockIdx.x) ||
      bidh_tl != int(blockIdx.y) ||
      bidb_tl != int(blockIdx.z) ||
      group_tl <= 0 ||
      bidh_kv_tl != int(blockIdx.y) / group_tl;
  if (mismatch) {{
    asm volatile("trap;");
  }}
}}
"""


def _write_extern_helper_header(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    stages: int,
    query_smem: bool,
    static_persistent_call_extern: bool,
) -> Path:
    TMP.mkdir(exist_ok=True)
    source = _extern_helper_header_source(
        batch,
        seq_len,
        heads,
        heads_kv,
        dim,
        stages,
        query_smem,
        static_persistent_call_extern,
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = TMP / f"tileops_fa3_shaped_extern_helper_{digest}.h"
    if not path.exists():
        path.write_text(source)
    return path


def build_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    smem_bytes: int,
    stages: int,
    query_smem: bool,
    pack_gqa: bool,
    softcap: float,
    force_rebuild_launcher: bool,
    legacy_raw_device: bool,
    host_launcher: bool,
    role_run: bool,
    static_persistent_call_extern: bool,
    validate_wg_branch: bool,
    producer_split: bool,
    producer_tma_hlir_probe: bool,
    producer_tma_hlir_q_boundary_probe: bool,
    producer_tma_hlir_q_stage_probe: bool,
    producer_tma_hlir_k_buffer_probe: bool,
    producer_tma_hlir_v_buffer_probe: bool,
    producer_tma_hlir_k_pipeline_probe: bool,
    producer_tma_hlir_v_pipeline_probe: bool,
    producer_tma_hlir_kv_boundary_probe: bool,
    producer_tma_hlir_core_shadow_probe: bool,
):
    source = _source(batch, seq_len, heads, heads_kv, dim, stages, query_smem) if legacy_raw_device else _dummy_cuda_source()
    group = heads // heads_kv
    if legacy_raw_device or host_launcher:
        num_m_blocks = (seq_len * group + 127) // 128
        grid_heads = heads_kv
    else:
        num_m_blocks = (seq_len + 127) // 128
        grid_heads = heads
    num_sms = 132
    kernel_grid_x = num_sms if static_persistent_call_extern else num_m_blocks
    kernel_grid_heads = 1 if static_persistent_call_extern else grid_heads
    kernel_grid_batch = 1 if static_persistent_call_extern else batch
    launcher_so = None
    helper_header = None
    if host_launcher:
        launcher_so = _compile_launcher(
            batch,
            seq_len,
            heads,
            heads_kv,
            dim,
            pack_gqa,
            softcap,
            force=force_rebuild_launcher,
        )
        _register_host_postproc(launcher_so)
    elif not legacy_raw_device:
        helper_header = _write_extern_helper_header(
            batch,
            seq_len,
            heads,
            heads_kv,
            dim,
            stages,
            query_smem,
            static_persistent_call_extern,
        )
    launcher_tag = launcher_so.stem.rsplit("_", 1)[-1] if launcher_so is not None else "legacy"
    helper_flag = ["-include", str(helper_header)] if helper_header is not None else []

    @tilelang.jit(
        out_idx=[6, 7],
        execution_backend="tvm_ffi",
        compile_flags=[
            "-O3",
            "-DNDEBUG",
            f"-DTILEOPS_FA3_SHAPED_LAUNCHER_TAG_{launcher_tag}",
            "-Xptxas=-v",
            "--expt-relaxed-constexpr",
            "-DENABLE_BF16",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90A_ENABLED",
            f"-I{FA3_INC}",
        ] + helper_flag,
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        descale_shape = (batch, heads_kv)

        @T.macro
        def producer_core_shadow(
            q_tensor,
            k_tensor,
            v_tensor,
            stage,
            vt_stage,
            q_full,
            q_done,
            k_full,
            k_done,
            v_full,
            v_done,
            tx: T.int32,
            tile_m: T.int32,
            bidh: T.int32,
            bidb: T.int32,
            bidh_kv: T.int32,
        ) -> None:
            q_row_base = tile_m * 128
            if q_row_base + 128 <= seq_len:
                T.tma_copy(
                    q_tensor[bidb, q_row_base:q_row_base + 128, bidh, 0:dim],
                    stage[0:128, 0:dim],
                    barrier=full,
                )
                T.barrier_arrive(full)
                T.barrier_wait(done, 0)
            if seq_len >= 224:
                for k_tile_n in T.Pipelined(seq_len // 224, num_stages=0):
                    k_phase = (k_tile_n + 1) % 2
                    T.tma_copy(
                        k_tensor[
                            bidb,
                            k_tile_n * 224:(k_tile_n + 1) * 224,
                            bidh_kv,
                            0:dim,
                        ],
                        stage,
                        barrier=full,
                    )
                    T.barrier_arrive(full)
                    T.barrier_wait(done, k_phase)
                for v_n_idx in T.Pipelined(seq_len // 224, num_stages=0):
                    if v_n_idx > 0:
                        v_tile_n = v_n_idx - 1
                        v_phase = (1 + seq_len // 224 + v_tile_n) % 2
                        T.tma_copy(
                            v_tensor[
                                bidb,
                                v_tile_n * 224:(v_tile_n + 1) * 224,
                                bidh_kv,
                                0:dim,
                            ],
                            stage,
                            barrier=full,
                        )
                        T.barrier_arrive(full)
                        T.barrier_wait(done, v_phase)
                v_tail_tile_n = seq_len // 224 - 1
                v_tail_phase = (1 + seq_len // 224 + v_tail_tile_n) % 2
                T.tma_copy(
                    v_tensor[
                        bidb,
                        v_tail_tile_n * 224:(v_tail_tile_n + 1) * 224,
                        bidh_kv,
                        0:dim,
                    ],
                    stage,
                    barrier=full,
                )
                T.barrier_arrive(full)
                T.barrier_wait(done, v_tail_phase)

        @T.macro
        def consumer_core_shadow_wg1(
            q_tensor,
            k_tensor,
            v_tensor,
            stage,
            full,
            done,
            tx: T.int32,
            tile_m: T.int32,
            bidh: T.int32,
            bidb: T.int32,
            bidh_kv: T.int32,
        ) -> None:
            q_row_base = tile_m * 128
            if q_row_base + 128 <= seq_len:
                T.barrier_wait(full, 0)
                if tx == 128:
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary",
                        stage.access_ptr("r"),
                        q_tensor.data,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                    )
                T.barrier_arrive(done)
            if seq_len >= 224:
                for k_tile_n in T.Pipelined(seq_len // 224, num_stages=0):
                    k_phase = (k_tile_n + 1) % 2
                    T.barrier_wait(full, k_phase)
                    if tx == 128:
                        T.call_extern(
                            "handle",
                            "tileops_fa3_shaped_check_kv_hlir_consumer_tensor_boundary",
                            0,
                            stage.access_ptr("r"),
                            k_tensor.data,
                            tx,
                            k_tile_n,
                            bidh_kv,
                            bidb,
                        )
                    T.barrier_arrive(done)
                for v_tile_n in T.Pipelined(seq_len // 224, num_stages=0):
                    v_phase = (1 + seq_len // 224 + v_tile_n) % 2
                    T.barrier_wait(full, v_phase)
                    if tx == 128:
                        T.call_extern(
                            "handle",
                            "tileops_fa3_shaped_check_kv_hlir_consumer_tensor_boundary",
                            1,
                            stage.access_ptr("r"),
                            v_tensor.data,
                            tx,
                            v_tile_n,
                            bidh_kv,
                            bidb,
                        )
                    T.barrier_arrive(done)

        @T.macro
        def consumer_core_shadow_wg2(
            q_tensor,
            stage,
            full,
            done,
            tx: T.int32,
            tile_m: T.int32,
            bidh: T.int32,
            bidb: T.int32,
        ) -> None:
            q_row_base = tile_m * 128
            if q_row_base + 128 <= seq_len:
                T.barrier_wait(full, 0)
                if tx == 256:
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary",
                        stage.access_ptr("r"),
                        q_tensor.data,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                    )
                T.barrier_arrive(done)
            if seq_len >= 224:
                for k_tile_n in T.Pipelined(seq_len // 224, num_stages=0):
                    T.barrier_wait(full, (k_tile_n + 1) % 2)
                    T.barrier_arrive(done)
                for v_tile_n in T.Pipelined(seq_len // 224, num_stages=0):
                    T.barrier_wait(full, (1 + seq_len // 224 + v_tile_n) % 2)
                    T.barrier_arrive(done)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_descale: T.Tensor(descale_shape, "float"),
            k_descale: T.Tensor(descale_shape, "float"),
            v_descale: T.Tensor(descale_shape, "float"),
            output: T.Tensor(q_shape, "bfloat16"),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            if legacy_raw_device:
                # The external CUDA body owns the actual layout; this allocation
                # only teaches TileLang's launch wrapper to pass dynamic smem
                # for FA3's extern shared storage.
                _cuda_source_kernel_with_smem(
                    num_m_blocks,
                    heads_kv,
                    batch,
                    threads=384,
                    source_code_or_path=source,
                    entry_name="tileops_fa3_shaped_shell",
                    dynamic_smem_bytes=smem_bytes,
                )
            elif host_launcher:
                T.CUDASourceCodeKernel(
                    num_m_blocks,
                    heads_kv,
                    batch,
                    threads=384,
                    source_code_or_path=source,
                    entry_name="tileops_fa3_shaped_shell",
                )
            else:
                with T.Kernel(kernel_grid_x, kernel_grid_heads, kernel_grid_batch, threads=384) as (_bx, _by, _bz):
                    smem = T.alloc_shared((smem_bytes,), "uint8")
                    if producer_tma_hlir_probe:
                        q_hlir_probe = T.alloc_shared((128, dim), "float8_e4m3fn")
                        q_hlir_probe_ready = T.alloc_barrier(arrive_count=128)
                    if producer_tma_hlir_q_boundary_probe:
                        q_hlir_boundary_probe = T.alloc_shared((128, dim), "float8_e4m3fn")
                        q_hlir_boundary_ready = T.alloc_barrier(arrive_count=128)
                    if producer_tma_hlir_k_buffer_probe:
                        k_hlir_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
                        k_hlir_probe_ready = T.alloc_barrier(arrive_count=128)
                    if producer_tma_hlir_k_pipeline_probe:
                        k_hlir_pipe_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
                        k_hlir_pipe_full = T.alloc_barrier(arrive_count=128)
                        k_hlir_pipe_empty = T.alloc_barrier(arrive_count=256)
                    if producer_tma_hlir_v_buffer_probe:
                        v_hlir_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
                        v_hlir_probe_ready = T.alloc_barrier(arrive_count=128)
                    if producer_tma_hlir_v_pipeline_probe:
                        v_hlir_pipe_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
                        v_hlir_pipe_full = T.alloc_barrier(arrive_count=128)
                        v_hlir_pipe_empty = T.alloc_barrier(arrive_count=256)
                    if producer_tma_hlir_kv_boundary_probe:
                        kv_hlir_boundary_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
                        kv_hlir_boundary_full = T.alloc_barrier(arrive_count=128)
                        kv_hlir_boundary_done = T.alloc_barrier(arrive_count=256)
                    if producer_tma_hlir_core_shadow_probe:
                        core_shadow_stage = T.alloc_shared((224, dim), "float8_e4m3fn")
                        core_shadow_full = T.alloc_barrier(arrive_count=128)
                        core_shadow_done = T.alloc_barrier(arrive_count=256)
                    if producer_tma_hlir_q_stage_probe:
                        q_fa3_stage_offset = T.call_extern(
                            "int32",
                            "tileops_fa3_shaped_smem_q_offset_bytes",
                        )
                        smem_fp8_view = T.view(
                            smem,
                            (smem_bytes // dim, dim),
                            dtype="float8_e4m3fn",
                        )
                        q_fa3_stage_row_offset = q_fa3_stage_offset // dim
                        q_fa3_stage_view = T.match_buffer(
                            smem_fp8_view[q_fa3_stage_row_offset:q_fa3_stage_row_offset + 128, 0:dim],
                            (128, dim),
                            "float8_e4m3fn",
                            scope="shared.dyn",
                        )
                        q_fa3_stage_ready = T.alloc_barrier(arrive_count=128)
                    tx = T.get_thread_binding()
                    lane_id = tx % 32
                    warp_id = tx // 32
                    warpgroup_id = tx // 128
                    warpgroup_lane = tx % 128
                    producer_warp = warpgroup_lane // 32
                    producer_lane = warpgroup_lane % 32
                    tile_m = _bx
                    bidh = _by
                    bidb = _bz
                    bidh_kv = bidh // group
                    T.reads(
                        q[0:batch, 0:seq_len, 0:heads, 0:dim],
                        k[0:batch, 0:seq_len, 0:heads_kv, 0:dim],
                        v[0:batch, 0:seq_len, 0:heads_kv, 0:dim],
                        q_descale[0:batch, 0:heads_kv],
                        k_descale[0:batch, 0:heads_kv],
                        v_descale[0:batch, 0:heads_kv],
                    )
                    if producer_tma_hlir_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            q_hlir_probe[0:128, 0:dim],
                        )
                        T.annotate_layout({
                            q_hlir_probe: tilelang.layout.make_swizzled_layout(q_hlir_probe),
                        })
                    elif producer_tma_hlir_q_boundary_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            q_hlir_boundary_probe[0:128, 0:dim],
                        )
                        T.annotate_layout({
                            q_hlir_boundary_probe: tilelang.layout.make_swizzled_layout(q_hlir_boundary_probe),
                        })
                    elif producer_tma_hlir_k_buffer_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            k_hlir_probe[0:224, 0:dim],
                        )
                        T.annotate_layout({
                            k_hlir_probe: tilelang.layout.make_swizzled_layout(k_hlir_probe),
                        })
                    elif producer_tma_hlir_k_pipeline_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            k_hlir_pipe_probe[0:224, 0:dim],
                        )
                        T.annotate_layout({
                            k_hlir_pipe_probe: tilelang.layout.make_swizzled_layout(k_hlir_pipe_probe),
                        })
                    elif producer_tma_hlir_v_buffer_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            v_hlir_probe[0:224, 0:dim],
                        )
                        T.annotate_layout({
                            v_hlir_probe: tilelang.layout.make_swizzled_layout(v_hlir_probe),
                        })
                    elif producer_tma_hlir_v_pipeline_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            v_hlir_pipe_probe[0:224, 0:dim],
                        )
                        T.annotate_layout({
                            v_hlir_pipe_probe: tilelang.layout.make_swizzled_layout(v_hlir_pipe_probe),
                        })
                    elif producer_tma_hlir_kv_boundary_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            kv_hlir_boundary_probe[0:224, 0:dim],
                        )
                        T.annotate_layout({
                            kv_hlir_boundary_probe: tilelang.layout.make_swizzled_layout(kv_hlir_boundary_probe),
                        })
                    elif producer_tma_hlir_core_shadow_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                            core_shadow_stage[0:224, 0:dim],
                        )
                        T.annotate_layout({
                            core_shadow_stage: tilelang.layout.make_swizzled_layout(core_shadow_stage),
                        })
                    elif producer_tma_hlir_q_stage_probe:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                        )
                        T.annotate_layout({
                            q_fa3_stage_view: tilelang.layout.make_swizzled_layout(q_fa3_stage_view),
                        })
                    else:
                        T.writes(
                            output[0:batch, 0:seq_len, 0:heads, 0:dim],
                            lse[0:batch, 0:heads, 0:seq_len],
                            smem[0:smem_bytes],
                        )
                    q_tma_desc = T.create_tma_descriptor(
                        0, 4, q.data,
                        dim, seq_len, heads, batch,
                        1, heads * dim, dim, seq_len * heads * dim,
                        dim, 128, 1, 1,
                        1, 1, 1, 1,
                        0, 3, 2, 0,
                    )
                    k_tma_desc = T.create_tma_descriptor(
                        0, 4, k.data,
                        dim, seq_len, heads_kv, batch,
                        1, heads_kv * dim, dim, seq_len * heads_kv * dim,
                        dim, 224, 1, 1,
                        1, 1, 1, 1,
                        0, 3, 2, 0,
                    )
                    v_tma_desc = T.create_tma_descriptor(
                        0, 4, v.data,
                        dim, seq_len, heads_kv, batch,
                        1, heads_kv * dim, dim, seq_len * heads_kv * dim,
                        dim, 224, 1, 1,
                        1, 1, 1, 1,
                        0, 3, 2, 0,
                    )
                    output_tma_desc = T.create_tma_descriptor(
                        9, 5, output.data,
                        dim, seq_len, heads, batch, 1,
                        2, heads * dim * 2, dim * 2, seq_len * heads * dim * 2,
                        batch * seq_len * heads * dim * 2,
                        64, 128, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        0, 3, 2, 0,
                    )
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_prepare_params",
                        q_tma_desc,
                        k_tma_desc,
                        v_tma_desc,
                        output_tma_desc,
                        q.data,
                        k.data,
                        v.data,
                        q_descale.data,
                        k_descale.data,
                        v_descale.data,
                        output.data,
                        lse.data,
                        tx,
                        lane_id,
                        warp_id,
                        warpgroup_id,
                        warpgroup_lane,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                        smem.access_ptr("rw"),
                    )
                    if role_run:
                        T.call_extern(
                            "handle",
                            "tileops_fa3_shaped_prepare_runtime",
                            tile_m,
                            bidh,
                            bidb,
                            smem.access_ptr("rw"),
                        )
                    if role_run:
                        if tx < 128:
                            if producer_split:
                                if producer_tma_hlir_probe:
                                    q_row_base_producer = tile_m * 128
                                    if q_row_base_producer + 128 <= seq_len:
                                        T.tma_copy(
                                            q[bidb, q_row_base_producer:q_row_base_producer + 128, bidh, 0:dim],
                                            q_hlir_probe,
                                            barrier=q_hlir_probe_ready,
                                        )
                                        T.barrier_arrive(q_hlir_probe_ready)
                                if producer_tma_hlir_q_boundary_probe:
                                    q_boundary_row_base_producer = tile_m * 128
                                    if q_boundary_row_base_producer + 128 <= seq_len:
                                        T.tma_copy(
                                            q[
                                                bidb,
                                                q_boundary_row_base_producer:q_boundary_row_base_producer + 128,
                                                bidh,
                                                0:dim,
                                            ],
                                            q_hlir_boundary_probe,
                                            barrier=q_hlir_boundary_ready,
                                        )
                                        T.barrier_arrive(q_hlir_boundary_ready)
                                if producer_tma_hlir_k_buffer_probe:
                                    if seq_len >= 224:
                                        T.tma_copy(
                                            k[bidb, 0:224, bidh_kv, 0:dim],
                                            k_hlir_probe,
                                            barrier=k_hlir_probe_ready,
                                        )
                                        T.barrier_arrive(k_hlir_probe_ready)
                                if producer_tma_hlir_k_pipeline_probe:
                                    if seq_len >= 224:
                                        for k_pipe_tile_n in T.Pipelined(seq_len // 224, num_stages=0):
                                            T.barrier_wait(k_hlir_pipe_empty, (k_pipe_tile_n + 1) % 2)
                                            T.tma_copy(
                                                k[
                                                    bidb,
                                                    k_pipe_tile_n * 224:(k_pipe_tile_n + 1) * 224,
                                                    bidh_kv,
                                                    0:dim,
                                                ],
                                                k_hlir_pipe_probe,
                                                barrier=k_hlir_pipe_full,
                                            )
                                            T.barrier_arrive(k_hlir_pipe_full)
                                if producer_tma_hlir_v_buffer_probe:
                                    if seq_len >= 224:
                                        T.tma_copy(
                                            v[bidb, 0:224, bidh_kv, 0:dim],
                                            v_hlir_probe,
                                            barrier=v_hlir_probe_ready,
                                        )
                                        T.barrier_arrive(v_hlir_probe_ready)
                                if producer_tma_hlir_v_pipeline_probe:
                                    if seq_len >= 224:
                                        for v_pipe_n_idx in T.Pipelined(seq_len // 224, num_stages=0):
                                            if v_pipe_n_idx > 0:
                                                v_pipe_tile_n = v_pipe_n_idx - 1
                                                T.barrier_wait(v_hlir_pipe_empty, (v_pipe_tile_n + 1) % 2)
                                                T.tma_copy(
                                                    v[
                                                        bidb,
                                                        v_pipe_tile_n * 224:(v_pipe_tile_n + 1) * 224,
                                                        bidh_kv,
                                                        0:dim,
                                                    ],
                                                    v_hlir_pipe_probe,
                                                    barrier=v_hlir_pipe_full,
                                                )
                                                T.barrier_arrive(v_hlir_pipe_full)
                                        v_pipe_tail_tile_n = seq_len // 224 - 1
                                        T.barrier_wait(v_hlir_pipe_empty, (v_pipe_tail_tile_n + 1) % 2)
                                        T.tma_copy(
                                            v[
                                                bidb,
                                                v_pipe_tail_tile_n * 224:(v_pipe_tail_tile_n + 1) * 224,
                                                bidh_kv,
                                                0:dim,
                                            ],
                                            v_hlir_pipe_probe,
                                            barrier=v_hlir_pipe_full,
                                        )
                                        T.barrier_arrive(v_hlir_pipe_full)
                                if producer_tma_hlir_kv_boundary_probe:
                                    if seq_len >= 224:
                                        T.tma_copy(
                                            k[bidb, 0:224, bidh_kv, 0:dim],
                                            kv_hlir_boundary_probe,
                                            barrier=kv_hlir_boundary_full,
                                        )
                                        T.barrier_arrive(kv_hlir_boundary_full)
                                        T.barrier_wait(kv_hlir_boundary_done, 0)
                                        T.tma_copy(
                                            v[bidb, 0:224, bidh_kv, 0:dim],
                                            kv_hlir_boundary_probe,
                                            barrier=kv_hlir_boundary_full,
                                        )
                                        T.barrier_arrive(kv_hlir_boundary_full)
                                        T.barrier_wait(kv_hlir_boundary_done, 1)
                                if producer_tma_hlir_core_shadow_probe:
                                    producer_core_shadow(
                                        q,
                                        k,
                                        v,
                                        core_shadow_stage,
                                        core_shadow_full,
                                        core_shadow_done,
                                        tile_m,
                                        bidh,
                                        bidb,
                                        bidh_kv,
                                    )
                                if producer_tma_hlir_q_stage_probe:
                                    q_row_base_stage_producer = tile_m * 128
                                    if q_row_base_stage_producer + 128 <= seq_len:
                                        T.tma_copy(
                                            q[bidb, q_row_base_stage_producer:q_row_base_stage_producer + 128, bidh, 0:dim],
                                            q_fa3_stage_view,
                                            barrier=q_fa3_stage_ready,
                                        )
                                        T.barrier_arrive(q_fa3_stage_ready)
                                T.dec_max_nreg(24)
                                T.call_extern(
                                    "handle",
                                    "tileops_fa3_shaped_producer_load_one_tile",
                                    0,
                                    tx,
                                    warpgroup_id,
                                    producer_warp,
                                    producer_lane,
                                    tile_m,
                                    bidh,
                                    bidb,
                                    bidh_kv,
                                    group,
                                    smem.access_ptr("rw"),
                                )
                                T.call_extern(
                                    "handle",
                                    "tileops_fa3_shaped_producer_load_tail",
                                    0,
                                    tx,
                                    warpgroup_id,
                                    producer_warp,
                                    producer_lane,
                                    tile_m,
                                    bidh,
                                    bidb,
                                    bidh_kv,
                                    group,
                                    smem.access_ptr("rw"),
                                )
                            else:
                                T.call_extern(
                                    "handle",
                                    "tileops_fa3_shaped_run_producer",
                                    0,
                                    tx,
                                    warpgroup_id,
                                    producer_warp,
                                    producer_lane,
                                    tile_m,
                                    bidh,
                                    bidb,
                                    bidh_kv,
                                    group,
                                smem.access_ptr("rw"),
                            )
                        elif tx < 256:
                            if producer_tma_hlir_probe:
                                q_row_base_consumer_1 = tile_m * 128
                                if q_row_base_consumer_1 + 128 <= seq_len:
                                    T.barrier_wait(q_hlir_probe_ready, 0)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_q_hlir_tma_probe",
                                            q_hlir_probe.access_ptr("r"),
                                            q.data,
                                            tx,
                                            tile_m,
                                            bidh,
                                            bidb,
                                        )
                            if producer_tma_hlir_q_boundary_probe:
                                q_boundary_row_base_consumer_1 = tile_m * 128
                                if q_boundary_row_base_consumer_1 + 128 <= seq_len:
                                    T.barrier_wait(q_hlir_boundary_ready, 0)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary",
                                            q_hlir_boundary_probe.access_ptr("r"),
                                            q.data,
                                            tx,
                                            tile_m,
                                            bidh,
                                            bidb,
                                        )
                            if producer_tma_hlir_k_buffer_probe:
                                if seq_len >= 224:
                                    T.barrier_wait(k_hlir_probe_ready, 0)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_k_hlir_tma_probe",
                                            k_hlir_probe.access_ptr("r"),
                                            k.data,
                                            tx,
                                            0,
                                            bidh_kv,
                                            bidb,
                                        )
                            if producer_tma_hlir_k_pipeline_probe:
                                if seq_len >= 224:
                                    for k_pipe_tile_n_consumer_1 in T.Pipelined(seq_len // 224, num_stages=0):
                                        T.barrier_wait(k_hlir_pipe_full, k_pipe_tile_n_consumer_1 % 2)
                                        if tx == 128:
                                            T.call_extern(
                                                "handle",
                                                "tileops_fa3_shaped_check_k_hlir_tma_probe",
                                                k_hlir_pipe_probe.access_ptr("r"),
                                                k.data,
                                                tx,
                                                k_pipe_tile_n_consumer_1,
                                                bidh_kv,
                                                bidb,
                                            )
                                        T.barrier_arrive(k_hlir_pipe_empty)
                            if producer_tma_hlir_v_buffer_probe:
                                if seq_len >= 224:
                                    T.barrier_wait(v_hlir_probe_ready, 0)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_v_hlir_tma_probe",
                                            v_hlir_probe.access_ptr("r"),
                                            v.data,
                                            tx,
                                            0,
                                            bidh_kv,
                                            bidb,
                                        )
                            if producer_tma_hlir_v_pipeline_probe:
                                if seq_len >= 224:
                                    for v_pipe_tile_n_consumer_1 in T.Pipelined(seq_len // 224, num_stages=0):
                                        T.barrier_wait(v_hlir_pipe_full, v_pipe_tile_n_consumer_1 % 2)
                                        if tx == 128:
                                            T.call_extern(
                                                "handle",
                                                "tileops_fa3_shaped_check_v_hlir_tma_probe",
                                                v_hlir_pipe_probe.access_ptr("r"),
                                                v.data,
                                                tx,
                                                v_pipe_tile_n_consumer_1,
                                                bidh_kv,
                                                bidb,
                                            )
                                        T.barrier_arrive(v_hlir_pipe_empty)
                            if producer_tma_hlir_kv_boundary_probe:
                                if seq_len >= 224:
                                    T.barrier_wait(kv_hlir_boundary_full, 0)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_kv_hlir_consumer_tensor_boundary",
                                            0,
                                            kv_hlir_boundary_probe.access_ptr("r"),
                                            k.data,
                                            tx,
                                            0,
                                            bidh_kv,
                                            bidb,
                                        )
                                    T.barrier_arrive(kv_hlir_boundary_done)
                                    T.barrier_wait(kv_hlir_boundary_full, 1)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_kv_hlir_consumer_tensor_boundary",
                                            1,
                                            kv_hlir_boundary_probe.access_ptr("r"),
                                            v.data,
                                            tx,
                                            0,
                                            bidh_kv,
                                            bidb,
                                        )
                                    T.barrier_arrive(kv_hlir_boundary_done)
                            if producer_tma_hlir_core_shadow_probe:
                                consumer_core_shadow_wg1(
                                    q,
                                    k,
                                    v,
                                    core_shadow_stage,
                                    core_shadow_full,
                                    core_shadow_done,
                                    tx,
                                    tile_m,
                                    bidh,
                                    bidb,
                                    bidh_kv,
                                )
                            if producer_tma_hlir_q_stage_probe:
                                q_row_base_stage_consumer_1 = tile_m * 128
                                if q_row_base_stage_consumer_1 + 128 <= seq_len:
                                    T.barrier_wait(q_fa3_stage_ready, 0)
                                    if tx == 128:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_q_fa3_stage_probe",
                                            smem.access_ptr("r"),
                                            q.data,
                                            tx,
                                            tile_m,
                                            bidh,
                                            bidb,
                                        )
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_run_consumer_wg1",
                                1,
                                tx,
                                warpgroup_id,
                                tile_m,
                                bidh,
                                bidb,
                                bidh_kv,
                                group,
                                smem.access_ptr("rw"),
                            )
                        else:
                            if producer_tma_hlir_probe:
                                q_row_base_consumer_2 = tile_m * 128
                                if q_row_base_consumer_2 + 128 <= seq_len:
                                    T.barrier_wait(q_hlir_probe_ready, 0)
                            if producer_tma_hlir_q_boundary_probe:
                                q_boundary_row_base_consumer_2 = tile_m * 128
                                if q_boundary_row_base_consumer_2 + 128 <= seq_len:
                                    T.barrier_wait(q_hlir_boundary_ready, 0)
                                    if tx == 256:
                                        T.call_extern(
                                            "handle",
                                            "tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary",
                                            q_hlir_boundary_probe.access_ptr("r"),
                                            q.data,
                                            tx,
                                            tile_m,
                                            bidh,
                                            bidb,
                                        )
                            if producer_tma_hlir_k_buffer_probe:
                                if seq_len >= 224:
                                    T.barrier_wait(k_hlir_probe_ready, 0)
                            if producer_tma_hlir_k_pipeline_probe:
                                if seq_len >= 224:
                                    for k_pipe_tile_n_consumer_2 in T.Pipelined(seq_len // 224, num_stages=0):
                                        T.barrier_wait(k_hlir_pipe_full, k_pipe_tile_n_consumer_2 % 2)
                                        T.barrier_arrive(k_hlir_pipe_empty)
                            if producer_tma_hlir_v_buffer_probe:
                                if seq_len >= 224:
                                    T.barrier_wait(v_hlir_probe_ready, 0)
                            if producer_tma_hlir_v_pipeline_probe:
                                if seq_len >= 224:
                                    for v_pipe_tile_n_consumer_2 in T.Pipelined(seq_len // 224, num_stages=0):
                                        T.barrier_wait(v_hlir_pipe_full, v_pipe_tile_n_consumer_2 % 2)
                                        T.barrier_arrive(v_hlir_pipe_empty)
                            if producer_tma_hlir_kv_boundary_probe:
                                if seq_len >= 224:
                                    T.barrier_wait(kv_hlir_boundary_full, 0)
                                    T.barrier_arrive(kv_hlir_boundary_done)
                                    T.barrier_wait(kv_hlir_boundary_full, 1)
                                    T.barrier_arrive(kv_hlir_boundary_done)
                            if producer_tma_hlir_core_shadow_probe:
                                consumer_core_shadow_wg2(
                                    q,
                                    core_shadow_stage,
                                    core_shadow_full,
                                    core_shadow_done,
                                    tx,
                                    tile_m,
                                    bidh,
                                    bidb,
                                )
                            if producer_tma_hlir_q_stage_probe:
                                q_row_base_stage_consumer_2 = tile_m * 128
                                if q_row_base_stage_consumer_2 + 128 <= seq_len:
                                    T.barrier_wait(q_fa3_stage_ready, 0)
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_run_consumer_wg2",
                                2,
                                tx,
                                warpgroup_id,
                                tile_m,
                                bidh,
                                bidb,
                                bidh_kv,
                                group,
                                smem.access_ptr("rw"),
                            )
                    if validate_wg_branch:
                        if tx < 128:
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_role_probe",
                                0,
                                tx,
                                warpgroup_id,
                                tile_m,
                                bidh,
                                bidb,
                                bidh_kv,
                                group,
                                smem.access_ptr("rw"),
                            )
                        elif tx < 256:
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_role_probe",
                                1,
                                tx,
                                warpgroup_id,
                                tile_m,
                                bidh,
                                bidb,
                                bidh_kv,
                                group,
                                smem.access_ptr("rw"),
                            )
                        else:
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_role_probe",
                                2,
                                tx,
                                warpgroup_id,
                                tile_m,
                                bidh,
                                bidb,
                                bidh_kv,
                                group,
                                smem.access_ptr("rw"),
                            )
                    if not role_run:
                        T.call_extern(
                            "handle",
                            "tileops_fa3_shaped_run_prepared",
                            tile_m,
                            bidh,
                            bidb,
                            smem.access_ptr("rw"),
                        )

        return main

    return func(), launcher_so


def build_core_shadow_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    smem_bytes: int,
    stages: int,
    query_smem: bool,
    static_persistent_call_extern: bool,
    core_shadow_kind: str,
):
    group = heads // heads_kv
    num_m_blocks = (seq_len + 127) // 128
    shadow_k = core_shadow_kind in ("qk", "qkv", "qkvv")
    shadow_v = core_shadow_kind in ("qkv", "qkvv")
    shadow_v_to_v = core_shadow_kind == "qkvv"
    helper_header = _write_extern_helper_header(
        batch,
        seq_len,
        heads,
        heads_kv,
        dim,
        stages,
        query_smem,
        static_persistent_call_extern,
    )

    @tilelang.jit(
        out_idx=[6, 7],
        execution_backend="tvm_ffi",
        compile_flags=[
            "-O3",
            "-DNDEBUG",
            "-DTILEOPS_FA3_SHAPED_LAUNCHER_TAG_core_shadow",
            "-Xptxas=-v",
            "--expt-relaxed-constexpr",
            "-DENABLE_BF16",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90A_ENABLED",
            f"-I{FA3_INC}",
            "-include",
            str(helper_header),
        ],
    )
    def func():
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        descale_shape = (batch, heads_kv)

        @T.macro
        def producer_core_shadow(
            q_tensor,
            k_tensor,
            v_tensor,
            stage,
            vt_stage,
            q_full,
            q_done,
            k_full,
            k_done,
            v_full,
            v_transform_full,
            v_done,
            tx: T.int32,
            tile_m: T.int32,
            bidh: T.int32,
            bidb: T.int32,
            bidh_kv: T.int32,
        ) -> None:
            q_row_base = tile_m * 128
            if q_row_base + 128 <= seq_len:
                T.tma_copy(
                    q_tensor[bidb, q_row_base:q_row_base + 128, bidh, 0:dim],
                    stage[0:128, 0:dim],
                    barrier=q_full,
                )
                T.barrier_arrive(q_full)
                T.barrier_wait(q_done, 0)

            if shadow_k and seq_len >= 224:
                for k_iter in T.Pipelined(T.ceildiv(seq_len, 224), num_stages=0):
                    k_tile_n = T.ceildiv(seq_len, 224) - 1 - k_iter
                    k_phase = k_iter % 2
                    T.tma_copy(
                        k_tensor[
                            bidb,
                            k_tile_n * 224:(k_tile_n + 1) * 224,
                            bidh_kv,
                            0:dim,
                        ],
                        stage,
                        barrier=k_full,
                    )
                    T.barrier_arrive(k_full)
                    T.barrier_wait(k_done, k_phase)
            if shadow_v and seq_len >= 224:
                vt_tma_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8, 4, v_tensor.data,
                    dim, heads_kv, seq_len, batch,
                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                    dim, 1, 224, 1,
                    1, 1, 1, 1,
                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                )
                for v_iter in T.Pipelined(T.ceildiv(seq_len, 224), num_stages=0):
                    v_tile_n = T.ceildiv(seq_len, 224) - 1 - v_iter
                    v_phase = v_iter % 2
                    if tx == 0:
                        T.mbarrier_expect_tx(v_full, dim * 224)
                        T.tma_load(
                            vt_tma_desc,
                            v_full[0],
                            T.access_ptr(vt_stage, "w"),
                            0,
                            bidh_kv,
                            v_tile_n * 224,
                            bidb,
                            EVICT_NORMAL,
                        )
                    T.barrier_arrive(v_full)
                    if shadow_v_to_v:
                        T.barrier_wait(v_full, v_phase)
                        T.call_extern(
                            "handle",
                            "tileops_fa3_shaped_vt_to_v_inplace_boundary",
                            vt_stage.access_ptr("rw"),
                            tx,
                        )
                        T.barrier_arrive(v_transform_full)
                    T.barrier_wait(v_done, v_phase)

        @T.macro
        def consumer_core_shadow_wg1(
            q_tensor,
            k_tensor,
            v_tensor,
            stage,
            vt_stage,
            q_full,
            q_done,
            k_full,
            k_done,
            v_full,
            v_transform_full,
            v_done,
            tx: T.int32,
            tile_m: T.int32,
            bidh: T.int32,
            bidb: T.int32,
            bidh_kv: T.int32,
        ) -> None:
            q_row_base = tile_m * 128
            if q_row_base + 128 <= seq_len:
                T.barrier_wait(q_full, 0)
                if tx == 128:
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary",
                        stage.access_ptr("r"),
                        q_tensor.data,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                    )
                T.barrier_arrive(q_done)

            if shadow_k and seq_len >= 224:
                for k_iter in T.Pipelined(T.ceildiv(seq_len, 224), num_stages=0):
                    k_tile_n = T.ceildiv(seq_len, 224) - 1 - k_iter
                    k_phase = k_iter % 2
                    T.barrier_wait(k_full, k_phase)
                    if tx == 128:
                        T.call_extern(
                            "handle",
                            "tileops_fa3_shaped_check_kv_hlir_consumer_tensor_boundary",
                            0,
                            stage.access_ptr("r"),
                            k_tensor.data,
                            tx,
                            k_tile_n,
                            bidh_kv,
                            bidb,
                        )
                    T.barrier_arrive(k_done)
            if shadow_v and seq_len >= 224:
                for v_iter in T.Pipelined(T.ceildiv(seq_len, 224), num_stages=0):
                    v_tile_n = T.ceildiv(seq_len, 224) - 1 - v_iter
                    v_phase = v_iter % 2
                    if shadow_v_to_v:
                        T.barrier_wait(v_transform_full, v_phase)
                    else:
                        T.barrier_wait(v_full, v_phase)
                    if tx == 128:
                        if shadow_v_to_v:
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_check_v_mma_hlir_stage_boundary",
                                vt_stage.access_ptr("r"),
                                v_tensor.data,
                                tx,
                                v_tile_n,
                                bidh_kv,
                                bidb,
                            )
                        else:
                            T.call_extern(
                                "handle",
                                "tileops_fa3_shaped_check_vt_hlir_stage_boundary",
                                vt_stage.access_ptr("r"),
                                v_tensor.data,
                                tx,
                                v_tile_n,
                                bidh_kv,
                                bidb,
                            )
                    T.barrier_arrive(v_done)

        @T.macro
        def consumer_core_shadow_wg2(
            q_tensor,
            stage,
            q_full,
            q_done,
            k_full,
            k_done,
            v_full,
            v_transform_full,
            v_done,
            tx: T.int32,
            tile_m: T.int32,
            bidh: T.int32,
            bidb: T.int32,
        ) -> None:
            q_row_base = tile_m * 128
            if q_row_base + 128 <= seq_len:
                T.barrier_wait(q_full, 0)
                if tx == 256:
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary",
                        stage.access_ptr("r"),
                        q_tensor.data,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                    )
                T.barrier_arrive(q_done)

            if shadow_k and seq_len >= 224:
                for k_iter in T.Pipelined(T.ceildiv(seq_len, 224), num_stages=0):
                    T.barrier_wait(k_full, k_iter % 2)
                    T.barrier_arrive(k_done)

            if shadow_v and seq_len >= 224:
                for v_iter in T.Pipelined(T.ceildiv(seq_len, 224), num_stages=0):
                    if shadow_v_to_v:
                        T.barrier_wait(v_transform_full, v_iter % 2)
                    else:
                        T.barrier_wait(v_full, v_iter % 2)
                    T.barrier_arrive(v_done)

        @T.prim_func
        def main(
            q: T.Tensor(q_shape, "float8_e4m3fn"),
            k: T.Tensor(kv_shape, "float8_e4m3fn"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            q_descale: T.Tensor(descale_shape, "float"),
            k_descale: T.Tensor(descale_shape, "float"),
            v_descale: T.Tensor(descale_shape, "float"),
            output: T.Tensor(q_shape, "bfloat16"),
            lse: T.Tensor([batch, heads, seq_len], "float"),
        ) -> None:
            with T.Kernel(num_m_blocks, heads, batch, threads=384) as (_bx, _by, _bz):
                smem = T.alloc_shared((smem_bytes,), "uint8")
                core_shadow_stage = T.alloc_shared((224, dim), "float8_e4m3fn")
                core_shadow_vt_stage = T.view(
                    core_shadow_stage,
                    (dim, 224),
                    dtype=T.float8_e4m3fn,
                )
                core_shadow_q_full = T.alloc_barrier(arrive_count=128)
                core_shadow_q_done = T.alloc_barrier(arrive_count=256)
                core_shadow_k_full = T.alloc_barrier(arrive_count=128)
                core_shadow_k_done = T.alloc_barrier(arrive_count=256)
                core_shadow_v_full = T.alloc_barrier(arrive_count=128)
                core_shadow_v_transform_full = T.alloc_barrier(arrive_count=128)
                core_shadow_v_done = T.alloc_barrier(arrive_count=256)

                tx = T.get_thread_binding()
                lane_id = tx % 32
                warp_id = tx // 32
                warpgroup_id = tx // 128
                warpgroup_lane = tx % 128
                producer_warp = warpgroup_lane // 32
                producer_lane = warpgroup_lane % 32
                tile_m = _bx
                bidh = _by
                bidb = _bz
                bidh_kv = bidh // group

                T.reads(
                    q[0:batch, 0:seq_len, 0:heads, 0:dim],
                    k[0:batch, 0:seq_len, 0:heads_kv, 0:dim],
                    v[0:batch, 0:seq_len, 0:heads_kv, 0:dim],
                    q_descale[0:batch, 0:heads_kv],
                    k_descale[0:batch, 0:heads_kv],
                    v_descale[0:batch, 0:heads_kv],
                )
                T.writes(
                    output[0:batch, 0:seq_len, 0:heads, 0:dim],
                    lse[0:batch, 0:heads, 0:seq_len],
                    smem[0:smem_bytes],
                    core_shadow_stage[0:224, 0:dim],
                    core_shadow_vt_stage[0:dim, 0:224],
                )
                T.annotate_layout({
                    core_shadow_stage: tilelang.layout.make_swizzled_layout(core_shadow_stage),
                })

                q_tma_desc = T.create_tma_descriptor(
                    0, 4, q.data,
                    dim, seq_len, heads, batch,
                    1, heads * dim, dim, seq_len * heads * dim,
                    dim, 128, 1, 1,
                    1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                k_tma_desc = T.create_tma_descriptor(
                    0, 4, k.data,
                    dim, seq_len, heads_kv, batch,
                    1, heads_kv * dim, dim, seq_len * heads_kv * dim,
                    dim, 224, 1, 1,
                    1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                v_tma_desc = T.create_tma_descriptor(
                    0, 4, v.data,
                    dim, seq_len, heads_kv, batch,
                    1, heads_kv * dim, dim, seq_len * heads_kv * dim,
                    dim, 224, 1, 1,
                    1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                output_tma_desc = T.create_tma_descriptor(
                    9, 5, output.data,
                    dim, seq_len, heads, batch, 1,
                    2, heads * dim * 2, dim * 2, seq_len * heads * dim * 2,
                    batch * seq_len * heads * dim * 2,
                    64, 128, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    0, 3, 2, 0,
                )
                T.call_extern(
                    "handle",
                    "tileops_fa3_shaped_prepare_params",
                    q_tma_desc,
                    k_tma_desc,
                    v_tma_desc,
                    output_tma_desc,
                    q.data,
                    k.data,
                    v.data,
                    q_descale.data,
                    k_descale.data,
                    v_descale.data,
                    output.data,
                    lse.data,
                    tx,
                    lane_id,
                    warp_id,
                    warpgroup_id,
                    warpgroup_lane,
                    tile_m,
                    bidh,
                    bidb,
                    bidh_kv,
                    smem.access_ptr("rw"),
                )
                T.call_extern(
                    "handle",
                    "tileops_fa3_shaped_prepare_runtime",
                    tile_m,
                    bidh,
                    bidb,
                    smem.access_ptr("rw"),
                )

                if tx < 128:
                    producer_core_shadow(
                        q,
                        k,
                        v,
                        core_shadow_stage,
                        core_shadow_vt_stage,
                        core_shadow_q_full,
                        core_shadow_q_done,
                        core_shadow_k_full,
                        core_shadow_k_done,
                        core_shadow_v_full,
                        core_shadow_v_transform_full,
                        core_shadow_v_done,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                    )
                    T.dec_max_nreg(24)
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_producer_load_one_tile",
                        0,
                        tx,
                        warpgroup_id,
                        producer_warp,
                        producer_lane,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                        group,
                        smem.access_ptr("rw"),
                    )
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_producer_load_tail",
                        0,
                        tx,
                        warpgroup_id,
                        producer_warp,
                        producer_lane,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                        group,
                        smem.access_ptr("rw"),
                    )
                elif tx < 256:
                    consumer_core_shadow_wg1(
                        q,
                        k,
                        v,
                        core_shadow_stage,
                        core_shadow_vt_stage,
                        core_shadow_q_full,
                        core_shadow_q_done,
                        core_shadow_k_full,
                        core_shadow_k_done,
                        core_shadow_v_full,
                        core_shadow_v_transform_full,
                        core_shadow_v_done,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                    )
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_run_consumer_wg1",
                        1,
                        tx,
                        warpgroup_id,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                        group,
                        smem.access_ptr("rw"),
                    )
                else:
                    consumer_core_shadow_wg2(
                        q,
                        core_shadow_stage,
                        core_shadow_q_full,
                        core_shadow_q_done,
                        core_shadow_k_full,
                        core_shadow_k_done,
                        core_shadow_v_full,
                        core_shadow_v_transform_full,
                        core_shadow_v_done,
                        tx,
                        tile_m,
                        bidh,
                        bidb,
                    )
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_run_consumer_wg2",
                        2,
                        tx,
                        warpgroup_id,
                        tile_m,
                        bidh,
                        bidb,
                        bidh_kv,
                        group,
                        smem.access_ptr("rw"),
                    )

        return main

    return func(), None


def build_vt_to_v_boundary_probe_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    stages: int,
    query_smem: bool,
    static_persistent_call_extern: bool,
):
    helper_header = _write_extern_helper_header(
        batch,
        seq_len,
        heads,
        heads_kv,
        dim,
        stages,
        query_smem,
        static_persistent_call_extern,
    )

    @tilelang.jit(
        out_idx=[1],
        execution_backend="tvm_ffi",
        compile_flags=[
            "-O3",
            "-DNDEBUG",
            "-DTILEOPS_FA3_SHAPED_LAUNCHER_TAG_vt_to_v_boundary",
            "-Xptxas=-v",
            "--expt-relaxed-constexpr",
            "-DENABLE_BF16",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90A_ENABLED",
            f"-I{FA3_INC}",
            "-include",
            str(helper_header),
        ],
    )
    def func():
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def main(
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            status: T.Tensor((1,), "int32"),
        ) -> None:
            with T.Kernel(T.ceildiv(seq_len, 224), heads_kv, batch, threads=128) as (tile_n, bidh_kv, bidb):
                vt_stage = T.alloc_shared((dim, 224), "float8_e4m3fn")
                v_stage = T.alloc_shared((dim, 224), "float8_e4m3fn")
                o_stage = T.alloc_shared((128, dim), "bfloat16")
                v_full = T.alloc_barrier(arrive_count=128)
                tx = T.get_thread_binding()

                T.reads(v[0:batch, 0:seq_len, 0:heads_kv, 0:dim])
                T.writes(
                    status[0:1],
                    vt_stage[0:dim, 0:224],
                    v_stage[0:dim, 0:224],
                )

                vt_tma_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8, 4, v.data,
                    dim, heads_kv, seq_len, batch,
                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                    dim, 1, 224, 1,
                    1, 1, 1, 1,
                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                )
                if tx == 0:
                    T.mbarrier_expect_tx(v_full, dim * 224)
                    T.tma_load(
                        vt_tma_desc,
                        v_full[0],
                        T.access_ptr(vt_stage, "w"),
                        0,
                        bidh_kv,
                        tile_n * 224,
                        bidb,
                        EVICT_NORMAL,
                    )
                T.barrier_arrive(v_full)
                T.barrier_wait(v_full, 0)
                T.call_extern(
                    "handle",
                    "tileops_fa3_shaped_vt_to_v_boundary",
                    vt_stage.access_ptr("r"),
                    v_stage.access_ptr("w"),
                    tx,
                )
                T.sync_threads()
                if tx == 0:
                    T.call_extern(
                        "handle",
                        "tileops_fa3_shaped_check_v_mma_layout_boundary",
                        v_stage.access_ptr("r"),
                        tx,
                    )
                    status[0] = 1

        return main

    return func()


def build_pv_correctness_probe_kernel(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    stages: int,
    query_smem: bool,
    static_persistent_call_extern: bool,
):
    if seq_len % 224 != 0:
        raise ValueError("PV correctness probe currently requires --seq-len to be a multiple of 224")

    helper_header = _write_extern_helper_header(
        batch,
        seq_len,
        heads,
        heads_kv,
        dim,
        1,
        query_smem,
        static_persistent_call_extern,
    )

    @tilelang.jit(
        out_idx=[2],
        execution_backend="tvm_ffi",
        compile_flags=[
            "-O3",
            "-DNDEBUG",
            "-DTILEOPS_FA3_SHAPED_LAUNCHER_TAG_pv_correctness",
            "-Xptxas=-v",
            "--expt-relaxed-constexpr",
            "-DENABLE_BF16",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90_ENABLED",
            "-DCUTLASS_ARCH_MMA_SM90A_ENABLED",
            f"-I{FA3_INC}",
            "-include",
            str(ROOT / "tileops" / "kernels" / "attention" / "_fp8_gqa_helper.h"),
            "-include",
            str(helper_header),
        ],
    )
    def func():
        num_n_blocks = seq_len // 224
        kv_shape = (batch, seq_len, heads_kv, dim)
        p_shape = (batch, num_n_blocks, heads_kv, 128, 224)
        out_shape = (batch, num_n_blocks, heads_kv, 128, dim)

        @T.prim_func
        def main(
            p: T.Tensor(p_shape, "float32"),
            v: T.Tensor(kv_shape, "float8_e4m3fn"),
            out: T.Tensor(out_shape, "bfloat16"),
        ) -> None:
            with T.Kernel(num_n_blocks, heads_kv, batch, threads=256) as (tile_n, bidh_kv, bidb):
                vt_stage = T.alloc_shared((dim, 224), "float8_e4m3fn")
                v_stage = T.alloc_shared((dim, 224), "float8_e4m3fn")
                o_stage = T.alloc_shared((128, dim), "bfloat16")
                v_full = T.alloc_barrier(arrive_count=1)
                tx = T.get_thread_binding()

                T.reads(
                    p[0:batch, 0:num_n_blocks, 0:heads_kv, 0:128, 0:224],
                    v[0:batch, 0:seq_len, 0:heads_kv, 0:dim],
                )
                T.writes(
                    out[0:batch, 0:num_n_blocks, 0:heads_kv, 0:128, 0:dim],
                    vt_stage[0:dim, 0:224],
                    v_stage[0:dim, 0:224],
                    o_stage[0:128, 0:dim],
                )

                vt_tma_desc = T.create_tma_descriptor(
                    TMA_DTYPE_UINT8, 4, v.data,
                    dim, heads_kv, seq_len, batch,
                    1, dim, heads_kv * dim, seq_len * heads_kv * dim,
                    dim, 1, 224, 1,
                    1, 1, 1, 1,
                    TMA_INTERLEAVE_NONE, TMA_SWIZZLE_128B,
                    TMA_L2_PROMOTION_128B, TMA_OOB_FILL_NONE,
                )
                if tx == 0:
                    T.mbarrier_expect_tx(v_full, dim * 224)
                    tir.call_extern(
                        "handle",
                        "tl::fp8_tma_load_4d_ptx",
                        vt_tma_desc,
                        v_full[0],
                        T.access_ptr(vt_stage, "w"),
                        0,
                        bidh_kv,
                        tile_n * 224,
                        bidb,
                    )
                    T.barrier_arrive(v_full)
                T.mbarrier_wait_parity(v_full, 0)
                if tx < 128:
                    T.call_extern(
                        "handle",
                        "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm",
                        vt_stage.access_ptr("r"),
                        v_stage.access_ptr("w"),
                    )
                T.sync_threads()
                T.call_extern(
                    "handle",
                    "tileops_fa3_shaped_pv_correctness_boundary",
                    p.data,
                    v_stage.access_ptr("r"),
                    o_stage.access_ptr("rw"),
                    out.access_ptr("w"),
                    tx,
                    tile_n,
                    bidh_kv,
                    bidb,
                )

        return main

    return func()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--heads-kv", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pack-gqa", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--smem-bytes", type=int, default=196608)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--query-smem", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--force-rebuild-launcher", action="store_true")
    parser.add_argument("--host-launcher", action="store_true")
    parser.add_argument("--role-run", action="store_true")
    parser.add_argument("--static-persistent-call-extern", action="store_true")
    parser.add_argument("--validate-wg-branch", action="store_true")
    parser.add_argument("--producer-split", action="store_true")
    parser.add_argument("--producer-tma-hlir-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-q-boundary-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-q-stage-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-k-buffer-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-v-buffer-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-k-pipeline-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-v-pipeline-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-kv-boundary-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-core-shadow-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-core-shadow-kind", choices=["q", "qk", "qkv", "qkvv"], default="q")
    parser.add_argument("--producer-tma-hlir-vt-to-v-boundary-probe", action="store_true")
    parser.add_argument("--producer-tma-hlir-pv-correctness-probe", action="store_true")
    parser.add_argument("--legacy-raw-device", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.role_run and args.static_persistent_call_extern:
        raise ValueError("--role-run cannot be combined with --static-persistent-call-extern yet")
    if args.producer_split and not args.role_run:
        raise ValueError("--producer-split requires --role-run")
    if args.producer_tma_hlir_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-probe requires --producer-split")
    if args.producer_tma_hlir_q_boundary_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-q-boundary-probe requires --producer-split")
    if args.producer_tma_hlir_q_stage_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-q-stage-probe requires --producer-split")
    if args.producer_tma_hlir_k_buffer_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-k-buffer-probe requires --producer-split")
    if args.producer_tma_hlir_v_buffer_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-v-buffer-probe requires --producer-split")
    if args.producer_tma_hlir_k_pipeline_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-k-pipeline-probe requires --producer-split")
    if args.producer_tma_hlir_v_pipeline_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-v-pipeline-probe requires --producer-split")
    if args.producer_tma_hlir_kv_boundary_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-kv-boundary-probe requires --producer-split")
    if args.producer_tma_hlir_core_shadow_probe and not args.producer_split:
        raise ValueError("--producer-tma-hlir-core-shadow-probe requires --producer-split")
    producer_tma_probe_count = sum([
        bool(args.producer_tma_hlir_probe),
        bool(args.producer_tma_hlir_q_boundary_probe),
        bool(args.producer_tma_hlir_q_stage_probe),
        bool(args.producer_tma_hlir_k_buffer_probe),
        bool(args.producer_tma_hlir_v_buffer_probe),
        bool(args.producer_tma_hlir_k_pipeline_probe),
        bool(args.producer_tma_hlir_v_pipeline_probe),
        bool(args.producer_tma_hlir_kv_boundary_probe),
        bool(args.producer_tma_hlir_core_shadow_probe),
        bool(args.producer_tma_hlir_vt_to_v_boundary_probe),
        bool(args.producer_tma_hlir_pv_correctness_probe),
    ])
    if producer_tma_probe_count > 1:
        raise ValueError("producer TMA HLIR probe flags are mutually exclusive")
    pack_gqa = _parse_pack_gqa(args.pack_gqa, args.seq_len, args.heads, args.heads_kv)
    legacy_raw_device = args.legacy_raw_device
    if args.host_launcher:
        print(
            "fa3_mode",
            "host_launcher_run_flash_fwd<90,128,128,ClusterM=1,e4m3,bf16,"
            f"causal=false,local=false,softcap={str(args.softcap > 0.0).lower()},varlen=false,"
            "PagedKVNonTMA=false,append=false,qv=false,"
            f"PackGQA={str(pack_gqa).lower()},split=false,V_colmajor=false>",
        )
    elif legacy_raw_device:
        print("fa3_mode legacy_raw_device_manual_params")
    elif args.static_persistent_call_extern:
        print("fa3_mode tilelang_primfunc_call_extern_fa3_static_persistent")
    elif args.producer_tma_hlir_core_shadow_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_tma_hlir_core_shadow_probe")
    elif args.producer_tma_hlir_vt_to_v_boundary_probe:
        print("fa3_mode tilelang_vt_to_v_out_of_place_boundary_probe")
    elif args.producer_tma_hlir_pv_correctness_probe:
        print("fa3_mode tilelang_pv_correctness_probe_vt_to_v_out_of_place")
    elif args.producer_tma_hlir_q_boundary_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_q_tma_hlir_consumer_tensor_boundary_probe")
    elif args.producer_tma_hlir_kv_boundary_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_kv_tma_hlir_stage_boundary_probe")
    elif args.producer_tma_hlir_v_pipeline_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_v_tma_hlir_pipeline_probe")
    elif args.producer_tma_hlir_k_pipeline_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_k_tma_hlir_pipeline_probe")
    elif args.producer_tma_hlir_v_buffer_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_v_tma_hlir_buffer_probe")
    elif args.producer_tma_hlir_k_buffer_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_k_tma_hlir_buffer_probe")
    elif args.producer_tma_hlir_q_stage_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_q_tma_hlir_fa3_stage_probe")
    elif args.producer_tma_hlir_probe:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_q_tma_hlir_probe")
    elif args.producer_split:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_producer_split_hlir_enter")
    elif args.role_run:
        print("fa3_mode tilelang_wg_branch_fa3_prepare_runtime_named_role_helpers_producer_gating_work_coord")
    elif args.validate_wg_branch:
        print("fa3_mode tilelang_primfunc_call_extern_fa3_device_helper_wg_validated")
    else:
        print("fa3_mode tilelang_primfunc_call_extern_fa3_device_helper")

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seq_len, args.heads, args.dim, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(args.batch, args.seq_len, args.heads_kv, args.dim, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(args.batch, args.seq_len, args.heads_kv, args.dim, device="cuda", dtype=torch.float16) * 0.25
    q_fp8, q_descale = _quantize_q_fa3_gqa_descale(q, args.heads_kv)
    k_fp8, k_descale = _quantize_kv_fa3_descale(k)
    v_fp8, v_descale = _quantize_kv_fa3_descale(v)

    if args.producer_tma_hlir_vt_to_v_boundary_probe:
        kernel = build_vt_to_v_boundary_probe_kernel(
            args.batch,
            args.seq_len,
            args.heads,
            args.heads_kv,
            args.dim,
            args.stages,
            args.query_smem,
            args.static_persistent_call_extern,
        )
        status = kernel(v_fp8)
        torch.cuda.synchronize()
        print("vt_to_v_boundary_status", status.cpu().tolist())
        if args.bench:
            ms = bench_kernel(
                kernel,
                args=(v_fp8,),
                n_warmup=args.warmup,
                n_repeat=args.repeat,
                n_trials=3,
            )
            print(f"latency_ms={ms:.6f}")
        return

    if args.producer_tma_hlir_pv_correctness_probe:
        kernel = build_pv_correctness_probe_kernel(
            args.batch,
            args.seq_len,
            args.heads,
            args.heads_kv,
            args.dim,
            args.stages,
            args.query_smem,
            args.static_persistent_call_extern,
        )
        num_n_blocks = args.seq_len // 224
        p = torch.randn(
            args.batch,
            num_n_blocks,
            args.heads_kv,
            128,
            224,
            device="cuda",
            dtype=torch.float16,
        ) * 0.25
        p_logical = p.float().contiguous()
        p_fp8 = torch.clamp(p_logical, -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
        out = kernel(p_logical, v_fp8)
        torch.cuda.synchronize()
        v_tiles = (
            v_fp8.float()
            .reshape(args.batch, num_n_blocks, 224, args.heads_kv, args.dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        ref = torch.einsum("bthmn,bthnd->bthmd", p_fp8.float(), v_tiles)
        ref_cmp = ref.to(torch.bfloat16).float()
        out_cmp = out.float()
        max_abs = (out_cmp - ref_cmp).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(out_cmp.flatten(), ref_cmp.flatten(), dim=0).item()
        print("pv_correctness_out", out.shape, out.dtype, "finite", bool(torch.isfinite(out).all()))
        print(f"pv_correctness max_abs={max_abs:.6g} cosine={cos:.8f}")
        print(
            "pv_correctness_norms",
            f"out={out_cmp.norm().item():.6g}",
            f"ref={ref_cmp.norm().item():.6g}",
            f"out_min={out_cmp.min().item():.6g}",
            f"out_max={out_cmp.max().item():.6g}",
            f"ref_min={ref_cmp.min().item():.6g}",
            f"ref_max={ref_cmp.max().item():.6g}",
        )
        out2 = out_cmp[0, 0, 0]
        ref2 = ref_cmp[0, 0, 0]
        row_cos = torch.nn.functional.normalize(out2, dim=1) @ torch.nn.functional.normalize(ref2, dim=1).T
        col_cos = torch.nn.functional.normalize(out2.T, dim=1) @ torch.nn.functional.normalize(ref2.T, dim=1).T
        print(
            "pv_correctness_best",
            f"row_cos={row_cos.max(dim=1).values.mean().item():.6g}",
            f"col_cos={col_cos.max(dim=1).values.mean().item():.6g}",
            "col_argmax_first32",
            col_cos.argmax(dim=1)[:32].tolist(),
        )
        ref_first = torch.einsum("bthmn,bthnd->bthmd", p_fp8.float()[..., :112], v_tiles[..., :112, :]).to(torch.bfloat16).float()
        ref_second = torch.einsum("bthmn,bthnd->bthmd", p_fp8.float()[..., 112:], v_tiles[..., 112:, :]).to(torch.bfloat16).float()
        print(
            "pv_correctness_halves",
            f"first_cos={torch.nn.functional.cosine_similarity(out_cmp.flatten(), ref_first.flatten(), dim=0).item():.8f}",
            f"first_max={(out_cmp - ref_first).abs().max().item():.6g}",
            f"second_cos={torch.nn.functional.cosine_similarity(out_cmp.flatten(), ref_second.flatten(), dim=0).item():.8f}",
            f"second_max={(out_cmp - ref_second).abs().max().item():.6g}",
        )
        chunk_refs = []
        for start in range(0, 224, 32):
            stop = min(start + 32, 224)
            chunk_refs.append(
                torch.einsum(
                    "bthmn,bthnd->bthmd",
                    p_fp8.float()[..., start:stop],
                    v_tiles[..., start:stop, :],
                ).to(torch.bfloat16).float()
            )
        chunk_cos = [
            torch.nn.functional.cosine_similarity(out_cmp.flatten(), r.flatten(), dim=0).item()
            for r in chunk_refs
        ]
        prefix_cos = [
            torch.nn.functional.cosine_similarity(out_cmp.flatten(), sum(chunk_refs[:i + 1]).flatten(), dim=0).item()
            for i in range(len(chunk_refs))
        ]
        suffix_cos = [
            torch.nn.functional.cosine_similarity(out_cmp.flatten(), sum(chunk_refs[i:]).flatten(), dim=0).item()
            for i in range(len(chunk_refs))
        ]
        print("pv_correctness_chunk_cos", [round(x, 6) for x in chunk_cos])
        print("pv_correctness_prefix_cos", [round(x, 6) for x in prefix_cos])
        print("pv_correctness_suffix_cos", [round(x, 6) for x in suffix_cos])
        if args.bench:
            ms = bench_kernel(
                kernel,
                args=(p_logical, v_fp8),
                n_warmup=args.warmup,
                n_repeat=args.repeat,
                n_trials=3,
            )
            print(f"latency_ms={ms:.6f}")
        return

    if args.producer_tma_hlir_core_shadow_probe:
        kernel, launcher_so = build_core_shadow_kernel(
            args.batch,
            args.seq_len,
            args.heads,
            args.heads_kv,
            args.dim,
            args.smem_bytes,
            args.stages,
            args.query_smem,
            args.static_persistent_call_extern,
            args.producer_tma_hlir_core_shadow_kind,
        )
    else:
        kernel, launcher_so = build_kernel(
            args.batch,
            args.seq_len,
            args.heads,
            args.heads_kv,
            args.dim,
            args.smem_bytes,
            args.stages,
            args.query_smem,
            pack_gqa,
            args.softcap,
            args.force_rebuild_launcher,
            legacy_raw_device,
            args.host_launcher,
            args.role_run,
            args.static_persistent_call_extern,
            args.validate_wg_branch,
            args.producer_split,
            args.producer_tma_hlir_probe,
            args.producer_tma_hlir_q_boundary_probe,
            args.producer_tma_hlir_q_stage_probe,
            args.producer_tma_hlir_k_buffer_probe,
            args.producer_tma_hlir_v_buffer_probe,
            args.producer_tma_hlir_k_pipeline_probe,
            args.producer_tma_hlir_v_pipeline_probe,
            args.producer_tma_hlir_kv_boundary_probe,
            args.producer_tma_hlir_core_shadow_probe,
        )
    if launcher_so is not None:
        print("launcher_so", launcher_so)
    out, lse = kernel(q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)
    torch.cuda.synchronize()
    if args.query_smem:
        print("shared_storage_size", int(lse.flatten()[0].item()), "sizeof_shared_storage", int(lse.flatten()[1].item()))
        return
    print("out", out.shape, out.dtype, "finite", bool(torch.isfinite(out.float()).all()))
    print("lse", lse.shape, lse.dtype, "finite", bool(torch.isfinite(lse.float()).all()))
    if not legacy_raw_device:
        ref = _reference_attention(
            q_fp8,
            k_fp8,
            v_fp8,
            q_descale,
            k_descale,
            v_descale,
            softcap=args.softcap,
        )
        out_f = out.float()
        ref_f = ref.float()
        max_abs = (out_f - ref_f).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0).item()
        print(f"correctness max_abs={max_abs:.6g} cosine={cos:.8f}")

    if args.bench:
        ms = bench_kernel(
            kernel,
            args=(q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale),
            n_warmup=args.warmup,
            n_repeat=args.repeat,
            n_trials=3,
        )
        flops = 4.0 * args.batch * args.heads * args.seq_len * args.seq_len * args.dim
        print(f"latency_ms={ms:.6f} tflops={flops / ms * 1e-9:.2f}")


if __name__ == "__main__":
    main()
