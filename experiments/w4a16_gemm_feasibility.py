"""Local W4A16 GEMM feasibility experiment.

This is intentionally not a public TileOps Op.  It tests whether TileLang can
fuse signed-INT4 unpacking and group-wise dequantization into an A16 GEMM
mainloop before a manifest/API contract is proposed upstream.

Physical weight contract used by the experiment:

* logical weight: ``[N, K]`` group-wise INT4;
* storage: ``uint8[N, K / 2]``;
* low nibble stores even K, high nibble stores odd K;
* nibbles are unsigned and dequantize as ``(q_u4 - zero) * scale``;
* the TileLang LOP3 path additionally uses its required 32-bit-word interleave;
* scale: ``float32[N, K / group_size]`` with ``group_size=128``.
* zero: ``uint8[N, K / group_size]``.
"""

from __future__ import annotations

import argparse
import functools
from typing import Callable

import tilelang
import tilelang.language as T
import torch
from tilelang.quantize import (
    _tir_packed_to_unsigned_convert,
    get_lop3_intrin_group,
    interleave_weight,
)


GROUP_SIZE = 128


def _schedule_permutation(size: int, mode: int) -> tuple[int, ...]:
    """Static traversal orders used by structural AKO schedule rounds."""
    indices = list(range(size))
    if mode == 0:
        return tuple(indices)
    if mode == 1:
        return tuple(reversed(indices))
    if mode == 2:
        return tuple(indices[::2] + indices[1::2])
    if mode == 3:
        return tuple(indices[1::2] + indices[::2])
    if mode == 4:
        return tuple(index ^ (size // 2) for index in indices)
    if mode == 5:
        bits = (size - 1).bit_length()
        return tuple(
            int(f"{index:0{bits}b}"[::-1], 2) for index in indices
        )
    if mode == 6:
        return tuple(index ^ (index >> 1) for index in indices)
    if mode == 7:
        return tuple(reversed(
            [index ^ (index >> 1) for index in indices]
        ))
    raise ValueError(f"unknown schedule permutation mode {mode}")


def _schedule_index_expr(index, size: int, mode: int):
    """Return a TileLang expression for one compile-time-selected order."""
    if mode == 0:
        return index
    if mode == 1:
        return size - 1 - index
    if mode == 2:
        return T.if_then_else(
            index < size // 2,
            index * 2,
            (index - size // 2) * 2 + 1,
        )
    if mode == 3:
        return T.if_then_else(
            index < size // 2,
            index * 2 + 1,
            (index - size // 2) * 2,
        )
    if mode == 4:
        return index ^ (size // 2)
    bits = (size - 1).bit_length()
    if mode == 5:
        result = index & 1
        for bit in range(1, bits):
            result = (
                (result << 1)
                | ((index >> bit) & 1)
            )
        return result
    if mode == 6:
        return index ^ (index >> 1)
    if mode == 7:
        reversed_index = size - 1 - index
        return reversed_index ^ (reversed_index >> 1)
    raise ValueError(f"unknown schedule expression mode {mode}")


MMA_SYNC_W4_SOURCE = r"""
#include <cuda_fp16.h>

namespace tl {

__device__ __forceinline__ unsigned w4_magic_pair(unsigned char packed) {
  // Place the low and high nibbles into the mantissas of two FP16 values
  // representing 1024 + q.  The following half2 subtraction removes both
  // the exponent bias and the affine zero point.
  unsigned q = (static_cast<unsigned>(packed) & 0x0fu)
             | ((static_cast<unsigned>(packed) & 0xf0u) << 12);
  unsigned result;
  asm volatile(
      "lop3.b32 %0, %1, %2, %3, 0xea;\n"
      : "=r"(result)
      : "r"(q), "r"(0x000f000fu), "r"(0x64006400u));
  return result;
}

__device__ __forceinline__ void w4_mma(
    const unsigned* a,
    const unsigned* b,
    float& c0,
    float& c1,
    float& c2,
    float& c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

template <int MODE>
__device__ __forceinline__ int w4_mma_schedule_index(int index) {
  if constexpr (MODE == 0) {
    return index;
  } else if constexpr (MODE == 1) {
    return 7 - index;
  } else if constexpr (MODE == 2) {
    return index < 4 ? index * 2 : (index - 4) * 2 + 1;
  } else if constexpr (MODE == 3) {
    return index < 4 ? index * 2 + 1 : (index - 4) * 2;
  } else if constexpr (MODE == 4) {
    return index ^ 4;
  } else if constexpr (MODE == 5) {
    return ((index & 1) << 2) | (index & 2) | ((index & 4) >> 2);
  } else if constexpr (MODE == 6) {
    return index ^ (index >> 1);
  } else {
    const int reverse = 7 - index;
    return reverse ^ (reverse >> 1);
  }
}

template <int BLOCK_K>
__device__ __forceinline__ void w4_prepare_transposed_mma_fragments(
    const __half* activation,
    const unsigned char* packed,
    int packed_n_tile,
    int group,
    int sub,
    __half2 zero_row0,
    __half2 zero_row8,
    unsigned* a,
    unsigned* b) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int k_inner = group * 128 + sub * 16;
  const int packed_offset =
      (((packed_n_tile * (BLOCK_K / 16) + k_inner / 16) * 32)
       + lane) * 4;

  // PTX mma.m16n8k16 row-major A ownership:
  //   a0/a2 -> output row lane_group
  //   a1/a3 -> output row lane_group + 8
  // and the two register pairs cover K[0:8] and K[8:16].
  const unsigned magic0 = w4_magic_pair(packed[packed_offset + 0]);
  const unsigned magic1 = w4_magic_pair(packed[packed_offset + 1]);
  const unsigned magic2 = w4_magic_pair(packed[packed_offset + 2]);
  const unsigned magic3 = w4_magic_pair(packed[packed_offset + 3]);
  const __half2 value0 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic0), zero_row0);
  const __half2 value1 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic1), zero_row8);
  const __half2 value2 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic2), zero_row0);
  const __half2 value3 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic3), zero_row8);
  a[0] = *reinterpret_cast<const unsigned*>(&value0);
  a[1] = *reinterpret_cast<const unsigned*>(&value1);
  a[2] = *reinterpret_cast<const unsigned*>(&value2);
  a[3] = *reinterpret_cast<const unsigned*>(&value3);

  // Matrix B is Kx8.  Only batch column zero is real for M=1, so only
  // lane_group zero owns non-zero B fragments.
  b[0] = 0u;
  b[1] = 0u;
  if (lane_group == 0) {
    const int k0 = k_inner + lane_in_group * 2;
    const __half2 b01 =
        __halves2half2(activation[k0], activation[k0 + 1]);
    const __half2 b23 =
        __halves2half2(activation[k0 + 8], activation[k0 + 9]);
    b[0] = *reinterpret_cast<const unsigned*>(&b01);
    b[1] = *reinterpret_cast<const unsigned*>(&b23);
  }
}

template <int BLOCK_K, int BATCH, int SUB_MODE>
__device__ __forceinline__ void w4_transposed_compute_group(
    const __half* activation,
    const unsigned char* packed,
    int packed_n_tile,
    int group,
    __half2 zero_row0,
    __half2 zero_row8,
    float& g0,
    float& g1,
    float& g2,
    float& g3) {
  static_assert(8 % BATCH == 0, "BATCH must divide K128");
  #pragma unroll
  for (int sub_base = 0; sub_base < 8; sub_base += BATCH) {
    unsigned a_batch[BATCH][4];
    unsigned b_batch[BATCH][2];
    #pragma unroll
    for (int item = 0; item < BATCH; ++item) {
      const int sub =
          w4_mma_schedule_index<SUB_MODE>(sub_base + item);
      w4_prepare_transposed_mma_fragments<BLOCK_K>(
          activation,
          packed,
          packed_n_tile,
          group,
          sub,
          zero_row0,
          zero_row8,
          a_batch[item],
          b_batch[item]);
    }
    #pragma unroll
    for (int item = 0; item < BATCH; ++item) {
      w4_mma(
          a_batch[item],
          b_batch[item],
          g0,
          g1,
          g2,
          g3);
    }
  }
}

template <
    int BLOCK_K,
    int SCALE_STRIDE,
    int SCHEDULE_ID>
__device__ __forceinline__ void w4a16_mma_sync_transposed_tile(
    const void* activation_raw,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zeros,
    int packed_n_tile,
    int output_col_base,
    int scale_group_base,
    float& c0,
    float& c1,
    float& c2,
    float& c3) {
  constexpr int SUB_MODE = SCHEDULE_ID & 7;
  constexpr int BATCH_LOG2 = (SCHEDULE_ID >> 3) & 3;
  constexpr int BATCH = 1 << BATCH_LOG2;
  constexpr bool REVERSE_GROUP = ((SCHEDULE_ID >> 5) & 1) != 0;
  constexpr bool DEFER_SCALE = ((SCHEDULE_ID >> 6) & 1) != 0;
  const __half* activation =
      reinterpret_cast<const __half*>(activation_raw);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;

  float deferred[2][4] = {
      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f},
  };
  #pragma unroll
  for (int group_iter = 0; group_iter < BLOCK_K / 128; ++group_iter) {
    const int group = REVERSE_GROUP
        ? BLOCK_K / 128 - 1 - group_iter
        : group_iter;
    const int scale_group = scale_group_base + group;
    const int output_row0 = output_col_base + lane_group;
    const int output_row8 = output_row0 + 8;
    const unsigned zero0 = static_cast<unsigned>(
        zeros[output_row0 * SCALE_STRIDE + scale_group]);
    const unsigned zero8 = static_cast<unsigned>(
        zeros[output_row8 * SCALE_STRIDE + scale_group]);
    const unsigned zero_magic0 =
        0x64006400u + zero0 + (zero0 << 16);
    const unsigned zero_magic8 =
        0x64006400u + zero8 + (zero8 << 16);
    const __half2 zero_row0 =
        *reinterpret_cast<const __half2*>(&zero_magic0);
    const __half2 zero_row8 =
        *reinterpret_cast<const __half2*>(&zero_magic8);
    float g0 = 0.0f;
    float g1 = 0.0f;
    float g2 = 0.0f;
    float g3 = 0.0f;
    w4_transposed_compute_group<BLOCK_K, BATCH, SUB_MODE>(
        activation,
        packed,
        packed_n_tile,
        group,
        zero_row0,
        zero_row8,
        g0,
        g1,
        g2,
        g3);

    if constexpr (DEFER_SCALE) {
      deferred[group][0] = g0;
      deferred[group][1] = g1;
      deferred[group][2] = g2;
      deferred[group][3] = g3;
    } else if (lane_in_group == 0) {
      const float scale0 =
          scales[output_row0 * SCALE_STRIDE + scale_group];
      const float scale8 =
          scales[output_row8 * SCALE_STRIDE + scale_group];
      c0 = fmaf(g0, scale0, c0);
      c2 = fmaf(g2, scale8, c2);
    }
  }

  if constexpr (DEFER_SCALE) {
    #pragma unroll
    for (int group_iter = 0; group_iter < BLOCK_K / 128; ++group_iter) {
      const int group = REVERSE_GROUP
          ? BLOCK_K / 128 - 1 - group_iter
          : group_iter;
      const int scale_group = scale_group_base + group;
      if (lane_in_group == 0) {
        const int output_row0 = output_col_base + lane_group;
        const int output_row8 = output_row0 + 8;
        const float scale0 =
            scales[output_row0 * SCALE_STRIDE + scale_group];
        const float scale8 =
            scales[output_row8 * SCALE_STRIDE + scale_group];
        c0 = fmaf(deferred[group][0], scale0, c0);
        c2 = fmaf(deferred[group][2], scale8, c2);
      }
    }
  }
}

template <int BLOCK_K, bool GROUP_POST_SCALE>
__device__ __forceinline__ void w4_prepare_mma_fragments(
    const __half* activation,
    const unsigned char* packed,
    int group,
    int sub,
    int output_col_base,
    __half2 zero20,
    __half2 zero21,
    __half2 scale20,
    __half2 scale21,
    unsigned* a,
    unsigned* b0,
    unsigned* b1) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int k_inner = group * 128 + sub * 16;
  a[0] = 0u;
  a[1] = 0u;
  a[2] = 0u;
  a[3] = 0u;
  if (lane_group == 0) {
    const int k0 = k_inner + lane_in_group * 2;
    const __half2 a01 =
        __halves2half2(activation[k0], activation[k0 + 1]);
    const __half2 a23 =
        __halves2half2(activation[k0 + 8], activation[k0 + 9]);
    a[0] = *reinterpret_cast<const unsigned*>(&a01);
    a[2] = *reinterpret_cast<const unsigned*>(&a23);
  }

  const int packed_offset0 =
      ((output_col_base / 8) * (BLOCK_K / 16) + k_inner / 16) * 64
      + lane * 2;
  const int packed_offset1 =
      ((output_col_base / 8 + 1) * (BLOCK_K / 16) + k_inner / 16) * 64
      + lane * 2;
  const unsigned packed_pair0 =
      *reinterpret_cast<const unsigned short*>(&packed[packed_offset0]);
  const unsigned packed_pair1 =
      *reinterpret_cast<const unsigned short*>(&packed[packed_offset1]);

  const unsigned magic000 =
      w4_magic_pair(static_cast<unsigned char>(packed_pair0));
  const unsigned magic001 =
      w4_magic_pair(static_cast<unsigned char>(packed_pair0 >> 8));
  __half2 value000 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic000), zero20);
  __half2 value001 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic001), zero20);
  if constexpr (!GROUP_POST_SCALE) {
    value000 = __hmul2(value000, scale20);
    value001 = __hmul2(value001, scale20);
  }
  b0[0] = *reinterpret_cast<const unsigned*>(&value000);
  b0[1] = *reinterpret_cast<const unsigned*>(&value001);

  const unsigned magic100 =
      w4_magic_pair(static_cast<unsigned char>(packed_pair1));
  const unsigned magic101 =
      w4_magic_pair(static_cast<unsigned char>(packed_pair1 >> 8));
  __half2 value100 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic100), zero21);
  __half2 value101 = __hsub2(
      *reinterpret_cast<const __half2*>(&magic101), zero21);
  if constexpr (!GROUP_POST_SCALE) {
    value100 = __hmul2(value100, scale21);
    value101 = __hmul2(value101, scale21);
  }
  b1[0] = *reinterpret_cast<const unsigned*>(&value100);
  b1[1] = *reinterpret_cast<const unsigned*>(&value101);
}

template <
    int BLOCK_K,
    int SCALE_STRIDE,
    bool GROUP_POST_SCALE,
    int MMA_BATCH,
    bool MMA_PINGPONG>
__device__ __forceinline__ void w4a16_mma_sync_tile(
    const void* activation_raw,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zeros,
    int output_col_base,
    int scale_group_base,
    float& c00,
    float& c01,
    float& c02,
    float& c03,
    float& c10,
    float& c11,
    float& c12,
    float& c13) {
  const __half* activation =
      reinterpret_cast<const __half*>(activation_raw);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;

  #pragma unroll
  for (int group = 0; group < BLOCK_K / 128; ++group) {
    const int scale_group = scale_group_base + group;
    const int output_col0 = output_col_base + lane_group;
    const int output_col1 = output_col_base + 8 + lane_group;
    const unsigned zero0 = static_cast<unsigned>(
        zeros[output_col0 * SCALE_STRIDE + scale_group]);
    const unsigned zero1 = static_cast<unsigned>(
        zeros[output_col1 * SCALE_STRIDE + scale_group]);
    const unsigned zero_magic0 =
        0x64006400u + zero0 + (zero0 << 16);
    const unsigned zero_magic1 =
        0x64006400u + zero1 + (zero1 << 16);
    const __half scale_half0 = __float2half_rn(
        scales[output_col0 * SCALE_STRIDE + scale_group]);
    const __half scale_half1 = __float2half_rn(
        scales[output_col1 * SCALE_STRIDE + scale_group]);
    const __half2 scale20 =
        __halves2half2(scale_half0, scale_half0);
    const __half2 scale21 =
        __halves2half2(scale_half1, scale_half1);
    const __half2 zero20 =
        *reinterpret_cast<const __half2*>(&zero_magic0);
    const __half2 zero21 =
        *reinterpret_cast<const __half2*>(&zero_magic1);
    float g00 = 0.0f;
    float g01 = 0.0f;
    float g02 = 0.0f;
    float g03 = 0.0f;
    float g10 = 0.0f;
    float g11 = 0.0f;
    float g12 = 0.0f;
    float g13 = 0.0f;

    if constexpr (MMA_PINGPONG) {
      unsigned a_slot[2][4];
      unsigned b0_slot[2][2];
      unsigned b1_slot[2][2];
      w4_prepare_mma_fragments<BLOCK_K, GROUP_POST_SCALE>(
          activation, packed, group, 0, output_col_base,
          zero20, zero21, scale20, scale21,
          a_slot[0], b0_slot[0], b1_slot[0]);
      #pragma unroll
      for (int sub = 0; sub < 7; ++sub) {
        const int current = sub & 1;
        const int next = current ^ 1;
        w4_prepare_mma_fragments<BLOCK_K, GROUP_POST_SCALE>(
            activation, packed, group, sub + 1, output_col_base,
            zero20, zero21, scale20, scale21,
            a_slot[next], b0_slot[next], b1_slot[next]);
        if constexpr (GROUP_POST_SCALE) {
          w4_mma(
              a_slot[current], b0_slot[current],
              g00, g01, g02, g03);
          w4_mma(
              a_slot[current], b1_slot[current],
              g10, g11, g12, g13);
        } else {
          w4_mma(
              a_slot[current], b0_slot[current],
              c00, c01, c02, c03);
          w4_mma(
              a_slot[current], b1_slot[current],
              c10, c11, c12, c13);
        }
      }
      if constexpr (GROUP_POST_SCALE) {
        w4_mma(a_slot[1], b0_slot[1], g00, g01, g02, g03);
        w4_mma(a_slot[1], b1_slot[1], g10, g11, g12, g13);
      } else {
        w4_mma(a_slot[1], b0_slot[1], c00, c01, c02, c03);
        w4_mma(a_slot[1], b1_slot[1], c10, c11, c12, c13);
      }
    } else {
      static_assert(8 % MMA_BATCH == 0, "MMA_BATCH must divide K128");
      #pragma unroll
      for (int sub_base = 0; sub_base < 8; sub_base += MMA_BATCH) {
        unsigned a_batch[MMA_BATCH][4];
        unsigned b0_batch[MMA_BATCH][2];
        unsigned b1_batch[MMA_BATCH][2];
        #pragma unroll
        for (int batch = 0; batch < MMA_BATCH; ++batch) {
          w4_prepare_mma_fragments<BLOCK_K, GROUP_POST_SCALE>(
              activation, packed, group, sub_base + batch,
              output_col_base, zero20, zero21, scale20, scale21,
              a_batch[batch], b0_batch[batch], b1_batch[batch]);
        }
        #pragma unroll
        for (int batch = 0; batch < MMA_BATCH; ++batch) {
          if constexpr (GROUP_POST_SCALE) {
            w4_mma(
                a_batch[batch], b0_batch[batch],
                g00, g01, g02, g03);
            w4_mma(
                a_batch[batch], b1_batch[batch],
                g10, g11, g12, g13);
          } else {
            w4_mma(
                a_batch[batch], b0_batch[batch],
                c00, c01, c02, c03);
            w4_mma(
                a_batch[batch], b1_batch[batch],
                c10, c11, c12, c13);
          }
        }
      }
    }
    if constexpr (GROUP_POST_SCALE) {
      // C-fragment columns are owned by lane_in_group, not lane_group
      // (which owns the corresponding B-fragment values).  Only lane_group
      // zero carries the real M=1 row; all other MMA rows were zero-filled.
      if (lane_group == 0) {
        const int c_col0 = output_col_base + lane_in_group * 2;
        const int c_col1 = c_col0 + 1;
        const int c_col8 = c_col0 + 8;
        const int c_col9 = c_col1 + 8;
        const float c_scale0 =
            scales[c_col0 * SCALE_STRIDE + scale_group];
        const float c_scale1 =
            scales[c_col1 * SCALE_STRIDE + scale_group];
        const float c_scale8 =
            scales[c_col8 * SCALE_STRIDE + scale_group];
        const float c_scale9 =
            scales[c_col9 * SCALE_STRIDE + scale_group];
        c00 = fmaf(g00, c_scale0, c00);
        c01 = fmaf(g01, c_scale1, c01);
        c02 = fmaf(g02, c_scale0, c02);
        c03 = fmaf(g03, c_scale1, c03);
        c10 = fmaf(g10, c_scale8, c10);
        c11 = fmaf(g11, c_scale9, c11);
        c12 = fmaf(g12, c_scale8, c12);
        c13 = fmaf(g13, c_scale9, c13);
      }
    }
  }
}

template <int BLOCK_K, int SCALE_STRIDE, int N_TILES>
__device__ __forceinline__ void w4a16_mma_sync_nreuse_tile(
    const void* activation_raw,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zeros,
    int output_col_base,
    int scale_group_base,
    float* accum) {
  const __half* activation =
      reinterpret_cast<const __half*>(activation_raw);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;

  #pragma unroll
  for (int group = 0; group < BLOCK_K / 128; ++group) {
    const int scale_group = scale_group_base + group;
    __half2 zero_frag[N_TILES];
    __half2 scale_frag[N_TILES];
    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
      const int output_col =
          output_col_base + n_tile * 8 + lane_group;
      const unsigned zero = static_cast<unsigned>(
          zeros[output_col * SCALE_STRIDE + scale_group]);
      const unsigned zero_magic =
          0x64006400u + zero + (zero << 16);
      zero_frag[n_tile] =
          *reinterpret_cast<const __half2*>(&zero_magic);
      const __half scale_half = __float2half_rn(
          scales[output_col * SCALE_STRIDE + scale_group]);
      scale_frag[n_tile] =
          __halves2half2(scale_half, scale_half);
    }

    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      const int k_inner = group * 128 + sub * 16;
      unsigned a[4] = {0u, 0u, 0u, 0u};
      if (lane_group == 0) {
        const int k0 = k_inner + lane_in_group * 2;
        const __half2 a01 =
            __halves2half2(activation[k0], activation[k0 + 1]);
        const __half2 a23 =
            __halves2half2(activation[k0 + 8], activation[k0 + 9]);
        a[0] = *reinterpret_cast<const unsigned*>(&a01);
        a[2] = *reinterpret_cast<const unsigned*>(&a23);
      }

      #pragma unroll
      for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
        const int packed_offset =
            ((output_col_base / 8 + n_tile) * (BLOCK_K / 16)
             + k_inner / 16) * 64
            + lane * 2;
        const unsigned packed_pair =
            *reinterpret_cast<const unsigned short*>(
                &packed[packed_offset]);
        const unsigned magic0 = w4_magic_pair(
            static_cast<unsigned char>(packed_pair));
        const unsigned magic1 = w4_magic_pair(
            static_cast<unsigned char>(packed_pair >> 8));
        const __half2 value0 = __hmul2(
            __hsub2(
                *reinterpret_cast<const __half2*>(&magic0),
                zero_frag[n_tile]),
            scale_frag[n_tile]);
        const __half2 value1 = __hmul2(
            __hsub2(
                *reinterpret_cast<const __half2*>(&magic1),
                zero_frag[n_tile]),
            scale_frag[n_tile]);
        unsigned b[2] = {
            *reinterpret_cast<const unsigned*>(&value0),
            *reinterpret_cast<const unsigned*>(&value1),
        };
        w4_mma(
            a,
            b,
            accum[n_tile * 4 + 0],
            accum[n_tile * 4 + 1],
            accum[n_tile * 4 + 2],
            accum[n_tile * 4 + 3]);
      }
    }
  }
}

}  // namespace tl
"""

HALF2_MIXED_DOT_SOURCE = r"""
#include <cuda_fp16.h>

namespace tl {

template <int HALF2_PAIRS, int TOTAL_PAIRS>
__device__ __forceinline__ float w4_dot_half2_mixed_impl(
    const void* activation_raw,
    const void* weight_raw) {
  const __half2* activation =
      reinterpret_cast<const __half2*>(activation_raw);
  const __half2* weight =
      reinterpret_cast<const __half2*>(weight_raw);
  float result = 0.0f;
  #pragma unroll
  for (int pair = 0; pair < HALF2_PAIRS; ++pair) {
    const __half2 product = __hmul2(activation[pair], weight[pair]);
    result += __half2float(__low2half(product));
    result += __half2float(__high2half(product));
  }
  #pragma unroll
  for (int pair = HALF2_PAIRS; pair < TOTAL_PAIRS; ++pair) {
    const __half2 a = activation[pair];
    const __half2 b = weight[pair];
    result = fmaf(
        __half2float(__low2half(a)),
        __half2float(__low2half(b)),
        result);
    result = fmaf(
        __half2float(__high2half(a)),
        __half2float(__high2half(b)),
        result);
  }
  return result;
}

#define TL_DEFINE_W4_MIXED_DOT(NAME, HALF2_PAIRS)                       \
  __device__ __forceinline__ float NAME(                               \
      const void* activation, const void* weight, int total_pairs) {   \
    if (total_pairs == 4) {                                             \
      return w4_dot_half2_mixed_impl<HALF2_PAIRS, 4>(                  \
          activation, weight);                                         \
    }                                                                  \
    return w4_dot_half2_mixed_impl<HALF2_PAIRS, 8>(                    \
        activation, weight);                                           \
  }

TL_DEFINE_W4_MIXED_DOT(w4_dot_half2_1, 1)
TL_DEFINE_W4_MIXED_DOT(w4_dot_half2_2, 2)
TL_DEFINE_W4_MIXED_DOT(w4_dot_half2_3, 3)
TL_DEFINE_W4_MIXED_DOT(w4_dot_half2_4, 4)

#undef TL_DEFINE_W4_MIXED_DOT

template <int SEGMENT_PAIRS, int TOTAL_PAIRS>
__device__ __forceinline__ float w4_dot_half2_accum_impl(
    const void* activation_raw,
    const void* weight_raw) {
  const __half2* activation =
      reinterpret_cast<const __half2*>(activation_raw);
  const __half2* weight =
      reinterpret_cast<const __half2*>(weight_raw);
  float result = 0.0f;
  #pragma unroll
  for (int base = 0; base < TOTAL_PAIRS; base += SEGMENT_PAIRS) {
    __half2 partial = __float2half2_rn(0.0f);
    #pragma unroll
    for (int pair = 0; pair < SEGMENT_PAIRS; ++pair) {
      partial = __hfma2(
          activation[base + pair],
          weight[base + pair],
          partial);
    }
    result += __half2float(__low2half(partial));
    result += __half2float(__high2half(partial));
  }
  return result;
}

#define TL_DEFINE_W4_HALF2_ACCUM(NAME, SEGMENT_PAIRS)                  \
__device__ __forceinline__ float NAME(                                \
    const void* activation, const void* weight, int total_pairs) {    \
  if (total_pairs == 4) {                                             \
    return w4_dot_half2_accum_impl<SEGMENT_PAIRS, 4>(                 \
        activation, weight);                                          \
  }                                                                   \
  return w4_dot_half2_accum_impl<SEGMENT_PAIRS, 8>(                   \
      activation, weight);                                            \
}

TL_DEFINE_W4_HALF2_ACCUM(w4_dot_half2_accum_1, 1)
TL_DEFINE_W4_HALF2_ACCUM(w4_dot_half2_accum_2, 2)
TL_DEFINE_W4_HALF2_ACCUM(w4_dot_half2_accum_4, 4)

#undef TL_DEFINE_W4_HALF2_ACCUM

__device__ __forceinline__ float w4_dot_half2_accum_8(
    const void* activation, const void* weight, int) {
  return w4_dot_half2_accum_impl<8, 8>(activation, weight);
}

}  // namespace tl
"""


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = GROUP_SIZE,
    quant_mode: str = "symmetric",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise quantize and pack a logical ``[N, K]`` weight tensor."""
    if quant_mode not in ("symmetric", "affine"):
        raise ValueError(
            f"quant_mode must be symmetric or affine, got {quant_mode}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if k % 2 != 0:
        raise ValueError(f"K must be even for nibble packing, got {k}")

    grouped = weight.float().reshape(n, k // group_size, group_size)
    if quant_mode == "symmetric":
        scale = grouped.abs().amax(dim=-1).clamp_min(1e-12) / 7.0
        zero = torch.full_like(scale, 8, dtype=torch.uint8)
        quantized = (
            torch.round(grouped / scale.unsqueeze(-1)) + 8
        ).clamp(1, 15).to(torch.uint8)
    else:
        group_min = grouped.amin(dim=-1)
        group_max = grouped.amax(dim=-1)
        scale = ((group_max - group_min) / 15.0).clamp_min(1e-12)
        zero = torch.round(-group_min / scale).clamp(0, 15).to(torch.uint8)
        quantized = torch.round(
            grouped / scale.unsqueeze(-1)
            + zero.float().unsqueeze(-1)
        ).clamp(0, 15).to(torch.uint8)

    unsigned = quantized.reshape(n, k)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    dequantized = (
        (quantized.float() - zero.float().unsqueeze(-1))
        * scale.unsqueeze(-1)
    ).reshape(n, k)
    return (
        packed.contiguous(),
        scale.contiguous(),
        zero.contiguous(),
        dequantized,
    )


@functools.lru_cache(maxsize=16)
def w4a16_gemm_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
    decode_mode: str = "staged",
    force_mma: bool = False,
) -> Callable:
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if decode_mode not in ("scalar", "staged", "lop3"):
        raise ValueError(
            f"decode_mode must be scalar, staged, or lop3, got {decode_mode}")
    if decode_mode == "lop3" and dtype != "float16":
        raise ValueError("TileLang INT4 LOP3 decoding currently requires float16")

    decode_unsigned_int4 = _tir_packed_to_unsigned_convert("uint", 8)
    lop3_source = ""
    lop3_func = ""
    if decode_mode == "lop3":
        lop3_group = get_lop3_intrin_group(
            out_dtype=T.float16,
            source_format=T.uint,
            source_bit=4,
            storage_dtype=T.int8,
            # TileLang 0.1.12's fused scaling intrinsic emits
            # __pack_half2(cutlass::half_t, ...), which does not compile under
            # NVCC. Decode with LOP3 first and vector-multiply the scale below.
            with_scaling=False,
        )
        lop3_source = lop3_group["c_source"]
        lop3_func = lop3_group["func_name"]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_WGMMA: force_mma,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: force_mma,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 2,
        threads: int = 128,
    ) -> Callable:
        if group_size % block_k != 0:
            raise ValueError(
                f"group_size={group_size} must be divisible by block_k={block_k}")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),
            packed_weight: T.Tensor((n, k // 2), "uint8"),
            weight_scale: T.Tensor((n, k // group_size), "float32"),
            weight_zero: T.Tensor((n, k // group_size), "uint8"),
            output: T.Tensor((m, n), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                activation_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                output_local = T.alloc_fragment((block_m, block_n), "float")
                if decode_mode in ("staged", "lop3"):
                    packed_weight_shared = T.alloc_shared(
                        (block_n, block_k // 2), "uint8")
                    weight_scale_shared = T.alloc_shared((block_n,), "float32")
                    weight_zero_shared = T.alloc_shared((block_n,), "uint8")
                    # One 128-bit A16 transaction is eight FP16/BF16 values,
                    # sourced from four packed INT4 bytes.
                    packed_local = T.alloc_local((4,), "uint8")
                    dequantized_local = T.alloc_local((8,), dtype)
                    scale_local = T.alloc_local((1,), dtype)
                    zero_local = T.alloc_local((1,), dtype)
                    if decode_mode == "lop3":
                        T.import_source(lop3_source)

                T.annotate_layout({
                    activation_shared: tilelang.layout.make_swizzled_layout(
                        activation_shared),
                    weight_shared: tilelang.layout.make_swizzled_layout(weight_shared),
                })

                m_start = by * block_m
                n_start = bx * block_n
                T.clear(output_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k
                    for i, j in T.Parallel(block_m, block_k):
                        activation_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            activation[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )

                    if decode_mode in ("staged", "lop3"):
                        # Model workloads are block-aligned and take the vector
                        # copy path. Keep a guarded path so arbitrary N tails
                        # remain a valid experiment input.
                        if n % block_n == 0:
                            T.copy(
                                packed_weight[
                                    n_start,
                                    k_start // 2,
                                ],
                                packed_weight_shared,
                            )
                        else:
                            for i, j in T.Parallel(block_n, block_k // 2):
                                packed_weight_shared[i, j] = T.if_then_else(
                                    n_start + i < n,
                                    packed_weight[
                                        n_start + i,
                                        k_start // 2 + j,
                                    ],
                                    T.cast(0, "uint8"),
                                )

                        for i in T.Parallel(block_n):
                            weight_scale_shared[i] = T.if_then_else(
                                n_start + i < n,
                                weight_scale[
                                    n_start + i,
                                    k_start // group_size,
                                ],
                                T.cast(0, "float32"),
                            )
                            weight_zero_shared[i] = T.if_then_else(
                                n_start + i < n,
                                weight_zero[
                                    n_start + i,
                                    k_start // group_size,
                                ],
                                T.cast(0, "uint8"),
                            )

                        tx = T.get_thread_binding(0)
                        for chunk in T.serial(
                            block_n * block_k // 2 // (threads * 4)
                        ):
                            packed_base = chunk * threads * 4 + tx * 4
                            scale_row = packed_base // (block_k // 2)
                            scale_local[0] = T.cast(
                                weight_scale_shared[scale_row], dtype)
                            zero_local[0] = T.cast(
                                weight_zero_shared[scale_row], dtype)

                            for v in T.vectorized(4):
                                packed_index = packed_base + v
                                packed_local[v] = packed_weight_shared[
                                    packed_index // (block_k // 2),
                                    packed_index % (block_k // 2),
                                ]

                            if decode_mode == "lop3":
                                T.call_extern(
                                    lop3_func,
                                    T.access_ptr(
                                        packed_local[0], "r", 4),
                                    T.access_ptr(
                                        dequantized_local[0], "w", 8),
                                    dtype=dtype,
                                )
                                for v in T.vectorized(8):
                                    dequantized_local[v] = (
                                        dequantized_local[v] - zero_local[0]
                                    ) * scale_local[0]
                            else:
                                for v in T.serial(8):
                                    dequantized_local[v] = (
                                        T.cast(
                                            decode_unsigned_int4(
                                                4,
                                                packed_local[v // 2],
                                                v % 2,
                                                dtype,
                                            ),
                                            dtype,
                                        )
                                        - zero_local[0]
                                    ) * scale_local[0]

                            for v in T.vectorized(8):
                                output_index = (
                                    chunk * threads * 8 + tx * 8 + v
                                )
                                weight_shared[
                                    output_index // block_k,
                                    output_index % block_k,
                                ] = dequantized_local[v]
                    else:
                        for i, j in T.Parallel(block_n, block_k):
                            logical_k = k_start + j
                            packed = T.if_then_else(
                                (n_start + i < n) & (logical_k < k),
                                T.cast(
                                    packed_weight[
                                        n_start + i,
                                        logical_k // 2,
                                    ],
                                    "int32",
                                ),
                                T.cast(0, "int32"),
                            )
                            nibble = T.if_then_else(
                                logical_k % 2 == 0,
                                T.bitwise_and(packed, T.cast(15, "int32")),
                                packed >> T.cast(4, "int32"),
                            )
                            weight_shared[i, j] = T.if_then_else(
                                (n_start + i < n) & (logical_k < k),
                                (
                                    T.cast(nibble, dtype)
                                    - T.cast(
                                        weight_zero[
                                            n_start + i,
                                            logical_k // group_size,
                                        ],
                                        dtype,
                                    )
                                ) * T.cast(
                                    weight_scale[
                                        n_start + i,
                                        logical_k // group_size,
                                    ],
                                    dtype,
                                ),
                                T.cast(0, dtype),
                            )

                    T.gemm(
                        activation_shared,
                        weight_shared,
                        output_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        output[m_start + i, n_start + j] = output_local[i, j]

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_gemm_2wg_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """Explicit two-warpgroup W4A16 pipeline for Hopper.

    WG0 loads A/packed-W/metadata and dequantizes the next K tile. WG1 consumes
    the resulting A16 tiles with WGMMA. The auto warp-specialization pass is
    disabled; all ring-buffer barriers are explicit.
    """
    if dtype != "float16":
        raise ValueError("The experimental two-WG LOP3 path currently requires float16")
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    lop3_group = get_lop3_intrin_group(
        out_dtype=T.float16,
        source_format=T.uint,
        source_bit=4,
        storage_dtype=T.int8,
        with_scaling=False,
    )
    lop3_source = lop3_group["c_source"]
    lop3_func = lop3_group["func_name"]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 2,
    ) -> Callable:
        threads = 256
        producer_threads = 128
        if group_size % block_k != 0:
            raise ValueError(
                f"group_size={group_size} must be divisible by block_k={block_k}")
        if k % block_k != 0:
            raise ValueError(f"K={k} must be divisible by block_k={block_k}")
        if block_n * block_k // 2 % (producer_threads * 4) != 0:
            raise ValueError("packed tile must divide into four-byte producer chunks")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),
            packed_weight: T.Tensor((n, k // 2), "uint8"),
            weight_scale: T.Tensor((n, k // group_size), "float32"),
            weight_zero: T.Tensor((n, k // group_size), "uint8"),
            output: T.Tensor((m, n), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                activation_smem = T.alloc_shared(
                    (num_stages, block_m, block_k), dtype)
                packed_smem = T.alloc_shared(
                    (num_stages, block_n, block_k // 2), "uint8")
                scale_smem = T.alloc_shared(
                    (num_stages, block_n), "float32")
                zero_smem = T.alloc_shared(
                    (num_stages, block_n), "uint8")
                dequantized_smem = T.alloc_shared(
                    (num_stages, block_n, block_k), dtype)
                output_local = T.alloc_fragment(
                    (block_m, block_n), "float32")

                packed_local = T.alloc_local((4,), "uint8")
                dequantized_local = T.alloc_local((8,), dtype)
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)

                T.annotate_layout({
                    activation_smem: tilelang.layout.make_swizzled_layout(
                        activation_smem),
                    dequantized_smem: tilelang.layout.make_swizzled_layout(
                        dequantized_smem),
                })

                load_full = T.alloc_barrier([producer_threads] * num_stages)
                ready = T.alloc_barrier([producer_threads] * num_stages)
                empty = T.alloc_barrier([producer_threads] * num_stages)
                gi_producer = T.alloc_var("int32", init=0)
                gi_consumer = T.alloc_var("int32", init=0)

                m_start = by * block_m
                n_start = bx * block_n
                tx = T.get_thread_binding()
                T.import_source(lop3_source)

                if tx < producer_threads:
                    T.dec_max_nreg(48)
                    for ki in T.serial(k // block_k):
                        stage = gi_producer % num_stages
                        phase = (gi_producer // num_stages) % 2
                        k_start = ki * block_k
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(empty[s], phase ^ 1)
                                T.tma_copy(
                                    activation[
                                        m_start:m_start + block_m,
                                        k_start:k_start + block_k,
                                    ],
                                    activation_smem[s, :, :],
                                    barrier=load_full[s],
                                )
                                T.tma_copy(
                                    packed_weight[
                                        n_start:n_start + block_n,
                                        k_start // 2:(k_start + block_k) // 2,
                                    ],
                                    packed_smem[s, :, :],
                                    barrier=load_full[s],
                                )
                                for i in T.Parallel(block_n):
                                    scale_smem[s, i] = T.if_then_else(
                                        n_start + i < n,
                                        weight_scale[
                                            n_start + i,
                                            k_start // group_size,
                                        ],
                                        T.cast(0, "float32"),
                                    )
                                    zero_smem[s, i] = T.if_then_else(
                                        n_start + i < n,
                                        weight_zero[
                                            n_start + i,
                                            k_start // group_size,
                                        ],
                                        T.cast(0, "uint8"),
                                    )
                                T.barrier_arrive(load_full[s])
                                T.barrier_wait(load_full[s], phase)

                                for chunk in T.serial(
                                    block_n * block_k // 2
                                    // (producer_threads * 4)
                                ):
                                    packed_base = (
                                        chunk * producer_threads * 4 + tx * 4
                                    )
                                    scale_row = packed_base // (block_k // 2)
                                    scale_local[0] = T.cast(
                                        scale_smem[s, scale_row], dtype)
                                    zero_local[0] = T.cast(
                                        zero_smem[s, scale_row], dtype)
                                    for v in T.vectorized(4):
                                        packed_index = packed_base + v
                                        packed_local[v] = packed_smem[
                                            s,
                                            packed_index // (block_k // 2),
                                            packed_index % (block_k // 2),
                                        ]
                                    T.call_extern(
                                        lop3_func,
                                        T.access_ptr(
                                            packed_local[0], "r", 4),
                                        T.access_ptr(
                                            dequantized_local[0], "w", 8),
                                        dtype=dtype,
                                    )
                                    for v in T.vectorized(8):
                                        dequantized_local[v] = (
                                            dequantized_local[v] - zero_local[0]
                                        ) * scale_local[0]
                                    for v in T.vectorized(8):
                                        output_index = (
                                            chunk * producer_threads * 8
                                            + tx * 8
                                            + v
                                        )
                                        dequantized_smem[
                                            s,
                                            output_index // block_k,
                                            output_index % block_k,
                                        ] = dequantized_local[v]
                                # Dequant writes use the generic shared-memory
                                # proxy; WGMMA consumes through the async proxy
                                # in a different warpgroup.
                                T.fence_proxy_async()
                                T.barrier_arrive(ready[s])
                        gi_producer = gi_producer + 1
                else:
                    T.inc_max_nreg(224)
                    T.clear(output_local)
                    for ki in T.serial(k // block_k):
                        stage = gi_consumer % num_stages
                        phase = (gi_consumer // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(ready[s], phase)
                                T.wgmma_gemm(
                                    activation_smem[s, :, :],
                                    dequantized_smem[s, :, :],
                                    output_local,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(
                                    output_local, num_regs=64)
                                T.barrier_arrive(empty[s])
                        gi_consumer = gi_consumer + 1

                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            output[m_start + i, n_start + j] = output_local[i, j]

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_gemm_3wg_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """Explicit three-warpgroup load/dequant/WGMMA pipeline for Hopper."""
    if dtype != "float16":
        raise ValueError(
            "The experimental three-WG LOP3 path currently requires float16")
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    lop3_group = get_lop3_intrin_group(
        out_dtype=T.float16,
        source_format=T.uint,
        source_bit=4,
        storage_dtype=T.int8,
        with_scaling=False,
    )
    lop3_source = lop3_group["c_source"]
    lop3_func = lop3_group["func_name"]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 3,
    ) -> Callable:
        threads = 384
        warpgroup_threads = 128
        if group_size % block_k != 0:
            raise ValueError(
                f"group_size={group_size} must be divisible by block_k={block_k}")
        if k % block_k != 0:
            raise ValueError(f"K={k} must be divisible by block_k={block_k}")
        if block_n * block_k // 2 % (warpgroup_threads * 4) != 0:
            raise ValueError("packed tile must divide into four-byte dequant chunks")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),
            packed_weight: T.Tensor((n, k // 2), "uint8"),
            weight_scale: T.Tensor((n, k // group_size), "float32"),
            weight_zero: T.Tensor((n, k // group_size), "uint8"),
            output: T.Tensor((m, n), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                activation_smem = T.alloc_shared(
                    (num_stages, block_m, block_k), dtype)
                packed_smem = T.alloc_shared(
                    (num_stages, block_n, block_k // 2), "uint8")
                scale_smem = T.alloc_shared(
                    (num_stages, block_n), "float32")
                zero_smem = T.alloc_shared(
                    (num_stages, block_n), "uint8")
                dequantized_smem = T.alloc_shared(
                    (num_stages, block_n, block_k), dtype)
                output_local = T.alloc_fragment(
                    (block_m, block_n), "float32")

                packed_local = T.alloc_local((4,), "uint8")
                dequantized_local = T.alloc_local((8,), dtype)
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)

                T.annotate_layout({
                    activation_smem: tilelang.layout.make_swizzled_layout(
                        activation_smem),
                    dequantized_smem: tilelang.layout.make_swizzled_layout(
                        dequantized_smem),
                })

                load_full = T.alloc_barrier([warpgroup_threads] * num_stages)
                ready = T.alloc_barrier([warpgroup_threads] * num_stages)
                empty = T.alloc_barrier([warpgroup_threads] * num_stages)
                gi_loader = T.alloc_var("int32", init=0)
                gi_dequant = T.alloc_var("int32", init=0)
                gi_compute = T.alloc_var("int32", init=0)

                m_start = by * block_m
                n_start = bx * block_n
                tx = T.get_thread_binding()
                T.import_source(lop3_source)

                if tx < warpgroup_threads:
                    # WG0: fill packed/A/metadata ring slots.
                    T.dec_max_nreg(24)
                    for ki in T.serial(k // block_k):
                        stage = gi_loader % num_stages
                        phase = (gi_loader // num_stages) % 2
                        k_start = ki * block_k
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(empty[s], phase ^ 1)
                                T.tma_copy(
                                    activation[
                                        m_start:m_start + block_m,
                                        k_start:k_start + block_k,
                                    ],
                                    activation_smem[s, :, :],
                                    barrier=load_full[s],
                                )
                                T.tma_copy(
                                    packed_weight[
                                        n_start:n_start + block_n,
                                        k_start // 2:(k_start + block_k) // 2,
                                    ],
                                    packed_smem[s, :, :],
                                    barrier=load_full[s],
                                )
                                for i in T.Parallel(block_n):
                                    scale_smem[s, i] = T.if_then_else(
                                        n_start + i < n,
                                        weight_scale[
                                            n_start + i,
                                            k_start // group_size,
                                        ],
                                        T.cast(0, "float32"),
                                    )
                                    zero_smem[s, i] = T.if_then_else(
                                        n_start + i < n,
                                        weight_zero[
                                            n_start + i,
                                            k_start // group_size,
                                        ],
                                        T.cast(0, "uint8"),
                                    )
                                T.barrier_arrive(load_full[s])
                        gi_loader = gi_loader + 1
                elif tx < 2 * warpgroup_threads:
                    # WG1: decode the loaded packed tile into A16 shared memory.
                    T.dec_max_nreg(64)
                    local_tx = tx - warpgroup_threads
                    for ki in T.serial(k // block_k):
                        stage = gi_dequant % num_stages
                        phase = (gi_dequant // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(load_full[s], phase)
                                for chunk in T.serial(
                                    block_n * block_k // 2
                                    // (warpgroup_threads * 4)
                                ):
                                    packed_base = (
                                        chunk * warpgroup_threads * 4
                                        + local_tx * 4
                                    )
                                    scale_row = packed_base // (block_k // 2)
                                    scale_local[0] = T.cast(
                                        scale_smem[s, scale_row], dtype)
                                    zero_local[0] = T.cast(
                                        zero_smem[s, scale_row], dtype)
                                    for v in T.vectorized(4):
                                        packed_index = packed_base + v
                                        packed_local[v] = packed_smem[
                                            s,
                                            packed_index // (block_k // 2),
                                            packed_index % (block_k // 2),
                                        ]
                                    T.call_extern(
                                        lop3_func,
                                        T.access_ptr(
                                            packed_local[0], "r", 4),
                                        T.access_ptr(
                                            dequantized_local[0], "w", 8),
                                        dtype=dtype,
                                    )
                                    for v in T.vectorized(8):
                                        dequantized_local[v] = (
                                            dequantized_local[v] - zero_local[0]
                                        ) * scale_local[0]
                                    for v in T.vectorized(8):
                                        output_index = (
                                            chunk * warpgroup_threads * 8
                                            + local_tx * 8
                                            + v
                                        )
                                        dequantized_smem[
                                            s,
                                            output_index // block_k,
                                            output_index % block_k,
                                        ] = dequantized_local[v]
                                T.fence_proxy_async()
                                T.barrier_arrive(ready[s])
                        gi_dequant = gi_dequant + 1
                else:
                    # WG2: consume A16 tiles with WGMMA.
                    T.inc_max_nreg(224)
                    T.clear(output_local)
                    for ki in T.serial(k // block_k):
                        stage = gi_compute % num_stages
                        phase = (gi_compute // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(ready[s], phase)
                                T.wgmma_gemm(
                                    activation_smem[s, :, :],
                                    dequantized_smem[s, :, :],
                                    output_local,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(
                                    output_local, num_regs=64)
                                T.barrier_arrive(empty[s])
                        gi_compute = gi_compute + 1

                    for i, j in T.Parallel(block_m, block_n):
                        if m_start + i < m and n_start + j < n:
                            output[m_start + i, n_start + j] = output_local[i, j]

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_decode_wgmma_nt_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """M=1 W4A16 as ``W[N,K] @ A.T[K,1]`` using async Hopper WGMMA.

    The transposed formulation maps 64 real output channels to WGMMA's fixed
    64-row M tile and pads only the one-column N dimension to eight.  Three
    warpgroups independently load, dequantize, and consume the K-tile ring.
    """
    if m != 1:
        raise ValueError("The transposed WGMMA decode experiment requires M=1")
    if dtype != "float16":
        raise ValueError("The transposed WGMMA decode path requires float16")
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    lop3_group = get_lop3_intrin_group(
        out_dtype=T.float16,
        source_format=T.uint,
        source_bit=4,
        storage_dtype=T.int8,
        with_scaling=False,
    )
    lop3_source = lop3_group["c_source"]
    lop3_func = lop3_group["func_name"]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_out: int = 64,
        block_k: int = 128,
        num_stages: int = 2,
        wgmma_depth: int = 0,
        tma_activation: bool = False,
    ) -> Callable:
        block_batch = 8
        threads = 384
        warpgroup_threads = 128
        groups_per_tile = max(1, block_k // group_size)
        activation_tma_block = min(block_k, 256)
        activation_tma_partitions = block_k // activation_tma_block
        if block_out != 64:
            raise ValueError("WGMMA M tile requires block_out=64")
        if n % block_out != 0:
            raise ValueError(f"N={n} must be divisible by block_out={block_out}")
        if (
            group_size % block_k != 0
            and block_k % group_size != 0
        ):
            raise ValueError(
                "block_k and group_size must divide one another, got "
                f"block_k={block_k}, group_size={group_size}")
        if k % block_k != 0:
            raise ValueError(f"K={k} must be divisible by block_k={block_k}")
        if block_k % activation_tma_block != 0:
            raise ValueError(
                "block_k must divide into contiguous activation TMA tiles")
        if block_out * block_k // 2 % (warpgroup_threads * 4) != 0:
            raise ValueError("packed tile must divide into four-byte dequant chunks")
        if wgmma_depth not in (0, 1, 2):
            raise ValueError("wgmma_depth must be 0, 1, or 2")
        if wgmma_depth > 0 and num_stages <= wgmma_depth:
            raise ValueError(
                "num_stages must exceed the number of in-flight WGMMA groups")

        @T.prim_func
        def main(
            activation: T.Tensor((1, k), dtype),
            packed_weight: T.Tensor((n, k // 2), "uint8"),
            weight_scale: T.Tensor((n, k // group_size), "float32"),
            weight_zero: T.Tensor((n, k // group_size), "uint8"),
            output: T.Tensor((1, n), dtype),
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_out), threads=threads) as bx:
                activation_tma_smem = T.alloc_shared(
                    (
                        num_stages,
                        activation_tma_partitions,
                        activation_tma_block,
                    ),
                    dtype,
                )
                activation_smem = T.alloc_shared(
                    (num_stages, block_batch, block_k), dtype)
                packed_smem = T.alloc_shared(
                    (num_stages, block_out, block_k // 2), "uint8")
                scale_smem = T.alloc_shared(
                    (
                        num_stages,
                        block_out,
                        groups_per_tile,
                    ),
                    "float32",
                )
                zero_smem = T.alloc_shared(
                    (
                        num_stages,
                        block_out,
                        groups_per_tile,
                    ),
                    "uint8",
                )
                weight_smem = T.alloc_shared(
                    (num_stages, block_out, block_k), dtype)
                output_local = T.alloc_fragment(
                    (block_out, block_batch), "float32")

                packed_local = T.alloc_local((4,), "uint8")
                dequantized_local = T.alloc_local((8,), dtype)
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)

                T.annotate_layout({
                    activation_smem: tilelang.layout.make_swizzled_layout(
                        activation_smem),
                    weight_smem: tilelang.layout.make_swizzled_layout(
                        weight_smem),
                })

                load_full = T.alloc_barrier(
                    [warpgroup_threads] * num_stages)
                ready = T.alloc_barrier(
                    [warpgroup_threads] * num_stages)
                empty = T.alloc_barrier(
                    [warpgroup_threads] * num_stages)
                gi_loader = T.alloc_var("int32", init=0)
                gi_dequant = T.alloc_var("int32", init=0)
                gi_compute = T.alloc_var("int32", init=0)
                pending_compute_stages = T.alloc_local((2,), "int32")

                n_start = bx * block_out
                tx = T.get_thread_binding()
                T.import_source(lop3_source)

                if tx < warpgroup_threads:
                    # WG0: load one real activation row, pad seven rows, and
                    # asynchronously fetch packed weights and metadata.
                    T.dec_max_nreg(24)
                    for ki in T.serial(k // block_k):
                        stage = gi_loader % num_stages
                        phase = (gi_loader // num_stages) % 2
                        k_start = ki * block_k
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(empty[s], phase ^ 1)
                                if tma_activation:
                                    for part in range(
                                        activation_tma_partitions
                                    ):
                                        part_k_start = (
                                            k_start
                                            + part * activation_tma_block
                                        )
                                        T.tma_copy(
                                            activation[
                                                0,
                                                part_k_start:
                                                (
                                                    part_k_start
                                                    + activation_tma_block
                                                ),
                                            ],
                                            activation_tma_smem[
                                                s, part, :
                                            ],
                                            barrier=load_full[s],
                                        )
                                else:
                                    for i, j in T.Parallel(
                                        block_batch, block_k
                                    ):
                                        activation_smem[s, i, j] = (
                                            T.if_then_else(
                                                i == 0,
                                                activation[0, k_start + j],
                                                T.cast(0, dtype),
                                            )
                                        )
                                    T.fence_proxy_async()
                                T.tma_copy(
                                    packed_weight[
                                        n_start:n_start + block_out,
                                        k_start // 2:(k_start + block_k) // 2,
                                    ],
                                    packed_smem[s, :, :],
                                    barrier=load_full[s],
                                )
                                for i, g in T.Parallel(
                                    block_out, groups_per_tile
                                ):
                                    scale_smem[s, i, g] = weight_scale[
                                        n_start + i,
                                        k_start // group_size + g,
                                    ]
                                    zero_smem[s, i, g] = weight_zero[
                                        n_start + i,
                                        k_start // group_size + g,
                                    ]
                                T.barrier_arrive(load_full[s])
                        gi_loader = gi_loader + 1
                elif tx < 2 * warpgroup_threads:
                    # WG1: LOP3 decode directly from the packed ring into the
                    # FP16 WGMMA A tile.
                    T.dec_max_nreg(64)
                    local_tx = tx - warpgroup_threads
                    for ki in T.serial(k // block_k):
                        stage = gi_dequant % num_stages
                        phase = (gi_dequant // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(load_full[s], phase)
                                if tma_activation:
                                    # Present one complete logical FP16 tile to
                                    # layout inference. TileLang maps this into
                                    # the physical WGMMA shared-memory swizzle.
                                    for i, j in T.Parallel(
                                        block_batch, block_k
                                    ):
                                        activation_smem[s, i, j] = (
                                            T.if_then_else(
                                                i == 0,
                                                activation_tma_smem[
                                                    s,
                                                    (
                                                        j
                                                        // activation_tma_block
                                                    ),
                                                    (
                                                        j
                                                        % activation_tma_block
                                                    ),
                                                ],
                                                T.cast(0, dtype),
                                            )
                                        )
                                for chunk in T.serial(
                                    block_out * block_k // 2
                                    // (warpgroup_threads * 4)
                                ):
                                    packed_base = (
                                        chunk * warpgroup_threads * 4
                                        + local_tx * 4
                                    )
                                    scale_row = packed_base // (block_k // 2)
                                    packed_in_row = (
                                        packed_base % (block_k // 2))
                                    scale_group = (
                                        packed_in_row * 2
                                    ) // group_size
                                    scale_local[0] = T.cast(
                                        scale_smem[
                                            s, scale_row, scale_group
                                        ],
                                        dtype,
                                    )
                                    zero_local[0] = T.cast(
                                        zero_smem[
                                            s, scale_row, scale_group
                                        ],
                                        dtype,
                                    )
                                    for v in T.vectorized(4):
                                        packed_index = packed_base + v
                                        packed_local[v] = packed_smem[
                                            s,
                                            packed_index // (block_k // 2),
                                            packed_index % (block_k // 2),
                                        ]
                                    T.call_extern(
                                        lop3_func,
                                        T.access_ptr(
                                            packed_local[0], "r", 4),
                                        T.access_ptr(
                                            dequantized_local[0], "w", 8),
                                        dtype=dtype,
                                    )
                                    for v in T.vectorized(8):
                                        dequantized_local[v] = (
                                            dequantized_local[v]
                                            - zero_local[0]
                                        ) * scale_local[0]
                                    for v in T.vectorized(8):
                                        output_index = (
                                            chunk * warpgroup_threads * 8
                                            + local_tx * 8
                                            + v
                                        )
                                        weight_smem[
                                            s,
                                            output_index // block_k,
                                            output_index % block_k,
                                        ] = dequantized_local[v]
                                T.fence_proxy_async()
                                T.barrier_arrive(ready[s])
                        gi_dequant = gi_dequant + 1
                else:
                    # WG2: C.T[64,8] = W[64,K] @ padded_A.T[K,8].
                    T.inc_max_nreg(224)
                    T.clear(output_local)
                    for ki in T.serial(k // block_k):
                        stage = gi_compute % num_stages
                        phase = (gi_compute // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(ready[s], phase)
                                T.wgmma_gemm(
                                    weight_smem[s, :, :],
                                    activation_smem[s, :, :],
                                    output_local,
                                    transpose_B=True,
                                    policy=T.GemmWarpPolicy.FullRow,
                                    clear_accum=(ki == 0),
                                )
                                if wgmma_depth == 0:
                                    T.wait_wgmma(0)
                                    T.barrier_arrive(empty[s])
                                elif ki >= wgmma_depth:
                                    # Keep `wgmma_depth` committed groups in
                                    # flight and release only the oldest
                                    # shared-memory stage proven complete.
                                    T.wait_wgmma(wgmma_depth)
                                    T.barrier_arrive(
                                        empty[
                                            pending_compute_stages[
                                                ki % wgmma_depth
                                            ]
                                        ])
                                if wgmma_depth > 0:
                                    pending_compute_stages[
                                        ki % wgmma_depth
                                    ] = s
                        gi_compute = gi_compute + 1

                    if wgmma_depth > 0:
                        T.wait_wgmma(0)
                        for pending in T.serial(wgmma_depth):
                            T.barrier_arrive(
                                empty[pending_compute_stages[pending]])
                    T.warpgroup_fence_operand(
                        output_local, num_regs=64)
                    for i, j in T.Parallel(block_out, block_batch):
                        if j == 0:
                            output[0, n_start + i] = output_local[i, j]

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_decode_wgmma_rs_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """M=1 W4A16 with register-dequant and register/shared WGMMA.

    WG0 loads packed INT4 weights and a padded activation tile.  WG1 decodes
    the weight tile directly into WGMMA's register A operand and computes
    ``W[64,K] @ padded_A.T[K,8]``.  Keeping one WGMMA group in flight overlaps
    its Tensor Core work with decoding the following packed tile.
    """
    if m != 1:
        raise ValueError("The register/shared WGMMA experiment requires M=1")
    if dtype != "float16":
        raise ValueError("The register/shared WGMMA path requires float16")
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    lop3_group = get_lop3_intrin_group(
        out_dtype=T.float16,
        source_format=T.uint,
        source_bit=4,
        storage_dtype=T.int8,
        with_scaling=False,
    )
    lop3_source = lop3_group["c_source"]
    lop3_func = lop3_group["func_name"]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_out: int = 64,
        block_k: int = 128,
        num_stages: int = 2,
        wgmma_depth: int = 0,
        post_scale_wgmma: bool = False,
    ) -> Callable:
        block_batch = 8
        threads = 256
        warpgroup_threads = 128
        if block_out != 64:
            raise ValueError("WGMMA M tile requires block_out=64")
        if n % block_out != 0:
            raise ValueError(f"N={n} must be divisible by block_out={block_out}")
        if group_size % block_k != 0:
            raise ValueError(
                f"group_size={group_size} must be divisible by block_k={block_k}")
        if k % block_k != 0:
            raise ValueError(f"K={k} must be divisible by block_k={block_k}")
        if wgmma_depth not in (0, 1):
            raise ValueError("wgmma_depth must be 0 or 1")
        if wgmma_depth > 0 and num_stages < 2:
            raise ValueError("async register/shared WGMMA requires two stages")
        if post_scale_wgmma and wgmma_depth:
            raise ValueError(
                "post-scale WGMMA currently requires depth zero")

        @T.prim_func
        def main(
            activation: T.Tensor((1, k), dtype),
            packed_weight: T.Tensor((n, k // 2), "uint8"),
            weight_scale: T.Tensor((n, k // group_size), "float32"),
            weight_zero: T.Tensor((n, k // group_size), "uint8"),
            output: T.Tensor((1, n), dtype),
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_out), threads=threads) as bx:
                activation_smem = T.alloc_shared(
                    (num_stages, block_batch, block_k), dtype)
                packed_smem = T.alloc_shared(
                    (num_stages, block_out, block_k // 2), "uint8")
                scale_smem = T.alloc_shared(
                    (num_stages, block_out), "float32")
                zero_smem = T.alloc_shared(
                    (num_stages, block_out), "uint8")

                weight_fragment = T.alloc_fragment(
                    (block_out, block_k), dtype)
                output_local = T.alloc_fragment(
                    (block_out, block_batch), "float32")
                group_output_local = T.alloc_fragment(
                    (block_out, block_batch), "float32")
                packed_local = T.alloc_local((4,), "uint8")
                dequantized_local = T.alloc_local((8,), dtype)
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)

                T.annotate_layout({
                    activation_smem: tilelang.layout.make_swizzled_layout(
                        activation_smem),
                })

                ready = T.alloc_barrier(
                    [warpgroup_threads] * num_stages)
                empty = T.alloc_barrier(
                    [warpgroup_threads] * num_stages)
                gi_loader = T.alloc_var("int32", init=0)
                gi_compute = T.alloc_var("int32", init=0)
                previous_compute_stage = T.alloc_local((1,), "int32")

                n_start = bx * block_out
                tx = T.get_thread_binding()
                T.import_source(lop3_source)

                if tx < warpgroup_threads:
                    # WG0: packed-weight TMA plus a single real activation
                    # row; the other seven rows are WGMMA padding.
                    T.dec_max_nreg(24)
                    for ki in T.serial(k // block_k):
                        stage = gi_loader % num_stages
                        phase = (gi_loader // num_stages) % 2
                        k_start = ki * block_k
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(empty[s], phase ^ 1)
                                for i, j in T.Parallel(
                                    block_batch, block_k
                                ):
                                    activation_smem[s, i, j] = (
                                        T.if_then_else(
                                            i == 0,
                                            activation[0, k_start + j],
                                            T.cast(0, dtype),
                                        )
                                    )
                                T.fence_proxy_async()
                                T.tma_copy(
                                    packed_weight[
                                        n_start:n_start + block_out,
                                        k_start // 2:(k_start + block_k) // 2,
                                    ],
                                    packed_smem[s, :, :],
                                    barrier=ready[s],
                                )
                                for i in T.Parallel(block_out):
                                    scale_smem[s, i] = weight_scale[
                                        n_start + i,
                                        k_start // group_size,
                                    ]
                                    zero_smem[s, i] = weight_zero[
                                        n_start + i,
                                        k_start // group_size,
                                    ]
                                T.barrier_arrive(ready[s])
                        gi_loader = gi_loader + 1
                else:
                    # WG1: decode the register A operand while the preceding
                    # WGMMA group remains in flight.
                    T.inc_max_nreg(240 if post_scale_wgmma else 232)
                    T.clear(output_local)
                    for ki in T.serial(k // block_k):
                        stage = gi_compute % num_stages
                        phase = (gi_compute // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(ready[s], phase)
                                # RS-WGMMA distributes each eight-value LOP3
                                # group across four lanes (two values/lane).
                                # Each owner lane decodes the common group and
                                # retains its two register-fragment elements.
                                for i, pair in T.Parallel(
                                    block_out, block_k // 2
                                ):
                                    scale_local[0] = T.cast(
                                        scale_smem[s, i], dtype)
                                    zero_local[0] = T.cast(
                                        zero_smem[s, i], dtype)
                                    for v in T.vectorized(4):
                                        packed_local[v] = packed_smem[
                                            s, i, (pair // 4) * 4 + v
                                        ]
                                    T.call_extern(
                                        lop3_func,
                                        T.access_ptr(
                                            packed_local[0], "r", 4),
                                        T.access_ptr(
                                            dequantized_local[0], "w", 8),
                                        dtype=dtype,
                                    )
                                    for v in T.vectorized(2):
                                        weight_fragment[
                                            i, pair * 2 + v
                                        ] = (
                                            (
                                                dequantized_local[
                                                    (pair % 4) * 2 + v
                                                ]
                                                - zero_local[0]
                                            )
                                            if post_scale_wgmma
                                            else (
                                                (
                                                    dequantized_local[
                                                        (pair % 4) * 2 + v
                                                    ]
                                                    - zero_local[0]
                                                )
                                                * scale_local[0]
                                            )
                                        )

                                if post_scale_wgmma:
                                    T.wgmma_gemm(
                                        weight_fragment,
                                        activation_smem[s, :, :],
                                        group_output_local,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=True,
                                    )
                                    T.wait_wgmma(0)
                                    for i, j in T.Parallel(
                                        block_out, block_batch
                                    ):
                                        if j == 0:
                                            output_local[i, j] += (
                                                group_output_local[i, j]
                                                * scale_smem[s, i]
                                            )
                                    T.barrier_arrive(empty[s])
                                else:
                                    T.wgmma_gemm(
                                        weight_fragment,
                                        activation_smem[s, :, :],
                                        output_local,
                                        transpose_B=True,
                                        policy=T.GemmWarpPolicy.FullRow,
                                        clear_accum=(ki == 0),
                                    )
                                    if wgmma_depth == 0:
                                        T.wait_wgmma(0)
                                        T.barrier_arrive(empty[s])
                                    else:
                                        if ki > 0:
                                            T.wait_wgmma(1)
                                            T.barrier_arrive(
                                                empty[
                                                    previous_compute_stage[0]
                                                ])
                                        previous_compute_stage[0] = s
                        gi_compute = gi_compute + 1

                    if wgmma_depth > 0 and not post_scale_wgmma:
                        T.wait_wgmma(0)
                        T.barrier_arrive(
                            empty[previous_compute_stage[0]])
                    T.warpgroup_fence_operand(
                        output_local, num_regs=64)
                    for i, j in T.Parallel(block_out, block_batch):
                        if j == 0:
                            output[0, n_start + i] = output_local[i, j]

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_gemv_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """Small-M FP16 path using register decode and warp reduction."""
    if dtype != "float16":
        raise ValueError("The experimental LOP3 GEMV path currently requires float16")
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    lop3_group = get_lop3_intrin_group(
        out_dtype=T.float16,
        source_format=T.uint,
        source_bit=4,
        storage_dtype=T.int8,
        with_scaling=False,
    )
    lop3_source = lop3_group["c_source"]
    lop3_func = lop3_group["func_name"]

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def build(
        n_partition: int = 4,
        reduce_threads: int = 32,
        split_k_warps: int = 1,
        outputs_per_warp: int = 1,
    ) -> Callable:
        values_per_thread = 8
        packed_per_thread = values_per_thread // 2
        block_k = reduce_threads * values_per_thread
        split_block_k = block_k * split_k_warps
        total_threads = reduce_threads * n_partition * split_k_warps
        if split_k_warps not in (1, 2, 4, 8, 16):
            raise ValueError(
                "split_k_warps must be one of (1, 2, 4, 8, 16), "
                f"got {split_k_warps}")
        if outputs_per_warp not in (1, 2, 4, 8):
            raise ValueError(
                "outputs_per_warp must be one of (1, 2, 4, 8), "
                f"got {outputs_per_warp}")
        if total_threads > 1024:
            raise ValueError(
                f"GEMV launch requests {total_threads} threads, exceeding 1024")
        if k % split_block_k != 0:
            raise ValueError(
                f"K={k} must be divisible by split GEMV block_k="
                f"{split_block_k}")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),
                packed_weight: T.Tensor((n, k // 2), "uint8"),
                weight_scale: T.Tensor((n, k // group_size), "float32"),
                weight_zero: T.Tensor((n, k // group_size), "uint8"),
                output: T.Tensor((m, n), dtype),
            ) -> None:
            with T.Kernel(
                T.ceildiv(n, n_partition * outputs_per_warp),
                m,
                threads=(reduce_threads, n_partition * split_k_warps),
            ) as (bx, by):
                activation_local = T.alloc_local((values_per_thread,), dtype)
                packed_local = T.alloc_local((packed_per_thread,), "uint8")
                dequantized_local = T.alloc_local((values_per_thread,), dtype)
                accumulator = T.alloc_local(
                    (outputs_per_warp,), "float32")
                cta_reduced = T.alloc_local(
                    (outputs_per_warp,), "float32")
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)
                partials = T.alloc_shared(
                    (
                        n_partition,
                        outputs_per_warp,
                        split_k_warps,
                    ),
                    "float32",
                )

                kr = T.thread_binding(
                    0, reduce_threads, thread="threadIdx.x")
                warp_slot = T.thread_binding(
                    0,
                    n_partition * split_k_warps,
                    thread="threadIdx.y",
                )
                ni = warp_slot // split_k_warps
                k_partition = warp_slot % split_k_warps
                output_col_base = (
                    bx * n_partition * outputs_per_warp
                    + ni * outputs_per_warp
                )

                T.import_source(lop3_source)
                T.clear(accumulator)

                for ko in T.serial(k // split_block_k):
                    logical_k = (
                        (ko * split_k_warps + k_partition) * block_k
                        + kr * values_per_thread
                    )
                    for v in T.vectorized(values_per_thread):
                        activation_local[v] = activation[
                            by, logical_k + v]
                    for output_slot in T.serial(outputs_per_warp):
                        output_col = output_col_base + output_slot
                        for v in T.vectorized(packed_per_thread):
                            packed_local[v] = T.if_then_else(
                                output_col < n,
                                packed_weight[
                                    output_col,
                                    logical_k // 2 + v,
                                ],
                                T.cast(0, "uint8"),
                            )

                        T.call_extern(
                            lop3_func,
                            T.access_ptr(
                                packed_local[0],
                                "r",
                                packed_per_thread,
                            ),
                            T.access_ptr(
                                dequantized_local[0],
                                "w",
                                values_per_thread,
                            ),
                            dtype=dtype,
                        )
                        scale_local[0] = T.if_then_else(
                            output_col < n,
                            T.cast(
                                weight_scale[
                                    output_col,
                                    logical_k // group_size,
                                ],
                                dtype,
                            ),
                            T.cast(0, dtype),
                        )
                        zero_local[0] = T.if_then_else(
                            output_col < n,
                            T.cast(
                                weight_zero[
                                    output_col,
                                    logical_k // group_size,
                                ],
                                dtype,
                            ),
                            T.cast(0, dtype),
                        )

                        for v in T.serial(values_per_thread):
                            accumulator[output_slot] += (
                                T.cast(
                                    activation_local[v], "float32")
                                * T.cast(
                                    (
                                        dequantized_local[v]
                                        - zero_local[0]
                                    ) * scale_local[0],
                                    "float32",
                                )
                            )

                for output_slot in T.serial(outputs_per_warp):
                    for reduction_step in T.serial(5):
                        accumulator[output_slot] += T.shfl_down(
                            accumulator[output_slot],
                            16 >> reduction_step,
                            width=reduce_threads,
                        )

                if split_k_warps == 1:
                    if kr == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            output_col = output_col_base + output_slot
                            if output_col < n:
                                output[
                                    by, output_col
                                ] = accumulator[output_slot]
                else:
                    if kr == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            partials[
                                ni, output_slot, k_partition
                            ] = accumulator[output_slot]
                    T.sync_threads(
                        barrier_id=1,
                        arrive_count=total_threads,
                    )
                    if kr == 0 and k_partition == 0:
                        for output_slot in T.serial(outputs_per_warp):
                            output_col = output_col_base + output_slot
                            cta_reduced[output_slot] = T.cast(
                                0, "float32")
                            for part in T.serial(split_k_warps):
                                cta_reduced[output_slot] += partials[
                                    ni, output_slot, part
                                ]
                            if output_col < n:
                                output[by, output_col] = (
                                    cta_reduced[output_slot]
                                )

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_gemv_tma_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    """Long-K GEMV with TMA-fed N64 tiles and activation reuse.

    One producer warpgroup fills a shared-memory ring with one activation tile
    and 64 packed-weight rows.  Two consumer warpgroups retain eight output
    accumulators per warp, reusing each activation register vector across all
    eight rows while the producer fetches the following K tile.
    """
    if m != 1:
        raise ValueError("The TMA GEMV experiment requires M=1")
    if dtype != "float16":
        raise ValueError("The TMA GEMV experiment currently requires float16")
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    lop3_group = get_lop3_intrin_group(
        out_dtype=T.float16,
        source_format=T.uint,
        source_bit=4,
        storage_dtype=T.int8,
        with_scaling=False,
    )
    lop3_source = lop3_group["c_source"]
    lop3_func = lop3_group["func_name"]

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={"tl.disable_warp_specialized": True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_n: int = 64,
        block_k: int = 256,
        num_stages: int = 2,
        outputs_per_warp: int = 8,
        split_k_warps: int = 1,
        tma_activation: bool = True,
        tma_weight: bool = True,
        global_split_k: int = 1,
        consumer_warpgroups: int = 2,
        group_post_scale: bool = False,
        group_reduce_post_scale: bool = False,
        group_metadata_fp16: bool = False,
        raw_multiply_fp16: bool = False,
        raw_fp16_values: int = 0,
        half2_accum_pairs: int = 0,
        cache_activation_fp32: bool = False,
        producer_activation_sum: bool = False,
        mma_sync: bool = False,
        mma_batch: int = 1,
        mma_pingpong: bool = False,
        mma_direct_activation: bool = False,
        mma_n_reuse: bool = False,
        mma_transpose: bool = False,
        mma_schedule_id: int = 0,
        scalar_schedule_id: int = -1,
        consumer_max_nreg: int = 112,
    ) -> Callable:
        producer_threads = 128
        consumer_threads = consumer_warpgroups * 128
        threads = producer_threads + consumer_threads
        reduce_threads = 32
        consumer_warps = consumer_threads // reduce_threads
        if consumer_warpgroups not in (1, 2, 4):
            raise ValueError("consumer_warpgroups must be 1, 2, or 4")
        if split_k_warps not in (1, 2, 4, 8, 16):
            raise ValueError(
                "split_k_warps must be 1, 2, 4, 8, or 16")
        output_warps = consumer_warps // split_k_warps
        values_per_thread = block_k // reduce_threads
        lanes_per_group = group_size // values_per_thread
        group_reduction_steps = lanes_per_group.bit_length() - 1
        packed_per_thread = values_per_thread // 2
        super_block_k = block_k * split_k_warps
        groups_per_partition = block_k // group_size
        effective_raw_fp16_values = (
            values_per_thread if raw_multiply_fp16 else raw_fp16_values)
        metadata_local_dtype = (
            dtype if group_metadata_fp16 or not group_post_scale else "float32")
        output_rows = m * global_split_k
        super_tiles_per_split = (
            k // super_block_k // global_split_k)
        scalar_output_mode = (
            scalar_schedule_id % 8 if scalar_schedule_id >= 0 else 0)
        scalar_value_mode = (
            (scalar_schedule_id // 8) % 8
            if scalar_schedule_id >= 0 else 0
        )
        scalar_decode_reverse = (
            (scalar_schedule_id // 64) % 2
            if scalar_schedule_id >= 0 else 0
        )
        scalar_release_policy = (
            (scalar_schedule_id // 128) % 3
            if scalar_schedule_id >= 0 else 0
        )
        packed_weight_shape = (
            (
                n // block_n,
                k // block_k,
                block_n // 64,
                32,
                4 * block_k // 16 * 4,
            )
            if mma_transpose
            else
            (
                n // block_n,
                k // block_k,
                block_n // 64,
                32,
                64 * block_k // (2 * 32),
            )
            if mma_sync
            else (n, k // 2)
        )
        packed_smem_shape = (
            (
                num_stages,
                split_k_warps,
                block_n // 64,
                32,
                4 * block_k // 16 * 4,
            )
            if mma_transpose
            else
            (
                num_stages,
                split_k_warps,
                block_n // 64,
                32,
                64 * block_k // (2 * 32),
            )
            if mma_sync
            else (
                num_stages,
                split_k_warps,
                block_n,
                block_k // 2,
            )
        )

        if consumer_warps % split_k_warps != 0:
            raise ValueError("consumer warps must divide evenly over split-K")
        if global_split_k not in (1, 2, 4, 8):
            raise ValueError("global_split_k must be 1, 2, 4, or 8")
        if block_n != output_warps * outputs_per_warp:
            raise ValueError(
                "block_n must equal output_warps * outputs_per_warp")
        if n % block_n != 0:
            raise ValueError(f"N={n} must be divisible by block_n={block_n}")
        if block_k % group_size != 0:
            raise ValueError(
                f"block_k={block_k} must be divisible by group_size={group_size}")
        if k % super_block_k != 0:
            raise ValueError(
                f"K={k} must be divisible by split tile K={super_block_k}")
        if (k // super_block_k) % global_split_k != 0:
            raise ValueError(
                "K tiles must divide evenly across global split-K")
        if values_per_thread not in (8, 16, 32):
            raise ValueError(
                "The LOP3 path currently requires block_k=256, 512, or 1024")
        if num_stages not in (2, 3, 4):
            raise ValueError("num_stages must be 2, 3, or 4")
        if group_reduce_post_scale and not group_post_scale:
            raise ValueError(
                "group_reduce_post_scale requires group_post_scale")
        if effective_raw_fp16_values not in (0, 2, 4, 6, 8):
            raise ValueError(
                "raw_fp16_values must be 0, 2, 4, 6, or 8")
        if (
            group_metadata_fp16
            or effective_raw_fp16_values
            or half2_accum_pairs
        ) and not group_post_scale:
            raise ValueError(
                "post-scale precision options require group_post_scale")
        if half2_accum_pairs not in (0, 1, 2, 4, 8):
            raise ValueError(
                "half2_accum_pairs must be 0, 1, 2, 4, or 8")
        if half2_accum_pairs and (
            effective_raw_fp16_values or block_k != 512
        ):
            raise ValueError(
                "half2_accum_pairs currently requires K512 group-post-scale "
                "without the older raw-fp16 path")
        if producer_activation_sum and (
            not group_post_scale or tma_activation
        ):
            raise ValueError(
                "producer_activation_sum requires group_post_scale "
                "and cooperative activation copy")
        if mma_sync and block_k != 256:
            raise ValueError("mma_sync requires block_k=256")
        if mma_sync and not mma_n_reuse and not mma_transpose and (
            block_n != 64
            or outputs_per_warp != 16
            or (
                (consumer_warpgroups, split_k_warps)
                not in ((2, 2), (4, 4))
            )
        ):
            raise ValueError(
                "the base mma_sync path requires N64, outputs16, "
                "and either 2WG/split2 or 4WG/split4")
        if mma_n_reuse and (
            not mma_sync
            or mma_transpose
            or block_n != 128
            or outputs_per_warp not in (32, 64)
            or group_post_scale
            or mma_batch != 1
            or mma_pingpong
            or mma_direct_activation
        ):
            raise ValueError(
                "mma_n_reuse requires base MMA pre-scale with "
                "N128, K256, and outputs_per_warp 32 or 64")
        if mma_transpose and (
            not mma_sync
            or block_n % 64
            or outputs_per_warp != 16
            or not group_post_scale
            or mma_n_reuse
            or mma_batch != 1
            or mma_pingpong
            or mma_direct_activation
            or consumer_max_nreg > 112
        ):
            raise ValueError(
                "mma_transpose requires group-post-scale MMA with "
                "N64-aligned tiles, K256, outputs_per_warp=16, and "
                "consumer_max_nreg<=112")
        if mma_schedule_id not in range(128):
            raise ValueError("mma_schedule_id must be in [0, 128)")
        if mma_schedule_id and not mma_transpose:
            raise ValueError(
                "non-zero mma_schedule_id requires mma_transpose")
        if scalar_schedule_id >= 0 and (
            mma_sync
            or not group_post_scale
            or block_k != 512
            or outputs_per_warp != 8
        ):
            raise ValueError(
                "scalar_schedule_id requires the K512/O8 scalar "
                "group-post-scale path")
        if scalar_schedule_id >= 384:
            raise ValueError("scalar_schedule_id must be below 384")
        if mma_batch not in (1, 2, 4, 8):
            raise ValueError("mma_batch must be 1, 2, 4, or 8")
        if mma_batch != 1 and not mma_sync:
            raise ValueError("mma_batch applies only to mma_sync")
        if mma_pingpong and (not mma_sync or mma_batch != 1):
            raise ValueError(
                "mma_pingpong requires mma_sync and mma_batch=1")
        if mma_direct_activation and (not mma_sync or tma_activation):
            raise ValueError(
                "mma_direct_activation requires mma_sync and "
                "cooperative activation copy disabled")
        if consumer_max_nreg not in (
            96, 104, 112, 120, 128, 160, 192, 224, 232
        ):
            raise ValueError(
                "unsupported consumer_max_nreg")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),
            packed_weight: T.Tensor(
                (
                    (
                        n // block_n,
                        k // block_k,
                        block_n // 64,
                        32,
                        4 * block_k // 16 * 4,
                    )
                    if mma_transpose
                    else
                    (
                        n // block_n,
                        k // block_k,
                        block_n // 64,
                        32,
                        64 * block_k // (2 * 32),
                    )
                    if mma_sync
                    else (n, k // 2)
                ),
                "uint8",
            ),
            weight_scale: T.Tensor((n, k // group_size), "float32"),
            weight_zero: T.Tensor((n, k // group_size), "uint8"),
            output: T.Tensor(
                (output_rows, n),
                "float32" if global_split_k > 1 else dtype,
            ),
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                output_rows,
                threads=threads,
            ) as (bx, by):
                activation_smem = T.alloc_shared(
                    (
                        num_stages,
                        split_k_warps,
                        block_k,
                    ),
                    dtype,
                )
                activation_sum_smem = T.alloc_shared(
                    (num_stages, split_k_warps, 32),
                    "float32",
                )
                packed_smem = T.alloc_shared(
                    packed_smem_shape, "uint8")
                scale_smem = T.alloc_shared(
                    (
                        num_stages,
                        block_n,
                        split_k_warps * groups_per_partition,
                    ),
                    "float32",
                )
                zero_smem = T.alloc_shared(
                    (
                        num_stages,
                        block_n,
                        split_k_warps * groups_per_partition,
                    ),
                    "uint8",
                )

                activation_local = T.alloc_local(
                    (values_per_thread,), dtype)
                activation_float_local = T.alloc_local(
                    (values_per_thread,), "float32")
                packed_local = T.alloc_local(
                    (
                        outputs_per_warp,
                        packed_per_thread,
                    ),
                    "uint8",
                )
                dequantized_local = T.alloc_local(
                    (values_per_thread,), dtype)
                accumulator = T.alloc_local(
                    (outputs_per_warp,), "float32")
                raw_partial = T.alloc_local(
                    (outputs_per_warp,), "float32")
                activation_sum = T.alloc_local((1,), "float32")
                mma_accum0 = T.alloc_local((4,), "float32")
                mma_accum1 = T.alloc_local((4,), "float32")
                mma_reuse_accum = T.alloc_local(
                    (outputs_per_warp // 8, 4), "float32")
                cta_reduced = T.alloc_local(
                    (outputs_per_warp,), "float32")
                output_sum = T.alloc_local((1,), "float32")
                producer_activation_sum_local = T.alloc_local(
                    (1,), "float32")
                scale_local = T.alloc_local(
                    (outputs_per_warp,), metadata_local_dtype)
                zero_local = T.alloc_local(
                    (outputs_per_warp,), metadata_local_dtype)
                partials = T.alloc_shared(
                    (
                        output_warps,
                        outputs_per_warp,
                        split_k_warps,
                    ),
                    "float32",
                )

                ready = T.alloc_barrier(
                    [producer_threads] * num_stages)
                empty = T.alloc_barrier(
                    [consumer_threads] * num_stages)
                gi_producer = T.alloc_var("int32", init=0)
                gi_consumer = T.alloc_var("int32", init=0)

                tx = T.get_thread_binding()
                n_start = bx * block_n
                logical_m = by % m
                global_k_partition = by // m
                split_tile_start = (
                    global_k_partition
                    * (k // super_block_k // global_split_k)
                )
                T.import_source(
                    (
                        MMA_SYNC_W4_SOURCE
                        if mma_sync
                        else lop3_source + HALF2_MIXED_DOT_SOURCE
                    )
                )

                if tx < producer_threads:
                    T.dec_max_nreg(24)
                    for ko in T.serial(super_tiles_per_split):
                        stage = gi_producer % num_stages
                        phase = (gi_producer // num_stages) % 2
                        k_start = (
                            split_tile_start + ko) * super_block_k
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(empty[s], phase ^ 1)
                                T.fence_proxy_async()
                                if mma_direct_activation:
                                    pass
                                elif tma_activation:
                                    for part in range(split_k_warps):
                                        part_k_start = (
                                            k_start + part * block_k)
                                        T.tma_copy(
                                            activation[
                                                logical_m,
                                                part_k_start:
                                                part_k_start + block_k,
                                            ],
                                            activation_smem[s, part, :],
                                            barrier=ready[s],
                                        )
                                else:
                                    for part, j in T.Parallel(
                                        split_k_warps, block_k
                                    ):
                                        activation_smem[
                                            s, part, j
                                        ] = activation[
                                            logical_m,
                                            (
                                                k_start
                                                + part * block_k
                                            + j
                                            ),
                                        ]
                                if producer_activation_sum:
                                    T.sync_threads(
                                        barrier_id=3,
                                        arrive_count=producer_threads,
                                    )
                                    if tx < split_k_warps * 32:
                                        producer_part = tx // 32
                                        producer_lane = tx % 32
                                        producer_activation_sum_local[
                                            0
                                        ] = T.cast(0, "float32")
                                        for value in T.serial(
                                            values_per_thread
                                        ):
                                            producer_activation_sum_local[
                                                0
                                            ] += T.cast(
                                                activation_smem[
                                                    s,
                                                    producer_part,
                                                    (
                                                        producer_lane
                                                        * values_per_thread
                                                        + value
                                                    ),
                                                ],
                                                "float32",
                                            )
                                        activation_sum_smem[
                                            s,
                                            producer_part,
                                            producer_lane,
                                        ] = (
                                            producer_activation_sum_local[0]
                                        )
                                for part in range(split_k_warps):
                                    part_k_start = (
                                        k_start + part * block_k)
                                    if mma_transpose and tma_weight:
                                        for mma_n_slice in range(
                                            block_n // 64
                                        ):
                                            T.tma_copy(
                                                packed_weight[
                                                    bx,
                                                    (
                                                        part_k_start
                                                        // block_k
                                                    ),
                                                    mma_n_slice,
                                                    :,
                                                    :,
                                                ],
                                                packed_smem[
                                                    s,
                                                    part,
                                                    mma_n_slice,
                                                    :,
                                                    :,
                                                ],
                                                barrier=ready[s],
                                            )
                                    elif mma_transpose:
                                        for (
                                            mma_n_slice,
                                            mma_lane,
                                            mma_value,
                                        ) in T.Parallel(
                                            block_n // 64,
                                            32,
                                            4 * block_k // 16 * 4,
                                        ):
                                            packed_smem[
                                                s,
                                                part,
                                                mma_n_slice,
                                                mma_lane,
                                                mma_value,
                                            ] = packed_weight[
                                                bx,
                                                part_k_start // block_k,
                                                mma_n_slice,
                                                mma_lane,
                                                mma_value,
                                            ]
                                    elif mma_sync and tma_weight:
                                        for mma_n_slice in range(
                                            block_n // 64
                                        ):
                                            T.tma_copy(
                                                packed_weight[
                                                    bx,
                                                    (
                                                        part_k_start
                                                        // block_k
                                                    ),
                                                    mma_n_slice,
                                                    :,
                                                    :,
                                                ],
                                                packed_smem[
                                                    s,
                                                    part,
                                                    mma_n_slice,
                                                    :,
                                                    :,
                                                ],
                                                barrier=ready[s],
                                            )
                                    elif mma_sync:
                                        for (
                                            mma_n_slice,
                                            mma_lane,
                                            mma_value,
                                        ) in T.Parallel(
                                            block_n // 64,
                                            32,
                                            (
                                                64
                                                * block_k
                                                // (2 * 32)
                                            ),
                                        ):
                                            packed_smem[
                                                s,
                                                part,
                                                mma_n_slice,
                                                mma_lane,
                                                mma_value,
                                            ] = packed_weight[
                                                bx,
                                                part_k_start // block_k,
                                                mma_n_slice,
                                                mma_lane,
                                                mma_value,
                                            ]
                                    elif tma_weight:
                                        weight_chunks = max(
                                            1, block_k // 512)
                                        weight_chunk_bytes = (
                                            block_k // 2 // weight_chunks)
                                        for weight_chunk in range(
                                            weight_chunks
                                        ):
                                            byte_start = (
                                                part_k_start // 2
                                                + weight_chunk
                                                * weight_chunk_bytes
                                            )
                                            shared_byte_start = (
                                                weight_chunk
                                                * weight_chunk_bytes
                                            )
                                            T.tma_copy(
                                                packed_weight[
                                                    n_start:
                                                    n_start + block_n,
                                                    byte_start:
                                                    (
                                                        byte_start
                                                        + weight_chunk_bytes
                                                    ),
                                                ],
                                                packed_smem[
                                                    s,
                                                    part,
                                                    :,
                                                    shared_byte_start:
                                                    (
                                                        shared_byte_start
                                                        + weight_chunk_bytes
                                                    ),
                                                ],
                                                barrier=ready[s],
                                            )
                                    else:
                                        for row, byte in T.Parallel(
                                            block_n, block_k // 2
                                        ):
                                            packed_smem[
                                                s, part, row, byte
                                            ] = packed_weight[
                                                n_start + row,
                                                part_k_start // 2 + byte,
                                            ]
                                T.tma_copy(
                                    weight_scale[
                                        n_start:n_start + block_n,
                                        k_start // group_size:
                                        (
                                            k_start + super_block_k
                                        ) // group_size,
                                    ],
                                    scale_smem[s, :, :],
                                    barrier=ready[s],
                                )
                                T.copy(
                                    weight_zero[
                                        n_start:n_start + block_n,
                                        k_start // group_size:
                                        (
                                            k_start + super_block_k
                                        ) // group_size,
                                    ],
                                    zero_smem[s, :, :],
                                )
                                T.barrier_arrive(ready[s])
                        gi_producer = gi_producer + 1
                else:
                    T.inc_max_nreg(consumer_max_nreg)
                    consumer_tx = tx - producer_threads
                    warp_id = consumer_tx // reduce_threads
                    lane_id = consumer_tx % reduce_threads
                    output_warp = warp_id // split_k_warps
                    k_partition = warp_id % split_k_warps
                    output_col_base = (
                        n_start + output_warp * outputs_per_warp)
                    if not mma_sync:
                        T.clear(accumulator)
                    if mma_sync:
                        if mma_n_reuse:
                            T.clear(mma_reuse_accum)
                        else:
                            T.clear(mma_accum0)
                            T.clear(mma_accum1)

                    for ko in T.serial(super_tiles_per_split):
                        stage = gi_consumer % num_stages
                        phase = (gi_consumer // num_stages) % 2
                        for s in range(num_stages):
                            if stage == s:
                                T.barrier_wait(ready[s], phase)
                                if mma_sync:
                                    if mma_transpose:
                                        T.call_extern(
                                            "handle",
                                            (
                                                "tl::w4a16_mma_sync_"
                                                "transposed_tile<"
                                                f"{block_k},"
                                                f"{split_k_warps * groups_per_partition},"
                                                f"{mma_schedule_id}"
                                                ">"
                                            ),
                                            T.address_of(
                                                activation_smem[
                                                    s, k_partition, 0
                                                ]
                                            ),
                                            T.address_of(
                                                packed_smem[
                                                    s,
                                                    k_partition,
                                                    (
                                                        output_warp
                                                        // 4
                                                    ),
                                                    0,
                                                    0,
                                                ]
                                            ),
                                            T.address_of(
                                                scale_smem[s, 0, 0]
                                            ),
                                            T.address_of(
                                                zero_smem[s, 0, 0]
                                            ),
                                            output_warp % 4,
                                            (
                                                output_warp
                                                * outputs_per_warp
                                            ),
                                            (
                                                k_partition
                                                * groups_per_partition
                                            ),
                                            mma_accum0[0],
                                            mma_accum0[1],
                                            mma_accum0[2],
                                            mma_accum0[3],
                                        )
                                    elif mma_n_reuse:
                                        T.call_extern(
                                            "handle",
                                            (
                                                "tl::w4a16_mma_sync_nreuse_tile<"
                                                f"{block_k},"
                                                f"{split_k_warps * groups_per_partition},"
                                                f"{outputs_per_warp // 8}"
                                                ">"
                                            ),
                                            T.address_of(
                                                activation_smem[
                                                    s, k_partition, 0
                                                ]
                                            ),
                                            T.address_of(
                                                packed_smem[
                                                    s,
                                                    k_partition,
                                                    0,
                                                    0,
                                                    0,
                                                ]
                                            ),
                                            T.address_of(
                                                scale_smem[s, 0, 0]
                                            ),
                                            T.address_of(
                                                zero_smem[s, 0, 0]
                                            ),
                                            output_warp * outputs_per_warp,
                                            (
                                                k_partition
                                                * groups_per_partition
                                            ),
                                            T.address_of(
                                                mma_reuse_accum[0, 0]
                                            ),
                                        )
                                    else:
                                        T.call_extern(
                                            "handle",
                                            (
                                                "tl::w4a16_mma_sync_tile<"
                                                f"{block_k},"
                                                f"{split_k_warps * groups_per_partition},"
                                                f"{str(group_post_scale).lower()},"
                                                f"{mma_batch},"
                                                f"{str(mma_pingpong).lower()}"
                                                ">"
                                            ),
                                            T.address_of(
                                                (
                                                    activation[
                                                        logical_m,
                                                        (
                                                            (
                                                                split_tile_start
                                                                + ko
                                                            )
                                                            * super_block_k
                                                            + k_partition
                                                            * block_k
                                                        ),
                                                    ]
                                                    if mma_direct_activation
                                                    else activation_smem[
                                                        s, k_partition, 0
                                                    ]
                                                )
                                            ),
                                            T.address_of(
                                                packed_smem[
                                                    s,
                                                    k_partition,
                                                    0,
                                                    0,
                                                    0,
                                                ]
                                            ),
                                            T.address_of(
                                                scale_smem[s, 0, 0]
                                            ),
                                            T.address_of(
                                                zero_smem[s, 0, 0]
                                            ),
                                            output_warp * outputs_per_warp,
                                            (
                                                k_partition
                                                * groups_per_partition
                                            ),
                                            mma_accum0[0],
                                            mma_accum0[1],
                                            mma_accum0[2],
                                            mma_accum0[3],
                                            mma_accum1[0],
                                            mma_accum1[1],
                                            mma_accum1[2],
                                            mma_accum1[3],
                                        )
                                    T.barrier_arrive(empty[s])
                                else:
                                    for v in T.vectorized(values_per_thread):
                                        activation_local[v] = activation_smem[
                                            s,
                                            k_partition,
                                            lane_id * values_per_thread + v,
                                        ]

                                    group_in_partition = (
                                        lane_id * values_per_thread
                                    ) // group_size
                                    for output_iter in T.serial(
                                        outputs_per_warp
                                    ):
                                        output_slot = _schedule_index_expr(
                                            output_iter,
                                            outputs_per_warp,
                                            scalar_output_mode,
                                        )
                                        output_col_local = (
                                            output_warp * outputs_per_warp
                                            + output_slot
                                        )
                                        for v in T.vectorized(
                                            packed_per_thread
                                        ):
                                            packed_local[
                                                output_slot, v
                                            ] = packed_smem[
                                                s,
                                                k_partition,
                                                output_col_local,
                                                (
                                                    lane_id
                                                    * packed_per_thread
                                                    + v
                                                ),
                                            ]
                                        if (
                                            not group_reduce_post_scale
                                            or lane_id % lanes_per_group == 0
                                        ):
                                            scale_local[
                                                output_slot
                                            ] = T.cast(
                                                scale_smem[
                                                    s,
                                                    output_col_local,
                                                    (
                                                        k_partition
                                                        * groups_per_partition
                                                        + group_in_partition
                                                    ),
                                                ],
                                                metadata_local_dtype,
                                            )
                                            zero_local[
                                                output_slot
                                            ] = T.cast(
                                                zero_smem[
                                                    s,
                                                    output_col_local,
                                                    (
                                                        k_partition
                                                        * groups_per_partition
                                                        + group_in_partition
                                                    ),
                                                ],
                                                metadata_local_dtype,
                                            )

                                    # The scalar path consumes the rest of the
                                    # tile exclusively from registers.
                                    if scalar_release_policy == 0:
                                        T.barrier_arrive(empty[s])

                                    if group_post_scale:
                                        if producer_activation_sum:
                                            activation_sum[0] = (
                                                activation_sum_smem[
                                                    s,
                                                    k_partition,
                                                    lane_id,
                                                ]
                                            )
                                        else:
                                            activation_sum[0] = T.cast(
                                                0, "float32")
                                            for value_iter in T.serial(
                                                values_per_thread
                                            ):
                                                v = _schedule_index_expr(
                                                    value_iter,
                                                    values_per_thread,
                                                    scalar_value_mode,
                                                )
                                                if cache_activation_fp32:
                                                    activation_float_local[
                                                        v
                                                    ] = T.cast(
                                                        activation_local[v],
                                                        "float32",
                                                    )
                                                    activation_sum[0] += (
                                                        activation_float_local[v]
                                                    )
                                                else:
                                                    activation_sum[0] += T.cast(
                                                        activation_local[v],
                                                        "float32",
                                                    )
                                        if group_reduce_post_scale:
                                            for reduction_step in T.serial(
                                                group_reduction_steps
                                            ):
                                                activation_sum[
                                                    0
                                                ] += T.shfl_down(
                                                    activation_sum[0],
                                                    (
                                                        lanes_per_group // 2
                                                        >> reduction_step
                                                    ),
                                                    width=lanes_per_group,
                                                )

                                    if scalar_release_policy == 1:
                                        T.barrier_arrive(empty[s])

                                    for output_iter in T.serial(
                                        outputs_per_warp
                                    ):
                                        output_slot = _schedule_index_expr(
                                            output_iter,
                                            outputs_per_warp,
                                            scalar_output_mode,
                                        )
                                        for decode_iter in T.serial(
                                            values_per_thread // 8
                                        ):
                                            decode_chunk = (
                                                (
                                                    values_per_thread // 8
                                                    - 1
                                                    - decode_iter
                                                )
                                                if scalar_decode_reverse
                                                else decode_iter
                                            )
                                            T.call_extern(
                                                lop3_func,
                                                T.access_ptr(
                                                    packed_local[
                                                        output_slot,
                                                        decode_chunk * 4,
                                                    ],
                                                    "r",
                                                    4,
                                                ),
                                                T.access_ptr(
                                                    dequantized_local[
                                                        decode_chunk * 8
                                                    ],
                                                    "w",
                                                    8,
                                                ),
                                                dtype=dtype,
                                            )
                                        if group_post_scale:
                                            if half2_accum_pairs:
                                                raw_partial[
                                                    output_slot
                                                ] = T.call_extern(
                                                    "float32",
                                                    (
                                                        "tl::w4_dot_half2_accum_"
                                                        f"{half2_accum_pairs}"
                                                    ),
                                                    T.address_of(
                                                        activation_local[0]
                                                    ),
                                                    T.address_of(
                                                        dequantized_local[0]
                                                    ),
                                                    values_per_thread // 2,
                                                )
                                            elif effective_raw_fp16_values:
                                                raw_partial[
                                                    output_slot
                                                ] = T.call_extern(
                                                    "float32",
                                                    (
                                                        "tl::w4_dot_half2_"
                                                        f"{effective_raw_fp16_values // 2}"
                                                    ),
                                                    T.address_of(
                                                        activation_local[0]
                                                    ),
                                                    T.address_of(
                                                        dequantized_local[0]
                                                    ),
                                                    values_per_thread // 2,
                                                )
                                            else:
                                                raw_partial[
                                                    output_slot
                                                ] = T.cast(0, "float32")
                                                for value_iter in T.serial(
                                                    values_per_thread
                                                ):
                                                    v = _schedule_index_expr(
                                                        value_iter,
                                                        values_per_thread,
                                                        scalar_value_mode,
                                                    )
                                                    raw_partial[
                                                        output_slot
                                                    ] += (
                                                    (
                                                        activation_float_local[
                                                            v
                                                        ]
                                                        if cache_activation_fp32
                                                        else T.cast(
                                                            activation_local[v],
                                                            "float32",
                                                        )
                                                    )
                                                    * T.cast(
                                                        dequantized_local[v],
                                                        "float32",
                                                    )
                                                )
                                            if group_reduce_post_scale:
                                                for reduction_step in T.serial(
                                                    group_reduction_steps
                                                ):
                                                    raw_partial[
                                                        output_slot
                                                    ] += T.shfl_down(
                                                        raw_partial[
                                                            output_slot
                                                        ],
                                                        (
                                                            lanes_per_group
                                                            // 2
                                                            >> reduction_step
                                                        ),
                                                        width=lanes_per_group,
                                                    )
                                                if (
                                                    lane_id
                                                    % lanes_per_group
                                                    == 0
                                                ):
                                                    accumulator[
                                                        output_slot
                                                    ] += (
                                                        (
                                                            raw_partial[
                                                                output_slot
                                                            ]
                                                            - zero_local[
                                                                output_slot
                                                            ]
                                                            * activation_sum[0]
                                                        )
                                                        * scale_local[
                                                            output_slot
                                                        ]
                                                    )
                                            else:
                                                accumulator[
                                                    output_slot
                                                ] += (
                                                    (
                                                        raw_partial[
                                                            output_slot
                                                        ]
                                                        - zero_local[
                                                            output_slot
                                                        ]
                                                        * activation_sum[0]
                                                    )
                                                    * scale_local[
                                                        output_slot
                                                    ]
                                                )
                                        else:
                                            for value_iter in T.serial(
                                                values_per_thread
                                            ):
                                                v = _schedule_index_expr(
                                                    value_iter,
                                                    values_per_thread,
                                                    scalar_value_mode,
                                                )
                                                accumulator[
                                                    output_slot
                                                ] += (
                                                    T.cast(
                                                        activation_local[v],
                                                        "float32",
                                                    )
                                                    * T.cast(
                                                        (
                                                            dequantized_local[v]
                                                            - zero_local[
                                                                output_slot
                                                            ]
                                                        )
                                                        * scale_local[
                                                            output_slot
                                                        ],
                                                        "float32",
                                                    )
                                                )
                                    if scalar_release_policy == 2:
                                        T.barrier_arrive(empty[s])
                        gi_consumer = gi_consumer + 1

                    if mma_sync:
                        if mma_transpose:
                            if lane_id % 4 == 0:
                                partials[
                                    output_warp,
                                    lane_id // 4,
                                    k_partition,
                                ] = mma_accum0[0]
                                partials[
                                    output_warp,
                                    lane_id // 4 + 8,
                                    k_partition,
                                ] = mma_accum0[2]
                        elif lane_id < 4:
                            if mma_n_reuse:
                                for n_tile in T.serial(
                                    outputs_per_warp // 8
                                ):
                                    partials[
                                        output_warp,
                                        n_tile * 8 + lane_id * 2,
                                        k_partition,
                                    ] = mma_reuse_accum[n_tile, 0]
                                    partials[
                                        output_warp,
                                        (
                                            n_tile * 8
                                            + lane_id * 2
                                            + 1
                                        ),
                                        k_partition,
                                    ] = mma_reuse_accum[n_tile, 1]
                            else:
                                partials[
                                    output_warp,
                                    lane_id * 2,
                                    k_partition,
                                ] = mma_accum0[0]
                                partials[
                                    output_warp,
                                    lane_id * 2 + 1,
                                    k_partition,
                                ] = mma_accum0[1]
                                partials[
                                    output_warp,
                                    8 + lane_id * 2,
                                    k_partition,
                                ] = mma_accum1[0]
                                partials[
                                    output_warp,
                                    8 + lane_id * 2 + 1,
                                    k_partition,
                                ] = mma_accum1[1]
                        T.sync_threads(
                            barrier_id=1,
                            arrive_count=consumer_threads,
                        )
                        if lane_id == 0 and k_partition == 0:
                            for output_slot in T.serial(
                                outputs_per_warp
                            ):
                                output_sum[0] = T.cast(0, "float32")
                                for part in T.serial(split_k_warps):
                                    output_sum[0] += partials[
                                        output_warp,
                                        output_slot,
                                        part,
                                    ]
                                output[
                                    by, output_col_base + output_slot
                                ] = output_sum[0]
                    else:
                        for output_slot in T.serial(outputs_per_warp):
                            for reduction_step in T.serial(5):
                                accumulator[output_slot] += T.shfl_down(
                                    accumulator[output_slot],
                                    16 >> reduction_step,
                                    width=reduce_threads,
                                )
                        if split_k_warps == 1:
                            if lane_id == 0:
                                for output_slot in T.serial(
                                    outputs_per_warp
                                ):
                                    output[
                                        by, output_col_base + output_slot
                                    ] = accumulator[output_slot]
                        else:
                            if lane_id == 0:
                                for output_slot in T.serial(
                                    outputs_per_warp
                                ):
                                    partials[
                                        output_warp,
                                        output_slot,
                                        k_partition,
                                    ] = accumulator[output_slot]
                            T.sync_threads(
                                barrier_id=1,
                                arrive_count=consumer_threads,
                            )
                            if lane_id == 0 and k_partition == 0:
                                for output_slot in T.serial(
                                    outputs_per_warp
                                ):
                                    cta_reduced[
                                        output_slot
                                    ] = T.cast(0, "float32")
                                    for part in T.serial(split_k_warps):
                                        cta_reduced[output_slot] += (
                                            partials[
                                                output_warp,
                                                output_slot,
                                                part,
                                            ]
                                        )
                                    output[
                                        by, output_col_base + output_slot
                                    ] = cta_reduced[output_slot]

        return main

    return build


@functools.lru_cache(maxsize=16)
def w4a16_splitk_reduce_kernel(
    m: int,
    n: int,
    split_k: int,
    dtype: str,
) -> Callable:
    """Reduce FP32 global split-K partials into the final A16 output."""
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3"])
    def build(threads: int = 256) -> Callable:
        @T.prim_func
        def main(
            partials: T.Tensor((m * split_k, n), "float32"),
            output: T.Tensor((m, n), dtype),
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, threads),
                m,
                threads=threads,
            ) as (bx, by):
                tx = T.get_thread_binding()
                col = bx * threads + tx
                accumulator = T.alloc_local((1,), "float32")
                accumulator[0] = T.cast(0, "float32")
                if col < n:
                    for part in T.serial(split_k):
                        accumulator[0] += partials[
                            part * m + by, col]
                    output[by, col] = T.cast(accumulator[0], dtype)

        return main

    return build


def run_case(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    decode_mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    quant_mode: str,
    kernel_mode: str,
    dump_lowering: bool = False,
) -> None:
    torch.manual_seed(0)
    activation = torch.randn((m, k), device="cuda", dtype=dtype)
    weight = torch.randn((n, k), device="cuda", dtype=torch.float32) * 0.25
    packed, scale, zero, dequantized = quantize_weight_int4(
        weight, quant_mode=quant_mode)
    if decode_mode == "lop3":
        packed = interleave_weight(
            packed, nbits=4, target_dtype="float16").view(torch.uint8)

    dtype_name = str(dtype).removeprefix("torch.")
    if kernel_mode == "gemv":
        if decode_mode != "lop3":
            raise ValueError("kernel_mode=gemv currently requires decode_mode=lop3")
        compiled = w4a16_gemv_kernel(m, n, k, dtype_name)()
    elif kernel_mode == "gemm_2wg":
        if decode_mode != "lop3":
            raise ValueError("kernel_mode=gemm_2wg currently requires decode_mode=lop3")
        compiled = w4a16_gemm_2wg_kernel(m, n, k, dtype_name)(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_stages=num_stages,
        )
    elif kernel_mode == "gemm_3wg":
        if decode_mode != "lop3":
            raise ValueError("kernel_mode=gemm_3wg currently requires decode_mode=lop3")
        compiled = w4a16_gemm_3wg_kernel(m, n, k, dtype_name)(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_stages=num_stages,
        )
    else:
        compiled = w4a16_gemm_kernel(
            m, n, k, dtype_name, decode_mode=decode_mode)(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_stages=num_stages,
                threads=threads,
            )
    actual = compiled(activation, packed, scale, zero)
    # Kernel-correctness reference: W4A16 dequantizes weights to the activation
    # dtype before MMA, then accumulates in fp32 and casts the output to A16.
    expected = activation @ dequantized.to(dtype).T
    # The fp32-dequant result is a separate quantization-quality diagnostic; it
    # must not be used as the implementation-equivalence gate for W4A16.
    ideal = (activation.float() @ dequantized.T).to(dtype)

    if dtype == torch.bfloat16 and k > 1024:
        tolerances = {"atol": 2.5e-1, "rtol": 5e-2}
    elif dtype == torch.bfloat16:
        tolerances = {"atol": 5e-2, "rtol": 3e-2}
    elif k > 1024:
        tolerances = {"atol": 5e-2, "rtol": 3e-2}
    else:
        tolerances = {"atol": 2e-2, "rtol": 2e-2}
    torch.testing.assert_close(actual, expected, **tolerances)
    # Explicit pipelines are synchronization-sensitive. Re-run correctness
    # before timing so a one-off successful launch cannot hide a barrier race.
    for _ in range(2):
        repeated = compiled(activation, packed, scale, zero)
        torch.testing.assert_close(repeated, expected, **tolerances)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0).item()
    ideal_cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), ideal.float().flatten(), dim=0).item()
    print(
        f"PASS shape=({m},{n},{k}) dtype={dtype_name} "
        f"kernel_mode={kernel_mode} decode_mode={decode_mode} "
        f"quant_mode={quant_mode} "
        f"config=({block_m},{block_n},{block_k},s{num_stages},t{threads})")
    print(
        f"max_abs={max_abs:.6f} cosine={cosine:.8f} "
        f"ideal_fp32_dequant_cosine={ideal_cosine:.8f}")

    # Warm-up and report kernel-only latency.  This is diagnostic only; the
    # feasibility phase has no production performance acceptance threshold.
    for _ in range(5):
        compiled(activation, packed, scale, zero)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        compiled(activation, packed, scale, zero)
    end.record()
    end.synchronize()
    latency_ms = start.elapsed_time(end) / 20
    tflops = 2.0 * m * n * k / (latency_ms * 1e-3) / 1e12
    print(f"latency_ms={latency_ms:.6f} tflops={tflops:.3f}")

    dequantized_a16 = dequantized.to(dtype)
    for _ in range(5):
        torch.matmul(activation, dequantized_a16.T)
    torch.cuda.synchronize()
    start.record()
    for _ in range(20):
        torch.matmul(activation, dequantized_a16.T)
    end.record()
    end.synchronize()
    torch_latency_ms = start.elapsed_time(end) / 20
    torch_tflops = 2.0 * m * n * k / (torch_latency_ms * 1e-3) / 1e12
    print(
        f"torch_a16_latency_ms={torch_latency_ms:.6f} "
        f"torch_a16_tflops={torch_tflops:.3f} "
        f"relative_throughput={tflops / torch_tflops:.3f}")

    if dump_lowering:
        source = compiled.get_kernel_source()
        print("LOWERING_EVIDENCE")
        for line in source.splitlines():
            lowered = line.lower()
            if "wgmma_ss<" in lowered:
                print(line.strip())
        print(
            "packed_global_loads_in_source="
            f"{source.count('packed_weight[')} "
            "scale_global_loads_in_source="
            f"{source.count('weight_scale[')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--decode-mode",
        choices=("scalar", "staged", "lop3"),
        default="staged",
    )
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument(
        "--quant-mode",
        choices=("symmetric", "affine"),
        default="symmetric",
    )
    parser.add_argument(
        "--kernel-mode",
        choices=("gemm", "gemm_2wg", "gemm_3wg", "gemv"),
        default="gemm",
    )
    parser.add_argument("--dump-lowering", action="store_true")
    args = parser.parse_args()
    run_case(
        args.m,
        args.n,
        args.k,
        getattr(torch, args.dtype),
        args.decode_mode,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        args.threads,
        args.quant_mode,
        args.kernel_mode,
        args.dump_lowering,
    )


if __name__ == "__main__":
    main()
