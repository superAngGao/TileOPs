
namespace clk {
__device__ __forceinline__ int64_t read_clock() {
    int64_t ret; asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)); return ret;
}
__device__ __forceinline__ void clock_accum(int64_t* addr, int64_t delta) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr), static_cast<unsigned long long>(delta));
}
__device__ __forceinline__ void clock_accum_count(int64_t* addr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr), 1ULL);
}
}  // namespace clk
#include <tl_templates/cuda/instruction/wgmma.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap k_desc, float* __restrict__ lse, __grid_constant__ const CUtensorMap output_desc, __grid_constant__ const CUtensorMap q_desc, int64_t* __restrict__ timing, __grid_constant__ const CUtensorMap v_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap k_desc, float* __restrict__ lse, __grid_constant__ const CUtensorMap output_desc, __grid_constant__ const CUtensorMap q_desc, int64_t* __restrict__ timing, __grid_constant__ const CUtensorMap v_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t k_full_mem[1];
  auto k_full = reinterpret_cast<Barrier*>(k_full_mem);
  __shared__ __align__(16) uint64_t k_empty_mem[1];
  auto k_empty = reinterpret_cast<Barrier*>(k_empty_mem);
  __shared__ __align__(16) uint64_t v_full_mem[1];
  auto v_full = reinterpret_cast<Barrier*>(v_full_mem);
  __shared__ __align__(16) uint64_t v_empty_mem[1];
  auto v_empty = reinterpret_cast<Barrier*>(v_empty_mem);
  __shared__ __align__(16) uint64_t wg_sched_12_mem[1];
  auto wg_sched_12 = reinterpret_cast<Barrier*>(wg_sched_12_mem);
  __shared__ __align__(16) uint64_t wg_sched_21_mem[1];
  auto wg_sched_21 = reinterpret_cast<Barrier*>(wg_sched_21_mem);
  __shared__ __align__(16) uint64_t q_full_1_mem[1];
  auto q_full_1 = reinterpret_cast<Barrier*>(q_full_1_mem);
  __shared__ __align__(16) uint64_t q_full_2_mem[1];
  auto q_full_2 = reinterpret_cast<Barrier*>(q_full_2_mem);
  int gi_kp = 0;
  int gi_vp = 0;
  int gi_q1 = 0;
  float acc_o_1[64];
  float ls_1[2];
  float sm_1[2];
  int gi_kc1 = 0;
  float acc_s_1[64];
  float smp_1[2];
  float ss_1[2];
  float ssum_1[2];
  half_t acc_s_cast_1[64];
  int gi_vc1 = 0;
  int gi_q2 = 0;
  float acc_o_2[64];
  float ls_2[2];
  float sm_2[2];
  int gi_kc2 = 0;
  float acc_s_2[64];
  int gi_vc2 = 0;
  half_t acc_s_cast_2[64];
  float smp_2[2];
  float ss_2[2];
  float ssum_2[2];
  tl::GmmaDescriptor desc_a;
  tl::GmmaDescriptor desc_b;
  tl::GmmaDescriptor desc_a_1;
  tl::GmmaDescriptor desc_b_1;
  float sm_1_clear[2];
  tl::GmmaDescriptor desc_a_2;
  tl::GmmaDescriptor desc_b_2;
  tl::GmmaDescriptor desc_a_3;
  tl::GmmaDescriptor desc_b_3;
  tl::GmmaDescriptor desc_b_4;
  tl::GmmaDescriptor desc_b_5;
  float sm_1_clear_1[2];
  tl::GmmaDescriptor desc_b_6;
  tl::GmmaDescriptor desc_b_7;
  tl::GmmaDescriptor desc_a_4;
  tl::GmmaDescriptor desc_b_8;
  tl::GmmaDescriptor desc_a_5;
  tl::GmmaDescriptor desc_b_9;
  tl::GmmaDescriptor desc_b_10;
  tl::GmmaDescriptor desc_b_11;
  float sm_2_clear[2];
  tl::GmmaDescriptor desc_b_12;
  tl::GmmaDescriptor desc_b_13;
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(v_desc);
    tl::prefetch_tma_descriptor(q_desc);
    tl::prefetch_tma_descriptor(output_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    k_full[0].init(128);
    k_empty[0].init(256);
    v_full[0].init(128);
    v_empty[0].init(256);
    wg_sched_12[0].init(128);
    wg_sched_21[0].init(128);
    q_full_1[0].init(128);
    q_full_2[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    for (int w = 0; w < 32; ++w) {
      if (1024 <= ((w * 33) + (((int)blockIdx.x) >> 2))) {
        break;
      }
      for (int sub_idx = 0; sub_idx < 2; ++sub_idx) {
        for (int n_idx = 0; n_idx < ((((((w * 132) + ((int)blockIdx.x)) % 128) / 8) + (sub_idx * (31 - (((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 2)))) + 1); ++n_idx) {
          k_empty[0].wait(((gi_kp + 1) & 1));
          if ((gi_kp % 2) == 0) {
            if (((int)threadIdx.x) == 0) {
              k_full[0].expect_transaction(32768);
              tl::tma_load(k_desc, k_full[0], (&(((half_t*)buf_dyn_shmem)[0])), 0, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), (n_idx * 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
              tl::tma_load(k_desc, k_full[0], (&(((half_t*)buf_dyn_shmem)[8192])), 64, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), (n_idx * 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            }
          } else {
            if (((int)threadIdx.x) == 0) {
              k_full[0].expect_transaction(32768);
              tl::tma_load(k_desc, k_full[0], (&(((half_t*)buf_dyn_shmem)[16384])), 0, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), (n_idx * 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
              tl::tma_load(k_desc, k_full[0], (&(((half_t*)buf_dyn_shmem)[24576])), 64, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), (n_idx * 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            }
          }
          k_full[0].arrive();
          if (0 < n_idx) {
            v_empty[0].wait(((gi_vp + 1) & 1));
            if ((gi_vp % 2) == 0) {
              if (((int)threadIdx.x) == 0) {
                v_full[0].expect_transaction(32768);
                tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[32768])), 0, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((n_idx * 128) - 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
                tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[40960])), 64, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((n_idx * 128) - 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
              }
            } else {
              if (((int)threadIdx.x) == 0) {
                v_full[0].expect_transaction(32768);
                tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[49152])), 0, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((n_idx * 128) - 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
                tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[57344])), 64, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((n_idx * 128) - 128), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
              }
            }
            v_full[0].arrive();
            gi_vp = (gi_vp + 1);
          }
          gi_kp = (gi_kp + 1);
        }
        v_empty[0].wait(((gi_vp + 1) & 1));
        if ((gi_vp % 2) == 0) {
          if (((int)threadIdx.x) == 0) {
            v_full[0].expect_transaction(32768);
            tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[32768])), 0, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx * (31 - (((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[40960])), 64, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx * (31 - (((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
          }
        } else {
          if (((int)threadIdx.x) == 0) {
            v_full[0].expect_transaction(32768);
            tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[49152])), 0, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx * (31 - (((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_load(v_desc, v_full[0], (&(((half_t*)buf_dyn_shmem)[57344])), 64, ((((w * 132) + ((int)blockIdx.x)) % 1024) / 128), ((((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx * (31 - (((((w * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w * 132) + ((int)blockIdx.x)) % 4096) / 1024));
          }
        }
        v_full[0].arrive();
        gi_vp = (gi_vp + 1);
      }
    }
  } else {
    if (((int)threadIdx.x) < 256) {
      tl::warpgroup_reg_alloc<240>();
      wg_sched_21[0].arrive();
      for (int w_1 = 0; w_1 < 32; ++w_1) {
        if (1024 <= ((w_1 * 33) + (((int)blockIdx.x) >> 2))) {
          break;
        }
        for (int sub_idx_1 = 0; sub_idx_1 < 2; ++sub_idx_1) {
          if (((int)threadIdx.x) == 128) {
            q_full_1[0].expect_transaction(16384);
            tl::tma_load(q_desc, q_full_1[0], (&(((half_t*)buf_dyn_shmem)[65536])), 0, ((((((w_1 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_1 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_1 * 132) + ((int)blockIdx.x)) % 8)), ((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w_1 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_load(q_desc, q_full_1[0], (&(((half_t*)buf_dyn_shmem)[69632])), 64, ((((((w_1 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_1 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_1 * 132) + ((int)blockIdx.x)) % 8)), ((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w_1 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
          }
          q_full_1[0].arrive();
          q_full_1[0].wait((gi_q1 & 1));
          gi_q1 = (gi_q1 + 1);
          #pragma unroll
          for (int i = 0; i < 16; ++i) {
            float broadcast_var = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(acc_o_1 + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
          }
          float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
          *(float2*)(ls_1 + 0) = make_float2(broadcast_var_1, broadcast_var_1);
          float broadcast_var_2 = -CUDART_INF_F;
          *(float2*)(sm_1 + 0) = make_float2(broadcast_var_2, broadcast_var_2);
          for (int n_idx_1 = 0; n_idx_1 < ((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) + (sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2)))) + 1); ++n_idx_1) {
            k_full[0].wait((gi_kc1 & 1));
            wg_sched_21[0].wait((gi_kc1 & 1));
            #pragma unroll
            for (int i_1 = 0; i_1 < 16; ++i_1) {
              float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(acc_s_1 + (i_1 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
            }
            if (n_idx_1 == 0) {
              if ((gi_kc1 % 2) == 0) {
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((half_t*)buf_dyn_shmem)[65536])));
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((half_t*)buf_dyn_shmem)[0])));
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_1 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki = 0; ki < 8; ++ki) {
                  tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((ki >> 2) * 16384) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_1 + 0)), 1);
                }
                tl::warpgroup_commit_batch();
              } else {
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((half_t*)buf_dyn_shmem)[65536])));
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((half_t*)buf_dyn_shmem)[16384])));
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_1 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
                  tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 16384) + ((ki_1 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_1 + 0)), 1);
                }
                tl::warpgroup_commit_batch();
              }
              wg_sched_12[0].arrive();
              tl::wait_wgmma<0>();
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_1 + 0), 64);
              k_empty[0].arrive();
              if (n_idx_1 == (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) + (sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))))) {
                #pragma unroll
                for (int i_2 = 0; i_2 < 64; ++i_2) {
                  float condval;
                  if (((((((n_idx_1 * 128) + ((i_2 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_2 & 1)) + 64) <= (((((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
                    condval = acc_s_1[i_2];
                  } else {
                    condval = -CUDART_INF_F;
                  }
                  acc_s_1[i_2] = condval;
                }
              }
              *(float2*)(smp_1 + 0) = *(float2*)(sm_1 + 0);
              float broadcast_var_4 = -CUDART_INF_F;
              *(float2*)(sm_1 + 0) = make_float2(broadcast_var_4, broadcast_var_4);
              #pragma unroll
              for (int i_3 = 0; i_3 < 2; ++i_3) {
                sm_1_clear[i_3] = -CUDART_INF_F;
                #pragma unroll
                for (int rv = 0; rv < 32; ++rv) {
                  sm_1_clear[i_3] = max(sm_1_clear[i_3], acc_s_1[((((rv & 15) * 4) + (i_3 * 2)) + (rv >> 4))]);
                }
                sm_1_clear[i_3] = tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run(sm_1_clear[i_3]);
                sm_1[i_3] = max(sm_1[i_3], sm_1_clear[i_3]);
              }
              #pragma unroll
              for (int i_4 = 0; i_4 < 2; ++i_4) {
                sm_1[i_4] = max(sm_1[i_4], smp_1[i_4]);
              }
              #pragma unroll
              for (int i_5 = 0; i_5 < 2; ++i_5) {
                sm_1[i_5] = max(sm_1[i_5], -0x1.2ced32a16a1b1p+126f/*-1.000000e+38*/);
              }
              #pragma unroll
              for (int i_6 = 0; i_6 < 2; ++i_6) {
                ss_1[i_6] = exp2f(((smp_1[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (sm_1[i_6] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
              }
              #pragma unroll
              for (int i_7 = 0; i_7 < 64; ++i_7) {
                acc_s_1[i_7] = exp2f(((acc_s_1[i_7] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (sm_1[((i_7 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
              }
              #pragma unroll
              for (int i_8 = 0; i_8 < 2; ++i_8) {
                ssum_1[i_8] = 0x0p+0f/*0.000000e+00*/;
                #pragma unroll
                for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
                  ssum_1[i_8] = (ssum_1[i_8] + acc_s_1[((((rv_1 & 15) * 4) + (i_8 * 2)) + (rv_1 >> 4))]);
                }
                ssum_1[i_8] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(ssum_1[i_8]);
              }
              #pragma unroll
              for (int i_9 = 0; i_9 < 2; ++i_9) {
                ls_1[i_9] = ((ls_1[i_9] * ss_1[i_9]) + ssum_1[i_9]);
              }
              #pragma unroll
              for (int i_10 = 0; i_10 < 16; ++i_10) {
                uint2 __1;
                float4 v_ = *(float4*)(acc_s_1 + (i_10 * 4));
                ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
                ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
                *(uint2*)(acc_s_cast_1 + (i_10 * 4)) = __1;
              }
            } else {
              int64_t t0 = clk::read_clock();
              if ((gi_kc1 % 2) == 0) {
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((half_t*)buf_dyn_shmem)[65536])));
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((half_t*)buf_dyn_shmem)[0])));
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_1 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
                  tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + ((((ki_2 >> 2) * 16384) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_1 + 0)), 1);
                }
                tl::warpgroup_commit_batch();
              } else {
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((half_t*)buf_dyn_shmem)[65536])));
                tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((half_t*)buf_dyn_shmem)[16384])));
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_1 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
                  tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 16384) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_1 + 0)), 1);
                }
                tl::warpgroup_commit_batch();
              }
              int64_t t2 = clk::read_clock();
              v_full[0].wait((gi_vc1 & 1));
              int64_t t3 = clk::read_clock();
              if ((gi_vc1 % 2) == 0) {
                tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_4, (&(((half_t*)buf_dyn_shmem)[32768])));
                tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_1 + 0), 32);
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_1 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
                  tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_1 + (ki_4 * 8)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_1 + 0), 1);
                }
                tl::warpgroup_commit_batch();
              } else {
                tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_5, (&(((half_t*)buf_dyn_shmem)[49152])));
                tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_1 + 0), 32);
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_1 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
                  tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_1 + (ki_5 * 8)), uint64_t(desc_b_5 + ((ki_5 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_1 + 0), 1);
                }
                tl::warpgroup_commit_batch();
              }
              int64_t t4 = clk::read_clock();
              wg_sched_12[0].arrive();
              tl::wait_wgmma<1>();
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_1 + 0), 64);
              int64_t t5 = clk::read_clock();
              k_empty[0].arrive();
              if (n_idx_1 == (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) + (sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))))) {
                #pragma unroll
                for (int i_11 = 0; i_11 < 64; ++i_11) {
                  float condval_1;
                  if (((((((n_idx_1 * 128) + ((i_11 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) + 64) <= (((((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_11 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
                    condval_1 = acc_s_1[i_11];
                  } else {
                    condval_1 = -CUDART_INF_F;
                  }
                  acc_s_1[i_11] = condval_1;
                }
              }
              int64_t t6 = clk::read_clock();
              *(float2*)(smp_1 + 0) = *(float2*)(sm_1 + 0);
              float broadcast_var_5 = -CUDART_INF_F;
              *(float2*)(sm_1 + 0) = make_float2(broadcast_var_5, broadcast_var_5);
              #pragma unroll
              for (int i_12 = 0; i_12 < 2; ++i_12) {
                sm_1_clear_1[i_12] = -CUDART_INF_F;
                #pragma unroll
                for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
                  sm_1_clear_1[i_12] = max(sm_1_clear_1[i_12], acc_s_1[((((rv_2 & 15) * 4) + (i_12 * 2)) + (rv_2 >> 4))]);
                }
                sm_1_clear_1[i_12] = tl::AllReduce<tl::MaxOp, 4, 1, 128, tl::NamedBarrier<128>>::run(sm_1_clear_1[i_12]);
                sm_1[i_12] = max(sm_1[i_12], sm_1_clear_1[i_12]);
              }
              #pragma unroll
              for (int i_13 = 0; i_13 < 2; ++i_13) {
                sm_1[i_13] = max(sm_1[i_13], smp_1[i_13]);
              }
              #pragma unroll
              for (int i_14 = 0; i_14 < 2; ++i_14) {
                sm_1[i_14] = max(sm_1[i_14], -0x1.2ced32a16a1b1p+126f/*-1.000000e+38*/);
              }
              #pragma unroll
              for (int i_15 = 0; i_15 < 2; ++i_15) {
                ss_1[i_15] = exp2f(((smp_1[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (sm_1[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
              }
              #pragma unroll
              for (int i_16 = 0; i_16 < 64; ++i_16) {
                acc_s_1[i_16] = exp2f(((acc_s_1[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (sm_1[((i_16 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
              }
              #pragma unroll
              for (int i_17 = 0; i_17 < 2; ++i_17) {
                ssum_1[i_17] = 0x0p+0f/*0.000000e+00*/;
                #pragma unroll
                for (int rv_3 = 0; rv_3 < 32; ++rv_3) {
                  ssum_1[i_17] = (ssum_1[i_17] + acc_s_1[((((rv_3 & 15) * 4) + (i_17 * 2)) + (rv_3 >> 4))]);
                }
                ssum_1[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 128, tl::NamedBarrier<128>>::run(ssum_1[i_17]);
              }
              #pragma unroll
              for (int i_18 = 0; i_18 < 2; ++i_18) {
                ls_1[i_18] = ((ls_1[i_18] * ss_1[i_18]) + ssum_1[i_18]);
              }
              int64_t t7 = clk::read_clock();
              tl::wait_wgmma<0>();
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_1 + 0), 64);
              int64_t t8 = clk::read_clock();
              v_empty[0].arrive();
              #pragma unroll
              for (int i_19 = 0; i_19 < 64; ++i_19) {
                acc_o_1[i_19] = (acc_o_1[i_19] * ss_1[((i_19 & 3) >> 1)]);
              }
              #pragma unroll
              for (int i_20 = 0; i_20 < 16; ++i_20) {
                uint2 __2;
                float4 v__1 = *(float4*)(acc_s_1 + (i_20 * 4));
                ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
                ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
                *(uint2*)(acc_s_cast_1 + (i_20 * 4)) = __2;
              }
              gi_vc1 = (gi_vc1 + 1);
              clk::clock_accum((&(timing[0])), (t2 - t0));
              clk::clock_accum((&(timing[1])), 0);
              clk::clock_accum((&(timing[2])), (t3 - t2));
              clk::clock_accum((&(timing[3])), (t4 - t3));
              clk::clock_accum((&(timing[4])), (t5 - t4));
              clk::clock_accum((&(timing[5])), (t6 - t5));
              clk::clock_accum((&(timing[6])), (t7 - t6));
              clk::clock_accum((&(timing[7])), (t8 - t7));
              clk::clock_accum_count((&(timing[8])));
            }
            gi_kc1 = (gi_kc1 + 1);
          }
          v_full[0].wait((gi_vc1 & 1));
          if ((gi_vc1 % 2) == 0) {
            tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_6, (&(((half_t*)buf_dyn_shmem)[32768])));
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_1 + 0), 32);
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_1 + 0), 64);
            tl::warpgroup_arrive();
            #pragma unroll
            for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
              tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_1 + (ki_6 * 8)), uint64_t(desc_b_6 + ((ki_6 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_1 + 0), 1);
            }
            tl::warpgroup_commit_batch();
          } else {
            tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_7, (&(((half_t*)buf_dyn_shmem)[49152])));
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_1 + 0), 32);
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_1 + 0), 64);
            tl::warpgroup_arrive();
            #pragma unroll
            for (int ki_7 = 0; ki_7 < 8; ++ki_7) {
              tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_1 + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_1 + 0), 1);
            }
            tl::warpgroup_commit_batch();
          }
          tl::wait_wgmma<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_1 + 0), 64);
          v_empty[0].arrive();
          gi_vc1 = (gi_vc1 + 1);
          #pragma unroll
          for (int i_21 = 0; i_21 < 64; ++i_21) {
            acc_o_1[i_21] = (acc_o_1[i_21] / ls_1[((i_21 & 3) >> 1)]);
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 8; ++i_22) {
            tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((i_22 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 65536)])), __pack_half2(((half_t)acc_o_1[(i_22 * 8)]), ((half_t)acc_o_1[((i_22 * 8) + 1)])), __pack_half2(((half_t)acc_o_1[((i_22 * 8) + 2)]), ((half_t)acc_o_1[((i_22 * 8) + 3)])), __pack_half2(((half_t)acc_o_1[((i_22 * 8) + 4)]), ((half_t)acc_o_1[((i_22 * 8) + 5)])), __pack_half2(((half_t)acc_o_1[((i_22 * 8) + 6)]), ((half_t)acc_o_1[((i_22 * 8) + 7)])));
          }
          tl::fence_proxy_async();
          tl::__sync_thread_partial<3, 128>();
          if (((int)threadIdx.x) == 128) {
            tl::tma_store(output_desc, (&(((half_t*)buf_dyn_shmem)[65536])), 0, ((((((w_1 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_1 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_1 * 132) + ((int)blockIdx.x)) % 8)), ((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w_1 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_store(output_desc, (&(((half_t*)buf_dyn_shmem)[69632])), 64, ((((((w_1 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_1 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_1 * 132) + ((int)blockIdx.x)) % 8)), ((((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)), ((((w_1 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_store_arrive();
            tl::tma_store_wait<0>();
          }
          #pragma unroll
          for (int i_23 = 0; i_23 < 2; ++i_23) {
            ls_1[i_23] = (log2f(ls_1[i_23]) + (sm_1[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
          }
          if ((((int)threadIdx.x) % 4) == 0) {
            #pragma unroll
            for (int i_24 = 0; i_24 < 2; ++i_24) {
              lse[((((((((((((w_1 * 132) + ((int)blockIdx.x)) / 4096) * 32768) + (((((w_1 * 132) + ((int)blockIdx.x)) % 4096) / 128) * 32768)) + ((((w_1 * 132) + ((int)blockIdx.x)) % 8) * 4096)) + (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128)) + ((sub_idx_1 * (31 - (((((w_1 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (i_24 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] = ls_1[i_24];
            }
          }
        }
      }
    } else {
      tl::warpgroup_reg_alloc<240>();
      for (int w_2 = 0; w_2 < 32; ++w_2) {
        if (1024 <= ((w_2 * 33) + (((int)blockIdx.x) >> 2))) {
          break;
        }
        for (int sub_idx_2 = 0; sub_idx_2 < 2; ++sub_idx_2) {
          if (((int)threadIdx.x) == 256) {
            q_full_2[0].expect_transaction(16384);
            tl::tma_load(q_desc, q_full_2[0], (&(((half_t*)buf_dyn_shmem)[73728])), 0, ((((((w_2 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_2 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_2 * 132) + ((int)blockIdx.x)) % 8)), (((((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + 64), ((((w_2 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_load(q_desc, q_full_2[0], (&(((half_t*)buf_dyn_shmem)[77824])), 64, ((((((w_2 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_2 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_2 * 132) + ((int)blockIdx.x)) % 8)), (((((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + 64), ((((w_2 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
          }
          q_full_2[0].arrive();
          q_full_2[0].wait((gi_q2 & 1));
          gi_q2 = (gi_q2 + 1);
          #pragma unroll
          for (int i_25 = 0; i_25 < 16; ++i_25) {
            float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(acc_o_2 + (i_25 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
          }
          float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
          *(float2*)(ls_2 + 0) = make_float2(broadcast_var_7, broadcast_var_7);
          float broadcast_var_8 = -CUDART_INF_F;
          *(float2*)(sm_2 + 0) = make_float2(broadcast_var_8, broadcast_var_8);
          for (int n_idx_2 = 0; n_idx_2 < ((((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) + (sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2)))) + 1); ++n_idx_2) {
            wg_sched_12[0].wait((gi_kc2 & 1));
            k_full[0].wait((gi_kc2 & 1));
            #pragma unroll
            for (int i_26 = 0; i_26 < 16; ++i_26) {
              float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
              *(float4*)(acc_s_2 + (i_26 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
            }
            if ((gi_kc2 % 2) == 0) {
              tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_4, (&(((half_t*)buf_dyn_shmem)[73728])));
              tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_8, (&(((half_t*)buf_dyn_shmem)[0])));
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_2 + 0), 64);
              tl::warpgroup_arrive();
              #pragma unroll
              for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
                tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_4 + ((((ki_8 >> 2) * 8192) + ((ki_8 & 3) * 32)) >> 4)), uint64_t(desc_b_8 + ((((ki_8 >> 2) * 16384) + ((ki_8 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_2 + 0)), 1);
              }
              tl::warpgroup_commit_batch();
            } else {
              tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_5, (&(((half_t*)buf_dyn_shmem)[73728])));
              tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_9, (&(((half_t*)buf_dyn_shmem)[16384])));
              tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_2 + 0), 64);
              tl::warpgroup_arrive();
              #pragma unroll
              for (int ki_9 = 0; ki_9 < 8; ++ki_9) {
                tl::wgmma_ss<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, false, 1, 1>(uint64_t(desc_a_5 + ((((ki_9 >> 2) * 8192) + ((ki_9 & 3) * 32)) >> 4)), uint64_t(desc_b_9 + ((((ki_9 >> 2) * 16384) + ((ki_9 & 3) * 32)) >> 4)), ((uint32_t*)(acc_s_2 + 0)), 1);
              }
              tl::warpgroup_commit_batch();
            }
            if (0 < n_idx_2) {
              v_full[0].wait((gi_vc2 & 1));
              if ((gi_vc2 % 2) == 0) {
                tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_10, (&(((half_t*)buf_dyn_shmem)[32768])));
                tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_2 + 0), 32);
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_10 = 0; ki_10 < 8; ++ki_10) {
                  tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_2 + (ki_10 * 8)), uint64_t(desc_b_10 + ((ki_10 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_2 + 0), 1);
                }
                tl::warpgroup_commit_batch();
              } else {
                tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_11, (&(((half_t*)buf_dyn_shmem)[49152])));
                tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_2 + 0), 32);
                tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 64);
                tl::warpgroup_arrive();
                #pragma unroll
                for (int ki_11 = 0; ki_11 < 8; ++ki_11) {
                  tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_2 + (ki_11 * 8)), uint64_t(desc_b_11 + ((ki_11 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_2 + 0), 1);
                }
                tl::warpgroup_commit_batch();
              }
            }
            wg_sched_21[0].arrive();
            if (0 < n_idx_2) {
              tl::wait_wgmma<1>();
            } else {
              tl::wait_wgmma<0>();
            }
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_s_2 + 0), 64);
            k_empty[0].arrive();
            if (n_idx_2 == (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) + (sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))))) {
              #pragma unroll
              for (int i_27 = 0; i_27 < 64; ++i_27) {
                float condval_2;
                if (((((((n_idx_2 * 128) + ((i_27 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_27 & 1)) + 64) <= (((((((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_27 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
                  condval_2 = acc_s_2[i_27];
                } else {
                  condval_2 = -CUDART_INF_F;
                }
                acc_s_2[i_27] = condval_2;
              }
            }
            *(float2*)(smp_2 + 0) = *(float2*)(sm_2 + 0);
            float broadcast_var_10 = -CUDART_INF_F;
            *(float2*)(sm_2 + 0) = make_float2(broadcast_var_10, broadcast_var_10);
            #pragma unroll
            for (int i_28 = 0; i_28 < 2; ++i_28) {
              sm_2_clear[i_28] = -CUDART_INF_F;
              #pragma unroll
              for (int rv_4 = 0; rv_4 < 32; ++rv_4) {
                sm_2_clear[i_28] = max(sm_2_clear[i_28], acc_s_2[((((rv_4 & 15) * 4) + (i_28 * 2)) + (rv_4 >> 4))]);
              }
              sm_2_clear[i_28] = tl::AllReduce<tl::MaxOp, 4, 1, 256, tl::NamedBarrier<128>>::run(sm_2_clear[i_28]);
              sm_2[i_28] = max(sm_2[i_28], sm_2_clear[i_28]);
            }
            #pragma unroll
            for (int i_29 = 0; i_29 < 2; ++i_29) {
              sm_2[i_29] = max(sm_2[i_29], smp_2[i_29]);
            }
            #pragma unroll
            for (int i_30 = 0; i_30 < 2; ++i_30) {
              sm_2[i_30] = max(sm_2[i_30], -0x1.2ced32a16a1b1p+126f/*-1.000000e+38*/);
            }
            #pragma unroll
            for (int i_31 = 0; i_31 < 2; ++i_31) {
              ss_2[i_31] = exp2f(((smp_2[i_31] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (sm_2[i_31] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
            }
            #pragma unroll
            for (int i_32 = 0; i_32 < 64; ++i_32) {
              acc_s_2[i_32] = exp2f(((acc_s_2[i_32] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (sm_2[((i_32 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
            }
            #pragma unroll
            for (int i_33 = 0; i_33 < 2; ++i_33) {
              ssum_2[i_33] = 0x0p+0f/*0.000000e+00*/;
              #pragma unroll
              for (int rv_5 = 0; rv_5 < 32; ++rv_5) {
                ssum_2[i_33] = (ssum_2[i_33] + acc_s_2[((((rv_5 & 15) * 4) + (i_33 * 2)) + (rv_5 >> 4))]);
              }
              ssum_2[i_33] = tl::AllReduce<tl::SumOp, 4, 1, 256, tl::NamedBarrier<128>>::run(ssum_2[i_33]);
            }
            #pragma unroll
            for (int i_34 = 0; i_34 < 2; ++i_34) {
              ls_2[i_34] = ((ls_2[i_34] * ss_2[i_34]) + ssum_2[i_34]);
            }
            tl::wait_wgmma<0>();
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 64);
            if (0 < n_idx_2) {
              v_empty[0].arrive();
              #pragma unroll
              for (int i_35 = 0; i_35 < 64; ++i_35) {
                acc_o_2[i_35] = (acc_o_2[i_35] * ss_2[((i_35 & 3) >> 1)]);
              }
              gi_vc2 = (gi_vc2 + 1);
            }
            #pragma unroll
            for (int i_36 = 0; i_36 < 16; ++i_36) {
              uint2 __3;
              float4 v__2 = *(float4*)(acc_s_2 + (i_36 * 4));
              ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
              ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
              *(uint2*)(acc_s_cast_2 + (i_36 * 4)) = __3;
            }
            gi_kc2 = (gi_kc2 + 1);
          }
          v_full[0].wait((gi_vc2 & 1));
          if ((gi_vc2 % 2) == 0) {
            tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_12, (&(((half_t*)buf_dyn_shmem)[32768])));
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_2 + 0), 32);
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 64);
            tl::warpgroup_arrive();
            #pragma unroll
            for (int ki_12 = 0; ki_12 < 8; ++ki_12) {
              tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_2 + (ki_12 * 8)), uint64_t(desc_b_12 + ((ki_12 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_2 + 0), 1);
            }
            tl::warpgroup_commit_batch();
          } else {
            tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_13, (&(((half_t*)buf_dyn_shmem)[49152])));
            tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(acc_s_cast_2 + 0), 32);
            tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 64);
            tl::warpgroup_arrive();
            #pragma unroll
            for (int ki_13 = 0; ki_13 < 8; ++ki_13) {
              tl::wgmma_rs<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(acc_s_cast_2 + (ki_13 * 8)), uint64_t(desc_b_13 + ((ki_13 * 2048) >> 4)), reinterpret_cast<uint32_t*>(acc_o_2 + 0), 1);
            }
            tl::warpgroup_commit_batch();
          }
          tl::wait_wgmma<0>();
          tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc_o_2 + 0), 64);
          v_empty[0].arrive();
          gi_vc2 = (gi_vc2 + 1);
          #pragma unroll
          for (int i_37 = 0; i_37 < 64; ++i_37) {
            acc_o_2[i_37] = (acc_o_2[i_37] / ls_2[((i_37 & 3) >> 1)]);
          }
          #pragma unroll
          for (int i_38 = 0; i_38 < 8; ++i_38) {
            tl::ptx_stmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((i_38 >> 2) * 4096) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_38 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_38 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 73728)])), __pack_half2(((half_t)acc_o_2[(i_38 * 8)]), ((half_t)acc_o_2[((i_38 * 8) + 1)])), __pack_half2(((half_t)acc_o_2[((i_38 * 8) + 2)]), ((half_t)acc_o_2[((i_38 * 8) + 3)])), __pack_half2(((half_t)acc_o_2[((i_38 * 8) + 4)]), ((half_t)acc_o_2[((i_38 * 8) + 5)])), __pack_half2(((half_t)acc_o_2[((i_38 * 8) + 6)]), ((half_t)acc_o_2[((i_38 * 8) + 7)])));
          }
          tl::fence_proxy_async();
          tl::__sync_thread_partial<4, 128>();
          if (((int)threadIdx.x) == 256) {
            tl::tma_store(output_desc, (&(((half_t*)buf_dyn_shmem)[73728])), 0, ((((((w_2 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_2 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_2 * 132) + ((int)blockIdx.x)) % 8)), (((((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + 64), ((((w_2 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_store(output_desc, (&(((half_t*)buf_dyn_shmem)[77824])), 64, ((((((w_2 * 132) + ((int)blockIdx.x)) / 4096) * 8) + (((((w_2 * 132) + ((int)blockIdx.x)) % 1024) / 128) * 8)) + (((w_2 * 132) + ((int)blockIdx.x)) % 8)), (((((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128) + ((sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + 64), ((((w_2 * 132) + ((int)blockIdx.x)) % 4096) / 1024));
            tl::tma_store_arrive();
            tl::tma_store_wait<0>();
          }
          #pragma unroll
          for (int i_39 = 0; i_39 < 2; ++i_39) {
            ls_2[i_39] = (log2f(ls_2[i_39]) + (sm_2[i_39] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
          }
          if ((((int)threadIdx.x) % 4) == 0) {
            #pragma unroll
            for (int i_40 = 0; i_40 < 2; ++i_40) {
              lse[((((((((((((w_2 * 132) + ((int)blockIdx.x)) / 4096) * 32768) + (((((w_2 * 132) + ((int)blockIdx.x)) % 4096) / 128) * 32768)) + ((((w_2 * 132) + ((int)blockIdx.x)) % 8) * 4096)) + (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 128)) + ((sub_idx_2 * (31 - (((((w_2 * 132) + ((int)blockIdx.x)) % 128) / 8) * 2))) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (i_40 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 64)] = ls_2[i_40];
            }
          }
        }
      }
    }
  }
}

