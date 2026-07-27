# Gated DeltaNet Backward Prefill-Style Optimization Plan

本文档记录 Gated DeltaNet backward 的下一轮性能改造计划。目标不是重写一个和 prefill 完全相同的 kernel，而是把 prefill 优化中已经证明有效的结构迁移到 backward：用 chunk / segment 级 affine summary 缩短跨 chunk 的长递推依赖，并让主要 workload 形状更接近 GPU 擅长的 GEMM / tiled compute。

## 1. 当前状态

当前 `GatedDeltaNetBwdKernel` 的 pipeline 是：

1. `fused_prepare_compute_w_u`: 重算 forward 中的 `w` / `u`。
2. `bwd_parallel`: 按 chunk 并行计算不依赖跨 chunk `dh` 的局部梯度。
3. segment-carry path: 生成每个 chunk 的 successor-side `dh` carry，并行计算
   `dk_corr` / `du_corr` / `dg_corr`。
4. `compute_dw_correction`: 把 future-state `du_corr` 传回 `w`。
5. `compute_w_u_bwd_full`: 完整反传 `w/u` 和 chunk-local
   `A=(I+L)^{-1}`，得到 `dk_prepare` / `dv` / `dbeta` / `dg_prepare`。
6. merge: 合并各路 `dk`，并对 `dg_cum` 做 chunk-local reverse cumsum。

这里原来的主要长依赖在第 3 步。`bwd_parallel` 已经是
`num_chunks x B x H` 的并行 grid；旧实现像 prefill wall 的部分，是
`dh_recurrence_bwd` 对每个 `(B,H)` stream 做完整反向 chunk loop。当前
segment-carry path 已经把它拆成 segment summary、boundary scan、local expansion
和 chunk-parallel correction。

初始 smoke 也支持这个判断。当前本地 H200 环境下，`B=1,S=4096,H=16,DK=DV=64,chunk=64,fp16,BTHD-wrapper` 的 legacy backward 是 `0.530 ms`；stage timing 中 `dh_recurrence_bwd` 是 `0.420 ms`，约占 full backward 的 79%。这个数字不是发布 benchmark，只作为第一轮定位瓶颈的工程记录。

### 1.1 Correctness closure

优化过程中发现，历史 Gated DeltaNet backward 虽然计算了 `dAw/dAu`，wrapper
却没有继续反传 chunk-local inverse，也没有加入 future carry 对 `dw` 的修正。
小输入和 `atol=rtol=5e-2` 的 smoke gate 可以通过，但 accuracy diagnostic 显示
`dk/dg/dbeta` 的 L2 relative error 分别约为 `4.21%/3.89%/2.48%`。

当前实现补齐了：

```text
dw_corr = -(du_corr @ S_start^T) * exp(g_i + g_last)
dL      = -A^T @ dA @ A^T
dL -> dk_A, dbeta_A, dg_A
```

在 `B=1,H=2,S=128,DK=DV=128,chunk=64,fp16,seed=42` 上，相对 fp32
differentiable reference 的结果为：

| gradient | max abs | L2 relative | cosine |
| --- | ---: | ---: | ---: |
| `dq` | `2.253e-4` | `0.376%` | `0.999993` |
| `dk` | `2.829e-4` | `0.376%` | `0.999993` |
| `dv` | `2.173e-4` | `0.337%` | `0.999995` |
| `dg` | `1.679e-4` | `0.887%` | `0.999961` |
| `dbeta` | `2.990e-4` | `0.321%` | `0.999995` |

完整梯度链下的 H200 backward-only timing：

| sequence | TileOps complete bwd | FLA 0.5.1 bwd | TileOps speedup |
| ---: | ---: | ---: | ---: |
| 4K | `0.552103 ms` | `0.830285 ms` | `1.50x` |
| 8K | `1.019251 ms` | `1.100744 ms` | `1.08x` |
| 16K | `1.935167 ms` | `2.137829 ms` | `1.10x` |

backward recompute、`bwd_parallel`、完整 prepare backward 和 carry correction
分别启用 TileLang fast-math；forward 默认 lowering 保持不变。主要收益来自
`bwd_parallel`，其 16K stage latency 从约 `0.742 ms` 降到 `0.670 ms`，其余
三个阶段提供较小的累积收益。完整五梯度诊断与关闭 fast-math 时逐项一致，
因此这些选择保留在当前实现中，而不是用放宽正确性门限换取速度。

合同为 `B=1,H=16,DK=DV=128,chunk=64,fp16,BHSD`，
`warmup=5,repeat=20,trials=3`。旧的 `0.620904/1.148775/2.112842 ms`
来自 incomplete-gradient historical path，只用于说明优化前的实现状态，不再作为
最终 correctness/performance claim。

## 2. 和 Prefill 对齐的核心结构

Prefill 的跨 chunk 状态可以写成 affine transition：

```text
H_end[i] = M_i @ H_start[i] + B_i
```

其中 `M_i` 表示第 `i` 个 chunk 对输入 state 的线性作用，`B_i` 表示该 chunk 自己产生的新 state。CP split 的关键不是消除因果性，而是把长链拆成：

1. 每个 chunk / segment 并行生成 summary `(M, B)`。
2. 对 summary 做 prefix / correction，得到各 segment 的正确 `H_start`。
3. segment 内部只保留短 replay。

Backward 的跨 chunk `dh` 传播是这个结构的反向伴随形式：

```text
dH_start[i] = M_i^T @ dH_end[i] + G_i
```

其中 `G_i` 是第 `i` 个 chunk 局部 loss 对 `H_start[i]` 的梯度贡献。当前 `dh_recurrence_bwd` 做的是逐 chunk 反向串行应用这个关系；优化目标是把它改成 chunk / segment summary + reverse scan。

## 3. Layout 计划

输入 layout 需要纳入本次改造。当前训练 forward/backward 采用 BHSD：

```text
q/k/do: [B, H, S, DK]
v:      [B, H, S, DV]
g/beta: [B, H, S]
```

而优化后的 serving prefill 已经支持并主要使用 BTHD：

```text
q/k/do: [B, S, H, DK]
v:      [B, S, H, DV]
g/beta: [B, S, H]
```

本计划采用 BTHD-first optimized path：

1. 新的高性能 backward hot path 以 `layout="bthd"` 为主要目标，和 prefill / Qwen / FLA 口径对齐。
2. 现有 BHSD `GatedDeltaNetBwdOp` 保留兼容，不在第一轮删除。
3. 如果 public op surface 需要统一，优先给 `GatedDeltaNetBwdOp` 增加 `layout` 参数，而不是立刻新增一个名字相近的 backward op。
4. 内部 hot path 不做 materialized transpose；BHSD 只作为 wrapper / fallback / compatibility path。

这个选择的原因是：如果 backward 优化继续只围绕 BHSD 写，后续与 prefill benchmark、Qwen-shaped workload、FLA reference 对齐时都会引入额外 transpose 和证据口径问题。

## 4. Backward Artifact Contract

Backward 和 inference prefill 的 contract 不同。Prefill public op 只输出：

```text
o, final_state
```

Backward 需要更多信息：

```text
do, q, k, v, g, beta, forward boundary states or recomputed summaries
```

第一版优先采用 recompute-first contract：

1. `q/k/v/g/beta/do` 是 backward 输入。
2. `S` 或等价 chunk boundary state 仍可作为 legacy path 输入。
3. 优化 path 可以重算 chunk-local prepare artifacts，避免把 `Aw/Au` 继续作为 public prefill artifact。
4. 如果后续 profiling 证明 recompute 太贵，再考虑 forward 保存 compact summary。

这和 prefill 的方向一致：对外 API 保持干净，把训练 backward 需要的中间量留在 implementation contract 中，而不是暴露成 serving prefill 的输出。

## 5. 第一阶段实现切口

第一阶段不直接替换全量 backward。先做三个可验证切口：

### 5.1 Baseline 和 stage timing

新增 benchmark harness，记录：

1. full legacy backward latency。
2. `fused_prepare_compute_w_u` latency。
3. `bwd_parallel` latency。
4. `dh_recurrence_bwd` latency。
5. `compute_w_u_bwd` latency。

目标是量化 `dh_recurrence_bwd` 在长序列中的占比，避免在错误位置做局部 tuning。

还有一个必须先处理的实现约束：原始 legacy backward 在 Qwen-style `DK=DV=128` row 上会因为 `dh_recurrence_bwd` 的 `DK x DV` shared buffer 超过动态 shared memory 限制而失败。一次本地 smoke 在 `B=1,S=4096,H=16,DK=DV=128,chunk=64,fp16` 下报错：

```text
Failed to set the allowed dynamic shared memory size to 298528
```

因此 optimized backward 不能只把现有 `dh_recurrence_bwd` 外层改成 scan；它还需要像 prefill forward 的 `h_block_v` 一样，对 `DV` 维做 tiling 或采用等价的分块 state layout。第一版实现已经给 `dh_recurrence_bwd` 加了 `recurrence_block_v`，让 `DK=DV=128` 的 smoke row 可以进入 correctness 和 timing。这个 patch 解除的是 shared-memory blocker，不是最终的 reverse-scan 性能优化。

### 5.2 Layout adapter baseline

同一组随机输入分别测：

1. BHSD legacy backward。
2. BTHD 输入经过 wrapper transpose 后调用 legacy backward。
3. 未来 BTHD-native optimized backward。

这样可以把 layout 成本和 kernel 算法成本分开。

### 5.3 Reverse affine scan prototype

先实现一个 reference-level prototype，验证：

```text
dH_start[i] = M_i^T @ dH_end[i] + G_i
```

可以按 segment 组合：

```text
dH_start[left] =
    (M_left_to_right)^T @ dH_end[right] + G_left_to_right
```

第一版 prototype 可以是 Torch / debug kernel，不要求立即比现有 kernel 快；它的作用是固定数学和 correction 口径。

当前已加入一个 Torch reference prototype：

```text
experiments/gated_deltanet_bwd_prefill_style/prototype_reverse_affine_scan.py
```

它先覆盖当前 `dh_recurrence_bwd` 暴露出来的跨 chunk carry 形态：

```text
X[i] = G[i] + alpha[i] * X[i + 1]
```

并验证 segment summary：

```text
X[left] = A_segment * X[right] + B_segment
```

可以通过 affine composition 与原始逐 chunk reverse recurrence 对齐。`chunks=64,DK=DV=128,segment=8,fp32`
本地检查的 max abs 为 `1.19e-07`。这个 prototype 只证明组合律和数值口径，尚未替换 hot path。

随后又加入了真实中间量 gate：

```text
experiments/gated_deltanet_bwd_prefill_style/prototype_reverse_affine_real.py
```

这个脚本运行现有 forward 和 `bwd_parallel`，拿到真实的 `dh_local`、`v_new`、`S`、`k`、`g_cum`，
再用 Torch segmented reverse affine scan 生成每个 chunk 的 successor-side carry，并重算
`dk_corr` / `du_corr` / `dg_corr`。本地检查结果：

1. `B=1,H=1,S=128,DK=DV=64,chunk=64,segment=1,fp16`: pass。
2. `B=1,H=2,S=128,DK=DV=128,chunk=64,segment=2,fp16`: pass，`dk_corr` max abs `6.84e-03`。

这一步把 reverse scan 的 kernel 边界固定为：先产生每个 chunk 在当前 sequential kernel 中使用的
`dh_buf` / successor carry，再并行重算 correction。后续 TileLang hot path 可以按这个 boundary
拆成 carry-scan producer 和 correction consumer。

### 5.4 TileLang split-carry / segment-carry hot path

第一版 TileLang hot path 已经按 5.3 的边界接入。对 `dim_v=128,chunk=64`
这类 d128 backward 热点，当前默认开启 segment-carry 版本：

```text
threads=128
parallel_threads=256
recurrence_threads=128
recurrence_block_v=64
recurrence_split_carry=2
```

`recurrence_split_carry=1` 先把旧的 `dh_recurrence_bwd` 拆成两个 kernel：

1. `_dh_carry_after_scan_tl`
   - 输入：`g_cum`、`dh_local`。
   - 输出：每个 chunk 的 successor-side carry。
   - 这个 kernel 仍然按 chunk 反向扫描，因此它不是最终 CP split / segment scan。

2. `_dh_correction_from_carry_tl`
   - 输入：`g_cum`、`k`、`v_new`、`S` 和上一步 materialized carry。
   - 输出：`dk_corr_partial`、`du_corr`、`dg_corr_partial`。
   - grid 改成 `(V tile, chunk, B*H)`，把 correction GEMM 从原来的长 chunk loop 中拆出来。

这个切口的意义是：先把“carry 传播”和“correction 计算”分离。真正需要跨 chunk 依赖的是 carry；
`du_corr = k_scaled @ dh_carry`、`dk_corr = v_new @ dh_carry^T`、`dg_corr` 的局部 reduction
都可以在给定 carry 后按 chunk 并行。

`recurrence_split_carry=2` 进一步把 carry producer 做成 segment 结构：

1. `_dh_segment_summary_tl`
   - 每个 segment 内部把 short reverse recurrence 总结成
     `X[left] = B_segment + A_segment * X[right]`。
   - `A_segment` 是 segment 内 chunk boundary decay 的乘积；
     `B_segment` 是 segment 自己产生的 local carry contribution。

2. `_dh_segment_boundary_scan_tl`
   - 对 segment summary 做 reverse suffix scan。
   - 对 `S=4096,chunk=64,segment_chunks=8`，cross-segment 依赖长度从 64 个 chunk
     降到 8 个 segment。

3. `_dh_segment_local_carry_tl`
   - 给定每个 segment 的 successor-side boundary carry，在 segment 内展开每个 chunk 的
     successor carry。
   - 这一步仍有短 recurrence，但 segment 之间已经并行。

4. `_dh_correction_from_carry_tl`
   - 复用 split-carry 的 correction consumer。

本地 H200、`B=1,S=4096,H=16,DK=DV=128,chunk=64,fp16,BTHD-wrapper`、
`warmup=5,repeat=20,trials=3` 的 stage timing：

| path | full backward | recurrence / carry | correction | partial reduction |
| --- | ---: | ---: | ---: | ---: |
| default recurrence | `1.146564 ms` | `dh_recurrence_bwd 0.587648 ms` | included | `0.039275 ms` |
| split carry | `0.842685 ms` | `dh_carry_after_scan 0.088928 ms` | `dh_correction_from_carry 0.186823 ms` | `0.039422 ms` |
| split carry, block_v=64 default | `0.781947 ms` | `dh_carry_after_scan 0.118798 ms` | `dh_correction_from_carry 0.129728 ms` | `0.021413 ms` |
| segment carry, block_v=64 default | `0.705536 ms` | summary `0.016203 ms` + boundary scan `0.008784 ms` + local carry `0.023086 ms` | `dh_correction_from_carry 0.099558 ms` | `0.020293 ms` |

随后跑了两轮 300-record in-process AKO。第一轮加入 split-carry 维度：

```text
records=300, pass=300, fail=0
best median candidate:
  num_stages=2
  threads=128
  parallel_threads=256
  recurrence_threads=128
  recurrence_block_v=64
  recurrence_split_carry=1
best=0.701216 ms, median=0.706207 ms  # warmup=1,repeat=3,trials=1 搜索口径
```

第二轮加入 segment-carry 维度：

```text
records=300, pass=300, fail=0
best short-repeat median candidate:
  num_stages=2
  threads=128
  parallel_threads=256
  recurrence_threads=256
  recurrence_block_v=64
  recurrence_split_carry=2
best=0.612958 ms, median=0.671614 ms  # warmup=1,repeat=3,trials=1 搜索口径
```

这个 short-repeat best 的 formal rerun 是 `0.768857 ms`，比
`recurrence_threads=128` 的 segment-carry default `0.705536 ms` 慢。因此默认保持
`recurrence_threads=128`。这说明当前 AKO loop 的分工是合理的：短 repeat 用来扩大搜索，
最终默认必须由 stable repeat gate 决定。

当前判断：split-carry 已经把 heavy correction work 从串行 recurrence 中移出；segment-carry
则把 carry producer 从全长 chunk 链推进到 segment summary / boundary scan / local expansion。
下一轮 AKO 应该继续优化 segment summary 与 local expansion 的访存和 launch 结构，而不是回到
monolithic `dh_recurrence_bwd` 里做局部调参。

## 6. Kernel 实现路线

后续高性能 kernel 分三层推进：

1. **Summary producer**
   - 每个 chunk 产生 backward 需要的 `M_i^T` 作用和局部 gradient summary `G_i`。
   - 尽量复用 prefill 中的 prepare / blocksolve / chunk summary 思路。

2. **Reverse correction / scan**
   - 对 chunk summary 做 reverse prefix / segment correction。
   - 目标是把 `num_chunks` 长度的串行 `dh_buf` 链缩短为 segment 级依赖。
   - 当前 segment-carry path 已经把 carry producer 推进到 segment summary / boundary scan；
     剩余工作集中在减少 segment/local expansion 的 launch 与访存开销。

3. **Local backward replay**
   - 给定正确的 `dH_end` 或 corrected adjoint state，在 segment 内计算 `dk_corr` / `du_corr` / `dg_corr`。
   - segment 内可以先保留短 recurrence，再逐步矩阵化。

## 7. Correctness Gates

第一轮 correctness 不改变现有 tolerance policy，先对齐现有 autograd reference：

1. small smoke: `B=2, S=64, H=2, DK=DV=64, chunk=32`。
2. medium: `B=1, S=128/256, H=4, DK=DV=64, chunk=32/64`。
3. dtype: fp32 / fp16 / bf16。
4. layout: BHSD legacy path 与 BTHD wrapper path 结果一致。
5. gradient outputs: `dq/dk/dv/dg/dbeta` 全部检查。

长上下文性能 gate 单独处理，不和 small correctness gate 混在一个 claim 里。

## 8. Performance Gates

建议第一批 benchmark rows：

| row | shape | dtype | layout | purpose |
| --- | --- | --- | --- | --- |
| smoke | B=2, S=4096, H=4, DK=DV=64, chunk=64 | fp16/bf16 | BHSD | legacy comparison |
| qwen-d64 | B=1, S=4096/8192, H=16, DK=DV=64, chunk=64 | fp16/bf16 | BTHD | current legacy stage breakdown |
| qwen-d128-smoke | B=1, S=4096, H=16, DK=DV=128, chunk=64 | fp16/bf16 | BTHD | shared-memory feasibility gate |
| qwen-small | B=1, S=8192, H=16, DK=DV=128, chunk=64 | fp16/bf16 | BTHD | optimized path target |
| qwen-mid | B=1, S=32768, H=16, DK=DV=128, chunk=64 | fp16/bf16 | BTHD | long-chain pressure |
| qwen-head-scale | B=1, S=32768, H=32/64, DK=DV=128, chunk=64 | fp16/bf16 | BTHD | head scaling |

每一行记录：

1. full backward latency。
2. stage breakdown。
3. `dh_recurrence_bwd` share。
4. transpose/wrapper overhead。
5. correctness max abs / allclose status。

## 9. PR 拆分

建议拆成三轮 PR：

1. **PR-1: baseline and layout contract**
   - 增加 design doc。
   - 增加 benchmark / stage breakdown harness。
   - 增加 BTHD wrapper correctness test。
   - 不改变默认 backward kernel。

2. **PR-2: reverse affine scan prototype**
   - 增加 reference prototype。
   - 加 small correctness tests。
   - 记录与 legacy `dh_recurrence_bwd` 的数学对应关系。

3. **PR-3: optimized BTHD backward kernel**
   - 替换或分发到 BTHD-native optimized path。
   - 保留 BHSD fallback。
   - 增加 long-context benchmark rows。

## 10. 非目标

第一轮不做这些事：

1. 不删除现有 BHSD training backward。
2. 不把 inference prefill 的 public contract 改成暴露 `Aw/Au`。
3. 不声称 backward kernel 已经和 prefill 同等成熟。
4. 不把 layout wrapper 的性能当作 BTHD-native kernel 性能。
5. 不把 FLA public benchmark 和 TileOps same-kernel attribution 混在一个结论里。

## 11. 当前判断

Gated DeltaNet backward 目前还没有吸收 prefill 的 CP split / blocksolve 优化。最清楚的高价值切口是 `dh_recurrence_bwd`：它是反向跨 chunk 长链，数学上可以看成 forward affine transition 的 adjoint scan。输入 layout 应该从一开始纳入设计，optimized path 以 BTHD 为主，BHSD 保留兼容。这样后续 kernel、benchmark 和 Qwen/FLA reference 的证据口径会更干净。
