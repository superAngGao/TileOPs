# TileLang FP8 GQA 对齐 FA3 Mainloop 的能力缺口报告

日期：2026-06-07

## 摘要

本报告描述在 TileLang 中实现 Hopper FP8 GQA attention，并对齐
FlashAttention-3（FA3）FP8 forward mainloop 时遇到的 DSL 与 lowering 能力缺口。
目标读者是假定已经具备 TileLang 使用经验、Hopper WGMMA 编程经验，以及基本的
FlashAttention-3 mainloop 背景的开发者。

结论如下：

1. TileLang 可以表达数值正确的 FA3-like source schedule，也支持 inline PTX。
2. 当前主要问题不是无法写出 QK/PV/wait 的源代码顺序，而是 TileLang 缺少
   FA3 FP8 mainloop 所需的两个关键高级 API / lowering contract：
   - in-register fragment reinterpret / layout conversion。TileLang 目前不能以
     高级 API 表达 FA3 中 `tSrS -> tOrP` 的寄存器布局契约。
   - 跨 QK/PV 的 FA3-style grouped WGMMA mainloop contract。TileLang 支持单个
     `T.gemm` 内部的 grouped WGMMA，但缺少跨两个 GEMM 的 scoreboard window
     contract；即使 source/PTX intent 接近 FA3，ptxas 仍可能在 SASS 层将
     WGMMA scoreboard 拆成 per-QGMMA 形式。
3. 当前 kernel 数值正确，但应定位为“能力边界样例”，用于展示 TileLang 在
   FA3-style scoreboard 与“寄存器片段主循环契约”（register-fragment mainloop
   contract）上的能力边界；它不是完整复现 FA3 FP8 寄存器片段主循环契约的实现。

## 1. 目标 Mainloop 契约

目标是在 TileLang 中表达 Hopper FP8 GQA forward attention，并尽量复现 FA3 的
intra-warpgroup overlap。期望的 steady-state 顺序如下：

```text
prepare P[n-1] from previous QK

issue QK[n]
issue PV[n-1]
wait_group<1>        # QK[n] is readable, PV[n-1] remains outstanding
softmax/update P[n]
wait_group<0>        # PV[n-1] is complete
update O / prepare next iteration
```

该契约包含四个关键要求：

1. `PV[n-1]` 消费上一轮 softmax 后准备好的 `tOrP` register fragment。
2. `wait_group<1>` 之后读取当前 QK accumulator `tSrS`，用于 mask、scale 和
   online softmax。
3. 当前轮 softmax 后，`tSrS` 需要被转换为下一轮 PV 所需的 `tOrP`。
4. QK 与 PV 的 WGMMA issue 应保持 FA3-style grouped scoreboard，而不是在每条
   QGMMA 附近形成独立的 `WARPGROUP.ARRIVE` / `WARPGROUP.DEPBAR` 边界。

术语约定：

```text
tSrS:
  QK GEMM 的 accumulator fragment，保存当前 Q tile 与当前 K tile 的 score。

tOrP:
  PV GEMM 的 A operand register fragment，保存 P = softmax(QK)。
  在 FA3 mainloop 中，tOrP 是长期存在的 CUTE register tensor。
  本轮 softmax 后准备 P[n]，下一轮 PV 直接消费该 register fragment。

tOrO:
  PV GEMM 的 output accumulator fragment，保存累积的 O。
```

## 2. FP8 相对 FP16/BF16 FA3 的额外要求

FP16/BF16 FA3 forward mainloop 已经包含 QK/PV overlap、`wait_group<1>`、
online softmax 和 O 累积。FP8 版本在此基础上引入额外的 scale/descale、
register layout conversion，以及 FP8-specific P operand 准备流程。

### 2.1 Scale / Descale Contract

FP8 Q/K/V 需要 scale metadata。对于 GQA，FA3 风格的 descale 通常按 KV head
group 组织；TileOps 内部 kernel 目前常用展开后的 scale buffer：

```text
q_scale:   [batch, heads,    seq_len / 128]
k/v_scale:[batch, heads_kv, seq_len / 128]
```

FA3 Python/CUDA 接口暴露的 compact descale 更接近：

```text
q/k/v descale: [batch, heads_kv]
```

因此，FP8 kernel 需要显式定义 scale contract。scale adapter 问题应与
mainloop overlap / WGMMA grouping 问题区分处理。

### 2.2 QK C Accumulator 到 PV RS-A Operand 的寄存器转换

FP8 FA3 的关键额外操作是：softmax 后的 P 需要作为 PV GEMM 的 A operand，并且
应尽量保留在 register fragment 中。FA3 中第一次准备 P 的结构可概括为：

```cpp
softmax(tSrS);
if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
Tensor tOrP_acc =
    make_tensor(tSrS.data(),
        flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
convert_type_out(tOrP_acc, tOrP);
```

该代码不是普通 elementwise copy。它表达的寄存器契约是：

```text
QK accumulator C-layout fp32 registers
-> FP8 C-register permutation
-> reinterpret as PV RS-A operand layout
-> dtype conversion to fp8 tOrP
```

FP16/BF16 路径也需要将 softmax 后的 P 交给 PV，但 FP8 路径对 register layout
和 dtype conversion 的要求更严格。该转换是否能被 DSL 以 zero-cost
reinterpret 的形式表达，是判断 TileLang kernel 是否等价于 FA3 FP8 mainloop 的
核心条件之一。

### 2.3 Persistent `tOrP` 与跨迭代 Register Layout Contract

Persistent `tOrP` 不是 FP8 独有概念。FP16/BF16 FA3 中也存在长期存活的 `tOrP`
mainloop fragment：

```text
iteration n:
  prepare P[n] after softmax

iteration n + 1:
  PV consumes P[n] through tOrP
```

FP8 的额外困难在于：长期存活本身不足以构成 FA3-equivalent `tOrP`。该 fragment
还必须保留跨迭代的 register layout / swizzle contract。steady-state 中 FA3
继续复用同一个 `tOrP.layout()`：

```cpp
if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
```

因此，需要表达的不是 shared memory / TMA 层面的 swizzle flag，而是跨 loop 的
register fragment layout / swizzle contract：

```text
loop n:
  QK produces tSrS in WGMMA accumulator C layout
  softmax(tSrS)
  apply FP8 C-register permutation
  reinterpret the same physical registers as PV RS-A tOrP

loop n + 1:
  PV consumes the previously prepared tOrP directly
```

如果 DSL 只能表达“另一个 fragment”或“从一个 fragment copy 到另一个 fragment”，
ptxas 看到的可能不再是 FA3 mainloop 中同一个 register fragment 的稳定 def/use
和 layout 语义。

### 2.4 Shared Layout 与 TMA Destination

FP8 FA3 对 K/V shared layout、V transpose / VtMma layout，以及 TMA destination
的物理布局较敏感。该问题属于完整 FA3-equivalent kernel 的验证条件，但不是当前
能力边界样例的主要展示点。相关验证细节见附录 B。

### 2.5 需要的 DSL / Lowering 抽象

一个 FA3-equivalent TileLang FP8 GQA kernel 至少需要以下抽象或验证能力：

| 能力 | 作用 |
| --- | --- |
| loop-external persistent fragment lifetime | 支持跨迭代保存 P fragment |
| in-register fragment layout reinterpret + dtype conversion | 表达 `tSrS -> tOrP` 寄存器契约 |
| QK/PV grouped WGMMA issue + wait contract | 保持 FA3-style scoreboard |
| scale/descale contract adapter | 统一 FA3 compact descale 与 TileOps expanded scale |
| CUTE-compatible smem/TMA layout 表达或验证工具 | 验证 TMA destination 与 WGMMA consumer layout 一致，见附录 B |

## 3. TileLang 当前能力评估

本节直接列出当前实现 FA3-equivalent FP8 GQA mainloop 所需、但 TileLang 尚未完整
提供的能力。这些判断已经通过与 TileLang DeepWiki 的逐项讨论进行了核对：DeepWiki
确认了 loop-external `T.alloc_fragment` 生命周期可用，同时确认当前 TileLang
高级 API 中不存在 FA3 FP8 所需的 in-register fragment reinterpret / layout
conversion、跨 loop register layout contract、跨 QK/PV 的 grouped WGMMA scoreboard
window API，以及 CUTE layout 逐元素等价性 checker。DeepWiki 查询和回复原文见附录 C；
对 TileLang 现有能力的归纳见附录 A；TMA destination 与 CUTE layout 的验证问题见
附录 B。

### 3.1 缺口总览

| 编号 | 能力项 | 当前判断 | 对 FP8 GQA kernel 的影响 |
| --- | --- | --- | --- |
| A | loop-external fragment lifetime | 已支持或理论上支持 | 可表达 P fragment 跨迭代存活；不是主要缺口 |
| B | in-register fragment reinterpret / layout conversion | 缺失 | 无法表达 FA3 `tSrS -> tOrP` 寄存器契约 |
| C | 跨 loop register fragment layout / swizzle contract | 缺失 | 无法声明上一轮 QK C accumulator 的同一组物理 registers 是下一轮 PV RS-A operand |
| D | 跨 QK/PV 的 FA3-style grouped WGMMA mainloop contract | 缺失 | 单个 `T.gemm` grouped WGMMA 已支持；跨 GEMM scoreboard window 缺少高层 API |
| E | exact CUTE smem/TMA layout 验证 | 次级缺口 | 影响完整 FA3-equivalent kernel 的最终验证，当前 boundary sample 不作为主问题 |

后续小节只展开 B、C、D。A 的结论是“已支持，不是主要阻塞项”；E 的验证细节放入
附录 B。tail fence / atom-level 控制不作为独立缺口列出，而作为 D 的诊断变量处理。

### 3.2 缺口 B/C：In-register Fragment Reinterpret 与跨迭代 Layout Contract

TileLang 目前缺少 FA3 FP8 mainloop 所需的寄存器级 fragment reinterpret 能力。具体
表现为：无法以高层 API 表达 `permute_Cregs_fp8 + convert_layout_acc_Aregs +
convert_type_out` 这一组操作。

该缺口包含两个层次：

1. **单次转换层次**：将 QK accumulator C layout 的 `tSrS` 解释为 PV RS-A operand
   layout，并完成 FP8 dtype conversion。
2. **跨迭代契约层次**：声明该转换后的同一组物理 registers 在下一轮 mainloop 中
   作为 `tOrP` 被 PV 直接消费。

TileLang 可以表达不同 fragment layout，也可以通过 `T.Parallel` 或 `T.copy` 在两个
fragment 之间赋值。但此类赋值会依据两个 fragment 各自的 thread/index mapping
生成数据搬运，不能等价替代 FA3 对同一组物理 registers 的 CUTE layout reinterpret。

所需能力可概括为：

```text
跨 loop 的 register fragment layout / swizzle contract
```

建议的 API 方向：

```text
T.reinterpret_fragment(src, dst_layout, dst_dtype)
T.convert_fragment_layout(src, dst_fragment)
```

这些 API 需要表达寄存器片段契约，而不是普通 elementwise copy。

当前缺口导致的替代路径是：

```text
softmax 后将 P pack 到 p_smem
PV helper 从 p_smem 读取 P
```

该路径可以数值正确，但不等价于 FA3 register `tOrP` contract。

### 3.3 缺口 D：跨 QK/PV 的 FA3-style WGMMA Mainloop Contract

该问题不是 PTX injection gap。TileLang 支持 inline PTX，当前实验也可以在
source/PTX intent 层表达：

```text
issue QK[n]
issue PV[n-1]
wait_group<1>
softmax
wait_group<0>
```

缺失的是一个可组合的 DSL/lowering contract，使以下信息在完整 mainloop 中以
coherent 的形式暴露给 ptxas：

```text
QK/PV WGMMA issue
operand fence
warpgroup arrive / commit / wait
accumulator def/use lifetime
softmax 对 QK accumulator 的消费
PV operand fragment lifetime
helper boundary / TIR region boundary
```

FA3 CUTE/C++ mainloop 将这些对象置于同一个强类型 CUTE scope 中。TileLang 当前实现
中，即使使用 inline PTX，完整上下文仍可能分散在 TensorIR、`T.call_extern` helper、
inline PTX、TileLang fragment buffer 和 C++ helper raw pointer/register view 中。
这些边界可能使 ptxas 无法证明它们属于同一个 FA3-style WGMMA group contract。

DeepWiki 的解释进一步收窄了问题边界：单个 `T.gemm` 已经会形成 grouped WGMMA
结构，即一次 warpgroup arrive、若干 WGMMA atom、一次 commit batch。当前缺口不是
单个 GEMM 的 grouping，而是跨 `QK[n]` 与 `PV[n-1]` 两个 GEMM 的 shared scoreboard
window；TileLang 当前没有 `T.gemm_overlap`、`T.wgmma_group_begin/end` 这类能够跨
两个 `T.gemm` 调用维持同一 scoreboard window 的高层 API。

当前观察结果：

```text
source/PTX intent 接近 FA3
correctness 通过
SASS 仍可能退化为 per-QGMMA WARPGROUP.ARRIVE / DEPBAR
```

因此，该缺口应定义为：

```text
DSL expressiveness gap / lowering contract gap
```

需要 TileLang 提供或澄清的能力：

1. `T.gemm(..., wg_wait=-1)` 是否足以表达 full-context FA3 grouped overlap。
2. 是否需要 `T.gemm_overlap(qk, pv, inter_wait=1)` 这样的高层 API。
3. 是否需要公开 atom-level WGMMA recipe，说明 issue/fence/commit/wait 与
   fragment lifetime 的合法组合方式。
4. TileLang 如何保持 QK/PV WGMMA groups、accumulator def/use、operand fences、
   waits、softmax consumption 和 PV fragment lifetime 在同一个主循环契约中
   对 ptxas 可见。

在当前能力边界样例中，`plain-wait` 路径试图表达 FA3 steady-state 的
`QK[n] + PV[n-1] + wait_group<1> + softmax + wait_group<0>`。性能结果表明该
source schedule 没有转化为 FA3-style SASS grouping。因此，后续需要对 WGMMA issue、
operand fence、commit、wait 与 helper boundary 做 A/B。tail fence 控制属于该 A/B
集合中的一个诊断变量，而不是本报告已确认的独立根因。

DeepWiki 将 `wgmma_rs()` / `body_rs()` 中 commit 后无条件执行的 operand fence
识别为导致 scoreboard window 断裂的可能机制。该判断与本报告的 feature request
方向一致：需要 `no_tail_fence` 或 atom-level API 做最小差异 A/B，以确认该机制在
当前能力边界样例中是否是直接触发因素。

相关调试能力需求包括：

```text
T.gemm(..., no_tail_fence=True)
T.wgmma_group_begin/end
documented atom-level WGMMA recipe
```

### 3.4 汇总

| TileLang 功能点 | 当前判断 | 对 kernel 的影响 |
| --- | --- | --- |
| loop 外 `T.alloc_fragment` 生命周期 | 已支持或理论上支持 | 可表达 persistent fragment lifetime，不是主要阻塞项 |
| QK C-layout -> PV RS-A in-register reinterpret | 缺失 | 可实现 `p_smem` 替代路径或 fragment-P probe，但无法证明为 FA3-equivalent `tOrP` |
| grouped WGMMA mainloop contract | 单个 `T.gemm` grouped WGMMA 已支持；跨 QK/PV scoreboard window 缺失 | source/PTX intent 可接近 FA3，SASS 仍可能 per-QGMMA scoreboard |
| `wg_wait=-1` / tail fence / atom-level 控制 | 作为 grouped WGMMA contract 的诊断变量 | 需要 A/B 和文档化 atom recipe；不是已证实独立根因 |
| exact CUTE smem/TMA layout | 需要独立验证 | 当前样例中作为次级问题；验证细节见附录 B |

## 4. 能力边界样例状态

本报告将当前实验 kernel 统一称为“能力边界样例”（boundary demo）。该名称表示：
该 kernel 数值正确，source schedule 接近 FA3 steady-state，但仍保留 TileLang
高级 API / lowering contract 无法完整表达的部分，用于展示能力边界而不是作为最终
高性能实现。

当前能力边界样例使用如下 kernel：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel
```

该 kernel 的目的不是提供最终高性能实现，而是构造一个数值正确、source schedule
接近 FA3、但仍暴露 TileLang 能力边界的样例。

### 4.1 Correctness 与最小测试形状

最小 correctness 测试形状：

```text
batch=1, seq_len=896, heads=2, heads_kv=1, dim=128
```

选择 `seq_len=896` 的原因：

```text
BN224 path requires seq_len % 224 == 0
WS parent requires seq_len % 128 == 0
lcm(224, 128) == 896
```

Correctness gate：

```text
python3 -m pytest experiments/ws_kernel_evolution/fa3_tilelang_gap/test_boundary_demo_kernel.py -s -q

1 passed
```

新目录全量测试：

```text
python3 -m pytest experiments/ws_kernel_evolution/fa3_tilelang_gap -q

8 passed
```

### 4.2 已对齐的结构

当前能力边界样例在 TileLang source / generated CUDA 的结构意图上对齐了 FA3。
本报告不声明其 PTX 层已经完整对齐；PTX 层仍需通过 dump 和 checker 固化证据。
SASS 层则已通过性能结果和历史统计显示 grouped scoreboard 未能稳定保持。

已对齐的 source/CUDA-level 结构包括：

| 项目 | 当前状态 |
| --- | --- |
| correctness | 对 serial TMA-V baseline 通过 correctness test |
| QK/PV FIFO order | 表达 `QK[n] -> PV[n-1] -> wait_group<1> -> softmax -> wait_group<0>` |
| scale contract | 使用 FP8 Q/K/V，并采用 TileOps expanded scale |
| P handoff intent | 当前通过 `p_smem` 保存上一轮 P，下一轮 PV helper 消费 |
| 能力边界样例价值 | 展示 source intent 接近 FA3 但 SASS/性能未达到 FA3 |

建议按层级表述当前状态：

```text
source/generated CUDA layer:
  已表达 FA3-like QK/PV FIFO schedule 与 P handoff intent。

PTX layer:
  仍需 dump generated PTX，检查 WGMMA issue / wait_group / helper call
  是否保持预期顺序。

SASS layer:
  grouped WGMMA scoreboard 未稳定保持，这是当前性能差距的主要观察点。
```

### 4.3 未对齐的结构

当前能力边界样例尚未满足以下三项 FA3-equivalent 要求：

1. 未建立 FA3-equivalent `tOrP` 寄存器布局契约。现有 fragment-P probe
   仍需要 `T.copy` 或 helper conversion，不能证明 QK accumulator 与 PV RS-A
   operand 共享同一组物理 registers。
2. 未表达 QK accumulator -> PV RS-A 的 zero-cost reinterpret。当前通过
   `p_smem` 替代路径规避了 TileLang 缺失的 in-register fragment reinterpret API。
3. 未证明 exact FA3 CUTE shared layout。该项是次级问题，本报告不展开。

补充说明：当前能力边界样例使用 `p_smem` 保存 P，而不是使用 loop-external
fragment-P。这不是一个独立的 TileLang 能力缺口，而是当前样例的隔离性取舍。repo
中已有 `p_pv_frag_1/2` probe，说明 loop-external P fragment lifetime 可以表达；
但 fragment-P probe 只能证明 P fragment 可以跨迭代存活，不能证明它已经是 PV 可直接
消费的 FA3-equivalent RS-A register operand。因此，当前样例保留 `p_smem`，避免将
“P 存储介质从 shared memory 改为 fragment”和“FA3 `tOrP` 寄存器布局契约
是否成立”混在同一个样例中。

当前 `p_smem` 替代路径的具体形式如下：

```python
p_smem_1 = T.alloc_shared([half_m, block_n], fp8_dtype)
p_smem_2 = T.alloc_shared([half_m, block_n], fp8_dtype)
```

能力边界样例在 softmax 后用 helper 将 P 写入 shared memory，下一轮 PV helper 再从
shared memory 读取 P。该路径可以用于获得数值正确的实现，但不等价于 FA3 的 register
`tOrP` contract。

因此，该 kernel 的定位是：

```text
数值正确的能力边界样例；
不是完整复现 FA3 FP8 寄存器片段主循环契约的 kernel。
```

### 4.4 性能结果

#### 4.4.1 Serial Baseline 定义

`TMA-V serial baseline` 指：

```text
GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel
```

该 kernel 不是朴素串行实现。它已经实现了 FA3 pipeline 的一部分：

```text
producer / consumer warpgroup ping-pong: yes
V TMA staging: yes
V layout transform: FA3-style CUTE helper
PV: extern PTX/CUTE helper
consumer 内部 QK -> softmax/P conversion -> PV overlap: no
```

称其为 serial baseline，是因为同一 consumer warpgroup 内部仍按阶段边界推进：

```text
QK[n]
wait QK[n]
softmax[n] / prepare P[n]
PV[n]
update O/LSE
```

它没有实现 FA3 steady-state 中的 consumer 内部 overlap：

```text
QK[n] + PV[n-1] + wait_group<1>
softmax[n] / prepare P[n]
wait_group<0>
```

该 baseline 的作用是提供“已知正确、局部优化较充分”的性能参照。能力边界样例
则用于展示：当 source schedule 进一步对齐 FA3 consumer 内部 overlap
时，TileLang/ptxas contract 如何成为限制因素。

#### 4.4.2 4k 规模实测

Nightly docker 环境：

```text
image: tileops-runner-sshd:nightly-tl019-fullstack-no-tileops-ldfix-registered-tmux
device: NVIDIA H200
python: 3.12.13
torch: 2.10.0+cu128
```

测试形状：

```text
batch=1, seq_len=3584, heads=32, heads_kv=8, dim=128
```

说明：严格 `seq_len=4096` 不满足当前 BN224 TileOps kernels 的
`seq_len % 224 == 0` 要求。这里使用 `3584 = lcm(224, 128) * 4`，同时保持 BN224
tiling 与 FA3 scale-block alignment。

运行命令：

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/ga/TileOPs:/workspace/TileOPs_live \
  -w /workspace/TileOPs_live \
  tileops-runner-sshd:nightly-tl019-fullstack-no-tileops-ldfix-registered-tmux \
  bash -lc 'IMPLS="tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v,\
tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_streaming_p_plain_wait,fa3"
  python3 benchmarks/ops/attention/bench_gqa_fp8.py \
    --quick \
    --seq-lens 3584 \
    --heads 32 \
    --heads-kv 8 \
    --batch 1 \
    --impls "$IMPLS" \
    --scale-mode fa3'
```

结果：

| impl | latency_ms | TFLOP/s | 相对 FA3 |
| --- | ---: | ---: | ---: |
| FA3 | 0.382061 | 550.84 | 1.00x |
| TMA-V serial baseline | 0.569670 | 369.43 | 0.67x |
| 能力边界样例：streaming-P plain-wait | 0.932893 | 225.59 | 0.41x |

结论：

```text
能力边界样例数值正确，但吞吐约为 FA3 的 41%。
serial TMA-V baseline 吞吐约为 FA3 的 67%。
source/PTX intent 接近 FA3 并不自动转化为 FA3-style grouped WGMMA SASS。
```

## 5. TileLang 高级 API Feature Requests

目标是使 FP8 GQA forward kernel 能够使用 TileLang 高级 API 表达 FA3-equivalent
mainloop，而不依赖 `p_smem` 替代路径、手写 C++ helper 边界或无法由 DSL 描述的
register reinterpret 约定。

### 5.1 Fragment Reinterpret / Layout Conversion API

**需求**：提供寄存器级 fragment reinterpret / layout conversion API。

候选接口形式：

```text
T.reinterpret_fragment(src, dst_layout, dst_dtype)
T.convert_fragment_layout(src, dst_fragment)
```

**需要表达的语义**：

```text
QK accumulator C-layout registers
-> FP8 C-register permutation
-> PV RS-A operand layout reinterpret
-> dtype conversion to FP8 P fragment
```

该 API 需要 lower 为 FA3 等价的
`permute_Cregs_fp8 + convert_layout_acc_Aregs + convert_type_out`，而不是普通
elementwise copy、register shuffle 或 shared-memory 中转。

**验收标准**：

```text
1. TileLang source 中可以直接表达 tSrS -> tOrP。
2. 生成代码中不需要 p_smem 作为 P handoff。
3. SASS 中不出现由 fragment copy 引入的额外 shared-memory round trip。
```

### 5.2 Cross-loop Register Fragment Layout Contract

**需求**：提供跨迭代 register fragment layout / swizzle contract。

**需要表达的语义**：

```text
loop n:
  QK produces tSrS
  softmax updates tSrS
  tSrS is reinterpreted as tOrP

loop n + 1:
  PV consumes the same logical tOrP as RS-A operand
```

该能力应允许 TileLang 声明：上一轮 QK C accumulator 的同一组物理 registers，在
FP8 swizzle/layout reinterpret 后，就是下一轮 PV 的 RS-A operand。

**验收标准**：

```text
1. loop-external P fragment 不仅能存活，而且具有明确的 PV RS-A layout contract。
2. PV GEMM 可以直接消费该 fragment，而不是通过 helper raw pointer 约定隐式解释。
3. lowering/ptxas 能看到稳定的 accumulator/operand def-use 边界。
```

### 5.3 FA3-style Grouped WGMMA Mainloop API

**需求**：提供可组合的 grouped WGMMA mainloop API，或正式文档化 atom-level recipe。

候选接口形式：

```text
T.gemm_overlap(qk_gemm, pv_gemm, inter_wait=1)
T.wgmma_group_begin()
T.wgmma_group_commit()
T.wgmma_group_wait(n)
T.wgmma_group_end()
```

**需要表达的语义**：

```text
issue QK[n]
issue PV[n-1]
wait_group<1>
softmax / prepare P[n]
wait_group<0>
```

该 API 需要将 WGMMA issue、operand fence、commit、wait、accumulator lifetime、
softmax consumption、PV operand lifetime 和 helper boundary 作为同一个 coherent
mainloop contract 暴露给 lowering/ptxas。

**验收标准**：

```text
1. TileLang source 能表达 FA3 steady-state QK/PV overlap。
2. PTX 中 WGMMA issue / wait 顺序符合 source intent。
3. SASS 中 QK/PV 保持 grouped scoreboard，不退化为 per-QGMMA DEPBAR/ARRIVE。
```

### 5.4 Atom-level A/B Controls for WGMMA Diagnostics

**需求**：提供用于定位 grouped scoreboard 退化原因的 atom-level A/B 控制。

候选接口形式：

```text
T.gemm(..., no_tail_fence=True)
T.wgmma_fence_operand(...)
T.wgmma_commit(...)
T.wgmma_wait(n)
```

**用途**：隔离以下因素对 ptxas scoreboard 分组的影响：

```text
operand fence placement
commit / wait placement
helper boundary
TIR region boundary
accumulator read/write lifetime
PV operand fragment lifetime
```

**验收标准**：

```text
1. 可以构造只改变 fence/wait 边界的 A/B kernel。
2. 可以将 source/PTX 差异与 SASS QGMMA/DEPBAR/ARRIVE 差异对应起来。
3. 该能力用于诊断 grouped WGMMA contract，不要求作为最终用户主 API。
```

### 5.5 CUTE-compatible Shared Layout / TMA Destination API

**需求**：提供 CUTE-compatible shared layout / TMA destination 的显式表达或验证工具。

**用途**：

```text
1. 验证 TileLang TMA destination 与 FA3 SmemLayoutK 逐元素一致。
2. 验证 V transpose / VtMma layout 与 FA3 PV consumer layout 逐元素一致。
3. 避免仅依赖 swizzle flag 判断 layout 等价。
```

**验收标准**：

```text
1. TMA producer layout 与 WGMMA consumer layout 可由同一 layout object 或显式契约绑定。
2. 提供 TileLang layout 与 CUTE layout 的 mapping checker。
3. 对 FP8 K-major 与 VtMma layout 给出可重复的等价性验证方法。
```

## 6. 后续 Boundary Sample 升级条件

若要将当前能力边界样例升级为完整 FA3 FP8 contract 样例，需要同时满足以下条件：

```text
loop-external persistent P fragment
+ QK accumulator C layout -> PV RS-A layout zero-cost reinterpret
+ prove that the resulting P fragment is FA3-equivalent tOrP
+ same QK -> PV -> wait<1> -> softmax -> wait<0> order
+ correctness pass
+ SASS QGMMA/DEPBAR/ARRIVE statistics
```

其中第二项是当前 TileLang 缺失的核心能力，因此更适合作为明确的 DSL/lowering
feature request，而不是继续依赖替代路径弱化问题边界。

## 附录 A：DeepWiki 对 TileLang 现有能力的解释

本附录记录 DeepWiki 对 TileLang 当前能力的解释，用作正文第 3 节判断的依据。正文
不依赖这些解释的逐字表述，只使用其归纳结论。

### A.1 Loop-external Fragment Lifetime

DeepWiki 解释指出：

```text
T.alloc_fragment 分配的 local.fragment buffer 生命周期绑定 T.Kernel scope。
MultiVersionBufferRewriter 只对 shared memory buffer 做 pipeline staging。
fragment buffer 不受影响。
```

由此得到的结论是：

```text
loop 外分配 fragment 并跨迭代存活可以表达。
persistent P fragment lifetime 不是当前主要缺口。
```

该结论只覆盖 fragment lifetime，不覆盖 FA3-equivalent `tOrP` 寄存器布局契约。

### A.2 In-register Layout Reinterpret

DeepWiki 与本地检索均显示，TileLang 当前没有以下 FA3 FP8 helper 的高层等价 API：

```text
permute_Cregs
convert_layout_acc
convert_type_out
reinterpret_fragment
```

`T.Parallel` 或 `T.copy` 可以在不同 fragment 之间赋值，但赋值会按照各自 fragment
layout 的 thread/index mapping 生成数据搬运，不能表达 FA3 对同一组物理 registers
的 zero-cost CUTE layout reinterpret。

对应正文结论：

```text
缺口 B/C:
  in-register fragment reinterpret / layout conversion
  cross-loop register fragment layout / swizzle contract
```

### A.3 WGMMA Wait 与 Tail Fence

DeepWiki 解释指出，`wg_wait=-1` 可以跳过 wait，但部分 WGMMA lowering path 可能在
commit 后仍执行 operand fence。该行为可能影响 ptxas 的 scoreboard 分组。

本报告不将 tail fence 写成已证实根因，原因是：

```text
FA3 flash::gemm 本身也包含 operand fence。
当前观察到的是 full-context grouping 未稳定保持。
tail fence 只是需要 A/B 的诊断变量之一。
```

对应正文结论：

```text
需要 atom-level WGMMA recipe 或可控 A/B API，
以隔离 issue/fence/commit/wait/helper boundary 对 SASS grouping 的影响。
```

## 附录 B：TMA Destination 与 CUTE Layout 验证问题

本附录记录 TMA destination / shared layout 的验证问题。该问题影响完整
FA3-equivalent kernel 的最终证明，但不是当前能力边界样例的主要展示点。

### B.1 K-major FP8 Shared Layout

DeepWiki 解释指出，TileLang TMA lowering 可以根据 TIR layout 推导 TMA swizzle flag。
对于标准 FP8 K-major layout，当连续维度满足 128B swizzle 条件时，TileLang 与 FA3
可能都会选择 128B swizzle。

需要注意的是：

```text
swizzle flag 一致不等于逐元素 CUTE layout 一致。
```

FA3 的 K shared layout 由 CUTE/CUTLASS layout selector 构造；TileLang 的 TMA 写入端
和 WGMMA 消费端可能通过两套独立路径构造 layout。对于标准 K-major FP8 情况，两者
经验上可能一致，但报告中不将其作为已证明事实。

### B.2 VtMma / V Transpose Layout

FA3 FP8 PV 对 V 的 MMA consumer layout 更敏感。FA3 中 V transpose / VtMma layout
可能通过 CUTE `tile_to_shape(..., Step<_2, _1>{})` 等方式构造。TileLang 中对应的
MN-major swizzled layout 即使选择相同 swizzle flag，也仍需证明逐元素 mapping 与
FA3 一致。

当前报告不把该问题作为 boundary sample 的主缺口，原因是当前性能差距和展示目标主要
来自：

```text
1. tSrS -> tOrP register contract 缺失；
2. grouped WGMMA mainloop contract 未稳定保持。
```

### B.3 后续验证需求

完整 FA3-equivalent kernel 仍需要以下验证工具或 API：

```text
1. CUTE-compatible TMA destination API；
2. TileLang TIR layout 与 CUTE layout 的逐元素 mapping checker；
3. V transpose / VtMma layout 的 SASS 与数值双重验证；
4. K/V shared layout producer 与 WGMMA consumer layout 的统一来源或显式绑定。
```

## 附录 C：DeepWiki 查询原文与回复原文

### C.1 查询原文

```text
I am trying to implement an FP8 GQA forward attention kernel in TileLang that matches the FlashAttention-3 FP8 mainloop contract on Hopper.

TileLang already appears to provide basic primitives such as:
- T.alloc_fragment for local/register fragments
- T.copy / T.Parallel for data movement and elementwise assignment
- T.gemm / T.wgmma_gemm for WGMMA
- T.wait_wgmma or wg_wait for warpgroup wait control
- layout annotation / swizzled shared-memory layouts
- TMA copy into shared memory
- inline extern/PTX helper calls

The question is whether current TileLang has higher-level APIs or lowering contracts equivalent to the following FA3 FP8 requirements.

1. In-register fragment reinterpret / layout conversion

FA3 FP8 prepares the PV A operand from the QK accumulator roughly like:

  softmax(tSrS);
  permute_Cregs_fp8(tSrS);
  tOrP_acc = make_tensor(
      tSrS.data(),
      convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
  tOrP = make_tensor_like<Element>(tOrP_acc);
  convert_type_out(tOrP_acc, tOrP);

Does TileLang currently provide a high-level API equivalent to:
- permute_Cregs_fp8
- convert_layout_acc_Aregs
- convert_type_out
- reinterpret_fragment / convert_fragment_layout

Specifically, can TileLang express that the same physical registers of a WGMMA C accumulator are reinterpreted as a PV RS-A operand layout, without generating an elementwise copy, register shuffle, or shared-memory round trip?

2. Cross-loop register fragment layout contract

FA3 keeps tOrP as a persistent CUTE register tensor across mainloop iterations:

  loop n:
    QK produces tSrS
    softmax updates tSrS
    tSrS is converted/reinterpreted into tOrP

  loop n + 1:
    PV consumes the previous tOrP directly as its RS-A operand

TileLang can allocate a loop-external T.alloc_fragment, but does it have a way to declare that this persistent fragment has the exact PV RS-A register layout contract and is derived from the previous QK C accumulator by zero-cost reinterpret?

Or does T.copy / T.Parallel between fragments necessarily lower through each fragment’s own thread/index mapping and therefore fail to represent this FA3-style same-register reinterpret?

3. FA3-style grouped WGMMA mainloop contract

FA3 steady state issues:

  QK[n]
  PV[n-1]
  wait_group<1>
  softmax / prepare P[n]
  wait_group<0>

The issue is not whether TileLang can emit PTX or call T.wgmma_gemm. The question is whether TileLang has a composable high-level API or documented atom-level recipe that keeps the following visible to lowering/ptxas as one coherent WGMMA group contract:
- QK/PV WGMMA issue
- operand fences
- warpgroup arrive / commit / wait
- accumulator def/use lifetime
- softmax reading QK accumulator
- PV consuming the previous P fragment
- helper/TIR region boundaries

Is T.gemm(..., wg_wait=-1) or T.wgmma_gemm + T.wait_wgmma sufficient to preserve FA3-style grouped scoreboard in SASS, or is there currently no TileLang API equivalent to something like T.gemm_overlap(qk, pv, inter_wait=1) / wgmma_group_begin/end?

4. Atom-level A/B controls for WGMMA diagnostics

Does TileLang expose controls to separately test:
- no tail operand fence after commit
- explicit operand fence placement
- explicit commit/wait placement
- grouping multiple WGMMA operations in one batch
- preventing helper/TIR boundaries from breaking the WGMMA scoreboard window

For example, is there an API equivalent to:
  T.gemm(..., no_tail_fence=True)
  T.wgmma_group_begin/end
  T.wgmma_commit
  T.wgmma_wait
  T.wgmma_fence_operand

5. CUTE-compatible shared layout / TMA destination

Does TileLang provide a way to explicitly bind the TMA destination layout to the same CUTE/CUTLASS shared-memory layout used by WGMMA consumption?

In particular, for FP8 FA3:
- K shared layout should match FA3 SmemLayoutK
- V transpose / VtMma layout should match FA3 PV consumer layout

Is swizzle flag inference enough, or is there a checker/API that proves the TileLang TIR layout is elementwise equivalent to the FA3 CUTE layout?

Please answer by pointing to existing TileLang APIs/source files if these capabilities already exist. If they do not exist, please identify which ones are current API/lowering gaps.
```

### C.2 DeepWiki 回复原文

```text
1. In-register fragment reinterpret / layout conversion
Not supported. Confirmed gap.

A codebase-wide search finds zero occurrences of permute_Cregs, convert_layout_acc, convert_type_out, reinterpret_fragment, or convert_fragment anywhere in TileLang.

T.reinterpret exists but operates only on scalar PrimExpr values: op.py:2198-2222

For two T.alloc_fragment buffers with different Fragment layouts, any assignment via T.Parallel lowers through each buffer's own forward_thread_fn / forward_index_fn. If the WGMMA-C layout (make_mma_store_layout) and the WGMMA-RS-A layout (make_mma_load_layout) have different thread→slot mappings — which they do — the lowered code will contain register shuffles or shared-memory round-trips. There is no TIR node that expresses "these two fragment buffers share the same physical registers."

FA3's permute_Cregs_fp8 + convert_layout_acc_Aregs + convert_type_out is zero-cost precisely because it exploits the FP8-specific physical register coincidence between WGMMA-C and WGMMA-RS-A after permutation. TileLang has no equivalent primitive.

2. Cross-loop register fragment layout contract
Persistent allocation: yes. Zero-cost reinterpret contract: no.

T.alloc_fragment allocates into local.fragment scope, which is bound to the T.Kernel block, not to any inner loop scope. MultiVersionBufferRewriter only multi-versions shared-memory buffers for pipeline staging; fragment buffers are unaffected. A tOrP-equivalent buffer allocated outside T.Pipelined will persist across iterations. allocate.py:66-77

However, TileLang has no mechanism to declare that this persistent fragment carries the PV RS-A layout contract and is derived from the QK C accumulator by zero-cost reinterpret. The layout annotation (T.annotate_layout) is a static compile-time hint; it does not express a runtime derivation relationship between two fragment buffers. Any T.Parallel copy between them will lower through each buffer's own layout mapping, as described in §1.

3. FA3-style grouped WGMMA mainloop contract
Single-gemm grouped WGMMA: yes. Cross-gemm scoreboard window: no high-level API.

Each T.gemm call already emits a single grouped WGMMA: one warpgroup_arrive, N atom instructions, one warpgroup_commit_batch: wgmma_macro_generator.py:222-241

wg_wait=-1 (any negative value) skips the warpgroup_wait call:

if wg_wait >= 0:  
    self.wgmma_wait(wg_wait)
wgmma_macro_generator.py:237-239

The blocking issue is the unconditional tail fence. After commit, wgmma_rs() always emits wgmma_fence_c(C_buf) + wgmma_fence_a(A_buf) regardless of wg_wait: wgmma_macro_generator.py:273-278

The C++ template body_rs() mirrors this — warpgroup_fence_operand(acc) and warpgroup_fence_operand(tCrA) are unconditional: gemm_sm90.h:138-144

These tail fences "release" tOrP and acc_o immediately after PV commit. When the next iteration writes tOrP (QK→PV conversion), ptxas sees a new acquire fence and inserts a DEPBAR between the two WGMMA groups, collapsing the FA3 scoreboard window.

There is no T.gemm_overlap, T.wgmma_group_begin/end, or any composable API that spans two T.gemm calls with a shared scoreboard window. The FA3 sequence QK[n] → PV[n-1] → wait_group<1> → softmax → wait_group<0> cannot be expressed at the T.gemm level.

4. Atom-level A/B controls for WGMMA diagnostics
Most atom-level primitives exist; no_tail_fence and group boundary markers do not.

Available in tilelang/language/builtin.py: builtin.py:531-559

T.warpgroup_arrive() ✓
T.warpgroup_commit_batch() ✓
T.warpgroup_wait(n) ✓
T.warpgroup_fence_operand(buf, num_regs=...) ✓ (6 call sites in builtin.py)
Available in tilelang/language/tir/op.py:

T.ptx_wgmma_ss(...) ✓
T.ptx_wgmma_rs(...) ✓
Not available:

T.gemm(..., no_tail_fence=True) — grep confirms zero matches for no_tail_fence or tail_fence anywhere in the codebase
T.wgmma_group_begin / T.wgmma_group_end — does not exist
Any TIR annotation that prevents helper/region boundaries from breaking the scoreboard window
The atom-level path (T.warpgroup_fence_operand + T.warpgroup_arrive + T.ptx_wgmma_ss/rs + T.warpgroup_commit_batch + T.warpgroup_wait) is the only way to manually construct the FA3 scoreboard sequence, but it requires manually computing WGMMA descriptor offsets (the logic currently lives inside wgmma_ss_atom() / wgmma_rs_atom() and is not separately exported): wgmma_macro_generator.py:510-530

5. CUTE-compatible shared layout / TMA destination
Swizzle flag alignment only. No elementwise equivalence proof or checker.

TileLang's TMA lowering selects CU_TENSOR_MAP_SWIZZLE_* by StructuralEqual-comparing the TIR layout object against a fixed set of known layouts (MakeFullBankSwizzleLayout2D → 128B, MakeHalfBankSwizzleLayout2D → 64B, MakeQuarterBankSwizzleLayout2D → 32B): atomic_add.cc:358-384

For FP8 with mat_continuous % 128 == 0, makeWgmmaSwizzledLayout produces MakeFullBankSwizzleLayout2D → CU_TENSOR_MAP_SWIZZLE_128B, which matches FA3's Swizzle<3,4,3> at the swizzle-flag level. gemm_layouts.cc:943-952

However, the C++ WGMMA consumer (GemmTensorOp::body()) independently reconstructs SmemLayoutA/B using CUTE's ss_smem_selector, not the TIR layout: gemm_sm90.h:37-47

The TMA write path (TIR layer) and the WGMMA consume path (C++ CUTE layer) are aligned only through the swizzle flag, not through a shared layout object. There is no API or static checker that proves the TIR layout is elementwise equivalent to the FA3 CUTE layout.

For SmemLayoutVtMma specifically — FA3 constructs it as tile_to_shape(ss_smem_selector<GMMA::Major::K, fp8, ...>(), ..., Step<_2,_1>{}) — TileLang's make_wgmma_swizzled_layout(V_shared, k_major=False) produces an MN-major layout with the same swizzle flag, but whether the tile-internal element ordering matches FA3's Step<_2,_1> transposition is not verified at the TIR level and would need SASS-level validation.

Summary table
Capability	Status	Gap
Persistent tOrP across loop iterations	Supported	—
permute_Cregs_fp8 / convert_layout_acc_Aregs / convert_type_out	Not supported	Need T.reinterpret_fragment(src, dst_layout, dst_dtype)
Zero-cost in-register layout reinterpret between two fragment buffers	Not supported	Same as above
Single T.gemm as grouped WGMMA (arrive+atoms+commit)	Supported	—
wg_wait=-1 to skip wait	Supported	—
Suppress tail warpgroup_fence_operand after commit	Not supported	Need no_tail_fence=True parameter or separate wgmma_rs_no_fence()
Cross-gemm scoreboard window (FA3 QK→PV→wait<1> pattern)	Not supported	Need T.gemm_overlap(qk, pv, inter_wait=1) or wgmma_group_begin/end
Atom-level arrive/commit/wait/fence_operand/ptx_wgmma_ss/rs	Supported	Descriptor offset computation not exported separately
TMA → swizzled smem (swizzle flag)	Supported	—
TMA → smem elementwise equivalent to FA3 CUTE layout	Not verified	No checker; SmemLayoutVtMma tile structure unconfirmed
```
