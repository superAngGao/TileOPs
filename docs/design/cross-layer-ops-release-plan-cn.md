# Cross-Layer 算子族发布计划

## 1. 计划定位

这份文档用于讨论 TileOps 是否需要引入 `cross_layer` 算子族，以及第一批工作如何落地。它不是 manifest spec，也不冻结任何新 op 的公开 API；真正的 op contract 仍以后续 tracking issue 和 manifest PR 为准。

这里的立场是：`cross_layer` 值得建立，但它应该是一个窄边界的 operator family，而不是所有“跨层架构机制”的集合。第一批工作聚焦两件事：

1. 对齐 TileOps 已有的 MHC / Hyper-Connection taxonomy。本文倾向将它归入 `cross_layer.connection`；如果既有 missing-ops / manifest 计划已先把 MHC 放入 `sequence_modeling`，第一步应先在 tracking issue 中完成 family reconciliation，避免重复 manifest 条目。
2. 为 Kimi Attention Residuals，尤其是 Block AttnRes，建立清晰的后续落地路径。

## 2. 为什么需要 Cross-Layer

TileOps 现有 family 并不是单纯按底层计算 primitive 分类。`elementwise`、`reduction`、`scan` 更接近计算方法；`attention`、`moe`、`normalization` 则按模型中稳定的算子语义分类。

近年的模型结构开始显式使用 depth / layer / block / expanded-residual 方向的数据流：

```text
多个 layer / block / residual-channel states
    -> 加权、选择、路由、注意力聚合或仿射混合
    -> 当前层继续使用的表示或状态
```

这些操作不适合简单归入 `reduction`。底层可能有 reduction、softmax、matmul 或 scatter/gather，但用户关心的不是“沿某个 tensor 轴求和”，而是模型深度方向的 state mixing / routing / aggregation。

它们也不适合全部归入 `attention`。以 AttnRes 为例，它确实使用 attention-like weights，但 attention axis 是 layer/block depth，不是 sequence token。

## 3. Family 边界

`cross_layer` 的核心不是“输入来自上一层”。普通 attention、MLP、RMSNorm 也都接收上一层 hidden state；如果按这个标准，几乎所有 Transformer 子层都会变成 cross-layer。

本计划采用更严格的边界：

```text
cross_layer operator:
  算子签名中显式包含多个 layer / block / residual-channel / source states，
  并且算子本身负责沿 depth / layer / expanded-residual / source axis
  做 combine、routing、selection 或 transform。
```

进入 TileOps manifest 的 `cross_layer` op 还需要满足一个工程条件：

```text
它必须能形成稳定 tensor signature、独立 correctness reference、
独立 benchmark workload，并具有明确的 kernel boundary。
```

### 3.1 功能子类与算子边界

`cross_layer` 不应该只对应一种数学 primitive。更合理的组织方式是先按功能语义拆成几个子类，再判断每个子类是否已经有足够清晰的 TileOps kernel boundary。

| 子类 | 代表模型 / 工作 | 模型里的功能 | TileOps 中可复用的算子边界 |
| --- | --- | --- | --- |
| `cross_layer.connection` | MHC / Hyper-Connection | 扩展 residual stream，并学习跨 residual-channel 的 mixing / projection | `MHCPreFwdOp`、`MHCPostFwdOp` 这类 expanded-residual mixing kernel；现有 wrapper 名称可通过 alias 兼容 |
| `cross_layer.aggregation` | Kimi Attention Residuals / Block AttnRes | 显式读取多个前序 layer/block states，并沿 depth axis 做 attention-style aggregation | `BlockAttnResFwdOp`，以及内部 weighted-sum reference |
| `cross_layer.scheduling` | Mixture-of-Depths, LayerSkip / self-speculative decoding, early-exit routing | 在模型深度方向决定哪些 token/layer/block 继续计算，或者在较浅层提前退出 | token-depth routing、layer execution mask、gather/scatter/compaction；更多是 runtime scheduling + data movement，第一批不直接做 manifest |
| `cross_layer.cache_sharing` | CLA, LCKV, YOCO-style KV/state reuse | 跨层共享、选择、复用或重排 KV/cache/state | `LayerKVSelect`、`LayerKVGather`、`CrossLayerCacheRemap`、`SharedKVRead/Write`、可选 `CrossLayerKVMix`；纯指针 alias 仍属于 runtime policy |

这里的分类回答两个不同问题：

1. 现代模型是否已经使用这种 cross-layer 机制。
2. TileOps 是否已经能抽象出稳定、可测试、可 benchmark 的算子边界。

例如 Mixture-of-Depths 和 LayerSkip 明确属于 `cross_layer.scheduling`，因为它们的核心就是沿模型深度方向调度 token 或提前退出；但它们第一阶段更像 runtime policy、token compaction、execution mask 和 graph integration，不适合直接写成一个泛化 `DepthRoutingOp`。CLA、LCKV、YOCO-style 机制也属于 cross-layer 版图，但如果实现只是“当前层读另一个层的 KV 指针”，那不是 TileOps kernel；只有发生 layout movement、packing、projection、numerical mixing 或 shared cache read/write 时，才形成明确的算子目标。

### 3.2 Core Operator Candidates

第一批 release target 保持克制，只放已经有清楚 kernel boundary 的部分：

| 类别 | 代表 | 为什么进入第一批 |
| --- | --- | --- |
| Expanded residual-channel mixing | MHC / Hyper-Connection | TileOps 已有 kernel、tests、benchmarks；缺的是 manifest / taxonomy alignment |
| Depth-wise residual aggregation | Kimi AttnRes / Block AttnRes | 现代模型已经显式使用；算子边界可以收敛为 RMSNorm + projection + depth softmax + weighted aggregation |

### 3.3 Adjacent / Future Tracks

这些方向属于 `cross_layer` 版图，但第一版不作为 operator manifest 目标：

| 方向 | 代表 | 为什么先不放入 manifest |
| --- | --- | --- |
| Cross-layer scheduling | Mixture-of-Depths, LayerSkip, early-exit routing | 需要先明确 runtime execution mask、token compaction、verification / rollback 和 graph boundary |
| Cross-layer cache/state sharing | CLA, LCKV, YOCO-style KV reuse | 需要先拆清 cache select/gather/remap/read/write/mix 哪些是真 kernel，哪些只是 runtime alias |
| Cross-layer attention variants inside attention | Depth-Attention, cross-layer value mixing | 可能更适合作为 attention op 配置或后续 hybrid family；公开实现和 contract 成熟度需要单独评估 |
| Simple layer weighted aggregation | ELMo scalar mix, BERT layer pooling, sentence-transformer weighted layer pooling | 常见于表征抽取、下游 pooling 或研究工具；可以作为 reference pattern，但不是第一批 release target |

这类机制会继续作为参考，但不会直接把 `DepthRoutingOp` 或 `CrossLayerKVShareOp` 之类的宽泛抽象塞进第一版计划。后续如果出现明确的 kernel boundary，应该按更具体的名字进入，例如 `LayerKVGather`、`CrossLayerCacheRemap`、`DepthExecutionMaskPack`，而不是按模型结构命名为 `CLAOp`、`LCKVOp` 或 `YOCOOp`。

### 3.4 Out of Scope

| 类别 | 当前归类 | 不纳入原因 |
| --- | --- | --- |
| 普通 residual add | `elementwise` | 只是 `x + residual`，没有显式跨层轴 |
| FusedAddRMSNorm / FusedAddLayerNorm | `normalization` | residual add 是 norm fusion 的一部分，不做跨 layer/block 混合 |
| 普通 sequence attention | `attention` | attention axis 是 token sequence，不是 layer/block depth |
| MoE expert combine | `moe` | 聚合轴是 expert/route，不是 layer/block |
| TopK selector / attention indexing | `attention_indexing` | selection axis 是 attention candidate / index，不是模型 depth；只有 depth execution mask / token-depth routing 才进入 `cross_layer.scheduling` |
| Engram GateConv | `sequence_modeling` / model-specific fused op | 使用当前 hidden stream 和 n-gram memory，不显式聚合多个 layer/block states |

Engram 的边界尤其重要。Engram 会利用当前 hidden stream 中已经累积的前层信息，也包含 local residual path；但它的主计算是 n-gram memory lookup、gating、causal/depthwise conv 和 local residual add，并没有在算子接口中暴露多个 layer/block states 作为待混合对象。因此它是 cross-layer-adjacent，但不是 `cross_layer` family 的核心成员。

## 4. 第一批工作

### 4.1 MHC：第一个 Manifest Alignment 目标

TileOps 已经有 MHC 实现：

```text
tileops/ops/mhc.py
tileops/kernels/mhc/mhc_pre.py
tileops/kernels/mhc/mhc_post.py
tests/ops/test_mhc_pre.py
tests/ops/test_mhc_post.py
benchmarks/ops/bench_mhc_pre.py
benchmarks/ops/bench_mhc_post.py
```

MHC 的计划状态需要先和现有 missing-ops / manifest 计划对齐。若 MHC 已经在 upstream 计划或 PR 中进入 `sequence_modeling`，本计划不应再创建重复 manifest 条目；第一步是开 tracking issue 明确最终 taxonomy：

```text
preferred final taxonomy: cross_layer.connection
existing/planned taxonomy: sequence_modeling, if already landed through missing-ops work
allowed resolution:
  1. migrate or retag MHC into cross_layer.connection, if maintainers agree;
  2. keep manifest family as sequence_modeling for compatibility, but document cross-layer semantics;
  3. defer MHC manifest changes and use it only as taxonomy evidence for Block AttnRes.
```

本文倾向第一种长期归属，因为 MHC 的稳定算子语义是 expanded residual-channel mixing，而不是 sequence-axis modeling。实际 PR scope 以后续 tracking issue 结论为准。

MHCPre 的实际接口是：

```text
forward(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps)
```

参考语义：

```text
x:       [B, n_expand * C]
phi:     [n_expand * C, n_expand * n_expand + 2 * n_expand]
b:       [n_expand * n_expand + 2 * n_expand]

生成:
  h_pre: [B, n_expand]
  h_res: [B, n_expand, n_expand]  # includes Sinkhorn-style normalization

x_res   = h_res @ reshape(x, [B, n_expand, C])
x_layer = h_pre @ reshape(x, [B, n_expand, C])
```

这里的公式只用于说明 MHC 为什么属于 `cross_layer`：它显式操作 `n_expand` 这个 expanded residual-channel axis。完整的 shape contract、`phi @ x + b` 的中间拆分、`h_res` 的 Sinkhorn normalization 细节，以及对应 shape rules，会在 MHC manifest PR 中写清楚。

MHCPost 的实际接口是：

```text
forward(x_layer_out, h_post, x_res)
```

参考语义：

```text
x_out = h_post[:, :, None] @ x_layer_out[:, None, :] + x_res
```

Manifest 命名应优先遵循 TileOps 的 `*FwdOp` 约定，倾向使用 `MHCPreFwdOp` / `MHCPostFwdOp` 作为 manifest-facing 名字。现有 Python wrapper 如果已经公开为 `MHCPreOp` / `MHCPostOp`，则由 Phase 1 tracking issue 决定是否保留 alias；manifest PR 不应在没有兼容策略时顺手破坏已有 API。

### 4.2 Block AttnRes：第一个新 Op 目标

Kimi Attention Residuals 将标准 residual 的固定累加替换为对前序 layer outputs 的 learned, input-dependent softmax attention。Block AttnRes 进一步将 layer 分块，聚合 block-level representations，以降低 memory / communication 开销。

Block AttnRes 是 `cross_layer` 的第一个新 op 目标。它比泛化的 weighted-sum 更能验证这个 family 是否成立，因为它包含完整的 depth-axis operator 形态：

```text
RMSNorm over H
+ projection H -> 1
+ softmax over source/block axis
+ weighted reduction of original states
```

用于讨论的 first contract strawman 是：

```text
BlockAttnResFwdOp(
    states:     Tensor[L, M, H],
    score_weight: Tensor[H],
    rms_weight: Tensor[H],
    rms_eps:    float,
) -> output: Tensor[M, H]
```

其中：

```text
M = B * S
L = number of source blocks/states, with the current partial block included by caller
softmax axis = L
norm axis = H
score_weight is a shared learned projection vector [H], broadcast over M after RMSNorm to produce one logit per source state
output uses original states, not normalized states
logits accumulation = fp32
weighted accumulation = fp32 or explicitly documented mixed precision
```

这里不使用 `query: [H]` 这个名字，是为了避免把共享投影权重误读成 per-sample / per-token query。第一版 strawman 假设每个 `(batch, token)` 使用同一个 `[H]` scoring vector 计算 depth logits。如果 official Kimi contract 需要 batch-varying 或 token-varying query，则 tracking issue 需要在两个方案中选择：

```text
1. caller 先生成 logits: [L, M]，TileOps op 只做 depth softmax + weighted aggregation；
2. op signature 扩展为 query: [M, H] 或 query/state-dependent projection。
```

这两种方案都比含糊的 `query: [H]` 更清楚。当前 strawman 只表达第一版最小 kernel boundary：RMSNorm + shared projection + depth softmax + weighted sum。

第一版倾向使用连续 workspace：

```text
states: contiguous [L, M, H]
```

workspace 的生命周期由模型/runtime 管理。operator 不负责 append/update history，也不接收 Python list of tensors。pointer array 和 paged/indexed state store 可以等第一版跑通后再评估。

这里的 `L` 是运行时 source-state 数量，不要求 compile-time constant。第一版 op 只消费 caller 已经准备好的 compact contiguous `[L, M, H]` view，不负责把每一层 / 每个 block 的历史 state append 到 workspace。若 prefill 中 `L` 随 depth 从 1, 2, 3... 增长，朴素 contiguous rebuild 可能产生额外拷贝；这属于 runtime state-store 设计问题，必须在 Phase 2 tracking issue 中作为 known limitation 记录。后续可以评估 ring buffer、paged state store、index array 或 pointer-array kernel boundary，但不在第一版 contract 中提前承诺。

这个 contract 不是 manifest freeze。它是我们拿去开 tracking issue 和同事讨论的起点。最终 contract 由 tracking issue 收敛后，再进入 spec-only manifest PR。

### 4.3 Depth Weighted Sum：内部垫脚石，不作为第一批公开目标

`CrossLayerWeightedSumFwdOp` 或 `DepthWeightedSum` 这种形式：

```text
states:  [L, M, H]
weights: [L, M]
output = sum_l weights[l, m] * states[l, m, :]
```

很适合作为 PyTorch reference、prototype benchmark 或 Block AttnRes 的分解 baseline。类似思想在 ELMo scalar mix、BERT layer pooling、sentence-transformer weighted layer pooling 里出现过，但这些更多是表征抽取或下游 pooling 场景，不是现代 LLM 推理主路径中明确的性能热点。

因此它本身太接近 fused weighted reduction，不足以单独证明新 family 的必要性。

第一批计划不把它作为公开 manifest 目标。只有当它被 MHC、AttnRes 或后续 op 真实复用，或者有明确独立用户时，再考虑进入 manifest。

## 5. Manifest 与实现节奏

这份计划不直接输出可提交的 manifest YAML。TileOps manifest 是人工审查的 source of truth，实际 PR 会按 manifest trust model 分开：

1. **Manifest PR 与 implementation PR 分离。**
2. **`spec-only` 到 `implemented` 由通过 CI 的实现 PR 触发。**
3. **完整 manifest 条目必须补齐 `signature`、`shape_rules`、`workloads`、`roofline`、`source`。**
4. **已有公开 API 不在 manifest PR 中顺手重命名。**

### Phase 0：Taxonomy

在 `docs/design/` 新增本文档，确立 `cross_layer` 的 family 边界和 admission rule。本文档是讨论用 release plan，不是 manifest spec。

### Phase 1：MHC Manifest Alignment

Phase 1 先处理 MHC family reconciliation，再决定是否新增：

```text
tileops/manifest/cross_layer.yaml
```

如果 tracking issue 决定迁移或新增 `cross_layer` manifest，第一批只放 MHC：

```text
MHCPreFwdOp
MHCPostFwdOp
```

这个阶段分两步推进。

Phase 1a 先补完整 manifest spec，并保持 `spec-only`。如果 MHC 已经以 `sequence_modeling` family 在其他计划中落地，则这个阶段必须明确是 migration、retag，还是只补 semantic note：

```text
signature
shape_rules
workloads
roofline
source metadata
```

MHCPre 的 `sinkhorn_repeat` 和 `sinkhorn_eps` 不能藏在 prose 里。manifest workload / roofline 需要把 Sinkhorn 迭代次数作为 workload descriptor 或 shape parameter 记录，否则 flop / byte 估算会漂移。`n_expand`、`C`、`sinkhorn_repeat` 应共同决定 MHCPre 的 roofline entry。

Phase 1b 对齐现有 tests / benchmarks / source metadata。MHC 现有测试以 cosine similarity 为主；如果要把 status 提升到 `implemented`，需要先补齐更严格的数值 gate。通过改进后的测试和 CI 后，再单独提交 status promotion。如果发现 kernel 或 wrapper 需要修复，则保持 `spec-only`，由后续 implementation PR 处理。

### Phase 2：Block AttnRes Tracking Issue

为 Block AttnRes 单独开 issue。issue 中收敛：

```text
official Kimi contract
state storage layout
fusion boundary
correctness reference
benchmark workloads
dtype / accumulation policy
dtype_combos for states / score_weight / rms_weight / output
workspace append/update ownership
contiguous vs non-contiguous state storage
dynamic L and compact workspace rebuild cost
causal depth mask representation
sequence-parallel compatibility
```

这个阶段不改主仓实现，也不把半定稿 API 写进 manifest。

### Phase 3：Block AttnRes Spec-Only Manifest

issue 收敛后，再提交 `BlockAttnResFwdOp` 的完整 spec-only manifest 条目。此时需要明确：

```text
signature
shape_rules
workloads
roofline
source placeholders
correctness policy
benchmark policy
```

### Phase 4：Block AttnRes Implementation

实现 op wrapper、kernel、tests、benchmarks。CI 通过后，再单独提升 manifest status。

### Phase 5：L2 / Helper Extraction

等 MHC 和 Block AttnRes 都稳定后，再判断是否存在值得抽取的 L2 helper。可能有价值的是：

```text
state-axis shape validation
contiguous [L, M, H] layout checks
causal-depth / source-axis mask checks
shared benchmark helpers
```

不会在第一版提前创造 `CrossLayerOp` 基类。

### Phase 6：Adjacent Architecture Follow-Up

`cross_layer.scheduling` 和 `cross_layer.cache_sharing` 后续分别开独立 tracking issue。前者围绕 Mixture-of-Depths、LayerSkip、early-exit routing 这类 depth execution policy，先明确 execution mask、token compaction、verification / rollback 和 graph boundary；后者围绕 CLA、LCKV、YOCO-style KV/state reuse，先明确 cache select、gather、remap、shared read/write 和 optional numerical mix 哪些是真正的 kernel boundary。

这两个方向都属于 `cross_layer` 大类，但不在第一批 manifest PR 中提前创造宽泛 op。后续进入 manifest 时，应使用具体算子名，例如 `DepthExecutionMaskPack`、`LayerKVGather` 或 `CrossLayerCacheRemap`，而不是按模型结构命名。

## 6. 测试与 Benchmark

### 6.1 MHC

MHC 现有测试使用 cosine similarity。后续 manifest alignment 中，cosine similarity 更适合作为诊断指标，不应是唯一 gate。数值 gate 应补充：

```text
torch.testing.assert_close 或明确的 max_abs / max_rel threshold
shape check
nonfinite check
repeatability check
```

原因是 cosine similarity 对尺度误差不敏感，例如 `actual = 2 * expected` 仍可能有很高 cosine。

### 6.2 Block AttnRes

Block AttnRes 的测试要覆盖 depth-axis softmax 的数值边界：

```text
L = 1, 2, 8, 16
H 非 tile 整数倍
M 很小和很大
全相同 logits
极大正负 logits
单一 depth 权重接近 1
states 中存在大幅值差异
bf16/fp16 input + fp32 reference
```

dtype contract 需要在 tracking issue 中收敛成表格，而不是只写一句 mixed precision。至少要明确：

```text
states dtype
score_weight dtype
rms_weight dtype
logits accumulation dtype
weighted accumulation dtype
output dtype
whether fp8 states are out of scope, reference-only, or a future extension
```

额外检查：

```text
softmax sum ~= 1
finite input 不产生 NaN/Inf
输出在有限精度容差下保持 convex-combination 直觉
同一输入重复执行的确定性边界
```

### 6.3 Benchmark Baseline

Block AttnRes 至少需要比较：

```text
PyTorch eager reference
torch.compile reference
unfused operator composition:
  RMSNorm
  projection
  softmax
  weighted sum
TileOps fused implementation
```

roofline 需要区分：

```text
algorithmic bytes
materialized implementation bytes
```

例如朴素 composition 可能需要多次 HBM 往返：

```text
states [L,M,H] read -> RMSNorm -> temp [L,M,H] write
temp read -> projection -> logits [L,M] write
logits read -> softmax -> weights [L,M] write
weights + states read -> weighted sum -> output [M,H] write
```

而 fused kernel 的目标是把这些阶段压进一个 kernel boundary，在 tile 内尽量保留中间量：

```text
states [L,M,H] tiled read
score_weight/rms_weight read
output [M,H] write
```

`algorithmic bytes` 表示公式层面的理论下界；`materialized implementation bytes` 表示实际 kernel 因临时张量、layout、spill 或多 kernel composition 产生的 HBM traffic，后者需要通过 profiler 校验。

因为 fused kernel 如果在片上保留 state tile，实际 memory traffic 会和朴素多 kernel composition 不同。

## 7. References

- Attention Residuals, Kimi Team: https://arxiv.org/abs/2603.15031
- MoonshotAI Attention Residuals official repository: https://github.com/MoonshotAI/Attention-Residuals
- Hyper-Connections: https://arxiv.org/abs/2409.19606
- mHC: Manifold-Constrained Hyper-Connections: https://arxiv.org/abs/2512.24880
- Mixture-of-Depths: Dynamically allocating compute in transformer-based language models: https://arxiv.org/abs/2404.02258
- LayerSkip: Enabling early exit inference and self-speculative decoding: https://arxiv.org/abs/2404.16710
- Reducing Transformer Key-Value Cache Size with Cross-Layer Attention: https://arxiv.org/abs/2405.12981
- Layer-Condensed KV Cache for Efficient Inference of Large Language Models: https://arxiv.org/abs/2405.10637
- LCKV official repository: https://github.com/whyNLP/LCKV
- Depth-Attention: Cross-Layer Value Mixing for Language Models: https://arxiv.org/abs/2606.05014
- TileOps existing MHC implementation:
  - `tileops/ops/mhc.py`
  - `tileops/kernels/mhc/mhc_pre.py`
  - `tileops/kernels/mhc/mhc_post.py`
  - `tests/ops/test_mhc_pre.py`
  - `tests/ops/test_mhc_post.py`
