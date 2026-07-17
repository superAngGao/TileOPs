# Cross-Layer 算子族发布计划

## 1. 背景与需求

TileOps 现有算子族并不是单纯按底层计算 primitive 分类。`elementwise`、`reduction`、`scan` 更接近计算方法；`attention`、`moe`、`normalization` 则按模型中稳定的算子语义分类。随着新模型结构开始显式使用 depth / layer / block / expanded-residual 方向的数据流，仅靠 `attention`、`moe`、`normalization` 或 `reduction` 已经不够自然。

`cross_layer` 算子族的目标是覆盖这类操作：

```text
多个 layer / block / residual-channel / KV-source states
    -> 选择、路由、加权、注意力聚合或仿射混合
    -> 当前层继续使用的表示或状态
```

建立该算子族的直接动机有三点：

1. **新模型结构已经出现明确需求。** Kimi Attention Residuals 对前序 layer/block outputs 做 depth-wise attention aggregation；MHC / Hyper-Connection 显式维护 expanded residual channels 并做跨通道混合。
2. **现有分类会造成语义混淆。** AttnRes 名字里有 residual，但不是普通 `x + residual`；它使用 attention-like weights，但 attention axis 是 layer/block depth，而不是 token sequence。
3. **需要可复用的接口和 benchmark 维度。** 这类 op 共享 depth-like axis、causal-depth 约束、layer/block state layout、routing/mixing metadata，而这些并不属于普通 `reduction` 或 `normalization` 的职责。

## 2. 算子族边界

`cross_layer` 的核心不是“输入来自上一层”。普通 Transformer 的 attention、MLP、RMSNorm 也都接收上一层 hidden state；如果按这个标准，所有子层都会被错误归入 cross-layer。

本计划采用更严格的定义：

```text
cross_layer operator:
  算子签名中显式包含多个 layer / block / residual-channel / source-layer states，
  并且算子本身负责沿 depth / layer / expanded-residual / KV-source axis
  做 combine、routing、selection、sharing 或 transform。
```

### 2.1 In Scope

| 类别 | 代表 | 纳入原因 |
| --- | --- | --- |
| Depth-wise residual aggregation | Kimi AttnRes / Block AttnRes | 显式对 preceding layer/block outputs 做 softmax attention aggregation |
| Expanded residual-channel mixing | MHC / Hyper-Connection | 显式维护 `n_expand` residual channels，并生成 mixing matrix 做通道间传递 |
| Layer weighted aggregation | ELMo-style scalar mix, BERT layer pooling | 显式读取多层 hidden states 并加权合成表示 |
| Cross-layer KV/state sharing | CLA, LCKV, YOCO-style KV reuse | 当前层复用、共享或聚合其他层产生的 KV/state |
| Depth routing / layer selection | Mixture-of-Depths, dynamic layer routing | token 或 batch 沿 depth/layer path 做动态选择 |

### 2.2 Out of Scope

| 类别 | 当前归类 | 不纳入原因 |
| --- | --- | --- |
| 普通 residual add | `elementwise` | 只是 `x + residual`，没有显式跨层轴 |
| FusedAddRMSNorm / FusedAddLayerNorm | `normalization` | residual add 是 norm fusion 的一部分，不做跨 layer/block 混合 |
| 普通 sequence attention | `attention` | attention axis 是 token sequence，不是 layer/block depth |
| MoE expert combine | `moe` | 聚合轴是 expert/route，不是 layer/block |
| Engram GateConv | `sequence_modeling` / model-specific fused op | 使用当前 hidden stream 和 n-gram memory，不显式聚合多个 layer/block states |
| 训练专用 auxiliary heads | training-only | 推理主路径不稳定，暂不作为 TileOps release target |

Engram 的边界尤其需要写清楚：Engram 会利用当前 hidden stream 中已经累积的前层信息，也包含 local residual path；但它的主计算是 n-gram memory lookup、gating、causal/depthwise conv 和 local residual add，并没有在算子接口中暴露多个 layer/block states 作为待混合对象。因此它是 cross-layer-adjacent，但不是 `cross_layer` family 的核心成员。

## 3. 计划添加或迁移的算子

### 3.1 第一批：已有 TileOps 基础

#### MHCPreOp / MHCPostOp

当前 TileOps 已有：

```text
tileops/ops/mhc.py
tileops/kernels/mhc/mhc_pre.py
tileops/kernels/mhc/mhc_post.py
tests/ops/test_mhc_pre.py
tests/ops/test_mhc_post.py
benchmarks/ops/bench_mhc_pre.py
benchmarks/ops/bench_mhc_post.py
```

但 MHC 当前还没有进入 `tileops/manifest/*.yaml`。因此第一阶段不是“迁移 family 字段”，而是为已有实现从零补齐 manifest spec，并确认它是否满足 `implemented` 状态要求。

MHCPre 的参考语义可概括为：

```text
forward(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat, sinkhorn_eps)

x:       [B, n_expand * C]
phi:     [n_expand * C, n_expand * n_expand + 2 * n_expand]
b:       [n_expand * n_expand + 2 * n_expand]

生成:
  h_pre: [B, n_expand]
  h_res: [B, n_expand, n_expand]  # includes Sinkhorn-style normalization

x_res   = h_res @ reshape(x, [B, n_expand, C])
x_layer = h_pre @ reshape(x, [B, n_expand, C])
```

MHCPost 的参考语义可概括为：

```text
forward(x_layer_out, h_post, x_res)

x_out = h_post[:, :, None] @ x_layer_out[:, None, :] + x_res
```

它显式在 expanded residual-channel axis 上混合和回写，是 `cross_layer` family 的现成例子。

计划：

1. 保留现有公开 Python API：`MHCPreOp` / `MHCPostOp`。
2. 新增完整 manifest 条目时，manifest key 第一阶段先沿用现有公开 API 名，避免破坏下游调用。
3. 如果后续要改成 `MHCPreFwdOp` / `MHCPostFwdOp`，需要单独 deprecation PR，提供 alias wrapper 和迁移说明。
4. 为 `cross_layer` family 建立 workload 命名和 roofline 表达，但不在同一个 PR 中移动源码目录。

### 3.2 第二批：Kimi Attention Residuals

Kimi Attention Residuals 将标准 residual 的固定累加替换为对前序 layer outputs 的 learned, input-dependent softmax attention。Block AttnRes 进一步将 layer 分块，聚合 block-level representations，以降低 memory / communication 开销。

候选算子：

```text
AttnResFwdOp
BlockAttnResFwdOp
```

建议第一版只做 forward / inference-facing path，先明确 shape contract：

```text
current_state:        [B, S, H] or [B, H]
history_states:       [B, L, S, H] or [B, L, H]
query/proj weights:   model-dependent
mask / causal_depth:  optional
output:               [B, S, H] or [B, H]
```

其中 `L` 是 layer/block depth axis。该 op 不应放入普通 `attention`，因为它的 attention axis 不是 sequence token；也不应放入 `normalization`，因为它不是 add-norm fusion。

### 3.3 第三批：基础 Cross-Layer Weighted Sum

为了给 AttnRes 和 MHC 之间提供最小公共 primitive，可以考虑加入：

```text
CrossLayerWeightedSumFwdOp
```

参考语义：

```text
states:  [B, L, ..., H]
weights: [B, L, ...] or [L]
output = sum_l weights[..., l] * states[:, l, ...]
```

该算子可覆盖 layer scalar mix、weighted layer pooling、block-level residual mix 等简单模式。它也可以作为 AttnRes 的 correctness reference baseline。

### 3.4 后续观察：Cross-Layer KV / State Sharing

这些工作暂不作为第一批实现目标，但应作为 `cross_layer` family 的中长期参考：

| 方向 | 代表 | 备注 |
| --- | --- | --- |
| Cross-Layer Attention | CLA | 相邻层共享 K/V activations，减少 unique KV cache layers |
| Layer-Condensed KV Cache | LCKV | 只计算/cache 少数层的 K/V，让其他层 query 配这些共享 K/V |
| YOCO-style KV reuse | YOCO / YOCO++ | 通过结构化 decoder 设计减少 persistent KV cache |
| Depth-Attention | Depth-wise value mixing | 在 attention module 内沿 depth 方向选择和混合 value |

这些方向的实现可能更接近 `attention` 与 `cross_layer` 的交叉地带。第一版可以只在设计文档中占位，不急于加入 manifest。

## 4. 可执行 PR 边界

本计划当前是 taxonomy / release plan，不直接输出可提交的 manifest YAML。实际 PR 必须遵守 TileOps manifest trust model：

1. **Manifest PR 与 implementation PR 分开。** manifest spec 是人工审查的 source of truth；implementation PR 不应同时修改 manifest spec。
2. **`spec-only` 到 `implemented` 必须由通过 CI 的实现 PR 触发。** 只有当 op wrapper、kernel、test、benchmark、source metadata 都齐备并通过 CI 后，才能在后续 manifest PR 中提升状态。
3. **完整 manifest 条目不能省略 schema 字段。** 每个条目需要按 `docs/design/manifest.md` 和现有 manifest 文件补齐 `signature`、`shape_rules`、`workloads`、`roofline`、`source` 等字段。
4. **公开 Python API 不在 manifest PR 中重命名。** 对已有 `MHCPreOp` / `MHCPostOp`，第一步只补 manifest，不做破坏性命名迁移。

### 4.1 Manifest PR：新增 family 与 spec-only 条目

第一份 PR 只做：

```text
tileops/manifest/cross_layer.yaml
```

并确认 `tileops.manifest` 是否自动 glob 所有 `*.yaml`。如果不是自动发现，需要在同一 manifest-only PR 中加入 family 注册。

第一批条目建议：

| Manifest key | family | status | 说明 |
| --- | --- | --- | --- |
| `MHCPreOp` | `cross_layer` | `spec-only` 或 `implemented` 需按 CI 状态确认 | 已有 Python wrapper/kernel，但目前没有 manifest；补齐完整 spec 后再判断状态 |
| `MHCPostOp` | `cross_layer` | `spec-only` 或 `implemented` 需按 CI 状态确认 | 同上 |
| `CrossLayerWeightedSumFwdOp` | `cross_layer` | `spec-only` | 最小 reference primitive，先不要求实现 |
| `BlockAttnResFwdOp` | `cross_layer` | `spec-only` | 等官方 shape/API 对齐后再实现 |

这里不放简化 YAML 片段，避免误导为可直接提交的 manifest。实际 PR 中每个 op 必须包含完整的 `signature`、`shape_rules`、合法 `workloads`、`roofline` 和 `source`。

### 4.2 Implementation PR：从具体 T2 Op 开始

第一批 implementation PR 不抽 `CrossLayerOp` L2 基类。根据 `ops-design.md` 的原则，应先实现 2-3 个具体 T2 op，理解重复模式后再提取 L2。

建议顺序：

1. 保持现有 MHC 源码路径不变，补齐 manifest source metadata、tests、benchmarks 的一致性。
2. 如需新增实现，优先做 `CrossLayerWeightedSumFwdOp` 作为最小 T2 op。
3. 等 MHC、WeightedSum、AttnRes 至少两个具体 op 的接口稳定后，再讨论 L2 基类。

### 4.3 Future L2：只在模式稳定后提取

如果后续确实出现共享逻辑，L2 层可以只承担轻量职责：

```text
cross-axis shape validation
layout normalization helpers
causal-depth mask validation
shared benchmark/workload utilities
```

不建议第一版定义 `CrossLayerOp.family`、`mode`、`layout` 这类未被现有 Op 框架消费的类变量；family 属于 manifest，不属于 Python base class。

## 5. Manifest 与目录布局

第一阶段建议只新增 manifest 文件，不移动源码：

```text
tileops/manifest/cross_layer.yaml
```

第二阶段若出现新实现，再新增：

```text
tileops/ops/cross_layer/
tileops/kernels/cross_layer/
tests/ops/test_cross_layer_*.py
benchmarks/ops/bench_cross_layer_*.py
```

MHC 的现有路径可先保留：

```text
tileops/ops/mhc.py
tileops/kernels/mhc/
tests/ops/test_mhc_*.py
benchmarks/ops/bench_mhc_*.py
```

如果未来决定迁移到 `tileops/ops/cross_layer/mhc.py`，应单独开路径迁移 PR，并保留 import compatibility。

命名建议：

| 现有/候选算子 | 第一阶段 manifest key | 后续可选规范名 | 说明 |
| --- | --- | --- | --- |
| MHC pre | `MHCPreOp` | `MHCPreFwdOp` | 第一阶段保留现有公开 API |
| MHC post | `MHCPostOp` | `MHCPostFwdOp` | 第一阶段保留现有公开 API |
| Layer scalar mix | `CrossLayerWeightedSumFwdOp` | 同左 | 最小 cross-layer primitive |
| Kimi AttnRes | `AttnResFwdOp` | 同左 | full previous-layer attention residual |
| Kimi Block AttnRes | `BlockAttnResFwdOp` | 同左 | block-level aggregation target |

## 6. 测试与 Benchmark 维度

`cross_layer` benchmark 应明确暴露 depth-like axis：

```text
B: batch
S: sequence length, optional
L: number of source layers / blocks / residual channels
H: hidden size
K: selected source count, optional
dtype
layout
causal_depth
```

Manifest 中不能使用范围表达式。每个 workload 必须展开成完整 shape / dtype dict，例如：

```yaml
workloads:
  - {batch: 1, n_expand: 4, c_x: 1280, dtypes: [bfloat16], label: "mhc-small"}
  - {batch: 2, n_expand: 4, c_x: 1920, dtypes: [bfloat16], label: "mhc-medium"}
  - {batch: 4, n_expand: 4, c_x: 2560, dtypes: [bfloat16], label: "mhc-large"}
```

不同 op 的 workload 字段应按该 op 的 `signature.params` 命名，不引入孤立的 workload key API。

正确性策略：

1. 对 weighted sum / affine mix 使用 PyTorch fp32 reference。
2. 对 AttnRes 使用官方实现或论文公式构造 reference。
3. 对 MHC 保留现有 cosine similarity 检查，同时补充 shape / nonfinite / deterministic seed 检查。

## 7. 发布节奏

### Phase 0: Design

- 新增本设计文档。
- 建立 `cross_layer` family 的边界说明。
- 建立 AttnRes tracking issue，与 KDA issue 分开。

### Phase 1: Manifest-only PR

- 新增 `tileops/manifest/cross_layer.yaml`。
- 为 MHC pre/post 从零写完整 manifest 条目。
- 为 `CrossLayerWeightedSumFwdOp` 和 `BlockAttnResFwdOp` 写 `spec-only` 条目。
- 确认新 family 是否需要显式注册。
- 不改 Python implementation。

### Phase 2: MHC manifest alignment / CI

- 对照现有 MHC tests/benchmarks/source metadata，决定 `MHCPreOp` / `MHCPostOp` 是否满足 `implemented`。
- 若满足，在单独 manifest status PR 中从 `spec-only` 升级到 `implemented`。
- 若不满足，先补 tests/benchmarks/source metadata，再升级。

### Phase 3: First new concrete T2 op

- 新增 `CrossLayerWeightedSumFwdOp` implementation，或直接实现 `BlockAttnResFwdOp` 的最小版本。
- 不抽 L2 基类，先让具体 op 通过 CI。
- 对应 manifest status 在 CI 通过后单独升级。

### Phase 4: AttnRes

- 根据 MoonshotAI / Kimi 官方实现确认输入输出契约。
- 先实现 `BlockAttnResFwdOp` 或 spec-only manifest entry。
- 建立 correctness reference 和 benchmark workload。

### Phase 5: L2 extraction

- 在 MHC、WeightedSum、AttnRes 至少两个实现稳定后，再评估是否提取 `CrossLayerOp` 或 layout helper。
- L2 只抽重复 shape/layout validation，不抽虚构 forward 模板。

### Phase 6: KV / State Sharing

- 观察 CLA / LCKV / YOCO / Depth-Attention 的模型生态和官方实现。
- 如果出现稳定推理接口，再加入 `CrossLayerKVShareOp` 或相关 tracking issue。

## 8. References

- Attention Residuals, Kimi Team: https://arxiv.org/abs/2603.15031
- MoonshotAI Attention Residuals official repository: https://github.com/MoonshotAI/Attention-Residuals
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
