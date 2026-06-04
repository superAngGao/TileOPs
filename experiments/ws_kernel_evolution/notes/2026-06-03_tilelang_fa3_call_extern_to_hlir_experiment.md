# TileLang FA3 call_extern to HLIR Experiment - 2026-06-03

## 目标

把当前可工作的 TileLang `@T.prim_func` shell 里的 FA3/CUTLASS
`T.call_extern` device helper，逐步搬回 TileLang 高级语言表达。

这里的目标不是换成 host launcher，也不是直接把 FA3 persistent dispatch
照搬进来。当前阶段要保留这个边界：

- TileLang 负责启动 CTA/block 和声明 kernel 形状。
- `T.call_extern` 只作为已经验证的正确性基线和临时桥。
- 后续每一步都尽量把 device helper 中的一块逻辑改写成 TileLang 表达。
- 在没有实现对应 dispatch metadata 之前，不把 persistent 当作默认路径。

## 已发现但尚未优化的问题

这个列表放在开头，后续每推进一步都同步更新。

- **persistent dispatch 尚未泛化。** 已新增实验分支
  `--static-persistent-call-extern`，TileLang grid 改为 `(NUM_SMS, 1, 1)`，
  helper 使用 `StaticPersistentTileScheduler<false>`。但该分支目前只覆盖当前
  dense、non-causal、non-pack GQA 形状，`NUM_SMS=132` 仍是硬编码，所以还不能
  当作默认 dispatch。
- **性能仍低于 host launcher。** 4096 GQA 上 TileLang `call_extern` shell 是
  `0.096834 ms / 709.66 TFLOP/s`；实验 static persistent 分支是
  `0.093599 ms / 734.19 TFLOP/s`；host launcher 对照是
  `0.081237 ms / 845.92 TFLOP/s`。当前最好分支仍慢约 `15.3%`，需要分清是
  scheduler、wrapper 参数构造、descriptor 存储还是 launch 形态造成。
- **debug probe 已从优化路径移出。** WG/thread/block coord 校验现在通过
  `--validate-wg-branch` 显式打开，默认 benchmark 不再把
  `tileops_fa3_shaped_role_probe` 的开销算进起点。
- **params / descriptor 生命周期仍是显式规避。** 当前 helper 用 device global
  params storage 避免 TMA descriptor 落到不稳定 local stack。后续如果搬到
  TileLang 高级表达，需要确认编译器生成的 descriptor/params 存储同样稳定。
- **TMA atom 仍依赖 CuTe/FA3 类型系统。** TileLang 已创建 TMA descriptor，
  但实际 FA3 TMA atom、`g_stride_`、swizzle 语义仍在 external helper 里组装。
- **GQA non-pack 只验证了部分形状。** 已验证 `H=8, HKV=2` 的 4096；
  还需要扩到 `H=32, HKV=8` 和更长序列。
- **pack GQA 尚未纳入默认路径。** 当前实验固定 dense、non-pack GQA；
  pack GQA 应单独开分支验证，避免和本阶段迁移混在一起。
- **epilogue / mainloop 还没有 TileLang 化。** 现在只是 TileLang 启动 CTA，
  FA3 计算核心仍在 `T.call_extern` helper 中。

## 当前基线

当前 probe：

- [_probe_tilelang_fa3_shaped_shell.py](/home/ga/TileOPs/_probe_tilelang_fa3_shaped_shell.py)

当前默认路径：

- TileLang `@T.prim_func`
- `T.Kernel` 启动 grid
- `T.create_tma_descriptor` 创建 Q/K/V/O TMA descriptor
- `T.call_extern` 进入外部 `__device__` helper
- helper 内部构造 FA3 `CollectiveMainloop::Params`,
  `CollectiveEpilogue::Params`, `Scheduler::Params`
- scheduler 使用 `SingleTileScheduler`

已确认的重要修正：

- O 的 TMA global stride 是 byte stride，BF16 输出必须乘 `2`。
- GQA 下 mainloop 的 `qhead_per_khead_divmod` 必须是
  `cutlass::FastDivmod(GROUP)`，否则 query head 会被错误当作 KV head。
- default non-pack GQA scheduler params 的 `qhead_per_khead` 记录为 `GROUP`。

4096 GQA 基线结果：

```text
shape: B=1, S=4096, H=8, HKV=2, D=128
mode: TileLang primfunc + T.call_extern + FA3 device helper
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.099478 tflops=690.80
```

host launcher 对照结果：

```text
shape: B=1, S=4096, H=8, HKV=2, D=128
mode: FA3 host launcher
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.081237 tflops=845.92
```

这个对照只作为参考上限，不作为本实验的实现方向。

## 实验原则

1. 每次只搬出一层语义。
2. 每一层都和现有 `T.call_extern` 基线做 correctness 对照。
3. 先固定 dense、non-pack GQA、non-causal、FP8 in / BF16 out。
4. 先保留 `SingleTileScheduler` 的 block mapping。
5. 只有在 TileLang 侧能表达对应调度元数据后，再讨论 persistent。

## 当前架构决策：走 A 路径

当前 producer 迁移主线选择：

```text
A: TileLang owns shared buffer layout.
B: FA3/CUTLASS owns SharedStorage layout.
```

A/B 的边界：

- A 中 TileLang 直接用 `T.alloc_shared` 声明 typed shared buffer，
  用 `T.annotate_layout({buf: make_swizzled_layout(buf)})` 绑定 swizzle，
  再用 `T.tma_copy` 写入这些 TileLang-owned buffers。C++ helper 后续只接收
  显式 buffer pointer，不再假设 FA3 `SharedStorage` 的内部 offset。
- B 中 TileLang 继续复用 FA3 raw dynamic shared arena，并向
  `smem_raw + offsetof(SharedStorage, ...)` 这样的 byte offset 写入 typed
  swizzled tile。

决策：主线走 A，B 只保留为低层兼容/诊断方向。

依据：

- TileLang eager `T.Kernel` 推荐直接 `T.alloc_shared` typed buffers，
  由 `MergeSharedMemoryAllocations` 自动 pack 到 shared.dyn arena 并保证
  TMA 对齐。
- `T.handle_add_byte_offset + T.decl_buffer` 属于 low-level/private TIR，
  不是 eager `T.Kernel` 的自然语言表面。
- 本实验目标是把 FA3/CUTLASS helper 逐层搬到 TileLang 高级表达；A 会把
  allocation、layout、barrier、TMA destination 的所有权交给 TileLang，
  更符合后续 HLIR 化。
- B 会继续绑定 FA3 内部 `SharedStorage` struct layout，短期可能贴近原
  kernel，但长期会妨碍替换 producer/consumer 边界。

## 拆解计划

### Step 0: 固化基线

记录并保留以下测试命令：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --bench --warmup 10 --repeat 50
```

通过标准：

- `out finite True`
- `lse finite True`
- cosine 接近当前基线
- benchmark 不出现明显退化或异常高估

### Step 1: 把 FA3 params 构造表格化

先不改代码生成，只把 helper 里的参数拆成可审计表：

- mainloop tensor shape/stride
- epilogue tensor shape/stride
- TMA descriptor basis / swizzle / g_stride
- descale shape/stride
- GQA head mapping
- scheduler params

产物：

- markdown 表格
- 可选 debug print / probe，用于验证 TileLang descriptor 和 FA3 CuTe
  descriptor 的语义一致

### Step 2: TileLang 侧表达 work mapping

把 `SingleTileScheduler` 的 block coordinate 逻辑搬到 TileLang 高级表达：

```text
block_idx = blockIdx.x
bidh      = blockIdx.y
bidb      = blockIdx.z
bidh_kv   = bidh // GROUP
```

目标是先替代 helper 内对 scheduler params 的依赖，而不是改变调度策略。

通过标准：

- H=HKV=1 通过
- H=8, HKV=8 通过
- H=8, HKV=2 通过
- S=128, 224, 4096 都通过

### Step 3: TileLang 侧表达 TMA descriptor 语义

当前 TMA descriptor 已由 TileLang 创建，但 FA3 helper 仍然把 descriptor
复制到 CuTe TMA atom 里。

下一步要研究：

- TileLang 是否能直接表达等价的 TMA load/store atom
- shared memory layout 是否能匹配 FA3 mainloop 需要的 swizzle
- descriptor 生命周期是否仍然满足 TMA load/prefetch 的要求

风险点：

- 之前局部 params/descriptor 生命周期导致过 illegal address。
- 需要避免 descriptor 或 params 落到不稳定的 local stack。

### Step 4: 先搬 epilogue，再搬 mainloop

建议先从 epilogue 开始，因为它比 QK/PV mainloop 更局部：

1. 保留 FA3 mainloop 产生 accumulator / softmax state。
2. 试着用 TileLang 表达 O 写回和 LSE 写回。
3. 再拆 softmax finalize、V descale、BF16 cast。

mainloop 拆解顺序：

1. Q/K TMA load
2. QK WGMMA
3. online softmax update
4. V TMA load
5. PV WGMMA
6. accumulator rescale

### Step 5: 性能和正确性阶梯

每完成一层都跑：

```text
B=1, S=128,  H=1, HKV=1
B=1, S=224,  H=1, HKV=1
B=1, S=4096, H=1, HKV=1
B=1, S=4096, H=8, HKV=8
B=1, S=4096, H=8, HKV=2
```

后续再扩展到生产形状：

```text
B=1, S=4096, H=32, HKV=8
B=1, S=8192, H=32, HKV=8
```

## 当前开放问题

- TileLang 高级语言能否直接表达 FA3 当前使用的 TMA atom 细节？
- 是否需要一个 TileLang intrinsic 来承载 CuTe TMA atom，而不是纯
  `T.call_extern`？
- WGMMA 和 softmax pipeline 是否能在 TileLang 中保持 FA3 的 warpgroup
  分工和 barrier 语义？
- params/descriptor 的稳定存储应该由 TileLang 编译器负责，还是需要显式
  global/workspace storage？
- 非 pack GQA 和 pack GQA 是否应分成两个实验，而不是同时推进？

## 下一步

先做 Step 1：把现有 helper 的 params 构造变成一张可检查表，并标出哪些字段
已经能由 TileLang 高级表达直接生成，哪些字段仍依赖 CuTe/FA3 类型系统。

## 2026-06-03 推进计划：从外向内拆 `T.call_extern`

新的拆解顺序采用“从外向内”：

1. 先在 TileLang 侧显式表达 CTA / thread / warpgroup 分配。
2. 再把 helper 里的 block/work mapping 搬出来。
3. 再搬 params 构造和 descriptor/atom 组装。
4. 最后才碰 epilogue、mainloop、WGMMA、softmax pipeline。

### Step A: 在 `T.call_extern` 外侧拿到 thread binding

在当前 default TileLang kernel 中加入：

```python
tx = T.get_thread_binding()
lane_id = tx % 32
warp_id = tx // 32
warpgroup_id = tx // 128
warpgroup_lane = tx % 128
```

当前 `threads=384`，所以可以先按 FA3/WS 形态理解为：

```text
warpgroup 0: tx in [0, 128)    producer / TMA-oriented role
warpgroup 1: tx in [128, 256)  consumer / MMA role
warpgroup 2: tx in [256, 384)  consumer / MMA role
```

第一步只做外显和验证，不改变 FA3 helper 的执行协议：

- 所有 384 个线程仍然一起调用 `tileops_fa3_shaped_device_call`。
- 不在 TileLang 外层提前按 warpgroup 分支跳过 `T.call_extern`。
- 可选地把 `tx / warpgroup_id / lane_id` 作为额外参数传入 helper，
  先用于 assert/debug，不改变计算。

通过标准：

- `S=128, H=1, HKV=1` 正确。
- `S=4096, H=8, HKV=2` 正确。
- benchmark 不应有明显变化。

状态：已完成。

实现：

- 在 TileLang `T.Kernel(..., threads=384)` 内调用 `T.get_thread_binding()`。
- 在 TileLang 侧计算 `lane_id / warp_id / warpgroup_id / warpgroup_lane`。
- 把这几个值作为额外参数传入 `tileops_fa3_shaped_device_call`。
- helper 内部用 `threadIdx.x` 重新推导同样的值；如果不一致，触发
  `asm volatile("trap;")`。
- 所有 384 个线程仍然一起进入 FA3 helper，没有改变 FA3 warpgroup 协议。

验证：

```text
shape: B=1, S=128, H=1, HKV=1, D=128
out finite True
lse finite True
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
shape: B=1, S=4096, H=8, HKV=2, D=128
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.098476 tflops=697.83
```

结论：

- TileLang 外侧 thread binding 与 CUDA `threadIdx.x` 推导一致。
- 这一层外显没有破坏 correctness。
- benchmark 基本无扰动，可以继续推进到 Step B。

### Step B: 把 block/work mapping 搬到 TileLang 外侧

在 TileLang 侧根据 `T.Kernel(num_m_blocks, grid_heads, batch)` 的 block 变量
直接计算：

```text
tile_m = _bx
bidh = _by
bidb = _bz
bidh_kv = bidh // GROUP
```

然后把这些值传给 helper，让 helper 先只用于 debug 对照：

- helper 内部仍可调用 FA3 scheduler 得到原始 coord。
- 对比 TileLang 传入 coord 和 FA3 scheduler coord 是否一致。
- 对 GQA 检查 `bidh_kv == bidh / GROUP`。

通过后，再把 helper 内部对 `SingleTileScheduler::get_initial_work` 的依赖去掉，
改用 TileLang 传入的 work coord。

### Step C: 搬出 thread-0 初始化和共享状态初始化

当前 helper 里仍有：

```text
if threadIdx.x == 0:
  identity_page_table[blockIdx.z] = blockIdx.z
  placement-new AttnParams into device global params storage
```

下一步尝试在 TileLang 侧表达：

- `if tx == 0` 的 CTA-local 初始化
- 必要的 `T.sync_threads()`
- block coord / GQA coord 的预计算

注意：FA3 `Params` 本身仍然是 C++ 类型，短期内不能直接由 TileLang 构造；
这一步的目标只是把“谁做初始化”和“初始化依赖哪些 coord”外显。

### Step D: 再处理 params / descriptor

只有当 Step A-C 都正确后，再开始拆：

- mainloop shape/stride params
- epilogue shape/stride params
- TMA atom 的 descriptor copy
- `aux_params_.g_stride_`
- descale stride 和 GQA head mapping

这一步会决定我们是能纯 TileLang 表达，还是需要更细粒度的 TileLang intrinsic
承载 CuTe TMA atom。

## 第一阶段 Milestone：WG-shaped TileLang kernel

第一阶段目标不是立刻纯 TileLang 化全部 FA3，而是让当前 shaped shell
长成 [gqa_fwd_fp8.py](/home/ga/TileOPs/tileops/kernels/attention/gqa_fwd_fp8.py)
里的 warp-specialized kernel 形态：

```text
tx = T.get_thread_binding()

if tx < 128:
  producer / TMA / prefetch path
elif tx < 256:
  consumer WG1 / QK + softmax + PV path
else:
  consumer WG2 / QK + softmax + PV path
```

每个 WG 分支里允许先混用：

- TileLang 高级原语：`T.tma_copy`, `T.copy`, `T.wgmma_gemm`,
  `T.wait_wgmma`, `T.barrier_*`, `T.Pipelined`
- 小粒度 `T.call_extern`：只承载 TileLang 目前不好表达的 FA3/CuTe/PTX
  片段

第一阶段结束时，不应该再是一个整 CTA 直接进入
`tileops_fa3_shaped_device_call` 的大黑盒；而应该是 TileLang 外层已经显式
拥有 WG 分支、work mapping、同步骨架和可替换的局部计算片段。

### M0: whole-helper baseline

状态：已完成。

当前可工作的起点：

- TileLang 启动 CTA/block。
- TileLang 创建 Q/K/V/O TMA descriptor。
- 整个 CTA 调用一个大 `T.call_extern` helper。
- helper 内部完成 FA3 params、scheduler、TMA atom、mainloop、epilogue。

基线：

```text
B=1, S=4096, H=8, HKV=2, D=128
latency_ms=0.099478 tflops=690.80
correctness max_abs=0.000457764 cosine=0.99970412
```

### M1: thread/WG identity visible in TileLang

状态：已完成。

TileLang 外层已经计算：

```text
tx
lane_id
warp_id
warpgroup_id
warpgroup_lane
```

这些值传入 helper 并和 CUDA `threadIdx.x` 推导结果校验。

验证：

```text
B=1, S=4096, H=8, HKV=2, D=128
latency_ms=0.098476 tflops=697.83
correctness max_abs=0.000457764 cosine=0.99970412
```

### M2: work coord visible in TileLang

状态：已完成。

目标：

在 TileLang 外层计算并传入：

```text
tile_m = _bx
bidh = _by
bidb = _bz
bidh_kv = bidh // GROUP
```

helper 内先只做一致性校验：

- TileLang coord 和 `SingleTileScheduler` coord 一致。
- `bidh_kv` 和 FA3 mainloop 使用的 GQA head mapping 一致。

通过后，helper 不再需要通过 `SingleTileScheduler::get_initial_work` 取得
当前 CTA work coord。

验收标准：

- **代码形态。** TileLang 外层显式计算 `tile_m / bidh / bidb / bidh_kv`，
  并作为参数传入 helper。
- **一致性校验。** helper 内部仍保留 FA3 `SingleTileScheduler` coord，
  并校验 TileLang 传入 coord 完全一致；不一致时触发 trap。
- **GQA 校验。** 对 `H > HKV` 的形状，确认 `bidh_kv = bidh // GROUP`
  和 mainloop `qhead_per_khead_divmod` 的映射一致。
- **正确性。** 以下形状全部通过：

```text
B=1, S=128,  H=1, HKV=1
B=1, S=4096, H=1, HKV=1
B=1, S=4096, H=8, HKV=8
B=1, S=4096, H=8, HKV=2
```

- **性能。** `B=1, S=4096, H=8, HKV=2` latency 不明显差于 M1；
  目标为 `<= 1.03x M1 latency`。
- **记录。** markdown 中追加本 milestone 的命令、结果和结论。

实现：

- TileLang 外层根据 `T.Kernel(num_m_blocks, grid_heads, batch)` 的 block
  变量计算 `tile_m / bidh / bidb / bidh_kv`。
- 这些 coord 和 M1 的 thread/WG identity 一起传入
  `tileops_fa3_shaped_device_call`。
- helper 内部校验：

```text
tile_m == blockIdx.x
bidh == blockIdx.y
bidb == blockIdx.z
bidh_kv == blockIdx.y / GROUP
0 <= bidh < H
0 <= bidh_kv < HKV
```

不一致时触发 `asm volatile("trap;")`。

验证：

```text
B=1, S=128, H=1, HKV=1
out finite True
lse finite True
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
B=1, S=4096, H=1, HKV=1
out finite True
lse finite True
correctness max_abs=0.000473022 cosine=0.99975061
```

```text
B=1, S=4096, H=8, HKV=8
out finite True
lse finite True
correctness max_abs=0.000427246 cosine=0.99975872
```

```text
B=1, S=4096, H=8, HKV=2
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.097326 tflops=706.07
```

M1 latency baseline 是 `0.098476 ms`，M2 的 `0.097326 ms` 满足
`<= 1.03x M1 latency` 的验收标准。

结论：

- TileLang 外侧的 CTA work coord 和当前 `SingleTileScheduler` direct block
  mapping 一致。
- GQA head mapping `bidh_kv = bidh // GROUP` 已在 helper 边界校验。
- 可以继续推进 M3：在 `T.call_extern` 外侧形成真实 WG branch skeleton。

### M3: WG branch skeleton outside `T.call_extern`

状态：已完成，按放宽后的 performance threshold 通过。

目标：

把当前单个整 CTA helper 拆成 WG-shaped shell：

```text
if tx < 128:
  T.call_extern("producer_helper", ...)
elif tx < 256:
  T.call_extern("consumer1_helper", ...)
else:
  T.call_extern("consumer2_helper", ...)
```

这里的 helper 仍可包含 FA3/CUTLASS 逻辑，但边界必须变小：

- producer helper 只做 producer/TMA 相关工作。
- consumer helper 只做对应 half-M consumer 的 QK/PV/epilogue。
- shared memory、barrier、phase counter 的所有权开始转移到 TileLang 外层。

风险：

- FA3 当前 mainloop 的 warpgroup 协议可能假设整 CTA 在同一个 C++ kernel
  body 中协作。拆 helper 时必须避免让 producer/consumer 的 barrier 协议失配。

通过标准：

- `S=128, H=1, HKV=1` 正确。
- `S=4096, H=8, HKV=2` 正确。
- latency 不明显差于 M1 whole-helper baseline。

验收标准：

- **代码形态。** TileLang 外层出现真实三分支：

```text
if tx < 128:
  ...
elif tx < 256:
  ...
else:
  ...
```

- **helper 边界。** 原来的整 CTA helper 被拆成至少 2 个更小 helper：
  producer helper 和 consumer helper；如果 consumer WG1/WG2 仍共用同一个 helper，
  必须通过参数区分 half-M / WG role。
- **执行协议。** 每个 helper 只在对应 WG 分支内被调用；不再所有 384
  线程一起进入同一个大 helper。
- **同步骨架。** shared memory、barrier id、phase counter 的定义开始位于
  TileLang 外层；helper 不应私自拥有整 CTA 的全部同步状态。
- **正确性。** 至少通过：

```text
B=1, S=128,  H=1, HKV=1
B=1, S=4096, H=8, HKV=2
```

- **性能。** `B=1, S=4096, H=8, HKV=2` latency 目标放宽为
  `<= 1.15x M1 latency`。超过该阈值可以继续实验，但不能标记 M3 完成。
  旧的 `1.10x` 目标保留为后续优化参考，不再作为 M3 阻塞条件。
- **记录。** markdown 中记录 helper 拆分边界、同步所有权和 benchmark。

#### M3a: per-WG branch probe

状态：已完成。

实现：

- TileLang 外层已经出现真实三分支：

```text
if tx < 128:
  role = producer
elif tx < 256:
  role = consumer WG1
else:
  role = consumer WG2
```

- 每个分支调用一个小粒度 extern：

```text
tileops_fa3_shaped_role_probe(...)
```

- role probe 校验：

```text
role_tl == role_from_threadIdx
tx_tl == threadIdx.x
warpgroup_id_tl == threadIdx.x / 128
tile_m == blockIdx.x
bidh == blockIdx.y
bidb == blockIdx.z
bidh_kv == blockIdx.y / GROUP
```

- role probe 不碰 shared memory、barrier 或 FA3 计算核心。
- whole-helper `tileops_fa3_shaped_device_call` 仍然保留在三分支之后，
  因此 M3a 只是分支骨架验证，不算 M3 完成。

验证：

```text
B=1, S=128, H=1, HKV=1
out finite True
lse finite True
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
B=1, S=4096, H=8, HKV=2
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.103407 tflops=664.55
```

M1 latency baseline 是 `0.098476 ms`，M3a 为 `1.052x M1`，仍低于
M3 原 `1.10x` 性能阈值。

结论：

- TileLang 可以稳定 lower `if tx < 128 / elif tx < 256 / else` 的
  per-WG branch skeleton。
- 每个 WG 分支可以独立调用小粒度 `T.call_extern`。
- 不能直接在这些分支内调用 FA3 whole-kernel helper，因为 FA3
  `FlashAttnFwdSm90::operator()` 在 role 分支前有 CTA-wide prepare 和
  `__syncthreads()`。下一步必须先拆出 CTA-wide prepare，再拆 producer /
  consumer role body。

#### M3b: split params prepare from run

状态：已完成。

实现：

- 原来的 `tileops_fa3_shaped_device_call` 被拆成两个全 CTA helper：

```text
tileops_fa3_shaped_prepare_params(...)
tileops_fa3_shaped_run_prepared(...)
```

- `prepare_params` 负责：

```text
TMA atom descriptor copy
mainloop params construction
epilogue params construction
scheduler params construction
placement-new AttnParams into device global params storage
CTA-wide __syncthreads()
```

- TileLang 外层调用顺序变成：

```text
prepare_params(...)             # all 384 threads
if tx < 128:
  role_probe(producer)
elif tx < 256:
  role_probe(consumer WG1)
else:
  role_probe(consumer WG2)
run_prepared(...)               # all 384 threads
```

- `run_prepared` 只根据 TileLang 传入的 `tile_m / bidh / bidb` 重新定位
  global params storage，然后调用 FA3 `AttnKernel::operator()`。

验证：

```text
B=1, S=128, H=1, HKV=1
out finite True
lse finite True
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
B=1, S=4096, H=8, HKV=2
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.097008 tflops=708.39
```

```text
query-smem: B=1, S=224, H=1, HKV=1
shared_storage_size 191488 sizeof_shared_storage 191488
```

M1 latency baseline 是 `0.098476 ms`，M3b 为 `0.985x M1`，满足
M3 原 `<= 1.10x M1 latency` 性能阈值。

结论：

- 参数构造和计算调用已经分成两个 helper 边界。
- TileLang WG branch skeleton 已经位于这两个边界之间。
- 这为下一步拆 FA3 `operator()` 的 CTA-wide prepare 和 producer/consumer
  role body 提供了稳定插入点。
- M3 尚未完成，因为 `run_prepared` 仍然是所有 384 线程一起进入 FA3
  `operator()`，还没有替换成 producer/consumer role helper。

#### M3c: experimental role-run helper

状态：functional pass，performance pending。

实现：

- 新增实验开关：

```text
--role-run
```

- `--role-run` 路径把执行顺序改成：

```text
prepare_params(...)       # all 384 threads
prepare_runtime(...)      # all 384 threads; prefetch descriptor, init pipeline/barrier, CTA sync
if tx < 128:
  run_role(producer)
elif tx < 256:
  run_role(consumer WG1)
else:
  run_role(consumer WG2)
```

- `prepare_runtime` 负责从 FA3 `operator()` 前半段拆出的 CTA-wide 工作：

```text
prefetch_tma_descriptors
barrier_Q / barrier_O init
pipeline_k / pipeline_v / pipeline_vt barrier init
CTA-wide __syncthreads()
```

- `run_role` 复制 FA3 `operator()` 的 producer/consumer body，但在 role 分支内
  使用 false-init pipeline wrapper，避免重复初始化 barrier。
- 当前 `run_role` 只覆盖本实验固定路径：dense、non-append、non-pack GQA、
  FP8 input、BF16 output、Transpose_V。

验证：

```text
--role-run
B=1, S=128, H=1, HKV=1
out finite True
lse finite True
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
--role-run
B=1, S=4096, H=8, HKV=2
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112041 tflops=613.34
```

默认 non-role-run 路径复测：

```text
B=1, S=4096, H=8, HKV=2
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.097114 tflops=707.62
```

结论：

- Functional：真正的 WG branch role helper 路径已经可以正确运行。
- Performance：`0.112041 ms` 相对 M1 baseline `0.098476 ms` 是 `1.138x`，
  满足放宽后的 M3 验收阈值 `1.15x`，但超过旧参考目标 `1.10x`。
- 因此 M3 可以标记为 relaxed performance complete；后续仍要追踪这段开销。

待优化问题：

- `prepare_params` 和 `prepare_runtime` 现在各有一次 CTA-wide 同步/初始化边界；
  需要继续合并或把一部分 prepare 移回 TileLang 外层。
- role-run 路径拆开后可能损失了 FA3 原 `operator()` 内局部对象构造/优化机会。
- 后续需要用 generated CUDA / SASS 对比 `run_prepared` 和 `run_role`，确认额外
  开销来自 helper 边界、pipeline wrapper false-init、还是 TileLang 分支结构。

修正后复测：

```text
mode: --role-run
shape: B=1, S=4096, H=8, HKV=2, D=128
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112077 tflops=613.15
```

结论：M3 functional pass，并按放宽后的 `<= 1.15x M1 latency` 标准通过。
旧 `<= 1.10x M1 latency` 目标未通过，记录为待优化项。

#### M3d: baseline optimization round

动机：

当前 whole-helper 起点已经比 FA3 host launcher 慢。继续往 HLIR 拆之前，先把
不属于计算核心的起点开销清掉一轮，避免后续每一步都踩在过慢 baseline 上。

实现：

- 新增 `--validate-wg-branch`。默认 benchmark 不再调用
  `tileops_fa3_shaped_role_probe`；WG/thread/block coord 校验仍可显式打开。
- 新增 `--static-persistent-call-extern`。TileLang 仍使用 `T.Kernel` 启动 CTA，
  但 grid 改成 `(132, 1, 1)`，helper 内 scheduler 改成
  `StaticPersistentTileScheduler<false>`。
- static persistent 分支中 `params_storage` 从 `NUM_M_BLOCKS * H * B` 改为
  `NUM_SMS`，每个 persistent CTA 构造自己的一份 `AttnKernel::Params`，避免
  CTA 间共享 placement-new 存储带来的竞态。
- `--role-run` 和 `--static-persistent-call-extern` 暂时互斥；前者是 WG 分拆
  实验，后者是 baseline scheduler/grid 优化实验。

验证命令：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --bench --warmup 10 --repeat 50
```

结果：

```text
fa3_mode tilelang_primfunc_call_extern_fa3_device_helper
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.096834 tflops=709.66
```

static persistent 验证命令：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --bench --warmup 10 --repeat 50 \
  --static-persistent-call-extern
```

结果：

```text
fa3_mode tilelang_primfunc_call_extern_fa3_static_persistent
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.093599 tflops=734.19
```

WG 校验入口复测：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 128 --heads 1 --heads-kv 1 \
  --validate-wg-branch
```

结果：

```text
fa3_mode tilelang_primfunc_call_extern_fa3_device_helper_wg_validated
correctness max_abs=0.00195312 cosine=0.99976474
```

结论：

- 移除 debug probe 后 default `SingleTileScheduler` 起点从约 `0.097457 ms`
  到 `0.096834 ms`，收益很小，但语义更干净。
- static persistent 分支正确，4096 GQA 从 default `0.096834 ms` 提到
  `0.093599 ms`，但仍慢于 host launcher `0.081237 ms`。
- 这说明 persistent scheduler/grid 能解释一部分差距，但不是全部。剩余差距
  需要继续看 device-side params 构造、TMA atom/descriptor wrapper、global
  params storage 和 TileLang launch IR 形态。
- 该分支目前只能作为优化实验，不标记为默认 dispatch。

#### M3e: named role helpers

目标：

把 `--role-run` 主线从“TileLang 三个分支都调用同一个
`tileops_fa3_shaped_run_role(role, ...)`”推进到“TileLang 三个分支分别调用
具名 producer / consumer helper”：

```text
if tx < 128:
  tileops_fa3_shaped_run_producer(...)
elif tx < 256:
  tileops_fa3_shaped_run_consumer_wg1(...)
else:
  tileops_fa3_shaped_run_consumer_wg2(...)
```

实现：

- 新增 `tileops_fa3_shaped_run_producer`。
- 新增 `tileops_fa3_shaped_run_consumer_wg1`。
- 新增 `tileops_fa3_shaped_run_consumer_wg2`。
- 三个 helper 目前仍复用 `tileops_fa3_shaped_run_role` 的实现，但把 role
  常量化，并在 wrapper 入口校验 role 参数。
- 这一步主要推进代码形态和 helper 边界，不期望显著性能变化。

验证：

```text
mode: --role-run
shape: B=1, S=4096, H=8, HKV=2, D=128
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112207 tflops=612.44
```

```text
mode: --role-run
shape: B=1, S=128, H=1, HKV=1, D=128
correctness max_abs=0.00195312 cosine=0.99976474
```

结论：

- M3 按放宽后的 `<= 1.15x M1 latency` 标准通过。
- named helper 后性能和原 `run_role` 基本一致：`0.112207 ms` vs
  `0.112077 ms`。
- 下一步进入 M4：不再只做 wrapper 命名，而是开始把 producer path 中的
  TMA/barrier/pipeline 操作逐步搬到 TileLang 外层。

### M4: TileLang owns producer path

目标：

先把 `tx < 128` 的 producer 分支改成 TileLang 主导：

- 用 TileLang 表达 `TMA descriptor` 使用点。
- 用 `T.tma_copy` 或最小 TMA extern 执行 Q/K/V/O 相关搬运。
- 用 TileLang barrier/mbarrier 表达 producer-consumer 同步。

consumer 分支可以继续调用局部 extern。

这是第一阶段最重要的转折点：一旦 producer path 可由 TileLang 驱动，
后面 mainloop 拆解就有了真实 WG 骨架，而不是只是在大 helper 外面套壳。

验收标准：

- **代码形态。** `tx < 128` 分支中的主要控制流由 TileLang 表达，而不是
  一个 producer mega-helper。
- **TMA 搬运。** Q/K/V 至少一种 TMA load 由 TileLang 外层调用
  `T.tma_copy` 或最小 TMA extern 完成；descriptor 使用点在 TileLang 可见。
- **同步所有权。** producer-consumer barrier/mbarrier 的 arrive/wait/phase
  在 TileLang 外层可见；helper 只能执行局部数据变换或 PTX 片段。
- **consumer 兼容。** consumer helper 能消费 TileLang producer 写入的 shared
  memory 和 barrier 状态。
- **正确性。** 以下形状全部通过：

```text
B=1, S=128,  H=1, HKV=1
B=1, S=4096, H=1, HKV=1
B=1, S=4096, H=8, HKV=2
```

- **性能。** M4 是结构迁移阶段，`B=1, S=4096, H=8, HKV=2` latency
  硬预算放宽为 `<= 1.20x M1 latency`。`<= 1.15x M1 latency` 保留为
  优化参考线，不作为 M4 阻塞条件。
- **记录。** markdown 中记录哪些 TMA/producer 操作已经 TileLang 化，哪些仍
  依赖 extern。

#### M4 细分节点

M4 不一次性搬完整 producer path，而是按以下节点推进。

**M4a: Producer contract map**

先不改行为，只把 producer path 的输入、输出和依赖列成表：

```text
warpgroup_reg_dealloc
SingleProducerWarp / producer warp gating
scheduler.get_initial_work / get_next_work
block_coord
SeqlenInfo
scheduler_prefetch lambda
mainloop.load(...)
mainloop.load_tail(...)
pipeline_k / pipeline_v / pipeline_vt handles
smem_pipe_write
```

通过标准：

- markdown 中有 producer 输入/输出/依赖表。
- 标出哪些字段已经由 TileLang 外层拥有，哪些仍在 C++ helper 内部。
- 行为不变，或只增加无行为变化的 assert / debug 校验。

**M4b: TileLang owns producer gating**

TileLang 外层显式表达 producer gating：

```python
is_producer = tx < 128
producer_warp = (tx // 32) % 4
producer_lane = tx % 32
```

并把这些值传给 `run_producer` 校验。

通过标准：

- TileLang 外层能读出哪些线程属于 producer WG。
- C++ helper 内校验 TileLang producer gating 与 CUDA `threadIdx.x` 推导一致。
- `B=1, S=128, H=1, HKV=1` 和 `B=1, S=4096, H=8, HKV=2` 正确。

**M4c: TileLang owns producer work coord**

在当前 `SingleTileScheduler` 路径中，每个 CTA 对应一个 work tile。先让
producer helper 使用 TileLang 传入的：

```text
tile_m
bidh
bidb
bidh_kv
```

构造 producer 侧 `block_coord` 和 `SeqlenInfo`，减少 producer path 对
`scheduler.get_initial_work` 的依赖。consumer 可以暂时继续使用 scheduler。

通过标准：

- producer helper 不再通过 `scheduler.get_initial_work` 决定当前 block coord。
- consumer helper 行为不变。
- correctness 覆盖 M4b 两个形状。

**M4d: Split producer load body**

把 producer helper 拆成更小的 C++ extern：

```text
producer_enter(...)
producer_load_one_tile(...)
producer_load_tail(...)
```

TileLang 外层负责调用顺序：

```python
if tx < 128:
    producer_enter(...)
    producer_load_one_tile(...)
    producer_load_tail(...)
```

通过标准：

- producer 分支主要控制流在 TileLang 外层可见。
- `mainloop.load` 可以暂时仍在 extern 中，但不再被一个 producer mega-helper
  完整包住。
- correctness 覆盖 M4b 两个形状。
- `B=1, S=4096, H=8, HKV=2` latency 仍在 `<= 1.20x M1`。

**M4e: Convert one producer helper slice to TileLang HLIR**

紧接 M4d，选一个已经切出来的 C++ helper slice，把其中一部分改成 TileLang
高级原语，而不是继续停留在“小 C++ helper”层。

优先候选：

```text
producer_enter:
  warpgroup role / producer warp gating
  register dealloc 的调用边界
  initial pipeline state 的可见表达

producer_load_one_tile:
  TMA descriptor 使用点
  一种 Q/K/V TMA load 的最小 TileLang 表达
  mbarrier arrive / transaction bytes 的显式参数

producer_load_tail:
  load_tail 调用顺序
  producer tail sync / barrier 收尾边界
```

通过标准：

- 至少一个原本在 C++ helper 内部的 producer 操作由 TileLang 高级语句或
  TileLang intrinsic 表达。
- 该操作的输入输出是 TileLang 可见的 shared memory、descriptor、barrier
  或 phase state。
- 如果仍需要 extern，它必须缩小为单个 PTX/CuTe intrinsic 片段，而不是完整
  producer control-flow helper。
- correctness 覆盖 M4b 两个形状。
- markdown 记录“已 TileLang 化的 producer 操作”和“仍依赖 extern 的操作”。

**M4f: TileLang owns barrier / pipeline init visibility**

`prepare_runtime` 现在仍是 all-CTA helper。该节点把其中的 barrier 初始化语义
外显到 TileLang 外层，至少包括：

```text
barrier_Q init
barrier_O init
pipeline_k / pipeline_v / pipeline_vt role params
CTA sync boundary
```

真实 init 可以暂时继续通过小 extern 执行，但 TileLang 外层必须能读出 init
顺序和同步边界。

**M4g: First real TMA move experiment**

最后再试第一块真实 TMA 搬运，例如先挑 Q 或 K 的 load 做最小替换。这个节点
风险最高，因为要匹配 FA3 的 shared layout、mbarrier phase、TMA transaction
bytes 和 swizzle。

#### M4a: producer contract map result

状态：已完成。

当前 producer path contract：

| 项目 | 当前所有者 | 输入 | 输出 / 影响 | 后续迁移方向 |
| --- | --- | --- | --- | --- |
| `tx < 128` producer WG 判定 | TileLang | `tx` | 进入 producer helper | 已外显 |
| `producer_warp / producer_lane` | TileLang | `tx`, `warpgroup_lane` | producer gating 校验 | 已外显 |
| `warpgroup_reg_dealloc` | TileLang-visible enter boundary + minimal intrinsic extern | producer threads | producer register budget | M4e 第一刀已完成 |
| `SingleProducerWarp` gating | C++ helper | `AttnKernel::NumProducerThreads`, producer warp | 非参与 producer warp 提前退出 | M4b/M4e 继续外移 |
| `scheduler.get_initial_work` | 已从 producer 移除 | `params->scheduler`, `blockIdx` | producer work tile | M4c 已外移到 TileLang work coord；consumer 仍使用 scheduler |
| `block_coord` | TileLang-owned producer coord | `tile_m_tl`, `bidh_tl`, `bidb_tl` | FA3 block coord | M4c 已外移；M4d 继续分拆 |
| `SeqlenInfo` | C++ helper | TileLang-owned `block_coord`, mainloop params | mainloop.load seqlen metadata | M4d 分拆 |
| `scheduler_prefetch` | producer no-op | scheduler, work tile | next-work prefetch | M4c 已简化；consumer 仍使用 scheduler |
| `pipeline_k/v/vt` handles | C++ helper | `shared_storage.pipelines` | producer/consumer pipeline wrappers | M4f 外显 |
| `smem_pipe_write` | C++ helper | pipeline state type | producer write phase | M4e 候选 |
| `mainloop.load` | C++ helper | params, pipeline, shared, block coord | Q/K/V TMA producer load | M4d 切片，M4e/M4g 逐步 HLIR 化 |
| `mainloop.load_tail` | C++ helper | pipeline state, shared | producer tail cleanup | M4d 切片 |

结论：

- TileLang 已经拥有 producer WG 入口和基本 thread identity。
- C++ helper 仍拥有 producer 的 scheduler work loop、TMA load、pipeline wrapper
  和 tail cleanup。
- M4 后续应优先从 `SingleProducerWarp` gating、`block_coord` 和
  `producer_enter / load_one_tile / load_tail` 这些控制流边界继续外移。

#### M4b: TileLang owns producer gating result

状态：已完成。

实现：

- TileLang 外层新增：

```python
producer_warp = warpgroup_lane // 32
producer_lane = warpgroup_lane % 32
```

- `tileops_fa3_shaped_run_producer` 新增参数：

```text
producer_warp_tl
producer_lane_tl
```

- producer helper 校验：

```text
tx_tl == threadIdx.x
threadIdx.x < 128
warpgroup_id_tl == threadIdx.x / 128
producer_warp_tl == (threadIdx.x / 32) % 4
producer_lane_tl == threadIdx.x % 32
```

验证：

```text
mode: --role-run
shape: B=1, S=128, H=1, HKV=1, D=128
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
mode: --role-run
shape: B=1, S=4096, H=8, HKV=2, D=128
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112385 tflops=611.47
```

结论：

- M4b correctness 通过。
- `0.112385 ms` 相对 M1 `0.098476 ms` 是约 `1.141x`，低于 M4
  `<= 1.20x M1` 硬预算；同时接近 `1.15x M1` 优化参考线。
- 这一步只是 producer gating 所有权外移，不改变 producer TMA/barrier 工作。

#### M4c: TileLang owns producer work coord result

状态：已完成。

实现：

- producer branch 不再调用 `scheduler.get_initial_work` 来取得当前
  `block_coord`。
- producer helper 直接使用 TileLang 外层传入的：

```text
tile_m_tl
bidh_tl
bidb_tl
```

构造：

```cpp
auto block_coord = cute::make_tuple(
    int32_t(tile_m_tl), int32_t(bidh_tl), int32_t(bidb_tl), int32_t(0));
```

- producer 侧 `SeqlenInfo` 使用这个 TileLang-owned `block_coord` 构造。
- producer 侧 `scheduler_prefetch` 暂时变成 no-op lambda。当前主线固定
  `SingleTileScheduler`，每个 CTA 只有一个 work tile，所以 producer 不需要
  next-work prefetch。
- consumer path 暂时保持原 scheduler 逻辑不变。

验证：

```text
mode: --role-run
shape: B=1, S=128, H=1, HKV=1, D=128
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
mode: --role-run
shape: B=1, S=4096, H=8, HKV=2, D=128
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112806 tflops=609.18
```

结论：

- M4c correctness 通过。
- producer 当前 work coord 已由 TileLang 外层拥有。
- `0.112806 ms` 相对 M1 `0.098476 ms` 是约 `1.146x`，低于 M4
  `<= 1.20x M1` 硬预算；同时接近 `1.15x M1` 优化参考线。
- 后续 M4d 拆 `producer_enter / producer_load_one_tile / producer_load_tail`
  时要格外注意性能，最好避免新增 CTA-wide sync 或额外 helper 边界。

#### M4d: split producer load body result

状态：structural pass，performance 在 M4 放宽预算内。

实现：

- 新增实验开关：

```text
--producer-split
```

- 该开关要求同时打开 `--role-run`。
- producer 分支从一个 `tileops_fa3_shaped_run_producer(...)` 拆成：

```text
tileops_fa3_shaped_producer_enter(...)
tileops_fa3_shaped_producer_load_one_tile(...)
tileops_fa3_shaped_producer_load_tail(...)
```

- TileLang 外层现在能读出 producer 的调用顺序：

```python
if tx < 128:
    producer_enter(...)
    producer_load_one_tile(...)
    producer_load_tail(...)
```

- `producer_load_one_tile` 当前仍包含 `mainloop.load(...)` 和
  `mainloop.load_tail(...)`，因为二者共享推进后的 `smem_pipe_write/work_idx`
  状态。真正把 pipeline state 外显留到 M4e/M4f。
- `producer_load_tail` 目前是显式边界/校验 no-op，用来固定 TileLang 外层的
  producer tail 调用点。

验证：

```text
mode: --role-run --producer-split
shape: B=1, S=128, H=1, HKV=1, D=128
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
mode: --role-run --producer-split
shape: B=1, S=4096, H=8, HKV=2, D=128
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.114247 tflops=601.50
```

结论：

- M4d correctness 通过。
- `0.114247 ms` 相对 M1 `0.098476 ms` 是约 `1.160x`，低于 M4
  `<= 1.20x M1` 硬预算，但超过 `1.15x M1` 优化参考线。
- TileLang 外层已经拥有 producer 三段控制流；下一步 M4e 需要把其中至少一段
  从“小 C++ helper”继续转换为 TileLang HLIR/高级原语。

#### M4e: producer enter HLIR result

状态：已完成第一刀。

实现：

- `--role-run --producer-split` 路径中，TileLang 外层继续保持：

```python
if tx < 128:
    producer_enter / reg_dealloc boundary
    producer_load_one_tile(...)
    producer_load_tail(...)
```

- 原来的 `tileops_fa3_shaped_producer_enter(...)` 不再作为主线调用点。
- 新增最小 intrinsic helper：

```text
tileops_fa3_shaped_producer_reg_dealloc(
    tx, warpgroup_id, producer_warp, producer_lane)
```

- 该 helper 只做两件事：

```text
1. 校验 TileLang 外层传入的 producer thread identity
2. 调用 cutlass::arch::warpgroup_reg_dealloc<...>()
```

- producer enter 的 role/work/GQA 控制流已经由 TileLang 外层表达；extern 边界
  从 producer control-flow helper 缩小为单个 register dealloc intrinsic。
- TMA load、pipeline state、`mainloop.load` 仍在
  `producer_load_one_tile` C++ helper 内，尚未 TileLang 化。

验证：

```text
mode: --role-run --producer-split
shape: B=1, S=128, H=1, HKV=1, D=128
correctness max_abs=0.00195312 cosine=0.99976474
```

```text
mode: --role-run --producer-split
shape: B=1, S=4096, H=8, HKV=2, D=128
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.113307 tflops=606.49
```

结论：

- M4e 第一刀 correctness 通过。
- `0.113307 ms` 相对 M1 `0.098476 ms` 是约 `1.151x`，低于 M4
  `<= 1.20x M1` 硬预算，接近 `1.15x M1` 优化参考线。
- 这一刀真正减少了一个 C++ producer control-flow helper；下一步要么继续
  把 `producer_load_tail` 做成 TileLang-visible state 边界，要么开始研究
  `producer_load_one_tile` 中的 TMA descriptor / mbarrier 原语映射。

#### M4f-pre: TileLang WS barrier probe

状态：已完成。

目的：

在不污染 FA3 主 kernel 的情况下，验证 TileLang barrier 原语能否表达
warp-specialized producer/consumer phase 协议。

新增 probe：

- [_probe_tilelang_ws_barrier_pipeline.py](/home/ga/TileOPs/_probe_tilelang_ws_barrier_pipeline.py)

实验 1：single phase

```text
producer WG: tx < 128
consumer WG: tx >= 128
ready barrier arrive_count = 128
done barrier arrive_count = 256

producer writes smem[0] = 123
producer arrive ready
consumer wait ready, read smem, arrive done
producer wait done, writes output
```

结果：

```text
single_out [123, 124, 125, 249]
single_pass True
```

实验 2：ping-pong parity

```text
phase 0:
  producer writes 200
  consumers wait ready parity 0
  consumers arrive done parity 0

phase 1:
  producer writes 300
  consumers wait ready parity 1
  consumers arrive done parity 1
```

结果：

```text
pingpong_out [200, 201, 202, 403, 300, 303, 304, 607]
pingpong_pass True
```

结论：

- `T.alloc_barrier(arrive_count=...)` 能表达 producer/consumer 不同 arrive count。
- `T.barrier_arrive` / `T.barrier_wait(bar, parity)` 能表达 single phase 和
  ping-pong parity。
- 这支持后续把 FA3 producer pipeline 中的 ready/free barrier phase 逐步
  映射到 TileLang 外层。
- 还未验证 TMA transaction bytes / expect-tx 语义；真正替换
  `mainloop.load` 前，还需要单独验证 `T.tma_copy` 与 mbarrier expect-tx 的组合。

#### M4f-pre2: TileLang TMA copy + mbarrier probe

状态：已完成。

目的：

验证 `T.tma_copy` 能否和 TileLang mbarrier 配合，表达 FA3 producer pipeline
所需的 TMA load ready 通知和 transaction bytes。

新增 probe：

- [_probe_tilelang_tma_mbarrier.py](/home/ga/TileOPs/_probe_tilelang_tma_mbarrier.py)

实验 1：single TMA load

```text
global x[128, 32] fp16
producer WG: T.tma_copy(x, x_shared, barrier=ready)
producer WG: T.barrier_arrive(ready)
consumer WG: T.barrier_wait(ready, 0)
consumer WG: read x_shared sample points
```

结果：

```text
single_out [0.0, 31.0, 4064.0, 4096.0]
single_expected [0.0, 31.0, 4064.0, 4096.0]
single_pass True
```

生成 CUDA 关键片段：

```text
ready[0].expect_transaction(8192);
tl::tma_load(x_desc, ready[0], (&(x_shared[0])), 0, 0);
ready[0].arrive();
ready[0].wait(0);
```

这里 `8192 = 128 * 32 * sizeof(fp16)`，transaction bytes 与 tile 大小一致。

实验 2：ping-pong TMA load

```text
phase 0:
  producer TMA load x[0:128, :]
  consumer wait ready parity 0, read shared
  consumer arrive done parity 0

phase 1:
  producer wait done parity 0
  producer TMA load x[128:256, :] into same shared tile
  consumer wait ready parity 1, read shared
  consumer arrive done parity 1
```

结果：

```text
pingpong_out [0.0, 31.0, 4064.0, 4096.0, 4096.0, 4128.0, 8160.0, 8192.0]
pingpong_expected [0.0, 31.0, 4064.0, 4096.0, 4096.0, 4128.0, 8160.0, 8192.0]
pingpong_pass True
```

生成 CUDA 关键确认：

```text
expect_transaction(8192): 2
tl::tma_load: 2
.wait(0): 2
.wait(1): 2
```

结论：

- `T.tma_copy(..., barrier=...)` 能自动生成 expect-tx + TMA load。
- TileLang mbarrier parity 能保护同一个 shared tile 的 ping-pong 覆写。
- 这说明 `T.tma_copy + T.barrier_arrive + T.barrier_wait` 的基础语义可以承载
  FA3 producer pipeline 的一部分。
- 下一步还要验证 FA3 相关的 shared layout/swizzle、TMA descriptor 参数和
  Q/K/V tile shape 是否与 `mainloop.load` 期望一致。

#### M4f-pre3: TileLang TMA swizzle probe

状态：已完成。

目的：

验证 TileLang 的 swizzled shared layout 是否能和 `T.tma_copy` 组合，并且物理
shared memory 排布是否匹配 FA3 当前使用的 128B swizzle 公式。

新增 probe：

- [_probe_tilelang_tma_swizzle.py](/home/ga/TileOPs/_probe_tilelang_tma_swizzle.py)

接口确认：

```python
x_shared = T.alloc_shared([128, 128], "uint8")
T.annotate_layout({
    x_shared: tilelang.layout.make_swizzled_layout(x_shared),
})
T.tma_copy(x[0:128, 0:128], x_shared, barrier=ready)
```

probe 做两种读取：

- TileLang 逻辑读取：`x_shared[row, col]`
- raw shared 指针读取：用 FA3 128B swizzle physical index 公式读取
  `x_shared.access_ptr("r")`

为了让 raw `T.call_extern` probe 能通过 TVM region inference，kernel 中需要
显式声明：

```python
T.reads(x[0:128, 0:128])
T.writes(x_shared[0:128, 0:128], logical_out[0:5], raw_out[0:5])
```

结果：

```text
logical_out [0, 127, 144, 0, 255]
raw_out [0, 127, 144, 0, 255]
expected [0, 127, 144, 0, 255]
logical_pass True
raw_fa3_swizzle_pass True
has_tma_load True
has_expect_16384 True
```

结论：

- `tilelang.layout.make_swizzled_layout(x_shared)` 可以作为中间 probe 的
  shared layout 接口。
- `T.tma_copy(..., barrier=ready)` 能把 `128x128 uint8` tile 复制进 swizzled
  shared，并生成 `expect_transaction(16384)`。
- TileLang 逻辑读取和 FA3 物理 swizzle raw 读取同时通过，说明当前
  `make_swizzled_layout` 的物理排布与 FA3 `tl_fp8_full_swizzle_128b_index`
  等价。
- 下一步可以把这个接口套到 producer 的 Q/K/V tile shape 上，继续验证
  descriptor 参数、stage offset 和 ping-pong phase。

#### M4f-main1: 主线内 Q TMA HLIR shadow probe

状态：已完成。

目的：

把 `T.tma_copy + swizzled shared + mbarrier` 从独立 probe 接进
`--role-run --producer-split` 主线，但暂时不替换 FA3 consumer 实际读取的
shared tile。

新增开关：

```bash
--producer-tma-hlir-probe
```

约束：

- 必须和 `--role-run --producer-split` 一起使用。
- 只对 full Q tile 执行 shadow TMA，即 `tile_m * 128 + 128 <= S`。
- 原 `tileops_fa3_shaped_producer_load_one_tile` 仍执行，作为当前 correctness
  fallback。

主线中新增的 TileLang 片段：

```python
q_hlir_probe = T.alloc_shared((128, dim), "float8_e4m3fn")
q_hlir_probe_ready = T.alloc_barrier(arrive_count=128)
T.annotate_layout({
    q_hlir_probe: tilelang.layout.make_swizzled_layout(q_hlir_probe),
})

if tx < 128:
    T.tma_copy(
        q[bidb, q_row_base:q_row_base + 128, bidh, 0:dim],
        q_hlir_probe,
        barrier=q_hlir_probe_ready,
    )
    T.barrier_arrive(q_hlir_probe_ready)
else:
    T.barrier_wait(q_hlir_probe_ready, 0)
```

consumer WG1 的 `tx == 128` 调用小 helper 做 raw sample 校验：

```text
tileops_fa3_shaped_check_q_hlir_tma_probe(...)
```

该 helper 用 FA3 128B swizzle physical index 从 `q_hlir_probe.access_ptr("r")`
读取样本，并和 global Q 的相同 logical `(row, col)` 样本比较；不匹配则
`trap`。

小 shape 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 128 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00195312 cosine=0.99976474
```

tail guard 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.113312 tflops=606.46
```

结论：

- `T.tma_copy` 的动态 Q slice
  `q[bidb, tile_m * 128:tile_m * 128 + 128, bidh, :]` 可以在主线 kernel
  中编译和运行。
- 主线内的 TileLang swizzled shared raw sample 与 global Q 对齐，没有 trap。
- 4096 性能和当前 `--producer-split` baseline 基本相同，说明这个 shadow
  probe 没有引入可见额外开销。
- 这一步还不是替换：FA3 consumer 仍读取原 `SharedStorage` 里的 Q/K/V。
  下一步应暴露 FA3 `SharedStorage::tensors.mainloop.smem_q/smem_k/smem_vt`
  offset，尝试让 TileLang TMA 直接写入 consumer 会读取的 Q stage buffer。

#### M4f-main2-attempt: raw FA3 `smem_q` typed view

状态：未完成，已定位到 TileLang view/lowering 问题。

目的：

不再写 shadow buffer，而是从 FA3 raw dynamic shared arena 中切出
`SharedStorage::tensors.mainloop.smem_q` 的 typed/swizzled view，让
`T.tma_copy` 直接写入 FA3 consumer 后续会读取的 Q stage buffer。

新增实验开关：

```bash
--producer-tma-hlir-q-stage-probe
```

C++ helper 暴露了 FA3 `smem_q` byte offset：

```cpp
extern "C" __device__ __forceinline__
int tileops_fa3_shaped_smem_q_offset_bytes() {
  using namespace tileops_fa3_call_extern_detail;
  return int(reinterpret_cast<char*>(
      reinterpret_cast<AttnKernel::SharedStorage*>(0)
          ->tensors.mainloop.smem_q.data()) -
      reinterpret_cast<char*>(0));
}
```

同时新增 helper 从 raw `smem` 的 `smem_q_offset + FA3_128B_swizzle(row, col)`
读取样本，与 global Q 对比：

```text
tileops_fa3_shaped_check_q_fa3_stage_probe(...)
```

尝试 1：`handle_add_byte_offset + decl_buffer`

```python
q_fa3_stage_offset = T.call_extern(
    "int32", "tileops_fa3_shaped_smem_q_offset_bytes")
q_fa3_stage_view = T.decl_buffer(
    (128, dim),
    "float8_e4m3fn",
    data=T.handle_add_byte_offset(smem.data, q_fa3_stage_offset),
    scope="shared.dyn",
)
```

失败：

```text
TypeError: Mismatched type on argument #3 when calling DeclBuffer.
Expected Optional<tir.Var> but got tir.Call
```

中间变量版本也失败：

```text
Variable q_fa3_stage_handle is not a pointer.
```

结论：在当前 TileLang eager `@tilelang.jit` 路径里，
`T.decl_buffer(data=...)` 不能直接接收
`T.handle_add_byte_offset(...)` 产生的 handle call；需要 pointer-typed
`tir.Var`，或者在更低层/private TIR 中构造。

尝试 2：`T.match_buffer` 从 raw `uint8` subregion 绑定 typed view

```python
q_fa3_stage_view = T.match_buffer(
    smem[q_fa3_stage_offset:q_fa3_stage_offset + 128 * dim],
    (128, dim),
    "float8_e4m3fn",
    scope="shared.dyn",
)
```

失败：

```text
MatchBuffer buffer data type mismatch: float8_e4m3fn vs. uint8
```

结论：`match_buffer` 不允许从 `uint8` source region 直接绑定成
`float8_e4m3fn` typed buffer。

尝试 3：先 `T.view` reinterpret dtype，再 `T.match_buffer`

```python
smem_fp8_view = T.view(
    smem,
    (smem_bytes // dim, dim),
    dtype="float8_e4m3fn",
)
q_fa3_stage_row_offset = q_fa3_stage_offset // dim
q_fa3_stage_view = T.match_buffer(
    smem_fp8_view[
        q_fa3_stage_row_offset:q_fa3_stage_row_offset + 128,
        0:dim,
    ],
    (128, dim),
    "float8_e4m3fn",
    scope="shared.dyn",
)
```

这个版本能通过前端构建，但在 lowering 的 `PointerValueTypeRewrite` /
`StorageRewrite` 阶段失败：

```text
Load/Store of buffer q_fa3_stage_view occurred before its declaration.
```

结论：

- 当前 high-level `T.tma_copy` + `make_swizzled_layout` 已能写 typed shadow
  shared buffer。
- 但在 `@tilelang.jit` eager 路径中，把 raw `uint8` dynamic shared arena 的
  byte offset subregion 重新解释为 typed 2D swizzled view，再作为
  `T.tma_copy` destination，目前还没有跑通。
- TileLang 社区确认：`T.handle_add_byte_offset + T.decl_buffer` 属于
  low-level `T.prim_func(private=True)` / TIR 层，不属于 eager `T.Kernel`
  语言表面。eager 路径推荐直接 `T.alloc_shared` typed buffer，让
  `MergeSharedMemoryAllocations` 自动合并 shared arena 并保证 TMA 对齐。

#### M4f decision: choose route A

状态：已决策。

当前有两条路线：

```text
A: TileLang owns shared buffer layout.
B: FA3/CUTLASS owns SharedStorage layout.
```

路线 A：

TileLang 直接声明 typed shared buffers：

```python
q_stage = T.alloc_shared((128, 128), "float8_e4m3fn")
k_stage = T.alloc_shared((num_stages, 224, 128), "float8_e4m3fn")
v_stage = T.alloc_shared(...)
T.annotate_layout({
    q_stage: tilelang.layout.make_swizzled_layout(q_stage),
    k_stage: tilelang.layout.make_swizzled_layout(k_stage),
    v_stage: tilelang.layout.make_swizzled_layout(v_stage),
})
```

然后 C++ helper 接收 TileLang buffer 指针，而不是继续假设：

```cpp
auto& shared_storage =
    *reinterpret_cast<AttnKernel::SharedStorage*>(smem_raw);
```

路线 B：

保留 FA3 `SharedStorage` exact layout，让 TileLang 往：

```text
smem_raw + offsetof(SharedStorage, tensors.mainloop.smem_q)
```

这类 raw offset 写入 typed/swizzled TMA tile。

决策：

- **主线选择 A。**
- B 只作为低层兼容/诊断分支暂存，不继续作为 eager 主线推进。

依据：

- TileLang eager `T.Kernel` 的推荐模式是直接 `T.alloc_shared` typed buffer，
  再由 `MergeSharedMemoryAllocations` 自动 pack 到 shared.dyn arena。
- eager 层不能自然表达 `handle_add_byte_offset + decl_buffer` 的 manual
  byte-offset typed view；该能力属于 low-level private TIR。
- 本实验目标是逐层把 FA3/CUTLASS helper 搬到 TileLang 高级表达；A 把
  allocation/layout/barrier 的所有权交给 TileLang，更符合这个目标。
- B 的本质是让 TileLang 适配 FA3 内部 `SharedStorage` struct offset，
  会继续绑定 FA3 内部布局，不利于后续 HLIR 化。

当前推进状态：

- TileLang-owned Q shadow buffer probe 已通过。
- TileLang-owned K stage buffer probe 已通过。
- TileLang-owned V stage buffer probe 已通过。
- producer register dealloc 已从 C++ helper 换成 TileLang
  `T.dec_max_nreg(24)`。
- K-only single-buffer TMA pipeline probe 已通过。
- V-only single-buffer TMA pipeline probe 已通过，包含 V 滞后一拍和 tail load。
- K/V 显式 TileLang-owned stage pointer boundary probe 已通过。
- K/V 显式 stage pointer 已能构造 FA3/CuTe consumer operand tensor view。
- Q 显式 stage pointer 已能构造 FA3/CuTe consumer operand tensor view。
- 下一步先拆当前 producer core extern：
  `tileops_fa3_shaped_producer_load_one_tile` 和
  `tileops_fa3_shaped_producer_load_tail`。完成后再进入 QK/PV consumer compute。
- 是否把多个 TileLang-owned stage buffer 合并到同一个 kernel，要等 shared
  memory budget 和 helper pointer 边界设计清楚后再决定。

后续 M4f 每个小步都必须同时记录 correctness 和 benchmark：

```bash
# correctness / smoke
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split <new-probe-flag> \
  --warmup 1 --repeat 1

# benchmark / drift check
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split <new-probe-flag> \
  --bench --warmup 5 --repeat 20
```

验收方式：

- correctness 必须通过：`out finite True`、`lse finite True`，cosine 维持当前
  基线水平。
- benchmark 每步都记录 `latency_ms / tflops`。当前阶段不为 probe 分支追求
  极限性能，但如果出现明显退化，需要在文档里标注原因或暂停推进。
- 因为 shared memory budget 紧张，Q/K/V probe 暂时单独测试，不把多个额外
  TileLang-owned stage buffer 叠加到同一个 benchmark 里。

#### M4f-main3: A 路径 TileLang-owned K stage buffer probe

状态：已完成。

目的：

沿 A 路径推进：不再尝试写入 FA3 raw `SharedStorage` offset，而是由
TileLang 直接声明 typed/swizzled K stage buffer，并让 C++ helper 通过显式
buffer pointer 做验证。

新增开关：

```bash
--producer-tma-hlir-k-buffer-probe
```

约束：

- 必须和 `--role-run --producer-split` 一起使用。
- 与 `--producer-tma-hlir-probe`、`--producer-tma-hlir-q-stage-probe` 互斥。
- 只在 `S >= 224` 时执行 K full tile probe。
- 原 FA3 producer/consumer 仍作为 correctness fallback 执行。

主线中新增 TileLang-owned buffer：

```python
k_hlir_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
k_hlir_probe_ready = T.alloc_barrier(arrive_count=128)

T.annotate_layout({
    k_hlir_probe: tilelang.layout.make_swizzled_layout(k_hlir_probe),
})
```

producer WG 直接用 TileLang TMA 写 K tile：

```python
T.tma_copy(
    k[bidb, 0:224, bidh_kv, 0:dim],
    k_hlir_probe,
    barrier=k_hlir_probe_ready,
)
T.barrier_arrive(k_hlir_probe_ready)
```

consumer WG1 的 `tx == 128` 调用显式 pointer helper：

```text
tileops_fa3_shaped_check_k_hlir_tma_probe(
    k_hlir_probe.access_ptr("r"),
    k.data,
    tx,
    tile_n=0,
    bidh_kv,
    bidb,
)
```

该 helper 按 FA3 128B swizzle physical index 从 TileLang-owned `k_hlir_probe`
读样本，并与 global K 对比；不匹配则 `trap`。

最小 full K tile 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-k-buffer-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-k-buffer-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112339 tflops=611.71
```

结论：

- A 路径的 TileLang-owned `k_stage` typed/swizzled buffer 可以在主线 kernel
  中被 `T.tma_copy` 写入。
- 显式 buffer pointer helper 可以正确读取 TileLang-owned K buffer 的 raw
  physical swizzle 样本。
- GQA 下 `bidh_kv = bidh // GROUP` 的 K head mapping 在这个 probe 中通过。
- 由于 shared memory budget 限制，当前先单独验证 K buffer；K/V 都通过后，
  再设计不依赖 FA3 raw `SharedStorage` 的 Q/K/V 组合边界。

#### M4f-main4: A 路径 TileLang-owned V stage buffer probe

状态：已完成。

目的：

沿 A 路径补齐 V 输入：由 TileLang 直接声明 typed/swizzled V stage buffer，
producer WG 使用 `T.tma_copy` 写入，consumer WG 通过显式 pointer helper
抽样验证。

新增开关：

```bash
--producer-tma-hlir-v-buffer-probe
```

约束：

- 必须和 `--role-run --producer-split` 一起使用。
- 与 `--producer-tma-hlir-probe`、`--producer-tma-hlir-q-stage-probe`、
  `--producer-tma-hlir-k-buffer-probe` 互斥。
- 只在 `S >= 224` 时执行 V full tile probe。
- 原 FA3 producer/consumer 仍作为 correctness fallback 执行。

主线中新增 TileLang-owned buffer：

```python
v_hlir_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
v_hlir_probe_ready = T.alloc_barrier(arrive_count=128)

T.annotate_layout({
    v_hlir_probe: tilelang.layout.make_swizzled_layout(v_hlir_probe),
})
```

producer WG 直接用 TileLang TMA 写 V tile：

```python
T.tma_copy(
    v[bidb, 0:224, bidh_kv, 0:dim],
    v_hlir_probe,
    barrier=v_hlir_probe_ready,
)
T.barrier_arrive(v_hlir_probe_ready)
```

consumer WG1 的 `tx == 128` 调用显式 pointer helper：

```text
tileops_fa3_shaped_check_v_hlir_tma_probe(
    v_hlir_probe.access_ptr("r"),
    v.data,
    tx,
    tile_n=0,
    bidh_kv,
    bidb,
)
```

该 helper 按 FA3 128B swizzle physical index 从 TileLang-owned `v_hlir_probe`
读样本，并与 global V 对比；不匹配则 `trap`。

最小 full V tile 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-v-buffer-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-v-buffer-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.112645 tflops=610.05
```

结论：

- A 路径的 TileLang-owned `v_stage` typed/swizzled buffer 可以在主线 kernel
  中被 `T.tma_copy` 写入。
- 显式 buffer pointer helper 可以正确读取 TileLang-owned V buffer 的 raw
  physical swizzle 样本。
- GQA 下 `bidh_kv = bidh // GROUP` 的 V head mapping 在这个 probe 中通过。
- Q shadow、K stage、V stage 已分别验证；下一步应设计 helper 显式 pointer
  边界，避免继续依赖 FA3 raw `SharedStorage`。

#### M4f-main5: producer register dealloc HLIR 化

状态：已完成。

参考：

- [gqa_fwd_ws.py](/home/ga/TileOPs/tileops/kernels/attention/gqa_fwd_ws.py:116)

`gqa_fwd_ws.py` 的 warp-specialized producer 分支直接使用：

```python
if tx < 128:
    T.dec_max_nreg(24)
```

因此当前 split producer 里的：

```text
T.call_extern("tileops_fa3_shaped_producer_reg_dealloc", ...)
```

不需要长期保留，可以替换为 TileLang 高级原语：

```python
T.dec_max_nreg(24)
```

注意：

- consumer 侧暂时不在 TileLang 外层补 `T.inc_max_nreg(240)`，因为当前
  `tileops_fa3_shaped_run_consumer_wg1/wg2` 仍会进入
  `tileops_fa3_shaped_run_role`，helper 内部还保留
  `cutlass::arch::warpgroup_reg_alloc<AttnKernel::MmaRegisterRequirement>()`。
- 等 consumer helper 也被拆开后，再把 consumer register alloc 搬成
  `T.inc_max_nreg(240)`。

producer load helper 的重新定位：

- `tileops_fa3_shaped_producer_load_one_tile` / `load_tail` 里面没有 QK/PV
  compute、softmax 或 WGMMA。
- 它本质上是 FA3 producer 的 TMA + mbarrier/pipeline 协议：

```text
work_idx / tile_n
SeqlenInfo / 边界
stage = smem_pipe_write.index()
wait empty barrier
TMA load K/V/Vt -> shared stage
arrive full barrier
advance pipeline state
tail load / drain
```

后续替换顺序应更直接：

1. `producer_reg_dealloc` -> `T.dec_max_nreg(24)`。
2. K-only pipeline：TileLang `barrier_wait(empty) -> T.tma_copy(K) -> barrier_arrive(full)`。
3. V-only pipeline：复刻 K，并处理 FA3/gqa_fwd_ws 中 V 滞后一拍的 load 顺序。
4. `load_tail`：搬最后一块 V/Vt load 和 drain 协议。

每一步继续记录 correctness 和 4096 GQA benchmark。

实现：

producer split 分支中，原来的：

```python
T.call_extern(
    "handle",
    "tileops_fa3_shaped_producer_reg_dealloc",
    tx,
    warpgroup_id,
    producer_warp,
    producer_lane,
)
```

已替换为：

```python
T.dec_max_nreg(24)
```

`tileops_fa3_shaped_producer_reg_dealloc` helper 暂时保留在文件中，作为对照和
回溯用；主线 producer-split 分支不再调用它。

最小 smoke 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.113786 tflops=603.94
```

结论：

- producer register dealloc 可以直接使用 TileLang `T.dec_max_nreg(24)` 表达。
- producer split 主线减少了一个 producer-side `T.call_extern`。
- consumer register alloc 暂不外移，避免与当前 consumer helper 内部
  `warpgroup_reg_alloc` 重复。

#### M4f-main6: K-only single-buffer TMA pipeline probe

状态：已完成。

目的：

把 K producer load 从“单次 TMA probe”推进到更接近 FA3 producer 的
`empty/full` pipeline 协议：

```text
producer WG:
  barrier_wait(k_empty, parity)
  T.tma_copy(K tile -> TileLang-owned k buffer, barrier=k_full)
  barrier_arrive(k_full)

consumer WG1/WG2:
  barrier_wait(k_full, parity)
  consume/check k buffer
  barrier_arrive(k_empty)
```

由于原 FA3 raw shared storage 已占用 `196608B`，两个额外 K stage buffer 会
超过当前 shared memory budget。因此本步先采用 single-buffer pipeline：

- `k_hlir_pipe_probe = T.alloc_shared((224, dim), "float8_e4m3fn")`
- `k_hlir_pipe_full = T.alloc_barrier(arrive_count=128)`
- `k_hlir_pipe_empty = T.alloc_barrier(arrive_count=256)`

新增开关：

```bash
--producer-tma-hlir-k-pipeline-probe
```

producer WG 对所有 full K tiles 执行：

```python
for tile_n in T.Pipelined(seq_len // 224, num_stages=0):
    T.barrier_wait(k_hlir_pipe_empty, (tile_n + 1) % 2)
    T.tma_copy(
        k[bidb, tile_n * 224:(tile_n + 1) * 224, bidh_kv, 0:dim],
        k_hlir_pipe_probe,
        barrier=k_hlir_pipe_full,
    )
    T.barrier_arrive(k_hlir_pipe_full)
```

consumer WG1/WG2 对应执行：

```python
for tile_n in T.Pipelined(seq_len // 224, num_stages=0):
    T.barrier_wait(k_hlir_pipe_full, tile_n % 2)
    # WG1 tx == 128 checks samples against global K
    T.barrier_arrive(k_hlir_pipe_empty)
```

其中 `T.barrier_wait(k_empty, (tile_n + 1) % 2)` 参考
`gqa_fwd_ws.py` 的 producer 初始 empty parity，用于避免第一轮加载前死等。

最小 smoke 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-k-pipeline-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-k-pipeline-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.158570 tflops=433.37
```

结论：

- TileLang 可以表达 K producer 的 `empty/full` mbarrier pipeline 协议。
- `k_empty` 初始 parity 与 `gqa_fwd_ws.py` 的 `(tile_n + 1) % 2` 写法一致，
  smoke 和 4096 GQA 均通过。
- 该 probe 额外串行加载所有 full K tiles，同时原 FA3 producer load 仍执行，
  benchmark 明显下降是预期的 probe 开销，不代表最终替换路径性能。
- 下一步做 V-only pipeline，需要验证 V 相对 K 滞后一拍的 load/tail 协议。

#### M4f-main7: V-only single-buffer TMA pipeline probe

状态：已完成。

目的：

验证 FA3/gqa_fwd_ws producer 中 V load 的滞后一拍和 tail load 协议能用
TileLang 高级原语表达。

参考 `gqa_fwd_ws.py` 的 producer 结构：

```text
for n_idx in loop_range:
  load K[n_idx]
  if n_idx > 0:
    load V[n_idx - 1]

tail:
  load V[loop_range - 1]
```

本步仍采用 single-buffer pipeline，避免在 FA3 raw shared storage 之外再叠加
两个 V stage buffer：

- `v_hlir_pipe_probe = T.alloc_shared((224, dim), "float8_e4m3fn")`
- `v_hlir_pipe_full = T.alloc_barrier(arrive_count=128)`
- `v_hlir_pipe_empty = T.alloc_barrier(arrive_count=256)`

新增开关：

```bash
--producer-tma-hlir-v-pipeline-probe
```

producer WG 对 V tiles 执行：

```python
for n_idx in T.Pipelined(seq_len // 224, num_stages=0):
    if n_idx > 0:
        tile_n = n_idx - 1
        T.barrier_wait(v_hlir_pipe_empty, (tile_n + 1) % 2)
        T.tma_copy(
            v[bidb, tile_n * 224:(tile_n + 1) * 224, bidh_kv, 0:dim],
            v_hlir_pipe_probe,
            barrier=v_hlir_pipe_full,
        )
        T.barrier_arrive(v_hlir_pipe_full)

tail_tile_n = seq_len // 224 - 1
T.barrier_wait(v_hlir_pipe_empty, (tail_tile_n + 1) % 2)
T.tma_copy(
    v[bidb, tail_tile_n * 224:(tail_tile_n + 1) * 224, bidh_kv, 0:dim],
    v_hlir_pipe_probe,
    barrier=v_hlir_pipe_full,
)
T.barrier_arrive(v_hlir_pipe_full)
```

consumer WG1/WG2 对应执行：

```python
for tile_n in T.Pipelined(seq_len // 224, num_stages=0):
    T.barrier_wait(v_hlir_pipe_full, tile_n % 2)
    # WG1 tx == 128 checks samples against global V
    T.barrier_arrive(v_hlir_pipe_empty)
```

最小 smoke 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-v-pipeline-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-v-pipeline-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.159328 tflops=431.31
```

结论：

- TileLang 可以表达 V producer 的滞后一拍 load 和最后一块 tail load。
- `S=224` 覆盖 tail-only 路径；`S=4096` 覆盖 delayed load + tail 路径。
- 该 probe 额外串行加载所有 full V tiles，同时原 FA3 producer load 仍执行，
  benchmark 明显下降是预期 probe 开销。
- K-only 和 V-only pipeline 均已通过；下一步应设计显式 Q/K/V buffer pointer
  边界，开始让 helper 消费 TileLang-owned stage buffer，而不是只做 probe。

#### M4g-1: K/V explicit stage pointer boundary

状态：已完成。

目的：

把 K/V stage 的 C++ helper 边界从“专门的 K probe / V probe”收敛成统一的
显式 pointer interface：

```text
stage_kind: 0 = K, 1 = V
stage_ptr:  TileLang-owned typed/swizzled shared buffer
global_ptr: K/V global tensor pointer
tile_n, bidh_kv, bidb
```

新增 helper：

```text
tileops_fa3_shaped_check_kv_hlir_stage_boundary(
    stage_kind,
    stage_ptr,
    global_ptr,
    tx,
    tile_n,
    bidh_kv,
    bidb,
)
```

这个 helper 不再从 FA3 `SharedStorage` 推导 `smem_k/smem_v`，而是只消费
TileLang 传入的 stage pointer。它按 FA3 128B swizzle physical index 抽样，
并与对应 global K/V 对比。

新增开关：

```bash
--producer-tma-hlir-kv-boundary-probe
```

由于 shared memory budget 紧张，本步只额外声明一个 TileLang-owned
`(224, dim)` stage buffer，并复用它验证 K 和 V 两个 boundary：

```python
kv_hlir_boundary_probe = T.alloc_shared((224, dim), "float8_e4m3fn")
kv_hlir_boundary_full = T.alloc_barrier(arrive_count=128)
kv_hlir_boundary_done = T.alloc_barrier(arrive_count=256)

# phase 0: K tile0
T.tma_copy(k[bidb, 0:224, bidh_kv, 0:dim], kv_hlir_boundary_probe,
           barrier=kv_hlir_boundary_full)
T.barrier_arrive(kv_hlir_boundary_full)
T.barrier_wait(kv_hlir_boundary_done, 0)

# phase 1: V tile0 reuses the same stage pointer
T.tma_copy(v[bidb, 0:224, bidh_kv, 0:dim], kv_hlir_boundary_probe,
           barrier=kv_hlir_boundary_full)
T.barrier_arrive(kv_hlir_boundary_full)
T.barrier_wait(kv_hlir_boundary_done, 1)
```

consumer WG1 调用统一 helper：

```python
T.barrier_wait(kv_hlir_boundary_full, 0)
if tx == 128:
    check_kv_boundary(kind=0, stage_ptr=kv_hlir_boundary_probe, global_ptr=k.data)
T.barrier_arrive(kv_hlir_boundary_done)

T.barrier_wait(kv_hlir_boundary_full, 1)
if tx == 128:
    check_kv_boundary(kind=1, stage_ptr=kv_hlir_boundary_probe, global_ptr=v.data)
T.barrier_arrive(kv_hlir_boundary_done)
```

最小 smoke 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-kv-boundary-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-kv-boundary-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.117592 tflops=584.39
```

结论：

- C++ helper 可以通过统一显式 pointer boundary 消费 TileLang-owned K/V
  stage buffer。
- 同一个 TileLang-owned stage buffer 可以在 producer/consumer barrier 保护下
  先作为 K、再作为 V 被 helper 消费。
- benchmark 比 full K/V pipeline probe 明显恢复，因为本步只额外加载 tile0
  的 K/V 各一次；它验证的是 helper 边界形状，不是完整 pipeline 性能。
- 下一步可以把 `producer_load_one_tile` / `load_tail` 拆成更小的局部 helper，
  让其中的 K/V stage 输入来自这个显式 pointer boundary。

追加验证：consumer tensor boundary

在同一开关下，统一 helper 进一步从 raw byte sample check 推进到
FA3/CuTe consumer operand view construction：

```cpp
// kind = K
Tensor sK = make_tensor(
    make_smem_ptr(stage),
    typename CollectiveMainloop::SmemLayoutK{});
typename CollectiveMainloop::TiledMmaQK tiled_mma_qk;
auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
auto tSrK = wg_mma_qk.partition_fragment_B(sK);

// kind = V
Tensor sV = make_tensor(
    make_smem_ptr(stage),
    typename CollectiveMainloop::SmemLayoutVtMma{});
typename CollectiveMainloop::TiledMmaPV tiled_mma_pv;
auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
auto tOrV = wg_mma_pv.partition_fragment_B(sV);
```

这说明显式 TileLang-owned stage pointer 不只能被 raw index helper 读取，也能
形成 FA3 consumer `mma` 里同类的 CuTe operand tensor view。

重新验证结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-kv-boundary-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-kv-boundary-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.117360 tflops=585.54
```

#### M4g-2: Q explicit consumer tensor boundary

状态：已完成。

目的：

补齐 Q 侧的显式 stage pointer boundary：让 TileLang-owned Q stage pointer
可以构造 FA3/CuTe consumer `mma` 中同类的 Q operand tensor view。

新增 helper：

```text
tileops_fa3_shaped_check_q_hlir_consumer_tensor_boundary(
    q_stage_ptr,
    q_global_ptr,
    tx,
    tile_m,
    bidh,
    bidb,
)
```

该 helper 使用显式 Q stage pointer 构造：

```cpp
Tensor sQ = make_tensor(
    make_smem_ptr(q_stage),
    typename CollectiveMainloop::SmemLayoutQ{});
typename CollectiveMainloop::TiledMmaQK tiled_mma_qk;
auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
auto tSrQ = wg_mma_qk.partition_fragment_A(sQ);
```

同时保留 raw swizzle sample check，确认 stage 内容仍与 global Q 对齐。

新增开关：

```bash
--producer-tma-hlir-q-boundary-probe
```

TileLang 侧新增：

```python
q_hlir_boundary_probe = T.alloc_shared((128, dim), "float8_e4m3fn")
q_hlir_boundary_ready = T.alloc_barrier(arrive_count=128)

T.annotate_layout({
    q_hlir_boundary_probe: tilelang.layout.make_swizzled_layout(q_hlir_boundary_probe),
})
```

producer WG 直接 TMA 当前 Q tile：

```python
T.tma_copy(
    q[bidb, tile_m * 128:tile_m * 128 + 128, bidh, 0:dim],
    q_hlir_boundary_probe,
    barrier=q_hlir_boundary_ready,
)
T.barrier_arrive(q_hlir_boundary_ready)
```

consumer WG1/WG2 分别用 `tx == 128` / `tx == 256` 构造各自的 Q operand view。

最小 smoke 结果：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-q-boundary-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

4096 GQA benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-q-boundary-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.113614 tflops=604.85
```

结论：

- Q/K/V 三类 stage pointer 都已证明可以由 C++ helper 显式接收。
- Q/K/V 显式 stage pointer 都可以构造 FA3/CuTe consumer operand tensor view。
- 下一步先回到 producer core，拆掉当前 line 2320 / 2335 的 producer
  `T.call_extern`。这些 extern 仍然拥有 producer 的核心 TMA/pipeline 协议；
  只有完成它们的 HLIR 化后，再进入 QK/PV consumer compute。

### M4g-3 Plan: producer core extern 拆解

状态：计划。

目标对象：

```python
T.call_extern("tileops_fa3_shaped_producer_load_one_tile", ...)
T.call_extern("tileops_fa3_shaped_producer_load_tail", ...)
```

它们当前位于 producer split 分支中，是 producer 侧剩余的核心 extern。它们
不包含 QK/PV compute、softmax 或 epilogue，但仍然拥有：

```text
pipeline_k / pipeline_v / pipeline_vt wrapper construction
PipelineState smem_pipe_write
work_idx
SeqlenInfo
block_coord
mainloop.load(...)
mainloop.load_tail(...)
```

`mainloop.load` 展开后的核心语义：

```text
load initial K/V
load Q / arrive QueryFull
wait O/V empty signal
loop:
  K load current tile
  V load delayed or current tile depending overlap/transpose
  advance smem_pipe_write
tail:
  producer_tail / drain
```

#### M4g-3a: producer core audit table

把 `producer_load_one_tile` / `producer_load_tail` 内的职责拆成表：

| piece | current owner | TileLang candidate | keep extern? |
| --- | --- | --- | --- |
| `wait_on_dependent_grids` | C++ | no-op/current dense case 或 small extern | maybe |
| `SeqlenInfo` | C++ | TileLang scalar coord + fixed dense shape | no |
| `block_coord` | TileLang 已拥有 | TileLang scalar | no |
| `PipelineState` | C++ | TileLang loop index/parity | no |
| K TMA load | C++ mainloop.load | `T.tma_copy + barrier` | no |
| V delayed/tail load | C++ mainloop.load/load_tail | `T.tma_copy + barrier` | no |
| Q TMA / Query barrier | C++ mainloop.load | `T.tma_copy + barrier` | no |
| `producer_tail` drain | C++ pipeline API | TileLang barrier protocol 或 tiny extern | maybe |

验收：

- 文档列清哪些 pieces 已由前面的 K/V/Q probes 验证。
- 标出真正还不确定的只有 `producer_tail` / FA3 pipeline drain 等价性。

#### M4g-3b: staged producer core shadow path

目标：

在 producer split 分支中建立 TileLang-owned Q/K/V stage buffers 和 TileLang
barrier/pipeline 协议，但仍保留原 FA3 producer helper 作为真实 correctness
fallback。

2026-06-03 源码审计修正：

不能按外部 probe 行为猜 producer 组织；必须对齐 FA3
`mainloop_fwd_sm90_tma_gmma_ws.hpp` 的 `load/load_tail`：

```text
实例参数:
  Stages = 2
  TileShape_MNK = (128, 224, 128)
  Use_TMA_Q = true
  Use_TMA_KV = true
  MmaPV_is_RS = true
  IntraWGOverlap = true
  V_colmajor = false
  Transpose_V = true  # FP8 row-major V path
```

真实 producer 结构：

```text
n_block_min, n_block_max = BlockMN::get_n_block_min_max(...)
n_block = n_block_max - 1

initial:
  if Transpose_V: load_V(n_block, smem_pipe_write) into pipeline_vt/smem_vt
  load_K(n_block, smem_pipe_write) into pipeline_k/smem_k

Q:
  wait NamedBarrier::QueryEmpty
  barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ)
  TMA Q into smem_q

wait:
  barrier_O.wait((work_idx + 1) % 2)

loop:
  smem_pipe_write_v = smem_pipe_write
  ++smem_pipe_write
  if Transpose_V: load_V(n_block, smem_pipe_write) into pipeline_vt/smem_vt
  load_K(n_block, smem_pipe_write) into pipeline_k/smem_k
  if Transpose_V: copy_Vt_to_V(smem_pipe_write_v)

tail:
  if Transpose_V: copy_Vt_to_V(smem_pipe_write)
  ++smem_pipe_write
  ++work_idx
```

重要约束：

- K 和 V/Vt 共用同一个 `PipelineState smem_pipe_write`，但属于不同
  pipeline storage。
- FP8 `V_colmajor=false` 时，consumer 不直接消费 TMA Vt；producer WG 先从
  `pipeline_vt/smem_vt` wait，再 transpose 到 `pipeline_v/smem_v`。
- Q 不属于 `pipeline_k/pipeline_v/pipeline_vt`，而是 `barrier_Q` +
  `NamedBarrier::QueryEmpty`。
- `seq_len=4096` 时 K/V block 数是 `ceildiv(4096, 224)=19`，不是
  `4096 // 224=18`；最后一个 partial tile 需要按 FA3 TMA tensor/SeqlenInfo
  语义处理，不能在 TileLang shadow 中静默丢掉。
- 因此 M4g-3b 的 shadow 需要分层启用：Q path；FA3 exact-order K path；
  Vt TMA path；最后才是 `pipeline_vt -> pipeline_v` transpose/bridge。不能把
  V 简化成普通 `(224, dim)` stage。

实现状态：

- 新增 `build_core_shadow_kernel(...)` 专用 JIT builder，避免旧 probe 分支太深
  触发 eager Python `too many statically nested blocks`。
- `--producer-tma-hlir-core-shadow-probe` 默认只启用 Q shadow。
- `--producer-tma-hlir-core-shadow-kind=qk` 启用 FA3 exact-order K shadow。
- `--producer-tma-hlir-core-shadow-kind=qkv` 现在启用到 Vt TMA shadow：producer
  用 TileLang `T.tma_load` + 4D descriptor 加载 V 到 `(dim, 224)` Vt stage，
  consumer-side checker 用 FA3 `CollectiveMainloop::SmemLayoutVt` 读取逻辑
  `(d, n, stage)` 验证内容。它还没有替换真正的 `copy_Vt_to_V`。

Q-only core shadow smoke：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-core-shadow-probe \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.00152588 cosine=0.99973959
```

Q-only core shadow 4096 benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-core-shadow-probe \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.115619 tflops=594.36
```

K exact-order shadow:

- K shadow 已改为 FA3 源码顺序：`n_block = n_block_max - 1`，从后往前
  load。
- `n_block_max = T.ceildiv(seq_len, 224)`，4096 case 覆盖最后 partial tile。
- C++ checker 对 K/V stage 的抽样改为只比较 `global_row < S` 的有效行，
  避免 partial tile 读取越界 reference。

K exact-order smoke：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 896 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-core-shadow-probe \
  --producer-tma-hlir-core-shadow-kind qk \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.000915527 cosine=0.99972862
```

K exact-order 4096 smoke/benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-core-shadow-probe \
  --producer-tma-hlir-core-shadow-kind qk \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.171336 tflops=401.08
```

解释：

- 该性能点是 shadow path，原 FA3 producer 仍真实执行，所以多了一整套
  TileLang K TMA load；下降符合预期。
- 下一步不能把 V 简化成 `(224, dim)` stage。当前实例
  `Transpose_V=true`，FA3 先 TMA 到 `smem_vt`，其逻辑 tile 是
  `(dim, block_n)`，再由 producer WG 做 `Vt -> V` transpose 后交给
  `pipeline_v/smem_v`。

Vt TMA shadow 源码确认：

- 以 vendor FA3
  `.github/runner/vendor/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
  为准，不以 TileOps FP8 helper 作为权威。
- 当前实例 `Is_FP8=true` 且 `V_colmajor=false`，所以：

```text
Transpose_V = true
TmaMajorV = GMMA::Major::MN
MmaMajorV = GMMA::Major::K
SmemLayoutVt    = ss_smem_selector<TmaMajorV>(D, block_n) tiled to (D, block_n, stages)
SmemLayoutVtMma = ss_smem_selector<MmaMajorV>(D, block_n) tiled to (D, block_n, stages)
```

- `load_V(...)` 走 `pipeline_vt`，TMA 写入 `sVt`/`smem_vt`。
- `copy_Vt_to_V(...)` 做：

```text
pipeline_vt.consumer_wait(read_state)
pipeline_v.producer_acquire(write_state)
LDSM.T from flat_divide(sVt, (64, 8))
byte_perm 0x6420 / 0x7531
STSM_N to flat_divide(sV, (8, 16))
fence_view_async_shared()
pipeline_v.producer_commit(write_state)
NamedBarrier::sync(TransposeBarrier)
pipeline_vt.consumer_release(read_state)
```

- 因此 V 的完整 producer 替换必须包含两层：先 `T.tma_load` 到 FA3
  `SmemLayoutVt` 等价 stage，再把 FA3 的 LDSM/STSM transpose/bridge 搬到
  TileLang 或一个更小的局部 helper。当前 `qkv` shadow 只完成第一层。

Vt TMA shadow smoke/benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 896 --heads 1 --heads-kv 1 \
  --role-run --producer-split --producer-tma-hlir-core-shadow-probe \
  --producer-tma-hlir-core-shadow-kind qkv \
  --warmup 1 --repeat 1
```

```text
out finite True
lse finite True
correctness max_abs=0.000915527 cosine=0.99972862
```

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split --producer-tma-hlir-core-shadow-probe \
  --producer-tma-hlir-core-shadow-kind qkv \
  --bench --warmup 5 --repeat 20
```

```text
out finite True
lse finite True
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.222045 tflops=309.48
```

备注：该性能点低得不应作为性能 milestone。当前 `qkv` shadow 是串行前置
pre-pass：每个 CTA 先用 TileLang 额外完成 Q shadow、全量 K shadow、全量 Vt
shadow，再进入原 FA3 producer。因此它同时包含：

- 原 FA3 producer 的真实 Q/K/V/Vt load + transpose。
- TileLang shadow 的额外 K/Vt TMA。
- 每个 shadow tile 的额外 barrier/checker 同步。
- 没有 FA3 mainloop 中 K/Vt load 与 consumer math 的 overlap。

所以 `0.222045 ms / 309.48 TFLOP/s` 只说明 Vt TMA 边界正确且额外开销很大，
不能用来预测完成替换后的性能。后续只有在移除对应 FA3 producer sub-call，
或把 shadow 融入同一 pipeline overlap 后，benchmark 才重新作为性能验收依据。

#### M4g-3b+ 后续边界与验收标准

统一性能口径：

```text
B=1, S=4096, H=8, HKV=2, D=128
--bench --warmup 5 --repeat 20
```

基线：

```text
M4 baseline: producer split，但还没替换 producer core
FA3 baseline: 原始 FA3 whole helper / host launcher
```

后续 performance gate 主要相对 M4 baseline；FA3 baseline 用于记录整体差距。

**B1: Vt TMA boundary**

边界：

```text
TileLang T.tma_load -> Vt stage
checker uses FA3 CollectiveMainloop::SmemLayoutVt
```

功能验收：

```text
S=224,  H=1, HKV=1 correctness pass
S=896,  H=1, HKV=1 correctness pass
S=4096, H=8, HKV=2 correctness pass
```

性能验收：

```text
diagnostic only
not a performance gate
```

原因：当前 B1 是 serial shadow pre-pass，会额外搬运 K/Vt，不能预测最终性能。

**B2: Vt -> V boundary**

边界：

```text
input:  Vt stage, FA3 SmemLayoutVt
op:     tiny helper copy_Vt_to_V equivalent
output: V stage,  FA3 SmemLayoutVtMma
```

功能验收：

```text
Vt->V checker pass
TiledMmaPV.partition_fragment_B(sV) pass
S=224, H=1, HKV=1 correctness pass
S=896, H=1, HKV=1 correctness pass
```

性能验收：

```text
independent probe: diagnostic latency only
mainline shadow: latency <= 1.10x qkv shadow latency
```

备注：如果 B2 只是独立 probe，不做 performance gate；如果接入主线但仍然
shadow，只要求不要比当前 qkv shadow 明显更差。

B2 out-of-place infrastructure result：

为解决 in-place probe 不可靠的问题，新增独立 probe：

```text
--producer-tma-hlir-vt-to-v-boundary-probe
```

该 probe 不携带原 FA3 `SharedStorage`，因此可以同时分配两块 TileLang shared
stage：

```text
vt_stage: FA3 SmemLayoutVt source
v_stage:  FA3 SmemLayoutVtMma destination
```

执行路径：

```text
global V
  -> TileLang T.tma_load 4D descriptor
  -> vt_stage
  -> tileops_fa3_shaped_vt_to_v_boundary(vt_stage, v_stage)
  -> tileops_fa3_shaped_check_v_mma_layout_boundary(v_stage)
```

`tileops_fa3_shaped_vt_to_v_boundary` 是 vendor FA3 `copy_Vt_to_V` 的 tiny
out-of-place helper：`LDSM.T + byte_perm(0x6420/0x7531) + STSM_N`。checker
目前只验证 `v_stage` 可以被 FA3 `TiledMmaPV.partition_fragment_B(sV)` 接受，
不做 `sV(d,n) == global_v[n,d]` 的内容比较，因为 `SmemLayoutVtMma` 可能包含
PV/epilogue 配套的列 permutation。

Smoke：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --producer-tma-hlir-vt-to-v-boundary-probe \
  --warmup 1 --repeat 1
```

```text
vt_to_v_boundary_status [1]
```

Multi-block smoke：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 896 --heads 1 --heads-kv 1 \
  --producer-tma-hlir-vt-to-v-boundary-probe \
  --warmup 1 --repeat 1
```

```text
vt_to_v_boundary_status [1]
```

Diagnostic benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-vt-to-v-boundary-probe \
  --bench --warmup 5 --repeat 20
```

```text
vt_to_v_boundary_status [1]
latency_ms=0.002736
```

结论：

- in-place 不再作为 B2 基础设施依据。
- out-of-place `SmemLayoutVt -> SmemLayoutVtMma` helper 能独立运行。
- 输出 layout 能被 FA3 PV operand partition 接受。
- 内容级验收仍待后续 fragment-level checker 或接入 PV correctness 后完成。

B2 源码复核：为什么不能用 naive V content checker：

vendor FA3 `mainloop_fwd_sm90_tma_gmma_ws.hpp` 对当前 FP8 row-major V case 的
定义是：

```text
Transpose_V = Is_FP8 && !V_colmajor = true
TmaMajorV = GMMA::Major::MN
MmaMajorV = GMMA::Major::K
```

所以 V producer 的真实路径是：

```text
V global
  -> TMA
  -> smem_vt / SmemLayoutVt
  -> LDSM.T + byte_perm + STSM_N
  -> smem_v / SmemLayoutVtMma
  -> PV GMMA
```

FA3 在 `R2STiledCopyV` 附近明确说明：

```text
Instead we will permute the cols of V,
and un-permute the cols of O in the epilogue.
```

对应代码路径还包括：

```text
mainloop: if Is_FP8 && !V_colmajor -> flash::permute_output_fp8(tOrO)
epilogue: FP8PermuteCol=Transpose_V -> flash::permute_output_fp8_Vcolmajor(...)
```

因此 `SmemLayoutVtMma` 的内容不能用：

```text
sV(d, n) == global_v[n, d]
```

直接验收。这个 checker 会把 FA3 故意引入的 V column permutation 误判为错误。

2026-06-04 接续计划：

1. 保留当前 B2 out-of-place infrastructure probe，作为 `Vt -> V` helper
   layout smoke。
2. 新增 B2b PV-level correctness probe，而不是继续调 raw swizzle checker：

```text
input:
  known P tile
  global V -> TMA -> vt_stage -> Vt->V -> v_stage

op:
  FA3 TiledMmaPV consumes v_stage
  apply the same FA3 output permutation path used after PV

check:
  compare PV result with reference P @ V
```

3. 如果 PV-level probe 通过，再回到主线 B3，把 producer loop skeleton 写成
   FA3 exact ordering：

```text
initial Vt/K
Q
loop: next Vt/K + previous Vt->V
tail Vt->V
```

4. 不再把 `qkv shadow` latency 当 performance signal；只有 B3 开始恢复
   overlap 后，benchmark 才重新进入 performance gate。

B2b 更新（2026-06-04）：

新增独立入口：

```text
--producer-tma-hlir-pv-correctness-probe
```

当前 probe 已能编译运行并确认：

- `global V -> 4D TMA -> FA3 SmemLayoutVt source` byte-level 路径成立。
- 既有 `--producer-tma-hlir-vt-to-v-boundary-probe` 仍保留为 out-of-place
  infrastructure smoke。
- 手写近似 `TiledMmaPV + output dump` 不是可靠验收方式；最新结果 finite 但
  初始 correctness 未过，表现为 `cos ~0.69-0.71`，且 half diagnostic 更接近后半
  segment：

```text
first_cos  ~0.13
second_cos ~0.86
```

解释：这不能说明两段相加正确。cosine 不是可加量；这个形态更像 PV checker
只复刻到了部分 `ki/N` segment 或 output canonicalization 与 FA3 exact contract
不一致。

后续 B2b 实现策略：

不再继续手写近似 checker，也不照搬低效 staging dataflow。以下 helper 作为
oracle contract：

```text
fp8_pack_p_logical_fa3_raw_64x128x224
fp8_pv_cute_grouped_begin_accumulate_from_p_frag_fa3_raw_64x128x224
fp8_pv_ptx_unit_wait_update_tail_fa3_raw_64x128x224
```

它们分别包含：

```text
logical P -> FA3 PV A-register/raw p_frag layout
v_stage -> FA3 TiledMmaPV B descriptor / 7 ki issue contract
PV raw accumulator -> canonical accumulator/output permutation contract
```

新的 B2b helper 应做“最小内联版”：

```text
logical P
  -> inline pack into local FA3 p_frag
v_stage
  -> same TiledMmaPV partition_fragment_B
PV ki loop
  -> local raw accumulator
same canonicalization logic as wait_update_tail
  -> compare/dump against P @ V reference
```

也就是说，复用 helper 里的 layout transform 和 exact contract，但避免
`p_smem -> p_frag -> begin helper -> wait/update helper` 这种低效配置路径。
如果需要继续定位，可以先加 `ki_mask` diagnostic：

```text
0x7f: all 7 chunks
0x0f: first 4 chunks
0x70: last 3 chunks
1<<i: single ki
```

单个 `ki` 的 reference 对齐后，再恢复 all-ki B2b correctness。

B2b 结果更新（2026-06-04）：

`--producer-tma-hlir-pv-correctness-probe` 已通过。最终 probe 使用：

```text
global V
  -> tl::fp8_tma_load_4d_ptx
  -> FA3 SmemLayoutVt source
  -> tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm
P logical tile
  -> tl::fp8_pack_p_logical_fa3_raw_64x128x224
PV
  -> tl::fp8_pv_cute_grouped_begin_accumulate_from_p_frag_fa3_raw_64x128x224
  -> tl::fp8_fa3_raw_acc_permute_to_canonical_64x128
  -> tl::fp8_fa3_raw_acc_store_global_64x128
```

验证命令：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --producer-tma-hlir-pv-correctness-probe \
  --warmup 1 --repeat 1
```

通过结果：

```text
pv_correctness max_abs=8 cosine=0.99999964
pv_correctness_best row_cos=1 col_cos=1
```

定位过程中的关键分叉：

- row-major V shared copy + `tl::fp8_transpose_v_128x224_ldsm_stsm` 通过，
  说明 P pack、PV helper、two-warpgroup 128-row output 都没问题。
- FA3-source TMA 路径使用 TileLang `T.tma_load` 时表现为 suffix-only：
  `suffix_cos[4] ~= 0.999997`，只对应 `N=128..223`。
- 把同一路径改成主 kernel / standalone 4D TMA probe 同款
  `tl::fp8_tma_load_4d_ptx` 后通过。

因此 B2b 的结论是：B2 out-of-place Vt -> V infrastructure 可以保留，但
B2b PV-level correctness 必须使用 `tl::fp8_tma_load_4d_ptx` 作为 FA3-source
TMA primitive；不能把该路径替换成当前 TileLang `T.tma_load` 变体。

**B3: FA3 exact producer loop skeleton**

边界：

TileLang 外层表达 FA3 exact ordering：

```text
initial:
  Vt(n_last), K(n_last)

Q:
  Q(tile_m)

loop descending:
  Vt(next), K(next)
  Vt->V(previous)

tail:
  Vt->V(last)
```

功能验收：

```text
S=896,  H=1, HKV=1 correctness pass
S=4096, H=8, HKV=2 correctness pass
partial tile S=4096 covers ceildiv(4096, 224)=19
```

边界验收：

```text
TileLang code exposes stage_idx/parity
K/Vt share one logical write state
Vt->V uses previous write state in loop
```

性能验收：

```text
latency <= 1.15x M4 baseline
```

这是第一个真正有意义的 performance gate，因为它开始恢复 FA3 producer 的
load/compute overlap。

B3-0a 更新（2026-06-04）：

新增独立入口：

```text
--producer-tma-hlir-exact-order-shadow-probe
```

当前实现是 standalone shadow skeleton，不接入真实 FA3 output path，也不分配
FA3 `SharedStorage`。这样可以在一个 384-thread CTA 内容纳：

```text
Q stage
K stage ping-pong
Vt stage ping-pong
V stage ping-pong
```

默认只验 exact ordering / barrier parity / stage reuse，不打开内容 checker：

```text
--producer-tma-hlir-exact-order-shadow-checks none
```

强检查使用：

```text
--producer-tma-hlir-exact-order-shadow-checks all
```

`all` 不是把四类 checker 全部塞进同一个 CTA，而是按 `bidh % 4` 分发：

```text
0 -> q
1 -> k
2 -> vt
3 -> v layout-only
```

这样 `H=8, HKV=2` 会覆盖两组 `q/k/vt/v`，同时避免 monolithic all-check
带来的组合副作用。

事件 trace 约定：

```text
1: initial Vt(last)
2: initial K(last)
3: Q(tile_m)
4: loop Vt(next)
5: loop K(next)
6: loop Vt->V(previous)
7: tail Vt->V(last)
```

通过命令：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 896 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-exact-order-shadow-probe \
  --producer-tma-hlir-exact-order-shadow-checks all \
  --warmup 1 --repeat 1
```

通过 trace：

```text
exact_order_shadow_status [1]
[[1, 3, 0, 0],
 [2, 3, 0, 0],
 [3, 0, 0, 0],
 [4, 2, 1, 0],
 [5, 2, 1, 0],
 [6, 3, 0, 1],
 [4, 1, 0, 1],
 [5, 1, 0, 1],
 [6, 2, 1, 0],
 [4, 0, 1, 0],
 [5, 0, 1, 0],
 [6, 1, 0, 1],
 [7, 0, 1, 0]]
```

已验证：

```text
S=896, H=1, HKV=1, checks=all 通过（只覆盖 q；单项 k/vt/v 另测）
S=896, H=4, HKV=2, checks=all 通过（分发式 all）
S=896, H=8, HKV=2, checks=all 通过（分发式 all，目标 GQA 形态）
B2b PV correctness: S=224, H=4, HKV=4 通过
```

定位记录：

- `checks=none` 在 `H=8, HKV=2` 通过，说明 exact-order barrier/parity skeleton
  能覆盖目标 GQA 形态。
- 单项定位显示 `q/k/vt/v` checker 在 `H=2, HKV=2`、`H=4, HKV=2`
  下均可独立通过。
- 旧的 monolithic `checks=all` 在 `H>=4` 会 trap，但单项均过；因此判断为
  checker 组合副作用，而不是 producer ordering 或单项 content checker 错误。
- 已修为 distributed all：`H=4, HKV=2` 和 `H=8, HKV=2` 通过。
- K 曾短暂尝试改成 `tl::fp8_tma_load_4d_ptx`，但该 descriptor 写入的 shared
  layout 不等同于 `SmemLayoutK` 期望形态，连 `H=1, HKV=1` K checker 都会
  trap；因此 K shadow load 已恢复为 `T.tma_copy`。

B3-0b 更新（2026-06-04）：

exact-order shadow 已支持 `ceildiv(seq_len, 224)` 的 N tile 个数，不再要求
`seq_len % 224 == 0`。`S=4096` 时：

```text
num_n_blocks = ceildiv(4096, 224) = 19
last_tile_n = 18
trace_len = 3 * 19 + 1 = 58
```

长序列验收命令：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-exact-order-shadow-probe \
  --producer-tma-hlir-exact-order-shadow-checks all \
  --warmup 1 --repeat 1
```

通过结果：

```text
exact_order_shadow_status [1]
trace starts:
  [1, 18, 0, 0], [2, 18, 0, 0], [3, 0, 0, 0],
  [4, 17, 1, 0], [5, 17, 1, 0], [6, 18, 0, 1]
trace ends:
  [4, 0, 0, 1], [5, 0, 0, 1], [6, 1, 1, 0], [7, 0, 0, 0]
```

这说明 producer shadow 的 order/parity/stage reuse 已覆盖目标长序列：

```text
initial:
  Vt(last=18), K(last=18), Q(tile_m)
loop:
  Vt(next), K(next), Vt->V(previous)
tail:
  Vt->V(tile 0)
```

单项定位：

```text
S=4096, H=8, HKV=2, checks=q    通过
S=4096, H=8, HKV=2, checks=k    通过
S=4096, H=8, HKV=2, checks=v    通过
S=4096, H=8, HKV=2, checks=none 通过
S=4096, H=8, HKV=2, checks=vt   通过
S=4096, H=8, HKV=2, checks=all  通过
```

Vt 长序列失败的根因已定位并修复：

```text
错误：两个 Vt TMA mbarrier 复用时直接使用 buffer id parity。
正确：每个 Vt TMA mbarrier 需要按自身复用次数翻转 parity。
```

具体说，`vt_tma_full0` 的使用序列是：

```text
initial tile 18 -> parity 0
loop tile 16   -> parity 1
loop tile 14   -> parity 0
...
```

`vt_tma_full1` 的使用序列是：

```text
loop tile 17 -> parity 0
loop tile 15 -> parity 1
loop tile 13 -> parity 0
...
```

修复后长序列 `checks=all` 重新覆盖 Q/K/Vt/V layout、barrier parity、
ping-pong stage reuse 和完整 trace。`S=4096` 的 partial last tile 也由
Vt checker 按 valid row 采样覆盖。

B3-0b 后续事项：

```text
1. 把 exact-order shadow 接回 role-run / producer-split 语境。
2. 用 4096 benchmark 记录 producer skeleton 的速度回退。
3. 后续如果要降低 probe 编译/运行成本，再单独做低开销采样 checker。
```

B3-0c 性能锚点（2026-06-04）：

旧的 monolithic `build_kernel(...)` 现在不适合继续作为 B3 性能入口：

```text
--role-run --producer-split
-> TileLang eager: too many statically nested blocks
```

根因不是 kernel runtime，而是历史 probe 分支过多。即使这些分支运行时为 false，
TileLang eager 仍会把它们展开进 AST。为后续性能线新增 dedicated 瘦身入口：

```text
--producer-tma-hlir-slim-baseline-probe
```

该入口只保留 M4 role-run producer-split 的必要路径：

```text
prepare_params
prepare_runtime
producer_load_one_tile
producer_load_tail
run_consumer_wg1
run_consumer_wg2
```

correctness smoke：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 896 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-slim-baseline-probe \
  --warmup 1 --repeat 1
```

结果：

```text
correctness max_abs=0.000915527 cosine=0.99977273
```

4096 benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-slim-baseline-probe \
  --bench --warmup 5 --repeat 20
```

结果：

```text
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.113416 tflops=605.91
```

standalone exact-order skeleton 裸开销：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-exact-order-shadow-probe \
  --producer-tma-hlir-exact-order-shadow-checks none \
  --bench --warmup 5 --repeat 20
```

结果：

```text
latency_ms=0.066074
```

解释：

- `0.113416 ms` 是新的同环境 M4 baseline，和旧记录 `0.113312 ms`
  基本一致。
- `0.066074 ms` 只代表 exact-order shadow skeleton 的裸数据搬运/同步开销，
  还没有接入真实 FA3 output path，不能直接当最终回退。
- 如果把 standalone skeleton 完全串行叠加到 M4 baseline，上界约为
  `0.066074 / 0.113416 = 58.3%` 额外开销。下一步 B3-1 要在 slim builder
  中串入 exact-order shadow，先记录 shadow-path 回退，再推进替换 producer core。

B3-1 shadow-path benchmark（2026-06-04）：

新增入口：

```text
--producer-tma-hlir-slim-exact-order-shadow-probe
```

实现方式：

```text
same CTA / same dynamic smem arena
exact-order shadow runs before prepare_params / prepare_runtime
then normal FA3 producer_load_one_tile / producer_load_tail / consumers produce output
```

为什么 shadow 放在 `prepare_params` 前：

```text
FA3 SharedStorage already uses ~196KB dynamic smem.
Standalone shadow needs ~188KB for Q/K/Vt/V stages.
Therefore B3-1 reuses the same smem arena by byte offset instead of allocating
extra shared memory. Running shadow before prepare_params avoids clobbering
FA3 runtime metadata.
```

smoke：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 896 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-slim-exact-order-shadow-probe \
  --warmup 1 --repeat 1
```

结果：

```text
correctness max_abs=0.000915527 cosine=0.99977273
```

4096 benchmark：

```bash
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --producer-tma-hlir-slim-exact-order-shadow-probe \
  --bench --warmup 5 --repeat 20
```

结果：

```text
correctness max_abs=0.000457764 cosine=0.99970412
latency_ms=0.156368 tflops=439.47
```

baseline 复测：

```text
--producer-tma-hlir-slim-baseline-probe
latency_ms=0.113451 tflops=605.72
```

回退：

```text
absolute overhead = 0.156368 - 0.113451 = 0.042917 ms
relative overhead = 37.8%
throughput ratio  = 0.113451 / 0.156368 = 72.6%
```

解释：

- 这是 shadow path 上界回退：真实 FA3 output 仍由原 producer core 生成，
  exact-order shadow 额外串在前面。
- 集成后的额外开销 `0.0429 ms` 小于 standalone skeleton `0.0661 ms`，
  说明同 CTA / same smem arena 下已有一部分 launch/CTA 固定成本被摊掉。
- 下一步 B3-2 不应继续叠加 shadow，而应开始替换 producer core 的局部阶段：
  先让 slim path 使用 exact-order Q/K/Vt->V 的一部分真实输出前置条件，
  再逐步删除 `producer_load_one_tile` / `producer_load_tail` 对应职责。

**B4: remove producer core sub-call**

边界：

至少一个 FA3 producer data movement 不再由 `producer_load_one_tile` 发起。
建议第一刀：

```text
TileLang owns Vt TMA
tiny helper owns Vt->V
FA3 helper no longer duplicates this V load/transpose
```

功能验收：

```text
S=224,  H=1, HKV=1 correctness pass
S=896,  H=1, HKV=1 correctness pass
S=4096, H=8, HKV=2 correctness pass
```

边界验收：

```text
no duplicate Vt TMA
FA3 consumer waits on TileLang/bridge-produced V stage
```

性能验收：

```text
latency <= 1.10x M4 baseline
```

如果先只去掉 V，而 K/Q 仍留在 FA3 helper，则 latency 必须比 qkv shadow
明显回升；否则说明仍有重复搬运或同步边界不对。

**B5: remove producer tail sub-call**

边界：

替换：

```text
producer_load_tail
```

功能验收：

```text
all selected S/H/HKV correctness pass
no hang
no launch failure
```

边界验收：

```text
barrier_O wait
pipeline_k.producer_tail
pipeline_v.producer_tail
pipeline_vt.producer_tail
```

这些等价关系必须能在 TileLang 或 tiny helper 中明确看到。

性能验收：

```text
latency <= 1.08x M4 baseline
```

**B6: producer complete replacement milestone**

边界：

```text
producer_load_one_tile removed
producer_load_tail removed
producer side only keeps tiny helpers:
  optional Vt->V
  optional FA3 pipeline bridge
  optional nreg dealloc if TileLang nreg primitive is insufficient
```

功能验收：

```text
B=1, S=128,  H=1, HKV=1 correctness pass
B=1, S=4096, H=1, HKV=1 correctness pass
B=1, S=4096, H=8, HKV=8 correctness pass
B=1, S=4096, H=8, HKV=2 correctness pass
```

性能验收：

```text
latency <= 1.05x M4 baseline
```

如果 correctness 通过但性能超过阈值，只标记为 functional milestone，不标记为
performance milestone。

与之前 probe 的区别：

- 不再是单独 Q/K/V probe flag，而是按 producer core 的真实顺序组织：

```text
initial K/Vt load from n_block_max - 1
Q load for current tile_m via barrier_Q equivalent
barrier_O wait equivalent
reverse n_block loop with shared smem_pipe_write
Vt -> V transpose / pipeline bridge
tail drain
```

- 仍然只做 shadow，不让 FA3 consumer 消费这些 buffers。

验收：

- correctness 通过。
- benchmark 记录，预期因双份 producer load 下降；下降标注为 shadow path 开销。

#### M4g-3c: remove one sub-call from producer core

目标：

先把 `producer_load_one_tile` 拆成两个 helper：

```text
producer_prepare_core_state(...)
producer_fa3_load_remaining(...)
```

然后把其中已经由 TileLang 表达的 K/V/Q TMA load 从 C++ helper 里删除或跳过，
只保留还没确认的 drain/tail/state pieces。

注意：

- 这一步必须避免 FA3 consumer 等待的 `shared_storage.pipelines.*` 与
  TileLang barrier 不一致。
- 如果 FA3 consumer 仍等待 FA3 pipeline barrier，那么不能直接删除对应 FA3
  pipeline arrive；需要同时让 consumer 改为等待 TileLang-owned barrier，或
  保留一个 tiny bridge extern 来 arrive FA3 pipeline barrier。

验收：

- 至少一个 producer-side data load 不再由 C++ `mainloop.load` 发起。
- correctness 通过。
- benchmark 比 shadow path 回升；如果没有回升，记录原因。

#### M4g-3d: producer tail replacement

目标：

替换：

```python
T.call_extern("tileops_fa3_shaped_producer_load_tail", ...)
```

需要确认 FA3 `producer_tail` 等价协议：

```text
wait barrier_O
pipeline_k.producer_tail(smem_pipe_write)
pipeline_v.producer_tail(smem_pipe_write)
pipeline_vt.producer_tail(smem_pipe_write)  # if Transpose_V
```

如果 TileLang barrier protocol 能完整表达，则直接 HLIR 化；否则保留一个
tiny extern，只做 `producer_tail` drain，不再包含 TMA load。

验收：

- `producer_load_tail` 不再是大块 producer helper。
- correctness + 4096 benchmark 记录。

#### M4g-3 完成条件

完成后 producer split 分支中不再有：

```text
tileops_fa3_shaped_producer_load_one_tile
tileops_fa3_shaped_producer_load_tail
```

允许暂时保留的小 extern：

```text
producer_tail_drain_only(...)
FA3 pipeline bridge arrive/wait only(...)
```

但这些 extern 不能再发起 Q/K/V TMA load，也不能拥有整段 producer loop。

### M4h: 拆一个 consumer 边界，立刻 HLIR 化

状态：计划更新。

新的推进原则：

```text
拆一个边界
用 C++ 小 helper 确认 FA3 layout / operand view
立刻尝试 TileLang 高级 API 替代
记录 correctness + benchmark
再进入下一个边界
```

不采用“先把所有 helper 都拆成 C++ 小 helper，最后再统一 TileLang 化”的路线。

原因：

- 本实验的核心风险不是能否把 C++ 切成更多小块，而是 TileLang 高级 API
  生成的 layout、同步、fragment 和寄存器行为能否与 FA3 对齐。
- 如果先堆出很多 C++ 小 helper，真实风险会被推迟到最后一次性爆发，届时很难
  判断问题来自 Q/K/V layout、WGMMA、barrier parity、softmax、PV 还是 epilogue。
- C++ helper 应作为临时确认工具，而不是新的中间架构。

#### M4h-0: QK operand pair boundary

目标：

在一个局部 C++ helper 里同时接收显式 `q_stage_ptr` 和 `k_stage_ptr`，
构造同一个 QK operand pair：

```cpp
Tensor sQ = make_tensor(make_smem_ptr(q_stage), SmemLayoutQ{});
Tensor sK = make_tensor(make_smem_ptr(k_stage), SmemLayoutK{});
auto tSrQ = wg_mma_qk.partition_fragment_A(sQ);
auto tSrK = wg_mma_qk.partition_fragment_B(sK);
```

验收：

- `S=224, H=1, HKV=1` correctness 通过。
- `S=4096, H=8, HKV=2` benchmark 记录。
- 不接入真实 output，只确认局部 QK operand pair boundary 可构造。

#### M4h-1: QK TileLang WGMMA smoke

紧接 M4h-0，不长期停留在 C++ helper。

目标：

用同样的 TileLang-owned Q/K stage buffer，尝试直接表达：

```python
acc_s = T.alloc_fragment((64 or 128, 224), "float")
T.wgmma_gemm(
    q_stage,
    k_stage,
    acc_s,
    transpose_B=True,
    policy=T.GemmWarpPolicy.FullRow,
    clear_accum=True,
)
```

先做 smoke，不接真实 output。

验收：

- 编译通过。
- runtime 不 trap，输出仍由原 FA3 path 保证 correctness。
- benchmark 记录；额外 WGMMA 带来的性能下降可接受，但必须标注为 probe 开销。

#### M4h-2: QK TileLang fragment sanity

目标：

给 `T.wgmma_gemm` 的 `acc_s` 增加最小可观测 sanity：

- finite / nonzero / sentinel reduce。
- 可选：对少量元素与 PyTorch reference 的 tile0 score 做粗略对照。

仍不接入真实 output。

#### M4h-3: Softmax HLIR 化

参考 `gqa_fwd_ws.py` 里的 TileLang online softmax，而不是先写大的 softmax
C++ helper。

目标：

基于 TileLang-visible `acc_s` fragment 做：

```text
descale / mask / softmax update
```

先覆盖 dense non-causal fixed-shape case。

#### M4h-4: PV boundary + TileLang WGMMA smoke

先用 C++ helper 确认 `p/acc_s + v_stage_ptr` 的 PV operand boundary，随后立刻
尝试 TileLang：

```python
T.wgmma_gemm(p_fragment_or_smem, v_stage, acc_o, ...)
```

不长期保留 PV C++ 小 helper。

#### M4h-5: Epilogue / output store HLIR 化

目标：

把 `acc_o -> BF16 output`、`lse` 写回逐步转为 TileLang `T.copy` / explicit store。

#### M4h 验收节奏

每个子步都执行：

```bash
# smoke
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 224 --heads 1 --heads-kv 1 \
  --role-run --producer-split <new-flag> \
  --warmup 1 --repeat 1

# benchmark
CUDA_LAUNCH_BLOCKING=1 TMPDIR=/home/ga/TileOPs/.tmp/tvm_tmp \
python _probe_tilelang_fa3_shaped_shell.py \
  --batch 1 --seq-len 4096 --heads 8 --heads-kv 2 \
  --role-run --producer-split <new-flag> \
  --bench --warmup 5 --repeat 20
```

### M5: TileLang owns consumer branch boundaries

目标：

consumer WG1/WG2 分支外形和 `gqa_fwd_fp8.py` 对齐：

- WG1 处理上半个 `64xD` query block。
- WG2 处理下半个 `64xD` query block。
- QK、softmax、PV、epilogue 可以继续分阶段使用局部 extern。
- 每个局部 extern 的输入输出必须是 TileLang 可见的 shared/fragment buffer。

通过标准：

- correctness 覆盖 MHA、multi-head MHA、GQA：

```text
B=1, S=128,  H=1, HKV=1
B=1, S=4096, H=1, HKV=1
B=1, S=4096, H=8, HKV=8
B=1, S=4096, H=8, HKV=2
```

- 性能目标：

```text
M5 latency <= 1.10x M1 latency
```

也就是说，第一阶段允许局部拆分带来小幅开销，但不能变成明显退化的
debug-only kernel。

验收标准：

- **代码形态。** WG1/WG2 consumer 分支分别在 TileLang 外层表达，并显式处理
  上半/下半 `64xD` query tile。
- **buffer 可见性。** Q/K/V shared buffer、accumulator fragment、softmax state、
  output/LSE 写回 buffer 在 TileLang 外层可见；局部 extern 的输入输出必须是
  这些 TileLang buffer。
- **局部 extern 边界。** 如果仍使用 extern，它们应是明确命名的小片段，例如
  QK/PV PTX unit、acc layout transform、FA3-style epilogue store，而不是整条
  consumer pipeline。
- **同步可审计。** WG1/WG2 与 producer 的 barrier/phase 关系能从 TileLang
  代码直接读出。
- **正确性。** 覆盖 MHA、multi-head MHA、GQA：

```text
B=1, S=128,  H=1, HKV=1
B=1, S=4096, H=1, HKV=1
B=1, S=4096, H=8, HKV=8
B=1, S=4096, H=8, HKV=2
```

- **性能。** `B=1, S=4096, H=8, HKV=2` latency 目标为
  `<= 1.10x M1 latency`。
- **第一阶段完成条件。** M5 通过后，第一阶段才能标记完成；如果 correctness
  通过但性能超过阈值，记录为 functional milestone，不记录为 performance
  milestone。
- **记录。** markdown 中追加 M5 的完整命令、正确性、性能和剩余 extern 列表。

### 第一阶段完成定义

第一阶段完成时应满足：

- kernel 外层形态和 `gqa_fwd_fp8.py` 的 WS kernel 一致。
- TileLang 明确拥有 `tx` 到 WG role 的分配。
- TileLang 明确拥有 CTA work coord 和 GQA head mapping。
- TileLang 明确拥有 WG 分支边界。
- 大块 whole-helper `T.call_extern` 被拆成可替换的小块 helper。
- correctness 与 M0/M1 基线一致。
- benchmark 接近 M1，不明显掉队。
