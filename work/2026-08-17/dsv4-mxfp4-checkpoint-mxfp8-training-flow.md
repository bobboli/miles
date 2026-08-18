# DeepSeek-V4：从发布 checkpoint 启动 MXFP8 训练

## 结论

DeepSeek-V4 发布的 Hugging Face checkpoint 可以作为 MXFP8 训练的初始权重来源，但不能把其中的 packed MXFP4 tensor 原样作为 Megatron 的可训练参数。

重复实验的推荐流程仍是：

```text
官方 hybrid FP8/MXFP4 HF checkpoint
  -> NVIDIA Megatron-Bridge 在导入时反量化
  -> BF16 Megatron 参数
  -> 一次性保存 model-only torch_dist checkpoint
  -> 后续实验复用 torch_dist
  -> Transformer Engine MXFP8 training recipe
```

这条流程同时满足两个目标：

- 不再落盘约 567 GB 的 BF16 HF 中间 checkpoint；
- 不把 HF 导入和分布式 checkpoint 生成成本搬进每一次 fresh training startup。

对于 P1/P2 的首次启动，现在也支持完全不落盘转换产物的 direct-HF 路径：

```text
官方 hybrid FP8/MXFP4 HF checkpoint
  -> Megatron-Bridge 在 trainer 初始化时直接反量化
  -> BF16 Megatron 参数 + MXFP8 training recipe
  -> trainer 直接生成 rollout 所需的 FP8/MXFP8/MXFP4 payload
  -> metadata-only 或官方 config 只负责 SGLang 参数布局
```

direct-HF 适合 smoke、一次性实验和消除离线转换依赖；重复扫参仍建议复用一次性生成的 model-only `torch_dist`，避免每个 fresh run 重付 HF 导入成本。

截至 2026-08-18，Miles 已实现 direct-HF + trainer-owned rollout。本机
8×B300 的零训练步初始同步和 OCI 32-GPU 的 P1/P2 单步训练 smoke 均已
通过；两个 OCI 作业都从官方 HF checkpoint 直接初始化 trainer，没有读取
离线 BF16 HF 或 `torch_dist` seed。现有两阶段转换仍保留为 fallback；多步
训练、数值一致性、save/resume 和 R3 验证尚未完成。

## 2026-08-18 实现状态

| 项目 | P1：MXFP8 rollout | P2：FP8 + MXFP4 expert rollout |
|---|---|---|
| Trainer 初始化 | 官方 HF 直接导入，Megatron 参数为 BF16，训练 recipe 为 MXFP8 | 同左 |
| SGLang 启动布局 | 从官方 checkpoint header 生成约 6.2 MB metadata-only MXFP8 schema；无 tensor/index payload | 使用官方 config/量化布局，但 `load_format=dummy`，不读取 checkpoint tensor |
| 首次权重来源 | trainer 在线量化为 MXFP8 | trainer 在线量化；non-routed 为 FP8，routed experts 为 packed MXFP4 |
| 本机结果 | 两个 TP4 engine 完成全部 bucket，`end_weight_update`/`continue_generation` 为 200，容器退出 0 | 同样完整通过；MoE backend 为 `flashinfer_mxfp4` |
| OCI no-R3 smoke | job `501870`，一轮 rollout/train + 训练后第二次 hot reload，`COMPLETED 0:0` | job `501657`，同样完成，`COMPLETED 0:0` |

本机验证使用单节点 8×B300、两个 TP4 colocated SGLang engine、零 optimizer step。P1 初始同步约 285--287 秒；P2 初始同步约 615--617 秒，其中约 358 秒是第一次 MHC/TileLang 冷编译。P2 的模型常驻约 46.3 GB/rank，在线更新期间峰值约 170 GB/rank，未发生 OOM。

本机验证证明了以下链路可以工作：官方 HF direct load、Bridge 名称/layout
转换、trainer-owned dummy rollout、原子 weight/scale bucket、MXFP4
restore/repack 生命周期以及更新结束后的恢复生成接口。

OCI smoke 在此基础上继续覆盖了一次真实 optimizer step 和第二次
trainer-to-rollout hot reload。两个作业都显式设置 `enable_r3=False`：

| Phase | OCI job | 首次 update | `compute_log_prob` | 完整 train | 第二次 update | 总状态 |
|---|---:|---:|---:|---:|---:|---|
| P1，MXFP8 rollout | `501870` | 428--430 s | 688 s | 1520--1522 s | 225--226 s | `COMPLETED 0:0`，59:25 |
| P2，FP8 + packed-MXFP4 experts | `501657` | 388--389 s | 667 s | 1492--1495 s | 163--164 s | `COMPLETED 0:0`，54:57 |

P1 使用 metadata-only MXFP8 schema；它只有 config/tokenizer 和从
safetensors header 提取的布局信息，不含 tensor 或 weight index。P2 直接使用
官方 config。两者都让 SGLang `load_format=dummy`，实际权重只来自 trainer
在线更新，因此这里的“不需要离线转换”准确指不需要生成或读取离线权重
artifact，而不是 P1 连 metadata 布局文件都不需要。

rollout 不是空跑：P1 的 `raw_reward=0.546875`、
`truncated_ratio=0.558594`，P2 分别为 `0.710938` 和 `0.371094`。P2
不再复现修复前接近 100% 截断和零 reward 的坏生成。单步结果的
`train_rollout_kl` 仍为 P1 `0.1267`、P2 `0.1310`，不能据此宣称数值 parity
已经解决。

这次 smoke 仍不证明两个以上 optimizer step、第二次 hot reload 后再次生成、
save/resume、跨拓扑 load 或 R3。两个作业在 Ray 已成功之后的进程 teardown
阶段仍打印 CUDA IPC shared-memory unlink 和 W&B broken-pipe 栈，但 Ray 报告
success，Slurm 均以 `0:0` 完成；这是待清理的退出路径问题，不是运行期失败。

相关实现边界：

- `scripts/run_deepseek_v4.py` 用 `init_model_source={hf,torch_dist}` 和 `rollout_weight_source={trainer,checkpoint}` 显式区分 trainer seed 与 rollout layout；
- `miles/utils/hf_rollout_schema.py` 只读取 safetensors header，为 P1 构造无权重 payload 的 MXFP8 schema；
- `miles_plugins/megatron_bridge/deepseek_v4.py` 补充 Miles DSV4 attention、HC、compressor 和 indexer 参数映射；
- `HfWeightIteratorBridge` 保证同一个 Megatron 参数产生的 weight/scale，以及跨参数的 DSV4 atomic group，不会被 bucket 边界拆开；
- SGLang TRT-LLM MXFP4 hot-reload 修复在内部 MR `lbo/sglang!1`；R3
  `HashTopK` routed-expert capture 已拆到独立 MR `lbo/sglang!2`，本次 smoke
  没有应用它；
- Megatron-Bridge DSV4 direct-HF 兼容修改已推送到
  `miles-dsv4-direct-hf`，验证 commit 为 `ca1a03de`；Miles 验证 commit 为
  `5cb48105e`。

## 不同精度分别代表什么

checkpoint 存储格式、Megatron 模型参数精度、optimizer master 精度、训练计算精度和 rollout 存储格式是不同层次：

| 层次 | 计划中的精度或格式 |
|---|---|
| 发布 checkpoint | Routed experts 为 packed MXFP4；其他量化权重主要为 E4M3 FP8；另有 BF16/FP32 参数 |
| Megatron 前向/反向参数 | BF16 |
| Distributed optimizer main/master 参数 | 通常为 FP32，并按 DP 分片 |
| Transformer Engine GEMM operand | 按 `--fp8-recipe mxfp8` 临时量化为 MXFP8 |
| Rollout checkpoint/update payload | 非 routed 权重为 FP8，routed experts 为 packed MXFP4，并携带各自 scale |

MXFP8 recipe 控制受支持训练 GEMM 的计算路径，不表示 optimizer 直接更新 packed MXFP4，也不表示模型以 MXFP8 master weight 形式常驻。

启用 `--rematerialize-param-from-master-weight` 时，Miles 可以从 optimizer 的 FP32 main parameter 重新生成 BF16 model parameter，从而省去 actor BF16 参数的 pinned CPU backup。它不改变上述精度分层，也不会消除训练时的 BF16 model parameter。

## 发布 checkpoint 的格式

DeepSeek-V4-Flash 发布 checkpoint 是混合量化格式：

| 参数 | checkpoint 中的存储格式 |
|---|---|
| Routed MoE experts | packed MXFP4 E2M1；每个 byte 保存两个 4-bit 元素；每 32 个 logical values 配一个 UE8M0 scale |
| Attention projection、shared expert 等量化权重 | E4M3 FP8；配 128×128 block scale，scale dtype 可能是 E8M0 或 F32 |
| Norm、embedding、router 和部分结构参数 | BF16 或 FP32 |

新版 Megatron-Bridge 的 DeepSeek-V4 importer 根据原始 weight dtype、对应 sibling `.scale` 以及 DSV4 checkpoint schema 选择解码路径：packed `int8` payload 走 MXFP4 解码，E4M3 payload 走 FP8 解码。这里判断的是**发布 checkpoint 中源 tensor 的编码格式**，不是训练 recipe 或 rollout 精度。

真实 checkpoint 中可以从 geometry 看出两种编码的差异：

```text
block FP8:  weight [2048, 4096], scale [16, 32]
packed MXFP4: weight [2048, 2048], scale [2048, 128]
```

MXFP4 weight 的最后一维是 logical K 的一半，而 scale 的最后一维是 `logical_K / 32`，因此相对于 packed K 的比例是 16。这个 geometry 是对 checkpoint 编码的说明，不是训练时的 scale granularity 配置。

旧 fallback 工具 `tools/fp8_cast_bf16.py` 也需要判断源 tensor 格式。它使用 dtype、scale shape 和 tensor geometry 区分历史 block-FP8 checkpoint 与发布版 packed MXFP4 checkpoint；名字或“一字节 dtype”本身不足以完成区分。推荐路径改用 Megatron-Bridge 后，不再由训练启动流程调用这个脚本。

## 当前 fallback 流程

当前 `scripts/run_deepseek_v4.py::full_train` 使用两个准备阶段：

```text
官方 hybrid HF checkpoint
  -> _prepare_single()
  -> BF16 HF checkpoint（约 567 GB）
  -> _prepare_spmd()
  -> model-only Megatron torch_dist checkpoint
  -> training
```

- `_prepare_single()` 调用 `tools/fp8_cast_bf16.py`，将 MXFP4/FP8 权重反量化并生成完整 BF16 HF artifact。
- `_prepare_spmd()` 调用 `tools/convert_hf_to_torch_dist.py`，把 BF16 HF 权重导入 Megatron 并保存 `torch_dist`。
- `full_train` 会检查已有产物并跳过这两个阶段，因此当前流程在一组重复实验中通常只支付一次转换成本。

当前 converter 使用的是旧 `ISEEKYAN/mbridge` package。这条代码可以继续作为迁移期间的 fallback，但不应继续在其中新增 DeepSeek-V4 量化导入功能。

## 推荐流程：release 直接生成 torch_dist

推荐把两个准备阶段合并为一个可缓存的转换阶段：

```text
官方 hybrid FP8/MXFP4 HF checkpoint
  -> Megatron-Bridge DeepSeek-V4 importer
       - 解析 DSV4 原生参数名
       - 同时读取 weight 和 sibling scale
       - MXFP4/FP8 分流反量化为 BF16
       - 写入分布式 Megatron model
  -> 一次性保存 model-only torch_dist checkpoint
  -> 所有后续 fresh runs 复用该 checkpoint
```

这就是应采用的“第三条路”：删除 BF16 HF 中间产物，但保留一次性、可复用的 Megatron checkpoint。

### 为什么使用 Megatron-Bridge

旧 [`ISEEKYAN/mbridge`](https://github.com/ISEEKYAN/mbridge#important) 已明确进入 deprecated 状态，并声明不再支持新模型。DeepSeek-V4 的新功能应基于 [`NVIDIA-NeMo/Megatron-Bridge`](https://github.com/NVIDIA-NeMo/Megatron-Bridge)。

新版 Megatron-Bridge 已经提供我们需要的 DSV4 导入语义：

- [`DeepSeekV4Bridge.maybe_modify_loaded_hf_weight`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py#L568-L583) 在导入阶段处理量化 weight；
- [`maybe_dequantize_hf_quantized_weight`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11/src/megatron/bridge/models/conversion/quantization_utils.py#L407-L435) 处理 packed MXFP4 和 block FP8；
- [DeepSeek-V4 文档](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11/examples/models/deepseek_v4/README.md#L47-L52) 明确说明导入不需要外部反量化脚本。

因此无需在旧 `miles_plugins/mbridge/deepseekv4.py::_weight_to_mcore_format()` 中重新实现 weight/scale pairing、native-name remap 和反量化。这些职责已经属于新版 Megatron-Bridge 的模型 importer。

### Miles 当前的双栈状态

Miles 目前同时安装并使用两个不同项目：

| 路径 | 当前 package | 状态 |
|---|---|---|
| `tools/convert_hf_to_torch_dist.py` | `mbridge.AutoBridge` | 旧 converter，保留为 fallback |
| `model_provider.py` 和 `checkpoint.py` 的 bridge mode | `megatron.bridge.AutoBridge` | 新 runtime 路径，应作为 DSV4 的目标实现 |
| `docker/Dockerfile` | 同时安装两者 | 迁移期间的双栈 |

当前 pin 的 `radixark/Megatron-Bridge` revision 早于 upstream DeepSeek-V4 import 支持。落地时需要：

1. 将 upstream DSV4 bridge/import changes 移植到 Miles 当前使用的 fork，或升级到包含这些改动的兼容 revision；
2. 保留当前 fork 中的 Transformer Engine grouped-linear 修复；
3. 验证新 revision 与 Miles 当前 Megatron-LM、Transformer Engine 和 DSV4 model provider 的兼容性；
4. 将一次性 `release -> torch_dist` converter 迁到 `megatron.bridge` API；
5. 旧 `mbridge` 仅服务尚未迁移的模型，不再承载新的 DSV4 实现。

不应为了获得 importer 而未经验证地替换整套 Megatron-LM。先对 Miles 当前 MCore 做 import-only compatibility test；如果新版 Bridge 依赖当前 fork 缺少的 API，再同步移植所需的最小 MCore changes。

## 已实现流程：trainer 直接从 HF 初始化

Miles 的 direct-HF 入口如下：

| 环节 | Miles 入口 |
|---|---|
| 从 HF config 构建 model provider | `miles/backends/megatron_utils/model_provider.py::get_model_provider_func` |
| 从 HF 加载初始权重 | `miles/backends/megatron_utils/checkpoint.py::_load_checkpoint_hf` |
| 传递 TP/PP/EP 和 MXFP8 配置 | `model_provider.py::_apply_bridge_runtime_config` |
| 从 Miles checkpoint 恢复 | 常规 Megatron `--load` / `latest_checkpointed_iteration.txt` 路径 |

direct-HF 模式可表达为：

```text
fresh run:
  official HF -> Megatron-Bridge import/dequant -> BF16 model -> training

resume:
  saved Miles torch_dist + optimizer state -> training
```

它不会在 resume 时重复导入 HF，但每一个没有可复用训练 checkpoint 的独立 fresh run 都要重新完成 HF 读取、解量化和分布式装载。因此：

- 适合 loader smoke、logit parity 和一次性实验；
- 不适合作为大规模参数 sweep 的默认初始化方式；
- 默认实验流程应先生成并复用 model-only `torch_dist`。

`--init-model-source` 显式区分 `hf` 和 `torch_dist`；`--rollout-weight-source` 独立区分 `trainer` 和 `checkpoint`。不能根据 `--hf-checkpoint` 是否存在推断，因为 rollout 初始化也需要一个 Hugging Face config/layout path。

## `_prepare_single()` 与 `_prepare_spmd()` 的成本

这两个阶段必须分开评估：

| 阶段 | 当前作用 | 推荐方案中的变化 |
|---|---|---|
| `_prepare_single()` | release -> BF16 HF；生成约 567 GB 中间 artifact | 删除；反量化并入 Megatron-Bridge import |
| `_prepare_spmd()` | BF16 HF -> model-only torch_dist | 被 release -> torch_dist 的一次性转换替代，而不是搬进每次训练启动 |
| direct-HF fresh startup | 在 trainer 启动时完成 import/dequant | 只作为可选路径；每个独立 fresh run 都会支付 |

现有记录不足以支持“`_prepare_spmd()` 约 10 分钟 × 8 节点”这个数字。已知的约 10 分钟记录来自单节点任务中的组合准备过程，不是单独的八节点 `_prepare_spmd()` 测量。实现后应分别记录：

- `_prepare_single()` 的 wall time、node/GPU-minutes、峰值内存和磁盘写入；
- 旧 BF16 HF -> `torch_dist` 的同组指标；
- 新 release -> `torch_dist` 的同组指标；
- direct-HF fresh startup 增加的启动时间。

### `torch_dist` 是否依赖转换拓扑

Miles 的 model-only `torch_dist` 使用 Megatron distributed checkpoint，设计上可在不同 TP/PP/EP 下 reshard；执行 converter 时仍需要一个并行拓扑让模型能够装载并完成分布式读写，但这不等于生成的 model-only checkpoint 被锁定到该拓扑。

DSV4 自定义 pipeline layout 仍需做一次跨拓扑加载测试后再宣称支持具体组合。另一方面，包含 disk-streamed optimizer state 的训练 resume 是 same-topology 约束，这是另一个问题，不能和 model-only seed checkpoint 混为一谈。

## Rollout 和在线权重更新

rollout engine 可以直接从官方 hybrid checkpoint 初始化：

```text
官方 checkpoint
  -> FP8 non-routed weights + MXFP4 routed experts rollout
```

训练更新后，rollout 必须收到新权重：

```text
updated BF16 Megatron weights
  -> non-routed weights 量化为 FP8 + scale
  -> routed experts 量化并打包为 MXFP4 + UE8M0 scale
  -> 原子发送 weight/scale groups
  -> 最后一个 bucket 后重建 rollout kernel layout
```

接入新版 Megatron-Bridge 后必须只有一个 rollout 量化 owner：

1. Bridge export BF16，由 Miles 的现有 FP8/MXFP4 processor 完成唯一一次量化；或
2. Bridge 直接输出最终量化 payload，Miles 不再重复调用 quantizer。

当前建议采用第一种方案，因为它对 Miles 已有 online update 路径的改动更小。BF16 export override 只应用于 trainer -> rollout 在线同步；保存完整 HF checkpoint 时应单独决定目标格式。

同一组 weight 和 scale 必须作为原子更新单元；不能在每个 bucket 后执行不可重复的 layout post-processing。

## 后续完整验证顺序

2026-08-18 的 no-R3 smoke 已为 P1/P2 各完成一个 optimizer update 和训练后的
第二次 hot reload。以下是从该结果继续扩展的验证顺序：

1. 将 OCI P1/P2 扩展到至少两个 optimizer updates。
2. 每个 phase 至少完成两次 trainer -> rollout hot reload，并在更新后执行真实 token generation。
3. 检查 direct-HF 导入后的 trainable parameter 中没有残留 packed `int8` 或 checkpoint FP8 payload。
4. 对比 direct-HF 和当前 `release -> BF16 HF -> torch_dist` 路径的参数、logits 和 loss。
5. 验证 MTP on/off、PP/EP layout，以及至少一组跨拓扑 model-only checkpoint load。
6. 保存并恢复包含 optimizer state 的训练 checkpoint；确认 resume 的 Bridge export 仍使用官方 HF config，而不是 metadata-only rollout schema。
7. 增加一次性 `official release -> model-only torch_dist` 转换入口，供重复实验复用。
8. 分别记录三条准备路径的 wall time、node/GPU-minutes、峰值 CPU/GPU memory 和磁盘占用。
9. 完整验证完成前保留当前两阶段 offline conversion 作为 fallback。

## 数值含义

发布 checkpoint 已经包含 MXFP4/FP8 量化误差。反量化得到的是这些量化值在 BF16 中的表示，无法恢复发布前的原始 BF16 权重。

训练又会在支持的 GEMM 中应用 MXFP8 runtime quantization。因此需要分别观察：

1. 发布 checkpoint 反量化后的 logit parity；
2. BF16 trainer 与 MXFP8 trainer 的 logits、loss 和梯度差异；
3. trainer -> rollout 更新后的 token log-prob 或 logits 一致性；
4. 多个 optimizer update 后的稳定性；
5. 多次 rollout hot reload 后的一致性；
6. save/resume 前后的模型和 optimizer 状态。

## 最终目标流程

```text
                     one-time initialization
DeepSeek-V4 hybrid FP8/MXFP4 HF checkpoint
                     |
                     | Megatron-Bridge import + dequantize
                     v
          reusable BF16 model-only torch_dist
                     |
                     | BF16 model params + FP32 optimizer masters
                     | Transformer Engine MXFP8 recipe
                     v
                MXFP8 training compute
                     |
                     | one rollout quantization owner
                     v
       FP8 non-routed + MXFP4 routed-expert rollout
```

目标不是直接训练 packed MXFP4 参数，而是以发布的量化 checkpoint 为初始来源，在 BF16 Megatron 参数上使用 MXFP8 训练计算，并维持量化 rollout。
