# 面试学习手册：llama.cpp Metal Flash-Attention

> 本文件是个人面试学习、复述与追问手册。仓库对外项目说明见根目录 [`README.md`](../README.md)。对应 llama.cpp 固定提交 `930e2fa5995789efbf249a8bf61325bb626e417b`，实验执行于 2026-09-16。
>
> 这份手册解释项目怎样讲、实验数字怎样解释、面试追问怎样回答，以及哪些话不能夸大；它不是 GitHub 仓库首页 README。

## 0. 先把项目定位说准确

这是一个基于 llama.cpp / GGML 的 Apple Silicon Metal Flash-Attention 实验复现、参数扫描和源码执行链分析项目。

我完成的工作是：

1. 将实验固定在指定 llama.cpp 提交，并在独立 worktree 中构建 Metal backend、专项测试和 tuner。
2. 用 llama.cpp 自带的 CPU reference 与 Metal backend 做数值正确性对比。
3. 在 Apple M5 上扫描 F16、`dk=dv=128` 的 36 个 workload，并比较 9 种 `(Q_cfg, NE)` 内核配置。
4. 从模型图构建一路追踪到 Metal pipeline、buffer、threadgroup dispatch 和 `fa.metal` kernel。
5. 分析 tiled QK、online softmax、PV 累加、内存层次和多 workgroup reduce 的融合边界。

我没有做的工作是：

- 没有发明或从零实现 Flash-Attention 算法。
- 没有修改 llama.cpp 的 Metal kernel。
- 没有把 tuner 结果写回上游调优表，也没有提交 PR。
- 没有下载模型做端到端推理吞吐测试。
- 没有证明所有 dtype、head size、序列长度或 Apple GPU 都有相同收益。

因此，简历中最准确的身份不是“llama.cpp 内核开发者”，而是“个人技术研究 / 实验复现与源码分析”。这个定位并不弱：它体现的是把一个真实大模型系统拆开、建立证据链、设计受控实验并解释底层机制的能力。

## 1. 一句话、30 秒和 2 分钟版本

### 一句话版本

我在 Apple M5 上复现了 llama.cpp Metal Flash-Attention 的正确性与参数调优流程，并从 GGML 图节点追踪到 Metal kernel，解释了 online softmax 如何避免完整注意力矩阵的 device-memory 往返。

### 30 秒版本

我把 llama.cpp 固定到一个可复现提交，在独立 worktree 中只构建 Metal tuner 和 backend test。正确性侧，我运行了过滤后的 `FLASH_ATTN_EXT` 测试，实际执行的 4,951 个用例全部通过，并补充验证了 240 个强制 `(Q_cfg, NE)` 组合。性能侧，我在 F16、`dk=dv=128` 下扫描 4 个 KV 长度和 9 个 query-row 数，共 36 个 workload；局部单点中，候选配置相对 baseline 的最高比值约为 1.82x。随后我从 `build_attn_mha` 追踪到 `fa.metal`，确认 Flash-Attention 在 tile 内完成 QK、online softmax 和 PV，但多 workgroup 路径仍会写临时 O/S/M 并启动 reduce kernel，所以“融合”不等于“只有一次 kernel launch”。

### 2 分钟版本

项目有三条主线。

第一条是可复现性。我没有直接使用原仓库已有 build，而是固定到提交 `930e2fa...`，用 detached worktree 隔离环境，只打开 Metal、tools 和 tests，关闭 examples 与 server。这样实验结果与源码版本一一对应，也不会破坏原仓库。

第二条是正确性和性能证据分离。`test-backend-ops` 会将 Metal 输出和 CPU reference 对比；过滤后的 Flash-Attention 测试实际通过 4,951/4,951，另外 240 个强制内核配置用例也全部通过。tuner 则只回答“哪组内核配置更快”，不提供数值正确性证明。它构造合成 Q/K/V/mask 图，扫描 36 个 workload，每个 workload 比较 9 个 `(Q_cfg, NE)` 候选，计时包含图提交与同步，不是纯 kernel 时间。最高 1.82x 出现在 KV 长度 32768、query rows 16，baseline 约 13402.4 us，候选 `Q2NE4` 约 7358.6 us。

第三条是源码机制。我追踪了模型图层、GGML op、Metal backend、pipeline/buffer 绑定、threadgroup dispatch 和 shader。普通 Attention 会显式形成 `K*Q`、softmax、`V*P` 节点；FA 路径在 KV tile 内维护 running max 和 running sum，并累计输出，从而不需要把完整 score/probability 矩阵写到 device memory。需要注意的是，dense vec 路径通常使用 32 个 workgroup，先写部分 O/S/M，再由独立 reduce kernel 合并。因此它减少的是大规模中间矩阵流量，而不是消除一切临时存储或保证单 kernel。

## 2. 项目边界与证据等级

| 结论 | 证据 | 能说到什么程度 |
|---|---|---|
| 固定版本 | `git rev-parse HEAD` | 仅对应 `930e2fa...` 或兼容代码 |
| Metal backend 可用 | CMake 日志、运行时设备日志 | 本机 Apple M5 上构建和运行成功 |
| 数值正确性 | `test-backend-ops` 与 CPU reference 对比 | 实际运行且受支持的测试形状通过 |
| 4,951/4,951 | `test_flash_attn.txt` | 过滤后的专项套件实际执行用例数，不是整个 llama.cpp 测试集 |
| 240/240 | forced-config slice | 两种 shape、两种 dtype、两种 query rows、两种 KV 长度等组合通过 |
| 最高约 1.82x | `fa_vec_sweep.log` 单个 workload | 本机、本次、局部候选相对 baseline 的比值，不是端到端加速 |
| FA 减少中间矩阵流量 | `fa.metal` 与非 FA 图对照 | 算法与源码路径结论，不等于所有中间量消失 |
| RoPE 未融合进本次 FA | 独立 op、pipeline 和 `rope.metal` | 当前固定提交的源码结论 |

最重要的表达原则是：测试证明的是“本次实际覆盖范围内没有发现数值错误”，tuner 证明的是“某些配置在某些 workload 上更快”。两者不能互相替代。

## 3. 仓库与实验产物地图

```text
llama.cpp-metal-lab/
├── src/llama-graph.cpp                         # 模型层 attention 图构建
├── ggml/include/ggml.h                         # GGML API 和张量约定
├── ggml/src/ggml.c                             # 创建 GGML_OP_FLASH_ATTN_EXT 节点
├── ggml/src/ggml-metal/
│   ├── ggml-metal-ops.cpp                      # Metal op 分派、workspace、buffer、grid
│   ├── ggml-metal-device.cpp                   # pipeline 名称与 function constants
│   ├── ggml-metal-device.m                     # Metal API 编码与 dispatch
│   ├── ggml-metal-tuning.cpp                   # runtime 配置表与 fallback
│   └── kernels/
│       ├── fa.metal                            # Flash-Attention kernel 与 reduce kernel
│       └── rope.metal                          # 独立 RoPE kernel
├── tools/tuning/
│   ├── fa-vec.cpp                              # workload、候选配置与选择规则
│   └── bench.cpp                               # warmup、重复运行、中位数计时
├── tests/test-backend-ops.cpp                  # CPU reference 对比与 FA 强制配置切片
└── codex_lab/
    ├── README.md                               # 本文，项目学习入口
    ├── RESULT.md                               # 实验事实摘要
    ├── source_trace.md                         # 七层源码追踪记录
    ├── env_report.txt                          # 环境与版本快照
    ├── build_configure.log / build.log         # 构建证据
    ├── test_flash_attn.txt                     # 正确性完整日志
    ├── test_flash_attn_exit.txt                # 正确性退出码
    ├── fa_vec_sweep.log                        # 36 个 workload 的完整计时
    ├── fa_vec_rows.txt                         # tuner 压缩后的 3 条候选表记录
    └── fa_vec_exit.txt                         # tuner 退出码
```

推荐阅读顺序：本文 -> `RESULT.md` -> `source_trace.md` -> 对照具体源码 -> 最后看完整日志。不要从 7,000 多行日志开始读。

## 4. 实验环境与复现命令

本次已记录环境：Apple M5、24 GB unified memory、macOS arm64、Apple clang 21.0.0、CMake 4.4.3。环境数据是 2026-09-16 的快照，不保证以后完全一致。

### 4.1 固定提交

```sh
git rev-parse HEAD
# 930e2fa5995789efbf249a8bf61325bb626e417b
```

固定提交的意义不是形式主义。tuner 的候选、fallback 表、kernel 实例和测试集合会随上游变化；如果源码版本变了，行号、结果和解释都可能漂移。

### 4.2 配置与构建

```sh
cmake -S . -B build-metal-lab \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF

cmake --build build-metal-lab \
  --target ggml-metal-tuning test-backend-ops \
  -j 8
```

这里只构建两个目标及其依赖，因为项目问题是 Metal FA 的正确性、调优和源码路径，不需要模型下载、server 或完整 examples。

### 4.3 正确性测试

```sh
./build-metal-lab/bin/test-backend-ops test \
  -o FLASH_ATTN_EXT \
  -b MTL0
```

关键结果：

```text
4951/4951 tests passed
fa_vec (Q,NE) slice: 240 cases run, 0 failed
Backend MTL0: OK
exit_code=0
```

日志中还有 349 条 `not supported`。这些输入被 backend 明确跳过，既不是失败，也不能计入通过数。末尾的 `3/3 backends passed` 包含被过滤后跳过的 BLAS 和 CPU 条目，不能说成三个 backend 都实际执行了 FA 测试。

### 4.4 受限 tuner

```sh
./build-metal-lab/bin/ggml-metal-tuning fa-vec \
  --dtype f16 \
  --dk 128 \
  --reps 3
```

这个过滤非常重要。无过滤的 sweep 会覆盖更多 dtype 和 head size，耗时与热漂移风险都更大，也超出本项目范围。

## 5. Attention 与 Flash-Attention 到底差在哪

### 5.1 数学结果没有变

单头 scaled dot-product attention 可以写成：

```text
S = Q K^T / sqrt(dk) + mask
P = softmax(S)
O = P V
```

Flash-Attention 改变的是计算顺序和中间数据移动，不是 Attention 的数学定义。

### 5.2 普通路径的问题

在固定提交的非 FA 路径中，模型图会显式创建：

```text
K*Q -> scale/bias/softcap -> softmax -> V*P
```

对应 [`src/llama-graph.cpp`](../src/llama-graph.cpp) 的 `2660-2705`。如果完整 `S` 和 `P` 物化到 device memory，随着 query 和 KV 长度增加，中间矩阵流量会变大。

### 5.3 FA 路径的核心

FA 将 KV 划分为 tile。每处理一个 tile，它完成：

```text
QK tile -> scale/mask -> online softmax update -> probability tile * V tile -> output accumulator
```

因此只保留 tile 级 score/probability、running maximum、running sum 和输出累加，而不是完整 `[query, kv]` score/probability 矩阵。

需要精确表述复杂度：FA 不改变主要矩阵乘计算量的数量级，仍要计算 QK 和 PV；它主要降低额外中间矩阵的显存/内存占用与外部内存流量。

## 6. 从模型图到 GPU kernel 的完整执行链

```text
llm_graph_context::build_attn_mha
  -> ggml_flash_attn_ext
  -> GGML_OP_FLASH_ATTN_EXT
  -> backend scheduler 选择 Metal
  -> ggml_metal_op_flash_attn_ext
  -> 选择 matrix / vec / sparse 路径
  -> fa_vec_pick 选择 Q_cfg 与 NE
  -> pipeline 特化 + buffer 绑定 + threadgroup memory
  -> ggml_metal_encoder_dispatch_threadgroups
  -> Metal dispatchThreadgroups
  -> kernel_flash_attn_ext 或 kernel_flash_attn_ext_vec
  -> 必要时 kernel_flash_attn_ext_vec_reduce
```

### 6.1 模型图层

入口是 [`src/llama-graph.cpp`](../src/llama-graph.cpp) 的 `build_attn_mha`：

- `cparams.flash_attn && kq_b == nullptr` 才进入 FA 分支。
- Q/K/V 先做 view/permute。
- 必要时把 F32 K/V 转成 F16。
- 创建 `ggml_flash_attn_ext`，再设置 sinks、KV 上界和 F32 累加精度。
- 如果不使用 FA，则创建显式 matmul、softmax 和第二次 matmul 节点。

这里仍然只是“建图”，没有调用 GPU。

### 6.2 GGML op 层

[`ggml/src/ggml.c`](../ggml/src/ggml.c) 的 `ggml_flash_attn_ext`：

- 检查 Q/K/V 与 mask 的 shape/type 约束。
- 创建 F32 输出张量。
- 把 scale、max bias、logit softcap 写入 op params。
- 设置 `result->op = GGML_OP_FLASH_ATTN_EXT`。
- 将 Q/K/V/mask 挂到 `src[0..3]`。

这一步把 Attention 表达成一个 GGML 图节点，仍然不等于 kernel 已经运行。

### 6.3 Metal backend 层

[`ggml-metal-ops.cpp`](../ggml/src/ggml-metal/ggml-metal-ops.cpp) 的 `ggml_metal_op_flash_attn_ext` 负责真正的 backend 决策：

- 从 op 中取 shape、stride、dtype 与 scale/mask 参数。
- 切分输出后的额外 workspace：pad、block、临时 O/S/M、F16 KV 和 sparse index。
- 小 query batch 且 head size 能被 32 整除时可走 vec kernel。
- 根据 device、dtype、`dk/dv`、KV 长度和 query rows 选择 tuned config。
- 检查 threadgroup memory，超限时回退 baseline。
- 绑定 pipeline、参数、输入输出 buffer 与动态 threadgroup memory。
- 提交主 kernel；如果有多个 workgroup，再提交 reduce kernel。

### 6.4 Metal API 层

backend 的 encoder 最终调用 Metal API：

```text
setComputePipelineState
setBytes / setBuffer
setThreadgroupMemoryLength
dispatchThreadgroups
```

这里才从框架抽象进入真实 GPU command encoding。

### 6.5 shader 层

[`fa.metal`](../ggml/src/ggml-metal/kernels/fa.metal) 包含两条主要实现：

- `kernel_flash_attn_ext`：matrix 路径，使用 simdgroup matrix 操作。
- `kernel_flash_attn_ext_vec`：小 query batch 路径，本次 tuner 的目标。

vec 路径模板参数包含 `DK`、`DV`、`NE`、`Q_cfg` 和每个 threadgroup 处理的 cache items 数。

## 7. 把最容易混淆的参数讲明白

本项目里有两个容易都被叫作 Q 的概念，面试时必须分开。

| 名称 | 源码变量 | 本次取值 | 含义 |
|---|---|---|---|
| head size | `dk`, `dv` | 128, 128 | Key/Query 与 Value 每个 head 的维度 |
| KV length | `ne11` | 2048, 3072, 8192, 32768 | 当前参与 Attention 的 KV cache 长度 |
| query rows | `ne01` | 1,2,3,4,5,6,7,8,16 | 本次图中的 query 行数或小 batch 维度 |
| tuning Q | `Q_cfg`，源码模板参数 `Q` | 1,2,4 | 每个 threadgroup 一起处理多少 query rows |
| tuning NE | `NE` | 1,2,4 | 每个线程负责的 head elements 数，影响 lane 分工 |
| SIMD groups | `NSG` | 1 到 4 | 一个 threadgroup 内的 SIMD-group 数 |
| workgroups | `NWG` | dense vec 通常 32 | 同一个输出行沿 KV 方向并行切分的 workgroup 数 |

“9 种 Query Shape”建议改成“9 个 query-row 数”，因为 head shape 在本次 sweep 中固定为 `dk=dv=128`。每个 workload 又比较 3 个 `Q_cfg` 乘 3 个 `NE`，即 9 个候选配置。

### 7.1 36 个 workload 是怎样来的

```text
4 个 KV length: 2048, 3072, 8192, 32768
x
9 个 query rows: 1, 2, 3, 4, 5, 6, 7, 8, 16
=
36 个 workload
```

每个 workload 测 9 个候选，因此日志中有 36 x 9 = 324 个候选计时。这里的 324 是候选测量格数，不等于 324 个独立正确性用例。

### 7.2 为什么 `Q_cfg > 1` 可能更快

`Q_cfg=2` 或 `4` 让一个 threadgroup 同时处理多行 query，可以对加载的 K/V 做复用，减少重复工作并提高并行利用率。但是当实际 query rows 不能被 `Q_cfg` 整除时会产生 padding 浪费；例如 5 行配 `Q_cfg=2` 会按 6 行处理。因此不存在一个对所有 query-row 数都最优的 `Q_cfg`。

### 7.3 为什么 `NE` 不是“线程数”

`NE` 控制一个线程负责多少 head elements，并通过 `NL = 32 / NE` 改变 SIMD lane 对 head 维度的切分。它会同时影响点积、V 累加、shuffle reduction 和共享内存访问。把 `NE=4` 简化成“线程更多”是不准确的。

## 8. vec kernel 内部到底做了什么

### 8.1 载入 Q 与分配 scratch

kernel 将 Q row 载入 threadgroup memory。共享区被解释为：

- `sq4`：Q tile。
- `ss/ss4`：attention score / probability tile。
- `sm`：mask tile。
- `so4`：局部输出累加。

同一块动态 threadgroup memory 会按阶段和类型重新解释，不能仅凭变量名 `shmem_f16` 就说所有中间量都是 FP16。

### 8.2 分块计算 QK

kernel 沿 KV cache 分块循环。每个线程读取 K 的一部分，与 Q 做点积，再通过 SIMD shuffle/reduce 合并为 tile score。

### 8.3 scale、mask、softcap 和 bias

score 会乘以 Attention scale，并按 function constants 决定是否加入 mask、ALiBi bias 或 logit softcap。pipeline 特化让不需要的分支在编译后 kernel 中消失或简化。

### 8.4 online softmax

对当前 tile 的 score，kernel 维护：

```text
M_old: 已处理 tile 的最大值
S_old: 在 M_old 标尺下的指数和
O_old: 在 M_old 标尺下的未归一化输出累加
```

新 tile 到来时：

```text
M_new = max(M_old, max(score_tile))
alpha = exp(M_old - M_new)
P_tile = exp(score_tile - M_new)
S_new = S_old * alpha + sum(P_tile)
O_new = O_old * alpha + P_tile * V_tile
```

最后输出 `O_new / S_new`。

`M_new` 保证指数不因大正数溢出；`alpha` 把旧累加结果重新缩放到新的最大值标尺。这个递推与对所有 score 一次性做 softmax 数学等价，但只需要 tile 级暂存。

### 8.5 PV 累加

此时共享区中的 `ss` 已不是原始 QK score，而是 exponentiated probability tile。kernel 读取对应 V tile，将 `P_tile * V_tile` 累加到局部 `O`。

### 8.6 SIMD-group 和 workgroup 内归约

多个 SIMD groups 各自得到部分 `(O,S,M)` 后，先在 threadgroup 内按照稳定 softmax 公式合并。合并时不能直接相加两个 O，因为它们可能分别以不同最大值 `M0`、`M1` 为标尺，必须先乘 `exp(Mi - max(M0,M1))`。

### 8.7 多 workgroup reduce

dense vec 路径通常设 `NWG=32`，即同一输出行的 KV 范围由 32 个 workgroup 分摊。主 kernel 先将每个 workgroup 的部分 O/S/M 写到 device temporary buffer，然后 `kernel_flash_attn_ext_vec_reduce`：

1. 找 32 份部分结果的最大 `M`。
2. 将每份 `S` 和 `O` 重新缩放到共同最大值。
3. 合并分子与分母。
4. 写最终归一化输出。

所以准确结论是“Attention 的核心数学步骤按 tile 融合”，而不是“一个 GGML FA op 永远只发一个 Metal kernel”。

## 9. 三层内存模型

| 层次 | 典型对象 | 生命周期 / 作用 | 本项目能确认什么 |
|---|---|---|---|
| device buffer | Q/K/V、mask、dst、workspace | 跨 threadgroup / kernel，可由 GPU 访问 | Apple Silicon 为统一物理内存，但 shader 中仍是 `device` 地址空间 |
| threadgroup memory | Q tile、score/probability tile、mask、部分 O | 单 threadgroup 共享，显式 barrier 同步 | 主机计算大小并用 `setThreadgroupMemoryLength` 分配 |
| thread-local / register candidate | running S/M、局部点积、局部 O 向量 | 单线程或单 lane 使用 | 源码可看见局部变量，但不能证明实际寄存器占用或零 spill |

三个概念不能混用：

- Apple unified memory 是 CPU/GPU 共享物理内存架构。
- Metal `device` 是 shader 地址空间限定符。
- Metal threadgroup memory 是 GPU 线程组共享的片上 scratch。

统一内存不代表 device buffer 自动变成 threadgroup memory，也不代表访问成本相同。

## 10. 为什么 FA 能减少流量，但不是“零中间量”

### 避免物化的主要对象

- 完整 `QK^T` score 矩阵。
- 完整 softmax probability 矩阵。
- 这两个大矩阵在 device memory 的写回与再次读取。

### 仍然存在的对象

- Q/K/V、mask 和最终输出。
- tile 级 threadgroup scratch。
- padding、mask block、量化 KV 转换或 sparse index 等条件性 workspace。
- 多 workgroup vec 路径的部分 O/S/M device temporary buffer。
- 独立 reduce kernel。

因此，面试中应说“减少完整 score/probability 矩阵的 device-memory 往返”，不要说“没有中间量”“完全不访问内存”或“所有东西都在寄存器里”。

## 11. matrix、vec 与辅助 kernel 的边界

### matrix 路径

适合较大的 query batch，使用 simdgroup matrix multiply-accumulate。主 host 代码提交到 `kernel_flash_attn_ext`。

### vec 路径

固定提交中的普通判定是 query batch `<20` 且 head size 可被 32 整除。本次 tuner 针对该路径的 `Q_cfg` 与 `NE` 做配置搜索。

### 条件性辅助 kernel

根据 dtype、mask、padding 与 sparse 情况，还可能出现：

- 量化 KV 预转换到 F16。
- KV/mask tail padding。
- mask block 分类。
- sparse index 构建。
- 多 workgroup O/S/M reduce。

不是每次运行都会调用全部辅助 kernel。源码里存在某个 kernel，不代表本次 36 个 workload 实际走过它。

### RoPE 为什么不算本次 FA 融合的一部分

RoPE 在 `ggml-metal-ops.cpp` 中有独立 op 分支、独立 pipeline，并执行 `rope.metal`。它做位置相关的二维旋转，不具有 FA 的 KV 分块、online softmax 或输出累加接口。因此本项目只能声称 Attention 内部融合，不能声称 RoPE 已融合进 FA kernel。

## 12. 正确性测试是怎样工作的

`test_flash_attn_ext` 构造 Q/K/V/mask，执行目标 backend，再与 CPU backend reference 比较。其允许的最大 NMSE 为 `5e-4`。

测试维度包含：

- 多种 head size、KV length、batch/query rows。
- mask、sinks、ALiBi、softcap 等特性。
- F16 与量化 K/V。
- KV cache 风格的 view。
- GQA 重复维度和 MLA 风格的共享 view。

额外的 forced-config slice 会绕过正常 `fa_vec_pick`，强制测试合法 `(Q_cfg, NE)`。固定切片包括：

```text
shape: (128,128), (576,512)
dtype: F16, Q4_0
query rows: 1, 3
KV length: 512, 4097
sinks: false, true
Q_cfg: 1, 2, 4
合法 NE: 由 head size 决定
```

总计实际运行 240 个 forced-config cases，0 failed。它特别覆盖 query-row padding、KV padding、多 SIMD-group reduction、量化 K/V 和 MLA view 等风险点。

为什么还不能说“所有 shape 都正确”：测试是有限集合，且 backend 会跳过不支持输入；真实模型还可能有不同 dtype、stride、mask、head size 与序列长度组合。

## 13. tuner 怎样计时、怎样选配置

### 13.1 合成图而不是模型推理

`fa-vec.cpp` 直接构造 Q/K/V/mask 张量和一个 `ggml_flash_attn_ext` 节点。这样能隔离 Attention 内核，不需要模型文件，也避免 tokenizer、采样、其他层和 I/O 干扰。

Q 是 F32，K/V 是指定 dtype，本次为 F16；mask 为 F16。K/V 使用较大 parent buffer 的 view，模拟生产中的 KV cache 布局。

### 13.2 计时口径

`bench.cpp` 的流程是：

1. 先 warmup，一并触发 pipeline compilation。
2. 每个样本执行 `ggml_backend_graph_compute`。
3. 调用 `ggml_backend_synchronize` 等待 GPU 完成。
4. 取 `reps=3` 的中位数。
5. 如果为摊薄固定开销复制了多个 op，则除以图中重复 op 数。

因此日志中的微秒数包含图提交与同步等开销，不是通过 GPU timestamp 得到的纯 shader 时间。

### 13.3 热漂移与随机顺序

候选顺序按固定 seed 打乱，减少总把某个候选放在机器更冷或更热阶段的偏差。tuner 还重复检查 baseline anchor，必要时执行 cooldown，并将不可信 cell 标记为 `DROP untrusted`。本次受限日志没有出现 untrusted cell 或 cooldown 失败。

### 13.4 为什么最快单点不一定写入表

tuner 不只是逐点选最小时间，还要把 workload 压缩成可运行时查表的 bucket 配置。候选必须：

- 在 bucket 覆盖的每个已测点上都不慢于 baseline。
- 对 query-row padding 的最坏 residue 有样本证据。
- 聚合速度至少达到 1.05x gate。
- domain default 的 pointwise regret 不超过阈值，否则需要 bucket exception。

日志明确记录：某个 bucket 的 `Q2NE4` 虽然很多点很快，但在 `KV=2048, query rows=5` 上比 baseline 慢 1.65%，因此被 no-harm 规则拒绝。这个细节比只报 1.82x 更能体现你理解调优系统的稳健性。

## 14. 1.82x 到底怎样算

最高局部比值出现在：

```text
dtype       = F16
dk = dv     = 128
KV length   = 32768
query rows  = 16
baseline    = Q1NE1 = 13402.4 us
best local  = Q2NE4 = 7358.6 us
speedup     = 13402.4 / 7358.6 = 1.821...
```

所以简历里写“局部候选配置相对 baseline 单点最高约 1.82x”是准确的。

以下说法不准确：

- “Flash-Attention 整体加速 1.82x”。没有与非 FA 实现做端到端对比。
- “模型推理加速 1.82x”。没有运行模型，也没有测 token/s 或 TTFT。
- “我优化内核获得 1.82x”。没有修改 kernel，做的是既有配置候选扫描。
- “所有 workload 都提升 1.82x”。不同点从 baseline 到约 1.82x 不等。

从结果看，KV 越长、query rows 越多时 `Q_cfg=2` 的 K/V 复用优势更容易覆盖调度与 padding 成本；这是结合源码和数据的合理解释，但没有硬件 counter，因此应称为“机制推断”，不要说成 profiler 已证明。

## 15. 把原简历四条逐条校正

### 原条目 1：Metal 内核验证

建议写成：

> 基于固定提交在 Apple M5 Metal backend 搭建隔离实验环境；过滤后的 `FLASH_ATTN_EXT` 数值测试实际执行 4,951/4,951 通过，另有 240 个强制 `(Q_cfg, NE)` 配置用例 0 failed，并与 CPU reference 对比。

比原文更强的地方：说明了固定版本、测试对象、额外切片和 reference。边界也更清楚：不是整个 llama.cpp 测试集。

### 原条目 2：Kernel 参数扫描

建议写成：

> 针对 F16、`dk=dv=128` 的 FA vec 路径，扫描 4 个 KV length x 9 个 query-row 数，共 36 个 workload；每点比较 9 种 `(Q_cfg, NE)` 配置，局部单点相对 baseline 最高约 1.82x，并分析 bucket no-harm 规则对最终配置的约束。

“9 种 Query Shape”改为“9 个 query-row 数”，避免与 head shape 混淆。

### 原条目 3：执行链路追踪

建议写成：

> 追踪 `build_attn_mha -> GGML_OP_FLASH_ATTN_EXT -> Metal backend -> pipeline/buffer binding -> threadgroup dispatch -> fa.metal`，梳理 vec/matrix 路径、动态 threadgroup memory、workspace 和条件性 reduce kernel。

### 原条目 4：Memory/Fusion 分析

建议写成：

> 对照普通 Attention 图与 FA kernel，分析 device buffer、threadgroup scratch 和 thread-local 数据路径；说明 tiled QK + online softmax + PV 如何避免完整 score/probability 矩阵的 device-memory 往返，并明确多 workgroup O/S/M 临时结果与独立 reduce 的融合边界。

### 项目标题与日期

当前可验证实验日期为 2026-09-16。若没有 2026-06 的独立代码或日志证据，简历日期建议改为 `2026.09`。

标题建议：

```text
llama.cpp Metal Flash-Attention 实验复现与源码分析 | 个人技术研究
```

不建议继续使用“个人开发者”，因为该词容易让面试官理解为你实现或提交了上游代码。

## 16. 与字节 JD 的真实关联

### 强关联能力

| JD 关注点 | 本项目能证明的能力 |
|---|---|
| 大模型训练和应用基础 | 理解 Attention、KV cache、GQA/MLA view、mask、量化 dtype |
| 大模型工程与性能 | 能把图算子追踪到 backend 与 GPU kernel，理解数据移动和调度 |
| 逻辑思考 | 区分 correctness、performance、coverage 和 generalization |
| 执行力与自驱 | 固定版本、隔离 worktree、构建、运行、日志、报告形成闭环 |
| 前沿技术理解 | 能解释 Flash-Attention、online softmax、kernel fusion 的真实边界 |

### 弱关联或未覆盖

这个项目没有直接覆盖：

- 短剧/游戏多模态数据集构建与数据合成。
- CPT、SFT、RLVR、多轮 Agent RL。
- 多模态视频理解或内容生成。
- Agent 架构与 Test-Time Scaling。
- 业务指标，如深转、ROI、用户兴趣挖掘。

因此它适合作为“技术深度和系统能力”项目，而不应伪装成“多模态业务项目”。面试中可以这样连接：

> 这个项目与岗位业务方向不是一一对应，但它训练了我理解大模型系统瓶颈和建立实验闭环的能力。多模态模型和 Agent 在训练后处理、长上下文推理、视频 token 增长时同样面临 Attention 计算与内存压力；我能从模型算子一直定位到 backend 和 kernel，而不是只会调用上层 API。对 JD 中数据、后训练和 Agent RL 的直接经验，我会用其他项目或后续实验补足。

这段回答既建立关联，也不越界。

## 17. 高频追问与参考回答

### Q1：为什么不用 CUDA，而选 Metal？

因为实验设备是 Apple Silicon，Metal 是 llama.cpp 在该平台的原生 GPU backend。目标不是比较 Metal 与 CUDA，而是在我可控的真实设备上完成从图算子到 kernel 的闭环验证。平台不同，但关于 tiling、online softmax、内存层次和稳健 benchmark 的方法可以迁移。

### Q2：Flash-Attention 为什么快？

关键不只是减少 FLOPs，而是改变 I/O。普通路径可能把完整 score 和 probability 矩阵写到外部 device memory；FA 在 tile 内完成 QK、稳定 softmax 更新和 PV 累加，减少大中间矩阵的往返。是否更快仍取决于 shape、dtype、并行度、occupancy 和额外 reduce 成本。

### Q3：online softmax 为什么数值稳定？

每个阶段都以当前最大值 M 为指数基准，所有指数项都不大于 1。最大值变化时，用 `exp(M_old-M_new)` 重新缩放旧的分母和输出累加，因此能与新 tile 在同一标尺下合并。

### Q4：1.82x 是对什么的加速？

是同一个 F16 FA vec workload 下，既有候选 `Q2NE4` 相对 tuner baseline `Q1NE1` 的单点计时比值。它不是 FA 相对普通 Attention，也不是端到端模型推理加速。

### Q5：为什么还需要 reduce kernel？

同一输出行的 KV 范围被多个 workgroup 并行处理，每个 workgroup 只有局部 O/S/M。不同局部 softmax 的最大值不同，必须用稳定缩放公式合并，不能直接求和，所以需要独立 reduce。

### Q6：统一内存是不是意味着没有数据搬运？

不是。统一内存表示 CPU/GPU 共享物理内存体系，但 GPU 仍有缓存、访问层次、同步和带宽成本；shader 的 device buffer 与 threadgroup scratch 也仍然不同。FA 减少的是不必要的大中间矩阵流量。

### Q7：怎么证明 kernel 正确？

使用 `test-backend-ops` 将 Metal 输出与 CPU reference 比较，并运行额外 forced-config slice。测试阈值、实际执行数、跳过数和退出码都保存在日志中。它证明覆盖范围内通过，不证明所有输入。

### Q8：为什么不把 1.82x 的配置直接写进表？

运行时表是按 device、dtype、shape bucket 做稳健选择，不应为一个峰值牺牲其他点。tuner 的 no-harm 规则要求候选在 bucket 内每个已测点都不慢于 baseline；日志中 `Q2NE4` 在一个较短 KV、5 query rows 的点上退化 1.65%，所以不能仅凭峰值直接采用。

### Q9：你的个人贡献是什么？

我的贡献是复现实验环境、建立正确性与性能证据、设计受限扫描范围、解释计时口径、追踪源码执行链并梳理融合边界。Flash-Attention 算法与 Metal kernel 来自 llama.cpp 上游，我没有把它们声称为原创实现。

### Q10：如果继续做，你下一步会做什么？

第一步是用 Metal System Trace 或 GPU counter 验证对 K/V 复用、带宽、occupancy 和 spill 的推断；第二步扩展 dtype/head size 并做重复运行置信区间；第三步加入端到端模型 benchmark，分别报告 TTFT、prefill tokens/s、decode tokens/s 和峰值内存。若要贴近该 JD，还需要另做多模态数据、后训练或 Agent 评测项目，不能靠这个项目替代。

## 18. 面试中绝对不要说的五句话

1. “我实现了 llama.cpp 的 Flash-Attention。”
2. “我的优化让 llama.cpp 推理加速了 1.82x。”
3. “4,951 个测试说明所有输入都正确。”
4. “Flash-Attention 没有任何中间内存，也只有一次 kernel launch。”
5. “Apple 统一内存等于 GPU threadgroup memory。”

对应的准确说法分别是：源码分析与实验复现、局部配置单点收益、有限覆盖、tile 融合但保留必要 workspace/reduce、统一物理内存与 shader memory hierarchy 分层。

## 19. 建议的现场白板结构

如果面试官让你画项目，按下面顺序：

```text
1. 数学：O = softmax(QK^T / sqrt(d) + mask)V

2. 普通路径：
   Q,K -> full score -> softmax -> full probability -> xV -> O

3. FA tile 路径：
   [Q tile, K/V tile]
       -> score tile
       -> running M/S
       -> partial O
       -> next KV tile

4. 工程调用链：
   model graph -> GGML op -> Metal backend -> pipeline/buffer -> dispatch -> kernel

5. 多 workgroup：
   32 partial (O,S,M) -> stable reduce -> final O

6. 证据：
   correctness 4951 + 240
   tuner 4 x 9 workloads, 9 candidates each
   peak local ratio 1.82x, not end-to-end
```

这个顺序先讲数学不变，再讲 I/O 优化，然后落到工程实现和实验边界，逻辑最稳。

## 20. 项目完成标准与当前状态

- [x] 固定提交与隔离 worktree。
- [x] Metal backend、tuner、专项测试构建成功。
- [x] `FLASH_ATTN_EXT` 4,951/4,951 通过。
- [x] forced `(Q_cfg, NE)` slice 240/240 通过。
- [x] 36 workload 受限 sweep 完成，退出码 0。
- [x] 1.82x 单点口径可由原始日志复算。
- [x] 图 -> op -> backend -> dispatch -> kernel 调用链完成。
- [x] device/threadgroup/thread-local 和 fusion/reduce 边界完成。
- [ ] 未做 GPU profiler counter。
- [ ] 未做端到端模型性能测试。
- [ ] 未覆盖多模态数据、后训练或 Agent RL。

项目当前状态：实验与源码分析闭环已完成；简历应将它作为可验证的系统研究项目，而不是原创内核开发项目。

## 21. 一手证据索引

- 实验摘要：[`RESULT.md`](RESULT.md)
- 七层源码追踪：[`source_trace.md`](source_trace.md)
- 正确性日志：[`test_flash_attn.txt`](test_flash_attn.txt)
- tuner 日志：[`fa_vec_sweep.log`](fa_vec_sweep.log)
- tuner 压缩结果：[`fa_vec_rows.txt`](fa_vec_rows.txt)
- 环境快照：[`env_report.txt`](env_report.txt)
- 模型图入口：[`src/llama-graph.cpp`](../src/llama-graph.cpp)
- GGML op：[`ggml/src/ggml.c`](../ggml/src/ggml.c)
- Metal host dispatch：[`ggml-metal-ops.cpp`](../ggml/src/ggml-metal/ggml-metal-ops.cpp)
- Metal kernel：[`fa.metal`](../ggml/src/ggml-metal/kernels/fa.metal)
- tuner workload：[`tools/tuning/fa-vec.cpp`](../tools/tuning/fa-vec.cpp)
- tuner timing：[`tools/tuning/bench.cpp`](../tools/tuning/bench.cpp)
- backend correctness test：[`tests/test-backend-ops.cpp`](../tests/test-backend-ops.cpp)

## 22. 9 月 30 日前的学习路线

不要从背 700 行 README 开始。按“能复述 -> 能画图 -> 能解释数字 -> 能应对反问”推进。

### 第 1 阶段：先建立骨架

目标：不看资料讲清 30 秒版本，并回答三个问题。

```text
算得对吗？4951/4951 + forced slice 240/240。
哪个配置快？36 workload，每点 9 个候选，局部最高约 1.82x。
为什么可能更快？多 query row 复用 K/V + 改变 lane 分工，但受 padding 和 shape 约束。
```

完成标准：能明确说出“1.82x 不是端到端模型加速”。

### 第 2 阶段：掌握 Attention 与 online softmax

目标：手写普通 Attention 公式，并从 `M_old/S_old/O_old` 推导一次 tile 更新。

练习：假设旧最大值为 3，新 tile 最大值为 5，解释为什么旧分母和旧 O 都要乘 `exp(3-5)`。

完成标准：能回答“为什么不能把不同 workgroup 的 O 直接相加”。

### 第 3 阶段：画出六层调用链

目标：不看代码画出：

```text
model graph -> GGML op -> Metal backend -> pipeline/buffer -> dispatch -> shader
```

然后为每层说出一个具体 symbol：

```text
build_attn_mha
ggml_flash_attn_ext / GGML_OP_FLASH_ATTN_EXT
ggml_metal_op_flash_attn_ext
fa_vec_pick + buffer binding
dispatchThreadgroups
kernel_flash_attn_ext_vec
```

完成标准：能解释“建图”和“执行 GPU”不是同一步。

### 第 4 阶段：吃透参数与实验表

目标：把 `ne01`、`ne11`、`Q_cfg`、`NE`、`NSG`、`NWG` 全部讲清。

练习：解释 query rows 为 5、`Q_cfg=2` 时为什么有 padding；复算 `13402.4 / 7358.6`。

完成标准：看到 `Q2NE4` 时不会把 Q 错讲成 Query tensor 或 head shape。

### 第 5 阶段：掌握内存与融合边界

目标：画三层内存表，并列出“避免物化”和“仍然存在”的对象。

完成标准：能主动说出 dense vec 的部分 O/S/M 和独立 reduce kernel，避免把 fused 说成 single-launch。

### 第 6 阶段：对照源码走一遍

按下面顺序各读一个小范围，不需要通读整个 llama.cpp：

1. `src/llama-graph.cpp:2591-2705`。
2. `ggml/src/ggml.c:5497-5538`。
3. `ggml-metal-ops.cpp:3144-3230, 3500-3750`。
4. `fa.metal:1203-1316, 1532-1800, 2287-2323`。
5. `tools/tuning/fa-vec.cpp:17-68, 226-291, 314-382, 408-557`。
6. `tools/tuning/bench.cpp:42-77`。
7. `tests/test-backend-ops.cpp:7649-7756, 11484-11583`。

完成标准：面试官指出任意一层时，你知道去哪个文件找，而不是背所有实现细节。

### 第 7 阶段：模拟追问

用第 17 节的 10 个问题做两轮：

- 第一轮每题限时 60 秒，目标是结论准确。
- 第二轮每题限时 30 秒，目标是先结论、后证据、最后边界。

推荐回答结构：

```text
结论：我观察到什么。
证据：哪条命令、哪段日志或哪层源码。
边界：它没有证明什么。
```

### 第 8 阶段：和 JD 对齐但不硬贴

准备两句话：

1. 这个项目证明你有大模型系统、实验和底层分析能力。
2. 它不直接证明你有多模态数据、SFT/RLVR 或 Agent RL 经验。

若简历上还有更贴近多模态或 Agent 的项目，先讲业务相关项目，再用本项目展示技术深度；若没有，也不要把本项目改名包装成多模态项目。
