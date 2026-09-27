# llama.cpp Metal Flash-Attention 实验与源码分析

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) / GGML 的 Apple Silicon Metal Flash-Attention 可复现实验项目。

本项目在固定源码版本上完成 Metal backend 构建、`FLASH_ATTN_EXT` 正确性验证、`fa-vec` 参数扫描和源码级执行链追踪，重点分析 Flash-Attention 在 Apple GPU 上的调度方式、online softmax、内存层次及融合边界。

> 上游基线：`ggml-org/llama.cpp`
>
> 固定提交：[`930e2fa5995789efbf249a8bf61325bb626e417b`](https://github.com/ggml-org/llama.cpp/commit/930e2fa5995789efbf249a8bf61325bb626e417b)
>
> 实验平台：Apple M5 / Metal 4 / 24 GB unified memory
>
> 实验日期：2026-09-16

本公开仓库是上述上游提交的源码快照，加上实验 README 与 `codex_lab/` 记录。发布提交的 Git 哈希与上游提交不同；上游版本记录在 [`UPSTREAM_COMMIT`](UPSTREAM_COMMIT)。Metal kernel 源码未作个人修改。

## 项目概览

Flash-Attention 的核心价值不只是减少计算量，更重要的是改变 Attention 的计算顺序和中间数据流：将 QK、softmax 与 PV 按 KV tile 组织，在片上共享存储中维护局部结果，避免完整 score/probability 矩阵在 device memory 中反复物化和读取。

本项目围绕三个问题展开：

1. **算得对吗？** Metal `FLASH_ATTN_EXT` 的输出能否在专项测试中与 CPU reference 保持一致？
2. **哪个配置更快？** 对给定 dtype、head size、KV length 和 query rows，`fa-vec` 的 `(Q, NE)` 参数如何影响耗时？
3. **为什么会快？** 从 GGML 图节点到 Metal kernel 的实际调用链是什么，数据在 device、threadgroup 和线程局部层次之间怎样流动？

项目最终形成了完整的证据闭环：

- 固定源码提交与实验环境。
- 只构建任务需要的 Metal tuner 和 backend test。
- 正确性测试与性能扫描分开执行。
- 原始日志、退出码、候选配置和环境快照完整保留。
- 从模型图构建追踪到 `fa.metal` kernel。
- 对性能数字给出明确适用范围，不将局部 benchmark 泛化为端到端模型收益。

## 背景与动机

### Apple Silicon 上的大模型推理

llama.cpp 是一个以 C/C++ 为核心的大模型推理项目，支持 CPU、Metal、CUDA、Vulkan 等多种 backend。在 Apple Silicon 上，CPU 与 GPU 共享统一物理内存，Metal backend 可以直接利用 Apple GPU 执行模型中的矩阵运算、Attention 和其他算子。

统一内存并不意味着所有访问成本相同。GPU 仍然存在 device buffer、cache、threadgroup memory、寄存器候选、同步和并行调度等层次。对于长上下文推理，Attention 需要访问不断增长的 KV cache，数据流和内存带宽往往与算术计算同样重要。

### 为什么研究 Flash-Attention

普通 scaled dot-product Attention 可以写成：

```text
S = Q K^T / sqrt(dk) + mask
P = softmax(S)
O = P V
```

如果 `S` 和 `P` 作为完整矩阵写入 device memory，再由后续算子读回，会产生额外的中间存储与内存流量。Flash-Attention 通过 tiling 和 online softmax 改变执行顺序，在处理每个 KV tile 时直接更新 softmax 状态和输出累加。

因此，本项目不只记录一次 benchmark 数字，而是希望回答：

- llama.cpp 在哪一层创建 Flash-Attention 图节点？
- GGML 如何把抽象 op 分派给 Metal backend？
- vec 与 matrix 路径如何选择？
- `Q`、`NE`、`NSG`、`NWG` 分别控制什么？
- 哪些中间量被避免，哪些 workspace 仍然存在？
- 为什么一个 fused GGML op 仍可能启动多个 Metal kernel？

## 项目定位与范围

### 本项目完成的工作

- 在独立 detached worktree 中固定 llama.cpp 提交。
- 显式启用 Metal、tools 和 tests，关闭不需要的 server 与 examples。
- 构建 `ggml-metal-tuning` 和 `test-backend-ops`。
- 执行过滤后的 `FLASH_ATTN_EXT` backend correctness test。
- 执行额外的 `(Q, NE)` forced-config 数值切片。
- 对 F16、`dk=dv=128` 的 FA vec 路径进行受限参数扫描。
- 保存构建、运行、计时、退出码和环境日志。
- 追踪模型图、GGML op、Metal backend、pipeline、buffer、dispatch 和 shader。
- 分析 online softmax、O/S/M 归约、内存层次和 RoPE 边界。

### 本项目没有声称的内容

- 没有从零发明或实现 Flash-Attention 算法。
- 没有修改 llama.cpp 上游 Metal kernel。
- 没有提交或合并上游 PR。
- 没有将本次 tuner 输出写回源码调优表。
- 没有下载模型执行端到端 token/s benchmark。
- 没有证明所有 dtype、head size、序列长度或 Apple GPU 都有相同收益。

这里的个人工作主要是实验复现、测试验证、参数分析和源码研究；算法与 kernel 实现归属于 llama.cpp / GGML 上游项目。

## 技术流程

项目分析的完整链路如下：

```text
模型 Attention 图构建
    |
    v
ggml_flash_attn_ext 创建 GGML 图节点
    |
    v
GGML_OP_FLASH_ATTN_EXT
    |
    v
Metal backend 支持检查与 op 分派
    |
    v
vec / matrix / sparse 路径选择
    |
    v
pipeline 特化 + device buffer + workspace + threadgroup memory
    |
    v
dispatchThreadgroups
    |
    v
kernel_flash_attn_ext[_vec]
    |
    +---- NWG == 1 ----> 直接写最终输出
    |
    +---- NWG > 1 ----> 写部分 O/S/M
                           |
                           v
                  kernel_flash_attn_ext_vec_reduce
                           |
                           v
                        最终输出
```

对应的主要函数链：

```text
llm_graph_context::build_attn_mha
  -> ggml_flash_attn_ext
  -> GGML_OP_FLASH_ATTN_EXT
  -> ggml_metal_op_flash_attn_ext
  -> ggml_metal_tuning::fa_vec_pick
  -> ggml_metal_encoder_set_pipeline
  -> ggml_metal_encoder_set_buffer
  -> ggml_metal_encoder_set_threadgroup_memory_size
  -> ggml_metal_encoder_dispatch_threadgroups
  -> dispatchThreadgroups
  -> kernel_flash_attn_ext / kernel_flash_attn_ext_vec
  -> kernel_flash_attn_ext_vec_reduce（条件性）
```

## 仓库结构

本仓库保留完整 llama.cpp 源码，同时增加 `codex_lab/` 作为本次实验的记录目录。

```text
llama.cpp-metal-lab/
├── README.md                                  # 本项目 GitHub 首页说明
├── LICENSE                                    # llama.cpp 上游 MIT License
├── CMakeLists.txt                             # llama.cpp 主构建入口
│
├── src/
│   └── llama-graph.cpp                        # 模型图构建与 Attention 路径选择
│
├── ggml/
│   ├── include/ggml.h                         # GGML tensor/op API
│   └── src/
│       ├── ggml.c                             # GGML_OP_FLASH_ATTN_EXT 节点创建
│       └── ggml-metal/
│           ├── ggml-metal-ops.cpp             # Metal op、workspace、buffer 与 grid
│           ├── ggml-metal-device.cpp          # pipeline 名称和 function constants
│           ├── ggml-metal-device.m            # Metal command encoder 与 dispatch
│           ├── ggml-metal-tuning.cpp          # 运行时配置表与 fallback
│           └── kernels/
│               ├── fa.metal                   # FA 主 kernel 与 vec reduce kernel
│               └── rope.metal                 # 独立 RoPE kernel
│
├── tools/tuning/
│   ├── main.cpp                               # tuner 子命令注册
│   ├── fa-vec.cpp                             # workload、候选与稳健选择规则
│   └── bench.cpp                              # warmup、重复运行与中位数计时
│
├── tests/
│   └── test-backend-ops.cpp                   # CPU reference 对比与 forced slice
│
└── codex_lab/
    ├── INTERVIEW_GUIDE.md                     # 个人面试学习与复述手册
    ├── RESULT.md                              # 实验结果摘要
    ├── source_trace.md                        # 七层源码追踪记录
    ├── env_report.txt                         # 设备、系统与工具链快照
    ├── build_configure.log                    # CMake 配置日志
    ├── build.log                              # 编译日志
    ├── test_flash_attn.txt                    # 正确性测试完整日志
    ├── test_flash_attn_exit.txt               # 正确性测试退出码
    ├── fa_vec_sweep.log                       # tuner 完整计时与选择过程
    ├── fa_vec_rows.txt                        # tuner 生成的候选表记录
    ├── fa_vec_exit.txt                        # tuner 退出码
    ├── tuner_help.txt                         # tuner 参数入口记录
    ├── tree_root.txt                          # 仓库顶层结构快照
    └── tree_focus.txt                         # 关键源码目录快照
```

### 关键源码索引

| 层次 | 文件与位置 | 作用 |
|---|---|---|
| 模型图 | [`src/llama-graph.cpp`](src/llama-graph.cpp#L2591-L2705) | 选择 FA 或普通 Attention 图路径 |
| GGML op | [`ggml/src/ggml.c`](ggml/src/ggml.c#L5497-L5538) | 创建 `GGML_OP_FLASH_ATTN_EXT` 节点 |
| vec 判定 | [`ggml-metal-ops.cpp`](ggml/src/ggml-metal/ggml-metal-ops.cpp#L2862-L2870) | 小 query batch 的 vec 路径条件 |
| Metal op | [`ggml-metal-ops.cpp`](ggml/src/ggml-metal/ggml-metal-ops.cpp#L3144-L3755) | workspace、配置选择、buffer 和 dispatch |
| vec kernel | [`fa.metal`](ggml/src/ggml-metal/kernels/fa.metal#L1189-L1806) | QK、online softmax 和 PV 累加 |
| reduce kernel | [`fa.metal`](ggml/src/ggml-metal/kernels/fa.metal#L2284-L2327) | 合并多 workgroup 的部分 O/S/M |
| tuner | [`tools/tuning/fa-vec.cpp`](tools/tuning/fa-vec.cpp#L17-L620) | workload、候选、计时结果压缩和 no-harm 规则 |
| 计时 | [`tools/tuning/bench.cpp`](tools/tuning/bench.cpp#L42-L77) | warmup、同步、中位数与重复 op 摊销 |
| 正确性 | [`tests/test-backend-ops.cpp`](tests/test-backend-ops.cpp#L7649-L7756) | 构造 FA 测试图并与 reference 对比 |
| 强制配置 | [`tests/test-backend-ops.cpp`](tests/test-backend-ops.cpp#L11484-L11583) | `(Q, NE)` forced-config 数值切片 |

## 实验环境

| 项目 | 本次环境 |
|---|---|
| 设备 | MacBook Air |
| SoC | Apple M5 |
| CPU cores | 10 |
| GPU cores | 10 |
| 内存 | 24 GB unified memory |
| GPU backend | Metal 4 |
| 运行时设备 | `MTL0 (Apple M5)` |
| macOS | 26.6.2，arm64 |
| Compiler | Apple clang 21.0.0 |
| CMake | 4.4.3 |
| Build type | Release |
| llama.cpp commit | `930e2fa5995789efbf249a8bf61325bb626e417b` |

这是一次固定环境快照。新的 llama.cpp 提交、不同 macOS/Metal 版本、设备温度和后台负载都可能改变编译行为或计时结果。

## 快速复现

### 1. 确认上游源码版本

```sh
cat UPSTREAM_COMMIT
```

期望输出：

```text
930e2fa5995789efbf249a8bf61325bb626e417b
```

如果提交不同，源码行号、测试集合、kernel 实例和 tuner 规则都可能发生变化。不要将本文结果直接套用到不兼容版本。

### 2. 配置 Metal 构建

```sh
cmake -S . -B build-metal-lab \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
```

关键配置：

- `GGML_METAL=ON`：编译 Metal backend。
- `LLAMA_BUILD_TOOLS=ON`：构建 tuner。
- `LLAMA_BUILD_TESTS=ON`：构建 backend correctness test。
- 关闭 examples 和 server：缩小构建范围，避免与实验目标无关的工作。

### 3. 构建实验目标

```sh
cmake --build build-metal-lab \
  --target ggml-metal-tuning test-backend-ops \
  -j 8
```

生成：

```text
build-metal-lab/bin/ggml-metal-tuning
build-metal-lab/bin/test-backend-ops
```

### 4. 运行正确性测试

```sh
./build-metal-lab/bin/test-backend-ops test \
  -o FLASH_ATTN_EXT \
  -b MTL0
```

本次记录结果：

```text
4951/4951 tests passed
fa_vec (Q,NE) slice: 240 cases run, 0 failed
Backend MTL0: OK
```

退出码记录在 [`codex_lab/test_flash_attn_exit.txt`](codex_lab/test_flash_attn_exit.txt)，完整日志见 [`codex_lab/test_flash_attn.txt`](codex_lab/test_flash_attn.txt)。

### 5. 运行受限 fa-vec tuner

```sh
./build-metal-lab/bin/ggml-metal-tuning fa-vec \
  --dtype f16 \
  --dk 128 \
  --reps 3
```

本项目只扫描 F16、`dk=dv=128`。不带过滤器的完整 sweep 会覆盖更多 dtype 和 head size，不属于本次实验范围。

完整输出见 [`codex_lab/fa_vec_sweep.log`](codex_lab/fa_vec_sweep.log)，压缩后的候选记录见 [`codex_lab/fa_vec_rows.txt`](codex_lab/fa_vec_rows.txt)。

## 正确性验证设计

### 主测试

`test_flash_attn_ext` 构造 Q/K/V/mask 图，在 Metal backend 执行后与 CPU backend reference 比较。测试最大 NMSE 阈值为 `5e-4`。

测试集合覆盖多种：

- head size 与 KV length。
- query rows / batch size。
- mask、sinks、ALiBi 和 logit softcap。
- F16 与量化 K/V。
- KV cache 风格 view。
- GQA 重复维度。
- MLA 风格的 V-as-K-subview。

本次过滤后的测试实际执行 4,951 个用例，全部通过。

日志中另有 349 条 `not supported`。这些是 backend 明确不支持并跳过的输入，不能算作失败，也不能计入通过数。因此正确结论是“实际运行且受支持的 4,951 个用例全部通过”，不是“所有可能输入都通过”。

### `(Q, NE)` forced-config slice

正常运行时，Metal backend 会通过调优表选择 `(Q, NE)`。为了验证非默认候选的数值正确性，测试代码还会强制覆盖配置选择。

切片包含：

```text
shape       = (128,128), (576,512)
KV dtype    = F16, Q4_0
query rows  = 1, 3
KV length   = 512, 4097
sinks       = false, true
Q           = 1, 2, 4
NE          = 对应 shape 的合法值
```

实际执行 240 个 forced-config cases，0 failed。该切片覆盖 query padding、KV padding、多 SIMD-group reduction、量化 K/V 和 MLA view 等代表性分支。

## 参数扫描设计

### Workload 维度

本次固定：

```text
dtype = F16
dk = dv = 128
```

扫描：

```text
KV length  = 2048, 3072, 8192, 32768
query rows = 1, 2, 3, 4, 5, 6, 7, 8, 16
```

因此 workload 数量为：

```text
4 个 KV length x 9 个 query-row 数 = 36 个 workload
```

这里的 9 个值是 query-row 数，不是 9 种 head shape。

### 候选配置

对 `dk=dv=128`，合法候选为：

```text
Q  = 1, 2, 4
NE = 1, 2, 4
```

每个 workload 比较 3 x 3 = 9 种候选，共产生 36 x 9 = 324 个候选计时格。

参数含义：

| 参数 | 含义 |
|---|---|
| `Q` | 每个 threadgroup 同时处理的 query rows 数量 |
| `NE` | 每个线程负责的 head elements 数，影响 SIMD lane 分工 |
| `NSG` | 一个 threadgroup 内的 SIMD-group 数量 |
| `NWG` | 沿 KV 方向并行处理同一输出行的 workgroup 数量 |

`Q>1` 可以让多个 query rows 复用 K/V，但 query rows 不能整除 Q 时会产生 padding。因此不存在对所有 workload 都最优的统一配置。

### 计时口径

`bench.cpp` 的计时流程：

1. 先执行 warmup，同时触发对应 pipeline 的编译。
2. 执行 `ggml_backend_graph_compute`。
3. 调用 `ggml_backend_synchronize` 等待 GPU 完成。
4. 取 `reps=3` 的中位数。
5. 如果同一图中复制了多个 op，则将总时间除以 op 数。

所以日志中的微秒数包含图提交与同步开销，不是纯 shader GPU timestamp。

tuner 会随机打乱候选顺序，并检查 baseline anchor 的漂移；不稳定测量可以被标记为 `DROP untrusted`。本次受限扫描没有出现 untrusted cell 或 cooldown 失败。

## 实验结果

### 构建结果

- CMake 配置退出码为 0。
- `ggml-metal-tuning` 构建成功。
- `test-backend-ops` 构建成功。
- CMake 检测到 Metal framework，并包含 Metal backend。
- 运行时设备为 `MTL0 (Apple M5)`。

### 正确性结果

| 项目 | 结果 |
|---|---|
| 过滤后的 `FLASH_ATTN_EXT` 测试 | 4,951 / 4,951 passed |
| forced `(Q, NE)` slice | 240 cases，0 failed |
| Metal backend 状态 | `Backend MTL0: OK` |
| 进程退出码 | 0 |

### 性能扫描结果

下面列出 query rows 为 16 时，不同 KV length 的代表性结果：

| KV length | Baseline | 最佳局部候选 | 候选耗时 | 相对 baseline |
|---:|---|---|---:|---:|
| 2048 | `Q1NE1`, 502.7 us | `Q2NE4` | 439.2 us | 1.14x |
| 3072 | `Q1NE1`, 825.2 us | `Q2NE4` | 675.0 us | 1.22x |
| 8192 | `Q1NE1`, 2710.9 us | `Q2NE2` | 1881.8 us | 1.44x |
| 32768 | `Q1NE1`, 13402.4 us | `Q2NE4` | 7358.6 us | 1.82x |

本次扫描中的最高局部比值：

```text
13402.4 us / 7358.6 us = 1.821...
```

即在 F16、`dk=dv=128`、KV length 32768、query rows 16 的单个 workload 上，`Q2NE4` 相对 baseline `Q1NE1` 约为 1.82x。

这个数字的准确含义是“同一个 FA vec workload 下，既有候选配置相对 tuner baseline 的单点比值”。它不是：

- Flash-Attention 相对普通 Attention 的总体加速。
- 完整模型的 prefill 或 decode 加速。
- 端到端 token/s、TTFT 或延迟提升。
- 所有 workload 的统一收益。

### 稳健配置选择

tuner 不会简单地把每个单点最快候选写入运行时表。它会把 workload 压缩成 bucket，并应用以下约束：

- 候选在 bucket 的每个已测点上都不能慢于 baseline。
- 必须覆盖 query-row padding 的不利 residue。
- 聚合收益至少达到 1.05x。
- domain default 的 pointwise regret 不能超过阈值。

本次日志中，`Q2NE4` 在多个长 KV workload 上表现较好，但在 `KV=2048, query rows=5` 时比 baseline 慢 1.65%，因此某个 bucket 的该配置被 no-harm 规则拒绝。

这说明单点峰值和可部署默认配置是两个不同问题。调优系统不仅追求最高速度，还要控制未对齐 shape 上的回退风险。

## Flash-Attention kernel 逻辑

### 1. 加载 query tile

vec kernel 将一个或多个 query rows 载入动态 threadgroup memory。`Q` 决定一个 threadgroup 同时处理多少 query rows。

### 2. 分块遍历 KV cache

kernel 沿 KV length 分块，每次只读取当前 K/V tile。每个 SIMD lane 负责 head 维度的一部分，`NE` 改变 lane 与 head elements 的分工方式。

### 3. 计算 QK 与应用 mask

对当前 tile 计算 QK 点积，然后应用 Attention scale。根据 pipeline function constants，还可以应用 mask、ALiBi bias、logit softcap 或 attention sinks。

### 4. online softmax

kernel 为每个 query row 维护：

```text
M: 当前已处理 score 的最大值
S: 在 M 标尺下的指数和
O: 在 M 标尺下的未归一化输出累加
```

处理新 tile 时：

```text
M_new = max(M_old, max(score_tile))
alpha = exp(M_old - M_new)
P_tile = exp(score_tile - M_new)
S_new = S_old * alpha + sum(P_tile)
O_new = O_old * alpha + P_tile * V_tile
```

最终输出：

```text
O_final = O / S
```

`M` 的动态更新避免指数溢出；`alpha` 将旧累加结果转换到新的最大值标尺，使分块计算与完整 softmax 保持数学等价。

### 5. 多 SIMD-group 归约

同一 threadgroup 内的多个 SIMD groups 会先分别处理 KV 子区间，再通过稳定缩放公式合并局部 O/S/M。不同局部结果具有不同最大值，因此不能直接相加。

### 6. 多 workgroup 归约

dense vec 路径通常使用 `NWG=32`，让多个 workgroup 沿 KV 方向并行处理同一输出行。

当 `NWG>1`：

1. 主 kernel 将每个 workgroup 的部分 O/S/M 写入 device temporary buffer。
2. backend 在两个 kernel 之间建立同步边界。
3. `kernel_flash_attn_ext_vec_reduce` 读取 32 份部分结果。
4. reduce kernel 重新对齐最大值，合并分子和分母，写入最终输出。

因此，“fused Flash-Attention”表示核心数学路径按 tile 融合，不表示一次 GGML op 永远只发起一次 Metal kernel launch。

## 内存层次与数据流

| 内存层次 | 主要对象 | 作用 |
|---|---|---|
| Device buffer | Q/K/V、mask、dst、workspace | 跨 kernel 和 threadgroup 访问 |
| Threadgroup memory | Q tile、score/probability tile、mask、部分 O | threadgroup 内共享的片上 scratch |
| Thread-local / register candidate | running M/S、局部点积、局部 O | 单线程或单 lane 的中间状态 |

Apple Silicon 的 unified memory 表示 CPU 与 GPU 共享物理内存体系；Metal shader 中的 `device` 地址空间和 threadgroup memory 仍然是不同的编程与性能层次。

### FA 避免的主要流量

- 完整 `QK^T` score 矩阵写回 device memory。
- 完整 softmax probability 矩阵写回 device memory。
- 后续算子再次读取这些完整中间矩阵。

### 仍然存在的数据

- Q/K/V、mask 和最终输出。
- tile 级 threadgroup scratch。
- KV/mask padding workspace。
- 条件性的量化 KV 转换和 sparse index。
- 多 workgroup vec 路径的部分 O/S/M。
- 独立 reduce kernel 的输入与输出。

因此，更准确的结论是“FA 避免完整 score/probability 矩阵的 device-memory 往返”，而不是“FA 没有中间量”或“所有数据都放在寄存器”。

## vec、matrix、sparse 与 RoPE 边界

### Vec 路径

固定提交中的普通判定为 query batch `<20` 且 head size 能被 32 整除。本项目的 tuner 针对该路径扫描 `(Q, NE)`。

### Matrix 路径

较大的 query batch 可以走 matrix kernel，使用 simdgroup matrix multiply-accumulate。它与 vec 路径有不同的 tile 和 shared-memory 组织。

### Sparse 与辅助 kernel

根据输入条件，backend 还可能启动：

- mask finite-entry sparse index 构建。
- KV/mask tail padding。
- mask block 分类。
- 量化 KV 转 F16。
- vec 多 workgroup reduce。

源码中存在这些 kernel，不表示本次 36 个 workload 全部执行了它们。

### RoPE

RoPE 在 Metal backend 中是独立 GGML op、独立 pipeline 和独立 `rope.metal` kernel。它不属于本次 FA kernel 的内部融合阶段。本项目的证据支持 Attention 内部 QK、online softmax 和 PV 融合，不支持“RoPE 已融合进 FA”的结论。

## 实验产物

| 文件 | 内容 |
|---|---|
| [`codex_lab/RESULT.md`](codex_lab/RESULT.md) | 环境、构建、测试、tuner 与结论摘要 |
| [`codex_lab/source_trace.md`](codex_lab/source_trace.md) | 从模型图到 kernel 的七层追踪 |
| [`codex_lab/INTERVIEW_GUIDE.md`](codex_lab/INTERVIEW_GUIDE.md) | 面试学习、复述和追问手册 |
| [`codex_lab/env_report.txt`](codex_lab/env_report.txt) | 设备、系统、工具链与 Git 状态 |
| [`codex_lab/build_configure.log`](codex_lab/build_configure.log) | CMake 配置日志 |
| [`codex_lab/build.log`](codex_lab/build.log) | 目标编译日志 |
| [`codex_lab/test_flash_attn.txt`](codex_lab/test_flash_attn.txt) | 正确性测试完整输出 |
| [`codex_lab/fa_vec_sweep.log`](codex_lab/fa_vec_sweep.log) | 36 个 workload 的全部候选计时 |
| [`codex_lab/fa_vec_rows.txt`](codex_lab/fa_vec_rows.txt) | tuner 输出的 3 条候选配置记录 |

## 如何阅读本项目

如果只想快速了解项目：

1. 阅读本 README 的“项目概览”“实验结果”和“项目边界”。
2. 打开 [`codex_lab/RESULT.md`](codex_lab/RESULT.md) 查看完整事实摘要。
3. 在 [`codex_lab/fa_vec_sweep.log`](codex_lab/fa_vec_sweep.log) 中核对性能数字。

如果希望理解实现：

1. 从 [`src/llama-graph.cpp`](src/llama-graph.cpp#L2591-L2705) 对比 FA 与普通 Attention 图。
2. 阅读 [`ggml/src/ggml.c`](ggml/src/ggml.c#L5497-L5538) 理解 GGML op 创建。
3. 阅读 [`ggml-metal-ops.cpp`](ggml/src/ggml-metal/ggml-metal-ops.cpp#L3144-L3755) 理解路径选择、workspace 和 dispatch。
4. 阅读 [`fa.metal`](ggml/src/ggml-metal/kernels/fa.metal#L1189-L1806) 理解 vec kernel。
5. 阅读 [`fa.metal`](ggml/src/ggml-metal/kernels/fa.metal#L2284-L2327) 理解多 workgroup reduce。
6. 对照 [`codex_lab/source_trace.md`](codex_lab/source_trace.md) 检查每层结论。

如果准备面试复述，可单独阅读 [`codex_lab/INTERVIEW_GUIDE.md`](codex_lab/INTERVIEW_GUIDE.md)。

## 项目限制

- 结果只对应固定提交和 Apple M5 环境。
- tuner 每点使用 `reps=3`，适合受限实验，不等同于完整统计性能研究。
- 计时包含图提交和同步，不是纯 kernel timestamp。
- 本次只扫描 F16、`dk=dv=128`。
- 没有使用 Metal System Trace 或硬件 counter 验证 cache、带宽、occupancy 和 register spill。
- 没有测量端到端模型的 TTFT、prefill tokens/s、decode tokens/s 或峰值内存。
- correctness test 只证明实际运行的受支持输入，没有覆盖所有可能 shape。

## 后续工作

可以在当前项目上继续扩展：

1. 增加多轮重复和置信区间，进一步控制温度与系统噪声。
2. 扩展 F16 以外的量化 dtype 和更多 `dk/dv` shape。
3. 使用 Metal System Trace 或 GPU counter 验证 K/V 复用、带宽和 occupancy 推断。
4. 对比 FA 与非 FA 的端到端 prefill/decode 指标。
5. 测量不同上下文长度下的峰值内存和临时 workspace。
6. 将 tuner 结果放入独立 patch 中验证运行时表收益，但在充分理解和测试前不向上游提交。

## 上游项目与许可证

本仓库的主体源码来自 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，固定到提交 [`930e2fa5995789efbf249a8bf61325bb626e417b`](https://github.com/ggml-org/llama.cpp/commit/930e2fa5995789efbf249a8bf61325bb626e417b)。

llama.cpp / GGML 源码及本仓库所保留的上游内容遵循 [MIT License](LICENSE)。使用、分发或修改时应保留原版权和许可证声明。

本项目 README、实验日志和分析文档用于技术学习、实验复现与源码研究；没有将 llama.cpp 上游算法或 kernel 实现声明为个人原创成果。

## 致谢

- [llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [GGML](https://github.com/ggml-org/ggml) 的维护者和贡献者。
- Apple Metal、Metal Shading Language 和 Apple Silicon 统一内存架构提供的本地 GPU 计算环境。
- llama.cpp 中的 backend test 与 tuning 基础设施，使算子正确性和配置性能可以分开验证。
