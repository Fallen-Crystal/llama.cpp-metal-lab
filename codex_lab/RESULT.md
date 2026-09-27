# llama.cpp Metal Flash-Attention Lab

执行日期：2026-09-16（Asia/Shanghai）。依据用户提供的 `llama_cpp_metal_codex_guide.pdf`，完成固定版本的最小 Metal 实验与源码追踪。

## 1. Environment

- macOS：26.6.2，build 25G83，arm64。
- Apple chip：Apple M5，10 CPU cores / 10 GPU cores，MacBook Air。
- RAM：24 GB unified memory。
- clang：Apple clang 21.0.0 (clang-2100.3.34.2)。
- CMake：4.4.3；Xcode Command Line Tools：`/Library/Developer/CommandLineTools`。
- original checkout HEAD：`c8e03ce8122b7af76f836d53efde6df1ce5ec437`。
- lab checkout HEAD：`930e2fa5995789efbf249a8bf61325bb626e417b`，detached worktree。
- 实验目录：本仓库根目录（公开副本）。
- Metal backend detected：yes。CMake 报告 `Metal framework found` / `Including METAL backend`；实际运行报告 `MTL0 (Apple M5)`，Apple10 / Metal4 GPU family。
- 环境记录：`env_report.txt`；原仓库快照在原仓库 `codex_lab_local_snapshot/env_and_git.txt`。

## 2. Repository map

- `tree_root.txt`：generated。
- `tree_focus.txt`：generated。
- `src/llama-graph.cpp` 构建模型 attention 图；`ggml/include/ggml.h` 与 `ggml/src/ggml.c` 定义 GGML op；`ggml/src/ggml-metal` 支持、分派、绑定与提交 Metal；`kernels/fa.metal` 执行 GPU 算法；`tools/tuning` 使用合成张量调优；`tests/test-backend-ops.cpp` 做 CPU reference 数值对比。
- 原仓库已有 `project_tree.txt`、`tests/.DS_Store` 均保留；未更改原分支、原 build 或任何跟踪源码。新增内容为实验 worktree、构建产物和实验记录。

## 3. Build

- ggml-metal-tuning built：yes。
- test-backend-ops built：yes。
- Release；`GGML_METAL=ON`，`LLAMA_BUILD_TOOLS=ON`，`LLAMA_BUILD_TESTS=ON`，`LLAMA_BUILD_EXAMPLES=OFF`，`LLAMA_BUILD_SERVER=OFF`。
- 仅构建 `ggml-metal-tuning test-backend-ops` 及其必需依赖，使用 `-j 8`。
- `tuner_help.txt` 已确认 `fa-vec`、`--dtype`、`--dk`、`--reps` 入口存在。
- 非阻塞提示：ccache / OpenMP 缺失、FP16_FORMAT_I3E 检测失败、使用 `-mcpu=native`、SVE 检测失败。按文档保留，未安装额外依赖或修改构建参数规避。
- 配置、编译均退出 0，日志为 `build_configure.log`、`build.log`。

## 4. Execution evidence

### FLASH_ATTN_EXT correctness test

```sh
cd llama.cpp-metal-lab
./build-metal-lab/bin/test-backend-ops test -o FLASH_ATTN_EXT -b MTL0
```

- 退出码：0，记录于 `test_flash_attn_exit.txt`。
- 实际摘要：`4951/4951 tests passed`。
- 附加配置切片：`fa_vec (Q,NE) slice: 240 cases run, 0 failed`。
- `Backend MTL0: OK`。该专项测试将 Metal 输出与 CPU reference 比较；这不属于完整项目测试套件。
- 日志中 349 条 `not supported` 是不支持形状的跳过，不计为通过。最后 `3/3 backends passed` 包含被过滤跳过的 BLAS 和 CPU 条目，不能解释为三个 backend 都执行了测试。
- 完整日志：`test_flash_attn.txt`。

### Restricted fa-vec tuner

```sh
cd llama.cpp-metal-lab
./build-metal-lab/bin/ggml-metal-tuning fa-vec --dtype f16 --dk 128 --reps 3
```

- 状态：completed；退出码 0，记录于 `fa_vec_exit.txt`。
- 设备：MTL0 / Apple M5；只运行 F16、dk=dv=128、reps=3。4 个 KV 长度（2048/3072/8192/32768）乘 9 个 query 行数，共 36 个形状点。
- `fa_vec_rows.txt` 输出 3 条候选配置；完整计时与筛选理由在 `fa_vec_sweep.log`。没有 DROP / untrusted cell 或冷却重试记录。
- 观察示例：KV=8192、query 行数=4 时，本次 Q1NE1 为 737.6 us，Q2NE4 为 521.6 us；query 行数=3 时最佳为 Q2NE2。数字是 `bench.cpp:59-78` 中包含图提交与同步的多次运行中位数除以重复 op 数，不能当作纯 kernel 时间。
- tuner 的规则还拒绝了某个桶内有退化的配置，说明单点最佳不等于整个桶可采用。3 条记录仅保留为实验结果，未写回源码调优表；本次运行不构成普遍性能承诺或 tuner 独立数值正确性证明。

## 5. Source-to-kernel chain

`src/llama-graph.cpp:build_attn_mha` 或 `tools/tuning/fa-vec.cpp:fa_build_graph` 构造输入，调用 `ggml_flash_attn_ext`；`ggml/src/ggml.c` 设置 `GGML_OP_FLASH_ATTN_EXT`；`ggml-metal-ops.cpp` 分派到 `ggml_metal_op_flash_attn_ext`，选择 vec / matrix 等路径、pipeline 与 workspace；通过 `ggml_metal_encoder_dispatch_threadgroups` 进入 `ggml-metal-device.m` 的 `dispatchThreadgroups`；最后执行 `kernels/fa.metal` 的 `kernel_flash_attn_ext[_vec]` 及条件性辅助 kernel。

精确源码位置和七段逐层说明见 `source_trace.md`。

## 6. Memory hierarchy mapping

- device/unified memory：CPU/GPU 共享统一物理内存；shader 的 `device` 地址空间用于 Q/K/V、mask、输出与 workspace。它不等于 threadgroup memory，也不应称作离散 GPU 的 HBM。
- threadgroup memory：显式 `threadgroup` scratch 保存 Q tile、score/softmax tile、mask 与输出累加等共享数据；矩阵路径还使用 K/V scratch。
- thread-local/register candidates：running max/sum、局部向量和矩阵等为寄存器候选；本次没有 GPU profiling，不能证明零 spill 或具体寄存器占用。
- synchronization：SIMD-group/threadgroup barrier 协调共享 scratch 的读写与复用；跨 kernel 由 backend 同步。
- 输出：矩阵路径归一化写到 `device dst`；vec 在 NWG>1 时先写部分 O/S/M，再由独立 reduce kernel 合并到最终输出。

## 7. Fusion conclusion

1. **避免什么流量？** 按 tile 完成 QK、online softmax、PV，避免将完整 `[query,kv]` 分数与概率矩阵写到 device memory 再读回。
2. **仍物化什么？** Q/K/V、mask、最终输出、必要辅助 workspace，以及多工作组 vec 路径的部分 O/S/M；不是所有中间量都消失。
3. **是否还有辅助 kernel？** 有。条件性 pad、mask block、量化 KV 转 F16、sparse index，以及 vec reduce。dense vec 当前设置 `nwg=32`，不能声称一次 op 必然一次 launch。
4. **RoPE 融合了吗？** 此处没有。`ggml-metal-ops.cpp` 对 ROPE 与 FA 分别分派，`rope.metal` 是独立旋转 kernel。本文证据只支持 Attention 内部融合。

## 8. Five takeaways

1. GGML op 是图节点，实际 GPU 执行要经过 backend、pipeline、encoder 和 dispatch。
2. 不需要下载模型，合成 Q/K/V 图就能验证与计时 Metal Flash Attention。
3. Apple unified memory、Metal device 地址空间和 threadgroup scratch 是不同概念。
4. FA 的关键是避免完整 score/probability 矩阵的 device 往返，不能用“只有一个 kernel”定义融合。
5. tuner 计时与数值正确性是两种证据；本次测试通过只覆盖实际运行的受支持形状，不证明所有输入或 RoPE 融合。

PROJECT STATUS: DONE
