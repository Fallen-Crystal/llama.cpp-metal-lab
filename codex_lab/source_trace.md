# Metal Flash Attention 源码阅读记录

审计版本：`930e2fa5995789efbf249a8bf61325bb626e417b`。源码根目录：本仓库根目录。先读 `AGENTS.md` 与 `codex_lab/tree_root.txt`、`tree_focus.txt`，再按文档指定顺序定点阅读。本文结论来自本地源码；源码分析不等同于某次运行已实际执行该分支，实际设备、形状、数值与耗时须由运行日志验证。

**验证范围说明**

`tools/tuning/README.md:3-7` 明确 tuner 不做性能 pass/fail，非零退出表示参数/环境错误；其数值正确性须另行跑 `test-backend-ops`。`tests/test-backend-ops.cpp:7649-7752` 是 FA 测试图，`11484-11565` 是 Metal 强制配置 slice：shape 仅 `(128,128)` 与 `(576,512)`，dtype 仅 F16/Q4_0，query 行数 1/3，KV 长度 512/4097，遍历合法 NE、Q 及 sinks 开关，并与 CPU backend 比较。它覆盖 padding、多 SIMD group、量化和 MLA views 等代表性分支，但不能把有限切片的通过扩展为所有 head/dtype/shape 的数值证明。源码读到有测试也不能写成测试已通过，实际结果应引用独立运行日志。

## 1. Graph entry：图构建入口

项目定位：`README.md:58,80` 明确 Apple silicon 的 ARM NEON / Accelerate / Metal 支持；`docs/build.md:140-145` 说明 macOS 默认启用 Metal，`GGML_METAL=OFF` 关闭构建支持，`--n-gpu-layers 0` 关闭 GPU 推理。构建成功、设备支持和一次执行确实选择 Metal 是三个不同证据。

图构建：[`src/llama-graph.cpp:2591`](../src/llama-graph.cpp#L2591) 的 `llm_graph_context::build_attn_mha` 先对 Q/K/V 做 view / permute。`2615-2639` 以 `cparams.flash_attn && kq_b == nullptr` 选 FA；必要时把 F32 K/V 转成 F16，再创建 `ggml_flash_attn_ext`，附加 sinks、稀疏 KV 上界与 F32 累加精度。

`tools/tuning/main.cpp:15,25` 注册 `fa-vec` tuner。`fa-vec.cpp:33-60` 构建无模型权重的合成 FA 图，Q 为 F32，mask 为 F16，K/V 使用 cache 风格 views；`dk=576,dv=512` 时 V 是 K 的子视图。

## 2. GGML op：算子定义与参数

GGML API 与节点：`ggml/include/ggml.h:2476-2495` 给出 Q `[dk,batch,head,ne3]`、K/V `[dk或dv,kv,head_kv,ne3]` 与输出 `[dv,head,batch,ne3]`。`ggml/src/ggml.c:5497-5539` 创建 F32 输出张量，保存 scale / bias / softcap，并设置 `result->op = GGML_OP_FLASH_ATTN_EXT`、`src[0..3]=q,k,v,mask`。这里只建图，不执行 GPU。

## 3. Metal dispatch：支持检查、workspace 与调优

Metal 节点分派：`ggml/src/ggml-metal/ggml-metal-ops.cpp:468-470` 将此 op 交给 `ggml_metal_op_flash_attn_ext`（`3144`）。普通路径的 vec 判断见 `2862-2870`：query batch `<20` 且 head size 可被 32 整除；稀疏路径另有判定，不能仅凭“小 batch”断定所有运行分支。

支持检查位于 `ggml/src/ggml-metal/ggml-metal-device.m:1718-1756`：限制可用 head sizes，要求 K/V dtype 相同，列出支持 dtype，BF16 还需设备支持，最后检查 `has_simdgroup_mm`（源码注释注明对 vec 过度限制）。不能把“图可创建”解释为“该 Metal backend 支持全部输入”。

配置选择：vec 主机代码 `3500-3514` 调用 `fa_vec_pick`，传入 device id、GPU family、KV dtype、dk/dv、KV 长度和 query 行数。`ggml-metal-tuning.cpp:4006-4051` 的顺序是强制 override、短 KV baseline、精确设备桶/域默认、family 代表设备、最终 baseline。`ggml-metal-ops.cpp:3646-3650` 在所需 threadgroup memory 超限时回退 Q=1。

`tools/tuning/fa-vec.cpp:226-242` 枚举合法 NE 和 Q={1,2,4}，并把 Q=1 与 baseline NE 作为基线。Q 是每组 query 数，不是模型 query 张量本身；NE 是模板中控制 lane/head 工作分配的参数（`fa.metal:1203-1207,1247-1252`），不能将变更简单等同于“更多线程”。

`ggml-metal.cpp:742-745` 将 backend 的 supports-op 查询交给 Metal device 检查；`230-236` 在 buffer allocation size 中为 FA 累加 pad、block、tmp、KV F16 和 index 空间。因此分派前分配的 device buffer 已为这些条件性 workspace 预留容量。

Workspace 绑定见 `ggml-metal-ops.cpp:3206-3230`：从输出 buffer 的 `ggml_nbytes(op)` 之后划分 pad、block、临时部分结果、KV F16 与 sparse index 区域。它们是 device workspace，不是第 5 节的 threadgroup scratch。

FA 辅助 kernel 的职责应分别说明：`fa.metal:11` 为 quantized KV 转 F16，`56` 为 padding，`128` 为 mask block 分类，`1087` 为稀疏索引建立，`2287` 为 vec 组间归约。主机 scratch 区分布见 `ggml-metal-ops.cpp:3210-3230`；KV 预转换启用条件见 `2874-2895`，包含 batch>=32 和指定量化 dtype；不是所有运行都会调用所有辅助 kernel。

## 4. Kernel launch：pipeline 与线程网格

Pipeline 特化：`ggml-metal-device.cpp:1713-1740` 组装 `kernel_flash_attn_ext_vec_<type>_dk<dk>_dv<dv>[_q<Q>_ne<NE>]` 名称；baseline 名称省略后缀。`1742-1756` 获取/建立 pipeline，使用 function constants 特化 mask、sinks、softcap、稀疏、stride、NSG/NWG 等。

绑定与提交：`ggml-metal-ops.cpp:3691-3729` 设置 pipeline、参数、Q/K/V/mask 等 buffer 和动态 threadgroup memory；主 kernel 网格为 `ceil(ne01/Q), ceil(ne02/nhptg), ne03*nwg`，每组线程为 `32,nsg,1`。`ggml-metal-device.m:867-875` 最终调用 Metal API `setThreadgroupMemoryLength:atIndex:` 和 `dispatchThreadgroups:threadsPerThreadgroup:`。

kernel 入口：`kernels/fa.metal:1209-1222` 的 `kernel_flash_attn_ext_vec` 接收 `device` buffer、`threadgroup(0)` scratch 和线程/组索引。普通矩阵路径则从 `ggml-metal-ops.cpp:3485-3500` 提交到 `fa.metal:867` 的 `kernel_flash_attn_ext`，内部实现从 `229` 开始。

完整链路：`build_attn_mha -> ggml_flash_attn_ext -> GGML_OP_FLASH_ATTN_EXT -> ggml_metal_op_flash_attn_ext -> pipeline + buffers -> ggml_metal_encoder_dispatch_threadgroups -> dispatchThreadgroups -> kernel_flash_attn_ext[_vec]`。

## 5. Memory path：device、threadgroup 与线程局部数据

| 层次 | 源码对象与位置 | 可以确认的结论 |
|---|---|---|
| Device buffer / 系统内存 | `fa.metal:1211-1218` 的 Q/K/V/mask/dst；`ggml-metal-device.m:1123` 读取 `hasUnifiedMemory`，`2127-2132` 可建立 Shared 或 Private MTLBuffer | Apple silicon 的统一物理内存不使所有访问都成为 threadgroup memory。`device` 是 shader 地址空间，Shared 是 Metal buffer 存储模式，两者也不是同一个术语。 |
| Threadgroup memory | `fa.metal:1219,1257-1261` 的 `shmem_f16`、`sq4`、`ss/ss4`、`sm`、`so4` | Q tile、attention/softmax scratch、mask tile、局部输出放在组共享区。`1309` 与 `1736,1767` 的 barrier 协调组内访问；主机显式计算并预留此区。 |
| Thread-local 候选 | `fa.metal:1312-1316` 的 S/M，`1441` 的 `mqk[Q][C/NE]`，`1586` 的 `lo[Q][DV4/NL]` | 这些局部标量/数组是寄存器候选；源码本身不能证明全部驻留硬件寄存器，也不能给出占用、spill 或 occupancy，需编译器/GPU profiling 证据。 |

矩阵路径同样有 `sq/so/ss` 和 K/V 解量化 scratch（`fa.metal:278-292`）。名称中的 `shmem_f16` 是分配/寻址基底，后续存在不同类型的 reinterpret；不能据此声称所有中间量均为 FP16。

最终输出写回也属于本节的数据路径：矩阵 kernel 从共享累加区归一化写入 `device dst`（`fa.metal:811-830`）；vec 写回最终输出或部分 O/S/M（`1770-1796`），多工作组的最终 reduce 见下一节。

## 6. Fusion path：online softmax 与中间结果

未启用 FA 时，`src/llama-graph.cpp:2660-2706` 显式构建 `K*Q`、mask/bias/softcap、`ggml_soft_max_ext` 和 `V*概率` 节点。FA 的主要收益是按 KV tile 做点积、online softmax 和 V 加权累加，不在 device memory 中物化完整的每 head `[query,kv]` 分数/概率矩阵。

vec 路径的具体过程：

- `fa.metal:1276-1294` 将 Q 行载入 threadgroup；`1346-1349` 分块循环 KV；`1431-1503` 计算 QK 点积，`1532-1550` 应用 scale、mask、softcap/ALiBi。
- `1557-1576` 更新 running maximum `M`、running sum `S`：`M_new=max(M_old, tile_max)`，`a=exp(M_old-M_new)`，`p=exp(score-M_new)`，`S_new=S_old*a+sum(p)`；把旧 O 乘 `a`，临时概率 tile 写回 `ss`。
- `1584-1650` 用该 tile 概率乘 V 并加到 O。代码注释写作 `O + (Q*K^T)*V`，但此时 `ss` 已是 exponentiated probabilities，不能把它解释成未 softmax 的原始 QK。
- `1725-1767` 将 S/M 写到共享 scratch，合并多个 SIMD groups；`1770-1796` 输出结果或部分结果与 S/M。普通矩阵路径也有 online softmax（`593-629`），并利用 simdgroup matrix 操作。

省去的是完整 QK 与概率中间矩阵的 device 读写；不是“没有中间量”，也不是“不访问设备内存”。输入、mask、KV cache、输出仍是 device buffer，tile scratch 仍存在。

尤其 vec 的 `NWG>1` 时，`ggml-metal-ops.cpp:3724-3749` 先把每组 O/S/M 写入 `bid_tmp`，同步后另发 reduce kernel。`fa.metal:2287-2321` 对组间 maximum 做稳定缩放、合并分母与分子后归一化。标准 dense vec 主机路径 `3638-3643` 设 `nwg=32`。因此“一次 GGML FA op”不等于“一次 Metal kernel launch”，也不意味着零 device 临时存储。


普通矩阵 FA 路径的 QK 运算可直接追到 `fa.metal:499,518-519` 的 `simdgroup_multiply_accumulate`；概率 tile 与 V 的乘加见 `676-677,699-702`。最终输出见 `811-830`：从 threadgroup `so4` 读取结果，以 `1/S` 归一化后写到 `device float4 * dst4`。这些行明确区分 SIMD-group 矩阵运算、共享暂存和 device 写回。

## 7. RoPE contrast：独立旋转算子对照

RoPE 是独立算子：`ggml-metal-ops.cpp:399-404` 有 ROPE/ROPE_BACK 分支，`4156` 是其编码函数，`4241` 选专用 pipeline；`ggml-metal-device.cpp:1965-1998` 依模式选择 norm/neox/multi/vision。`rope.metal:48-56` 的入口读取 device tensors，`72-90` 计算位置相关角度并旋转一对元素，`89-90` 写回 `x0*cos-x1*sin`、`x0*sin+x1*cos`。此 kernel 没有 FA 的 KV 分块、online softmax 或 threadgroup scratch 接口。不能把 RoPE 描述为本次 FA kernel 内融合阶段；本报告中的 RoPE 对照以该独立算子和 kernel 为边界。
