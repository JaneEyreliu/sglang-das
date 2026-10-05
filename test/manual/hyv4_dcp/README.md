# HY4 W4A8: DCP2 / DP16 / EP32

实现分支：`feat/sglang-hy4-dcp-adapt`。以现有 DCP 仓库为基础，移植
`sglang-hy4-w4a8` 的 HY4 主模型与单层 MTP；权重格式兼容部分参考已有 `sglang-das`
实现。gate 使用 attention TP2，不建立独立 gate 通信组。

| 改动 | 作用 / 隔离范围 |
| --- | --- |
| `configs/hy_v4.py`、config 注册和 MLA/DSA 识别 | 识别 HY4 target/NextN，主模型按 `indexer_types` 共享 TopK；原 GLM 等模型的判定保持不变 |
| `models/hunyuan_v4.py` | iHC、indexer RoPE 排列、FP32 router/LM-head；dense/shared expert 不做 routed expert 的 SwiGLU clipping |
| `models/hunyuan_v4_nextn.py` | 单层 pre-norm MTP（无 iHC），复用 gate TP2、sink、EP32；融合 D 维 target hidden 与 embedding，共享 target embedding/LM-head |
| `configs/hy_v4_mtp.py` | HY4 MTP 名称、完整性、shape/dtype 检查；运行时与离线 header 检查器共享，防止漏载后继续推理 |
| `resolve_spec_hidden_size` | HY4 target/draft 使用 D 维 hidden，不套用 DSV4 的 D×hc_mult 缓冲；其他架构计算规则保持不变 |
| HY4 DCP draft index-K | KV page128 与 index page64 解耦；target/draft 共同遵循 INT8 开关，保持 P/D ABI 一致。独立 INT8 索引页仅为 HY4 HCU draft 开启；预算计入 INT8 存储、反量化 workspace 和 DCP 复制的 draft KV |
| `linear_gate`（兼容 checkpoint 的 `g_proj` 名称） | attention TP2 按输出维切分，两个 rank 各保存一半 heads；在 DCP 输出合并和 value 投影后施加 sigmoid gate，再执行 `o_proj` |
| `layers/hy4_dcp.py` 和 FlashMLA KV 接入 | 从无 sink 的 output/LSE 补入 sink；每个 DCP rank 分配 `exp(sink)/D` 的分母，再走现有 LSE 合并，只累计一个 sink |
| iHC 的 DP token 布局 | attention 前 gather 二维输入，attention 后 reduce-scatter；三维 residual 保持本地，输出使用独立存储 |
| W4A8 显式格式标记 | 仅 `checkpoint_format=hy4_w4a8_v1` 转换 nibble 顺序和 scale；未标记的其他模型沿用原行为 |
| shared expert INT4 加载 | nibble 无损展开为 INT8，保留真实 scale，使用独立 MLP 避免错误 clipping |
| 可选 TileLang iHC | 默认关闭；不可用时回退 eager；只由 HY4 模型调用 |

32 张卡的布局是 `TP32 / DP16 / DCP2 / EP32`，attention TP 为 2。
每个 DP 组有相同的请求，两张卡分摊 KV token；attention 权重按 head 切分。
MoE 使用全 32 卡的 DeepEP。`g_proj` 不建立额外通信组。
小型 sink logits（每 head 一个 FP32）复制保存，使 DCP gathered heads 可直接取用，
不会为每层增加 sink all-gather。

启动需满足：

```text
--tp-size 32 --dp-size 16 --dcp-size 2 --ep-size 32
--enable-dp-attention --moe-dense-tp-size 1
--pp-size 1 --attn-cp-size 1 --dcp-comm-backend ag_rs
--moe-a2a-backend deepep --deepep-mode low_latency
--attention-backend dsa
--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv
--quantization slimquant_w4a8_marlin --kv-cache-dtype fp8_e4m3 --page-size 64
--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full
```

`SGLANG_DSA_FUSE_TOPK=false`。HY4 已接入普通解码与 EAGLE/NEXTN 单链 MTP
（topk=1）。不启用 EAGLE3、SBO/TBO、prefill CP 或分段图；
`index_share_for_mtp_iteration` 支持 `true` 和 `false`。`true` 复用 draft-extend
最后接受位置的索引；缺少种子时由首个 draft forward 计算，后续步骤复用。
`false` 在每个 draft forward 重新计算索引。两种模式均保持 topk=1；
开关由现有 EagleDraftWorker 管理，GLM 路径不变。
当前仓库外 `d1.sh`～`d4.sh` 已配置以下 MTP 参数，NEXTN 会规范化为 EAGLE：

```text
--speculative-algorithm NEXTN
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
--json-model-override-args '{"index_share_for_mtp_iteration": false}'
```

两次 draft step 复用一个 MTP 层，不需要两个 checkpoint MTP 层。
确保运行时 `PYTHONPATH`
指向本分支的 `python` 子目录，而不是另一份旧仓库。

`d1.sh`～`d4.sh` 直接从 `${SGLANG_ROOT}/python` 导入，不再复制源码到系统
`/tmp`。脚本禁用 `.pyc` 写入和 core dump；工作目录、Python 临时文件、Triton/
PyTorch 编译缓存默认放在 `${SGLANG_ROOT%/*}/runtime/node0`～`node3`。
可用绝对路径 `SGLANG_RUNTIME_DIR` 指向有空闲空间的挂载盘。启动前会实际创建、
写入并同步一个小临时文件检查可写性；历史 `/tmp/sglang-*` 文件不会自动删除。

HY4 v1 checkpoint 必须有明确的格式标记：`config.json` 的 `quantization_config`
包含 `"checkpoint_format": "hy4_w4a8_v1"`，或者无 HF 量化配置的导出目录中提供
`hy4-assets.json`，其中 `"format": "hy4_w4a8_v1"`。这表示 even-K 在低 nibble、signed INT4、
真实 per-channel scale。代码不会根据目录名称猜测格式。旧量化配置默认不转换。
启动前可运行下面的只读检查，检查配置和 attention gate header 形状：

```bash
python3 test/manual/hyv4_dcp/check_checkpoint.py /models/HY4_W4A8
python3 test/manual/hyv4_dcp/check_checkpoint.py /models/HY4_W4A8 --mtp
```

`--mtp` 要求单层 `model.mtp_layers.0`（兼容 `model.mtp.layers.0`），检查全部
attention/indexer、gate/sink、输入融合、norm、router、routed/shared expert 权重。
当前 HY4 v1 W4A8 路径要求 MTP experts 为 INT4 packed + 真实 channel scale，
attention/融合层保持浮点。如果导出目录删掉了 MTP 或 MTP 使用另一量化格式，
检查会明确失败；不能用随机初始化补齐。运行时 loader 同样检查这些条件。

CPU 验证（只需 PyTorch）：

```bash
python3 test/registered/dcp/test_hyv4_dcp_cpu.py
python3 test/registered/dcp/test_hyv4_mtp_cpu.py
python3 test/registered/dcp/test_hyv4_mtp_index_share_cpu.py
python3 test/registered/dcp/test_hyv4_mtp_cache_cpu.py
python3 test/registered/dcp/test_hyv4_retraction_cpu.py
python3 test/registered/dcp/test_hyv4_pd_bootstrap_cpu.py
```

测试隔离了 GPU 模块导入并执行实际生产函数；FlashMLA、通信和基础设施采用 CPU
替身。覆盖 gate 权重切分、CUDA/ROCm 门控执行顺序、sink 对照完整 softmax、空 KV、
空 batch、padding、INT4 全字节解码、shared expert 解包、非 HY4 router 和量化路径、
HY4 启动约束及原有 DSA launch contract 用例。它不等价于多机端到端验证。

本地环境没有 `/models/HY4_W4A8`、HCU/GPU 和完整 SGLang 运行依赖。
部署环境还必须验证：

1. 四节点先禁用 decode graph，检查权重加载和 all-rank 启动。
2. 同 checkpoint 在 DP32/EP32 与 DCP2/DP16/EP32 上比较固定 prompts 的 logits、
   greedy tokens 和任务准确率；覆盖短输入及超过 2048 token 的稀疏注意力输入。
3. 打开 full decode graph，比较 eager/graph 的输出；覆盖 batch 1、2、4、超过
   graph 捕获上限的 eager batch，以及部分 DP 组空闲。
4. 通过真实 P/D 服务验证 Mooncake KV + index-K 传输，包括跨 64-token 物理页边界；
   P/D 的 index-K cache 格式和 page ABI 必须一致。
5. 实测 GLM 5.2/5.3 原配置的启动与输出，记录两种布局的显存、吞吐和延迟。
6. MTP 先运行 eager、单步 draft，再运行 steps=2/topk=1/draft_tokens=3，比较
   MTP 开关前后 greedy tokens/logits；覆盖接受、拒绝、EOS、空闲 DP 和分页边界。
7. 开启 MTP full graph，比较 eager/graph；验证 P/D 的 D 维 hidden、KV 和 index-K
   传输；128K 输入及并发必须重新压测，MTP 会额外占用权重、KV 和图缓冲显存。

未实测前不能据 CPU 单测宣称多机正确性或性能验收完成。

本地验证记录（2026-09-23）：CPU PyTorch 隔离环境下，20 个 DCP 顶层测试和
11 个 MTP 专项测试和 7 个 MTP cache 回归测试通过；DCP 测试中包含原有 DSA
launch contract 的 8 个回归用例。
MTP 测试涵盖 GLM draft 映射保留、其他 HC 模型 hidden 宽度保留、量化配置复制隔离、
权重流名称处理及缺失检查、TP2 residual 布局、D 维 hidden 融合和合成 header 检查。
FlashMLA、RCCL/DeepEP、完整模型导入和 P/D 服务不在这些 CPU 测试覆盖范围内。
新增/修改 Python 文件通过语法检查，`git diff --check` 和四个启动脚本的
`bash -n` 检查通过。
用户提供的 115005 日志已证实普通 decode 启动；134815 日志中 node0 的 8 个 rank
均完成 HY4 target 和 NextN 加载（MTP 加载阶段记录 1.35 GB），随后在 draft 缓存
分配因虚拟 KV 页与 INT8 index-K 页布局不匹配而失败。
145038 日志证明之前的 BF16 draft 方案通过 target verify、draft decode、draft extend
图捕获，但在 P/D 注册时报 target INT8 / draft BF16 ABI 不一致，因此已撤掉强制 BF16。
当前修复让 packed index-K 的分配、量化写入、LightOp 读取、token 搬移和卸载恢复
统一使用 index_page_size=64；draft KV 仍为 page128。P/D 校验保持原样。
7 个 cache 测试覆盖实际 CPU packed buffer/量化参考实现的 63/64、127/128、末页及
padding 边界、搬移和两半页卸载恢复，复现旧 ABI 错误并验证新 P/D 注册和页字节数，
同时验证非 HY4 默认限制、BF16 回退和显存预算。
修复后的完整四机 MTP 启动、GPU INT8 图捕获与推理精度验收仍待部署环境执行。
四节点均需使用本分支最新代码；P 端 index-K 模式/page ABI 也必须与 D 端一致。


2026-09-23 审查后的修复与优化：

- **HY4 DCP 索引末页容量**：index-K、INT8 反量化 workspace、page claims 和
  BF16 回退统一按 `ceil((virtual_capacity + virtual_page_size) / index_page_size)`
  分配。DCP2 为 page128 的分配器保留完整虚拟零页；例如可用容量 512 时，合法
  槽位为 128～639，需要 10 个 index page64，旧的 9 页会漏掉最后 64 个槽位。
  target/draft 同时修复，显存预算计入该 padding；未传 HY4 参数的模型保持原容量规则。
- **HY4 DCP decode 普通内存回退**：专用 allocator 将 target 虚拟地址转换为当前
  DCP rank 的物理地址，同时保存/恢复全局 index-K 和完整复制的 MTP KV/index-K。
  恢复允许页号变化；调度器在内存回退前提交 pending overlap 结果并保存最新 MTP
  hidden/top-k seed，清除旧 DSA slot seed。原有其他模型继续使用原 allocator。
  当前 DCP 回退使用 `cpu_tensor`；基础分支已禁止 DCP 的 `host_pool` 回退。
- **sink 临时显存优化**：在 `where` 产生的独立 output 上做原位缩放，省去完整
  FP32 output 临时张量；BF16/FP16/FP32 CPU 对照原公式逐值一致。没有实测 GPU 提速比例。
- **shared indexer 缓存优化（默认关闭）**：新增
  `SGLANG_HY4_COMPACT_PD_INDEX_K=1`，仅 HY4 target 在 P/D 模式下跳过 shared 层
  的 index-K 分配；draft 保持完整。预算和 P/D 注册均使用实际 full-indexer 层，
  层号保留全局 ID，MTP 偏移仍为 target 总层数。HiSparse/HiCache 不开启此优化。
  **P、D 两端都升级到包含此实现的代码并设置该变量后才能启用**，不能单独打开 D 端；
  dense P → compact D 缺失层号时会被原传输校验拒绝。四个启动脚本默认不打开此项，
  兼容当前已部署的 P 端。固定容量下，INT8 每卡预计节省
  `shared层数 × (virtual_capacity + 128) × 132` 字节；具体层数以模型配置为准。
- 四个启动脚本统一清除继承的 `SGLANG_SIMULATE_ACC_*` 三项环境变量，避免节点间
  模拟接受率设置不一致。TileLang iHC 保持默认关闭。

新增 CPU 回归覆盖两个 DCP rank、INT8/BF16、MTP 开关、0/1/63/64/65/127/128/129/
255/256 token 的非连续页迁移、最后可分配页、相邻页不被改写、重复恢复、最新 MTP
seed、overlap 提交顺序、HY4/GLM allocator 分流和 P/D compact 层号注册。
本轮共 48 项 CPU 测试通过。部署验收还需在真实 HCU 四节点上通过至少每 DP 两条
请求或降低 KV 容量触发回退，核验恢复后的 greedy 输出与未回退基线一致，并分别
验证 MTP 开关与 eager/full graph；运行 `fake prefill` 只能检查容量和执行路径，
不能验收权重、KV 内容和输出准确性。本次未重启正在运行的服务。


170427 启动日志对应的默认后端修复：旧 HY4 仓库通过模型专属 override 自动设置
prefill/decode backend；本次 DCP 移植之前缺少该注册，省略 prefill 参数时值保持
`None`，被 HY4 启动校验拒绝。现为 HY4 target/NextN 补充专属默认解析，未填写的
`dsa_prefill_backend` / `dsa_decode_backend` 均使用本分支已接入 sink 的 `flashmla_kv`。
因此脚本可省略 `--dsa-prefill-backend flashmla_kv`。显式配置不会被覆盖；本分支
尚未移植旧仓库 `flashmla_sparse` 的 sink 路径，不能直接删除启动校验来支持它。
旧仓库 HCU 的默认 prefill 为 `flashmla_sparse`，这与当前分支的默认实现不同。
本轮新增默认值注册、解析及非 HY4 隔离回归，CPU 测试总数为 49。


P/D bootstrap 协议兼容修复：实际 P 端 `/route` 返回 `kv_cache_layout: null`，
旧 `sglang-das` 的 `PrefillServerInfo` 接受该字段，而 DCP 分支之前缺少字段，
`try_ensure_parallel_info` 会捕获 `unexpected keyword argument 'kv_cache_layout'`
并返回失败。现允许空值/省略字段，非空的未支持布局仍明确拒绝；page-size、KV dtype
和 CP/DCP 拓扑校验保持不变。使用实际 P 端返回的 PP2/CP8/layer-split 元数据复现并
测试握手解析流程，52 项 HY4 CPU 测试通过。此修复解决确定的 bootstrap 兼容缺口，
不能代替实际 router 注册、RDMA 传输和生成验收；未重启正在运行的服务。
