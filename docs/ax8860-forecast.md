# AX8860 部署预估（对照 AX8850 实测）

> 面向**后续 AX8860 上市后的转换与部署**：本仓库所有实测数字都来自 **AX8850**（AX650 同包兼容，
> 板端 AX Engine 2.12.0s、3 NPU core、6 GB CMM、8 核 A55、1.9 GB 系统内存）。
> 本文给出本模型的**逐模块计算量/内存带宽分析**与 **AX8860 速度预估**，AX8860 到手后按 §6 的表格
> 用真实数据替换/揭晓即可。

## 1. 基线：AX8850 实测（本仓库 warm 常驻服务，INT8 / bf16 混合）

| 阶段 | 实测 | 折算每音频帧（12.5 fps） |
|---|---|---|
| 文本编码器（两段 U16/S8，34 token） | 193–218 ms | — |
| talker prefill（L=34） | ~370–400 ms | — |
| talker decode（28 层，kv 512，INT8） | 133–146 ms/帧 | **~134 ms** |
| depth decoder（12 层 × 15 步，INT8 + U16 lm_head） | 500–640 ms/帧 | **~503 ms** |
| codec（8 层 transformer + 分块 vocoder） | ~9 ms/帧（整句 0.5–1.2 s） | ~9 ms |
| **端到端** | 5 s 音频 ≈ **37–42 s（RTF 8.2–8.3×）** | 帧合计 ≈ 646 ms |

## 2. 计算量分析（为什么是算力瓶颈）

按每 token/帧的 MACs 估算（INT8，含注意力和 MLP）：

| 模块 | 每帧 MACs | 每帧 FLOPs | AX8850 实测耗时 | **有效算力** |
|---|---|---|---|---|
| talker 28L（hidden 2048, inter 6144, kv 8） | ≈ 1.41 GMAC | ≈ 2.82 GFLOP | 134 ms | **≈ 21 TFLOPS** |
| depth 12L × 15 步（hidden 1024, inter 8192, kv 2） | ≈ 5.0 GMAC | ≈ 10.0 GFLOP | 503 ms | **≈ 20 TFLOPS** |
| codec 8L（小模型） | ≈ 0.1 GMAC | ≈ 0.2 GFLOP | 9 ms | ≈ 22 TFLOPS |
| 文本编码器（26L hidden 1152，34 token） | ≈ 45 GMAC | ≈ 90 GFLOP | ~200 ms | ≈ 0.45 TFLOPS（小图/低效） |

**结论**：talker 与 depth 都跑在 **≈20 TFLOPS 的有效 INT8 算力**上，接近 AX8850 标称 24 TOPS 的
~83%，是**计算 bound**（不是内存 bound）：每帧读权重 talker 1.63 GB / depth 0.43 GB，
对应带宽 12 GB/s / 0.9 GB/s，远低于 AX8850 的 LPDDR 带宽。→ **AX8860 的速度基本正比于 NPU 算力提升。**

## 3. AX8860 公开规格（2026-10 发布）

| 项 | 参数 | 来源 |
|---|---|---|
| NPU | **256 TOPS 混合精度**（INT4/8/16、FP4/8/16、BF16），原生 FP4、无稀疏 | 官方/IT之家 |
| CPU | 16 核 Cortex-A55 1.65 GHz | 官方 |
| 内存 | 最高 **128 GB LPDDR5/5X**，256-bit | 官方/IT之家 |
| 带宽 | **273 GB/s**（发布会口径）/ 官方产品页写 136 GB/s（不同配置口径） | IT之家 / 官方 |
| 互联 | PCIe 5.0、AXLink C2C，双芯 512 TOPS / 四芯 1024 TOPS（约 80% 效率） | IT之家 |
| 工艺/功耗 | 6 nm，约 40 W（模块/单槽卡） | IT之家 |
| 软件 | Pulsar2/AX-LLM 新工具链，支持 vLLM、SGLang、PyTorch、ONNX/HF、OpenAI API | IT之家 |

- IT之家发布稿：<https://m.ithome.com/html/1012551.htm>
- 官方产品页（中文）：<https://www.axera-tech.com/index.php/zh-hans/product/3294.html>
- 官方产品页（英文）：<https://www.axera-tech.com/en/node/3294>
- 补充报道：<https://ee.ofweek.com/2026-10/ART-8110-2801-30707753.html>、<https://www.cnx-software.com/news/axera/>

**算力比**：256 TOPS / 24 TOPS（AX8850 标称）≈ **10.7×**。按新芯片工具链成熟度打折：

| 场景 | 假设 AX8860 达到峰值的 | 相对 AX8850 加速 | talker/帧 | depth/帧 | 每帧合计 | 5 s 音频总耗时 | 预计 RTF |
|---|---|---|---|---|---|---|---|
| 保守 | 50% | **≈ 6.4×** | ~21 ms | ~79 ms | ~102 ms | ≈ 6.9 s | **≈ 1.4×** |
| 期望 | 65% | **≈ 8.3×** | ~16 ms | ~61 ms | ~79 ms | ≈ 5.4 s | **≈ 1.1×** |
| 乐观 | 80% | **≈ 10.2×** | ~13 ms | ~49 ms | ~64 ms | ≈ 4.4 s | **≈ 0.9×**（快于实时） |

（总耗时 = 帧数×每帧 + prefill ~0.2 s + textenc ~0.05–0.1 s + codec ~0.1–0.2 s；5 s 音频 = 62 帧。）

**结论：AX8860 单芯预计 RTF ≈ 0.9–1.5×（5 秒语音 4.5–7 秒出结果），相对 AX8850 提速 6–10×，
进入“准实时”区间；双芯 AXLink 级联（2×512 TOPS）若做流水线并行（如 textenc/talker 与 depth 分芯），
可进一步到 RTF < 0.5×。**

> ⚠️ **口径风险**：官方“256 TOPS”是**混合精度峰值（原生 FP4，无稀疏）**。若 INT8 峰值实际减半
> （128 TOPS，常见做法），则 INT8 模型的加速比约 **3.5–5×**，对应 RTF ≈ **1.7–2.4×**；
> 建议 AX8860 上尝试 **FP4 权重**重编译（体积减半、带宽减半、算力跑满 FP4 通道），以接近上表上限。
> 本文按“256 TOPS 可用于 INT8”估计，AX8860 到手后用 §6 实测揭晓。

## 4. 内存/CMM 预期

AX8860 是 128 GB 统一 LPDDR5X，不再有 AX8850 的 6 GB CMM 约束：

- 全部模型常驻约 **4.4 GB**（textenc 0.77 + talker 1.63 + depth 0.40 + codec 0.16 + embedding 表 0.27 + lm_head 0.03 + U16 版按需 1.3）；
- 每会话 KV（talker kv 512 ≈ 41 s 音频；depth kv 32）仅几十 MB → **可支持大量并发会话**（配合 AX-LLM 的 batching/paged attention，吞吐可再提升数倍）；
- 文本编码器、quality 模式（FP32 depth 1.3 GB）都可以常驻，无需像 AX8850 那样“用完即释放”。

## 5. 转换/部署指引（AX8860 到手后）

1. **工具链**：升级 Magnetar/Pulsar2 到支持 AX8860 的版本；确认 `pulsar2 build/llm_build` 的
   `--chip` 有 `AX8860`（或新命名）选项，axmodel 格式可能升级，需**重新编译**，不能直接复用 AX8850 产物。
2. **权重无需重打包**：本仓库 `scripts/convert/repack/repack_breeze.py` 产出的 HF 格式
   （Qwen3 talker / Llama depth + embeds + lm_head ONNX）与芯片无关，直接复用。
3. 编译参数沿用本仓库：talker `--model_type qwen3 --kv_cache_len 512 --prefill_len 128 --ret_postnorm`；
   depth `--model_type llama --kv_cache_len 32 --prefill_len 16 --ret_postnorm`；
   lm_head U16 版本（`build_lm_head_hp.sh`）继续用。
4. **量化**：AX8860 原生 FP4，可尝试 talker/depth 权重 FP4（体积减半、带宽减半，算力不变），
   需用本仓库的 golden（`docs/ref/GEN_LOGIC.md` + `_work/ref/golden_*` 流程）复核 cos/ASR；
   保守起见先 INT8（已验证）。
5. **运行时**：`runtime/run_breeze_tts.py` 的模型外逻辑（采样、EOS、depth 15 步、codec 分块）可复用；
   若 AX8860 的 AX-LLM 支持图捕获/批量执行，建议把 depth 的 15 步/180 层 launch 合并成单图，
   可去掉当前每帧 180 次 layer launch 的开销（AX8850 上这部分的 CPU 侧开销已到 ~50 ms/帧量级）。
6. **验证口径**（与本仓库一致）：文本编码器 vs ONNX cos ≥0.99；talker frame0 top5；整体 ASR
   （paraformer-zh / SenseVoice）；延迟分解与 RTF；CMM/内存峰值。

## 6. 真实数据回填表（AX8860 上市后揭晓）

| 项 | AX8850 实测 | AX8860 实测（待填） |
|---|---|---|
| 芯片/工具链版本 | AX8850 / Pulsar2 8.0 | |
| talker decode | 134 ms/帧 | |
| depth decoder | 503 ms/帧 | |
| codec | 9 ms/帧 | |
| 5 s 音频端到端 | 37–42 s（RTF 8.2×） | |
| 常驻内存 | CMM 4.37 GB | |
| ASR 通过率 | 4/4 | |
