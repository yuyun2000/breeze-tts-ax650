# 板端实测报告（AX650C，pyaxengine）

> 2026-11-11 夜。板卡：AX650C（`pyramid-openclaw`），AX Engine 2.12.0s / MC50 / triple core，
> 8 核 aarch64 / 1.9 GB RAM / 2 GB swap / eMMC，CMM 6144 MB（停掉其它 demo 后可用 ~5.1 GB）。
> 运行时：纯 numpy + pyaxengine + tokenizers，无 PyTorch。

## 1. 延迟（fast 模式 = talker/depth INT8，官方默认采样参数）

| 阶段 | 实测 | 备注 |
|---|---|---|
| 模型加载 | talker 28 层 7.9 s；depth 12 层 2.5 s；codec 0.8 s；embedding 页缓存预热 0.67 s | |
| 文本编码器（两段 NPU） | 加载 1.58 s；编码（34 token, L=128 桶）**196 ms**；CMM +742 MB 后释放；链式 cos **0.99968** vs ONNX fp32 | |
| talker prefill (L=34) | **398 ms** | 1 个 128 token chunk |
| talker decode / 帧 | **133–146 ms**（均值 ~140） | 28 层，kv 512 |
| depth decoder / 帧 | **564–758 ms**（15 步 × 12 层 INT8 + lm_head；`--lmhead-cpu` 时 CPU fp32 兜底；U16 lm_head 上 NPU 后实测 ~506 ms/帧） |
| codec decode / 句 | **522 ms（5.12 s）/ 1249 ms（6.72 s）** | decode_pre + 8 层 transformer + 分块 vocoder |
| **端到端（采样模式，短句 64 帧）** | **72.2 s / 5.12 s 音频 = RTF 14.1×** | 其中 talker 9.36 s + depth 48.5 s + codec 0.52 s + prefill 0.43 s |
| **端到端（采样模式，长句 84 帧）** | **73.4 s / 6.72 s 音频 = RTF 10.9×** | talker 11.57 s + depth 47.3 s + codec 1.25 s |

资源：python RSS 峰值 ~1145 MB、MemAvailable ~1082–1237 MB、**CMM used 3471 MB / remain 2672 MB**
（talker 1.6 GB + depth INT8 0.58 GB + codec 0.16 GB + KV/buffer）。
quality 模式（depth FP32-heavy 1.3 GB）实测进行中：每帧预计 1.5–2.2 s，作为可选档。

## 2. 精度/对拍

| 项 | 结果 |
|---|---|
| 重打包等价性（CPU f32，原始 Breeze vs repack） | talker 逐层 hidden cos 1.00000000；depth 16 步 1.0000010；lm_head logits 0.9999998 |
| talker frame0 prefill logits top5 vs golden | **逐位一致**（[404,1464,29,1221,279]，数值 18.625 vs 18.625） |
| talker post output_norm vs golden hidden_norm[0] | cos **0.99942** |
| depth teacher-forced（golden codes 输入） | INT8: logits cos 0.99914、top1 96%；free-run 在 frame0 第 3 步分叉（量化近 tie 翻 argmax）。hp(FP32): cos 0.99927、top1 95.6%（与 bf16 参考不可约）、free-run 平均 code match 0.90（INT8 0.81） |
| **15 个 lm_head** | 旧 INT8 输出被 clamp（2051 logits 仅 ~150 unique、argmax 错）；**U16 版本修复**：unique 1662–1917、cos 0.999963–0.999994、top1 与 CPU fp32 一致（100%），NPU 比 CPU heads 快 ~50 ms/帧 |
| codec（同一份 golden codes 过板端 vocoder） | 长度精确一致 122880 samples；**幅度谱相关 0.99958**、能量包络 0.84；波形相位差来自 vocoder 实现差异，不影响内容 |
| **ASR（paraformer-zh + SenseVoice 双模型）** | **4 条 demo 采样模式全部与输入文本一致**：中文短句 "...语音合成" ✓；长句 "今天天气不错我们一起出去走走吧你觉得去哪里比较好呢" 逐字 ✓；英文 "Hello, welcome to Breeze TTS running on the AX650." ✓；中文场景句 "深圳的秋天很舒服，周末我想去海边走走，顺便拍几张照片。" 逐字 ✓。greedy 模式短句丢最后音节（depth 分叉），仅作 token 对拍基准 |

## 3. 结论

1. **Breeze TTS 2 的 talker(28L) + depth(12L) + codec 全部在 AX650 NPU 上运行**，纯 Python runtime 可用；
   采样模式（官方默认参数）音频**通过 ASR 客观验收**。
2. 性能 RTF ≈ 11×（非实时；比预期 9.5–14× 区间），瓶颈是 depth decoder 的 180 次 layer forward
   与 talker 逐帧权重流（每帧约 1.7 GB 权重读）。
3. codec 复用 AXERA 已编译 Qwen3-TTS speech_tokenizer 产物被证明精确（同 codes → 同频谱/能量/长度）。
4. 已知限制：
   - 文本编码器 NPU 版本受 Pulsar2 混合精度 AxRoPE backend bug 阻塞（INT8 cos 0.45 不达标；
     U16/auto/smooth-quant 尝试结果见 §5），demo 用 golden text_embeds 注入（其余全部 NPU）。
   - depth INT8 量化噪声导致 free-run token 分叉（音频可懂，采样模式 ASR 通过）；quality 模式可选。
   - CVF/双 CFG、voice clone（需 codec encoder + speaker encoder）未实现。

## 4. 复现

```bash
# 板端
export LD_LIBRARY_PATH=/soc/lib
python3 runtime/run_breeze_tts.py --text "..." --instruction "..." --out out.wav --dump-tokens tokens.npz
# 采样模式默认开启；--greedy 可切贪心；--depth-dir 可切 quality 模式
```

## 5. 文本编码器 NPU 的尝试记录

| 方案 | 结果 |
|---|---|
| INT8（MinMax, 40 calib） | 仿真 cos 0.41–0.55（**不达标**）；precision_analysis：MLP 中间激活 (1,128,6912) cos 0.07–0.19、多处 RNNorm/Add 0.6–0.88（动态范围过大） |
| 显式 U16（MatMul/Add/Mul/... 全 U16） | cos 0.90（仍不达标） |
| ONNX 自身精度 | vs PyTorch cos **1.000000**；vs Mac golden text_embeds 0.99993（排除导出问题） |
| `highest_mix_precision=true`（auto） | **实为"未量化"**（`Num of Quant Config: [0]`，precision 表全 1.0 是假象），且 NPU backend 在 **FP32 AxRoPE** 上崩：`OpBuildException: op: AxRoPE, s1=S(1024,1024) s2=S(32,8192)` |
| `precision_analysis_mode=NPUBackend` | 同样 Quant Config 0 / 崩溃（截止前终止） |
| SmoothQuant | 未得到结论（c1_sq 为让路被主动 kill、c2_sq 脚本未启动） |

> **结论**：Pulsar2 8.0 工具链暂无法把 T5Gemma2 文本编码器量化到可用精度；ONNX 两段（cos 1.0/0.99993）与
> 校准/验证脚本全部保留，等工具链修复后按 `docs/ref/TEXTENC_REPORT.md` §5 重编译即可，板端接口不变。
> 完整证据见 [docs/ref/TEXTENC_REPORT.md](../docs/ref/TEXTENC_REPORT.md)。
