# Breeze TTS 2 on AX650

把 [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)（约 3.45 B 参数、中英双语、
Artificial Analysis TTS 榜单开源第一）转换并部署到 **AX650/AX8850 NPU**：完整的转换流水线 +
板端纯 Python 运行时（numpy + pyaxengine，无 PyTorch）+ 可复现实测报告。

> 🤗 **模型包（axmodel + runtime，约 4.5 GB）**：<https://huggingface.co/yunyu1258/breeze-tts-2-ax650>
>
> ⚠️ **许可证**：Breeze TTS 2 的**模型权重、衍生模型与自托管输出仅限研究与非商业用途**
> （BreezeBlue Research and Non-Commercial License 1.1，见 [licenses/BREEZEBLUE_LICENSE.txt](licenses/BREEZEBLUE_LICENSE.txt)）。
> 本仓库**代码**为 Apache-2.0（[LICENSE](LICENSE)）；本仓库/HF 发布的 axmodel 是 Breeze 权重的**衍生模型**，
> 同样受 BreezeBlue 非商业许可证约束，**不可商用**。Qwen3-TTS tokenizer 组件版权归 QwenLM（Apache-2.0）；
> 部分 codec 编译产物复用自 AXERA 公开仓库（见 [NOTICE](NOTICE)）。

## 1. 做了什么

| 组件 | 原始结构 | AX650 部署方式 | 状态 |
|---|---|---|---|
| 文本编码器 | T5Gemma2 encoder（26L, hidden 1152, 双向） | 两段静态 ONNX → `pulsar2 build`（U16 激活 + S8 权重，fused-op-aware 精度修复）；embedding 查表在 CPU（bf16 LUT mmap） | ✅ NPU（链式 cos 0.99968） |
| Backbone（talker） | Qwen3 式 28L，hidden 2048，预测 codebook0 | 重打包成 Qwen3ForCausalLM → `pulsar2 llm_build --model_type qwen3` 逐层 axmodel | ✅ NPU |
| Depth decoder | Llama 式 12L，hidden 1024，预测 codebook1..15 | 重打包成 LlamaForCausalLM → `--model_type llama` 逐层 axmodel + 15 个 head | ✅ NPU |
| 音频 codec | Qwen3-TTS Tokenizer 12Hz（Mimi 系），24 kHz | **直接复用 AXERA 已编译 Qwen3-TTS speech_tokenizer axmodel**（与 Breeze codec 权重字节一致） | ✅ NPU |

**关键发现**：Breeze 顶层 config 的 llama3 rope 对 backbone 是死配置（backbone 走嵌套 `backbone_config`：
qwen3 / θ=1e6 / 普通 RoPE）；只有 depth decoder 真正用 llama3 rope。Pulsar2 的 Qwen3-TTS 专用 model_type
（mRoPE + 强制 qk-norm）与 Breeze 不匹配，通用 `qwen3` / `llama` model_type 才是精确路径。详见
[docs/experience.md](docs/experience.md)。

## 2. 板端实测（AX650C 真板）

| 项 | 结果 |
|---|---|
| 文本编码器（两段串行） | 加载 1.58 s；单次编码（34 token, L=128 桶）**196 ms**；链式 cos **0.99968** vs ONNX fp32；CMM +742 MB（编码后释放） |
| talker decode | **133–146 ms/帧**（28 层 INT8，kv 512） |
| depth decoder | **≈500–570 ms/帧**（15 步 × 12 层 INT8 + U16 lm_head 在 NPU；`--lmhead-cpu` 可兜底） |
| codec decode | 522 ms（5.12 s 音频）/ 1249 ms（6.72 s） |
| 整句端到端（已交付 demo，含 CPU-head 兜底） | 短句 **72.2 s / 5.12 s = RTF 14.1×**；长句 **73.4 s / 6.72 s = RTF 10.9×**；U16 head 全 NPU 后 depth 约省 50 ms/帧（预期 RTF ~11×） |
| 资源 | CMM 3471 MB / 6144 MB；python RSS ~1.1 GB；模型加载 ~11 s |
| 精度 | talker frame0 logits 与 PyTorch 参考 **top5 逐位一致**、output_norm cos 0.99942；codec 同 codes 幅度谱相关 0.99958 |
| **ASR 验收** | 4 条 demo（中文短句/长句/英文/中文场景句）采样模式经 paraformer-zh + SenseVoice 转写**全部与输入文本一致**；hp 对照版本同样通过（见 `docs/ref/RUNTIME_REPORT.md`） |

非实时（RTF≈11–14×），适合离线合成 / demo；瓶颈是 depth decoder 的 180 次 layer forward 与 talker 每帧权重流。
demo wav 在 HF 包 `demos/`（4 条 fast + 1 条 quality 对照）。

## 3. 上手

```bash
# 1) 下载模型包
export HF_ENDPOINT=https://hf-mirror.com
hf download yunyu1258/breeze-tts-2-ax650 --local-dir ./breeze-tts-2-ax650

# 2) 板端（AX650，装 pyaxengine + ml-dtypes + tokenizers）
cd breeze-tts-2-ax650
export LD_LIBRARY_PATH=/soc/lib
python3 runtime/run_breeze_tts.py \
  --text "今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？" \
  --instruction "用温柔亲切的女声，语速稍慢。" \
  --out out.wav --dump-tokens tokens.npz
```

## 4. 目录

```
scripts/convert/   重打包 (repack_breeze.py) + 文本编码器导出 + pulsar2 编译脚本
scripts/board/     板端 runtime + 对拍/校验工具
docs/              架构 / 转换 / 板端部署 / 经验 / 参考生成规格 (ref/)
reports/           conversion.md / board.md（实测数字）
models/            模型包结构说明 + configuration.json
```

## 5. 已知限制 / 经验

1. **文本编码器混合精度的坑（已解决，值得记录）**：Pulsar2 8.0 里只按 ONNX op 名配 U16 不够——
   **融合后的节点名（FullyConnected / RMSNormalization / RotaryEmbedding / Gelu 等）也必须一起配 U16**，
   否则这些节点输出仍是 U8，cos 只有 0.90；INT8 更只有 0.45。auto 混合精度会退化成"未量化"并崩在 FP32 AxRoPE
   （工具链 bug）。最终方案：S8 权重 + 全 U16 激活 + fused-op-aware layer_configs，链式 cos 0.99968。
   详见 [docs/experience.md](docs/experience.md) 与 [docs/ref/TEXTENC_REPORT.md](docs/ref/TEXTENC_REPORT.md)。
2. depth decoder INT8 的量化噪声会让 greedy free-run 从第 3 步起分叉（音频仍可懂，采样模式 ASR 通过）；
   `quality` 模式（FP32-heavy depth 1.3 GB）可提升保真度，代价 ~2.5× 延迟。
3. Voice Clone / Voice Direction（需要 codec encoder + speaker encoder）未上板；CFG 未实现（默认 cfg_scale=1.0）。
4. **非商业**：见上方许可证说明。
