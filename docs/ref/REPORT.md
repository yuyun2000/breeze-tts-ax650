# Breeze TTS 2 参考推理与对拍报告（task-5）

> 负责人：ref-2038。写作用域：Mac `_work/ref/`、20.38 `~/work/breeze-tts/ref/`。
> 结论：**参考推理已跑通并完全可复现**；golden tokens/wav/probe 已就绪；对拍工具已验证；ASR 可懂度 3/3 逐字准确。板端只需按 `GEN_LOGIC.md` 实现采样循环，用本报告的 golden 做逐级对拍。

---

## 0. TL;DR（给 lead / 板端）

| 项 | 结论 |
|---|---|
| 参考推理环境 | Mac M4 Pro / macOS 26.6.2 / Python 3.12.8 / **torch 2.15.0 (macOS arm64)** / **transformers 4.57.3** / qwen-tts 0.1.1 / bf16 / eager / CPU |
| 可行性 | 20.38 为 QEMU 虚拟 CPU（无 AVX/AVX2/FMA）不可行 → 按 lead 指示改 Mac；Mac 上 **模型加载 ~0.95 s（mmap）、生成 ~0.29 s/帧（RTF ≈3.6）**，峰值 RSS ≈ 5.5 GB |
| golden | `golden_zh`（采样 seed42，96 帧 7.68 s，EOS@96）、`golden_zh_greedy`（贪心 100 帧 8.0 s，上限截断）、`golden_zh_greedy200`（贪心 111 帧 8.88 s，**EOS@111** 证明贪心能正常收敛） |
| 可复现性 | **贪心两次独立进程逐 token 100% 一致（首处差异帧 = 无）**；probe 数组逐字节一致；sampled seed42 两次 20 帧实验也完全一致 |
| 智能验证 | sherpa-onnx + SenseVoice-int8 对 3 段 golden wav 全部转写 **与输入文本逐字一致**（含标点），可懂度无问题 |
| tokens-only 路径 | **block 掉 `qwen_tts` 后仍能生成 tokens**（`model.generate(output_audio=False)` 不触发 codec）→ 板端可只用 NPU speech_tokenizer 解码 |
| 板端对拍入口 | `golden_zh_greedy/first20_frames.npz`（前 20 帧 tokens + 每帧 backbone top5 logits + hidden_norm 摘要）、`compare.py` |

---

## 1. 环境与版本

| 项 | 值 |
|---|---|
| 机器 | Apple M4 Pro，24 GB 统一内存，macOS 26.6.2 |
| Python | 3.12.8（venv：`/Users/yunyu/Documents/DSH/AX-model/_work/venv-ref`，1.6 GB） |
| torch | **2.15.0**（macOS arm64 wheel）⚠️ Breeze requirements 写 2.9.1，本机安装环境只有 2.15.0 可用；见 §10 |
| transformers | **4.57.3**（与 Breeze requirements 一致） |
| qwen-tts | 0.1.1（codec 解码；生成 tokens 不需要它，见 §7） |
| 其它 | numpy 2.5.4、librosa 1.0.0、soundfile 0.14.0、safetensors 0.8.0、sherpa-onnx 2.0.1、accelerate 1.12.0 |
| 模型 | `BreezeBlue/Breeze-TTS-2`：3 个组件 3.483 B 参数、6.97 GB bf16（2 shards）；`audio_tokenizer/model.safetensors` 682,293,092 B |
| 源码 | `_work/breeze-tts-src`（官方 repo checkout，未改动；`attn_implementation="eager"`，无 cudagraph/fast 路径） |
| 推理参数 | dtype bf16、device cpu、eager attention、`torch.set_num_threads(10)` |

**获取方式**：shard1 + audio_tokenizer 走 LAN rsync（20.38 → Mac，7.2 GB）；shard2 因 20.38 下载卡在 1.82/2.0 GB，Mac 直接从 hf-mirror 下载（2,004,567,152 B，23.5 MB/s，85 s）。

---

## 2. 交付物清单

### Mac `_work/ref/`

| 路径 | 大小 | 说明 |
|---|---|---|
| `GEN_LOGIC.md` | 24 KB | 生成逻辑规格书（板端 runtime 对照，含行号/实测修正） |
| `ref_generate.py` | — | 参考推理脚本（采样参数/seed/probe/贪心/voice clone） |
| `compare.py` | — | 对拍工具（tokens + wav） |
| `export_first20.py` | — | 从 golden+probe 导出板端对拍小文件 |
| `README.md` | — | 目录说明/复现命令 |
| `golden_zh/` | 3.3 MB | 采样 golden（seed42）：`out.wav` + `tokens.npz` + `probe.npz` + `meta.json` + `first20_frames.{npz,json}` |
| `golden_zh_greedy/` | 3.3 MB | 贪心 golden（100 帧）：同上 |
| `golden_zh_greedy200/` | 1.5 MB | 贪心 200 帧上限（EOS@111）：wav/tokens/probe/meta |
| `golden_zh_greedy_repeat/` | 404 KB | 贪心重跑（仅 tokens/probe/meta，复现性证据） |
| `asr-sensevoice/` | 241 MB | SenseVoice-int8 ASR 模型（临时，可删） |
| `model-breeze-tts-2/` | 7.2 GB | 完整模型（**临时**，板端对拍后删） |
| `_work/venv-ref/` | 1.6 GB | Mac 推理 venv（**临时**，对拍后删） |

### 20.38 `~/work/breeze-tts/ref/`
`GEN_LOGIC.md`、`ref_generate.py`、`compare.py`、`export_first20.py`、`README.md`、`REPORT.md`、`golden_zh*/`（与 Mac 同步；大文件不上传）。

---

## 3. golden 清单与实测数据

**公共 prompt**（Voice Design / `tts_instruction`，cfg_scale=1.0 无 CFG，speaker=S0）：
- text：`今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？`
- instruction：`用温柔亲切的女声，语速稍慢。`
- **prompt ids（35 token，以 BOS=2 开头，无 EOS）**：
```
[2, 262146, 262156, 237105, 171109, 239148, 238201, 146440, 238389, 236900,
 238384, 238121, 241066, 239364, 236924, 262157, 22601, 122362, 56033, 236900,
 6932, 24163, 67038, 238302, 238302, 238702, 236924, 237408, 28129, 237648,
 73189, 23533, 237389, 238463, 237536]
```
（`<bos>`=2、`[S0]`=262146、`<ins_bos>`=262156、`<ins_eos>`=262157、instruction 12 token、text 18 token）

| golden | 模式 | 帧数 | EOS | 音频时长 | 生成耗时 | RTF | codec 解码 | wav 采样点 | tokens.npz |
|---|---|---|---|---|---|---|---|---|---|
| `golden_zh` | 采样 seed42（temp .9/top_k 50/top_p 1.0/rep 1.1，depth 同） | 96 | **是 @96** | 7.68 s | 28.4 s | 3.70 | 2.37 s | 184,320 | 650 KB |
| `golden_zh_greedy` | 贪心（temp=1/top_k=0；depth 也贪心） | 100 | 否（达 100 上限） | 8.00 s | 29.2 s | 3.65 | 1.77 s | 192,000 | 664 KB |
| `golden_zh_greedy200` | 贪心，max 200 | 111 | **是 @111** | 8.88 s | 32.2 s | 3.63 | 1.56 s | 213,120 | — |
| `golden_zh_greedy_repeat` | 贪心重跑（skip decode） | 100 | 否 | — | 29.1 s | 3.63 | — | — | 5.6 KB(压缩后) |

- 模型加载：0.93–0.96 s（safetensors mmap）；**稳态 ≈0.29 s/帧**（约 3.5 帧/s，RTF ≈3.6，100 帧 ≈29 s）。
- 峰值 RSS：**5.53 GB**（`/usr/bin/time -l` 实测短序列；含模型 + codec 前向）。
- 音频电平：sampled peak 0.542 / rms 0.081；greedy peak 0.421 / rms 0.063（24 kHz，mono）。
- `tokens.npz` 字段：`tokens` [T,16] int32（有效帧，已截 EOS pad 帧）、`tokens_with_eos`、`prompt_ids`、`audio` float32。`meta.json` 含 prompt/sampling/耗时/config/eos_frame_index 等全部参数。
- probe（`probe.npz`，`--save-scores --probe-frames 20`）：`backbone_topk_val/idx` [20,5]（每帧选 cb0 的原始 logits top-5，**未加温度/未掩码**）、`hidden_norm` [20,2048]（该帧 depth decoder 输入的 post-norm hidden）、`text_embeds` [35,2048]（文本编码器→proj 输出=backbone prefill 输入）、`depth_logits` [20,15,2051] f16（15 步采样前 logits）。

---

## 4. 可复现性结论（重点）

| 实验 | 结果 |
|---|---|
| 贪心两独立进程（`golden_zh_greedy` vs `golden_zh_greedy_repeat`） | **tokens [100,16] 100% 逐 token 一致；第一处差异帧 = 无（-1）** |
| 同上 probe 数组（topk/hidden_norm/depth_logits/text_embeds） | **逐字节一致**（`np.array_equal` 全 True） |
| 贪心 200 帧的前 100 帧 vs 贪心 100 帧 | **完全一致** → max_frames 截断是稳定前缀 |
| sampled seed42 两次独立 20 帧（另一实验） | tokens + probe 完全一致 |
| ASR 三次转写 | 3/3 与输入文本逐字一致 |

结论：**同机同版本下参考侧完全确定（贪心与固定 seed 采样都逐 token 可复现）**。板端对拍建议用 `golden_zh_greedy`（贪心、确定性）作为主基准；`golden_zh`（采样）用于听感/质量对照。注意 int8 量化 + 不同硬件会引入 logits 抖动，验收口径应看「首分叉帧 + 一致帧比例 + ASR/听感」，而非要求全 750 帧逐 token 相等。

---

## 5. 与 GEN_LOGIC 的一致性实测核对

| GEN_LOGIC 结论 | 实测证据 |
|---|---|
| prompt ids 以 `<bos>`=2 开头、末尾无 EOS、`[S0]`/`<ins_bos>`/`<ins_eos>`=262146/262156/262157 | 35-token ids 实测（§3），tokenizer round-trip 行为已写进 GEN_LOGIC §2.1 |
| EOS 只认 backbone id 2051；id 1 不停止；无「前 N 帧禁 EOS」 | 采样在 96 帧自然 EOS；贪心 100 帧上限内未 EOS、放宽到 200 帧后在 **111 帧自然 EOS** |
| 默认 cfg_scale=1.0 → 无 CFG 分支（单次前向） | `lm_head` 调用次数 = 生成行数（96+EOS=97、贪心=100），无双分支 |
| depth decoder 固定 15 步、无 EOS | `depth_head` 调用 = 行数×15（1440=96×15、1500=100×15）；**EOS 行不跑 depth**（97 行只跑 96 帧 depth） |
| 音频帧 embedding = 16 组求和；文本位置用文本编码器输出 | 源码逐行核对（`breeze.py:792-803, 1567-1578`）；probe 保存 `text_embeds`[35,2048] 供板端直接对拍文本编码器 |
| backbone→depth 的 hidden 是 post-norm（2048 维） | probe `hidden_norm`[F,2048] 实测 mean≈0.007/std≈0.70/l2≈31.9（逐帧摘要见 §3） |
| 采样顺序：温度→top_k→掩码[2048:2051]→softmax→multinomial | 源码 + transformers 4.57.3 processor 顺序核对（GEN_LOGIC §3.2）；本轮未单独做 logits 级反向验证 |

---

## 6. 权重构成与可裁剪分析（供 repack / 板端预算）

从 checkpoint 实测（safetensors header 统计，3.483 B 参数 / 6.97 GB bf16）：

| 组件 | tensors | 参数量 | bf16 体积 | 推理是否必需（Voice Design） |
|---|---|---|---|---|
| `backbone_model`（28 层，不含音频表） | 309 | 1409.4 M | 2.82 GB | ✅ talker |
| `text_encoder`（T5Gemma2 26 层） | 341 | 999.9 M | 2.00 GB | ✅ |
| `text_encoder_proj`（1152→2048） | 1 | 2.4 M | 0.00 GB | ✅ |
| `depth_decoder`（12 层 + **16×2051×2048 音频表，与 backbone tied、checkpoint 只存一份**） | 112 | 434.3 M | 0.87 GB | ✅ code predictor |
| `lm_head`（2052×2048） | 1 | 4.2 M | 0.01 GB | ✅ |
| `embed_text_tokens`（262158×2048） | 1 | 536.9 M | 1.07 GB | ❌ 推理不用（有 text_encoder 时），**可丢** |
| `codec_model`（Mimi，config codec_config） | 350 | 96.2 M | 0.19 GB | ❌ token-only 推理不用（仅训练/编码路径 `_get_audio_token_from_batch`），**可丢** |
| **合计** | 1115 | **3.483 B** | **6.97 GB** | — |
| **Voice Design 最小部署集** | — | **2.850 B** | **5.70 GB**（丢上两项省 1.26 GB） | — |

注：`embed_text_tokens` 与 `codec_model` 可丢；音频 16 组 embedding 表只存一份（与 depth embed tied），板端需要 16 组表（可裁成 bf16/fp16 或按板端需要重新打包）。

---

## 7. tokens-only 路径验证（板端关键结论）

**结论：不需要 `qwen_tts` 包/codec 就能跑 talker+depth 得到 16×T codes。**

验证方法：在 `qwen_tts` import 处注入 blocker（任何 `qwen_tts` 导入直接 ImportError），再跑 `ref_generate.py --skip-decode`：
- 生成成功：4 帧 tokens（与正常路径同 seed 结果一致）；
- 结束后 `'qwen_tts' in sys.modules == False`；
- `model.generate(..., output_audio=False)` 全程未触碰 codec decoder / audio_tokenizer。

细节说明：
- `BreezeForConditionalGeneration` 类里仍会实例化 `codec_model`（Mimi，96.2M 参数，从 checkpoint 加载，~0.19 GB bf16），但**token-only 生成不调用它**；它会占内存/加载时间，repack/板端预算里可直接剔除（见 §6）。
- 真正把 codes 变波形的是**独立**的 `audio_tokenizer`（Qwen3TTSTokenizerV2，与 `Qwen/Qwen3-TTS-Tokenizer-12Hz` 字节一致）→ 板端完全可以用 AXERA 的 speech_tokenizer decoder axmodel 替换。
- 也就是说：**tokens 对拍不依赖任何 codec**，比音频对拍更可靠；板端先对齐 tokens（first20 + 全序列），再看 wav。

---

## 8. `compare.py` 用法与输出字段说明（板端交接）

### 8.1 命令行

```bash
# 1) 只对 tokens（板端导出 tokens.npz，键名 tokens/audio_codes/codes 均可）
python compare.py --golden golden_zh_greedy --cand ~/board_out/tokens.npz --out-dir cmp_board

# 2) tokens + wav
python compare.py --golden golden_zh_greedy \
  --cand-codes ~/board_out/tokens.npz --cand-wav ~/board_out/out.wav --out-dir cmp_board

# 3) 只对 wav
python compare.py --golden-wav golden_zh_greedy/out.wav --cand-wav ~/board_out/out.wav --out-dir cmp_wav

# 4) 板端产物是目录（含 tokens.npz + out.wav）时直接给目录
python compare.py --golden golden_zh_greedy --cand ~/board_out --out-dir cmp_board
```
支持输入：`.npz`（自动找 `tokens/audio_codes/codes/token_ids/...` 键）、`.npy`、`.json`、`.txt`（每行一帧的 16 个整数）。golden 也可直接给目录（自动读 `tokens.npz` + `out.wav` + `meta.json`）。
输出：`<out-dir>/report.md` + `report.json`（终端也打印 verdict）。

### 8.2 `report.json` 字段

`token_compare`：
- `golden_frames` / `cand_frames` / `overlap_frames`：帧数；
- `exact_frame_rate`：**整帧 16 个 codebook 全对的帧比例**（主指标）；
- `codebook_acc`[16] / `codebook_acc_mean`：逐 codebook 一致率 / 平均；
- `first_divergence_frame`：第一处任一 codebook 不一致的帧号（-1 = 全一致）；
- `identical_prefix_frames`：从头完全一致的帧数；
- `cb0_acc` / `cb0_first_divergence_frame`：只比 codebook0（talker 层，最敏感）；
- `mismatch_per_codebook`[16] / `worst_codebook` / `mismatch_abs_diff_mean`：不匹配分布；
- `best_offset`：在 ±32 帧内搜索最优全局偏移（`offset`/`acc`/`frames`），用于诊断起始错位；
- `prompt`：若双方 npz 都含 `prompt_ids`，报 `same` / `first_diff`（**板端 tokenizer 第一道验证**）。

`audio_compare`：
- `golden_sr`/`cand_sr`/`target_sr`、`*_duration_s`、`duration_diff_s`；
- `golden_rms`/`cand_rms`/`golden_peak`/`cand_peak`；
- `best_lag_samples`/`best_lag_ms`：波形互相关最优对齐 lag（诊断延迟/帧数偏差）；
- `waveform_pearson`：对齐后波形相关系数；`aligned_rms_ratio`；
- `logmel_cosine` / `logmel_pearson` / `logmel_l1_db` / `logmel_l1_db_voiced`：log-MEL（80 mel，n_fft 1024/hop 256，24 kHz）结构相似度与距离；
- `verdict`：自动结论（token ≥0.99 判等价；0.8–0.99 判主流程正确/数值分叉；<0.8 判需排查）。

**建议板端验收顺序**：prompt ids 相同 → 前 20 帧 tokens 一致（first20）→ 全序列 exact_frame_rate/首分叉帧 → wav log-MEL/ASR 听感。

---

## 9. 板端快速对齐数据：`first20_frames.{npz,json}`

位置：`golden_zh/first20_frames.*`（采样）与 `golden_zh_greedy/first20_frames.*`（贪心，**板端主用**）。体积：npz ~65 KB + json ~15 KB。

`first20_frames.npz` 字段：
| 键 | shape | 说明 |
|---|---|---|
| `tokens_first20` | [20,16] int32 | 前 20 帧 16 个 codebook token（CB0..CB15） |
| `prompt_ids` | [35] int32 | 板端必须逐 token 一致的 prompt ids（含 BOS=2） |
| `backbone_topk_idx` | [20,5] int32 | 每帧 talker 选 cb0 时 logits 的 top-5 token id |
| `backbone_topk_val` | [20,5] float32 | 对应原始 logits（未温度/未掩码；板端比对应一致到 ~1e-2） |
| `hidden_norm_first20` | [20,2048] float16 | 每帧 depth decoder 输入的 post-norm hidden |
| `hidden_norm_summary` | [20,5] float32 | `[mean, std, l2, first4_0, first4_1]` |
| `n_frames` / `eos_reached` | 标量 | 全序列帧数 / 是否自然 EOS |

`first20_frames.json` 是人类可读版：每帧 tokens + `backbone_top5`（token, logit）+ hidden 摘要（mean/std/l2/first4/min/max）。
示例（greedy 第 0 帧）：tokens `[1376,1684,1977,826,780,1608,1641,425,1133,1628,2033,531,472,629,1775,733]`；backbone top5 `1376(19.25), 1138(19.125), 1464(18.25), 1995(17.25), 1221(17.25)`；hidden_norm mean 0.006692 / std 0.704722 / l2 31.8935。

**定位逻辑**：prompt ids 不同→tokenizer；`text_embeds` 不同（需板端导出文本编码器输出）→文本编码器 axmodel；前几帧 backbone top5 不同→talker；cb0 对但 cb1..15 不同→depth/code predictor；全对但 wav 差→speech_tokenizer decoder。

---

## 10. ASR 智能验证（可懂度）

- 工具：`sherpa-onnx==2.0.1`（`_work/venv-ref` 已装）+ SenseVoice-int8 中文模型（239 MB，hf-mirror `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`），模型文件在 `_work/ref/asr-sensevoice/`。
- 命令（可直接复用）：
```python
import sherpa_onnx, soundfile as sf, librosa
rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model='asr-sensevoice/model.int8.onnx', tokens='asr-sensevoice/tokens.txt',
    num_threads=4, use_itn=True, language='zh', debug=False)
audio, sr = sf.read('golden_zh/out.wav', dtype='float32')   # 24 kHz
a16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
s = rec.create_stream(); s.accept_waveform(16000, a16); rec.decode_stream(s)
print(s.result.text)
```
- 结果：

| golden | 帧数 | ASR 转写 | 与输入文本 |
|---|---|---|---|
| `golden_zh`（采样） | 96 | 今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？ | ✅ 逐字一致（含标点） |
| `golden_zh_greedy` | 100（截断） | 同上 | ✅ 逐字一致 |
| `golden_zh_greedy200` | 111（自然 EOS） | 同上 | ✅ 逐字一致 |

- 性能：ASR 初始化 0.70 s，单段 7.7 s 音频解码 **0.09 s**。→ **golden 音频可懂度无问题**，可作为板端 wav 的智能验收基线（板端 wav 同样转写，应得同一句）。

---

## 11. 版本差异记录（torch 2.15.0 vs requirements 2.9.1）

- Breeze requirements pin `torch==2.9.1`；Mac 本机 pip 可用版本为 **2.15.0**（macOS arm64），已记录进每个 golden 的 `meta.json`（`torch` 字段）。
- 观察到的差异：无异常。证据：tokens 逐 token 可复现；ASR 3/3 逐字准确；wav 电平/时长正常；hidden/top5 logits 数值合理（top1 logit ≈19，hidden l2 ≈32）。
- 风险与对策：若板端对拍出现**系统性** logits/音质偏差，优先怀疑 torch/transformers 版本而非板端实现，可在有 AVX2 的 Linux x86 宿主上用 torch 2.9.1 复跑本 golden 对比（20.38 QEMU 无 AVX 不可行；可考虑其他 x86 机或 CI）。此条已写入 `GEN_LOGIC.md` §9。

---

## 12. 清理清单（等板端对拍完成后统一执行）

**保留（小文件，最终交付）**：`GEN_LOGIC.md`、`ref_generate.py`、`compare.py`、`export_first20.py`、`README.md`、`REPORT.md`、`golden_zh*/`（约 8.5 MB 总量）、`model-meta/`。

**待 lead 确认后删除（Mac，约 9 GB）**：
1. `_work/ref/model-breeze-tts-2/`（7.2 GB）— 板端对拍完成后再删；
2. `_work/venv-ref/`（1.6 GB，含 torch/transformers/qwen-tts/sherpa-onnx）— 同上；
3. `_work/ref/asr-sensevoice/`（241 MB）— ASR 模型，若不再复跑可删；
4. `/tmp/try_probe /tmp/try_noqwen /tmp/samp_rep1 /tmp/samp_rep2 /tmp/cmp_*`（临时）。
**注意**：golden 的 `tokens.npz` 里含 float32 音频（约 650 KB/段），保留即可，不必再压。

---

## 13. 命令速查（复现 / 板端对接）

```bash
cd ~/Documents/DSH/AX-model/_work/ref
PY=../venv-ref/bin/python
MODEL=./model-breeze-tts-2
SRC=~/Documents/DSH/AX-model/_work/breeze-tts-src

# 采样 golden（96 帧自然 EOS）
$PY ref_generate.py --outdir golden_zh --model $MODEL --src-dir $SRC \
  --text "今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？" \
  --instruction "用温柔亲切的女声，语速稍慢。" --seed 42 --max-frames 100 \
  --threads 10 --save-scores --probe-frames 20

# 贪心 golden（确定性基准）
$PY ref_generate.py ... --greedy --outdir golden_zh_greedy
# 贪心 200 帧（观察 EOS 收敛）
$PY ref_generate.py ... --greedy --max-frames 200 --outdir golden_zh_greedy200

# 导出板端 first20
$PY export_first20.py --golden-dir golden_zh_greedy

# 对拍（板端产物）
$PY compare.py --golden golden_zh_greedy --cand-codes board/tokens.npz \
  --cand-wav board/out.wav --out-dir cmp_board
```

---

## 15. RoPE scaling 消融（2026-11-11 追加）

**目的**：验证 pulsar2 通用 builder 若不支持 llama3 rope 时「default rope 近似」的质量代价；同时核实 §1 的 RoPE 事实。

**关键事实（实测修正）**：`backbone_model_type="qwen3"` → backbone 走 `BreezeBackboneAdapter`，用嵌套 `backbone_config`（Qwen3, **rope_theta=1e6、rope_scaling=null**）→ **talker 实际是 default rope，顶层 llama3 配置不生效**；**depth decoder 的 llama3（theta 5e5, orig 16, factor 32）真实生效**。

**变体（贪心 111 帧，vs 原版）**：

| 变体 | 整帧一致率 | 首分叉帧 | logMEL cos | 波形 r | ASR |
|---|---|---|---|---|---|
| b backbone rope=None | **1.000** | **-1** | **1.000** | **1.000** | 与基线逐字一致 |
| c depth rope=None | 0.010 | 1 | 0.325 | 0.161 | 词全对，字符准确率 96.4%（仅句末标点差异）；EOS 105 vs 111 |
| d 两者 None | 0.010 | 1 | 0.325 | 0.161 | 同 c |

**结论**：① talker 用 default rope(theta=1e6) 即**精确**，repack talker config 应改 `rope_theta=1e6, rope_scaling=null`（BREEZE_PLAN §3.1 需修正）；② depth 的 llama3 rope **必须精确实现**，否则 token 从第 1 帧全分叉（可懂度损失不大但 token 级对拍会失败）；③ 若 builder 不支持，最稳方案是 depth 固定 15 步/位置 0..15 → 预计算 llama3 cos/sin 常量导静态/展开 ONNX。详见 `GEN_LOGIC.md §11`、`rope_ablation{,_long}/ABLATION.md`、`rope_ablation/*/summary.json`。

## 14. 待办 / 需要 lead 处理

1. **任务板**：`task-5` 仍因 `blocked_by=["task-1"]`（task-1 处于 in_progress）无法 claim；本任务交付物已全部完成，请 lead 在 task-1 完成/关闭后把 task-5 置 ready（或直接标记完成）。
2. 板端发来 wav/tokens 后，我可用 `compare.py` 直接出对拍报告；如需我代跑，给出路径即可。
3. 清理节奏由 lead 统一决定（§12 清单，先别删模型/venv）。

---

*2026-11-11，ref-2038。所有数字均为本机实测；golden 与代码同步于 Mac `_work/ref/` 与 20.38 `~/work/breeze-tts/ref/`。*
