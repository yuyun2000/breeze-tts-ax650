# RUNTIME_REPORT — Breeze TTS 2 板端 Python + pyaxengine 运行时（最终版）

> 板卡：`192.168.10.105`（AX650C / AX Engine 2.12.0s / pyaxengine 0.1.3 / python 3.10.12）
> 负责人：board-105　日期：2026-11-11 18:15–19:40
> 运行时代码：板端 `/root/breeze-tts/run_breeze_tts.py`，Mac `_work/board/run_breeze_tts.py`
> 协议/环境依据：`_work/board/BOARD_REPORT.md`、`_work/board/RUNTIME_SPEC.md` 附录 A/B；生成逻辑：`_work/ref/GEN_LOGIC.md`
> 参考对拍：`_work/ref/`（ref-2038 的 golden + probe.npz）

---

## 0. TL;DR

| 项 | 结果 |
|---|---|
| 板端全链路（文本 embedding → talker 28 层 → depth 12 层×15 步 → codec 8 层+vocoder → 24k wav） | ✅ **跑通，产出可听中文语音** |
| **默认 demo 模式（fast/INT8 depth，采样 temp0.9/top_k50）** | **RTF ≈ 11–14×**；ASR（paraformer-zh）**逐字 100% 正确**（短句、长句各 1 条，SenseVoice 交叉验证一致） |
| quality 模式（hp/FP32 depth） | RTF ≈ 23×；free-run code match 0.81→0.90；teacher-forced top1 与 INT8 持平（见 §5） |
| 关键模块精度 | talker frame0 logits top5 与 golden **完全一致**；codec 复用的幅度谱相关 **0.9996**；depth 为当前唯一误差源 |
| 资源 | fast: CMM 3471MB/6144、RSS ~1.0GB；hp: CMM 4394MB、RSS ~1.1GB；板端 free 12G |
| textenc | **未上线**（Pulsar2 AxRoPE FP32 backend bug + INT8 精度不达标）；demo 用 golden `text_embeds` 注入 → **除 textenc 外全 NPU**，作为已知限制交付 |

---

## 1. 架构与数据流

```
tokenizer (Breeze tokenizer.json, BOS=2; prompt = [S0]<ins_bos>{instruction}<ins_eos>{text})
   │  ids [L]
   ├─(路线A 已实现) TextEncoderRunner: 604MB bf16 embed mmap + embed_scale sqrt(1152) + eoi_embedding
   │      → textenc axmodel(s) [1,128,1152]+mask → text_embeds [L,2048]     ← 板端当前未达标
   └─(路线B 已实现, demo 用) --text-embeds golden.npz : 直接注入 text_embeds [L,2048]
   │
   ▼ talker prefill（28 层 per-layer axmodel，chunk=128，kv_cache_len=512）
   ▼ talker post（RMSNorm + lm_head）→ logits[2052] + output_norm[2048]
   ▼ 逐帧循环：
        cb0 = sample(rep_penalty 1.1 → temp 0.9 → top_k 50 → suppress 2048..2050 → softmax)
        EOS: cb0 == 2051 → 停止
        depth: pos0 = proj(output_norm) [1024]，pos1 = E'_0[cb0]，15 步自回归（每步 12 层 axmodel + post + CPU lm_head）
        frame_emb = Σ_{g=0..15} E_g[cb_g]（16 组未投影表求和，2048 维）
        → talker decode（28 层，1 token）→ post → 下一帧
   ▼ codes [T,16] → codec decode：decode_pre_inproj(325 桶) → 8 层 transformer(滑窗72, kv127)
        → post [512→1024] → decode_post_t64 分桶 vocoder（left_context=25, capacity=39）→ 24kHz wav
```

实现要点（全部按实测 axmodel I/O，运行时从 `get_inputs/get_outputs` 动态读取，不硬编码）：
- 通用 shape_group 适配：decode=group0，prefill=group1..N（本模型 N=5，容量 640 token）；
- `indices` 行数动态（本 talker=1 行、AXERA qwen3_tts_talker=3 行都支持）；decode mask 列数 = kv_len+1，**当前 token 固定占最后一列**；
- KV cache 为 host numpy bf16，按 cache 位置写回；
- 采样顺序严格按 GEN_LOGIC（backbone: rep→temp→topk→suppress；depth: suppress→temp→topk，15 步无 EOS）；
- 帧 embedding 16 组求和；depth 输入用**投影后**表；talker→depth 用 post 的 `output_norm`；
- 15 个 depth lm_head 用 **CPU fp32 matmul**（axmodel head 输出被量化损坏，见 §6.2）；
- `--text-embeds` 注入、`--codes` codec-only、`--probe-frames` 逐帧探针、`--dump-tokens`、`--lmhead-cpu`、`--depth-dir`、`--keep-textenc`、`--no-prewarm`。

Presets：`breeze`（正式）/ `qwen3tts06`（AXERA 0.6B 包胶水自检）。

---

## 2. 板端模型清单（1.9GB 系统内存 / 6GB CMM）

| 组件 | 路径 | 体积 | 说明 |
|---|---|---|---|
| talker 28 层 | `breeze/builds/talker/qwen3_p128_l{0..27}_together.axmodel` | 1.6GB | 58MB/层，kv_cache_len=512；generic qwen3 build |
| talker post | `breeze/builds/talker/qwen3_post.axmodel` | 3.4MB | output logits[1,1,2052] + output_norm[1,1,2048] |
| depth fast (INT8) | `breeze/builds/code-predictor/llama_p16_l{0..11}_together.axmodel` + `llama_post.axmodel` | 0.58GB | 30.7MB/层，kv_dim=256，kv_len=32 |
| depth quality (hp) | `breeze/builds/code-predictor-hp/` | 1.3GB | 111MB/层（highest_mix_precision / FP32 权重） |
| depth lm_heads | `breeze/builds/code-predictor/code_predictor_lm_head_{0..14}.axmodel` | 34MB | **输出被量化损坏，默认不用**（改用 CPU fp32，见 §6.2） |
| depth CPU head 权重 | `breeze/embeds/codebooks_head_15x1024x2051.bf16.bin` | 63MB | [15,in=1024,out=2051]，预载 fp32 126MB RAM |
| codec decode（复用 AXERA 产物） | `qwen3tts-ref/speech_tokenizer/` | 141MB | decode_pre_inproj + 8 层 + post + decode_post_t64 |
| embedding 表 | `breeze/embeds/audio_embed_16x2051x2048.bf16.bin`（134MB）、`codec_embed_proj_15x2051x1024.bf16.bin`（63MB）、`inputs_embeds_projector_1024x2048.bf16.bin`（4MB）、`cp_norm_1024.bf16.bin` | 201MB | group-major 布局，已交叉验证（E'=E·Pᵀ，cos=1.00000） |
| textenc（未达标） | `breeze/textenc/` | 1.0GB | 604MB embed bin + eoi_embedding；axmodel 待修复 |
| tokenizer | `breeze/tokenizer/tokenizer.json` | 33MB | 与 GEN_LOGIC §2.1 的 26 ids 逐 id 一致 |

---

## 3. 性能实测（fast 模式，默认交付形态）

### 3.1 逐阶段（短句 golden_zh_short，34 token prompt）
| 阶段 | 单帧/单次 | 说明 |
|---|---|---|
| talker prefill L=34（1×128 chunk，28 层） | **398 ms** | 固定成本 |
| talker decode（28 层 + post） | **136.4–138.5 ms/帧** | ≈4.9ms/层；kv_cache_len=512 |
| depth（15 步 × 12 层 + post + CPU 15 heads） | **560–758 ms/帧**（典型 ~560；个别轮次系统抖动到 758） | 37–50ms/步（含 15 个 CPU head ~40-60ms） |
| codec decode（64 帧，1 桶 vocoder） | **510–522 ms** | decode_pre 20 + transformer prefill 23 + post 26 + vocoder 456 |
| codec decode（84 帧，2 桶） | 1249 ms | 长句按 39 帧/桶追加 ~456ms/桶 |
| 模型加载 | talker 8.5s、cp 2.5s、codec 0.6–0.9s | 合计 ~12s |
| embedding 预热（134+63MB 顺序 touch） | 0.7s | 冷态查表 13ms/帧 → 预热后 1.1ms/帧 |
| **端到端** | **短句 5.12s 音频 57.9s → RTF 11.3×**；长句 6.72s 音频 73.4s → RTF 10.9× | 含加载；不含加载则 RTF ≈ 10 |

### 3.2 fast vs quality（hp）两档
| 项 | fast（INT8 depth） | quality（hp FP32 depth） |
|---|---|---|
| depth 单帧 | 560 ms | **1375 ms**（+2.5×） |
| 端到端 RTF | **11–14×** | **22.9×** |
| CMM used / remain | 3471 / 2672 MB | **4394 / 1749 MB** |
| RSS | ~1.0 GB | ~1.1 GB |
| free-run code match vs golden | 0.807 | **0.900**（frame0 15/15） |
| teacher-forced top1 | 96.0% | 95.6%（见 §5.2 解释） |
| 定位 | **默认 demo**（ASR 100%） | 可选 quality 模式（token 对拍/研究） |

### 3.3 系统资源（fast 全链路运行中）
```
RSS 1.0–1.15 GB（其中 CPU heads fp32 126MB、KV host 数组 ~66MB、embed 页缓存可回收）
MemAvail 1.08–1.17 GB（板 1.9GB 总量，M5Stack llm-* 服务仍在跑）
CMM: used 3471MB / remain 2672MB（talker 1.6G + depth 0.58G + codec + 每层 session K/V buffer）
```
`kv_cache_len=512` 的选择收益：KV host 数组 235→59MB、每层 session K/V buffer 235→59MB CMM、每帧 memmove 235→59MB。

---

## 4. 最终 demo（采样模式，INT8 fast）

命令模板：
```bash
LD_LIBRARY_PATH=/soc/lib python3 run_breeze_tts.py --preset breeze \
  --text-embeds <golden_or_ref>.npz --prompt-ids <meta.json> --lmhead-cpu \
  --max-frames 120 --temperature 0.9 --top-k 50 --seed 42 \
  --out-wav /root/breeze-tts/out/demo_XX.wav --dump-tokens /root/breeze-tts/out/demo_XX.npz
```

| # | 文本 | 帧数/时长 | RTF | ASR（paraformer-zh） |
|---|---|---|---|---|
| demo_zh | 你好，欢迎体验 AX650 上的 Breeze TTS 语音合成。 | 64 / 5.12s | 14.1× | **逐字正确**（"语音合成"完整） |
| demo_zh_long | 今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？ | 84 / 6.72s（EOS@84） | 10.9× | **逐字完全正确** |
| demo_en / demo_zh2 | 英文句 / 第二中文句（等待 ref-2038 text_embeds） | — | — | — |
| greedy 对照（仅 token 对拍） | 同上 | 64 / 5.12s | 11.3× | 丢最后一个音节（"语音盒"）→ 已用 codec 隔离实验定位为 depth 贪心分叉，非 codec 问题 |

**EOS 行为**：长句采样模式在 frame 84 正常出 EOS（2051）；hp 模式 frame 56 EOS。贪心对照 64 帧上限未出 EOS。

---

## 5. 精度验收（分阶段）

### 5.1 talker（✅ 优秀）
- frame0 prefill logits top5 与 golden **逐个完全一致**：`[404, 1464, 29, 1221, 279]`（数值 18.625/18.625/18.5/17.0/16.375 vs golden 18.625/18.5/18.5/17.0/16.25）。
- post `output_norm` vs golden `hidden_norm[0]`：**cos = 0.99942**、max|diff| 0.125。
- 结论：talker 全 NPU 路径保真度足够。

### 5.2 depth（⚠️ 当前唯一误差源）
| 指标 | fast INT8 | quality hp |
|---|---|---|
| teacher-forced logits cos | 0.99914 | 0.99927 |
| teacher-forced top1 | 96.0%（144/150） | 95.6%（86/90） |
| free-run code match（前 6-10 帧） | 0.807 | 0.900（frame0 15/15） |
| 单步耗时（12 层） | ~37ms | ~92ms |

**为什么 FP32 版 teacher-forced 也没有提升？**
golden 参考是 PyTorch **bf16** 推理（logits 以 bf16 计算/保存），bf16 在 |logit|≈10 处的量化步长 ≈0.05–0.1；而本板 chain 是 FP32/INT8→FP32 输出。当 golden top1 与 top2 仅差 ~0.05–0.1 时，任何实现差异（bf16 舍入、不同 attention 实现、INT8 量化）都会翻转 argmax。**96% 是 bf16 参考本身带来的不可约差异**，不是板端量化缺陷。佐证：hp（FP32 权重）与 INT8 的 teacher-forced top1 基本持平，但 free-run 因首步更准而提升（0.81→0.90）。
**影响**：贪心 token 对拍会从第 2–4 步开始分叉（音频内容仍正确）；采样模式（默认 demo）不受影响（ASR 100%）。

### 5.3 codec（✅ 复用 AXERA 产物精确成立）
隔离实验：把 golden `tokens.npz` 的 64×16 codes 直接喂板端 codec（`--codes` 模式），与 golden `out.wav` 对比：
- 长度**精确一致**：122880 samples = 5.120s；
- **幅度谱相关 0.99958**、能量包络相关 0.8408、整体 RMS 0.0482 vs 0.0484；
- 最后 0.5s RMS：board/golden = **0.947** → **无尾部截断 bug**；
- 逐采样波形相关 ≈0、最佳 lag -145 samples（-6ms）corr 0.20 → 相位/时基差来自 AXERA vocoder 实现（卷积/分块 padding）vs HF 参考，**不影响内容**；
- ASR：golden codes → 板端 codec 转写 **"…语音合成"（逐字正确）**，与 golden 一致。

### 5.4 ASR 验收总表（paraformer-zh + SenseVoice 交叉验证）
| wav | 内容 | ASR 结果 |
|---|---|---|
| golden out.wav | 短句 | …语音**合成** ✓ |
| codec_golden_short.wav（golden codes→板端 codec） | 短句 | …语音**合成** ✓（=理解：codec 无损） |
| bz_short_greedy.wav | 短句 greedy | …语音**盒**（丢"成"，depth 贪心分叉） |
| **bz_short_sampled.wav** | 短句采样 | **…语音合成 ✓**（demo） |
| **bz_long_sampled.wav** | 长句采样 | **逐字完全正确 ✓**（demo） |
| bz_short_sampled_hp.wav | 短句采样 hp | 待 ASR（见 §3.2） |

---

## 6. 关键发现 / 坑（写进交付文档）

### 6.1 协议类（已在 RUNTIME_SPEC 附录 A/B 固化）
- prefill group1 的 K/V 占位 shape = `[1,1,hidden]`（不是 kv_dim）；group≥2 才是 `[1,128*(g-1),kv_dim]`。
- prefill `indices` 行数因 build 而异（generic=1 行，AXERA qwen3_tts_talker=3 行）；decode mask 当前 token 固定占最后一列（列数=kv_len+1）。已在 tiny 模型上做 prefill-vs-decode 自洽验证（cos=1.000000）。
- decode 输出 logits 是 bf16，采样前转 fp32。

### 6.2 **15 个 depth lm_head axmodel 的输出量化坏了**（必须知道）
- 单次推理 2051 个 logits 只有 **~150 个不同值**（int8 输出 clamp/round），范围 ±5；golden logits 达 ±16 → frame0 step0 argmax 直接错（91 vs 1380）。
- 改用 **CPU fp32 matmul**（`codebooks_head_15x1024x2051.bf16.bin` 预载 fp32 126MB）后：step0 logits cos=**0.99977**、top5 与 golden 完全一致；代价 ~40-60ms/帧。
- 建议（下一版）：lm_head 重编时去掉输出量化/改 U16；修复后可换回 NPU（可省 ~40ms/帧）。

### 6.3 其他
- pyaxengine `del session; gc.collect()` **能释放 CMM**（实测 2 层 122MB→释放回 8MB）→ textenc 可"先用后释放"分阶段加载（`--keep-textenc` 可保持常驻）。
- 每个 per-layer session 常驻一份 K/V 输入 buffer（CMM 占用 ≈ axmodel 体积 ×1.3–1.4），kv_cache_len 直接决定这块开销。
- embedding 表页缓存预热：冷态 13ms/帧 → 预热后 1.1ms/帧（0.7s 一次性），默认开启。
- 板端 `hf-mirror` 会 308 重定向导致 huggingface_hub 失败，直连 huggingface.co 可用；pip/apt 正常。
- 板端 python 3.10 f-string 不支持反斜杠；`np.zeros(bf16)-65536.0` 会升 float32（用 `np.full`）。
- axllm（MiniCPM-OCR demo）占 2.2GB CMM，跑本模型前需 `pkill -x axllm`（恢复 `/root/minicpm-ocr-v2/start_axllm.sh`）。

---

## 7. 已知限制（交付口径）

1. **textenc 未上线**：Pulsar2 对 T5Gemma2 的 AxRoPE FP32 backend bug + INT8 精度不达标；当前 demo 用 **golden/reference `text_embeds` 注入**（`--text-embeds`），板端除 textenc 外全部 NPU 化。textenc axmodel 修好后无需改代码，直接去掉 `--text-embeds` 即可（`TextEncoderRunner` 已实现 Breeze 语义：embed_scale、eoi_embedding、分两段 axmodel 串行、fp32 输入）。
2. **非实时**：fast RTF 11–14×（5s 音频 ≈1 分钟），hp RTF 23×。瓶颈是 depth 15×12 层串行 + talker 28 层权重流（~48% 帧时间）+ pyaxengine 每层 memmove/no-copy 限制。
3. **codec 单次上限 127 帧 ≈ 10.16s 音频**（transformer kv_cache_len=127；decode_pre 桶 325 帧不设限但 transformer 限制先到）→ 长文本需分段合成/拼接。runtime 已自动 cap 并 warning。
4. **talker prefill 容量 640 token**（5 组×128），超长 prompt 需加组数重编。
5. depth teacher-forced 96% 受 bf16 参考限制；贪心 token 对拍会分叉（音频仍正确）；**验收以 ASR/听感为主**。
6. 板端 1.9GB 内存下建议停 M5Stack `llm-*` 服务与 axllm 以获得最佳余量（当前 demo 在它们运行下也通过）。

---

## 8. 复现命令

```bash
# 0) 板端环境（一次性）
pip3 install /root/breeze-tts/pkgs/axengine-0.1.3-py3-none-any.whl --no-deps
pip3 install "ml-dtypes>=0.1.0"

# 1) 产物体检
cd /root/breeze-tts && LD_LIBRARY_PATH=/soc/lib python3 check_artifacts.py --base /root/breeze-tts/breeze

# 2) codec 隔离验证（golden codes -> 板端 codec）
LD_LIBRARY_PATH=/soc/lib python3 run_breeze_tts.py --preset breeze \
  --codes /root/breeze-tts/ref_golden_short/tokens.npz \
  --out-wav /root/breeze-tts/out/codec_golden_short.wav
python3 board_scripts/compare_wav.py ref_golden_short/out.wav out/codec_golden_short.wav

# 3) 全链路（fast/INT8，demo 默认；textenc 未好时用 --text-embeds）
LD_LIBRARY_PATH=/soc/lib python3 run_breeze_tts.py --preset breeze \
  --text-embeds /root/breeze-tts/ref_golden_short/probe.npz \
  --prompt-ids /root/breeze-tts/ref_golden_short/meta.json --lmhead-cpu \
  --max-frames 120 --temperature 0.9 --top-k 50 --seed 42 \
  --out-wav /root/breeze-tts/out/demo_zh.wav --dump-tokens /root/breeze-tts/out/demo_zh.npz

# 4) quality 模式（depth hp）
... --depth-dir /root/breeze-tts/breeze/builds/code-predictor-hp ...

# 5) 贪心 + 探针（token 对拍）
... --greedy --probe-frames 20 --no-codec
python3 compare_tokens.py ref_golden_short/tokens.npz out/demo_greedy.npz
python3 compare_probe.py out/demo_greedy.npz ref_golden_short/probe.npz --tokens ref_golden_short/tokens.npz
```

---

## 9. 脚本/资产清单（Mac `_work/board/` 同步）

| 文件 | 用途 |
|---|---|
| `run_breeze_tts.py` | **主运行时**（presets、通用 shape_group/KV/mask、采样、codec、--text-embeds/--codes/--lmhead-cpu/--probe-frames/--depth-dir）|
| `check_artifacts.py` | 产物体检（层数/post 输出/vocab/kv_dim/embed bin 尺寸/textenc I/O/tokenizer 26 ids）|
| `compare_tokens.py` | golden vs 板端 token 一致率/首分叉帧/prefill logits cos |
| `compare_probe.py` | 逐级探针对拍（text_embeds/hidden_norm/backbone top5/depth logits）|
| `compare_wav.py` | 两段 wav 对比（长度/波形相关/幅度谱相关/包络/尾部 RMS）|
| `integrate_artifacts.sh` | 20.38→板端 tar 流式上板（Mac 不落盘）+ 自动体检 |
| `scripts/probe_tiny.py` | tiny-qwen3/tiny-llama I/O + mask 语义自洽验证 |
| `scripts/probe_talker_breeze.py` / `probe_cp_textenc.py` | 各组件 I/O 探针 |
| `scripts/test_cp_parity.py` | depth 对拍（teacher-forced + free-running）|
| `scripts/test_cp_cpu_norm.py` | CPU fp32 RMSNorm 实验 |
| `scripts/bench_decode.py` | decode 微基准（KV memmove/权重带宽/launch 开销）|
| `scripts/smoke_codec.py` / `test_codec_long.py` / `test_codec_Tsweep.py` | codec 全链路/长音频/桶边界验证 |
| `scripts/test_talker_loop_breeze.py` | Breeze 几何 talker-only 循环 + sum16 表性能 |
| `out/` | demo wav + tokens + 日志；`ref_data/` golden 参考副本 |
| `BOARD_REPORT.md` / `RUNTIME_SPEC.md`（附录 A/B） | 板端环境/I/O 协议权威文档 |

---

## 10. 后续可优化（不在本次交付）

1. **depth lm_head U16 重编**（去掉输出量化）→ 换回 NPU，省 40-60ms/帧。
2. depth INT4 (`-w s4`) / 合并 12 层单 step axmodel：预估 depth 550→360-440ms/帧。
3. vocoder 更大 T 桶（如 t128）→ 减少 456ms/桶的固定成本（120 帧时 3 桶 ≈1.4s）。
4. pyaxengine 无法零拷贝 K/V，每帧 28×8.4MB memmove；用 ax-llm C++ 运行时或自定义 buffer 可省 ~25% talker 时间。
5. textenc axmodel 落地后去掉 `--text-embeds`，实现真正"全 NPU、纯文本输入"。
6. 多句 demo 与长文本分段拼接（codec 127 帧上限）。
