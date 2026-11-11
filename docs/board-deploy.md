# 板端部署（AX650）

> 状态：进行中（2026-11-11 夜）；数字与最终命令以 `reports/board.md` 为准。

## 1. 板端环境（AX650C 实测）

```
Linux aarch64, Ubuntu 22.04, 8 核, 1.9 GB 系统内存, 2 GB swap
AX Engine 2.12.0s / MC50 / triple core, CMM 6144 MB（停掉其它 demo 后 remain ~5.1 GB）
libax_engine.so 在 /soc/lib（运行时 LD_LIBRARY_PATH=/soc/lib）
```

依赖（无需 PyTorch/transformers）：

```bash
pip3 install axengine-0.1.3-py3-none-any.whl --no-deps   # AXERA pyaxengine，py3-none-any
pip3 install "ml-dtypes>=0.1.0" tokenizers numpy          # bf16 张量、tokenizer
```

模型包（HuggingFace `yunyu1258/breeze-tts-2-ax650`）解压到 `/root/breeze-tts/breeze/`。

## 2. 运行

```bash
cd /root/breeze-tts/breeze
export LD_LIBRARY_PATH=/soc/lib
python3 runtime/run_breeze_tts.py \
  --text "你好，欢迎体验 AX650 上的 Breeze TTS 语音合成。" \
  --instruction "一位温柔自信的年轻女性，声音清晰，语气亲切。" \
  --out out.wav --dump-tokens tokens.npz
```

输出 24 kHz 单声道 wav；`--dump-tokens` 保存 `[T,16]` codebook tokens 供与 PyTorch 参考对拍。

## 3. 运行时结构（纯 numpy + pyaxengine）

| 阶段 | 输入 → 输出 | 说明 |
|---|---|---|
| tokenizer | text → prompt ids（34–40 token） | `tokenizers` + Breeze tokenizer.json；模板 `[S0]<ins_bos>{instruction}<ins_eos>{text}`，BOS=2 |
| text encoder | embeddings [1,128,1152] → [1,128,2048] | CPU 查表（bf16 LUT 604 MB mmap + eoi 替换 + scale），2 段 INT8 axmodel 串联 |
| talker prefill | embeddings → 28 层 axmodel（chunk 128）→ post → logits(2052)+hidden_norm(2048) | 通用 `llm_build --model_type qwen3` |
| talker decode | 帧 embedding（16 组求和）→ 28 层 → post | 每帧 1 次；KV cache 512 帧（host numpy，每层 K/V [1,512,1024] bf16） |
| depth decoder | `[proj(hidden_norm)]` + 15 步 × 12 层（llm_build --model_type llama）→ post(norm) → 15 个 lm_head | 每帧 15 步；KV cache 32，kv_dim 256 |
| codec decode | codes [16,T] → decode_pre → 8 层 transformer(滑窗 72) → post → decode_post(t64 分块) | 复用 AXERA Qwen3-TTS speech_tokenizer axmodel |

采样（与官方默认一致）：backbone temperature 0.9 / top_k 50 / 后掩码 2048–2050；depth 同步 15 步；
EOS 只有 backbone 的 id **2051**（`vocab_size` 的额外类别）；EOS 帧整帧 pad(2050) 后截断。
`cfg_scale=1.0`（Voice Design 默认不做 CFG）。

## 4. 内存与性能

- CMM：talker ~1.4–1.9 GB + depth ~0.4 GB + codec ~0.16 GB + textenc ~0.7 GB，实测待填。
- host RAM：python RSS ~0.9–1.1 GB（模型文件 mmap 页 + KV cache + embedding 表页缓存）；
  建议启动时顺序预热 16 组音频 embedding 表（134 MB，冷态 13 ms/帧 → 热态 1.1 ms/帧）。
- 性能：talker decode ~0.2 s/帧（28 层 INT8），depth 15 步 × 12 层 ~0.55–0.9 s/帧，
  codec ~0.02 s/帧 → **RTF ≈ 10–15×**（5 s 音频约 1–1.5 分钟）。非实时，可用作离线合成 demo。

## 5. 验收方法

1. 文本编码器输出 vs PyTorch 参考（golden `text_embeds`）cosine；
2. 贪心前 20 帧 codebook 与 golden tokens 逐 token 一致率 / 首分叉帧；
3. 整句 wav 由 ASR（sensevoice/paraformer）转写，与输入文本比对；
4. wav log-MEL cosine / 波形相关系数（`runtime/compare.py`）。

golden 由 `docs/ref/` 里的 `ref_generate.py`（Mac CPU, torch 2.15/transformers 4.57.3）生成。
