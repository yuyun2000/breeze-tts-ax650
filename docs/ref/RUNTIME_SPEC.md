# Breeze TTS 2 板端运行时规格（给 board-105）

来源：`_work/ax-llm-qwen3tts/src/runner/`（C++ 权威实现）+ HF `Qwen3-TTS-12Hz-0.6B-Base-AX650/infer.py`（Python axengine 协议）+ `_work/breeze-tts-src/models/generation_breeze.py`（Breeze 生成逻辑）。
所有维度按 Breeze 改：`talker hidden = 2048`、`talker prefilla/decodel`、`depth hidden = 1024`、`depth layers = 12`、`num_code_groups = 16`、codebook_size=2048、audio vocab=2051、backbone EOS=2051（logits 2052 维）。

## 1. 总体流程（Voice Design，无参考音频）

```
text ──tokenizer──> token ids [1,T] ──text encoder axmodel──> embeds [1,T,2048]
  ──talker prefill（28 层 per-layer axmodel，chunk 128）──> raw hidden [1,T,2048]
  ──talker post（final norm + codec_head）──> logits [2052] + hidden_norm [2048]
  loop frame:
    cb0 = sample(logits, mask: 2048..2050 = -inf, EOS=2051 允许; temp 0.9/top_k 50/top_p 1.0)
    if cb0 == 2051: break
    depth: 12 层 per-layer axmodel：
        pos0 = hidden_norm @ proj.T (2048→1024，proj = depth_decoder.model.inputs_embeds_projector)
        pos1 = E'[0][cb0]（E' = proj 后的投影表：E'i = E_i @ proj.T，E_i 是 Breeze embed_audio_tokens 第 i 组）
        for step 0..14:
            if step>0: pos = 2+step-1 的 decode → hidden
            hidden_norm_d = depth post（final norm）
            logits_k = lm_head[step] @ hidden_norm_d（2051 维；mask 2048..2050=-inf? 按 Breeze: depth 采样也 suppress reserved 2048..2050，见 generation_breeze）
            cb_{step+1} = sample(logits_k, temp 0.9/top_k 50/top_p 1.0)
    帧 codes = [cb0..cb15]
    next_embed = Σ_{i=0..15} E_i[cb_i]（**未投影**的 16 组表，各 2048 维，bf16）
    ──talker decode_one(next_embed)──> raw hidden ──post──> logits/hidden_norm，进入下一帧
  codes [T,16] ──speech tokenizer decode──> wav 24kHz
```

## 2. talker（llm_build --model_type qwen3_tts_talker 产物）

文件名：`qwen3_tts_talker_p{P}_l{L}_together.axmodel` ×28，`qwen3_tts_talker_post.axmodel`；embedding bin 由重打包导出（talker codec_embedding=未投影 E0 + 额外 15 组未投影表）。
每层 axmodel 的 I/O（shape_group 语义，来自 infer.py）：
- **prefill**（shape_group=1..chunks）：输入 `indices` uint32 [1,P]、`input` bf16 [1,P,2048]、`mask` bf16 [1,P,chunk_end]、`K_cache`/`V_cache` bf16 [1,kv_len,kv_dim]（首 chunk 传 zeros，后续传已写部分 `[:, :start, :]`），输出 `output_N`、`K_cache_out_N`、`V_cache_out_N`（带组号后缀，N=group id）。P=128 每 chunk。
  - mask 构造：`mask[row, col]=0 if col<=row else -65536`，padding 行全 -65536。
- **decode**（shape_group=0）：输入 `K_cache`/`V_cache`（完整 [1,kv_len,kv_dim]）、`indices` uint32 [1,1]=当前长度、`input` bf16 [1,1,2048]、`mask` bf16 [1,kv_len+1]（前 current_len 个 0，最后 1 个 0，其余 -65536），输出 `output`、`K_cache_out`、`V_cache_out`。
- **post**：输入 `input` bf16 [1,1,2048]（或用 [1,1] 按 build 实际 shape），输出 `output` logits（fp32，2052 维）和 `output_norm`（bf16，2048）——需要 build 时 `--ret_postnorm`。
- KV cache: kv_dim = num_kv_heads*head_dim = 8*128 = 1024；kv_len 由 build `--kv_cache_len`（建议 2048）。

## 3. depth decoder / code_predictor（llm_build --model_type qwen3_tts_talker_code_predictor 产物）

文件名 `qwen3_tts_talker_code_predictor_p{P}_l{L}_together.axmodel` ×12，`_post.axmodel`，`code_predictor_lm_head_0..14.axmodel`，codec_embedding bin ×15（**投影后 1024 维**）。
- **prefill**（shape_group=1，P=64，实际只用前 2 个位置）：输入 `input` bf16 [1,P,1024]（前两位 = [hidden_norm@proj.T, E'0[cb0]]，其余 0）、`indices` uint32 [1,P]（0,1,其余 0）、`mask` bf16 [1,P,?]（2×2 可见，其余 -65536；参考 C++：mask[0]=0, mask[cols]=0, mask[cols+1]=0）、K/V zeros。
- **decode**（shape_group=0）：同 talker decode，hidden 1024。
- **post**：输入 bf16 [1,1,1024]，输出 `output_norm`（fp32 或 bf16，1024）。
- **lm_head[k]**：输入 fp32 [1,1,1024]，输出 fp32 `output`（2051 维）。
- 15 步循环见 §1；所有采样在 CPU。

## 4. speech tokenizer decode（AXERA 现成 Qwen3-TTS axmodel，权重与 Breeze 字节一致）

目录（AXERA HF 0.6B-AX650 包）：
- `speech_tokenizer/speech_tokenizer_decode_pre_inproj.axmodel`：输入 `codes` int32 [16, code_len_padded]（**q-major 排布**：`feed[q*L + t] = codes[t][q]`；L=decode_pre 固定长度），输出 `hidden_512` fp32 [L, 512]（取前 code_len 行）。
- `speech_tokenizer/decoder/qwen3_tts_tokenizer_12hz_p{P}_l{L}_together.axmodel` ×8 + `_post.axmodel`：llm_build 格式；输入 bf16 [1,P,512]，prefill 分块（P=64），解码同 talker 协议（KV cache / mask / indices）；post 输出 1024 维。
  - 注意：这是纯非因果还是因果？按 C++ 用法是 prefill 一次性输入整段（chunked），输出 transformed [code_len,1024]。K/V cache 长度 = vs cache_len 参数。
- `speech_tokenizer/speech_tokenizer_decode_post_t{T}.axmodel`：输入 `hidden` fp32 [T,1024]（T=64 为桶），输出 `wav` fp32 [T*1920] 采样点（24k，即 80ms/帧）。
- C++ 逻辑：code_len = codes 中首个 `codes[t][0] > 0` 之前的帧数（Breeze 用 0 作为 eos，实际以生成端截断为准）；decode_post 分块 64 帧。
- 输出 wav float32 [-1,1]，24kHz，16-bit PCM 保存。

## 5. 帧率/容量
- 12.5 帧/秒（1920 samples @24k）。kv_cache_len 2048 帧 ≈ 163s 音频；一次性合成 demo 用 max_new_tokens ≤ 512 帧（≈41s）足够。
- 每个 codebook token 采样后若为 reserved（2048..2050）需屏蔽；backbone EOS=2051；前 2 帧禁 EOS？

## 6. 验证顺序
1. 文本编码器 axmodel 输出 vs 20.38 PyTorch（cos ≥0.99）。
2. talker prefill 输出 hidden vs PyTorch（重打包对拍）；post logits top-k 一致。
3. 贪心前 3 帧 codebook 与 20.38 原始模型/golden tokens 对拍。
4. 整句音频：听感 + `_work/ref/compare.py` MEL 距离。

---

## 附录 A（board-105, 2026-11-11 板端实测修正，优先级高于上文）

以下全部来自板端 pyaxengine 实测（AX650C / AX Engine 2.12.0s / pyaxengine 0.1.3）。完整数据见 `_work/board/BOARD_REPORT.md`。

### A.1 talker（28 层 + post）
- shape group：**0=decode，1..5=prefill chunk**（build: prefill_len=128, last_kv_cache_len=128,256,384,512 → 总容量 640 token）。
- 每层 I/O（H = talker hidden；0.6B=1024，1.7B/Breeze=2048；kv_dim=1024）：
  - g0 decode：`K_cache` `[1,kv_len,1024]`、`V_cache` 同、`indices` `[1,1]` uint32、`input` `[1,1,H]`、`mask` `[1,1,kv_len+1]`；输出 `K_cache_out`/`V_cache_out` `[1,1,1024]`、`output` `[1,1,H]`，全部 bf16。
  - g1 prefill：`K_cache`/`V_cache` = **`[1,1,H]` 零占位**（注意是 H 不是 kv_dim！）、`indices` `[3,128]` uint32（3 行相同=绝对位置 0..valid-1，padding 填 1）、`input` `[1,128,H]`、`mask` `[1,128,128]`。
  - g2..g5 prefill：`K_cache`/`V_cache` = `[1,128*(g-1),1024]`（前 g-1 个 chunk 的累计 KV）、`mask` `[1,128,128*g]`；outputs 均为 `K_cache_out`/`V_cache_out` `[1,128,1024]`、`output` `[1,128,H]`。
- prefill mask 规则：列数 = 128*g；`mask[r][c]=0 iff c <= 128*(g-1)+r`（全局因果），否则 `-65536.0`（bf16）。
- decode mask 规则：行数不参与（固定 `[1,1,kv_len+1]`）；列 c：`c < cur` → 0；**c == kv_len（最后一列）→ 0（当前 token 固定占最后一列）**；否则 -65536。当前 token 的 KV 输出由 host 写到 cache 第 `cur` 行。
- post：`input [1,1,H]` bf16 → 输出 **`output`** logits **bf16** `[1,1,vocab]` + **`output_norm`** bf16 `[1,1,H]`。**必须 `--ret_postnorm`**（0.6B Base 有；1.7B VoiceDesign 包没有 output_norm，别拿它当参考）。
- 不含 embedding；时间 0.7–0.9 ms/帧；1.7B 几何 decode 6.69 ms/层（28 层=187.3 ms/帧）。

### A.2 code-predictor / depth（5 层 @0.6B；Breeze 为 12 层）
- group 0=decode，group 1=prefill（P=64，实际只用前 2 位）：
  - g0：`K_cache` `[1,128,kv_dim]`、`V_cache` 同、`indices` `[1,1]` uint32、`input` `[1,1,cp_hidden]`、`mask` `[1,1,kv_len+1]`；输出 `K_cache_out`/`V_cache_out` `[1,1,kv_dim]`、`output` `[1,1,cp_hidden]`。
  - g1：`K_cache`/`V_cache` `[1,1,cp_hidden]` 占位、`indices` **`[1,64]`（1 行！与 talker 的 3 行不同）**、`input` `[1,64,cp_hidden]`、`mask` `[1,64,64]`（2×2 可见），其余 pad 0。
- 0.6B 实测 `kv_dim=1024`（= Qwen3-TTS cp 的 num_kv_heads 8×128），`cp_hidden=1024`。**Breeze depth 是 2 kv heads×128=256，build 后必须重读 shape**。
- post：`input [1,1,cp_hidden]` bf16 → **只有 `output_norm`** `[1,1,cp_hidden]` bf16（没有 logits）。
- lm_head[k]：`input` **fp32** `[1,1,cp_hidden]` → `output` **fp32** `[1,1,codebook_size]`（0.6B=2048；Breeze=2051）。15 个 → 6.8 ms/帧。

### A.3 speech tokenizer decode（AXERA 0.6B 包，Breeze audio_tokenizer 字节一致）
- `decode_pre_inproj`：`codes` int32 `[1,16,325]`（**带 batch 维，q-major**：`feed[0,q,t]=codes[t][q]`）→ `hidden_512` fp32 `[1,325,512]`，取前 T 行。固定 20.4 ms/句。
- codec transformer 8 层（H=512，kv_dim=1024，**kv_cache_len=127**）：
  - g1 prefill：K/V `[1,1,512]` 占位、`indices` `[1,64]`=0..63、`input` `[1,64,512]`、`mask` `[1,64,64]`；输出 K/V `[1,64,1024]`、output `[1,64,512]`。
  - g0 decode：K/V `[1,127,1024]`、`indices` `[1,1]`=pos、`input` `[1,1,512]`、`mask` `[1,1,128]`；**滑窗 72**：可见 col ∈ [max(0,pos-71), pos-1] **外加最后一列（127）**，其余 -65536。
- post：`input [1,1,512]` bf16 → `output [1,1,1024]` bf16（每帧一次）。
- decode_post_t64：`hidden` fp32 `[1,64,1024]` → `wav` fp32 `[1,1,122325]`（**不是 64×1920=122880**）。分块：left_context=25，capacity=64-context 帧；取输出 `[context*1920 : context*1920+(end-start)*1920]`；最终 resize 到 `T*1920`。单次 456 ms（t64 桶）。

### A.4 其他实测约束
- 板端 **AX Engine 2.12.0s 原生支持 bf16**，不需要 SDK≥2.18。
- bf16 numpy 必须 `ml_dtypes.bfloat16`；`np.zeros(dtype=bf16) - 65536.0` 会升 float32，用 `np.full(..., -65536.0, dtype=bf16)`。
- `run(None, feed, shape_group=g)` 返回 list，顺序 = `get_outputs(g)` 顺序；feed 的 shape/dtype 必须严格一致（pyaxengine 先断言再拷贝）。
- 性能：talker 1.7B 几何 decode 187.3 ms/帧；CMM：talker28+post 1950MB、cp 5 层+15 head 76MB、codec decode 全套 158MB；停 axllm 后 remain 5152MB。
- 建议 Breeze talker build 用 `--kv_cache_len 512`（KV host 235MB→59MB）；若保持 2048 也能跑。

---

## 附录 B（tiny 单测模型实测，2026-11-11 18:35，board-105）

用 20.38 的 tiny 单测模型在板上 probe 真实 I/O（`probe_tiny.py`），并做 mask 语义自洽验证（prefill(P) vs prefill(k)+decode(k)）。

**tiny-qwen3（talker 1 层，通用 Qwen3 build）**：shape_count=2
- g0 decode：`K_cache`/`V_cache` `[1,128,1024]`、`indices` `[1,1]` uint32、`input` `[1,1,2048]`、`mask` `[1,1,129]`；out `K_cache_out`/`V_cache_out` `[1,1,1024]`、`output` `[1,1,2048]`（全 bf16）
- g1 prefill：`K_cache`/`V_cache` `[1,1,**2048**]`（占位=hidden，不是 kv_dim）、`indices` **`[1,64]`（1 行）**、`input` `[1,64,2048]`、`mask` `[1,64,64]`；out `[1,64,1024]`/`[1,64,1024]`/`[1,64,2048]`
- post：`input [1,1,2048]` → **`output [1,1,2052]` + `output_norm [1,1,2048]`**（Breeze 目标形状，ret_postnorm ✓）

**tiny-llama（depth 2 层，通用 Llama build）**：shape_count=2，两层 I/O 相同
- g0 decode：K/V `[1,32,**256**]`（kv_dim=256 ✓ Breeze depth 2 kv heads×128 被接受）、`indices [1,1]`、`input [1,1,1024]`、`mask [1,1,33]`（=kv_len+1）
- g1 prefill：K/V `[1,1,1024]`、`indices [1,16]`（1 行）、`input [1,16,1024]`、`mask [1,16,16]`
- post：`input [1,1,1024]` → `output [1,1,2051]` + `output_norm [1,1,1024]`（generic llama 的 post 融了 lm_head；真正 cp build 若用 `qwen3_tts_talker_code_predictor` 应只有 output_norm + 15 个独立 head，集成时以实测为准）

**mask 语义（决定性验证）**：prefill(P) 的 `hidden[k]` vs 「prefill(k) + decode 第 k token」的 `hidden[k]`：
- convA（cols<k →0，**最后一列=kv_len** →0）：cos = **1.000000**（tiny-qwen3 和 tiny-llama 都是）
- convB（cols≤k →0）：cos = 0.998398 / 0.999872
→ **AXERA "当前 token 占 mask 最后一列（列数=kv_len+1）" 语义对 Breeze 的 talker/depth 同样成立**；`indices` 行数必须以 axmodel 实际 shape 为准（generic=1 行，AXERA qwen3_tts_talker=3 行，我们的 runtime 动态适配）。
