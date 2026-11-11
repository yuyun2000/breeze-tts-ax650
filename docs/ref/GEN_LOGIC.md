# Breeze TTS 2 生成逻辑规格书（板端 runtime 对照用）

> 来源：`_work/breeze-tts-src/`（models/generation_breeze.py 1349 行、models/breeze.py 1898 行、models/logits_process.py、breeze_infer/{runtime,templates}.py、infer.py）+ HF 线上 `config.json` / `generation_config.json`（BreezeBlue/Breeze-TTS-2）。
> 环境口径：transformers **4.57.3**、torch 2.9.1、`attn_implementation="eager"`、`dtype=bfloat16`。
> 行号以本仓库当前 checkout 为准。**本文是板端 Python runtime 的规格书：默认路径 = Voice Design（tts_instruction），cfg_scale=1.0（无 CFG），eager 采样。**
> 写作用域：ref-2038（本文件 Mac `_work/ref/`、20.38 `~/work/breeze-tts/ref/`）。

---

## 0. 一页速查（板端 runtime 必须照抄的常数）

| 项 | 值 | 依据 |
|---|---|---|
| 音频 codebook 数 | **16** | config `num_codebooks` / `audio_num_codebooks` |
| 音频 token 词表 | **2051**（合法 id `0..2050`，但见下行） | config `vocab_size` / `audio_vocab_size` |
| codec 合法 id | **0..2047**（`codebook_size=2048`） | config `codec_config.codebook_size` |
| 保留掩码 id | **2048, 2049, 2050** → logits 置 `-inf` | `generation_breeze.py:133-146` + `logits_process.py:7-25` |
| backbone EOS id | **2051**（= `vocab_size`；lm_head 输出 **2052** 维） | `breeze.py:921-923`、`generation_breeze.py:912-914` |
| codebook pad id | **2050**（EOS 帧整帧填 pad；解码时被截断） | config `codebook_pad_token_id`、`generation_breeze.py:970-975, 1063-1066, 1269-1278` |
| codebook eos id | **0**（仅 voice-clone 的 prompt 音频 EOS 帧用；Voice Design 不用） | config `codebook_eos_token_id`、`breeze.py:1611-1628` |
| 帧率 / 采样率 | **12.5 fps / 24000 Hz**（每帧 1920 采样点） | config `codec_config._frame_rate / sampling_rate` |
| backbone 采样 | temperature **0.9**、top_k **50**、top_p **1.0**（不生效）、do_sample=True、vocab 2052 | `breeze_infer/runtime.py:40-66`、`infer.py:27`（repetition_penalty=1.1） |
| depth decoder 采样 | temperature **0.9**、top_k **50**、top_p **1.0**（不生效）、固定生成 **15** 步 | 同上 + `generation_breeze.py:412-432` |
| 重复惩罚 | **1.1**，只作用于「已生成帧的 codebook0」历史，prompt 不计 | `infer.py:27`、`logits_process.py:36-78`、`generation_breeze.py:453-492` |
| 默认最大帧数 | **750**（`max_new_tokens`；prompt 不计入） | `generation_config.json`、`runtime.py:52` |
| 默认 CFG | **cfg_scale=1.0 → 完全不走 CFG 分支**（单次前向） | `infer.py:24,39`、`api.py` DEFAULT_CFG_SCALE=1.0 |
| 首帧禁 EOS | **没有**（backbone 第 1 帧即可出 EOS 2051） | 无 `min_new_tokens`/EosTokenCriteria（见 §3.3） |
| 文本 prompt 位置 embedding | 文本编码器输出（T5Gemma2→proj 2048），**不是** `embed_text_tokens` | `breeze.py:1567-1578, 1352-1460` |
| prompt 首 token | **`<bos>` id=2**（templates encode→decode→encode 的 net 效果；末无 EOS） | `templates.py:182-205` + 实测（§2.1） |
| tokenizer 特殊 id | `<bos>`2 / `<eos>`1 / `<pad>`0 / `[S0]`262146 / `<ins_bos>`262156 / `<ins_eos>`262157 | `tokenizer.json` added tokens |
| 音频帧 embedding | `Σ_{g=0..15} E_g[cb_g]`（16 组 **求和**，无缩放、无平均） | `breeze.py:798-803` |
| **talker RoPE（实测修正）** | **default RoPE，theta=1e6，无 llama3 scaling**；config 顶层的 `rope_theta=5e5`/llama3 **不生效**（走嵌套 `backbone_config`） | §1、§11（实测 `rope_type=="default"`） |
| **depth RoPE（实测修正）** | **llama3 生效**：theta=5e5, orig=16, factor=32, high=1/128, low=1/512 → 35/64 频率被 /32 | §1、§11（实测 `rope_type=="llama3"`） |
| 跨帧位置 | prompt 占 `0..L-1`，第 k 帧占 `L+k`（RoPE 连续计数） | `breeze.py:861-884`、`generation_breeze.py:615-617` |

---

## 1. 模型组成与 token 空间（板端只看这一层）

**Backbone（talker）**：Qwen3 式 28 层，hidden 2048，16 heads × 128，8 KV heads，inter 6144，q_norm/k_norm。
- ⚠️ **RoPE 实测修正（2026-11-11，影响 repack）**：`config.backbone_model_type="qwen3"` → `BreezeBackboneFactory` 走 `BreezeBackboneAdapter`，**用嵌套 `backbone_config`（`Qwen3ForCausalLM`）构造 Qwen3 层 + `Qwen3RotaryEmbedding`**（`breeze_backbone_factory.py:105-180`）。`backbone_config` 里 **`rope_theta=1000000`、`rope_scaling=null` → 实际是 default RoPE(theta=1e6)**。**顶层 BreezeConfig 的 `rope_theta=5e5` / `llama3(factor32,orig1024,high0.5,low0.125)` 在该路径下完全不生效**（只有 `backbone_model_type="breeze"` 的原生 backbone 才读顶层 config）。
  实测确认：`model.backbone_model.rotary_emb.rope_type == "default"`；`inv_freq[-1]=1.2409e-6`，与 theta=1e6/head_dim=128 的 default 公式吻合（若按顶层 5e5+llama3 应为 7.67e-8）。
  → **repack 的 talker config 应写 `rope_theta=1e6`、`rope_scaling=null`**；pulsar2 `--model_type qwen3` 默认 rope 即精确匹配（消融 b：backbone rope=None 与原版 **token 100% 等价**，见 §11）。
- `lm_head`: `Linear(2048 → vocab_size + 1 = 2052)`（`breeze.py:923`）。**logits[2052] 中 2051 = EOS 类**，`config.vocab_size=2051` 不是 EOS id 本身。
- 音频帧 embedding：`embed_audio_tokens` 是 `Embedding(num_codebooks*vocab_size = 16*2051, 2048)`，按 offset `g*2051` 查表后 **对 16 组求和**：
  `frame_emb[t] = Σ_g E[g*2051 + cb[t][g]]`（`breeze.py:792-803`，`forward` 里 `input_ids + audio_tokens_offsets` 再 `.sum(dim=2)`）。
  → 板端实现：16 个 `[2051,2048]` 表（或 bf16/bin）逐组 gather 后相加。注意是**求和**不是平均，数值幅度约为单组 16 倍。
- 文本 embedding：**有 text_encoder 时不用 `embed_text_tokens`**（`breeze.py:1567-1578`）。Voice Design 的 prompt 全是文本 → 100% 走文本编码器路径。

**Depth decoder（code predictor）**：Llama 式 12 层，hidden 1024，8 heads × 128，**2 KV heads**，inter 8192，无 qk-norm，`max_position_embeddings=33`（一帧内序列长度最多 16）。
- **RoPE（实测）**：`BreezeDepthDecoderModel` 用 `BreezeRotaryEmbedding`，**llama3 scaling 真实生效**：theta=5e5、orig=16、factor=32、high_freq_factor=0.0078125（=1/128）、low_freq_factor=0.001953125（=1/512）；实测 `rope_type=="llama3"`，64 个频率中 **35 个被 /32 缩放**（相对 default 的 sin 角度差 max≈1.48e-2 @ pos 0..15）。**board 必须精确复现**（或按 §11 的静态 cos/sin 方案），否则 token 从第 1 帧分叉。
- `embed_tokens`: `Embedding(16*2051, 2048)`，与 backbone 音频表 **tied**（`breeze.py:913-916, 1102-1107`）→ 存一份即可。
- `inputs_embeds_projector`: `Linear(2048→1024, bias=False)`，对**整条** depth 输入序列（含位置 0 的 backbone hidden）都做投影（`breeze.py:487-489, 568`）。
- `codebooks_head.weight`: `[15, 1024, 2051]` = 15 个**位置相关** head；`forward` 用 `codebook_idxs = cache_position - 1` 选 head（`breeze.py:604-628`）。
- `backbone_hidden_state_projector` = None（因为 audio_embed_size == backbone_hidden_size == 2048，`breeze.py:491-496`）。

**Codec**：`Qwen3TTSTokenizerV2Model`（与 `Qwen/Qwen3-TTS-Tokenizer-12Hz` 字节一致），24 kHz，12.5 fps，16 codebook，decoder-only 即可出波形。

---

## 2. Prompt 构造（Voice Design = tts_instruction）

`breeze_infer/templates.py`：
- `select_template_name`（135-145）：有 `ref_audio_path` → `ref_edit_tata`/`ref_clone_tata`；否则有 instruction → **`tts_instruction`**；否则 `tts_plain`。
- 模板正文（44-51）：
  ```
  f"{speaker_prefix}<ins_bos>{instruction}<ins_eos>{text}"
  speaker_prefix = "[S0]"   # 默认 speaker="S0"；speaker 为空则不加
  ```
- Tokenize 细节（170-228，**已用真实 tokenizer 实测复核**，见 §2.1）：
  - 每个 text segment 先 `tokenizer(seg, add_special_tokens=True)`（Gemma post_processor **会加 BOS**），再 `tokenizer.decode(ids, skip_special_tokens=False)` 得到「rendered」字符串（里面保留字面 `<bos>`）；最后对拼接串 `tokenizer(final_text, add_special_tokens=False, return_tensors="pt")`，此时 `<bos>` 作为 added-token 被重新匹配回 **id 2**。
  - **所以最终 prompt ids 一定以 id=2（`<bos>`）开头**，随后一次是 `[S0]`(262146)、`<ins_bos>`(262156)、instruction tokens、`<ins_eos>`(262157)、text tokens。**不加 EOS**（`add_eos_token=False`）。
  - `text_ids_mask` 全 True（长度 = L，含 BOS），`text_ids_len = [L]`（batch=1；该长度是在 rendered 串上用 `add_special_tokens=False` 数的，恰好等于 final ids 长度）。`input_values` = None（无参考音频）。
  - tokenizer 的 `pad_token_id = 0`（`<pad>`），**注意不要和 codebook pad 2050 混淆**；batch=1 无 padding，实际不出现 0。
- 板端必须用**同一份 tokenizer**（Breeze `tokenizer.json`，33.4 MB；`GemmaTokenizerFast`）或等价纯 Python BPE + 相同的 added-token 表，`[S0]`/`<ins_bos>`/`<ins_eos>` 必须整 token 匹配、且要复现「先加 BOS 再 decode/encode 回 BOS」的 net 效果（= 直接前置 id 2）。
- **特殊 token 有两个来源，勿混用**：
  - 推理 tokenizer 实际 id（`tokenizer.json` added tokens，**板端用这个**）：`<bos>`=**2**、`<eos>`=1、`<pad>`=0、`<|AUDIO|>`=262144、`<|audio_eos|>`=262145、`[S0]`=**262146**…`[S9]`=262155、`<ins_bos>`=**262156**、`<ins_eos>`=**262157**。
  - config `text_encoder_special_tokens_config.initializer_token_ids` 里的 `<ins_bos>:[236820,22768,236813]`、`<ins_eos>:[954,22768,236813]` 是**训练时用于初始化 special-token embedding 的初始化文本 token 序列**（`"<instruction>"` 等），`enabled=false`，**不是推理 id**——不要用它做板端 tokenize。

### 2.1 实测 prompt（Voice Design，供板端逐 token 对齐）
`text="今天天气不错，我们出去走走吧。"`，`instruction="用温柔亲切的女声，语速稍慢。"`，`speaker="S0"`：
`L=26`，ids：
```
[2, 262146, 262156, 237105, 171109, 239148, 238201, 146440, 238389, 236900,
 238384, 238121, 241066, 239364, 236924, 262157, 22601, 122362, 56033, 236900,
 6932, 67038, 238302, 238302, 238702, 236924]
```
（`<bos>` idx0；`[S0]` idx1；`<ins_bos>` idx2；instruction 12 token；`<ins_eos>` idx15；text 10 token。golden 的 `meta.json` 会保存实际 ids，以生成时为准。）

**Prefill 序列（batch=1）**：`input_ids = [1, L]`（纯文本 token）→ 文本编码器 → `inputs_embeds = [1,L,2048]`（**全部位置**都是文本编码器输出）。

---

## 3. 主循环（backbone 逐帧 → depth decoder 15 步 → 下一帧）

核心在 `models/generation_breeze.py`（Breeze 覆写的 `_sample`，494-1136）与 `generate`（1138-1349）。板端等价伪代码：

```
# --- 0) 输入准备（breeze.py:1759-1774, 1567-1578, 1243-1460）---
ids = tokenize("[S0]<ins_bos>{instruction}<ins_eos>{text}")         # [L]
text_h = T5Gemma2_encoder(ids, mask=ones, pos=arange(L))            # [L,1152]（26 层，双向，sliding<512 等价全注意力）
embeds = proj_1152_to_2048(text_h)                                  # [L,2048]
# --- 1) backbone prefill（breeze.py:827-901）---
out = backbone(embeds, cache=None, positions=0..L-1)                # 28 层
hidden = final_norm_hidden(out)                                     # [2048]   （post-norm！见 §3.4）
logits = lm_head(hidden)                                            # [2052]
# --- 2) 逐帧循环（generation_breeze.py:727-1121）---
frames = []
for step in range(max_frames):
    # 2a) backbone 采样 cb0（~ 每帧 1 次 decode 前向；首帧用 prefill 的输出）
    scores = logits.float()                                         # [2052]
    scores = repetition_penalty(scores, history_cb0)                # 只惩罚已生成帧 cb0，penalty=1.1（首帧不惩罚）
    if do_sample:
        if temperature != 1.0: scores = scores / 0.9
        scores = top_k_filter(scores, 50)                           # 先 top-k（对 2052 全 logits）
        scores[2048:2051] = -inf                                    # 再掩码保留 token（注意顺序！见 §3.2）
        probs = softmax(scores)
        cb0 = multinomial(probs, 1)
    else:
        scores[2048:2051] = -inf; cb0 = argmax(scores)
    if cb0 == 2051:                                                 # 2b) EOS 判定：只认 backbone id 2051
        frames.append([2050]*16)                                    #     追加一帧全 pad（2050），然后跳出
        break
    # 2c) depth decoder 15 步（generation_breeze.py:1026-1054 + breeze.py:502-628）
    seq = [PLACEHOLDER, cb0]                                        # 位置 0 占位，位置 1 = cb0
    d_emb = [proj_2048_to_1024(backbone_hidden), proj_2048_to_1024(E0[cb0])]
    for i in range(15):                                             # 固定 15 步，无 EOS
        h = depth_decoder_12_layers(d_emb)                          # hidden 1024，causal，rope llama3(orig 16)
        d_logits = codebooks_head[i](h[-1])                         # 第 i 个位置相关 head → [2051]
        d_logits[2048:2051] = -inf                                  # suppress_tokens = [2048,2049,2050]
        d_logits = d_logits / 0.9                                   # 温度先于 top-k
        d_logits = top_k_filter(d_logits, 50)
        cb_i = multinomial(softmax(d_logits), 1)
        seq.append(cb_i); d_emb.append(proj(E_{i+1}[cb_i]))         # 下一步输入用 **group i+1** 的表
    frames.append([cb0, cb1, …, cb15])
    # 2d) 下一帧 backbone 输入（breeze.py:798-803）
    frame_emb = Σ_{g=0..15} E_g[frames[-1][g]]                       # 16 组求和
    out = backbone(frame_emb, cache, position=L+step)                # 单 token decode
    hidden = final_norm_hidden(out); logits = lm_head(hidden)       # [2052] → 给下一帧
```

### 3.1 帧表示与结束
- 生成序列内部为 3D `[batch, frames, 16]`；**EOS 帧 = 16 个 `2050`（pad）**（`generation_breeze.py:970-975, 984-985, 1056-1066`）。
- 结束条件只有两个（`generation_breeze.py:1078-1086`）：**backbone 输出 2051**，或达到 `max_new_tokens=750`。
- `generation_config.json` 里的 `eos_token_id: [1]` **不参与停止**：`_get_stopping_criteria`（358-374）把除 `MaxLengthCriteria` 外的 criteria（含 `EosTokenCriteria`）全部丢弃。板端不要用 id 1 停止。
- **没有前 N 帧禁 EOS**：backbone generation config 无 `min_new_tokens`；`MinNewTokensLengthLogitsProcessor` 只在 `min_new_tokens>0` 时添加（transformers 4.57.3 `_get_logits_processor`），而 `eos_token_id=[1]` 也不覆盖 2051。EOS 从第 1 帧就可能出现。
- `max_new_tokens` 语义：HF 先把 `max_length = prompt_len + 750`，Breeze 在 2D prompt 场景把它减去 `prompt_len`（`generation_breeze.py:709-716`）→ **实际就是最多 750 帧音频**（prompt 不计）。

### 3.2 每步 logits 处理顺序（**顺序敏感，板端逐条照抄**）
Backbone（`generation_breeze.py:878-909` + transformers 4.57.3 `_get_logits_processor`）：
1. `GeneratedTokenRepetitionPenaltyLogitsProcessor(1.1)`：**只取 `input_ids[...,0]`（历史 cb0）**做惩罚；`input_ids` 为 2D（首步 prompt）时直接返回原 scores（`logits_process.py:41-42`）。负值 `*penalty`、正值 `/penalty`，重复 id 只作用一次。
2. `TemperatureLogitsWarper(0.9)`：`scores /= 0.9`。
3. `TopKLogitsWarper(50)`：**对 2052 维全 logits** 取 top-50，其余置 `-inf`。
4. `TopPLogitsWarper(1.0)`：**不添加**（条件 `top_p < 1.0` 才加）。
5. Breeze 再 `_mask_reserved_codec_logits`：`scores[..., 2048:2051] = -inf`（**在 top-k 之后**！若 2048-2050 挤进 top-50，softmax 只剩 49 个候选）。
6. `softmax(dim=-1)` → `torch.multinomial(probs, num_samples=1)`（`generation_breeze.py:894-907`；`do_sample=False` 时 `argmax`，注意 argmax 前同样做了第 5 步掩码）。

Depth decoder（`breeze.py:1026-1035` 调 `depth_decoder.generate(..., suppress_tokens=[2048,2049,2050])`，标准 HF `_sample`）：
1. `SuppressTokensLogitsProcessor([2048,2049,2050])` → `-inf`（先掩码）
2. `TemperatureLogitsWarper(0.9)`
3. `TopKLogitsWarper(50)`；`top_p=1.0` 不添加
4. `softmax` → `multinomial`
5. 步数由 `min_new_tokens = max_new_tokens = num_codebooks-1 = 15` 强制（`generation_breeze.py:412-432`）；depth 的 `eos_token_id=None`，**没有 EOS 逻辑**，恒定 15 个 token。

> 随机数：两个采样器共用同一个 torch CPU RNG 流；`set_all_seeds(seed)` 在 `prepare_inputs` 之后、generate 之前调用（`infer.py:111-121`）。板端做**贪心**（`do_sample=False`）最容易逐帧对拍；采样模式只做主观质量对照（见 §7）。

### 3.3 为什么「首帧 EOS 判定」要单独说
EOS 帧本身不会进入音频：它在 `frames` 里是整帧 2050，解码前 `is_pad_frame = (codes == 2050).all(dim=-1)` 找到第一个整帧 pad 的位置截断（`generation_breeze.py:1269-1278`）。所以**有效音频帧 = EOS 帧之前的所有帧**。板端流程：出 EOS → 不再解码该帧 → `codes[:eos_idx]` 送 codec。

### 3.4 backbone → depth 的 hidden 是 **post-norm**
`outputs.hidden_states[-1]`（`generation_breeze.py:917`）。Breeze 里 `hidden_states` 若未被逐层收集，则回退成 `(last_hidden_state,)`（`breeze.py:1789-1795`）；`last_hidden_state` 是 `self.norm(...)` 之后的（`breeze.py:897`）。→ 板端 depth decoder 的位置 0 输入 = **backbone 最终 RMSNorm 之后**的 hidden 再做 `input_embeds_projector`(2048→1024)（注意不是 pre-norm）。

### 3.5 Depth decoder 的 16→15 与 offset 细节（复现要点）
- 输入序列 `[placeholder, cb0, cb1, …, cb14]` 共 16 个位置（`generation_breeze.py:995-997`），`cache_position = 0..15`。
- `embed_tokens` 索引 = `token + (cache_position-1)*2051`（clamp 到 ≥0，`breeze.py:550-552`）：
  - 位置 0 的 placeholder 不查表（被 backbone hidden 覆盖）；
  - 位置 1 cb0 → **group 0** 表；位置 2 cb1 → **group 1** 表；… 位置 15 cb14 → group 14 表。
- 位置相关 head：位置 p 的 logits 用 `head[p-1]`；`forward` 砍掉位置 0 的 logits（`slice(1,None)`，`breeze.py:711-718`）→ 15 个 head 依次对应 cb1..cb15。
- 生成完 `sequences`（长度 2+15=17）后去掉位置 0 占位 → 16 维整帧（`generation_breeze.py:1040-1054`）。
- 也就是说：**第 i 个 head（0-based）预测 cb_{i+1}**；输入 cb_i 用**第 i 组** embedding 表。

---

## 4. CFG（分类器无关引导）——什么时候启用

- `generation_breeze.py:537-562`：`cfg_scale = model_kwargs.pop("cfg_scale", 1.0)`；`use_cfg = cfg_scale != 1.0 and negative_prompt_ids is not None`。
- **默认 `cfg_scale=1.0`（infer.py / api.py 的 DEFAULT_CFG_SCALE）→ 完全不做 CFG，单次前向。Voice Design 默认路径就是无 CFG。**
- 单 CFG（`tts_instruction` 且 `cfg_scale != 1.0`，如 fast.json 里的 4.0）：
  - cond = 带 instruction 的 prompt `[S0]<ins_bos>ins<ins_eos>text`（`templates.py:44-51`）；
  - uncond = **去掉 instruction 的纯文本 prompt** `[S0]text`（`templates.py:54-56`）；
  - backbone：`logits = uncond + cfg_scale*(cond - uncond)`（`generation_breeze.py:850-856`）；
  - depth decoder：对 cond/uncond 两条 backbone hidden 各走一遍 depth 采样，再在 logits 上同样融合（`_depth_decoder_generate_with_cfg`，158-252；调用点 1016-1024）。成本约 2×。
- `cfg_scale == 0.0` 特例：直接用 negative prompt 跑单条（`generation_breeze.py:547-562`）。
- Dual CFG（`cfg_scale_ref` + `cfg_scale_ins`，三路 uncond/ref/ins）：只在**参考音频编辑** `ref_edit_tata` 模板启用（`templates.py:99-105, 315-340`）；Voice Design 不会触发。板端第一版可完全不实现。
- 板端建议：**默认 cfg_scale=1.0（不做 CFG）**，先把 greedy 对拍打通；CFG=4.0 作为第二阶段（会显著改变音色/表现力）。

---

## 5. 文本编码器注入位置（板端接口）

- 只在 **prefill** 发生一次：`input_ids[1,L]` →（`breeze.py:1352-1460`）
  1. `text_ids_mask` 选出的文本 token 按 segment 切分（batch=1 只有 1 段，长度 L）；
  2. 逐段跑 T5Gemma2 encoder（`_batched_text_encoder_forward` 1243-1350：padded batch，`pos_ids=arange`，`attention_mask` 1/0；sliding_window=512 ≥ L 时等价双向全注意力）；
  3. `last_hidden_state` → `text_encoder_proj`：**Linear(1152→2048, bias=False)**（`breeze.py:1000-1006`；config `text_encoder_proj_type="linear"`）；
  4. 写回 `inputs_embeds[text_ids_mask]`（1451-1458），其余位置为 0（Voice Design 无音频 prompt，不存在「其余位置」）。
- **不是** `embed_text_tokens`（262158×2048 的表在推理时不参与，可丢弃）。
- `position_ids`：文本段内 `arange(L)`；backbone 解码时 `cache_position` 连续递增（prompt 占 0..L-1，第一帧占 L）。RoPE 走 llama3 scaling，超过 1024 后按 scaled 频率。
- 音频 EOS token（`<|audio_eos|>`）在 prompt 里的 embedding 用「16 组全 `codebook_eos_token_id=0` 的帧 emb」（`breeze.py:1611-1628`）——**仅 voice clone / ref 模式需要，Voice Design 不需要**。

---

## 6. 音频解码（codec）

- 生成 `codes: [T,16]`（int），`T = 有效帧数`（不含 EOS 帧），范围 0..2047。
- `audio_tokenizer.decode({"audio_codes": codes})` → 波形（mono，24 kHz，float32）。（`generation_breeze.py:1261-1341`；Breeze 包内 `audio_tokenizer/` 与 `Qwen/Qwen3-TTS-Tokenizer-12Hz` 字节一致，板端直接用 AXERA 已编译的 speech_tokenizer decoder。）
- 每帧 1920 采样点（24000/12.5）；时长 = T × 80 ms。
- 板端 codec decode 参考 `_work/ax-llm-qwen3tts/src/runner/Qwen3TTSSpeechTokenizer.cpp`（decode_pre_inproj → 8 层 decoder transformer → decode_post）。

---

## 7. 对拍口径建议（golden vs 板端）

1. **token 对拍（主）**：贪心（temperature=0, do_sample=False 等价 argmax）跑同一 prompt、同一 `max_frames` 短样本（如 40-80 帧）；逐帧逐 codebook 计算一致率；允许数值抖动导致的偶发分叉，统计「首次分叉帧号」，之后帧不计入。
2. **logits 对拍（辅）**：第 1 帧 backbone top-5 token + 概率、depth 第 1 步 top-5；cos 相似度 > 0.99。
3. **音频对拍**：两段 wav 统一 24 kHz，MEL/log-MEL L1/相关、时长、能量包络；codec 与采样若一致则波形应高度相关。
4. **可复现性**：参考侧 `set_all_seeds(seed)`+`torch.multinomial`；板端无 torch，贪心路径无随机性，最容易作为第一验收。
5. 采样路径（temperature/top_k）板端可用 numpy RNG 实现，但**不保证**与 torch 逐 token 一致，只作听感对照。

---

## 8. 关键函数/行号索引

| 逻辑 | 文件:行 |
|---|---|
| backbone EOS 常数 + lm_head 2052 | `models/breeze.py:921-923` |
| 音频帧 16 组求和 | `models/breeze.py:792-803` |
| depth 输入 embeds/offset/位置 0 覆盖 | `models/breeze.py:549-568` |
| depth 位置相关 head | `models/breeze.py:604-628, 711-718` |
| depth generate 入口（suppress_tokens） | `generation_breeze.py:1026-1035` |
| 预留 token 掩码 | `generation_breeze.py:133-146` + `logits_process.py:7-25` |
| 重复惩罚（只 cb0 历史） | `logits_process.py:28-78` |
| CFG 判定/公式 | `generation_breeze.py:537-562, 850-876` |
| depth CFG 两/三路 | `generation_breeze.py:158-356` |
| 采样/温度/topk/top-p 顺序 | transformers 4.57.3 `generation/utils.py:1200-1300`（processor 列表）+ Breeze `generation_breeze.py:878-909` |
| EOS 判定 + pad 帧 | `generation_breeze.py:911-914, 970-985, 1078-1086` |
| 停止条件过滤（只留 MaxLength） | `generation_breeze.py:358-374` |
| depth 固定 15 步 | `generation_breeze.py:412-432` |
| 解码截断 EOS 帧 + codec decode | `generation_breeze.py:1246-1341` |
| 默认采样参数 | `breeze_infer/runtime.py:40-66`、`infer.py:24-27` |
| prompt 模板 | `breeze_infer/templates.py:31-56, 135-145, 170-228` |
| 文本编码器→proj→embeds | `models/breeze.py:1243-1460, 1567-1578` |
| 生成入口 / args | `models/generation_breeze.py:1138-1244`、`infer.py:30-140` |

---

## 9. 待确认/风险项（需要 repack / board 侧核实）

1. **talker 输出词表**：Breeze `lm_head` 是 2052（audio 2051 + EOS 1）。若 repack 走 Qwen3-TTS talker（其 `codec_head` 常按 3072/2158），编译产物 logits 维度/排布必须裁回 2052 或做显式映射，否则 EOS/保留 token 位置错位。板端掩码必须落在 **2048..2050**、EOS=**2051**。
2. **depth past_hidden 投影**：Breeze depth 位置 0 输入 = `proj_2048_to_1024(backbone post-norm hidden)`。Qwen3-TTS 的 code_predictor 假定 talker hidden 已是 1024 维 ── repack 时需要决定：(a) 把 2048→1024 投影放在 talker 输出侧（ret_postnorm 后接一个小 linear，或导出独立 ONNX），或 (b) 在 code predictor 输入前做。**这是 talker→code-predictor 的接口关键点。**
3. **codec 输出采样率/帧率**：24 kHz、12.5 fps、每帧 1920 点，需与 AXERA speech_tokenizer 的 upsample rate 对齐（`decode_upsample_rate=1920`）。
4. **eager vs 编译数值**：板端 int8 量化会引入 logits 抖动，贪心对拍大概率前若干帧一致、后续分叉；验收应看「一致帧比例 / 首分叉帧 / 听感」，不要期望 750 帧全等。
5. `infer.py` 里 `MAX_NEW_TOKENS=1500` 是 API 层上限（fast runtime 用），`generation_config` 默认 750；板端以 750 为默认、可按需降低。
6. **参考环境版本差异（实测记录）**：参考 golden 在 **torch 2.15.0 (macOS arm64) + transformers 4.57.3** 上产出（Breeze requirements pin torch 2.9.1；Mac 环境只有 2.15.0）。实测无异常：贪心两次独立进程 tokens/probe **逐 token/逐字节一致**；ASR（SenseVoice-int8）对 golden wav 转写与输入文本**逐字一致**。若板端对拍出现系统性偏移，优先怀疑 torch/transformers 版本，可换 torch 2.9.1 的 x86(有 AVX2) 宿主复跑本 golden 对比。版本写入每个 golden 的 `meta.json`。
7. **tokens-only 路径已验证**：block 掉 `qwen_tts` 导入后 `model.generate(..., output_audio=False)` 仍能生成 tokens（codec 完全不参与）→ 板端可只用 NPU speech_tokenizer 解码；`BreezeForConditionalGeneration` 内 `codec_model`(Mimi, 96.2M/0.19GB) 与 `embed_text_tokens`(536.9M/1.07GB) 在 Voice Design 推理中**均可丢弃**（部署权重 3.483B/6.97GB → 2.850B/5.70GB）。详见 `REPORT.md` §6/§7。

---

## 10. 参考 golden 实测（2026-11-11，供板端量化对拍）

环境：Mac M4 Pro / torch 2.15.0 / transformers 4.57.3 / bf16 / eager / CPU。模型加载 ~0.95 s（mmap），**稳态 ≈0.29 s/帧（RTF≈3.6）**，峰值 RSS 5.53 GB。

固定 prompt（Voice Design，cfg_scale=1.0 无 CFG，seed=42）：
`text=今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？`
`instruction=用温柔亲切的女声，语速稍慢。` → prompt ids **35 个**（以 2 开头，§2.1/REPORT §3 有完整数组）。

| golden | 模式 | 帧数 | EOS | 音频 | 生成耗时 |
|---|---|---|---|---|---|
| `golden_zh` | 采样（0.9/50/1.0，rep 1.1） | 96 | 是 @96 | 7.68 s | 28.4 s |
| `golden_zh_greedy` | 贪心（backbone+depth） | 100 | 否（上限截断） | 8.00 s | 29.2 s |
| `golden_zh_greedy200` | 贪心，max 200 | 111 | **是 @111** | 8.88 s | 32.2 s |

板端对拍数据（Mac `_work/ref/` 与 20.38 `~/work/breeze-tts/ref/` 同步）：
- `golden_zh_greedy/first20_frames.npz`：前 20 帧 tokens、每帧 backbone top5 logits（token+原始 logit）、post-norm hidden（[20,2048] f16 + mean/std/l2/first4）、prompt_ids；
- 同目录 `probe.npz`：`text_embeds`[35,2048]（文本编码器输出，可直接对拍文本编码器 axmodel）、`depth_logits`[F,15,2051]；
- 对拍工具：`compare.py`（用法/字段见 `REPORT.md` §8）。
- 实测一致性：贪心两次独立进程 tokens 100% 一致、probe 逐字节一致、首处差异帧 = 无；贪心 200 帧的前 100 帧与贪心 100 帧完全一致（截断前缀稳定）。

---

*初版：2026-11-11 由 ref-2038 依据源码逐行核对；后续若发现板端接口差异，在文末追加修订记录，不删原文。*

---

## 11. RoPE scaling 消融（2026-11-11，决定板端 rope 实现策略）

**目的**：Pulsar2 通用 `qwen3`/`llama` builder 是否支持 llama3 rope_scaling 尚未验证；本实验给出「若不支持、用 default rope 近似」的质量代价证据，并顺带核实了 §1 的 RoPE 事实。

**方法**：为每个变体建模型目录（**符号链接**原权重，仅改 `config.json`；`BreezeRotaryEmbedding/Qwen3RotaryEmbedding` 在 `__init__` 时读 config，故必须改 config 而非运行后改属性）。贪心、同 prompt/seed：text=`今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？`/instruction=`用温柔亲切的女声，语速稍慢。`/seed=42/S0。脚本：`rope_ablation.py`（`--max-frames 40` 快筛，`111` 做完整句质量对照）。

**变体**：a=原版；b=backbone `rope_scaling=None`；c=depth `rope_scaling=None`；d=两者 None。

### 11.1 结果 A：40 帧快筛（vs 原版 a）

| 变体 | 整帧一致率 | 首分叉帧 | cb0 一致率 | logMEL cos | 波形 r | lag |
|---|---|---|---|---|---|---|
| a 原版 | 1.000 | -1 | 1.000 | 1.000 | 1.000 | 0 |
| b backbone rope=None | **1.000** | **-1** | 1.000 | **1.000** | **1.000** | 0 |
| c depth rope=None | 0.025 | **1** | 0.050 | 0.619 | 0.231 | 11.9 ms |
| d 两者 None | 0.025 | 1 | 0.050 | 0.619 | 0.231 | 11.9 ms |

### 11.2 结果 B：111 帧完整句（贪心 EOS 点附近）

| 变体 | 帧数 | EOS | 整帧一致率 | 首分叉帧 | cb0 一致率 | logMEL cos | 波形 r | ASR 转写（vs 输入） |
|---|---|---|---|---|---|---|---|---|
| a 原版 | 111 | 否（上限） | 1.000 | -1 | 1.000 | 1.000 | 1.000 | `今天天气不错，我们一起出去走走吧。你觉得去哪里比较好呢？` ✅ 逐字 |
| b backbone rope=None | 111 | 否 | **1.000** | **-1** | 1.000 | **1.000** | **1.000** | 同上 ✅ 逐字 |
| c depth rope=None | **105** | **是@105** | 0.010 | 1 | 0.038 | 0.325 | 0.161 | `今天天气不错，我们一起出去走走吧，你觉得去哪里比较好呢？`（**词全对，仅句末标点 ，→。**，字符准确率 96.4%） |
| d 两者 None | 105 | 是@105 | 0.010 | 1 | 0.038 | 0.325 | 0.161 | 同 c |

### 11.3 结论（板端 rope 实现建议）

1. **Talker/backbone：default rope（theta=1e6）就是精确解，不存在近似代价。** 变体 b 与原版逐 token 100%、波形 r=1.0（原因是 §1 修正：原版 backbone 根本没应用顶层 llama3 配置）。→ **repack talker config 写 `rope_theta=1e6, rope_scaling=null`，pulsar2 `qwen3` 默认 rope 直接正确；无需 llama3 支持。**
2. **Depth decoder 的 llama3 rope 必须精确实现**：改成 default 后**第 1 帧 token 就分叉**（整帧一致率 1–2.5%、cb0 一致率 3.8–5%），音频波形明显不同（logMEL 0.32–0.62、波形 r 0.16–0.23、EOS 提前到 105 帧）。原因：depth 只有 12 层 × 15 步，greedy top-1 margin 小，llama3 在 35/64 个频率上的 /32 缩放带来的 sin 角度差（pos 0..15 上 max≈1.5e-2）足以翻转 argmax 并级联。
3. **但可懂度损失很小**：111 帧对照里词序列完全一致、ASR 字符准确率 96.4%（唯一差异是句末标点）。→ 若必须用 default 近似上线，听感/ASR 仍可接受，但**token 级对拍必须改用同近似生成的 golden**（不能拿本目录 llama3 golden 判失败）。
4. **推荐优先级**：
   - (i) 先让 repack 在 depth config 里保留 `rope_scaling` 试 pulsar2 llama builder（观察 build 报错 + `check_level` 的 cos；若支持则最优）；
   - (ii) 不支持时，利用 **depth 每帧固定 15 步、位置恒为 0..15** → 预先算好 llama3-scaled 的 cos/sin（16×64 表），把 depth 导成**静态/逐步展开 ONNX，cos/sin 作为常量或图输入**，彻底绕开 builder 的 rope 支持（这一路径不依赖 builder 的 rope 实现，最稳）；
   - (iii) 兜底才用 default 近似，并用 `ref_generate.py`（depth rope=None 变体目录）重新生成一份 approx golden 专用于 token 对拍。
5. 不要试图用「调 theta」近似 llama3：llama3 是对部分频率做 /32 缩放（35/64），不是全局 theta 缩放，即使对短序列也不是等价替换。
6. 复现命令：`python rope_ablation.py --model ./model-breeze-tts-2 --out-dir rope_ablation --max-frames 40`（长版 `--out-dir rope_ablation_long --max-frames 111`）；变体模型目录在 `<out-dir>/models/`（符号链接，可随时重建）。
