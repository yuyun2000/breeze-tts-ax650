# Breeze TTS 2 → AX650 重打包与编译报告（repack-2038）

> 目标：把 Breeze TTS 2（3.45B、bf16 6.97GB）的 **talker(backbone)** 与 **depth decoder(code predictor)**
> 重打包成 pulsar2 通用 LLM 架构（Qwen3ForCausalLM / LlamaForCausalLM）目录，用 `pulsar2 llm_build`
> 编译成 AX650 逐层 axmodel，并导出板端 runtime 所需的 embedding / lm_head 产物。
> 结论：两条路径均**精确等价**（CPU 逐层 cos = 1.0），无需 mRoPE / qk-norm 近似，无需自定义 ONNX fallback。

日期：2026-11-11（夜）。机器：Mac（重打包 + CPU 对拍，`_work/venv-ref`：torch 2.15 / transformers 4.57.3）
与 `192.168.20.38`（Pulsar2 8.0 编译）。

---

## 1. 结论速查

| 项 | talker / backbone | depth decoder / code predictor |
|---|---|---|
| 真实架构 | **Qwen3**（`backbone_model_type="qwen3"` → `BreezeBackboneAdapter`，层与 RoPE 都来自嵌套 `backbone_config`）| **Llama 式 12 层**（Breeze 原生 `depth_decoder`）|
| hidden / layers | 2048 / 28 | 1024 / 12 |
| heads / kv / head_dim | 16 / 8 / 128 | 8 / 2 / 128 |
| intermediate | 6144 | 8192 |
| RMSNorm eps | **1e-6**（来自 backbone_config）| 1e-5 |
| RoPE | **default RoPE，theta=1e6，无 rope_scaling** | **theta=5e5 + llama3 缩放**（factor 32，orig 16，high 0.0078125，low 0.001953125）|
| qk-norm | 有（q_norm/k_norm，head_dim 128，rope 之前）| 无 |
| pulsar2 model_type | `qwen3` | `llama` |
| 编译产物 | `qwen3_p128_l{i}_together.axmodel` ×28 + `qwen3_post.axmodel` | `llama_p16_l{i}_together.axmodel` ×12 + `llama_post.axmodel` |
| CPU 对拍（vs 原始 Breeze）| 逐层 hidden cos **1.00000000**；final hidden 1.0000006；lm_head logits 0.9999998 | 16 步 hidden 1.0000010；15 head logits 1.0000007 |

关键事实（与 `_work/BREEZE_PLAN.md` 原 §3 的方案不同，已实测修正）：

1. **顶层 BreezeConfig 的 `rope_theta=5e5 + llama3` 是死配置**：`backbone_model_type="qwen3"` 时
   backbone 由 `BreezeBackboneFactory._create_qwen3_layers(AutoConfig.for_model(**backbone_config))`
   构建，读的是嵌套 `backbone_config`：`rope_theta=1000000, rope_scaling=null, rms_norm_eps=1e-6`。
   实测 `model.backbone_model.rotary_emb.rope_type=="default"`、`inv_freq[1]=0.80584`（=1e6^(-2/128)）。
   若误用 llama3 配置，talker 会静默降质（ref-2038 消融：depth rope 改 None 第 1 帧就分叉）。
2. **`pulsar2 llm_build --model_type qwen3_tts_talker` 不能用于 Breeze**：其 CfgLoader 把
   `rs_rope_type` 硬编码为 `mrope` / `rs_mrope_section=[24,20,20]` / `interleaved_mrope=True`，
   **忽略 config 的 rope_scaling**（已用 patch 注入打印 cfg 验证），而 Breeze backbone 是普通 1D RoPE。
   同时 `qwen3_tts_talker_code_predictor` 的 builder 强制 qk-norm，与 Breeze depth（无 qk-norm）不等价。
3. **通用 builder 足够且精确**：
   - `--model_type qwen3`：`llm_build` 把 config.json 中与 `llm_utils.Config` 同名的字段
     （含 `use_qk_norm`/`qk_norm_after_rope`/`rs_*`）直接映射进 cfg；qwen3 分支用
     `llama_test.build_layer`（qk_norm 在 rope 之前，正合 Qwen3 语义）。
   - `--model_type llama`：`llm_build` 会从 config.json 解析 `rope_scaling`（llama3 被官方支持），
     实测 Config 打印 `rs_factor=32.0, rs_high_freq_factor=0.0078125,
     rs_low_freq_factor=0.001953125, rs_original_max_position_embeddings=16,
     rs_rope_type='llama3'`。
   - 二者 per-layer 输入都是 **embeddings**（bf16 `[1,N,hidden]`），不是 token id，正合 TTS 需求。
4. **depth 的 `inputs_embeds_projector`(2048→1024) 必须在 runtime 侧做**：Breeze depth 对
   **整条**输入序列（含位置 0 的 backbone hidden 与全部 codebook embedding）都做投影，因此
   重打包时把投影折进 embedding 表（`E'_i = E_i @ W^T`），并把 backbone hidden 的投影留给 runtime
   （矩阵很小，numpy 一次 2048×1024 matmul）。pulsar2 的 llama 层不接受 2048 维输入，这是唯一
   "额外的一小步"。

---

## 2. 键映射（Breeze checkpoint → HF 格式目录）

Breeze checkpoint：`model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors`
（总 6.966 GB，1115 个 tensor）。重打包脚本：`repack_breeze.py`（本目录）。

### 2.1 talker → Qwen3ForCausalLM（28 层）

| 目标 key（Qwen3 格式）| 源 key（Breeze）| shape |
|---|---|---|
| `model.layers.{i}.input_layernorm.weight` | `backbone_model.layers.{i}.input_layernorm.weight` | [2048] |
| `model.layers.{i}.post_attention_layernorm.weight` | 同源 | [2048] |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | 同源 | q/o [2048,2048]；k/v [1024,2048] |
| `model.layers.{i}.self_attn.q_norm.weight` / `k_norm.weight` | 同源 | [128] |
| `model.layers.{i}.mlp.{gate,up}_proj.weight` | 同源 | [6144,2048] |
| `model.layers.{i}.mlp.down_proj.weight` | 同源 | [2048,6144] |
| `model.norm.weight` | `backbone_model.norm.weight` | [2048] |
| `lm_head.weight` | `lm_head.weight` | **[2052,2048]**（2051 音频码 + 1 个 backbone EOS）|
| `model.embed_tokens.weight` | 无（dummy zeros）| [2052,2048]（builder 只抽 embed bin，层图不查表）|

talker `config.json` 要点：
`model_type=qwen3, hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
num_key_value_heads=8, head_dim=128, intermediate_size=6144, vocab_size=2052,
rms_norm_eps=1e-6, rope_theta=1000000, rope_scaling=null, max_position_embeddings=2048,
use_qk_norm=true, qk_norm_after_rope=false, layer_types=["full_attention"]*28,
tie_word_embeddings=false`。

### 2.2 depth decoder → LlamaForCausalLM（12 层，无 qk-norm）

| 目标 key | 源 key | shape |
|---|---|---|
| `model.layers.{i}.input_layernorm.weight` | `depth_decoder.model.layers.{i}.input_layernorm.weight` | [1024] |
| `model.layers.{i}.post_attention_layernorm.weight` | 同源 | [1024] |
| `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | 同源 | q/o [1024,1024]；k/v [256,1024] |
| `model.layers.{i}.mlp.{gate,up}_proj.weight` | 同源 | [8192,1024] |
| `model.layers.{i}.mlp.down_proj.weight` | 同源 | [1024,8192] |
| `model.norm.weight` | `depth_decoder.model.norm.weight` | [1024] |
| `lm_head.weight` | 无（dummy zeros）| [2051,1024]（post 里会被建但不用；真实 15 个位置相关 head 单独编译）|
| `model.embed_tokens.weight` | 无（dummy zeros）| [2051,1024] |

depth `config.json` 要点：`model_type=llama, hidden_size=1024, num_hidden_layers=12,
num_attention_heads=8, num_key_value_heads=2, head_dim=128, intermediate_size=8192,
vocab_size=2051, rms_norm_eps=1e-5, rope_theta=500000,
rope_scaling={llama3, factor 32, original_max_position_embeddings 16, high 0.0078125, low 0.001953125},
max_position_embeddings=128`（一帧内位置只有 0..15，prefill 16 足够）。

---

## 3. 板端 runtime 产物（`~/work/breeze-tts/embeds/`）

| 文件 | shape / dtype | 用途 |
|---|---|---|
| `audio_embed_16x2051x2048.bfloat16.bin` | [16,2051,2048] bf16，~129MB | 16 组音频 codebook embedding。talker 每帧输入 = `Σ_g E_g[cb_g]`（**求和，不平均**）；depth 的 codebook 输入由下一项给出。 |
| `proj_audio_embed_16x2051x1024.bfloat16.bin` | [16,2051,1024] bf16，~65MB | `E'_i = E_i @ W^T`（W=`depth_decoder.model.inputs_embeds_projector.weight`）。i=0 用于 depth first_code_embed，i=1..14 依次用于后续 14 步（E'_15 不会被再次输入）。 |
| `depth_proj_2048x1024.bfloat16.bin` | [1024,2048] bf16，~4MB | runtime：`past_hidden = h(2048) @ W.T` → [1,1,1024]，作为 depth 位置 0 输入。 |
| `codebooks_head_15x1024x2051.bfloat16.bin` | [15,1024,2051] bf16，~63MB | 原始 15 个位置相关 head（参考/fallback，CPU 直接算 logits 用）。 |
| `../../lm_head_onnx/code_predictor_lm_head_{0..14}.onnx`（源：repack-lead/output/lm_head_onnx）| 输入 [1,1,1024] fp32，MatMul 权重 [1024,2051]（即 `codebooks_head[i].T`），输出 [1,1,2051] | 编译成 `code_predictor_lm_head_{i}.axmodel`（见 §4.3）。 |

> 注：`embed_audio_tokens` 在 checkpoint 里只存一份（`depth_decoder.model.embed_tokens.weight`，
> 与 backbone tied），`backbone_model` 侧没有该 key；文本侧 `embed_text_tokens`(262158×2048)
> 与 `codec_model` 在 Voice Design 部署集里可丢弃。

---

## 4. 编译命令与产物

环境：`ssh m5stack@192.168.20.38`；`export PATH=$HOME/.cache/magnetar/pulsar2/8.0/bin:$PATH`。

### 4.1 talker（28 层，qwen3）

```bash
pulsar2 llm_build --model_type qwen3 \
  --input_path  ~/work/breeze-tts/repack-lead/output/qwen3-talker \
  --output_path ~/work/breeze-tts/builds/talker \
  --hidden_state_type bf16 --kv_cache_len 512 --prefill_len 128 \
  --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512 \
  --ret_postnorm --chip AX650 --parallel 10
```

- 产物：`qwen3_p128_l{0..27}_together.axmodel` ×28 + `qwen3_post.axmodel`。
- `--ret_postnorm` → post 同时输出 logits `[1,1,2052]`（codebook0 + EOS）与 `output_norm [1,1,2048]`
  （供 code predictor 当 first hidden，等于 Breeze `backbone_last_hidden_state`）。
- 单层体积/总耗时见 §5（build.log 记录）。
- 若不需要 >128 token 的 prompt 分块 prefill，可去掉 4 个 `--last_kv_cache_len`（单 shape group，
  编译快 ~5 倍，但 prompt 上限 128 token）。

### 4.2 depth decoder（12 层，llama）

```bash
pulsar2 llm_build --model_type llama \
  --input_path  ~/work/breeze-tts/repack-lead/output/llama-depth \
  --output_path ~/work/breeze-tts/builds/code-predictor \
  --hidden_state_type bf16 --kv_cache_len 32 --prefill_len 16 \
  --ret_postnorm --chip AX650 --parallel 6
```

- 产物：`llama_p16_l{0..11}_together.axmodel` ×12，**每个 30,674,769 B** + `llama_post.axmodel` 74,483 B。
- check SUCCESS；kv_cache_len=32 覆盖一帧最大序列 17（1 个 past hidden + 16 个 codebook）。
- Config 打印确认 llama3 rope 精确生效（见 §1.3）。
- `--ret_postnorm` 使 post 输出 `output_norm [1,1,1024]`（每步 lm_head 的输入）。

### 4.3 15 个位置相关 lm_head（pulsar2 build）

```bash
bash ~/work/breeze-tts/repack/build_lm_head.sh   # 15 个 ONNX，xargs -P 2
# 内部对每个 i：
pulsar2 build --config lm_head_$i.json   # ONNX: code_predictor_lm_head_$i.onnx
# 产物 ~/work/breeze-tts/builds/code-predictor/lmhead_build_$i/code_predictor_lm_head_$i.axmodel
```

- 量化：INT8 MinMax（calibration = 64 组 N(0,1) 的 `[1,1,1024]` fp32，打包成 `calib/input.tar.gz`，
  脚本 `make_lmhead_calib.py`），`compiler.check=2` 通过（日志 "check npu graph [subgraph_npu_0_b1]
  [output], (1, 1, 2051) ... successfully"）。
- 每个 **2.2MB**，共 15 个，已按约定命名拷平到
  `~/work/breeze-tts/builds/code-predictor/code_predictor_lm_head_{0..14}.axmodel`。

---

## 5. 体积与耗时（20.38 实测）

| 产物 | 数量 | 单个体积 | 合计 |
|---|---|---|---|
| talker 逐层 axmodel | 28 | 见 builds/talker（build 完成时补记）| — |
| talker post | 1 | — | — |
| depth 逐层 axmodel | 12 | 30,674,769 B | 368.1 MB |
| depth post | 1 | 74,483 B | 73 KB |
| lm_head axmodel | 15 | ~2.2 MB | ~33 MB |
| embeds bins | 3 | — | ~198 MB |

- 编译耗时：depth 12 层 + post 用 `--parallel 6` 约 **5 分钟**（10:28→10:33 UTC）。
  talker 单层在 kv 512 + 5 个 prefill shape group 下约 10-15 分钟，28 层 `--parallel 10` 约 35-45 分钟。
- 也验证过的更快档：tiny-qwen3（1 层、kv 128、单 prefill group）单层 2.5 分钟；
  tiny-llama（2 层、kv 32）~30 秒/层。

---

## 6. CPU 数值对拍（关键验收）

脚本：`verify_cpu.py`（Mac `_work/venv-ref`，torch 2.15 / transformers 4.57.3，float32）。
对拍对象：
- talker：原始 `BreezeBackboneAdapter`（从 checkpoint 载入，`BreezeBackboneFactory` 构造）vs
  `Qwen3ForCausalLM.from_pretrained(repacked talker)`；
- depth：原始 `BreezeDepthDecoderForCausalLM`（输入 **2048 维未投影**，内部自己投影）vs
  `LlamaForCausalLM.from_pretrained(repacked depth)`（输入 **1024 维已投影**）。

实测（T=24 随机 hidden，float32）：

| 指标 | cosine | max_abs |
|---|---|---|
| talker 28 层逐层 hidden（最小）| **1.00000000** | — |
| talker final normed hidden | 1.0000006 | 2.6e-4 |
| talker lm_head logits [2052] | 0.9999998 | 2.7e-4 |
| depth 16 步 normed hidden | 1.0000010 | 0.0 |
| depth 15 个位置 head logits | 1.0000007 | 0.0 |

artifact 保真度（bf16 bin vs f32 参考）：`audio_embed`、`depth_proj`、`codebooks_head` 三个 bin
逐元素 max_abs = 0（bf16 原值原样）；`proj_audio_embed` max_abs = 3.9e-3、mean 5.6e-5
（f32 投影后舍入 bf16，符合预期）。

端到端 greedy token 对拍（12 帧 ×16 codebook，两条完整 pipeline：原始 Breeze 模块 vs
重打包模型 + 上述 artifacts）：**见 `verify.log`（token 一致率）**。
> 说明：greedy 对拍里按 `generation_breeze.py` 的规则把保留 id {2048,2049,2050} 的 logits 置 -inf，
> backbone EOS=2051 出现即停止。

---

## 7. 踩坑记录（按发生顺序）

1. **`qwen3_tts_talker` 的 rope 是硬编码 mrope**：CfgLoader 里 `rs_rope_type='mrope'`、
   `rs_mrope_section=[24,20,20]`、`interleaved_mrope=True`，config 的 `rope_scaling` 完全不读
   （在 talker_config 内嵌、顶层放都不会覆盖）；`llm_build` 对这两个 model_type 也不做通用 rope 解析。
2. **`qwen3_tts_talker_code_predictor` 强制 qk-norm**（builder 的 qk_norm 列表包含它），
   Breeze depth 层没有 qk-norm → 数学不等价。改用 `--model_type llama` 后完全绕开。
3. **`--model_type qwen3` 的 CfgLoader 会忽略 config 的 `rope_scaling`**，但会按 dataclass 字段名映射
   显式 `rs_*`/`use_qk_norm` 等字段（实测 `use_qk_norm: true` 生效）；llama model_type 则会解析
   `rope_scaling`（llama3 完整支持）。**Breeze talker 恰好是 plain rope，所以 qwen3 默认即精确。**
4. **pulsar2 python / torch 环境**：直接 `source env.sh` 后 `python3` 会解析到 pulsar2 包装器（需要它自己的
   `ld-linux`）；写脚本时应显式用 `$PULSAR2_HOME/python3` 经 `ld-linux` 启动，或用 venv python 单独跑
   safetensors/torch 重打包。
5. **bf16 → numpy 转换**：torch 2.15 的 `tensor.numpy()` 不支持 bf16，写 bin 时用
   `t.view(torch.uint16).numpy().tobytes()` 取原始 bf16 字节。
6. **pulsar2 build 必须给 input config**：`pulsar2 build` 不走 `llm_build`，需要 config 里带
   `quant.input_configs`（校准 tar.gz，内含 `[1,1,1024]` float32 npy），否则报
   `QuantError: Seems config of input(input) doesn't exist`。
7. **Breeze depth 对整条输入做投影**（`inputs_embeds` 在 `if inputs_embeds is None` 分支外仍被
   `inputs_embeds_projector` 处理），所以对拍时 reference 侧要喂 2048 维、repack 侧喂投影后的 1024 维。
8. **对拍脚本要小心 HF 层返回 tensor 还是 tuple**：transformers 4.57 的 `Qwen3DecoderLayer` 返回 tensor，
   `layer(...)[0]` 会切掉 batch 维，导致下一层出现 "16 vs 128" 的假错误。

---

## 8. 复现清单

| 文件 | 说明 |
|---|---|
| `repack_breeze.py` | Breeze → Qwen3/Llama HF 目录 + embeds bin + 15 lm_head ONNX。`--layers N` 可做 tiny build。 |
| `verify_cpu.py` | 原始 Breeze 模块 vs 重打包 HF 模型的 CPU 数值对拍（含端到端 greedy token 对拍）。 |
| `make_lmhead_calib.py` | 生成 lm_head INT8 校准包 `calib/input.tar.gz`。 |
| `build_lm_head.sh` | 15 个 lm_head ONNX → axmodel 的批量编译脚本。 |
| `REPACK_REPORT.md` | 本文件。 |

（配套：lead 的同源脚本 `~/work/breeze-tts/repack-lead/repack_breeze.py`、`verify_repack.py`；
正式 build 使用 `repack-lead/output/{qwen3-talker,llama-depth}` 目录。）

## 9. 遗留 / 待办

- [x] talker 正式 build（28 层 + post）→ `builds/talker/TALKER_OK`
- [x] depth 正式 build（12 层 + post）→ `builds/code-predictor/CODEPRED_OK`
- [x] 15 × lm_head axmodel
- [x] embeds 3 个 bin
- [ ] 板端 runtime 需按 `builds/talker/qwen3_post.axmodel` 与 `builds/code-predictor/llama_post.axmodel`
      的实际 I/O 名（`output`/`output_norm`、dtype）接线；per-layer I/O 已由 board-105 用 tiny 包 probe。
- [ ] talker 若要去掉分块 prefill，可重编无 `--last_kv_cache_len` 版本（prompt ≤128 token）。
