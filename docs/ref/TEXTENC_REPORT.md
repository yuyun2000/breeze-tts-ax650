# Breeze TTS 2 文本编码器 → AX650：导出 / 编译 / 精度报告（task-4）

- 时间：2026-11-11 19:30 CST
- 负责人：exp-2038（20.38 / ai-group-vm01）
- 结论一句话：**ONNX（fp32 两段）导出与数值对齐全部通过；Pulsar2 8.0 的 NPU 量化/编译在本模型上未达 0.99 验收（INT8=0.44、显式 U16=0.90、auto 混合精度因 AxRoPE FP32 backend bug 崩）→ 本期 textenc axmodel 不交付，资产与接口全部保留，等待工具链修复。**

---

## 1. 环境

- venv：`~/work/breeze-tts/env/venv`（python 3.10.20，torch 2.9.1+cpu，transformers 4.57.3，onnx 1.23.2，onnxruntime 1.23.2）
- 入口：`source ~/work/breeze-tts/env/activate.sh`
- Pulsar2：8.0 独立包 `$HOME/.cache/magnetar/pulsar2/8.0/bin/pulsar2`
- 权重来源：HF `BreezeBlue/Breeze-TTS-2`（`text_encoder.*` + `text_encoder_proj`，bf16，共 999.9M 参数，其中纯 encoder+proj 697.6M，embedding 表 302M 留在 CPU）

## 2. ONNX 导出（成功，作为后续修复的基准）

| 项 | chunk1 | chunk2 |
|---|---|---|
| 层 | 0–12（13 层） | 13–25（13 层）+ `text_encoder_proj` |
| 文件 | `onnx/textenc_L128_c1of2.onnx` | `onnx/textenc_L128_c2of2.onnx` |
| 体积 | **1403.0 MB** | **1412.5 MB** |
| 输入 | `inputs_embeds` float32 [1,128,1152] + `attention_mask` float32 [1,128] | 同左 |
| 输出 | `text_embeds` float32 [1,128,1152] | `text_embeds` float32 [1,128,2048] |
| 导出 | opset17，TorchScript exporter，fp32，导出耗时 ~61 s/段 | 同左 |
| 校验 | ONNX vs PyTorch 逐段中间 hidden **cos=1.000000** | 最终输出 **cos=1.000000** |

- 自测：`verify_onnx.py`（8 条真实中英 prompt，token 长度 20–101，含 87/69 长句）→ 全位置/仅真实 token 的 cosine 均 **1.000000**（mean=min=1.000000）。
- 独立交叉：`crosscheck_ref_golden.py` 用 ref-2038 在 Mac（torch 2.15/transformers 4.57.3 bf16）产出的 `ref/golden_zh/probe.npz::text_embeds[35,2048]`：
  **cosine = 0.99993289**，max_abs_diff = 6.36e-2（golden |x| mean 0.197，差异即 bf16 参考的舍入量级）→ 通过。
- 导出实现要点（详见 `scripts/export_textenc_onnx.py` 注释）：
  - 双向注意力（无 causal）；mask 用图内 `(1-mask)*(-60)` 作为 attention bias（softmax 后 exp(-60)≈1e-26，等价硬 mask，同时避免 -inf 破坏 INT8 标定 scale）；
  - sliding window=512 在 L=128/256 下等价全注意力（已核对窗口边界，脚本对 L>256 有断言）；
  - 每层 RoPE 按 layer_type 区分（sliding 层 theta=1e4/default；full 层 theta=1e6 + linear factor 8）；scaling=`query_pre_attn_scalar^-0.5`=1/16；q/k RMSNorm(1+w, eps=1e-6)；GELU-tanh MLP；
  - embedding 不在图内：`embed = table[token]*sqrt(1152)`，`token==256000` 时用 `eoi_embedding`（不乘 scale）；padding 填 0。

## 3. 板端 I/O 协议（固定，http://不可变）

```
token_ids (Breeze tokenizer, 自动 BOS=2) --CPU LUT--> inputs_embeds f32[1,128,1152] + attention_mask f32[1,128]
  → axmodel chunk1 → hidden f32[1,128,1152]
  → axmodel chunk2（同一 mask）→ text_embeds f32[1,128,2048] → 取前 T 行喂 talker prefill
```
- 张量名：输入 `inputs_embeds`、`attention_mask`；输出 `text_embeds`（两段同名）。
- 输入是 **fp32 embedding，不是 token id**；embedding LUT 在 CPU（`text_embed_cpu.py`，纯 numpy：604MB bf16 mmap + `eoi_embedding.npy` + scale sqrt(1152)=33.9411）。
- 桶长 128；T≤128；更长的 prompt 需另编 256 桶（同脚本 `--seq-len 256`）。
- 冒烟样本：`calib/val/smoke_sample.npz`（真实 prompt：`[S0]<ins_bos>用温柔平静的语气说<ins_eos>我们下午去超市买点水果和蔬菜吧。`，21 token）。

## 4. NPU 尝试全记录（Pulsar2 8.0 / AX650 / NPU1）

### 4.1 INT8（MinMax，40 条真实中英校准）—— 编译成功，精度不达标
- 产物：`artifacts/textenc_L128_c1of2.axmodel`，**357,520,638 B（≈357.5 MB）**，build 18.2 min（校准 5.2 min + NPU 编译约 11 min）。
- 计算量（编译日志实测）：`group 0 QuantAxModel macs: 45,092,962,304` → **45.09 GMACs/段，两段合计 ≈90 GMACs @ L=128**。
- `pulsar2 run` 仿真：**200 s/样本/段**（cmodel；与真实 NPU 无关）。2 样本结果：
  - sample0：cos_all=0.553991，cos_real=0.463538
  - sample1：cos_all=0.546879，cos_real=0.441290
  - **mean cos(real tokens)=0.452414 → 不达标（<0.99）**
- Pulsar2 precision_analysis（EndToEnd）证据（`evidence/int8_c1_precision_table.txt`）：
  - 最终输出 `text_embeds` cos=**0.41239**；424/491 个逐层节点 cos<0.99；
  - 最差激活集中在 MLP 中间 (1,128,6912)：**cos=0.069 / 0.161 / 0.193**；另有 (1,128,1152) 层 cos 0.60–0.88；
  - Pulsar2 明确提示：`[Info]These op contains very big range, maybe cause low precision`，点名各层 `RMSNormalization`、残差 `Add`/`Add_1`、`self_attn/Mul_6`；
- 判定：**不是导出/mask 问题**（同一 ONNX fp32 对 PyTorch/golden cos≈1.0），是典型 transformer INT8 激活动态范围崩坏。

### 4.2 显式 U16 layel_configs（MatMul/Add/Mul/Sub/Div/Pow/ReduceMean/Sqrt/Tanh/Softmax/Neg/RMSNorm 全 U16，MatMul 权重 U16）—— 精度 0.90，仍不达标
- 16 样本校准（`calib/calib16`）下完成量化与 precision_analysis 后，在 NPU 编译阶段按时间预算主动 kill（未产出 axmodel）。
- precision_analysis（`evidence/u16_c1_precision_table.txt`）：最终输出 cos=**0.90153**（另两条终段 0.98362 / 0.83489）；424/491 节点 <0.99；最差 (1,128,1152) cos=0.6965（mse=23.09），MLP 中间 0.755–0.81。
- 判定：U16 比 INT8 好很多但仍不过 0.99；剩余误差说明还有未被 layer_configs 匹配到的高动态范围算子（Pulsar2 内部融合算子名与 ONNX op_type 不完全一致）。

### 4.3 auto 混合精度（`highest_mix_precision: true`）—— 模型未被量化，且 FP32 AxRoPE backend 崩溃
- 现象：`Num of Quant Config: [0]`、`Network Quantization Finished.` → **该开关下模型实际未量化（FP32）**；因此 precision_analysis 表 518 行全部 cos=1.000000（这是"不量化"的假象，不是可用结果）。
- 编译失败关键行（`evidence/auto_c2_compile_failure.txt` / `auto_c2_compile_tail.txt`）：
```
2205:│ op_193:… │ AxRoPE  │ /encode… │ (1, 1,  │ FP32  │  FP32  │ 1.00000 │ 0.00000 │
2322:2026-11-11 11:00:48.629 | INFO     | axnn.yamain.command.build:compile_ptq_model:1190 - group 0 QuantAxModel macs: 726,319,235,072
2337:AssertionError: s1 = S(1024, 1024) s2 = S(32, 8192)
2351:axnn.backend.base.OpBuildException: op: AxRoPE, s1 = S(1024, 1024) s2 = S(32, 8192)
2355:	 Exception: s1 = S(1024, 1024) s2 = S(32, 8192)
2373:axnn.yamain.common.error.CodeException: (<ErrorCode.NPUBackendError: 8>, OpBuildException("op: AxRoPE, s1 = S(1024, 1024) s2 = S(32, 8192)\n\tattrs = {'const_inputs': {'sin_param': {'shape': (128, 128), 'dtype': 'FP32'}, 'cos_param': {'shape': (128, 128), 'dtype': 'FP32'}}, 'output_dtype': 'FP32', 'prefix': 'op_185:onnx.RotaryEmbedding_s0_sp_extra'}\n\tinput = {'x': Tensor(FP32, name=/encoder/layers.0/self_attn/k_norm/Mul_1_output_0, shape=(1, 1, 32, 256), offset=0, bit_strides=(1048576, 1048576, 8192, 32), memory_type=0), 'indices': Tensor(S32, name=op_184:onnx.RotaryEmbedding_default_position_ids_op_185:onnx.RotaryEmbedding_s0_sp, shape=(1, 32), offset=0, bit_strides=(1024, 32), memory_type=0)}\n\toutput = {'r': Tensor(FP32, name=/encoder/layers.0/self_attn/Add_1_output_0, shape=(1, 1, 32, 256), offset=0, bit_strides=(1048576, 1048576, 8192, 32), memory_type=0)}\n\t Exception: s1 = S(1024, 1024) s2 = S(32, 8192)"))
```
- 另测 `precision_analysis_mode: NPUBackend + highest_mix_precision: true`（c1/c2 各一次）：同样 `Num of Quant Config: [0]`（仍未量化），在截止时间前主动终止，未等到编译（预期同样撞 AxRoPE FP32 bug）。
- 判定：Pulsar2 8.0 NPU backend 的 **AxRoPE FP32 codegen bug**（shape 广播分支 `s1 = S(1024,1024) s2 = S(32,8192)`）；若能让 rope 输入保持 U16/INT8 可绕开，但 auto 路线下 rope 必为 FP32。

### 4.4 SmoothQuant（`enable_smooth_quant: true`，INT8，16 样本）
- c1 启动后在 "Get Outlier 15/16" 阶段**被人工 kill**（为给 4.3 让出 CPU），未出结果；c2 因 shell 目录 bug 未真正启动。
- 说明：SmoothQuant 正对"激活 outlier"这一类问题，是**下一条最值得试的路线**（比 U16 体积小，~360MB/段），但本轮时间不够。

## 5. 资产清单（20.38 `~/work/breeze-tts/textenc/`）

| 路径 | 内容 | 说明 |
|---|---|---|
| `onnx/` (2.8G) | c1/c2 两段 ONNX + meta json | **保留**（修复后直接复用；不传 HF，太大） |
| `weights/` (2.5G) | textenc_state.pt(bf16)、`text_embed_262158x1152.bfloat16.bin`(604MB)、`eoi_embedding.npy`、`embedding_meta.json` | 保留 |
| `calib/` (30M) | 40 条真实校准 + 8 条验证 + `smoke_sample.npz/json` + `calib/calib16/`(16 条小校准) | 保留 |
| `artifacts/textenc_L128_c1of2.axmodel` (341M) | INT8 反例产物 | 保留作证据 |
| `evidence/` (1.1M) | INT8/U16/auto precision 表、auto 编译失败行、INT8 编译日志 | 保留 |
| `verify/` | `verify_c1c2.json`（cos 1.0）、`crosscheck_ref.json`（golden 0.99993） | 保留 |
| `simulate/` | INT8 chunk1 仿真 metrics + I/O bin | 保留（2.3M） |
| `scripts/` | 全部脚本（见 §6） | 保留（已同步 Mac `_work/textenc/scripts/`） |

### 复现命令（修复工具链后从 ONNX 重编译）
```bash
source ~/work/breeze-tts/env/activate.sh && export PATH=$PULSAR2_HOME/bin:$PATH
cd ~/work/breeze-tts/textenc
source-venv-python scripts/make_pulsar2_config.py --onnx onnx/textenc_L128_c1of2.onnx \
    --calib-dir calib/calib16 --build-root build/c1of2 --tag textenc_L128_c1of2 <精度选项>
(cd build/c1of2 && pulsar2 build --config textenc_L128_c1of2.pulsar2_config.json)
python scripts/run_simulate.py --axmodel build/c1of2/textenc_L128_c1of2.axmodel \
    --onnx onnx/textenc_L128_c1of2.onnx --n 1
```
（`make_pulsar2_config.py` 支持 `--mixed u16|auto`、`--precision-mode NPUBackend`、`--smooth`、`--n-calib`，便于后续一键复现。）

## 6. 脚本（均带注释、可复现）

| 脚本 | 作用 |
|---|---|
| `extract_textenc_weights.py` | 从 HF safetensors 抽 text_encoder+proj（3.7 s）；输出 bf16 embedding bin + eoi |
| `export_textenc_onnx.py` | 导出 ONNX（支持 `--layers a:b`、`--proj`、`--no-final-norm`、`--export-params`） |
| `verify_onnx.py` | ONNX 链 vs PyTorch（逐段中间 hidden + 最终输出 cosine） |
| `crosscheck_ref_golden.py` | 与 Mac ref golden `probe.npz::text_embeds` 交叉验证 |
| `make_calib_data.py` | 用 Breeze tokenizer 构造真实中英 prompt（40 校准 + 8 验证 + smoke 样本） |
| `text_embed_cpu.py` | 纯 numpy CPU embedding LUT（bf16 mmap / eoi / scale） |
| `make_pulsar2_config.py` | 生成 Pulsar2 build config（INT8/U16/auto/SmoothQuant 可切） |
| `run_simulate.py` | pulsar2 run 仿真 + vs ONNX cosine（支持两段串联） |

## 7. 后续建议（一句话各）

1. **升级/修复 Pulsar2**：AxRoPE FP32 codegen bug 是 auto 路线唯一阻塞；修好后 auto（会选高精度）可能直接得到 ≥0.99。
2. **SmoothQuant + 16 样本 + 逐层 U16 组合**：针对激活 outlier，体积仅 ~360-400MB/段，是 INT8 路线里最值得的一试（本轮没时间）。
3. **QAT**：AXERA 官方 QAT.axera（QDQ）可在 INT8 下恢复精度，但需要训练数据与时间。
4. **兜底部署方式（已与 lead 对齐）**：文本编码器在 host/服务器侧计算（ONNX fp32 cos=1.0），板端通过 `--text-embeds` 注入 golden text_embeds；textenc NPU 作为已知限制。
5. 若坚持 NPU：可把 encoder 再细分成更多段（每段层数更少）以降低单图动态范围——经验上对量化误差帮助有限，不建议优先。

## 8. 可清理清单（20.38 textenc 目录，已执行大部分）

- 已删（本轮清理）：`build/*/work/` 全部中间目录、失败的 `build/{c1of2_auto,c1of2_npuauto,c2of2_auto,c2of2_npuauto,c1of2_sq,c2of2_sq}`、无产物的 `build/{c1of2_u16,c2of2}`、重复的 `env/wheels/torch_cpu.whl`（184MB）→ 释放 **~16 GB**（/ 可用 28G→44G）。
- 仍可清理（后续确认后）：`build/` 剩余编译日志/配置（<400MB）、`simulate/`（2.3M）、`artifacts/` INT8 axmodel（341M，反例证据，建议留到报告定稿）。
- **不要删**：`onnx/`、`weights/`、`calib/`、`evidence/`、`verify/`、`scripts/`。

---

# BONUS（2026-11-11 20:40 CST）：fused 节点 U16 修复 → **达标交付**

## 结论
- **链式 cosine（pulsar2 run 仿真 vs ONNX fp32）= 0.9996766（实 token）/ 0.9995082（全位置）** → ≥0.99，**textenc axmodel 可交付**。
- 两段 axmodel：c1 **381,721,897 B (381.7MB)** `sha256 80bed2d8…2c2f`；c2 **384,311,393 B (384.3MB)** `sha256 57a9db93…9233d`（比 INT8 的 357.5MB 仅大 ~7%，CMM 友好）。
- 产物路径：`build/c1of2_hp/textenc_L128_c1of2.axmodel`、`build/c2of2_hp/textenc_L128_c2of2.axmodel`（config 同目录），标记 `build/TEXTENC_OK`。

## 根因（之前 0.90 的原因）
之前显式 U16 只覆盖了 **ONNX op_type**（MatMul/Add/Mul/…），但 Pulsar2 会先做算子融合，产生 `FullyConnected`(91)、`RMSNormalization`(51)、`RotaryEmbedding`(26)、`Gelu`(13) 等 fused 节点，**这些节点输出仍是 U8**（从 `quant_axmodel.json::tensor_configs` 逐节点 bit_width 可证：196 个节点输出 8-bit）→ RMSNorm/MLP 激活以 U8 进入后续计算，精度卡在 0.90。

## 修复（配置层面，无需改 ONNX）
在 `quant.layer_configs` 的 `op_types` 中**追加 Pulsar2 fused op 名**：
`FullyConnected, Gelu, RotaryEmbedding, RoPE, AxRoPE`（以及原有的 `RMSNormalization, LayerNormalization, MatMul, Add, Mul, Sub, Div, Pow, ReduceMean, Sqrt, Tanh, Softmax, Neg, Slice, Transpose, Reshape, Expand, Cast, Concat, Where, Equal, Unsqueeze, Squeeze, ConstantOfShape`），全部 `data_type=U16, output_data_type=U16`（权重仍 S8；Pulsar2 8.0 的 `weight_data_type` 只允许 S8/FP32）。
效果：QDType 分布 U16 476 / S16 26 / FP32 1，**503 个节点全部 cos≥0.99928**，终段 0.99998/0.99987/0.99991。

## 第二个坑：chunk2 必须用「chunk1 输出 hidden」做校准
第一版 c2（用 embedding 校准，链式 cos 只有 **0.754**）在板上也复现为 0.807：chunk2 的真实输入是 chunk1 hidden（实测 min=-437 / max=1105 / mean_abs=19.9），而 embedding 量级只有 ±3 → U16 输入量化 scale 严重偏小、hidden 被裁切。
修复：新目录 `calib/c2_hidden/` —— 把 16 条校准样本先过 ONNX chunk1，取 **hidden [1,128,1152]** 作为 c2 的 `inputs_embeds` 校准数据，attention_mask 沿用 calib16。重建后链式 cos 0.99968。

## A/B 记录
| 变体 | 配置 | 结果 |
|---|---|---|
| A | S8 权重 + 全 U16 激活（fused 名补齐） | **precision 表 final≈0.9999；仿真链 cos 0.99968 ✅ 采用** |
| B | A + MatMul `weight_data_type=FP32` | 与 A 的 QDType 分布/精度完全相同（该字段对本图未生效）→ 弃用 |
| 旧 c2（embedding 校准） | A 配置 | 链 cos 0.754（输入量化 clip 根因）→ 修正为 c2_hidden 校准 |
| auto（`highest_mix_precision=true`） | — | 实为未量化 FP32（Quant Config 0），AxRoPE FP32 backend 崩 → 排除 |
| 显式 U16（不含 fused 名） | U16 仅 ONNX op 名 | 节点 196 个仍 U8 → cos 0.90，不达标 |

## 复现命令（20.38）
```bash
source ~/work/breeze-tts/env/activate.sh && export PATH=$PULSAR2_HOME/bin:$PATH
cd ~/work/breeze-tts/textenc
PY=~/work/breeze-tts/env/venv/bin/python
# 1) chunk1
$PY scripts/make_pulsar2_config.py --onnx onnx/textenc_L128_c1of2.onnx --calib-dir calib/calib16 \
    --build-root build/c1of2_hp --tag textenc_L128_c1of2 --mixed u16 --n-calib 16
(cd build/c1of2_hp && pulsar2 build --config textenc_L128_c1of2.pulsar2_config.json)
# 2) chunk2（校准数据 = chunk1 hidden，见 calib/c2_hidden）
$PY scripts/make_pulsar2_config.py --onnx onnx/textenc_L128_c2of2.onnx --calib-dir calib/c2_hidden \
    --build-root build/c2of2_hp --tag textenc_L128_c2of2 --mixed u16 --n-calib 16
(cd build/c2of2_hp && pulsar2 build --config textenc_L128_c2of2.pulsar2_config.json)
# 3) 仿真：c1 run → 输出作为 c2 输入 → 对比 ONNX 链（详见 simulate/chain_A2_metrics.json、verify/smoke_ref.npz）
```
（`scripts/make_pulsar2_config.py` 的 `--mixed u16` 已内置完整 fused op_types 列表。）

## 板端注意
- 接口不变：`inputs_embeds` f32[1,128,1152] + `attention_mask` f32[1,128] → 段1 hidden[1,128,1152] → 段2 text_embeds[1,128,2048]，取前 T 行。
- 若板端用旧 c2（链 cos 0.807）请更新为 `sha256 57a9db93…9233d`；参考值见 `verify/smoke_ref.npz`。
- 两段 axmodel 共 766MB，CMM 可分时加载（段1 用完释放再载段2），不会与 talker 峰值冲突。
