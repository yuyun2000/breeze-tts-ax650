# 经验与踩坑（Breeze TTS 2 → AX650）

> 本文只记"下次还会踩"的坑，按重要性排序。转换细节见 [conversion.md](conversion.md)。

## 1. 先弄清模型"真实"结构，别信顶层 config

Breeze TTS 2 的 `config.json` 里有 `rope_theta=500000` + llama3 rope_scaling，但 **backbone 根本不用它**：
`backbone_model_type="qwen3"` → 走 `BreezeBackboneAdapter`，用嵌套 `backbone_config`
（**rope_theta=1e6、rope_scaling=null**、rms eps 1e-6）实例化 Qwen3 层与 RoPE。
只有 `depth_decoder_config` 的 llama3 rope 真正生效。

**教训**：遇到"自定义模型 + 嵌套 config"时，先 `print(eval(model).rotary_emb.rope_type, inv_freq)` 实测，
再动转换方案。rope 错了 token 会从第 1 帧分叉（消融：depth rope 改 default → 整帧一致率 0.010、首分叉帧 1），
但它未必影响可懂度（ASR 字符准确率仍有 96.4%）——所以"能用"和"精确"要分开验收。

## 2. 官方 model_type 未必匹配你的模型，通用 model_type 反而更准

Pulsar2 8.0 内置了 Qwen3-TTS 的转换支持（`qwen3_tts_talker` / `qwen3_tts_talker_code_predictor` /
`qwen3_tts_tokenizer_12hz`），看起来天作之合（Breeze 的 codec 就是 Qwen3-TTS tokenizer，字节一致）。
但：
- `qwen3_tts_talker` 实现是 **mRoPE**（prefill `indices` 是 [3,P] 三轴位置），Breeze backbone 是普通 1D RoPE；
- `qwen3_tts_talker_code_predictor` 的层固定带 **qk-norm**，Breeze depth decoder 是 Llama 式**没有 qk-norm**、
  KV 头数也不同（2 vs 8）。

→ 最终用通用 `--model_type qwen3`（talker）与 `--model_type llama`（depth），**结构完全同构、rope 精确、
配置简单**。判断依据：ax-llm C++ 源码里通用 LLM 的 prefill `input` 张量就是 **bf16 embeddings**
（`LLaMaEmbedSelector` 在 CPU 查表后灌入），所以"外部 embedding + 自管采样"的自定义推理链路
（TTS/多模态）用通用 LLM 编译产物完全可以实现。

## 3. 复用现成编译产物：先做字节级比对

Breeze 的 `audio_tokenizer/model.safetensors` 与 `Qwen/Qwen3-TTS-Tokenizer-12Hz` 的 sha256 完全一致
（LFS oid `836b7b35…571258`）→ 直接下载 AXERA 已编译的 speech_tokenizer axmodel 包，节省一整条 codec 转换/校准链路。
**教训**：模型里"基于 XXX 的组件"先查是不是原封不动（权重 hash、config archive 名），能省一天。

## 4. llm_build 产物是"per-layer + 外置 KV cache + 外置 embedding"

- 每层一个 axmodel；shape_group 0 = decode（单 token），1..N = prefill 的第 1..N 个 chunk；
- `indices` 是位置（talker 的 mRoPE 版是 3 行，通用版 1 行——**必须从 axmodel 实际 I/O 读，不要照抄**）；
- `mask` 是 bf16 的加法掩码（0 / -65536），decode 时**当前 token 固定放最后一列**（不是 col=cur）；
- KV cache 在 host numpy 里按层维护，每步把 `[1,1,kv_dim]` 写回第 cur 行；
- talker 的 post 用 `--ret_postnorm` 才能拿到 `output_norm`（depth decoder 位置 0 需要它，不是 pre-norm hidden）。

## 5. 转换机 CPU 与参考推理

本次转换机是 QEMU 虚拟 CPU（`QEMU Virtual CPU version 2.5+`，只有 SSE4.2，无 AVX/AVX2）：
Pulsar2 编译正常，但 **PyTorch CPU 参考推理慢到不可用**；参考生成/对拍放到 Apple M4 Pro（bf16 CPU，
0.29 s/帧，RTF≈3.6）完成，再与板端产物对拍。这也是"团队 + 本机协同"的一个实用分工。

## 6. 其它小坑

- `pulsar2` 独立包不会自动进 PATH（`$PULSAR2_HOME/bin`），且 `llm_build` 与 `llm_build2` 参数不同；
- HF 模型下载走 `hf-mirror.com` 时，部分 LFS 文件要用 `resolve/main/...`（`raw/...` 会拿到 pointer 文本）；
- 板端 bf16 数组用 `ml_dtypes.bfloat16`；`np.zeros(...,dtype=bf16) - 65536.0` 会把 dtype 提升成 fp32，
  要用 `np.full(..., -65536.0, dtype=bf16)`；
- 板端 pyaxengine 的 `run()` 会**严格断言** feed 的 shape/dtype，必须按 `get_inputs(shape_group)` 逐组构造；
- `speech_tokenizer_decode_post` 的 t64 桶输出 122325 采样（不是 64×1920=122880），按 C++ 的
  `left_context=25 / capacity=39` 分块并最终裁剪到 `T*1920`。


## 7. 文本编码器 NPU 混合精度（BONUS：从 0.45 → 0.99968）

T5Gemma2 文本编码器（26 层、hidden 1152，90 GMACs/128 token）在 Pulsar2 8.0 上走了完整的坑，
最终达标。关键经验（对其它 transformer 编码器同样适用）：

1. **INT8 直接量化不达标**（cos 0.41–0.55）：precision_analysis 显示 MLP 中间激活 (1,128,6912)
   的 cos 只有 0.07–0.19、"very big range"，RMSNorm/残差 Add 也只有 0.6–0.88。文本模型的
   embedding 输入动态范围远大于视觉模型，INT8 激活不够。
2. **只按 ONNX op 名配 U16 也不够（cos 0.90）**：Pulsar2 量化前会做算子融合，融合后的节点
   （`FullyConnected`、`RMSNormalization`、`RotaryEmbedding`、`Gelu` 等，本图共 196 个）**输出仍是 U8**。
   必须把这些**融合后的 op 名**也写进 `quant.layer_configs` 的 `op_types` 并设 `data_type/output_data_type=U16`。
   把这些补上后 precision 表 503 个节点全部 cos≥0.99928。
3. **`highest_mix_precision=true`（auto）在 Pulsar2 8.0 上不可用**：它实际产出"未量化"图
   （`Num of Quant Config: [0]`，precision 全 1.0 是假象），并崩在 FP32 `AxRoPE`
   （`OpBuildException: s1=S(1024,1024) s2=S(32,8192)`）。显式 U16 配置（`RotaryEmbedding` 保持 U16/非 FP32）
   可绕过。
4. **两段拆分时，第二段的校准数据必须是第一段 axmodel 的真实输出分布**：我们最初把两段都用
   text-embedding 分布校准，c2 的输入是 chunk1 hidden（min/max ≈ -437/1105），U16 clip 后链式 cos
   只有 0.75；改用「真实 prompt → ONNX chunk1 → hidden」做 c2 校准后 **链式 cos 0.99968**。
5. 权重用 S8、激活全 U16 即可（体积几乎不涨：两段共 766 MB vs INT8 715 MB），
   板端两段串行编码 196 ms/次，CMM +742 MB 且**用完即释放**（textenc 与 talker 不同时驻留）。

结论：文本编码器已 100% 在 NPU 上（U16/S8 混合精度），与 talker/depth/codec 组成完整 NPU 流水线。
