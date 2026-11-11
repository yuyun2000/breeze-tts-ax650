# 模型包目录结构

完整模型包（约 2.5–3 GB）发布在 HuggingFace：`yunyu1258/breeze-tts-2-ax650`（待上传）。
本目录只说明结构并提供下载/校验脚本；权重文件不进 git。

```
breeze-tts-2-ax650/
├── config/
│   └── configuration.json          # 运行时元数据：维度、prefill 桶、采样默认值、文件名模板
├── text-encoder/
│   ├── text_encoder.axmodel        # T5Gemma2(26L,1152) + proj(1152→2048)，输入 inputs_embeds [1,128,1152]
│   ├── embed_tokens.bfloat16.bin   # [262158, 1152] bf16，CPU mmap 查表（含 eoi_embedding 处理在 runtime）
│   └── tokenizer/                  # tokenizer.json / tokenizer_config.json / special_tokens_map.json
├── talker/
│   ├── qwen3_tts_talker_p128_l{0..27}_together.axmodel   # 28 层，bf16 hidden，外置 KV cache
│   ├── qwen3_tts_talker_post.axmodel                     # final RMSNorm + lm_head(2048→2052)，输出 logits + output_norm
│   ├── audio_embed_16x2051x2048.bfloat16.bin             # 16 组未投影音频 embedding 表（求和用）
│   └── talker_meta.json
├── code-predictor/
│   ├── qwen3_tts_talker_code_predictor_p64_l{0..11}_together.axmodel  # 12 层
│   ├── qwen3_tts_talker_code_predictor_post.axmodel                   # 12 层后的 final RMSNorm
│   ├── code_predictor_lm_head_{0..14}.axmodel                         # 位置相关 15 个 head（1024→2051）
│   ├── proj_audio_embed_{0..14}x1024.bfloat16.bin        # 投影后的音频表：0 用于 first_code_embed，1..13 用于后续步进
│   ├── depth_proj_2048x1024.bfloat16.bin                 # inputs_embeds_projector.weight  [1024,2048]
│   └── depth_meta.json
├── speech-tokenizer/                                     # 来自 AXERA Qwen3-TTS-12Hz-0.6B-Base-AX650（Breeze codec 与其字节一致）
│   ├── speech_tokenizer_decode_pre_inproj.axmodel        # codes int32[16,L] → hidden 512
│   ├── speech_tokenizer_decode_post_t64.axmodel          # hidden[64,1024] → wav（64×1920 采样）
│   └── decoder/
│       ├── qwen3_tts_tokenizer_12hz_p64_l{0..7}_together.axmodel
│       └── qwen3_tts_tokenizer_12hz_post.axmodel         # 512 → 1024
├── runtime/                                              # 板端 Python 运行时（numpy + pyaxengine + tokenizers）
└── LICENSE / NOTICE                                      # BreezeBlue 非商业许可 + 第三方声明
```

命名/尺寸以实际转换产物为准，`reports/` 下的报告与 `docs/` 会记录最终 sha256 清单。
