# BOARD_REPORT — AX650 板端环境 + AXERA Qwen3-TTS 编译包实机验证（task-2）

> 板卡：`192.168.10.105`（hostname `pyramid-openclaw`，root）
> 执行人：board-105　日期：2026-11-11 17:54–18:20
> 目的：给 Breeze TTS 2 板端 Python runtime 打地基——摸清 pyaxengine/bf16、AXERA 已编译 Qwen3-TTS axmodel 的**真实 I/O 协议**、性能、CMM/RAM 预算。
> 权威参考：HF `AXERA-TECH/Qwen3-TTS-12Hz-0.6B-Base-AX650/infer.py`（Python 协议）+ `_work/ax-llm-qwen3tts/src/runner/Qwen3TTSSpeechTokenizer.cpp`（C++ 权威）。**本报告所有 shape 都是在板端用 pyaxengine 读出来的实测值**。

---

## 0. 结论速览

1. **板端完全可用**：`axengine 0.1.3`（纯 Python wheel）+ `ml-dtypes 0.6.0` 装在系统 python3（3.10.12）上即可；无需装 torch/transformers。
2. **bf16 原生支持，板端 AX Engine 版本 2.12.0s 就够，不需要 SDK≥2.18**。bf16 axmodel 加载、prefill、decode、post 全部正常，输入/输出 NodeArg dtype = `bfloat16`，numpy 侧必须用 `ml_dtypes.bfloat16` 精确匹配（否则 pyaxengine 抛 AssertionError）。
3. **AXERA Qwen3-TTS-0.6B-AX650 包全部组件（talker 28 层、talker post、code-predictor、speech_tokenizer decode 全链路）在板上跑通**，包括随机 code → 24kHz wav 的 codec 全链路（`/root/breeze-tts/codec_random_20f.wav`）。
4. **性能代理**：下载了 AXERA **1.7B VoiceDesign** 包的 28 个 talker 层做 Breeze 等价几何体（hidden 2048 / 28 层 / kv_dim 1024）实测：
   - decode：稳态 **187.3 ms/帧（28 层，6.69 ms/层）**，首帧 292.8 ms（冷）；talker-only RTF = **2.34**（音频 80ms/帧）。
   - prefill：128 token = 375.6 ms；256 token（2 chunk）= 832.2 ms。
   - 28 层 + post 加载 9.2–9.4 s，**CMM 占用 +1950 MB**。
   - 预估 Breeze 端到端（talker+12 层 depth+16 帧 codec）：**≈ 3.0× 实时**（详见 §7）。
5. **CMM/内存可行**：停掉 MiniCPM-OCR demo（axllm，占 2.2GB CMM）后 CMM remain 5152MB；Breeze 全组件预估 ~2.45GB CMM（remain ~2.8GB）；python RSS 峰值 ~900–1000MB（多为 mmap 可回收页），1.9GB 系统内存下可行但建议 **talker build 用 `--kv_cache_len 512`**（host KV 从 235MB 降到 59MB）。
6. **必须 build 时加 `--ret_postnorm`**：0.6B Base 包的 talker post 有 `output_norm` 输出；1.7B VoiceDesign 包没有（没开 ret_postnorm）→ Breeze 的 depth decoder 需要 talker hidden，必须开。
7. 板端 `hf` 直连可用（~25MB/s），**不要用 hf-mirror**（308 重定向回 huggingface.co，导致 huggingface_hub 报 "Distant resource does not seem to be on huggingface.co"）。
8. 期间板子 18:01 发生过一次**主动重启**（journal 是干净的 systemd reboot，非 panic；时间点在 kill axllm 后 ~2min，也可能是操作员触发）。所有产物落盘在 `/root/breeze-tts/`，1.7B 下载自动续传。18:10 再次 kill axllm 后 20 分钟稳定。——长任务必须 nohup + 落盘 + 可续跑。

---

## 1. 板端环境（实测）

```
$ cat /proc/ax_proc/version     # 内容几乎为空，只有 "Ax_Version" 一行
$ cat /proc/ax_proc/chip_type   # AX650C_CHIP
$ cat /proc/ax_proc/uid         # ax_uid: 0x600840014c795202
$ uname -a                      # Linux pyramid-openclaw 5.15.73 #1 SMP PREEMPT aarch64
$ cat /etc/os-release           # Ubuntu 22.04 LTS
$ nproc                         # 8
$ free -m                       # Mem 1912MB total；Swap 2047MB
$ df -h /                       # 29G total, 14G free（含我们 2.5G 产物）
```

AX Engine 版本/能力（pyaxengine `InferenceSession` 初始化时打印）：

```
[INFO] Available providers:  ['AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.12.0s
[INFO] Model type: 2 (triple core)        # talker/codec transformer 是三核；lm_head/decode_post 是 single core
[INFO] Compiler version: 5.1-patch1-dirty
libax_engine.so -> /soc/lib/libax_engine.so（ldconfig 已注册）
```

CMM：`/proc/ax_proc/mem_cmm_info` → `total size=6291456KB(6144MB)`；**停 demo 后 used≈991MB，remain≈5152MB**。

python：`3.10.12`，pip 22.0.2；numpy 2.1.3（import 实测；pip 元数据里另有 2.2.6，不影响）；cffi 2.0.0 已有。
**板端有外网**：`huggingface.co` 直连可下（~25MB/s）、`pypi.org` 可达。所以 wheel/模型都在板端直接装/下，不用 Mac 中转。

---

## 2. 安装步骤（可复现）

```bash
# 1) pyaxengine：GitHub AXERA-TECH/pyaxengine releases，最新 0.1.3.rc3 → wheel 名 axengine-0.1.3-py3-none-any.whl
#    Mac 下载（直连 GitHub 可用）：
curl -L -o axengine-0.1.3-py3-none-any.whl \
  https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc3/axengine-0.1.3-py3-none-any.whl
scp axengine-0.1.3-py3-none-any.whl root@192.168.10.105:/root/breeze-tts/pkgs/

# 2) 板端安装（wheel 是 py3-none-any 纯 python + ctypes，无 aarch64 编译）
pip3 install /root/breeze-tts/pkgs/axengine-0.1.3-py3-none-any.whl --no-deps
pip3 install "ml-dtypes>=0.1.0"        # 实测装到 0.6.0；bf16 支持必需
# 注意 pip 22.0.2 没有 --break-system-packages 选项（Ubuntu 22.04 也不是 externally-managed，不需要）

# 3) 验证
LD_LIBRARY_PATH=/soc/lib python3 -c "import axengine; print('ok')"
```

API 要点（`axengine.InferenceSession`）：

```python
from axengine import InferenceSession
sess = InferenceSession(path)                 # 加载 axmodel
sess.get_inputs(shape_group=g) / sess.get_outputs(g)   # -> list[NodeArg(name, dtype, shape)]
outs = sess.run(None, feed_dict, shape_group=g)        # 返回 list[np.ndarray]，顺序 = get_outputs(g) 顺序
```
- `run()` 会**严格断言** feed 的 shape 与 dtype 和 `get_inputs(g)` 完全一致（先断言后拷贝），所以 feed 必须自己按组 shapes 构造。
- 输出无名字（`run(None,...)` 返回 list），用 `get_outputs(g)` 的顺序映射（talker/cp/codec transformer 的组输出顺序都是 `K_cache_out, V_cache_out, output`）。
- `shape_group` 是第 4 个参数（也接受关键字 `shape_group=`），和 HF infer.py 用法一致。

---

## 3. 板端已有模型产物

| 路径 | 内容 | 体积 |
|---|---|---|
| `/root/breeze-tts/qwen3tts-ref/` | AXERA `Qwen3-TTS-12Hz-0.6B-Base-AX650` 子集（talker 28 层+post、code-predictor 5 层+post+15 lm_head+bin、speech_tokenizer decode 全套、infer.py/config） | 921MB |
| `/root/breeze-tts/qwen17b/talker/` | AXERA `Qwen3-TTS-12Hz-1.7B-VoiceDesign-AX650` 的 talker（28 层 + post）——**Breeze talker 的等价几何体替身**（hidden 2048） | 1.6GB |
| `/root/breeze-tts/pkgs/` | axengine wheel | 19KB |
| `/root/breeze-tts/board_scripts/` | probe/smoke 脚本（见 §10） | — |

下载脚本：`/root/breeze-tts/dl_qwen3tts.py`、`dl_17b.py`（huggingface_hub `snapshot_download` + allow_patterns）。
**坑**：板端 `hf-mirror.com` 会 308 重定向到 `huggingface.co`，huggingface_hub 判定元数据不合规直接失败 → 板端直连 `HF_ENDPOINT=https://huggingface.co` 反而通。
**省空间**：0.6B 包里有 `talker.model.text_embedding.weight.bfloat16.bin`(622MB，Breeze 不用)、`speech_tokenizer_encoder`(68MB，Voice Design 不用)、`*_sim.onnx`(220MB) —— 全部没下。

---

## 4. axmodel I/O 协议（**最重要**，task-6 runtime 唯一依据）

通用约定：
- dtype：`input`/`K_cache`/`V_cache`/`mask`/`output` = **bf16**（numpy 用 `ml_dtypes.bfloat16`）；`indices` = **uint32**；codec 的 `codes` = **int32**；`hidden_512`/`wav`/lm_head 输入输出 = **fp32**。
- shape_group 语义：**group 0 = decode（单 token）**；**group 1..N = prefill 的第 1..N 个 chunk**（每 chunk 固定 P 个 token）。组数由 build 参数决定。
- mask 值：可见 = `0.0`，不可见 = `-65536.0`（bf16 可表示，不是 -inf）。

### 4.A talker（`llm_build --model_type qwen3_tts_talker`，28 层 per-layer axmodel + post）

0.6B（H=1024）与 1.7B（H=2048）实测，组数都是 **6**（build: `--prefill_len 128 --last_kv_cache_len 128,256,384,512` → 5 个 prefill 组 + 1 个 decode 组；**prefill 总容量 640 token**）：

| group | 用途 | K_cache / V_cache | indices | input | mask |
|---|---|---|---|---|---|
| 0 | decode 单 token | `[1, kv_len=2048, kv_dim=1024]` | `[1,1]` uint32 | `[1,1,H]` | `[1,1,2049]` |
| 1 | prefill chunk 1 | `[1,1,**H**]`（占位，注意是 hidden 不是 kv_dim） | `[3,128]` uint32 | `[1,128,H]` | `[1,128,128]` |
| 2 | chunk 2 | `[1,128,1024]` | `[3,128]` | `[1,128,H]` | `[1,128,256]` |
| 3 | chunk 3 | `[1,256,1024]` | `[3,128]` | `[1,128,H]` | `[1,128,384]` |
| 4 | chunk 4 | `[1,384,1024]` | `[3,128]` | `[1,128,H]` | `[1,128,512]` |
| 5 | chunk 5 | `[1,512,1024]` | `[3,128]` | `[1,128,H]` | `[1,128,640]` |

每层输出（三个，顺序固定）：`K_cache_out` `[1,P,kv_dim=1024]` bf16、`V_cache_out` 同上、`output` `[1,P,H]` bf16（group 0 的 P=1）。
- `indices`（prefill）：**3 行**，三行内容相同 = 每个位置的**绝对位置**（0..valid-1；padding 位置填 1）；decode = `[1,1]` = 当前 cache 长度。
- `mask`（prefill，group g）：列数 = 128*g；`mask[r][c]=0 当 c <= 128*(g-1)+r`（即全局因果），否则 -65536；padding 行也按全局位置算。
- `mask`（decode）：行 = 当前 cache 长度 cur；`col < cur → 0`；**最后一列（下标 kv_len=2048）→ 0**；其余 -65536。即当前 token 固定占 mask 最后一列，它的 K/V 由 kernel 内部追加，输出 `K_cache_out` 后写到 host cache 的第 cur 行。
- KV cache 在 host numpy 里维护（每层一份 K 一份 V），每步把 `[1,1,kv_dim]` 写回第 cur 行；**预填充时**把 `K_cache_out` reshape 成 `[P,kv_dim]` 写到 `[start:start+P]`。
- 最后一行 `output`（即 `raw_hidden`）→ talker post。

talker post（`qwen3_tts_talker_post.axmodel`，1 group）：
- 输入：`input` `[1,1,H]` bf16
- 输出（0.6B Base，build 带 `--ret_postnorm`）：`output` = logits **bf16** `[1,1,3072]`（vocab，0.6B 是 3072）+ `output_norm` **bf16** `[1,1,H]`
- 1.7B VoiceDesign 包**没有 output_norm**（只有 logits）→ **Breeze 必须 `--ret_postnorm`**。
- 耗时 0.7–0.9 ms/帧。

### 4.B code-predictor / depth decoder（`llm_build --model_type qwen3_tts_talker_code_predictor`）

0.6B 实测 **5 层**、hidden 1024、组数**2**（build: `--kv_cache_len 128 --prefill_len 64`）：

| group | 用途 | K_cache / V_cache | indices | input | mask |
|---|---|---|---|---|---|
| 0 | decode | `[1,128,1024]` | `[1,1]` uint32 | `[1,1,1024]` | `[1,1,129]` |
| 1 | prefill（一次吃 2 个 token，其余 0） | `[1,1,1024]` | `[1,64]` uint32（只有 1 行！） | `[1,64,1024]` | `[1,64,64]` |

- 输出：`K_cache_out`/`V_cache_out` `[1,P,kv_dim]`、`output` `[1,P,1024]` bf16。
- **cp 的 K_cache dim = 1024 = Qwen3-TTS cp 的 kv_dim（8 kv heads × 128）**，不是 hidden 的巧合（Qwen3-TTS 0.6B cp config 实测 `num_attention_heads=16, num_key_value_heads=8, head_dim=128`，kv_dim=1024）。→ **Breeze depth（2 kv heads ×128 = kv_dim 256）build 后必须重新读 shape，不能照抄 1024。**
- cp post：`input [1,1,1024]` bf16 → **只有** `output_norm [1,1,1024]` bf16（没有 logits）；0.37 ms/帧。
- `code_predictor_lm_head_{0..14}.axmodel`：`input` **fp32** `[1,1,1024]` → `output` **fp32** `[1,1,2048]`（0.6B 的 codebook 词表=2048；Breeze 应为 2051）。0.45 ms/个 → 15 个 ≈ 6.8 ms/帧。
- prefill 用法（HF infer.py/C++ 一致）：序列 = `[proj(talker hidden_norm), E'_0[cb0]]` 共 2 个有效位置，其余 pad 0；`indices[0]=0, indices[1]=1`，其余 0；mask 2×2 可见、其余 -65536；K/V 传 `[1,1,1024]` 的零占位；输出取第 2 行（valid_len-1）→ post → lm_head[0] 得 cb1 logits；之后每步把上一步 token 的 embedding（投影后 1024 维）喂 decode(shape_group=0) 再 post + lm_head[step]。

### 4.C speech tokenizer decode（AXERA 0.6B 包，**Breeze audio_tokenizer 与其权重字节一致 → 原样复用**）

1) `speech_tokenizer_decode_pre_inproj.axmodel`（1 group）
- 输入：`codes` int32 **`[1,16,325]`**；排布 = **q-major**：`feed[0,q,t] = codes[t][q]`（t<valid_len，其余 0），325 是固定桶（=26s 音频上限）。
- 输出：`hidden_512` fp32 **`[1,325,512]`**，取前 `code_len` 行。耗时 20.4 ms（固定开销，与 valid 长度无关）。

2) codec transformer：8 层 per-layer `qwen3_tts_tokenizer_12hz_p64_l{L}_together.axmodel` + `_post.axmodel`
- 组数 2（`--prefill_len 64 --kv_cache_len 127`）：

| group | 用途 | K_cache / V_cache | indices | input | mask |
|---|---|---|---|---|---|
| 0 | decode | `[1,127,1024]` | `[1,1]` uint32 | `[1,1,512]` | `[1,1,128]` |
| 1 | prefill（前 64 帧） | `[1,1,512]` | `[1,64]` uint32=0..63 | `[1,64,512]` | `[1,64,64]` |

- **hidden 512，kv_dim 1024，kv_cache_len 127，decode mask 列 128**；输出 `K_cache_out`/`V_cache_out` `[1,P,1024]`、`output [1,P,512]`。
- **注意力是滑窗 72**（不是全因果）：prefill 时 64<72 等价因果；decode 时第 pos 个 token 可见 `col ∈ [max(0,pos-71), pos-1]`，**外加最后一列（127）**；其余 -65536（见 C++ `run_transformer`）。
- 对 T>64 的音频：先 prefill 前 64 帧，然后 `pos=64..T-1` 用 group 0 逐帧 decode（8 层 ≈12.2 ms/帧）。
- post：`input [1,1,512]` bf16 → `output [1,1,1024]` bf16（每帧一次，0.4 ms）。

3) `speech_tokenizer_decode_post_t64.axmodel`（vocoder，single core）
- 输入：`hidden` fp32 **`[1,64,1024]`** → 输出：`wav` fp32 **`[1,1,122325]`**（注意不是 64×1920=122880）。
- 分块规则（照 C++）：`left_context=25`，每块 `capacity = 64 - context` 帧；feed = `hidden[start-context : end]` 补零到 64 帧；取输出的 `[context*1920 : context*1920+(end-start)*1920]`；最终 `wav_all.resize(T*1920)`。20 帧示例：38400 采样（1.6s@24k）。一次 t64 推理 ≈ **456 ms**（单核，主要成本）。

---

## 5. 性能实测（全部为板端 pyaxengine 实测，单位 ms）

> 环境：axllm demo 已停止（CMM 干净了，但 M5Stack `llm-*` 服务仍在）；除注明外为 warm 后平均。

| 组件 | 实测 | 折算 |
|---|---|---|
| talker 0.6B（H=1024） decode 单层 | 4.82–5.14 ms | 28 层 ≈ 135–144 ms/帧 |
| talker 0.6B prefill 单层（128 token，chunk1） | 8.37 ms | 28 层 ≈ 234 ms/128 token |
| **talker 1.7B（H=2048，Breeze 等价）decode 单层** | **6.69 ms**（首帧冷 292.8ms/28层） | **28 层 = 187.3 ms/帧** |
| talker 1.7B prefill 单层 chunk1 / chunk2 | 13.42 / 16.31 ms | 128 token=375.6ms；256 token=832.2ms |
| talker post | 0.7–0.9 ms | |
| cp decode 单层（0.6B） | 2.64 ms | 5 层 ≈ 13.2 ms/帧 |
| cp prefill 单层（64，仅 2 valid） | 3.87 ms | 5 层 ≈ 19 ms |
| cp post | 0.37 ms | |
| cp lm_head × 15 | 0.45 ms/个 | 6.8 ms/帧 |
| codec decode_pre（固定桶 325） | 20.4 ms | 每句固定 |
| codec transformer prefill 8 层 | 23.3 ms | 每句固定 |
| codec transformer decode 单层 | 1.52 ms | 8 层 ≈ 12.2 ms/帧（仅 T>64） |
| codec post | 0.40 ms | 每帧 |
| codec decode_post（t64 桶） | 456 ms | 64 帧/次；分块后 39 帧/次 |
| 模型加载 | talker28(1.7B)=9.2–9.4s；cp=0.7s；codec=0.8s | |

---

## 6. CMM / 系统内存预算

| 状态 | CMM used | CMM remain | 系统内存 |
|---|---|---|---|
| M5Stack `llm-*` 服务在跑 + axllm 在跑（默认） | 3200 MB | 2943 MB | used 870MB / total 1912MB |
| 停 axllm 后（只留 `llm-*`） | **991 MB** | **5152 MB** | available ≈ 970MB |
| + talker 1.7B 28 层 & post | 2941 MB（+1950） | 3202 MB | RSS 900MB（文件页居多，可回收） |
| + cp 0.6B（5 层+15 head） | 3067 MB（+76） | 3076 MB | |
| + codec decode 全套（pre/8 层/post/vocoder） | 3322 MB（+158） | 2821 MB | MemAvailable 941MB |

**Breeze 预估**（talker 几何=1.7B；cp 12 层 hidden 1024 kv_dim 256；codec 相同）：
- CMM：talker ≈1950 + cp（12 层 ≈211 + 15 lm_head ≈34 + post）+ codec 158 ≈ **2.4–2.5 GB**（6GB 分区内 remain ≈2.7GB）✅
- host RAM：python RSS 峰值 ≈ 0.9–1.1GB（其中 axmodel 文件 mmap 页可回收）；talker KV cache numpy 数组 = `28×2×kv_len×1024×2B`：**kv_len=2048 → 235MB；kv_len=512 → 59MB**。
- 建议：**Breeze talker build 用 `--kv_cache_len 512`**（41s 音频上限，足够 demo；省 176MB host RAM 且降低每帧 memmove 量），若坚持 2048 也能跑，但 1.9GB 系统内存下更紧张（有 2GB swap 兜底）。

**可行性结论**：纯 numpy+pyaxengine 可行；不要装 torch/transformers（板端无、也不需要）。每帧 talker decode 的 28 层都需要把整份 `[1,2048,1024]` K/V 从 host 拷进 axmodel 输入 buffer（每层 8.4MB×2/帧），这是 pyaxengine API 固有的，也是 6.69ms/层的一部分。

---

## 7. Breeze 端到端预估（task-6 基线）

| 阶段 | 单帧成本 |
|---|---|
| talker decode 28 层 + post | 187.3 + 0.8 = 188.1 ms |
| depth 12 层 + 15 lm_head + post | ≈ 12×2.64 + 6.8 + 0.4 ≈ 39 ms |
| codec（T≤64 一次性 prefill 摊销 / T>64 逐帧 12.2ms + post，vocoder 摊销 ~7–12ms/帧） | ≈ 11–13 ms |
| **合计 ≈ 238–240 ms/帧 → 音频 80ms/帧 → RTF ≈ 3.0** | |

含义：合成 5 秒音频（63 帧）≈ 15 s；合成 10 秒 ≈ 30 s。**不是实时流式**，但作为"尝试部署 + 音质验收 demo"完全可接受。后续若要把 RTF 压到 1 附近，唯一现实手段是权重更低比特量化（bf16→W8，层 decode 近似减半），属于下一阶段。

---

## 8. 与 `_work/board/RUNTIME_SPEC.md` 的差异 / 修正（重点）

1. **talker prefill group1 的 K/V 占位 shape = `[1,1,hidden]`**（1.7B 实测 `[1,1,2048]`；codec 也是 `[1,1,512]`=hidden），**不是 kv_dim**。group≥2 才是 `[1,128*(g-1),kv_dim]`。→ 运行时一律按 `get_inputs(g)` 动态取 shape，禁止硬编码。
2. **talker prefill `indices` 是 `[3,128]`（3 行内容相同）**，decode 才是 `[1,1]`；**cp / codec 的 prefill indices 是 `[1,64]`（1 行）**。三者不同，必须从 axmodel 读。
3. **mask 最后一列语义**：decode mask 列数 = `kv_len+1`，当前 token 用**最后一列**（talker 2048、cp 128、codec 127），不是 col=cur。spec 描述方向对，但"最后一列固定为 kv_len"要写死。
4. **cp 的 K_cache dim 实测 1024 = kv_dim（Qwen3-TTS cp 是 8 kv heads×128）**，不是 hidden；**Breeze cp 是 2 kv heads×128 = 256**，build 后要重新核对（这也说明 spec 里"cp decode hidden 1024"容易被误读为 cache 1024）。
5. **talker post 输出 2 个**（logits bf16 + output_norm bf16，需 `--ret_postnorm`）；**1.7B VoiceDesign 包没开 ret_postnorm，只有 logits** → Breeze 必须显式开。cp post **只有 output_norm**（没有 logits）。
6. **codec 细节**（spec 未写全）：decode_pre 输入 `[1,16,325]`（**带 batch 维，q-major**）、输出 `[1,325,512]` fp32；transformer **kv_cache_len=127**、kv_dim=1024、decode mask 列 128、**滑窗 72**；post `[1,1,512]→[1,1,1024]`；decode_post 输入 fp32 `[1,64,1024]`、输出 `[1,1,122325]`、left_context=25 / capacity=39、最终 resize 到 `T*1920`。
7. **cp lm_head 输出 2048**（0.6B 的 codebook 词表），Breeze 应为 2051；输入 fp32（不是 bf16）。
8. spec §5 的"前 2 帧禁 EOS"仍无参考实现确认（等 `_work/ref/GEN_LOGIC.md`），板端无法自证。
9. spec 说 "pyaxengine 文档要求 SDK>=2.18，确认 bf16" → **实测 2.12.0s 即可用 bf16，无需升级 SDK**。

---

## 9. 风险 / 注意事项

1. **不要随意 kill 长期占 NPU 的进程**：17:58 kill axllm 后 18:01 板子发生过一次干净重启（也可能是操作员触发；18:10 再 kill 后稳定 20min+）。判断是偶发；但**跑长任务请 nohup + 落盘 + 可续跑**，并避免在重启窗口丢结果。
2. **axllm（MiniCPM-OCR demo）占 2.2GB CMM + 几百 MB RAM**，跑 Breeze 前必须停：
   - 停：`pkill -x axllm`
   - 恢复：`/root/minicpm-ocr-v2/start_axllm.sh`（web UI 恢复：`/root/minicpm-ocr-v2/start_flask.sh`）。当前状态：**已停**（为 Breeze 让出 CMM）。
3. 板端 python 3.10 f-string 不支持反斜杠（写脚本注意）；pyaxengine 张量 dtype/shape 严格匹配；`np.zeros(...,dtype=ml_dtypes.bfloat16) - 65536.0` 会升成 float32，要 `np.full(...,-65536.0,dtype=bf16)`。
4. `talker post` logits 是 **bf16**，采样前转 fp32；EOS/禁 token 等逻辑参考 `_work/ref/GEN_LOGIC.md`。
5. talker/cp/codec 的 axmodel **不含 embedding 层**：音频/文本 embedding 都在 host（Breeze：16 组未投影表求和 2048 维 + depth 的 15 组投影表 1024 维，由 repack 导出）。
6. prefill 容量 640 token（5 组×128）是 build 参数决定的；Breeze 的 `[S0]<ins_bos>instruction<ins_eos>text` prompt 要控制在此之内（放不下就得加 `--last_kv_cache_len` 或多 chunk 配置）。
7. `decode_pre` 固定桶 325 帧（26s）→ 超过需分段（C++ 直接抛错）；`decode_post` t64 桶输出 122325 采样，注意按 C++ 截断/补齐，最终 `T*1920`。

---

## 10. 可复现资产

板端 `/root/breeze-tts/board_scripts/`（Mac 副本 `_work/board/scripts/`）：
- `probe_io.py` / `probe_all.py`：枚举 axmodel 所有 shape group 的 I/O + 单次延迟（talker/cp/codec）。
- `smoke_codec.py`：随机 16×T codes → decode_pre → 8 层 transformer（滑窗 mask）→ post → decode_post → 24k wav；参数 T。**已跑通**，产物 `/root/breeze-tts/codec_random_20f.wav`（1.6s）。
- `smoke_talker.py`：加载 28 层 talker（默认 1.7B 替身）+ post；多 chunk prefill、N 帧 decode、CMM/RSS 追踪。用法：
  `LD_LIBRARY_PATH=/soc/lib python3 board_scripts/smoke_talker.py /root/breeze-tts/qwen17b/talker 28 256 40`
- `smoke_all.py`：同时加载 talker(1.7B)+cp+codec 全组件，打印 CMM/RAM 预算。

复现要点：所有推理都要 `LD_LIBRARY_PATH=/soc/lib`（libax_engine.so 在 /soc/lib）。
日志：`/root/breeze-tts/logs/`（`probe_0b6b.log`、`smoke_talker17b_n64.log`、`smoke_talker17b_n256.log`）；Mac 副本 `_work/board/logs/`，wav 证据 `_work/board/artifacts/`。

---

## 附录 B（补充实测：axmodel 体积 / CMM / decode 分解，2026-11-11 18:40）

### B.1 体积（板上 `ls -l`）
| 文件 | 0.6B 包 | 1.7B 包（Breeze talker 等价） |
|---|---|---|
| talker 每层 `*_p128_l*_together.axmodel` | 21,777,175 B（21.78MB） | 60,967,575 B（60.97MB） |
| talker 28 层合计 | 609.8 MB | 1,707.1 MB |
| talker post | 3,446,575 B | 6,893,303 B |
| cp 每层（5 层） | 17,610,000 B 量级 | — |
| cp lm_head ×15 | 2,259,876 B/个 | — |
| codec `decode_pre_inproj` | 19,381,025 B | — |
| codec decoder 每层 ×8 | 4,244,000 B 量级 | — |
| codec decoder post | 0.59 MB | — |
| codec `decode_post_t64`（vocoder） | 92,519,852 B | — |

### B.2 CMM 实测（AXERA 0.6B 包 + 1.7B talker 代理）
| 加载内容 | CMM used 增量 |
|---|---|
| talker 28 层（1.7B 几何）+ post | **+1950 MB**（≈69.6MB/层，远大于文件 60.97MB/层） |
| talker 28 层（0.6B 几何）+ post（由 991→约1600 估算） | ≈+610 MB 文件 + 每层 K/V 输入 buffer |
| cp 0.6B 5 层 + 15 head + post | +126 MB |
| codec decode 全套（pre+8 层+post+vocoder） | +255 MB |
| 0.6B 包 talker+cp+codec 全加载（胶水运行） | 991 → 2258 MB（+1267 MB；文件合计 878MB） |

**发现**：CMM 占用 ≈ 文件体积 ×1.3–1.4，原因之一是**每个 per-layer session 在 CMM 里常驻一份该层的 K/V 输入 buffer**：kv_cache_len=2048、kv_dim=1024、bf16 → 每层 2×4.19MB=8.39MB，28 层 = **235MB**（仅 talker）。`--kv_cache_len 512` → 58.7MB（省 176MB CMM，同时 host KV 数组 235→59MB）。

### B.3 decode 187.3 ms/帧（1.7B talker，28 层）分解
微观基准（板上）：
- `np.copyto` 4.19MB：0.89ms → **host memcpy ≈4.7–4.9 GB/s**（pyaxengine 每层每 run 要 memmove K+V = 8.39MB ⇒ ≈**1.7 ms/层**，占 6.69ms 的 ~25%）
- `lm_head`（8.39MB fp32 权重、输入 4KB）run=0.44ms → **NPU 权重读 ≈19 GB/s**
- `cp_post`（极小模型）run=0.37ms → **纯 launch/同步开销下限 ≈0.37ms**
- talker post（12.6MB bf16 权重）run=0.67ms

据此 28 层 decode ≈ 187ms：
| 成分 | 估算 | 占比 |
|---|---|---|
| 权重流（28×60.97MB=1.71GB @≈19GB/s） | ≈90 ms | ~48% |
| host→CMM K/V memmove（28×8.39MB=235MB @4.9GB/s） | ≈48 ms | ~26% |
| cache flush/invalidate、mask/indices/input 拷贝、三核 RunSync 同步、Python 开销 | ≈49 ms | ~26% |

结论与建议：
- **KV 每帧 memmove 是主要可优化项之一**（~26%）：`kv_cache_len=512` 可把它从 48ms 降到 ~12ms（≈-19% 整帧时间，同时省 CMM/host RAM）。
- 权重流是硬瓶颈（~48%）：bf16/INT8 权重下无法再快；要接近实时只能更低比特量化（s4）或 C++ ax-llm 零拷贝运行时。
- per-layer 固定开销 ≈0.4ms（launch 下限）+ memmove；把 12 层 depth 合并成单 step 可省 ~11×开销×15 步。

---

## 附录 C（tiny 单测模型 + 预取实测，2026-11-11 18:40）

### C.1 tiny-qwen3（talker 1 层单测，generic Qwen3 build）完整 I/O
```
shape_count=2（decode + 1 个 prefill 组，prefill_len=64）
g0 decode: K_cache [1,128,1024] bf16 | V_cache [1,128,1024] | indices [1,1] uint32 | input [1,1,2048] bf16 | mask [1,1,129] bf16
           out: K_cache_out [1,1,1024] / V_cache_out [1,1,1024] / output [1,1,2048]
g1 prefill: K_cache [1,1,2048]（占位=hidden，不是 kv_dim）| V_cache [1,1,2048] | indices [1,64]（**1 行**）| input [1,64,2048] | mask [1,64,64]
post: input [1,1,2048] → output [1,1,**2052**] bf16 + output_norm [1,1,**2048**] bf16   ← Breeze talker 目标形状
```
### C.2 tiny-llama（depth 2 层单测，generic Llama build）完整 I/O
```
shape_count=2，两层 I/O 相同
g0 decode: K_cache [1,32,**256**] bf16（kv_dim=256 ✓）| V_cache [1,32,256] | indices [1,1] | input [1,1,1024] | mask [1,1,33]
           out: K_cache_out [1,1,256] / V_cache_out [1,1,256] / output [1,1,1024]
g1 prefill: K_cache [1,1,1024]（占位=hidden）| V_cache [1,1,1024] | indices [1,16]（1 行）| input [1,16,1024] | mask [1,16,16]
post: input [1,1,1024] → output [1,1,2051]（dummy lm_head，**忽略**）+ output_norm [1,1,1024] bf16
```
结论：Breeze 的 depth decoder 用通用 llama build 可行（2 kv heads 被接受，kv_dim 256）；post 取 output_norm 即可。

### C.3 mask 语义自洽验证（决定性）
prefill(P) 的 hidden[k] vs 「prefill 前 k token + decode 第 k token」的 hidden[k]：
- tiny-qwen3：convA（cols<k 置 0 + **最后一列=kv_len** 置 0）cos=**1.000000**；convB（cols≤k）cos=0.998398
- tiny-llama：convA cos=**1.000000**；convB cos=0.999872
→ AXERA「当前 token 占 decode mask 最后一列、列数=kv_len+1」语义对 Breeze talker/depth 均成立。
`indices` 行数：generic build = **1 行**；AXERA `qwen3_tts_talker` = 3 行 → runtime 按 axmodel 实际 shape 填，不写死。

### C.4 embedding 表页缓存预热（板端实测）
| 状态 | 16 组随机查表（1 帧音频的帧 embedding） |
|---|---|
| 冷态（drop_caches 后，134MB 表） | **13.0 ms/帧** |
| 启动时顺序 touch 134MB 表 | **399 ms 一次性** |
| 预热后 | **1.1 ms/帧**（省 12~16ms/帧） |

### C.5 Breeze 几何 talker-only 循环（1.7B 替身 + 真实 16 组表，20 帧）
- prefill L=8（1×128 chunk，28 层）：417 ms
- decode 稳态：**198.5 ms/帧**（其中 sum16 embedding 18.5ms→预热后 1.1ms、post 0.9ms）
- talker-only RTF ≈ 2.5（80ms/帧）
- CMM：talker(1.7B) + 表加载 ≈ 3051MB used；RSS 944MB；MemAvail 1146MB

---

## 附录 D（最终交付摘要，2026-11-11 19:40）

- **板端全链路已跑通**（Breeze TTS 2：talker 28 层 NPU + depth 12 层×15 步 NPU + codec NPU），采样模式 ASR 逐字正确。详见 `RUNTIME_REPORT.md`。
- 关键实测：talker decode **138.5ms/帧**（kv512）、depth **560ms/帧**（INT8 fast）/ **1375ms/帧**（hp FP32）、codec 510ms（64帧）/1249ms（84帧）；端到端 **RTF 11.3×（短句）/10.9×（长句）**；CMM 3471MB used（fast）/4394MB（hp）。
- codec 复用 AXERA 产物精确成立：同 golden codes → 长度一致、幅度谱相关 0.9996、包络 0.84、尾部 RMS 0.947（无 tail bug）；波形相位差异来自 vocoder 实现差异，不影响内容（ASR 一致）。
- 15 个 depth lm_head axmodel 输出被量化损坏（~150 distinct values、clamp ±5）→ runtime 改用 CPU fp32 matmul（cos 0.9998）；建议下一版 U16 重编。
- textenc 未上线（Pulsar2 AxRoPE backend bug + INT8 精度不达标），demo 用 golden `text_embeds` 注入，**除 textenc 外全部 NPU 化**；staged loading（textenc→释放 CMM→加载 talker/depth）已实现并实测 pyaxengine `del` 可释放 CMM。
