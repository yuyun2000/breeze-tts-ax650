#!/usr/bin/env python3
"""Build Pulsar2 calibration + validation data for the Breeze text encoder.

Real business-like prompts (Breeze Voice-Design template `[S0]<ins_bos>{instruction}<ins_eos>{text}`)
are tokenized with the *Breeze* tokenizer, the CPU embedding lookup turns them into
`inputs_embeds`, and padding to the fixed bucket is recorded in `attention_mask`.

Produces:
  calib/inputs_embeds.tar.gz     0000.npy.. float32 [1,L,1152]  (>= --n-calib samples)
  calib/attention_mask.tar.gz    0000.npy.. float32 [1,L]       (same order)
  val/val.npz                    inputs_embeds [N,L,1152], attention_mask [N,L], lengths [N]
  calib_meta.json                prompts, token lengths, template used

Run on 20.38 with the project venv (only numpy + tokenizers needed for this step).
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path

import numpy as np

from text_embed_cpu import BreezeEmbeddingLUT

INSTRUCTION_BOS = "<ins_bos>"
INSTRUCTION_EOS = "<ins_eos>"

LONG_TEXTS_ZH = [
    "亲爱的听众朋友们，大家晚上好，欢迎收听今天的晚间新闻。今天的主要内容有：本市地铁七号线将于下月正式开通运营，沿线共设置二十三个车站，将有效缓解东西向通勤压力；此外，市气象台发布暴雨蓝色预警，提醒市民朋友减少夜间外出，注意交通安全。",
    "人工智能技术正在快速发展，从语音识别到图像生成，从自动驾驶到智能医疗，越来越多的应用场景正在被重新定义。与此同时，如何保证数据安全、如何避免算法偏见、如何建立合理的监管框架，也成为了全社会共同关心的重要议题。",
    "在今天的会议上，我们回顾了过去一个季度的项目进展，讨论了下一阶段的重点任务。团队在模型压缩和推理加速方面取得了明显突破，端侧延迟下降了百分之四十，功耗也显著降低，这为我们后续的产品落地打下了坚实的基础。",
    "这本书讲述了作者在山西农村度过的童年时光。院子里有一棵老槐树，夏天的时候，树下总是坐满了乘凉的人。奶奶会在树下摇着蒲扇，讲那些听了无数遍的故事，而孩子们则追着萤火虫在田埂上跑来跑去，一直玩到很晚才肯回家。",
    "请注意，本次列车即将到达终点站，请乘客们带好自己的随身物品，依次从右侧车门下车。感谢您乘坐本次列车，祝您旅途愉快。如果您需要换乘其他线路，请按照站台上的指示标识前往换乘通道，谢谢配合。",
    "随着新能源汽车的普及，充电基础设施的建设也变得愈发重要。目前全市已建成公共充电桩超过三万个，覆盖主要商圈、居民小区和高速公路服务区。下一步，相关部门还将继续优化布局，提升充电效率，让市民出行更加便利。",
    "这道菜的做法其实并不复杂，关键在于火候的控制。先把五花肉切成小块，冷水下锅焯去血水，捞出沥干。锅中放少许油，加入冰糖小火炒出糖色，再放入肉块翻炒上色，加入葱姜八角和生抽老抽，倒入热水没过肉块，小火慢炖四十分钟即可。",
    "在过去的十年里，我们见证了移动互联网从兴起到繁荣的整个过程。从最初的简单网页浏览，到如今随时随地的视频通话、移动支付和在线协作，技术的进步深刻地改变了每一个人的日常生活，也创造了无数新的商业机会。",
]
LONG_TEXTS_EN = [
    "Ladies and gentlemen, welcome aboard flight CA1832 from Beijing to Shanghai. The flight time will be about two hours and fifteen minutes. We expect to encounter some light turbulence during the flight, so please keep your seat belts fastened whenever the sign is on, and let us know if there is anything we can do to make your journey more comfortable.",
    "In recent years, the rapid development of artificial intelligence has brought both opportunities and challenges. On the one hand, intelligent systems can help doctors detect diseases earlier, assist teachers in personalizing education, and allow engineers to design safer products. On the other hand, we must carefully consider privacy, fairness and accountability when these systems are deployed in the real world.",
    "The museum is open from nine in the morning until six in the evening, and it is closed on Mondays. Student tickets are available at half price with a valid identification card. Guided tours start every hour on the hour, and the exhibition on the second floor features more than three hundred paintings from the nineteenth century.",
    "After the meeting, we walked along the river and talked about the project schedule. The sunset turned the sky orange, and a cool breeze came from the water. Everyone agreed that the next milestone should be finished before the end of the month, and we would review the plan again on Friday afternoon.",
    "Our team has successfully deployed the model on the edge device, and the result exceeded our expectations. The inference latency dropped below fifty milliseconds, while the accuracy remained almost unchanged. We are now working on further improving memory efficiency, which will allow more models to run on the same hardware platform.",
]

INSTRUCTIONS_ZH = [
    "用温柔平静的语气说",
    "用兴奋的语气朗读",
    "请用低沉的声音慢慢说",
    "用标准的播音腔朗读",
    "用轻快的语气说",
    "用严肃认真的语气说",
]
INSTRUCTIONS_EN = [
    "Say it in a calm and gentle voice.",
    "Read it in an excited tone.",
    "Speak slowly with a low voice.",
]

TEXTS_ZH = [
    "今天天气真不错，我们一起去公园散步吧。",
    "欢迎使用 Breeze 语音合成系统，希望你喜欢这个声音。",
    "会议将在下午三点准时开始，请大家提前十分钟到场。",
    "人工智能正在改变我们的生活方式和工作方式。",
    "请把这份文件复制三份，分别送到三个部门。",
    "这家餐厅的红烧肉做得特别地道，下次带你来尝尝。",
    "前方路口右转，然后直行两百米就到了。",
    "春天的西湖边，柳树发芽，游客如织。",
    "他每天早上六点起床，先跑步再吃早餐。",
    "系统检测到异常流量，请及时联系管理员处理。",
    "小朋友，过马路的时候一定要注意看红绿灯。",
    "这款芯片在边缘计算场景下表现出色，功耗也很低。",
    "谢谢你昨天的帮助，要不是你我真不知道怎么办。",
    "请保持安静，图书馆内禁止大声喧哗。",
    "我最近在学习做菜，已经会做几道家常菜了。",
    "天气预报说明天有雨，出门记得带伞。",
    "这个项目的进度比预期提前了两周完成。",
    "海边的日落非常漂亮，天空被染成了橘红色。",
    "他说话的语气很温和，让人感觉特别亲切。",
    "请在收到邮件后尽快回复确认。",
    "我们下午去超市买点水果和蔬菜吧。",
    "这台设备的噪音有点大，需要检查一下风扇。",
    "请把空调温度调到二十六度，谢谢。",
    "这条路晚上路灯很暗，走路要小心一点。",
    "他昨天在图书馆借了三本关于历史的书。",
    "公司楼下新开了一家咖啡店，味道还不错。",
    "我们计划下个月去云南旅行，路线还在商量。",
    "请把这批货物在明天上午之前送到仓库。",
    "孩子的作业已经写完了，可以看一会儿动画片。",
    "这道数学题我想了很久，还是没找到思路。",
    "周末的早晨，阳光透过窗帘照进房间。",
    "地铁上人很多，大家都很安静地看着手机。",
    "这款手机拍照效果很好，夜景也很清晰。",
    "医生说要多喝水、多休息，少熬夜。",
    "明天上午十点在三楼会议室开会，请准时参加。",
    "窗外下起了小雨，远处的山若隐若现。",
    "他把最后一页文件签好字，长长地松了口气。",
    "请告诉我最近的火车站在哪个方向。",
    "这只小猫特别黏人，一直跟着我走来走去。",
    "最后一班公交车在晚上十一点发车。",
]

TEXTS_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the Breeze text to speech system.",
    "Please make sure the door is locked before you leave.",
    "The meeting has been rescheduled to next Tuesday.",
    "Artificial intelligence is transforming the way we work.",
    "I would like a cup of coffee with a little milk, please.",
    "The weather forecast says it will rain tomorrow afternoon.",
    "Our team has successfully deployed the model on the edge device.",
    "She opened the window and let the fresh air in.",
    "Thank you for your patience and understanding.",
    "Could you please turn the volume down a little?",
    "The library closes at nine in the evening.",
    "We need to finish the report before Friday.",
    "He walked slowly along the river and watched the sunset.",
    "Please send me the updated schedule when you have time.",
]


def build_prompts() -> list[dict]:
    """Voice-Design style prompts = speaker tag + instruction segment + text."""
    prompts: list[dict] = []
    for i, text in enumerate(TEXTS_ZH):
        ins = INSTRUCTIONS_ZH[(i * 3) % len(INSTRUCTIONS_ZH)]
        prompts.append({"text": text, "instruction": ins, "lang": "zh"})
    for i, text in enumerate(TEXTS_EN):
        ins = INSTRUCTIONS_EN[(i * 2) % len(INSTRUCTIONS_EN)]
        prompts.append({"text": text, "instruction": ins, "lang": "en"})
    for i, text in enumerate(LONG_TEXTS_ZH):
        ins = INSTRUCTIONS_ZH[i % len(INSTRUCTIONS_ZH)]
        prompts.append({"text": text, "instruction": ins, "lang": "zh", "long": True})
    for i, text in enumerate(LONG_TEXTS_EN):
        ins = INSTRUCTIONS_EN[i % len(INSTRUCTIONS_EN)]
        prompts.append({"text": text, "instruction": ins, "lang": "en", "long": True})
    return prompts


def render_prompt(prompt: dict, speaker: str = "S0") -> str:
    return f"[{speaker}]{INSTRUCTION_BOS}{prompt['instruction']}{INSTRUCTION_EOS}{prompt['text']}"


def pack_tar(files: list[Path], tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path("/home/m5stack/work/breeze-tts/data/breeze-tts-2"),
                    help="Breeze HF dir (tokenizer.json)")
    ap.add_argument("--weights", type=Path,
                    default=Path("/home/m5stack/work/breeze-tts/textenc/weights"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/m5stack/work/breeze-tts/textenc/calib"))
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tokenizers import Tokenizer  # noqa: PLC0415  (only needed here)

    tok_path = args.repo / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"missing Breeze tokenizer: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    lut = BreezeEmbeddingLUT(args.weights)

    prompts = build_prompts()
    rng = random.Random(args.seed)
    rng.shuffle(prompts)

    # tokenize everything first, then pick an explicit long/short mix for calibration
    # (prompts are ~20-30 tokens when short, but real requests can be 60-120 tokens;
    #   both regimes must be represented in the calibration set).
    items = []
    for prompt in prompts:
        prompt_text = render_prompt(prompt)
        ids = tokenizer.encode(prompt_text).ids
        if not ids or len(ids) > args.seq_len:
            continue
        items.append((prompt, prompt_text, ids))

    long_items = [x for x in items if len(x[2]) > 48]
    short_items = [x for x in items if len(x[2]) <= 48]
    n_long_calib = min(max(0, len(long_items) - 2), max(8, args.n_calib // 3))  # keep 2 long for val
    n_val_long = min(max(0, len(long_items) - n_long_calib), 2)
    n_val_short = max(0, args.n_val - n_val_long)
    calib_items = long_items[:n_long_calib] + short_items[: args.n_calib - n_long_calib]
    val_items = short_items[args.n_calib - n_long_calib: args.n_calib - n_long_calib + n_val_short] \
        + long_items[n_long_calib: n_long_calib + n_val_long]

    calib_embeds, calib_mask, val_embeds, val_mask = [], [], [], []
    records, lengths = [], []
    for prompt, prompt_text, ids in calib_items:
        embeds, mask = lut.build_graph_inputs(ids, args.seq_len)
        calib_embeds.append(embeds)
        calib_mask.append(mask)
        records.append({
            "prompt": prompt_text, "lang": prompt["lang"],
            "instruction": prompt["instruction"], "text": prompt["text"],
            "n_tokens": len(ids), "token_ids": ids,
        })
        lengths.append(len(ids))
    val_records = []
    for prompt, prompt_text, ids in val_items:
        embeds, mask = lut.build_graph_inputs(ids, args.seq_len)
        val_embeds.append(embeds)
        val_mask.append(mask)
        val_records.append({"prompt": prompt_text, "token_ids": ids, "n_tokens": len(ids)})

    if len(calib_embeds) < 3 or len(val_embeds) < 1:
        raise SystemExit(f"not enough usable prompts: calib={len(calib_embeds)} val={len(val_embeds)}; "
                         "increase text corpus or seq-len")

    out = args.out
    (out / "calib_inputs_embeds").mkdir(parents=True, exist_ok=True)
    (out / "calib_attention_mask").mkdir(parents=True, exist_ok=True)
    files_embeds, files_mask = [], []
    for i, (embeds, mask) in enumerate(zip(calib_embeds, calib_mask)):
        p = out / "calib_inputs_embeds" / f"{i:04d}.npy"
        np.save(p, embeds.astype(np.float32))
        files_embeds.append(p)
        q = out / "calib_attention_mask" / f"{i:04d}.npy"
        np.save(q, mask.astype(np.float32))
        files_mask.append(q)
    pack_tar(files_embeds, out / "inputs_embeds.tar.gz")
    pack_tar(files_mask, out / "attention_mask.tar.gz")

    (out / "val").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "val" / "val.npz",
        inputs_embeds=np.concatenate(val_embeds, axis=0).astype(np.float32),  # [N,L,H]
        attention_mask=np.concatenate(val_mask, axis=0).astype(np.float32),   # [N,L]
        lengths=np.array([int(m.sum()) for m in val_mask], dtype=np.int64),
    )
    # single-sample bundle for board-side smoke tests (real prompt, not random)
    smoke = {
        "inputs_embeds": val_embeds[0].astype(np.float32),   # [1,L,1152]
        "attention_mask": val_mask[0].astype(np.float32),    # [1,L]
    }
    (out / "val").mkdir(parents=True, exist_ok=True)
    np.savez(out / "val" / "smoke_sample.npz", **smoke)
    meta = {
        "seq_len": args.seq_len,
        "n_calib": len(calib_embeds),
        "n_val": len(val_embeds),
        "token_lengths": lengths,
        "token_length_min": int(min(lengths)),
        "token_length_max": int(max(lengths)),
        "template": "[S0]<ins_bos>{instruction}<ins_eos>{text}",
        "calib_records": records,
    }
    (out / "val" / "smoke_sample.json").write_text(
        json.dumps({"prompt": val_records[0]["prompt"],
                    "token_ids": val_records[0]["token_ids"],
                    "tensor_names": {"inputs_embeds": [1, args.seq_len, int(lut.hidden_size)],
                                     "attention_mask": [1, args.seq_len]},
                    "output": "text_embeds [1,128,2048] (chunk2) / [1,128,1152] (chunk1)"},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "calib_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[calib] {out}  calib={len(calib_embeds)} val={len(val_embeds)} "
          f"tokens {min(lengths)}..{max(lengths)} of {args.seq_len}")


if __name__ == "__main__":
    main()
