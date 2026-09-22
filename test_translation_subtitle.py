# coding=utf-8
"""
test_translation_subtitle.py - 字幕翻译的离线单测（不加载任何模型）

覆盖：
  [1] 说话人前缀剥离
  [2] 不完整句子的合并分组（直到遇到句末标点）
  [3] 说话人切换时强制分组
  [4] 合并块时长 = 参与各行时长之和
  [5] 译文按标点拆行，整体时长与原来一致
  [6] 译文拆行后每行不超过 max_line_duration
  [7] 提示词模板渲染（单段 / 多段）
  [8] 批量编号回复的解析（正常 / 缺段回退）
  [9] 译文清洗与字幕片段回填（含说话人标记）
  [10] SRT 解析 / 序列化往返，以及「只翻译已有字幕文件」的时长守恒

运行：
    python test_translation_subtitle.py
"""
import os
import sys
import tempfile
from pathlib import Path

from qwen_asr_gguf.inference.schema import SubtitleSegment
from qwen_asr_gguf.inference.subtitle import (
    build_translated_segments,
    group_segments_for_translation,
    load_srt_segments,
    parse_srt,
    segments_to_srt,
    split_speaker_prefix,
    split_translated_text,
)
from qwen_asr_gguf.inference.translator import (
    DEFAULT_TRANSLATION_PROMPT,
    build_translation_prompt,
    clean_translation,
    parse_numbered_translations,
)
from qwen_asr_gguf.pipeline import is_subtitle_file, split_files_by_kind

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def seg(index: int, start: float, end: float, text: str, speaker: str = "") -> SubtitleSegment:
    return SubtitleSegment(index=index, start_time=start, end_time=end, text=text, speaker=speaker)


# --------------------------------------------------------------------------- #
print("[1] 说话人前缀剥离")
spk, body = split_speaker_prefix("[spk0] Hello there.")
check("剥离 spk 标记", spk == "spk0" and body == "Hello there.", f"{spk!r} {body!r}")
spk, body = split_speaker_prefix("没有标记")
check("无标记时原样返回", spk == "" and body == "没有标记", f"{spk!r} {body!r}")

# --------------------------------------------------------------------------- #
print("\n[2] 不完整句子的合并分组")
segments = [
    seg(1, 0.0, 2.5, "Hello, I want to", "spk0"),
    seg(2, 2.5, 5.0, "ask you something.", "spk0"),
    seg(3, 5.0, 8.0, "Sure, go ahead.", "spk0"),
]
units = group_segments_for_translation(segments)
check("合并成 2 个翻译单元", len(units) == 2, f"实际 {len(units)}")
check(
    "第一块把两行并起来了",
    len(units[0].segments) == 2
    and units[0].text == "Hello, I want to ask you something.",
    repr(units[0].text),
)
check("第一块以句末标点闭合", units[0].complete)
check("第二块是独立完整句", units[1].text == "Sure, go ahead." and len(units[1].segments) == 1)

# --------------------------------------------------------------------------- #
print("\n[3] 说话人切换时强制分组")
segments = [
    seg(1, 0.0, 3.0, "I was", "spk0"),
    seg(2, 3.0, 6.0, "thinking about it.", "spk1"),
]
units = group_segments_for_translation(segments)
check("跨说话人不合并", len(units) == 2, f"实际 {len(units)}")
check("各自保留说话人", units[0].speaker == "spk0" and units[1].speaker == "spk1")

# 文本里带前缀、字段为空的情况也要能识别
segments = [
    seg(1, 0.0, 3.0, "[spk0] I was"),
    seg(2, 3.0, 6.0, "[spk1] thinking about it."),
]
units = group_segments_for_translation(segments)
check("正文里的 [spk] 前缀也能识别说话人切换", len(units) == 2, f"实际 {len(units)}")
check("正文前缀被剥离", units[0].text == "I was", repr(units[0].text))

# --------------------------------------------------------------------------- #
print("\n[4] 合并块时长 = 各行时长之和")
segments = [
    seg(1, 0.0, 2.0, "A long sentence that is not", "spk0"),
    seg(2, 4.0, 5.5, "finished yet.", "spk0"),
]
units = group_segments_for_translation(segments)
check("时长合计为 3.5 秒（不是区间 5.5 秒）", abs(units[0].duration - 3.5) < 1e-6, f"{units[0].duration}")

# --------------------------------------------------------------------------- #
print("\n[5] 译文拆行：整体时长不变")
rows = split_translated_text(
    "你好，我是第一个。第二句在这里。", 10.0, start_time=2.0, max_duration=6.0, max_chars=24
)
total = sum(e - s for _, s, e in rows)
check("拆出的行数 > 1", len(rows) > 1, f"{len(rows)}")
check("整体时长恒等于 10.0", abs(total - 10.0) < 1e-6, f"{total}")
check("首行起点正确", abs(rows[0][1] - 2.0) < 1e-6, f"{rows[0][1]}")
check("末行终点正确", abs(rows[-1][2] - 12.0) < 1e-6, f"{rows[-1][2]}")
check("时间轴单调不重叠", all(rows[i][2] <= rows[i + 1][1] + 1e-6 for i in range(len(rows) - 1)))
check("每行都不超过 6 秒", all((e - s) <= 6.0 + 1e-6 for _, s, e in rows))

# 回归：曾经按标点「无脑拆」，把 1.36 秒的短译文拆成 3 行（末行只剩 0.16 秒）
rows = split_translated_text("哦，你好！我都没注意到你在这里。", 1.36)
check("1.36 秒的短译文保持一行", len(rows) == 1, [r[0] for r in rows])
check("该行时长等于原时长", abs((rows[0][2] - rows[0][1]) - 1.36) < 1e-6, f"{rows[0][2] - rows[0][1]}")
check("短译文内容完整", rows[0][0] == "哦，你好！我都没注意到你在这里。", rows[0][0])

rows = split_translated_text("好的，请说。", 3.0)
check("带标点的短句不拆", len(rows) == 1, [r[0] for r in rows])

rows = split_translated_text("嗯，差别其实没那么大。", 2.08)
check("2 秒短句不拆", len(rows) == 1, [r[0] for r in rows])

# --------------------------------------------------------------------------- #
print("\n[6] 译文拆行：每行不超过 6 秒")
long_text = (
    "这是一句非常长的话它完全没有句号只是一直在说然后继续说下去也不停下来"
    "总之就是拼了命地要把时间撑满最后才肯结束掉"
)
rows = split_translated_text(long_text, 40.0, max_duration=6.0, max_chars=14)
check("确实发生了拆分", len(rows) >= 6, f"{len(rows)}")
check(
    "所有行都 <= 6 秒",
    all((e - s) <= 6.0 + 1e-6 for _, s, e in rows),
    f"最长 {max(e - s for _, s, e in rows):.3f}",
)
check("整体时长仍为 40.0", abs(sum(e - s for _, s, e in rows) - 40.0) < 1e-6)
check("拆出的文字没有丢字", "".join(t for t, _, _ in rows) == long_text)

# 单行短句应保持一行
rows = split_translated_text("好的。", 1.2)
check("短句不拆", len(rows) == 1 and abs(rows[0][2] - rows[0][1] - 1.2) < 1e-6)

# --------------------------------------------------------------------------- #
print("\n[7] 提示词渲染")
system, user = build_translation_prompt(DEFAULT_TRANSLATION_PROMPT, ["Hello world."])
check("单段 user 就是原文", user == "Hello world.", repr(user))
check("system 含翻译规则", "专业字幕翻译" in system and "英译中" in system)
check("system 不含占位符", "{text}" not in system and "{text}" not in user)

system, user = build_translation_prompt(DEFAULT_TRANSLATION_PROMPT, ["A", "B", "C"])
check("多段追加格式要求", "3 段" in system and "序号" in system, system[-60:])
check("多段 user 带编号", user.startswith("1. A") and "3. C" in user, repr(user))

system, user = build_translation_prompt("翻译成日语：{text}", ["こんにちは"])
check("自定义模板可用", system == "翻译成日语：" and user == "こんにちは", f"{system!r} {user!r}")

# --------------------------------------------------------------------------- #
print("\n[8] 批量编号回复解析")
ok = parse_numbered_translations("1. 你好\n2. 世界\n3. 再见", 3)
check("正常解析", ok == ["你好", "世界", "再见"], repr(ok))
ok = parse_numbered_translations("1. 你好\n2. 世界", 3)
check("缺段时返回 None（触发逐条回退）", ok is None, repr(ok))
ok = parse_numbered_translations("1. 第一行\n续写内容\n2. 第二行", 2)
check("续行并入上一段", ok is not None and ok[0] == "第一行续写内容", repr(ok))
ok = parse_numbered_translations("好的，这是译文", 1)
check("无编号的单段回复也能兜住", ok == ["好的，这是译文"], repr(ok))

# --------------------------------------------------------------------------- #
print("\n[9] 译文清洗与回填")
check("去掉引号", clean_translation('"你好"') == "你好", clean_translation('"你好"'))
check("去掉前缀", clean_translation("译文：你好") == "你好", clean_translation("译文：你好"))
check("去掉 markdown", clean_translation("**你好**") == "你好", clean_translation("**你好**"))

units = group_segments_for_translation([
    seg(1, 0.0, 2.0, "Hello, I am", "spk0"),
    seg(2, 2.0, 4.0, "very glad to meet you.", "spk0"),
])
out = build_translated_segments(
    units, ["你好，见到你很高兴，真的非常开心。"], with_speaker=True, max_duration=6.0, max_chars=24
)
check("回填后说话人标记保留", all(s.speaker == "spk0" for s in out), [s.speaker for s in out])
check("序号连续", [s.index for s in out] == list(range(1, len(out) + 1)))
check("整体时长与合并块一致", abs((out[-1].end_time - out[0].start_time) - units[0].duration) < 1e-6)

# 译文为空时回退原文（且保持原有的行结构，不按译文规则拆分）
out = build_translated_segments(units, [""], with_speaker=True)
check(
    "译文为空时回退原文且保留原行",
    len(out) == 2 and out[0].text == "Hello, I am" and "very glad" in out[1].text,
    [s.text for s in out],
)
check("回退时说话人标记仍在", all(s.speaker == "spk0" for s in out))

# --------------------------------------------------------------------------- #
print("\n[10] SRT 解析 / 序列化与「只翻译字幕文件」")

DEMO_SRT = """1
00:00:06,720 --> 00:00:08,480
[spk1] Hello. Hello.

2
00:00:08,480 --> 00:00:09,840
[spk0] Oh, hello! I didn't know
you were there.

3
00:00:09,920 --> 00:00:09,920
[spk0] Neither did I.
"""

parsed = parse_srt(DEMO_SRT)
check("解析出 3 条", len(parsed) == 3, len(parsed))
check("说话人前缀被拆进 speaker 字段",
      [s.speaker for s in parsed] == ["spk1", "spk0", "spk0"], [s.speaker for s in parsed])
check("正文不含前缀且多行正文被合并为一行",
      parsed[0].text == "Hello. Hello." and parsed[1].text == "Oh, hello! I didn't know you were there.",
      [s.text for s in parsed])
check("时间轴解析正确", abs(parsed[0].start_time - 6.72) < 1e-6 and abs(parsed[0].end_time - 8.48) < 1e-6,
      f"{parsed[0].start_time} {parsed[0].end_time}")
check("零长字幕被补齐为正时长", parsed[2].end_time > parsed[2].start_time)

# 往返：segments -> SRT -> segments 必须保持正文 / 说话人 / 时间轴
round_trip = parse_srt(segments_to_srt(parsed, with_speaker=True))
check("往返后条数一致", len(round_trip) == len(parsed), len(round_trip))
check("往返后说话人与正文一致",
      [(s.speaker, s.text) for s in round_trip] == [(s.speaker, s.text) for s in parsed],
      [(s.speaker, s.text) for s in round_trip])
check("往返后时间轴一致",
      all(abs(a.start_time - b.start_time) < 0.05 for a, b in zip(round_trip, parsed)))

# 不规范格式的兜底解析（缺序号 + 句点做小数点）
fallback = parse_srt("00:00:01.000 --> 00:00:02.500\n你好\n\n00:00:03,000 --> 00:00:04,000\n世界\n")
check("兜底解析容忍缺序号 / 句点小数点",
      len(fallback) == 2 and fallback[0].text == "你好" and abs(fallback[1].start_time - 3.0) < 1e-6,
      [(s.text, s.start_time) for s in fallback])

# 文件读取 + 编码兼容（GBK 字幕是国内常见情况）
with tempfile.TemporaryDirectory() as tmp:
    p_utf8 = Path(tmp) / "utf8.srt"
    p_gbk = Path(tmp) / "gbk.srt"
    p_plain = Path(tmp) / "plain.srt"
    p_utf8.write_text(DEMO_SRT, encoding="utf-8")
    p_gbk.write_text(DEMO_SRT.replace("Hello", "你好呀"), encoding="gb18030")
    p_plain.write_text("1\n00:00:01,000 --> 00:00:02,000\n没有说话人标记\n", encoding="utf-8")

    segs_u, has_u = load_srt_segments(p_utf8)
    segs_g, _ = load_srt_segments(p_gbk)
    segs_p, has_p = load_srt_segments(p_plain)
    check("UTF-8 字幕读取正常", len(segs_u) == 3 and has_u, len(segs_u))
    check("GB18030 字幕读取不报错且内容正确",
          len(segs_g) == 3 and segs_g[0].text.startswith("你好呀"), [s.text for s in segs_g])
    check("无说话人标记时 has_speaker=False", has_p is False and len(segs_p) == 1, has_p)

# 只翻译模式：合成译文后整体时长必须与输入字幕各行时长之和一致
units = group_segments_for_translation(parsed)
source_total = sum(max(0.0, s.end_time - s.start_time) for s in parsed)
check("分组后总时长守恒", abs(sum(u.duration for u in units) - source_total) < 1e-6,
      f"{sum(u.duration for u in units)} vs {source_total}")

zh = ["你好。你好。", "哦，你好！我没想到你在这里。", "我也没这么做。"]
translated = build_translated_segments(
    units, zh, with_speaker=True, max_duration=6.0, max_chars=24
)
check("译文回填条数合理", 3 <= len(translated) <= 6, len(translated))
check("译文从输入字幕的起点开始", abs(translated[0].start_time - parsed[0].start_time) < 1e-6)
check("译文不超出输入字幕的时间范围",
      translated[-1].end_time <= parsed[-1].end_time + 1e-6,
      f"{translated[-1].end_time} vs {parsed[-1].end_time}")
# 逐块校验时长守恒：每块的译文时长之和 == 该块各行原始时长之和
per_unit_ok = True
per_unit_detail = []
for i, u in enumerate(units):
    rows = build_translated_segments([u], [zh[i]], with_speaker=True, max_duration=6.0, max_chars=24)
    got = sum(s.duration for s in rows)
    per_unit_detail.append((round(got, 6), round(u.duration, 6)))
    if abs(got - u.duration) > 1e-6:
        per_unit_ok = False
check("每块译文时长与其原始各行时长之和一致", per_unit_ok, per_unit_detail)
check("译文每行不超过 6 秒", all(s.duration <= 6.0 + 1e-6 for s in translated),
      [round(s.duration, 3) for s in translated])
check("译文保留说话人标记",
      all(s.speaker in ("spk0", "spk1") for s in translated),
      [s.speaker for s in translated])
# 输入分流
check("is_subtitle_file 识别 .srt / 大小写",
      is_subtitle_file("a.srt") and is_subtitle_file("b.SRT") and not is_subtitle_file("c.mp4"))
media, subs = split_files_by_kind(["a.mp4", "b.srt", "c.wav", "d.SRT"])
check("输入分流正确", media == ["a.mp4", "c.wav"] and subs == ["b.srt", "d.SRT"], (media, subs))

# --------------------------------------------------------------------------- #
print(f"\n{'=' * 52}")
print(f"通过 {PASS} 项，失败 {FAIL} 项")
sys.exit(1 if FAIL else 0)
