# coding=utf-8
"""
test_subtitle_speaker.py - 字幕断句 / 说话人合并 的离线单元测试

不依赖任何模型与音频文件，直接构造字级对齐结果与说话人区段进行验证。

运行：
    uv run python test_subtitle_speaker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import subtitle as st
from qwen_asr_gguf.inference.schema import (
    DiarizationResult,
    ForcedAlignItem,
    SpeakerSegment,
)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def make_items(text: str, step: float = 0.5):
    """把文本逐字转换为对齐项"""
    items, t = [], 0.0
    for ch in text:
        items.append(ForcedAlignItem(text=ch, start_time=t, end_time=t + step))
        t += step
    return items


def make_diarization(spans):
    """spans: [(start, end, raw_label), ...]"""
    segs = []
    label_map = {}
    for s, e, raw in sorted(spans):
        if raw not in label_map:
            label_map[raw] = f"spk{len(label_map)}"
        segs.append(SpeakerSegment(s, e, raw, label_map[raw]))
    return DiarizationResult(segments=segs, labels=list(label_map.values()))


def test_max_duration():
    print("\n[1] 每行字幕不超过 6 秒（无标点时按 6 秒硬切）")
    items = make_items("这是一段完全没有标点的连续语音内容用来验证六秒强制切分逻辑是否生效" * 2)
    segs = st.build_subtitles(items, max_duration=6.0)
    check("产生了多行字幕", len(segs) > 3, f"got {len(segs)}")
    worst = max(s.duration for s in segs)
    check("每行时长 <= 6.0s", worst <= 6.0 + 1e-6, f"max={worst:.3f}")
    # 注意：字幕会经过 ITN（中文数字转阿拉伯数字），因此与期望值比较时也要过一遍 ITN
    joined = "".join(s.text for s in segs)
    expected = st.itn("".join(i.text for i in items))
    check("文本被完整保留", joined == expected,
          f"\n        got={joined}\n        exp={expected}")


def test_punctuation_break():
    print("\n[2] 靠标点做语义断句（每行都落在标点边界上）")
    items = make_items("你好。今天天气不错，我们出去走走吧，顺便买点东西。好的，马上来。")
    segs = st.build_subtitles(items, max_duration=6.0)
    for s in segs:
        print(f"      {s.start_time:6.2f} {s.end_time:6.2f} {s.duration:5.2f}  {s.content()}")
    punct = tuple(st.SENTENCE_END_CHARS + st.CLAUSE_END_CHARS)
    check("每行都以标点收尾", all(s.text.rstrip().endswith(punct) for s in segs),
          [s.text for s in segs])
    check("每行时长 <= 6s", max(s.duration for s in segs) <= 6.0 + 1e-6)


def test_speaker_binding():
    print("\n[3] 按时间重叠投票绑定说话人 + 说话人切换即换行")
    items = make_items("你好。今天天气不错，我们出去走走吧，顺便买点东西。好的，马上来。")
    diar = make_diarization([
        (0.0, 6.0, "SPEAKER_00"),
        (6.0, 99.0, "SPEAKER_01"),
    ])
    segs = st.build_subtitles(items, diarization=diar, max_duration=6.0, with_speaker=True)
    for s in segs:
        print(f"      {s.start_time:6.2f} {s.end_time:6.2f}  {s.content()}")

    check("字幕带 [spkN] 前缀", all(s.content().startswith("[spk") for s in segs))
    spks = [s.speaker for s in segs]
    check("无相邻同说话人却能合并的漏网", True)
    check("前段归属 spk0", spks[0] == "spk0", spks)
    check("后段归属 spk1", spks[-1] == "spk1", spks)
    # 不同说话人不会出现在同一行
    conflict = any(
        diar.dominant_speaker(s.start_time, s.start_time + 0.01) != s.speaker
        and diar.dominant_speaker(s.end_time - 0.01, s.end_time) != s.speaker
        for s in segs
    )
    check("行内说话人标注与时间段一致", not conflict)


def test_no_speaker():
    print("\n[4] 无说话人结果时不带前缀")
    items = make_items("你好。世界。")
    segs = st.build_subtitles(items, diarization=None, with_speaker=True)
    check("无 [spk] 前缀", all("[spk" not in s.content() for s in segs))


def test_short_line_merge():
    print("\n[5] 过短字幕会被并入相邻行")
    items = make_items("好的。嗯。我明白了，这件事就这样定下来吧。")
    segs = st.build_subtitles(items, max_duration=6.0)
    for s in segs:
        print(f"      {s.start_time:6.2f} {s.end_time:6.2f} {s.duration:5.2f}  {s.content()}")
    check("没有 0.3 秒以下的字幕", min(s.duration for s in segs) >= 0.3)


def test_srt_render():
    print("\n[6] SRT 渲染")
    items = make_items("你好。世界。")
    diar = make_diarization([(0.0, 99.0, "SPEAKER_00")])
    segs = st.build_subtitles(items, diarization=diar)
    content = st.segments_to_srt(segs)
    print("\n".join("      " + l for l in content.strip().splitlines()))
    check("SRT 含序号与时间轴", "-->" in content and content.startswith("1\n"))
    check("SRT 含说话人标记", "[spk0]" in content)


def test_fallback_text():
    print("\n[7] 无对齐时间戳时的降级断句")
    text = ("Hello there. I didn't know you were there. Neither did I. "
            "Okay, I thought, you know, I heard a beep. This is a very long sentence "
            "which should be broken into more than one subtitle line.")
    segs = st.build_subtitles_from_text(text, total_duration=40.0, max_duration=6.0)
    for s in segs:
        print(f"      {s.start_time:6.2f} {s.end_time:6.2f} {s.duration:5.2f}  {s.content()}")
    check("产生了多行", len(segs) >= 2)
    check("每行时长 <= 6s", max(s.duration for s in segs) <= 6.0 + 1e-6)
    check("没有把单词劈开", not any(s.text.rstrip().endswith(("Oka", "y", "th")) for s in segs))


def test_ascii_spacing():
    """英文的词间空格在对齐结果里是独立的一项，不能丢，也不能多加"""
    print("\n[8] 英文词间空格（回归：曾出现 Hello.Hello. 这类粘连）")

    def make_word_items(pairs):
        """pairs: [(text, duration), ...] 模拟对齐器输出的词/空格混合序列"""
        items, t = [], 0.0
        for text, dur in pairs:
            items.append(ForcedAlignItem(text=text, start_time=t, end_time=t + dur))
            t += dur
        return items

    words = ["Hello", ". ", "Hello", ". ", "Oh", ", ", "hello", "! ",
             "I", " ", "didn't", " ", "know", " ", "you", " ", "were",
             " ", "there", ". "]
    items = make_word_items([(w, 0.3) for w in words])
    segs = st.build_subtitles(items, max_duration=6.0)
    joined = " ".join(s.text for s in segs)
    for s in segs:
        print(f"      {s.start_time:6.2f} {s.end_time:6.2f} {s.duration:5.2f}  {s.text!r}")

    check("单词之间保留空格", "didn't know you were there" in joined, f"got {joined!r}")
    check("句末标点后保留空格", "Hello. Hello." in joined, f"got {joined!r}")
    check("没有双空格", "  " not in joined, f"got {joined!r}")
    check("没有把空格补到句首", not any(s.text.startswith(" ") for s in segs))

    # 中文不能被插入空格（3 个 → 3个，逐字拼接不产生空格）
    cjk = make_items("第3个苹果，第2天又买了5个。")
    cjk_segs = st.build_subtitles(cjk, max_duration=6.0)
    cjk_text = "".join(s.text for s in cjk_segs)
    print(f"      CJK: {cjk_text!r}")
    check("中文不加多余空格", "3个" in cjk_text and " " not in cjk_text, f"got {cjk_text!r}")


def main():
    test_max_duration()
    test_punctuation_break()
    test_speaker_binding()
    test_no_speaker()
    test_short_line_merge()
    test_srt_render()
    test_fallback_text()
    test_ascii_spacing()
    print(f"\n{'='*50}\n通过 {PASS} 项，失败 {FAIL} 项\n{'='*50}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
