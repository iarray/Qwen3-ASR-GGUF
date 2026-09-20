# coding=utf-8
"""
subtitle.py - 字幕断句与说话人合并

流程：
    字级对齐结果 (ForcedAlignItem)
        -> 按标点 / 停顿 / 时长聚合成语义从句 (clause)
        -> 绑定说话人 (来自 DiarizationResult 的重叠投票)
        -> 打包为字幕行 (每行 <= max_duration 秒，说话人切换即换行)
        -> 输出 SubtitleSegment 列表 / SRT 文本

设计要点：
- **标点优先**：从句边界一律落在标点上，保证语义完整；
- **6 秒兜底**：当一个从句（或连续从句）超过 max_duration 时强制切分；
- **说话人切换即换行**：字幕不跨说话人，便于阅读与配音。
"""
import re
from typing import Callable, List, Optional, Sequence

import srt

from .schema import DiarizationResult, ForcedAlignItem, SubtitleSegment
from .chinese_itn import chinese_to_num as itn

# 强断句标点（句子结束）
SENTENCE_END_CHARS = "。！？!?…～"
# 弱断句标点（从句结束）
CLAUSE_END_CHARS = "，、；：,;:"
# 所有可断句标点
BREAK_CHARS = SENTENCE_END_CHARS + CLAUSE_END_CHARS + "\n\r"

# 收尾符号（标点后可能紧跟引号 / 括号）
_TRAILING_CHARS = '”"’\')）)】》」』…～ '
# 英文句点单独处理：只有后接空白或行尾才算断句，避免切坏 3.14 / v2.0 这类数字
_DOT_BREAK = r"(?<=\.)(?=\s|$)"

# 用于判断某个对齐项是否落在断句边界
_BREAK_PATTERN = re.compile(rf"[{re.escape(BREAK_CHARS)}]|{_DOT_BREAK}")
# 用于把整段文本按标点切成从句（零宽切分）
_SPLIT_PATTERN = re.compile(rf"(?<=[{re.escape(BREAK_CHARS)}])|{_DOT_BREAK}")


class _Clause:
    """语义从句（标点闭合，时长受限）"""
    __slots__ = ("text", "start", "end", "speaker")

    def __init__(self, text: str, start: float, end: float, speaker: str = ""):
        self.text = text
        self.start = start
        self.end = end
        self.speaker = speaker

    def __repr__(self):
        return f"_Clause({self.text!r}, {self.start:.2f}-{self.end:.2f}, {self.speaker})"


# --------------------------------------------------------------------------- #
# 1. 从句切分
# --------------------------------------------------------------------------- #
def split_into_clauses(
    items: Sequence[ForcedAlignItem],
    max_duration: float = 6.0,
    max_gap: float = 1.2,
) -> List[_Clause]:
    """把字级对齐项聚合成语义从句

    Args:
        items:        字/词级对齐项（含 reconcile 补回的标点）
        max_duration: 单个从句的最长时长，超长则强制切分
        max_gap:      相邻对齐项之间的最大静默，超过则切分
    """
    clauses: List[_Clause] = []
    buf: List[str] = []
    buf_start: Optional[float] = None
    buf_end: float = 0.0

    def flush():
        nonlocal buf, buf_start, buf_end
        if buf:
            text = "".join(buf).strip()
            if text:
                clauses.append(_Clause(text, buf_start, max(buf_end, buf_start), ""))
        buf = []
        buf_start = None
        buf_end = 0.0

    for it in items:
        text = (it.text or "")
        if not text:
            continue
        # 注意：这里**不能**丢弃纯空白项。英文的词间空格在对齐结果里是独立的一项
        # （如 ' ' / '. '），丢掉会让整句英文粘成一串（Hello.Hello.）。
        # 从句首尾多余的空白由 flush() 里的 strip() 收掉，不受影响。

        start = float(it.start_time)
        end = float(it.end_time)
        if end < start:
            end = start

        if buf_start is not None:
            # 停顿过长 → 先断开
            if start - buf_end > max_gap:
                flush()
            # 加上该项将超过 6 秒 → 先断开（保证每行不超时长）
            elif (end - buf_start) > max_duration and buf:
                flush()

        if buf_start is None:
            buf_start = start
        buf_end = max(buf_end, end)
        buf.append(text)

        # 标点闭合，或从句自身已达时长上限
        if _BREAK_PATTERN.search(text) or (buf_end - buf_start) >= max_duration:
            flush()

    flush()
    return clauses


# --------------------------------------------------------------------------- #
# 2. 说话人绑定
# --------------------------------------------------------------------------- #
def attach_speakers(clauses: List[_Clause], diarization: Optional[DiarizationResult]) -> List[_Clause]:
    """为每个从句绑定说话人（取区间内重叠时长最大的说话人）"""
    if diarization is None or len(diarization) == 0:
        for cl in clauses:
            cl.speaker = ""
        return clauses
    for cl in clauses:
        cl.speaker = diarization.dominant_speaker(cl.start, cl.end, default="")
    return clauses


# --------------------------------------------------------------------------- #
# 3. 打包成字幕行
# --------------------------------------------------------------------------- #
def _ends_sentence(text: str) -> bool:
    """判断文本是否以句末标点结束（允许后面跟引号 / 括号）"""
    stripped = text.rstrip().rstrip(_TRAILING_CHARS)
    if not stripped:
        return False
    return stripped[-1] in SENTENCE_END_CHARS or stripped[-1] == "."


# 需要在右侧补一个空格的前一个字符（仅 ASCII，中/日文不加空格）
_SPACE_AFTER = set(
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ".,!?;:'\")]%-"
)
# 需要在左侧补一个空格的后一个字符（仅 ASCII）
_SPACE_BEFORE = set(
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "('\"[$¥#@"
)


def _smart_join(left: str, right: str) -> str:
    """拼接两段文本，必要时补回一个空格

    对齐结果按字/词切开后，`Hello.` 与 `Hello` 这类边界会丢掉分隔空格，
    直接相加会粘成 `Hello.Hello.`。这里按 ASCII 词边界补回空格；
    中文、日文不使用空格分词，两侧只要有一侧不是 ASCII 字符就不会被插入空格
    （例如 `3` + `个` 仍然输出 `3个`）。
    """
    if not left:
        return right
    if not right:
        return left
    a, b = left[-1], right[0]
    if a.isspace() or b.isspace():
        return left + right
    if a in _SPACE_AFTER and b in _SPACE_BEFORE:
        return f"{left} {right}"
    return left + right


def _join_pieces(pieces: Sequence[str]) -> str:
    """把若干文本片段按词边界规则拼成一行"""
    out = ""
    for p in pieces:
        out = _smart_join(out, p)
    return out


def pack_subtitles(
    clauses: List[_Clause],
    max_duration: float = 6.0,
    max_chars: int = 48,
    max_gap: float = 1.2,
    min_line_duration: float = 1.0,
    with_speaker: bool = True,
) -> List[SubtitleSegment]:
    """把语义从句打包为字幕行

    换行条件（满足其一即换行）：
    1. 当前行已以一个完整句子（句末标点）结束，且行时长已达 min_line_duration
    2. 加上新从句后总时长超过 max_duration
    3. 加上新从句后总字数超过 max_chars
    4. 说话人发生变化
    5. 与上一从句之间的静默超过 max_gap

    这样既能「靠标点做语义断句」，又能「每行控制在 max_duration 秒内」。
    """
    segments: List[SubtitleSegment] = []
    cur_texts: List[str] = []
    cur_start: float = 0.0
    cur_end: float = 0.0
    cur_speaker: str = ""

    def flush():
        nonlocal cur_texts, cur_start, cur_end, cur_speaker
        if cur_texts:
            text = _join_pieces(cur_texts).strip()
            if text:
                segments.append(
                    SubtitleSegment(
                        index=len(segments) + 1,
                        start_time=cur_start,
                        end_time=max(cur_end, cur_start),
                        text=itn(text),
                        speaker=cur_speaker if with_speaker else "",
                    )
                )
        cur_texts = []
        cur_start = 0.0
        cur_end = 0.0
        cur_speaker = ""

    for i, cl in enumerate(clauses):
        # ---- 加之前：判断是否需要先换行 ---- #
        if cur_texts:
            cur_len = sum(len(t) for t in cur_texts)
            too_long = (cl.end - cur_start) > max_duration
            too_many_chars = (cur_len + len(cl.text)) > max_chars
            speaker_changed = bool(cl.speaker) and cl.speaker != cur_speaker
            big_gap = (cl.start - cur_end) > max_gap
            if too_long or too_many_chars or speaker_changed or big_gap:
                flush()

        if not cur_texts:
            cur_start = cl.start
            cur_speaker = cl.speaker
        cur_texts.append(cl.text)
        cur_end = max(cur_end, cl.end)

        # ---- 加之后：句末标点收尾 ---- #
        if not _ends_sentence(cl.text):
            continue

        nxt = clauses[i + 1] if i + 1 < len(clauses) else None
        cur_dur = cur_end - cur_start
        if nxt is None:
            flush()
        elif cur_dur >= min_line_duration:
            flush()
        else:
            # 当前行太短，短暂延后断句，让下一从句补足阅读时长
            cur_len = sum(len(t) for t in cur_texts)
            if (
                nxt.speaker != cur_speaker
                or (nxt.end - cur_start) > max_duration
                or (cur_len + len(nxt.text)) > max_chars
                or (nxt.start - cur_end) > max_gap
            ):
                flush()

    flush()
    return segments


def merge_short_segments(
    segments: List[SubtitleSegment],
    min_duration: float = 1.0,
    max_duration: float = 6.0,
    max_chars_soft: int = 64,
) -> List[SubtitleSegment]:
    """把一闪而过的超短字幕并入上一行（不跨说话人）"""
    if not segments:
        return segments

    out: List[SubtitleSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        same_speaker = (seg.speaker == prev.speaker)
        if (
            seg.duration < min_duration
            and same_speaker
            and (seg.end_time - prev.start_time) <= max_duration
            and (len(prev.text) + len(seg.text)) <= max_chars_soft
        ):
            prev.text = _smart_join(prev.text, seg.text)
            prev.end_time = max(prev.end_time, seg.end_time)
        else:
            out.append(seg)

    # 若首行本身太短且能并入下一行，则向后合并
    while len(out) >= 2 and out[0].duration < min_duration:
        nxt = out[1]
        if (
            out[0].speaker == nxt.speaker
            and (nxt.end_time - out[0].start_time) <= max_duration
            and (len(out[0].text) + len(nxt.text)) <= max_chars_soft
        ):
            nxt.text = _smart_join(out[0].text, nxt.text)
            nxt.start_time = out[0].start_time
            out.pop(0)
        else:
            break

    return out


def normalize_timeline(
    segments: List[SubtitleSegment],
    min_duration: float = 0.6,
    allow_overlap: bool = False,
) -> List[SubtitleSegment]:
    """修正时间轴：保证单调递增、最短时长，并尽量不重叠"""
    for i, seg in enumerate(segments):
        if seg.end_time <= seg.start_time:
            seg.end_time = seg.start_time + min_duration

        next_start = segments[i + 1].start_time if i + 1 < len(segments) else None

        if seg.end_time - seg.start_time < min_duration:
            target = seg.start_time + min_duration
            if next_start is not None and not allow_overlap:
                target = min(target, max(seg.start_time + 0.1, next_start - 0.02))
            seg.end_time = target

        if next_start is not None and not allow_overlap and seg.end_time > next_start:
            seg.end_time = max(seg.start_time + 0.05, next_start - 0.02)

    for i, seg in enumerate(segments):
        seg.index = i + 1
    return segments


# --------------------------------------------------------------------------- #
# 4. 对外主入口
# --------------------------------------------------------------------------- #
def build_subtitles(
    items: Optional[Sequence[ForcedAlignItem]],
    diarization: Optional[DiarizationResult] = None,
    max_duration: float = 6.0,
    max_chars: int = 48,
    max_gap: float = 1.2,
    min_line_duration: float = 1.0,
    with_speaker: bool = True,
) -> List[SubtitleSegment]:
    """由字级对齐结果 + 说话人结果，生成最终字幕片段列表"""
    if not items:
        return []

    clauses = split_into_clauses(items, max_duration=max_duration, max_gap=max_gap)
    attach_speakers(clauses, diarization)
    segments = pack_subtitles(
        clauses,
        max_duration=max_duration,
        max_chars=max_chars,
        max_gap=max_gap,
        min_line_duration=min_line_duration,
        with_speaker=with_speaker,
    )
    segments = merge_short_segments(
        segments,
        min_duration=min_line_duration,
        max_duration=max_duration,
        max_chars_soft=max(max_chars, int(max_chars * 1.4)),
    )
    return normalize_timeline(segments)


def _split_oversize(piece: str, max_chars: int) -> List[str]:
    """把过长的文本拆成不超过 max_chars 的片段（优先在空格处断开，避免劈开单词）"""
    if len(piece) <= max_chars:
        return [piece]

    tokens = piece.split(" ")
    if len(tokens) > 1:
        out: List[str] = []
        cur = ""
        for tk in tokens:
            candidate = f"{cur} {tk}" if cur else tk
            if len(candidate) > max_chars and cur:
                out.append(cur)
                cur = tk
            else:
                cur = candidate
        if cur:
            out.append(cur)
        return out or [piece]

    # 纯 CJK 等无空格文本：按字符硬切
    return [piece[i:i + max_chars] for i in range(0, len(piece), max_chars)] or [piece]


def build_subtitles_from_text(
    text: str,
    total_duration: float,
    diarization: Optional[DiarizationResult] = None,
    max_duration: float = 6.0,
    max_chars: int = 48,
    with_speaker: bool = True,
) -> List[SubtitleSegment]:
    """无对齐时间戳时的降级方案：按标点切分，并按字数比例分配时间"""
    if not text or not text.strip():
        return []

    # 估算语速，用于保证单个从句的时长也不超过 max_duration
    total_len = len(text.strip()) or 1
    chars_per_sec = total_len / max(total_duration, 0.001)
    allowed_chars = max(4, int(max_duration * chars_per_sec * 0.95))
    split_len = max(1, min(max_chars, allowed_chars))

    pieces: List[str] = []
    for chunk in _SPLIT_PATTERN.split(text):
        chunk = chunk.strip()
        if chunk:
            pieces.extend(_split_oversize(chunk, split_len))
    if not pieces:
        pieces = [text.strip()]

    total_chars = sum(len(p) for p in pieces) or 1

    # 生成等价的"语义从句"，复用与主流程一致的打包逻辑
    clauses: List[_Clause] = []
    cursor = 0.0
    for piece in pieces:
        dur = max(0.4, total_duration * len(piece) / total_chars)
        start = cursor
        end = min(total_duration, cursor + dur)
        cursor = end
        clauses.append(_Clause(piece, start, max(end, start), ""))

    attach_speakers(clauses, diarization)
    segments = pack_subtitles(
        clauses,
        max_duration=max_duration,
        max_chars=max_chars,
        min_line_duration=1.0,
        with_speaker=with_speaker,
    )
    segments = merge_short_segments(
        segments,
        min_duration=1.0,
        max_duration=max_duration,
        max_chars_soft=max(max_chars, int(max_chars * 1.4)),
    )
    return normalize_timeline(segments)


# --------------------------------------------------------------------------- #
# 5. 序列化
# --------------------------------------------------------------------------- #
def segments_to_srt(segments: Sequence[SubtitleSegment], with_speaker: bool = True) -> str:
    """把字幕片段列表渲染为 SRT 文本"""
    subs = []
    for seg in segments:
        text = seg.content(with_speaker=with_speaker)
        if not text.strip():
            continue
        subs.append(
            srt.Subtitle(
                index=len(subs) + 1,
                start=srt.timedelta(seconds=max(0.0, seg.start_time)),
                end=srt.timedelta(seconds=max(seg.start_time + 0.05, seg.end_time)),
                content=text,
            )
        )
    return srt.compose(subs)


def segments_to_json(segments: Sequence[SubtitleSegment]) -> List[dict]:
    """把字幕片段列表渲染为可序列化的字典列表"""
    return [
        {
            "index": i + 1,
            "start": round(seg.start_time, 3),
            "end": round(seg.end_time, 3),
            "speaker": seg.speaker,
            "text": seg.text,
        }
        for i, seg in enumerate(segments)
    ]


def export_segments_to_srt(
    path: str,
    segments: Sequence[SubtitleSegment],
    with_speaker: bool = True,
) -> str:
    """写 SRT 文件，返回写入的文本"""
    content = segments_to_srt(segments, with_speaker=with_speaker)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return content
