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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

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


# --------------------------------------------------------------------------- #
# 5.5 SRT 解析（用于「只翻译已有字幕文件」的场景）
#
# 需求：允许把已经生成好的 .srt 直接丢进来，只补一份译文，不必再从音频重跑一遍
# 解码 + 说话人识别 + 语音识别。因此这里需要一个**健壮**的 SRT 读取器：
#   * 兼容 UTF-8 / UTF-8-BOM / GB18030 / BIG5 等编码（字幕文件常见各种编码）；
#   * 识别 `[spk0] 你好。` 形式的说话人前缀，拆进 SubtitleSegment.speaker；
#   * 序号、时间轴的小毛病（缺序号、逗号/句点做小数点）都要能容忍。
# --------------------------------------------------------------------------- #

# 按顺序尝试的编码；latin-1 永不失败，作为最后兜底
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1")

# 兜底解析用的时间轴正则：00:00:01,234 --> 00:00:03,456
_SRT_TIME_RE = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def read_text_auto(path) -> str:
    """按常见字幕编码读取文本文件（不会抛 UnicodeDecodeError）"""
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):        # UTF-8 BOM
        data = data[3:]
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _srt_time_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000.0


def _parse_srt_blocks(text: str) -> List[Tuple[float, float, str]]:
    """逐块扫描的兜底解析器：容忍缺失序号 / 时间轴格式小毛病

    返回 `[(起始秒, 结束秒, 正文), ...]`。
    """
    out: List[Tuple[float, float, str]] = []
    cur_times: Optional[Tuple[float, float]] = None
    cur_body: List[str] = []

    def flush():
        nonlocal cur_times, cur_body
        body = " ".join(x.strip() for x in cur_body if x.strip()).strip()
        if cur_times is not None and body:
            out.append((cur_times[0], cur_times[1], body))
        cur_times = None
        cur_body = []

    for line in text.split("\n"):
        m = _SRT_TIME_RE.search(line)
        if m:
            flush()
            g = m.groups()
            cur_times = (_srt_time_to_sec(*g[:4]), _srt_time_to_sec(*g[4:8]))
            continue
        if cur_times is None:
            continue                                # 序号行、空行等一律忽略
        if line.strip():
            cur_body.append(line)
        else:
            flush()
    flush()
    return out


def _normalize_srt_body(raw: str) -> str:
    """把多行正文压成一行，并去掉首尾空白"""
    return " ".join(part.strip() for part in (raw or "").splitlines() if part.strip()).strip()


def parse_srt(content: str) -> List[SubtitleSegment]:
    """把 SRT 文本解析为字幕片段列表

    - 支持 `[spk0] 你好。` 形式的说话人前缀（拆分到 `speaker` 字段，正文不含前缀）；
    - 多行正文按空格拼接为单行（本项目的字幕行都不含换行）；
    - `end <= start` 时补 0.05 秒，避免下游出现零长字幕。
    """
    if not content:
        return []

    text = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    raw: List[Tuple[float, float, str]] = []
    try:
        for sub in srt.parse(text):
            raw.append((
                sub.start.total_seconds(),
                sub.end.total_seconds(),
                _normalize_srt_body(sub.content),
            ))
    except Exception:
        raw = []
    if not raw:
        # srt 库解析不出东西（格式不规范）时走兜底扫描
        raw = _parse_srt_blocks(text)

    segments: List[SubtitleSegment] = []
    for start, end, body in raw:
        body = _normalize_srt_body(body)
        if not body:
            continue
        speaker, body = split_speaker_prefix(body)
        if not body:
            continue
        start = max(0.0, float(start))
        end = float(end)
        if end <= start:
            end = start + 0.05
        segments.append(
            SubtitleSegment(
                index=len(segments) + 1,
                start_time=start,
                end_time=end,
                text=body,
                speaker=speaker,
            )
        )
    return segments


def load_srt_segments(path) -> Tuple[List[SubtitleSegment], bool]:
    """读取 SRT 文件，返回 `(片段列表, 是否带说话人标记)`

    是否带说话人标记由内容自动判断（只要有一行带 `[spkX]` 前缀就算），
    这样 GUI / CLI 不必让用户再勾一个额外选项。
    """
    segments = parse_srt(read_text_auto(path))
    has_speaker = any(seg.speaker for seg in segments)
    return segments, has_speaker


# --------------------------------------------------------------------------- #
# 6. 翻译支持：合并成完整句 → 翻译 → 拆回多行
#
# 需求约定：
#   * 如果某行结尾没有句末标点（内容不完整），就与后面若干行合并，直到遇见句末标点，
#     合并后的多行一起送去翻译（保证模型看到的是完整语义）；
#   * 合并块的总时长 = 参与合并的各行时长之和；
#   * 译文再按标点拆回多行，每行尽量 ≤ max_duration 秒，拆完的总时长与原合并块一致；
#   * 每行译文保留原说话人标记。
# --------------------------------------------------------------------------- #

# 说话人前缀：字幕正文里可能已经带了 `[spk0] `，需要剥离后再翻译
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[([^\]]{1,32})\]\s*")

# 译文里允许作为断点的标点（中文优先，兼容英文）
_TRANS_BREAK_CHARS = "。！？!?…～；;，、：:"


@dataclass
class TranslationUnit:
    """翻译单元：若干连续字幕行合并出的一个完整语义块"""
    texts: List[str] = field(default_factory=list)      # 参与合并的各行正文（已剥离说话人前缀）
    segments: List[SubtitleSegment] = field(default_factory=list)
    text: str = ""                                      # 合并后的原文（送去翻译的字符串）
    start: float = 0.0                                  # 块的起始时间
    end: float = 0.0                                    # 块的结束时间
    duration: float = 0.0                               # 各行时长合计（译文按此分配时间）
    speaker: str = ""                                   # 说话人标记（组内唯一）
    complete: bool = False                              # 是否以句末标点收尾

    @property
    def merged_rows(self) -> int:
        return len(self.segments)


def split_speaker_prefix(text: str) -> Tuple[str, str]:
    """把 `[spk0] 你好。` 拆成 `("spk0", "你好。")`；没有前缀时返回 `("", text)`"""
    if not text:
        return "", ""
    m = _SPEAKER_PREFIX_RE.match(text)
    if not m:
        return "", text.strip()
    return m.group(1), text[m.end():].strip()


def group_segments_for_translation(
    segments: Sequence[SubtitleSegment],
    max_group_duration: float = 30.0,
) -> List[TranslationUnit]:
    """把字幕行按「完整句子」聚合为翻译单元

    聚合规则：
    1. 遇到句末标点（。！？!?…）就闭合一个单元；
    2. 说话人切换时强制闭合 —— 否则合并后的译文无法再挂回唯一的说话人标记；
    3. 单块时长超过 `max_group_duration` 时强制闭合，避免整段音频被并成一个超大请求。
    """
    units: List[TranslationUnit] = []
    cur: List[SubtitleSegment] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        texts: List[str] = []
        for seg in cur:
            _, body = split_speaker_prefix(seg.text)
            if body:
                texts.append(body)
        if not texts:
            cur = []
            return
        merged = _join_pieces(texts).strip()
        if merged:
            speaker = cur[0].speaker or split_speaker_prefix(cur[0].text)[0]
            units.append(
                TranslationUnit(
                    texts=texts,
                    segments=list(cur),
                    text=merged,
                    start=cur[0].start_time,
                    end=max(s.end_time for s in cur),
                    duration=sum(max(0.0, s.end_time - s.start_time) for s in cur),
                    speaker=speaker,
                    complete=_ends_sentence(merged),
                )
            )
        cur = []

    for seg in segments:
        body = split_speaker_prefix(seg.text)[1]
        if not body:
            continue
        spk = seg.speaker or split_speaker_prefix(seg.text)[0]

        if cur:
            prev_spk = cur[-1].speaker or split_speaker_prefix(cur[-1].text)[0]
            if spk and prev_spk and spk != prev_spk:
                flush()
            elif (seg.end_time - cur[0].start_time) > max_group_duration:
                flush()

        cur.append(seg)
        if _ends_sentence(body):
            flush()

    flush()
    return units


def _split_piece_once(text: str, max_chars: int) -> Tuple[str, str]:
    """把一段文本尽量平均地一分为二，优先断在从句标点或空格处"""
    n = len(text)
    if n < 2:
        return text, ""
    mid = n // 2
    win = max(2, n // 3)

    lo = max(1, mid - win)
    hi = min(n, mid + win + 1)

    cut = None
    for i in range(lo, hi):
        if text[i - 1] in _TRANS_BREAK_CHARS:
            if cut is None or abs(i - mid) < abs(cut - mid):
                cut = i
    if cut is None:
        for i in range(lo, hi):
            if text[i - 1] == " ":
                if cut is None or abs(i - mid) < abs(cut - mid):
                    cut = i
    if cut is None:
        cut = mid

    left, right = text[:cut].strip(), text[cut:].strip()
    if not right:
        return text, ""
    return left, right


def _enforce_piece_limits(
    pieces: List[str],
    total_duration: float,
    max_duration: float,
    max_chars: int,
    min_piece_chars: int = 4,
) -> List[str]:
    """拆到每一段的预估时长都不超过 max_duration、且字数不超过 max_chars"""
    for _ in range(400):
        total_chars = sum(len(p) for p in pieces) or 1
        target = None
        for i, p in enumerate(pieces):
            est = total_duration * len(p) / total_chars if total_duration > 0 else 0.0
            if len(p) > max_chars or (est > max_duration and len(p) > min_piece_chars):
                target = i
                break
        if target is None:
            break
        left, right = _split_piece_once(pieces[target], max_chars)
        if not right:
            break
        pieces[target:target + 1] = [left, right] if left else [right]
    return [p for p in pieces if p]

def split_translated_text(
    text: str,
    total_duration: float,
    start_time: float = 0.0,
    max_duration: float = 6.0,
    max_chars: int = 24,
    min_duration: float = 0.6,
) -> List[Tuple[str, float, float]]:
    """把一段译文拆成若干字幕行，并按字符数比例分配时间
    切分规则：
        1. 强制断点：。？，若句号/问号后面紧跟中文右双引号”，则在”之后强制切分
        2. 逗号，：只要当前片段字符数 >=5，遇到逗号立刻切分
        3. 禁止无标点中间硬截断，只允许在标点位置切分
    返回 `[(译文, 起始秒, 结束秒), ...]`，各行时长之和恒等于 total_duration。
    """
    text = (text or "").strip()
    if not text:
        return []
    total_duration = max(0.0, float(total_duration))
    max_duration = max(0.5, float(max_duration))
    trigger_len = 5
    min_duration = max(0.3, float(min_duration))

    if total_duration <= 0.01:
        total_duration = max(min_duration, 0.2 * len(text))

    force_char = {"。", "？"}
    comma_char = {"，"}
    right_quote = {"”"}

    pieces: List[str] = []
    pos = 0
    n = len(text)

    while pos < n:
        seg_start = pos
        while pos < n:
            ch = text[pos]
            # -------- 强制断点：。？兼容后面 ” --------
            if ch in force_char:
                if pos + 1 < n and text[pos + 1] in right_quote:
                    pos += 2
                else:
                    pos += 1
                piece = text[seg_start:pos].strip()
                if piece:
                    pieces.append(piece)
                break
            # -------- 逗号：片段>=5字符立刻切 --------
            if ch in comma_char:
                buf = text[seg_start:pos+1].strip()
                if len(buf) >= trigger_len:
                    piece = buf
                    if piece:
                        pieces.append(piece)
                    pos += 1
                    break
            pos +=1
        else:
            # 文本走到末尾，剩余部分
            remain = text[seg_start:].strip()
            if remain:
                pieces.append(remain)
            break

    rows = pieces

    row_chars = [len(r) for r in rows]
    sum_chars = sum(row_chars) or 1
    out: List[Tuple[str, float, float]] = []
    cursor = float(start_time)
    for i, (row, rc) in enumerate(zip(rows, row_chars)):
        if i == len(rows) - 1:
            end = float(start_time) + total_duration
        else:
            end = cursor + total_duration * rc / sum_chars
        if end - cursor < 0.05:
            end = cursor + 0.05
        out.append((row, cursor, end))
        cursor = end
    if out and out[-1][2] <= out[-1][1]:
        out[-1] = (out[-1][0], out[-1][1], out[-1][1] + 0.05)
    return out




def build_translated_segments(
    units: Sequence[TranslationUnit],
    translations: Sequence[str],
    with_speaker: bool = True,
    max_duration: float = 6.0,
    max_chars: int = 24,
    min_duration: float = 0.6,
    fallback_to_source: bool = True,
) -> List[SubtitleSegment]:
    """把译文回填成字幕片段列表（保留说话人标记与整体时间轴）

    某一块译文为空（翻译失败 / 被中断）时，若 `fallback_to_source=True` 则**原样保留**
    该块的原始字幕行，不做拆行，避免把源语言文本按译文规则切碎。
    """
    out: List[SubtitleSegment] = []
    for idx, unit in enumerate(units):
        translated = (translations[idx] if idx < len(translations) else "") or ""
        translated = translated.strip()

        if not translated:
            if not fallback_to_source:
                continue
            for src in unit.segments:
                _, body = split_speaker_prefix(src.text)
                if not body:
                    continue
                out.append(
                    SubtitleSegment(
                        index=len(out) + 1,
                        start_time=src.start_time,
                        end_time=max(src.end_time, src.start_time + 0.05),
                        text=body,
                        speaker=unit.speaker if with_speaker else "",
                    )
                )
            continue

        real_window_duration = max(0.001, unit.end - unit.start)
        rows = split_translated_text(
            translated,
            real_window_duration, #unit.duration,
            unit.start,
            max_duration=max_duration,
            max_chars=max_chars,
            min_duration=min_duration,
        )
        if not rows:
            rows = [(translated, unit.start, unit.start + max(unit.duration, 0.6))]
        for text, s, e in rows:
            if not text:
                continue
            out.append(
                SubtitleSegment(
                    index=len(out) + 1,
                    start_time=s,
                    end_time=e,
                    text=text,
                    speaker=unit.speaker if with_speaker else "",
                )
            )
    for i, seg in enumerate(out):
        seg.index = i + 1
    return out
