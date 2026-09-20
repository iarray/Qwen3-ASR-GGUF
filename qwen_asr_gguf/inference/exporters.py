# coding=utf-8
import json
import re
from typing import List, Optional

from .chinese_itn import chinese_to_num as itn
from .schema import DiarizationResult, ForcedAlignItem, SubtitleSegment, TranscribeResult
from . import subtitle as _subtitle


# --------------------------------------------------------------------------- #
# 转换函数
# --------------------------------------------------------------------------- #
def alignment_to_srt(
    items: Optional[List[ForcedAlignItem]],
    max_chars: int = 48,
    max_duration: float = 6.0,
    diarization: Optional[DiarizationResult] = None,
    with_speaker: bool = False,
) -> str:
    """将对齐结果转换为 SRT 格式内容。

    按标点做语义断句，每行时长尽量控制在 `max_duration` 秒（默认 6 秒）；
    当提供 `diarization` 且 `with_speaker=True` 时，每行字幕以 `[spk0]` 开头。
    """
    segments = _subtitle.build_subtitles(
        items,
        diarization=diarization,
        max_duration=max_duration,
        max_chars=max_chars,
        with_speaker=with_speaker,
    )
    return _subtitle.segments_to_srt(segments, with_speaker=with_speaker)


def alignment_to_json(
    items: Optional[List[ForcedAlignItem]],
    diarization: Optional[DiarizationResult] = None,
) -> List[dict]:
    """将对齐结果转换为可序列化的字典列表（可选附带说话人）"""
    if not items:
        return []
    result = []
    for it in items:
        row = {
            "text": it.text,
            "start": round(it.start_time, 3),
            "end": round(it.end_time, 3),
        }
        if diarization is not None and len(diarization):
            row["speaker"] = diarization.dominant_speaker(it.start_time, it.end_time, default="")
        result.append(row)
    return result


def _resolve_segments(
    result: TranscribeResult,
    with_speaker: bool,
    max_duration: float,
    max_chars: int,
) -> List[SubtitleSegment]:
    """优先复用结果中已算好的字幕片段，否则现场生成"""
    if result.subtitles:
        return result.subtitles
    if result.alignment:
        return _subtitle.build_subtitles(
            result.alignment.items,
            diarization=result.speakers,
            max_duration=max_duration,
            max_chars=max_chars,
            with_speaker=with_speaker,
        )
    return []


# --------------------------------------------------------------------------- #
# 导出函数
# --------------------------------------------------------------------------- #
def export_to_srt(
    path: str,
    result: TranscribeResult,
    with_speaker: bool = True,
    max_duration: float = 6.0,
    max_chars: int = 48,
) -> List[SubtitleSegment]:
    """将对齐结果（含说话人）保存为 SRT 文件"""
    segments = _resolve_segments(result, with_speaker, max_duration, max_chars)

    if not segments:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return []

    content = _subtitle.segments_to_srt(segments, with_speaker=with_speaker)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 已生成字幕文件: {path} ({len(segments)} 条)")
    return segments


def export_to_json(path: str, result: TranscribeResult):
    """将字级对齐结果（含说话人）保存为 JSON 文件"""
    if not result.alignment:
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")
        return

    data = alignment_to_json(result.alignment.items, diarization=result.speakers)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已导出时间戳: {path}")


def export_to_txt(path: str, result: TranscribeResult, with_speaker: bool = False):
    """将转录结果处理后保存为 TXT 文件 (含 ITN 和标点换行)"""
    # 1. ITN 处理
    final_text = itn(result.text)

    if with_speaker and result.subtitles:
        # 按字幕行输出，带说话人标记
        lines = [seg.content(with_speaker=True) for seg in result.subtitles]
        formatted_text = "\n".join(lines) + "\n"
    else:
        # 2. 按照标点符号换行，保留标点
        formatted_text = re.sub(r'([，。？！：])', r'\1\n', final_text)
        # 3. 对于英文字母后面的逗号空格、句号空格，也要换行
        formatted_text = re.sub(r'(?<=[a-zA-Z])([,\.] )', r'\1\n', formatted_text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    print(f"✅ 已保存文本文件: {path}")
