# coding=utf-8
"""
pipeline.py - 语音识别 + 说话人识别 端到端流水线

串联各模块，输出带说话人标记的 SRT 字幕：

    音频/视频文件
        ├─ ffmpeg / soundfile 解码为 16kHz 单声道
        ├─ pyannote 说话人分离      → DiarizationResult (spk0 / spk1 ...)
        ├─ Qwen3-ASR GGUF 转录      → 文本 + ForceAligner 字级时间戳
        ├─ 标点语义断句 + 6 秒限长   → SubtitleSegment 列表
        └─ 落盘: <原名>.srt / .txt / .json

通过 `on_event(kind, payload)` 向外持续汇报进度，供 CLI 与 GUI 复用。
"""
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .inference import (
    ASREngineConfig,
    AlignerConfig,
    DiarizationConfig,
    DiarizationResult,
    QwenASREngine,
    SpeakerDiarizer,
    SubtitleSegment,
    TranscribeResult,
    exporters,
    load_audio,
    subtitle as subtitle_mod,
)

# 各阶段在总进度中的权重
_STAGE_WEIGHTS = {"load": 0.03, "diarize": 0.35, "asr": 0.55, "export": 0.07}

EVENT = Callable[[str, Dict[str, Any]], None]


@dataclass
class PipelineConfig:
    """流水线配置"""
    model_dir: str = "model"
    precision: str = "int4"
    onnx_provider: str = "DML"
    llm_use_gpu: bool = True
    use_vulkan: bool = True
    n_ctx: int = 2048
    chunk_size: float = 40.0
    memory_num: int = 1
    language: Optional[str] = None
    context: str = ""
    temperature: float = 0.4
    enable_aligner: bool = True      # 关闭后字幕时间轴将按字数估算

    # 字幕
    max_line_duration: float = 6.0
    max_line_chars: int = 48

    # 说话人
    enable_diarization: bool = True
    diarization_backend: str = "auto"   # auto / onnx / pyannote
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    diarization_model_dir: Optional[str] = None   # ONNX 模型目录；None 表示 <model_dir>
    hf_token: Optional[str] = None
    diarization_device: str = "auto"    # auto / cpu / cuda / dml
    min_speakers: int = 1
    max_speakers: int = 20

    # 导出
    export_txt: bool = True
    export_json: bool = True

    verbose: bool = False


@dataclass
class PipelineResult:
    """单个文件的完整处理结果"""
    audio_path: str
    srt_path: Optional[str] = None
    txt_path: Optional[str] = None
    json_path: Optional[str] = None
    text: str = ""
    segments: List[SubtitleSegment] = field(default_factory=list)
    speakers: Optional[DiarizationResult] = None
    duration: float = 0.0
    elapsed: float = 0.0
    performance: Optional[dict] = None
    cancelled: bool = False


def get_model_filenames(precision: str, is_aligner: bool = False) -> Dict[str, str]:
    """根据精度返回对应的模型文件名"""
    prefix = "qwen3_aligner" if is_aligner else "qwen3_asr"
    return {
        "frontend": f"{prefix}_encoder_frontend.{precision}.onnx",
        "backend": f"{prefix}_encoder_backend.{precision}.onnx",
    }


def find_missing_models(model_dir: str, precision: str = "int4", enable_aligner: bool = True) -> List[str]:
    """检查模型文件是否齐全，返回缺失文件路径列表"""
    asr = get_model_filenames(precision, is_aligner=False)
    names = [
        "qwen3_asr_llm.q4_k.gguf",
        asr["frontend"],
        asr["backend"],
    ]
    if enable_aligner:
        align = get_model_filenames(precision, is_aligner=True)
        names += [
            "qwen3_aligner_llm.q4_k.gguf",
            align["frontend"],
            align["backend"],
        ]
    base = Path(model_dir)
    return [str(base / n) for n in names if not (base / n).exists()]


def build_engine(config: PipelineConfig, verbose: bool = False) -> QwenASREngine:
    """按配置构建 ASR 引擎"""
    if not config.use_vulkan:
        os.environ["VK_ICD_FILENAMES"] = "none"

    asr_files = get_model_filenames(config.precision, is_aligner=False)
    align_files = get_model_filenames(config.precision, is_aligner=True)

    align_config = None
    if config.enable_aligner:
        align_config = AlignerConfig(
            model_dir=config.model_dir,
            onnx_provider=config.onnx_provider,
            llm_use_gpu=config.llm_use_gpu,
            encoder_frontend_fn=align_files["frontend"],
            encoder_backend_fn=align_files["backend"],
            n_ctx=config.n_ctx,
        )

    engine_config = ASREngineConfig(
        model_dir=config.model_dir,
        onnx_provider=config.onnx_provider,
        llm_use_gpu=config.llm_use_gpu,
        encoder_frontend_fn=asr_files["frontend"],
        encoder_backend_fn=asr_files["backend"],
        n_ctx=config.n_ctx,
        chunk_size=config.chunk_size,
        memory_num=config.memory_num,
        enable_aligner=config.enable_aligner,
        align_config=align_config,
        verbose=verbose,
        quiet=not verbose,
    )
    return QwenASREngine(config=engine_config)


def diarization_model_dir(config: PipelineConfig) -> str:
    """ONNX 说话人模型所在目录（默认复用 ASR 的 model_dir）"""
    return config.diarization_model_dir or config.model_dir


def resolve_diarization_backend(config: PipelineConfig) -> str:
    """预览实际会使用的说话人后端：'onnx' / 'pyannote' / ''（不可用）

    只看能力是否具备，不做任何模型加载，供 GUI / CLI 提前提示用。
    """
    try:
        from .inference.diarization import (
            onnx_backend_available,
            onnxruntime_available,
            pyannote_available,
            resolve_backend,
        )
    except Exception:
        return ""

    want = (config.diarization_backend or "auto").lower()
    model_dir = diarization_model_dir(config)

    # 明确指定 onnx，或 auto 且本地已有 ONNX 模型
    if want == "onnx" or (want == "auto" and onnxruntime_available() and onnx_backend_available(model_dir)):
        return "onnx" if onnxruntime_available() and onnx_backend_available(model_dir) else ""
    if want == "pyannote":
        return "pyannote" if pyannote_available() else ""
    # auto 且无 ONNX 模型
    return "pyannote" if pyannote_available() else ""


def diarization_needs_hf_token(config: PipelineConfig) -> bool:
    """判断当前配置下的说话人识别是否依赖 HuggingFace Token

    只有真正会走 pyannote（其模型是 gated 资源）时才需要 Token；
    ONNX 后端使用本地已导出的模型，无需 Token。
    """
    if not config.enable_diarization:
        return False
    if config.hf_token:
        return False
    resolved = resolve_diarization_backend(config)
    if resolved == "onnx":
        return False
    if resolved == "pyannote":
        return True
    # 后端不可用：若可能回退 pyannote 则提示
    return (config.diarization_backend or "auto").lower() != "onnx"


def build_diarizer(config: PipelineConfig) -> SpeakerDiarizer:
    """按配置构建说话人识别器"""
    return SpeakerDiarizer(
        DiarizationConfig(
            backend=config.diarization_backend,
            model_dir=diarization_model_dir(config),
            model_name=config.diarization_model,
            hf_token=config.hf_token,
            device=config.diarization_device,
            num_speakers=None,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
            enabled=config.enable_diarization,
        )
    )


class TranscriptionPipeline:
    """语音识别 + 说话人识别 流水线"""

    def __init__(
        self,
        engine: QwenASREngine,
        diarizer: Optional[SpeakerDiarizer] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.engine = engine
        self.diarizer = diarizer
        self.config = config or PipelineConfig()
        self._diarization_disabled = False

    # ------------------------------------------------------------------ #
    def run(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        on_event: Optional[EVENT] = None,
        cancel_event: Optional[threading.Event] = None,
        with_speaker: bool = True,
        output_dir: Optional[str] = None,
        start_second: float = 0.0,
        duration: Optional[float] = None,
    ) -> PipelineResult:
        """处理单个音视频文件，产出 SRT 字幕"""
        cfg = self.config
        audio_path = str(audio_path)
        base = Path(audio_path)
        out_base = Path(output_dir) / base.stem if output_dir else base.with_suffix("")
        result = PipelineResult(audio_path=audio_path)

        t0 = time.time()

        def emit(event: str, **payload):
            if on_event is None:
                return
            try:
                on_event(event, payload)
            except Exception:
                pass

        def cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        self._diarization = None
        asr_started = {"value": False}

        def on_chunk(idx: int, total: int, s: float, e: float):
            emit(
                "chunk",
                index=idx,
                total=total,
                start=s,
                end=e,
                speakers=self._chunk_speakers(s, e),
            )
            if asr_started["value"]:
                emit("progress", progress=self._asr_progress(idx, total))

        # 让引擎的回调把流式文本 / 分片进度转发给外部
        if self.engine is not None:
            self.engine.set_callbacks(
                on_stream=lambda piece: emit("text", text=piece),
                on_chunk=on_chunk,
                on_status=lambda msg: emit("status", message=msg),
                cancel_event=cancel_event,
            )

        try:
            # ---------------- 1. 解码音频 ---------------- #
            emit("stage", stage="load", message=f"正在读取音频: {base.name}")
            audio = load_audio(
                audio_path,
                start_second=start_second if start_second else None,
                duration=duration,
            )
            duration = len(audio) / 16000.0
            result.duration = duration
            emit(
                "stage",
                stage="load",
                message=f"音频时长 {duration:.1f} 秒",
                progress=_STAGE_WEIGHTS["load"],
            )
            if cancelled():
                result.cancelled = True
                return result

            # ---------------- 2. 说话人识别 ---------------- #
            diarization: Optional[DiarizationResult] = None
            use_diarization = (
                cfg.enable_diarization
                and self.diarizer is not None
                and not self._diarization_disabled
            )
            if use_diarization:
                emit("stage", stage="diarize", message="正在识别说话人 ...")
                try:
                    diarization = self.diarizer.diarize_waveform(
                        audio,
                        sample_rate=16000,
                        num_speakers=num_speakers,
                        on_status=lambda m: emit("status", message=m),
                        uri=base.stem,
                    )
                    self._diarization = diarization
                    result.speakers = diarization
                    emit(
                        "stage",
                        stage="diarize",
                        message=f"检测到 {diarization.num_speakers} 位说话人",
                        speakers=diarization,
                        progress=_STAGE_WEIGHTS["load"] + _STAGE_WEIGHTS["diarize"],
                    )
                except Exception as e:
                    # 只报一次错，后续文件不再重复尝试说话人识别
                    self._diarization_disabled = True
                    diarization = None
                    self._diarization = None
                    emit(
                        "warning",
                        message=f"说话人识别失败，后续文件将只输出纯文本字幕：{e}",
                    )
            else:
                self._diarization = None

            if cancelled():
                result.cancelled = True
                return result

            # ---------------- 3. 语音识别 ---------------- #
            emit("stage", stage="asr", message="正在转录 ...")
            self._asr_offset = _STAGE_WEIGHTS["load"] + (
                _STAGE_WEIGHTS["diarize"] if diarization else 0.0
            )
            asr_started["value"] = True

            transcribe_result: TranscribeResult = self.engine.asr(
                audio=audio,
                context=cfg.context,
                language=cfg.language,
                chunk_size_sec=cfg.chunk_size,
                memory_chunks=cfg.memory_num,
                temperature=cfg.temperature,
            )
            result.text = transcribe_result.text
            result.performance = transcribe_result.performance
            result.cancelled = bool(transcribe_result.performance.get("cancelled"))

            # ---------------- 4. 断句 + 说话人合并 ---------------- #
            emit("stage", stage="export", message="正在生成字幕 ...")
            if transcribe_result.alignment and len(transcribe_result.alignment):
                segments = subtitle_mod.build_subtitles(
                    transcribe_result.alignment.items,
                    diarization=diarization,
                    max_duration=cfg.max_line_duration,
                    max_chars=cfg.max_line_chars,
                    with_speaker=with_speaker,
                )
            else:
                segments = subtitle_mod.build_subtitles_from_text(
                    transcribe_result.text,
                    duration,
                    diarization=diarization,
                    max_duration=cfg.max_line_duration,
                    max_chars=cfg.max_line_chars,
                    with_speaker=with_speaker,
                )
            result.segments = segments
            transcribe_result.speakers = diarization
            transcribe_result.subtitles = segments

            # ---------------- 5. 落盘 ---------------- #
            srt_path = f"{out_base}.srt"
            subtitle_mod.export_segments_to_srt(srt_path, segments, with_speaker=with_speaker)
            result.srt_path = srt_path
            emit("file", kind="srt", path=srt_path)

            if cfg.export_txt:
                txt_path = f"{out_base}.txt"
                exporters.export_to_txt(txt_path, transcribe_result, with_speaker=with_speaker)
                result.txt_path = txt_path
                emit("file", kind="txt", path=txt_path)

            if cfg.export_json and transcribe_result.alignment:
                json_path = f"{out_base}.json"
                exporters.export_to_json(json_path, transcribe_result)
                result.json_path = json_path
                emit("file", kind="json", path=json_path)

            result.elapsed = time.time() - t0
            emit(
                "stage",
                stage="done",
                message=f"完成，共 {len(segments)} 条字幕，耗时 {result.elapsed:.1f} 秒",
                progress=1.0,
            )
            return result

        finally:
            if self.engine is not None:
                self.engine.set_callbacks(None, None, None, None)

    # ------------------------------------------------------------------ #
    def _asr_progress(self, idx: int, total: int) -> float:
        """ASR 阶段在总进度中的占比"""
        offset = getattr(self, "_asr_offset", _STAGE_WEIGHTS["load"])
        ratio = (idx + 1) / max(1, total)
        return min(0.99, offset + _STAGE_WEIGHTS["asr"] * ratio)

    def _chunk_speakers(self, start: float, end: float) -> List[str]:
        """返回某个分片时间范围内出现过的说话人标签（去重、保持顺序）"""
        diar = getattr(self, "_diarization", None)
        if not diar or not len(diar):
            return []
        seen: List[str] = []
        for seg in diar.segments:
            if seg.end_time <= start or seg.start_time >= end:
                continue
            if seg.label not in seen:
                seen.append(seg.label)
        return seen

    def release(self):
        """释放说话人模型占用的资源"""
        if self.diarizer is not None:
            self.diarizer.release()
