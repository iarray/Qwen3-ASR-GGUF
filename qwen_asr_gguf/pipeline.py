# coding=utf-8
"""
pipeline.py - 语音识别 + 说话人识别 + 字幕翻译 端到端流水线

串联各模块，输出带说话人标记的 SRT 字幕：

    音频/视频文件
        ├─ ffmpeg / soundfile 解码为 16kHz 单声道
        ├─ 说话人分离（onnxruntime / pyannote）→ DiarizationResult (spk0 / spk1 ...)
        ├─ Qwen3-ASR GGUF 转录      → 文本 + ForceAligner 字级时间戳
        ├─ 标点语义断句 + 6 秒限长   → SubtitleSegment 列表
        ├─ （可选）Qwen3-8B 翻译     → 保留说话人与时间轴，另存 <原名>.zh.srt
        └─ 落盘: <原名>.srt / .txt / .json

也支持「只翻译已有字幕」：

    xxx.srt  ──(parse_srt)──> SubtitleSegment
             ──(Qwen3-8B 翻译)──> <原名>.zh.srt        （见 translate_subtitle_file）

    该模式不加载 ASR / 说话人识别模型，适合字幕已经生成、只想补译文时省时间。

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
    SubtitleTranslator,
    TranscribeResult,
    TranslationConfig,
    exporters,
    load_audio,
    subtitle as subtitle_mod,
)
from .inference.translator import DEFAULT_TRANSLATION_PROMPT

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

    # 翻译（可选，基于 llama.cpp + Qwen3-8B GGUF）
    enable_translation: bool = False
    translation_model: str = "Qwen3-8B-Q4_K_M.gguf"
    translation_backend: str = "auto"          # auto / vulkan / cuda / cpu
    translation_prompt: str = DEFAULT_TRANSLATION_PROMPT
    translation_batch_size: int = 1            # 单次请求翻译几段；>1 更快但质量略降
    translation_max_new_tokens: int = 512
    translation_temperature: float = 0.3
    translation_n_ctx: int = 4096
    translation_n_gpu_layers: int = -1         # -1 = 全部层卸载到 GPU
    translation_max_line_duration: float = 6.0
    translation_max_line_chars: int = 24
    translation_output_suffix: str = ".zh"     # 译文另存为 <原名>.zh.srt

    verbose: bool = False


@dataclass
class PipelineResult:
    """单个文件的完整处理结果"""
    audio_path: str
    srt_path: Optional[str] = None
    txt_path: Optional[str] = None
    json_path: Optional[str] = None
    translated_srt_path: Optional[str] = None
    text: str = ""
    segments: List[SubtitleSegment] = field(default_factory=list)
    translated_segments: List[SubtitleSegment] = field(default_factory=list)
    speakers: Optional[DiarizationResult] = None
    duration: float = 0.0
    elapsed: float = 0.0
    performance: Optional[dict] = None
    cancelled: bool = False
    # "media" = 走了完整的解码 + 说话人 + 识别流程；
    # "subtitle" = 输入本身就是 .srt，只做了翻译（srt_path 指向输入文件，未被改写）
    source_kind: str = "media"

    @property
    def translation_only(self) -> bool:
        return self.source_kind == "subtitle"


# 可直接「只翻译」的输入扩展名（输入本身就是字幕，无需重跑识别）
SUBTITLE_FILE_EXTS = (".srt",)


def is_subtitle_file(path) -> bool:
    """判断输入是否是字幕文件（.srt）——这类文件只会走翻译，不做识别"""
    return Path(str(path)).suffix.lower() in SUBTITLE_FILE_EXTS


def split_files_by_kind(paths) -> tuple:
    """把输入路径分成 `(音视频列表, 字幕列表)`，保持原有顺序"""
    media: List[str] = []
    subs: List[str] = []
    for p in paths:
        (subs if is_subtitle_file(p) else media).append(str(p))
    return media, subs


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


def find_missing_translation_model(config: PipelineConfig) -> List[str]:
    """检查翻译模型是否存在（未开启翻译时返回空列表）"""
    if not config.enable_translation:
        return []
    path = Path(config.model_dir) / config.translation_model
    return [] if path.exists() else [str(path)]


def build_translator(config: PipelineConfig) -> SubtitleTranslator:
    """按配置构建字幕翻译器（模型此时不会加载，首次翻译时才载入）"""
    return SubtitleTranslator(
        TranslationConfig(
            model_dir=config.model_dir,
            model_fn=config.translation_model,
            backend=config.translation_backend,
            n_gpu_layers=config.translation_n_gpu_layers,
            n_ctx=config.translation_n_ctx,
            max_new_tokens=config.translation_max_new_tokens,
            temperature=config.translation_temperature,
            batch_size=config.translation_batch_size,
            prompt_template=config.translation_prompt,
            max_line_duration=config.translation_max_line_duration,
            max_line_chars=config.translation_max_line_chars,
            output_suffix=config.translation_output_suffix,
            enabled=config.enable_translation,
            verbose=config.verbose,
        )
    )


class TranscriptionPipeline:
    """语音识别 + 说话人识别（+ 可选字幕翻译）流水线"""

    def __init__(
        self,
        engine: QwenASREngine,
        diarizer: Optional[SpeakerDiarizer] = None,
        config: Optional[PipelineConfig] = None,
        translator: Optional[SubtitleTranslator] = None,
    ):
        self.engine = engine
        self.diarizer = diarizer
        self.translator = translator
        self.config = config or PipelineConfig()
        self._diarization_disabled = False
        self._translation_disabled = False

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
        if self.engine is None:
            raise RuntimeError(
                "ASR 引擎未初始化，无法识别音视频文件；"
                "若要处理 .srt 字幕文件请改用 translate_subtitle_file()。"
            )
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

            # ---------------- 6. 字幕翻译（可选） ---------------- #
            # 注意顺序：原字幕已经落盘，翻译只**新增**一个译文文件，绝不覆盖原文。
            if (
                cfg.enable_translation
                and self.translator is not None
                and not self._translation_disabled
                and not result.cancelled
                and segments
            ):
                emit("stage", stage="translate", message="正在翻译字幕 ...")
                t_tr = time.time()
                try:
                    translated = self.translator.translate_segments(
                        segments,
                        with_speaker=with_speaker,
                        on_progress=lambda cur, tot: emit(
                            "stage",
                            stage="translate",
                            message=f"正在翻译字幕 {cur}/{tot} ...",
                            progress=0.95 + 0.04 * (cur / max(1, tot)),
                        ),
                        on_stream=lambda piece: emit("translate_text", text=piece),
                        cancel_event=cancel_event,
                    )
                    if translated:
                        tsrt = f"{out_base}{cfg.translation_output_suffix}.srt"
                        subtitle_mod.export_segments_to_srt(
                            tsrt, translated, with_speaker=with_speaker
                        )
                        result.translated_segments = translated
                        result.translated_srt_path = tsrt
                        emit("file", kind="translated_srt", path=tsrt)
                        emit(
                            "stage",
                            stage="translate",
                            message=(
                                f"翻译完成，共 {len(translated)} 条字幕，"
                                f"耗时 {time.time() - t_tr:.1f} 秒"
                            ),
                            progress=0.99,
                        )
                except Exception as e:
                    # 翻译失败只降级，不影响已经产出的原字幕
                    self._translation_disabled = True
                    emit("warning", message=f"字幕翻译失败，后续文件将跳过翻译：{e}")

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
    def translate_subtitle_file(
        self,
        srt_path: str,
        on_event: Optional[EVENT] = None,
        cancel_event: Optional[threading.Event] = None,
        with_speaker: Optional[bool] = None,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """只翻译一个已有的 SRT 字幕文件（**不跑** ASR / 说话人识别）

        用途：字幕已经生成好了（本工具上次跑过，或别的工具产出的），只想补一份译文时
        直接把 `.srt` 喂进来即可 —— 省掉解码音频 + 说话人识别 + 语音识别的时间。

        行为约定：
        * 输入字幕**原样保留**，绝不覆盖；译文写到 `<原名><translate_suffix>.srt`；
        * `with_speaker=None` 时按内容自动判断（只要有一行带 `[spkX]` 前缀就保留标记）；
        * 需要 `config.enable_translation=True` 且 `translator` 已就绪，否则只读取不翻译。

        Args:
            with_speaker: None = 自动检测；True/False 强制保留或丢弃说话人标记
            output_dir:   译文输出目录，默认与输入字幕同目录
        """
        cfg = self.config
        src = Path(str(srt_path))
        out_base = Path(output_dir) / src.stem if output_dir else src.with_suffix("")
        result = PipelineResult(audio_path=str(srt_path), source_kind="subtitle")
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

        def finish() -> PipelineResult:
            result.elapsed = time.time() - t0
            return result

        # ---------------- 1. 读取并解析字幕 ---------------- #
        emit("stage", stage="load", message=f"正在读取字幕: {src.name}")
        try:
            segments, has_speaker = subtitle_mod.load_srt_segments(src)
        except Exception as e:
            emit("warning", message=f"字幕解析失败：{e}")
            return finish()

        if not segments:
            emit("warning", message=f"字幕文件里没有可用的字幕条目：{src}")
            return finish()

        if with_speaker is None:
            with_speaker = has_speaker
        # load_srt_segments 已经把 [spkX] 前缀拆进 speaker 字段；
        # 若调用方明确要求不要说话人标记，就地把字段清空。
        if not with_speaker:
            for seg in segments:
                seg.speaker = ""

        result.segments = segments
        result.srt_path = str(src)
        result.duration = max(seg.end_time for seg in segments)
        emit(
            "stage",
            stage="load",
            message=(
                f"已读取 {len(segments)} 条字幕"
                + ("（含说话人标记）" if has_speaker else "（无说话人标记）")
            ),
            progress=0.05,
        )

        if cancelled():
            result.cancelled = True
            return finish()

        # ---------------- 2. 翻译前置检查 ---------------- #
        if not cfg.enable_translation or self.translator is None or self._translation_disabled:
            emit("warning", message="未启用字幕翻译，仅读取了字幕文件，没有产生译文。")
            return finish()

        tsrt = f"{out_base}{cfg.translation_output_suffix}.srt"
        try:
            same_path = Path(tsrt).resolve() == src.resolve()
        except Exception:
            same_path = False
        if same_path:
            emit(
                "warning",
                message=(
                    f"译文路径与输入字幕相同（{tsrt}），已放弃以免覆盖原文件；"
                    "请修改「译文后缀」后再试。"
                ),
            )
            return finish()

        # ---------------- 3. 翻译 ---------------- #
        emit("stage", stage="translate", message="正在翻译字幕 ...")
        t_tr = time.time()
        try:
            translated = self.translator.translate_segments(
                segments,
                with_speaker=with_speaker,
                on_progress=lambda cur, tot: emit(
                    "stage",
                    stage="translate",
                    message=f"正在翻译字幕 {cur}/{tot} ...",
                    progress=0.05 + 0.9 * (cur / max(1, tot)),
                ),
                on_stream=lambda piece: emit("translate_text", text=piece),
                cancel_event=cancel_event,
            )
        except Exception as e:
            self._translation_disabled = True
            emit("warning", message=f"字幕翻译失败：{e}")
            return finish()

        if not translated:
            emit("warning", message="没有产生任何译文，原字幕保持不变。")
            return finish()

        # ---------------- 4. 落盘译文（原字幕不动） ---------------- #
        subtitle_mod.export_segments_to_srt(tsrt, translated, with_speaker=with_speaker)
        result.translated_segments = translated
        result.translated_srt_path = tsrt
        emit("file", kind="translated_srt", path=tsrt)
        emit(
            "stage",
            stage="done",
            message=(
                f"翻译完成，共 {len(translated)} 条字幕，"
                f"耗时 {time.time() - t_tr:.1f} 秒"
            ),
            progress=1.0,
        )
        return finish()

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
        """释放说话人 / 翻译模型占用的资源"""
        if self.diarizer is not None:
            self.diarizer.release()
        if self.translator is not None:
            self.translator.release()
