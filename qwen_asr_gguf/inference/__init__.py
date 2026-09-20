# coding=utf-8
from .. import logger
try:
    from ...llama import llama
except:
    ...

from .asr import QwenASREngine
from .aligner import QwenForcedAligner
from .diarization import (
    SpeakerDiarizer,
    default_model_dir,
    describe_backends,
    onnx_backend_available,
    onnxruntime_available,
    pyannote_available,
    resolve_backend,
    resolve_hf_token,
)
from .schema import (
    ForcedAlignItem,
    ForcedAlignResult,
    DecodeResult,
    AlignerConfig,
    ASREngineConfig,
    TranscribeResult,
    # 说话人识别
    DiarizationConfig,
    DiarizationResult,
    SpeakerSegment,
    # 字幕
    SubtitleSegment,
)
from .chinese_itn import chinese_to_num as itn
from .audio import load_audio
from . import exporters
from . import subtitle
