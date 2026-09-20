# coding=utf-8
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Tuple
import numpy as np

class MsgType(Enum):
    CMD_ENCODE = auto()   # 主进程 -> Encoder: 编码请求
    CMD_ALIGN = auto()    # 主进程 -> Aligner: 对齐请求
    CMD_STOP = auto()     # 主进程 -> Worker: 停止请求
    MSG_EMBD = auto()     # Worker -> 主进程: 返回特征 (Encoder)
    MSG_ALIGN = auto()    # Worker -> 主进程: 返回对齐结果 (Aligner)
    MSG_READY = auto()    # Worker -> 主进程: 就绪信号
    MSG_DONE = auto()     # Worker -> 主进程: 已退出信号
    MSG_ERROR = auto()    # Worker -> 主进程: 错误信号

@dataclass
class StreamingMessage:
    """音频编码/对齐进程通用通信协议"""
    msg_type: MsgType
    data: Any = None         # 存放音频 chunk 或 embedding/align 结果
    text: Optional[str] = None # 用于对齐的文本
    offset_sec: float = 0.0  # 对齐的时间轴偏移
    language: Optional[str] = None # 语言
    is_last: bool = False    # 标记是否为最后一段音频
    encode_time: float = 0.0 # 耗时统计

@dataclass
class DecodeResult:
    """LLM 解码内核输出标准化"""
    text: str = ""           # 包含前缀的完整文本
    new_text: str = ""       # 本次增量生成的文本
    stable_tokens: List[int] = field(default_factory=list)
    t_prefill: float = 0.0   # 预填充耗时 (ms)
    t_generate: float = 0.0  # 生成耗时 (ms)
    n_prefill: int = 0       # 预填充 token 数
    n_generate: int = 0      # 生成 token 数
    is_aborted: bool = False # 是否因重复或其他原因熔断中断

@dataclass(frozen=True)
class ForcedAlignItem:
    """单个词/字符的对齐结果"""
    text: str
    start_time: float        # 单位：秒
    end_time: float          # 单位：秒

@dataclass
class ForcedAlignResult:
    """对齐结果标准化集合 (官方结构化输出格式)"""
    items: List[ForcedAlignItem]
    performance: Optional[dict] = None

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]

@dataclass
class AlignerConfig:
    """对齐引擎配置"""
    model_dir: str
    # 拆分为 Frontend 和 Backend
    encoder_frontend_fn: str = "qwen3_aligner_encoder_frontend.int4.onnx"
    encoder_backend_fn: str = "qwen3_aligner_encoder_backend.int4.onnx"
    
    llm_fn: str = "qwen3_aligner_llm.q4_k.gguf" 
    onnx_provider: str = 'CPU'  # CPU, CUDA, DML, TensorRT
    llm_use_gpu: bool = True
    n_ctx: int = 2048       # 对于 Aligner Decoder，每秒音频+文字，约占 30 个 token
    dml_pad_to: int = 40 # Encoder 填充时长

@dataclass
class ASREngineConfig:
    """ASR 识别引擎配置"""
    model_dir: str
    encoder_frontend_fn: str = "qwen3_asr_encoder_frontend.int4.onnx"
    encoder_backend_fn: str = "qwen3_asr_encoder_backend.int4.onnx"
    llm_fn: str = "qwen3_asr_llm.q4_k.gguf"

    onnx_provider: str = 'CPU'  # CPU, CUDA, DML, TensorRT
    llm_use_gpu: bool = True
    dml_pad_to: int = 40        # 使用 DirectML 加速 onnx 时，Encoder 填充时长
    n_ctx: int = 2048           # 对于 ASR Decoder，每秒音频+文字，约占 20 个 token
    chunk_size: float = 40.0    # 每个片段 40s，对应 800 个 token
    memory_num: int = 1         # 记忆一个片段，转录一个片段，对应 1600 个 token
    verbose: bool = True
    quiet: bool = False         # 为 True 时屏蔽所有控制台输出（UI 模式）
    enable_aligner: bool = False
    align_config: Optional[AlignerConfig] = None

    def __post_init__(self):
        # 如果没有显式设置 Encoder 填充时长，则默认与 LLM 分段识别时长对齐
        if self.dml_pad_to is None:
            object.__setattr__(self, 'pad_to', int(self.chunk_size))
            
        if self.align_config is None:
            object.__setattr__(self, 'align_config', AlignerConfig(
                model_dir=self.model_dir,
                onnx_provider=self.onnx_provider,
                llm_use_gpu=self.llm_use_gpu,
                dml_pad_to=self.dml_pad_to # Aligner 默认也跟随主 pad_to
            ))
        elif self.align_config.dml_pad_to is None:
             object.__setattr__(self.align_config, 'dml_pad_to', int(self.chunk_size))

@dataclass
class TranscribeResult:
    """ASR 转录结果 (含可选的对齐信息)"""
    text: str
    alignment: Optional[ForcedAlignResult] = None
    performance: Optional[dict] = None
    speakers: Optional["DiarizationResult"] = None       # 说话人识别结果
    subtitles: Optional[List["SubtitleSegment"]] = None  # 已断句的字幕片段


# ============================================================================
# 说话人识别 (Speaker Diarization)
# ============================================================================

@dataclass(frozen=True)
class SpeakerSegment:
    """单个说话人区段"""
    start_time: float                    # 单位：秒
    end_time: float                      # 单位：秒
    speaker: str                         # 原始标签，例如 SPEAKER_00
    label: str = ""                      # 归一化标签，例如 spk0

    def __str__(self) -> str:
        return f"[{self.label or self.speaker}] {self.start_time:.2f}-{self.end_time:.2f}"


@dataclass
class DiarizationResult:
    """说话人识别结果 (按时间排序的说话人区段集合)"""
    segments: List[SpeakerSegment] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)      # 归一化标签，按首次出现顺序 ["spk0", "spk1", ...]
    performance: Optional[dict] = None

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> SpeakerSegment:
        return self.segments[idx]

    @property
    def num_speakers(self) -> int:
        return len(self.labels)

    def speaker_at(self, time_sec: float, default: str = "") -> str:
        """返回指定时刻的说话人标签；有重叠时取该时刻最早开始的区段"""
        best = None
        for seg in self.segments:
            if seg.start_time <= time_sec < seg.end_time:
                if best is None or seg.start_time < best.start_time:
                    best = seg
        return best.label if best else default

    def dominant_speaker(self, start_time: float, end_time: float, default: str = "") -> str:
        """返回时间区间内语音重叠时长最大的说话人标签"""
        if end_time <= start_time:
            return self.speaker_at(start_time, default)

        overlap: dict = {}
        for seg in self.segments:
            ov = min(end_time, seg.end_time) - max(start_time, seg.start_time)
            if ov > 0:
                key = seg.label or seg.speaker
                overlap[key] = overlap.get(key, 0.0) + ov
        if not overlap:
            return self.speaker_at((start_time + end_time) / 2.0, default)
        return max(overlap.items(), key=lambda kv: kv[1])[0]

    def speaker_turns(self) -> List[Tuple[float, str]]:
        """返回 [(切换时刻, 说话人标签), ...]，用于按说话人切分"""
        return [(seg.start_time, seg.label or seg.speaker) for seg in self.segments]


@dataclass
class DiarizationConfig:
    """说话人识别引擎配置

    支持两种后端（`backend`）：
      - `onnx`     : 纯 onnxruntime 推理（DirectML / CUDA / CPU），不需要 torch，
                     模型由 `30-Export-Diarization-ONNX.py` 一次性导出
      - `pyannote` : pyannote.audio 原生 PyTorch 实现（CPU / CUDA）
      - `auto`     : 找到 ONNX 模型就用 onnx，否则回退 pyannote
    """
    backend: str = "auto"              # auto / onnx / pyannote
    model_dir: Optional[str] = None    # ONNX 模型目录；None 表示项目下的 model/
    model_name: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = None     # HuggingFace Token (pyannote 模型为 gated 资源)
    device: str = "auto"               # auto / cpu / cuda / dml
    num_speakers: Optional[int] = None # 指定说话人数；None 表示自动检测
    min_speakers: int = 1              # 自动检测时的下限
    max_speakers: int = 20             # 自动检测时的上限
    enabled: bool = True


# ============================================================================
# 最终字幕片段
# ============================================================================

@dataclass
class SubtitleSegment:
    """最终字幕片段 (已断句、已绑定说话人)"""
    index: int
    start_time: float
    end_time: float
    text: str
    speaker: str = ""    # 归一化说话人标签，例如 spk0；无说话人识别时为空

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def content(self, with_speaker: bool = True) -> str:
        """返回字幕正文，形如 `[spk0] 你好。`"""
        if with_speaker and self.speaker:
            return f"[{self.speaker}] {self.text}"
        return self.text
