# coding=utf-8
"""
diarization.py - 说话人识别 (Speaker Diarization) 模块

两种推理后端（由 `DiarizationConfig.backend` 选择）：

1. **onnx**（推荐）—— 纯 onnxruntime 推理，支持 DirectML / CUDA / CPU，
   不需要 torch，模型由 `30-Export-Diarization-ONNX.py` 一次性导出。
   实现在 `diarization_onnx.py`。
2. **pyannote** —— pyannote.audio 原生 PyTorch 实现（CPU / CUDA）。

两者输出完全一致的结构（`DiarizationResult`），说话人标签统一归一化为
`spk0`、`spk1` ...（按首次出现顺序）。

其它特性：
- 指定说话人数 / 自动检测说话人数
- 复用已加载的 16kHz 单声道波形，避免重复解码音频
- 设备自动选择 (dml / cuda / cpu) 与失败降级
- 惰性导入，缺少依赖时不影响其它功能
"""
import os
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .schema import DiarizationConfig, DiarizationResult, SpeakerSegment

# 常见的环境变量名，按优先级排列
_TOKEN_ENVS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)

# 已知的 pyannote 说话人分离流水线（按优先级尝试）
DEFAULT_MODEL_CANDIDATES = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization",
)

# 项目默认的模型目录
PROJECT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"


def default_model_dir() -> str:
    return str(PROJECT_MODEL_DIR)


def pyannote_available() -> bool:
    """检测 pyannote.audio 是否可用（不触发真正的模型加载）"""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def onnx_backend_available(model_dir: Optional[str] = None) -> bool:
    """检测 ONNX 说话人分离模型是否已导出"""
    try:
        from .diarization_onnx import onnx_models_available
        return onnx_models_available(model_dir or default_model_dir())
    except Exception:
        return False


def onnxruntime_available() -> bool:
    """检测 onnxruntime 是否可用"""
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(config: DiarizationConfig) -> str:
    """根据配置与本地文件决定实际使用的后端：'onnx' 或 'pyannote'"""
    want = (config.backend or "auto").lower()
    model_dir = config.model_dir or default_model_dir()

    if want in ("onnx", "auto") and onnxruntime_available() and onnx_backend_available(model_dir):
        return "onnx"
    if want == "onnx":
        raise RuntimeError(
            "未找到 ONNX 说话人分离模型或 onnxruntime 未安装。\n"
            "请先执行: uv run python 30-Export-Diarization-ONNX.py\n"
            f"（模型应位于 {model_dir}）"
        )
    return "pyannote"


def resolve_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """解析 HuggingFace Token：显式传入 > 环境变量"""
    if explicit:
        return explicit.strip()
    for name in _TOKEN_ENVS:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def describe_backends(model_dir: Optional[str] = None) -> Tuple[str, str]:
    """返回 (推荐后端, 说明文本)，供界面展示"""
    can_onnx = onnxruntime_available() and onnx_backend_available(model_dir)
    can_pyannote = pyannote_available()
    if can_onnx:
        return "onnx", "ONNX Runtime（DirectML / CUDA / CPU）"
    if can_pyannote:
        return "pyannote", "pyannote.audio（PyTorch，CPU / CUDA）"
    return "", "不可用：请导出 ONNX 模型或安装 pyannote.audio"


class SpeakerDiarizer:
    """说话人识别引擎（ONNX Runtime 或 pyannote.audio）"""

    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self._pipeline = None          # pyannote Pipeline
        self._onnx = None              # OnnxDiarizationEngine
        self._backend: Optional[str] = None
        self._device = None
        self._model_name_loaded: Optional[str] = None
        self._warned: List[str] = []

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _log(self, msg: str, on_status: Optional[Callable[[str], None]] = None):
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    @property
    def backend(self) -> str:
        """实际使用的后端（'onnx' / 'pyannote'）"""
        if self._backend is None:
            self._backend = resolve_backend(self.config)
        return self._backend

    def _resolve_device(self, warn: Optional[Callable[[str], None]] = None):
        """解析推理设备，返回 torch.device 或 None(表示 CPU)"""
        import torch

        requested = (self.config.device or "auto").lower()

        if requested in ("cuda", "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")

        if requested in ("dml", "directml", "auto"):
            try:
                import torch_directml  # type: ignore
                dev = torch_directml.device()
                # 只有能构造出 torch.device 才能被 pyannote 的 .to() 接受
                if isinstance(dev, torch.device):
                    return dev
            except Exception:
                pass

        if requested == "dml":
            self._warn(warn, "torch-directml 不可用，说话人识别将回退到 CPU")
        return torch.device("cpu")

    def _warn(self, warn: Optional[Callable[[str], None]], msg: str):
        if msg in self._warned:
            return
        self._warned.append(msg)
        if warn is not None:
            try:
                warn(msg)
            except Exception:
                pass
        else:
            warnings.warn(msg)

    # ------------------------------------------------------------------ #
    # 模型加载
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None or self._onnx is not None

    def _load_onnx(self, on_status: Optional[Callable[[str], None]] = None):
        from .diarization_onnx import OnnxDiarizationEngine

        model_dir = self.config.model_dir or default_model_dir()
        self._log("正在加载 ONNX 说话人分离模型 ...", on_status)
        self._onnx = OnnxDiarizationEngine(
            segmentation_path=os.path.join(model_dir, "diarization_segmentation.onnx"),
            embedding_path=os.path.join(model_dir, "diarization_embedding.onnx"),
            device=self.config.device or "auto",
            exclude_overlap=True,
            verbose=False,
        )
        self._onnx.load(on_status=on_status)
        self._device = self._onnx.providers[0] if self._onnx.providers else "CPUExecutionProvider"
        self._log(
            f"ONNX 说话人分离模型已就绪 (后端: {self._device})",
            on_status,
        )
        return self._onnx

    def _load_pyannote(self, on_status: Optional[Callable[[str], None]] = None):
        if not pyannote_available():
            raise RuntimeError(
                "未安装 pyannote.audio，且没有可用的 ONNX 模型。\n"
                "二选一：\n"
                "  1) 导出 ONNX 模型: uv run python 30-Export-Diarization-ONNX.py\n"
                "  2) 安装 pyannote:  uv add pyannote.audio"
            )

        import torch
        from pyannote.audio import Pipeline

        token = resolve_hf_token(self.config.hf_token)
        if not token:
            self._warn(
                on_status,
                "未提供 HuggingFace Token，pyannote 模型为 gated 资源，下载可能失败。"
                "请在界面填写 Token 或设置环境变量 HF_TOKEN。",
            )

        candidates = [self.config.model_name] if self.config.model_name else []
        for name in DEFAULT_MODEL_CANDIDATES:
            if name not in candidates:
                candidates.append(name)

        last_error: Optional[Exception] = None
        for model_name in candidates:
            try:
                self._log(f"正在加载说话人识别模型: {model_name} ...", on_status)
                for kwargs in ({"token": token}, {"use_auth_token": token}, {}):
                    try:
                        self._pipeline = Pipeline.from_pretrained(model_name, **kwargs)
                        break
                    except TypeError:
                        # 不同版本参数名不同，逐个降级尝试
                        continue
                if self._pipeline is not None:
                    self._model_name_loaded = model_name
                    break
            except Exception as e:  # 下载失败 / 未授权 / 网络问题
                last_error = e
                self._log(f"加载 {model_name} 失败: {e}", on_status)
                self._pipeline = None

        if self._pipeline is None:
            raise RuntimeError(
                f"说话人识别模型加载失败，请检查网络与 HuggingFace Token。\n最后一个错误: {last_error}"
            )

        device = self._resolve_device()
        try:
            self._pipeline.to(device)
            self._device = device
        except Exception as e:
            self._log(f"切换到设备 {device} 失败 ({e})，回退 CPU", on_status)
            self._pipeline.to(torch.device("cpu"))
            self._device = torch.device("cpu")

        # 允许及时释放显存
        if self._device.type == "cuda" and hasattr(self._pipeline, "segmentation_batch_size"):
            try:
                self._pipeline.segmentation_batch_size = 16
            except Exception:
                pass

        self._log(f"说话人识别模型已就绪 (设备: {self._device}, 模型: {self._model_name_loaded})", on_status)
        return self._pipeline

    def load(self, on_status: Optional[Callable[[str], None]] = None):
        """惰性加载模型（重复调用只会加载一次）"""
        if self.is_loaded:
            return self._onnx or self._pipeline

        if self.backend == "onnx":
            return self._load_onnx(on_status=on_status)
        return self._load_pyannote(on_status=on_status)

    def release(self):
        """释放模型占用的内存"""
        self._pipeline = None
        self._onnx = None
        self._model_name_loaded = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def diarize_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
        uri: str = "audio",
    ) -> DiarizationResult:
        """对已加载的波形执行说话人识别

        Args:
            waveform:    (N,) 或 (1, N) 的 float32 波形
            sample_rate: 采样率（默认 16000）
            num_speakers: 指定说话人数；None 表示自动检测
            on_status:   进度消息回调
            uri:         文件标识
        """
        pipeline = self.load(on_status=on_status)

        t0 = time.time()

        # 组装波形 (N,) -> (1, N)
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim == 2 and wav.shape[0] == 1:
            wav = wav[0]
        elif wav.ndim > 1:
            wav = wav.reshape(-1)

        # 说话人数参数
        num = int(num_speakers) if num_speakers and int(num_speakers) > 0 else None
        min_spk = max(1, int(self.config.min_speakers))
        max_spk = max(min_spk, int(self.config.max_speakers))

        # ---------------- ONNX Runtime 后端 ---------------- #
        if self.backend == "onnx":
            result = self._onnx.diarize(
                wav,
                sample_rate=int(sample_rate),
                num_speakers=num,
                min_speakers=min_spk,
                max_speakers=max_spk,
                on_status=on_status,
            )
            elapsed = time.time() - t0
            if result.performance is None:
                result.performance = {}
            result.performance.setdefault("total_time", elapsed)
            result.performance.setdefault("device", str(self._device))
            result.performance["backend"] = "onnx"
            return result

        # ---------------- pyannote 原生后端 ---------------- #
        import torch

        tensor = torch.from_numpy(wav[np.newaxis, :])

        call_kwargs = {}
        if num:
            call_kwargs["num_speakers"] = num
        else:
            call_kwargs["min_speakers"] = min_spk
            call_kwargs["max_speakers"] = max_spk

        def _hook(step_name, step_artefact=None, file=None, **kw):
            total = kw.get("total")
            completed = kw.get("completed")
            if on_status is not None:
                if total:
                    self._log(f"说话人识别: {step_name} ({completed}/{total})", on_status)
                else:
                    self._log(f"说话人识别: {step_name}", on_status)

        file = {"waveform": tensor, "sample_rate": int(sample_rate), "uri": uri}
        output = pipeline(file, hook=_hook, **call_kwargs)

        segments = self._parse_output(output)
        elapsed = time.time() - t0

        labels: List[str] = []
        for seg in segments:
            if seg.label not in labels:
                labels.append(seg.label)

        self._log(
            f"说话人识别完成: 共 {len(labels)} 位说话人 / {len(segments)} 个语音段，耗时 {elapsed:.2f}s",
            on_status,
        )

        return DiarizationResult(
            segments=segments,
            labels=labels,
            performance={"total_time": elapsed, "device": str(self._device), "backend": "pyannote"},
        )

    def diarize_file(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> DiarizationResult:
        """直接对音频文件执行说话人识别（内部会自行解码）"""
        # ONNX 后端只接受已解码波形，统一先解码
        if (self.config.backend or "auto").lower() in ("onnx", "auto"):
            try:
                from .audio import load_audio

                wav = load_audio(str(audio_path))
                return self.diarize_waveform(
                    wav,
                    sample_rate=16000,
                    num_speakers=num_speakers,
                    on_status=on_status,
                    uri=Path(audio_path).stem,
                )
            except Exception:
                # 解码失败则继续走 pyannote 的文件路径
                pass

        pipeline = self.load(on_status=on_status)
        t0 = time.time()

        call_kwargs = {}
        if num_speakers and int(num_speakers) > 0:
            call_kwargs["num_speakers"] = int(num_speakers)
        else:
            call_kwargs["min_speakers"] = max(1, int(self.config.min_speakers))
            call_kwargs["max_speakers"] = max(
                call_kwargs["min_speakers"], int(self.config.max_speakers)
            )

        output = pipeline(str(audio_path), **call_kwargs)
        segments = self._parse_output(output)

        labels: List[str] = []
        for seg in segments:
            if seg.label not in labels:
                labels.append(seg.label)

        return DiarizationResult(
            segments=segments,
            labels=labels,
            performance={"total_time": time.time() - t0, "device": str(self._device)},
        )

    # ------------------------------------------------------------------ #
    # 结果解析
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_annotation(output):
        """兼容 pyannote 3.x / 4.x 的不同返回结构"""
        if output is None:
            return None
        # 4.x: DiarizeOutput
        for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
            ann = getattr(output, attr, None)
            if ann is not None:
                return ann
        # 3.x: 直接返回 Annotation
        if hasattr(output, "itertracks"):
            return output
        return None

    def _parse_output(self, output) -> List[SpeakerSegment]:
        annotation = self._extract_annotation(output)
        if annotation is None:
            return []

        raw = []
        for turn, _track, speaker in annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue
            raw.append((start, end, str(speaker)))

        # 按时间排序后，按“首次出现顺序”归一化说话人标签
        raw.sort(key=lambda x: (x[0], x[1]))
        label_map = {}
        segments: List[SpeakerSegment] = []
        for start, end, speaker in raw:
            if speaker not in label_map:
                label_map[speaker] = f"spk{len(label_map)}"
            segments.append(
                SpeakerSegment(
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    speaker=speaker,
                    label=label_map[speaker],
                )
            )
        return segments
