# coding=utf-8
"""
diarization_onnx.py - 纯 onnxruntime 的说话人分离流水线

pyannote.audio 的原生实现**完全基于 PyTorch**（推理走 CPU/CUDA）。本模块用
onnxruntime（DirectML / CUDA / CPU）替换其中的两块神经网络，其余算法逐行复刻
`pyannote/speaker-diarization-3.1` 的配方，从而在 AMD 显卡上也能走 GPU：

    1. 分割    pyannote/segmentation-3.0 (PyanNet, powerset)   -> ONNX
    2. 嵌入    pyannote/wespeaker-voxceleb-resnet34-LM         -> ONNX
    3. 计数/聚类/时间轴重建  -> numpy + scipy（纯 CPU，计算量极小）

模型由 `30-Export-Diarization-ONNX.py` 一次性导出，运行期不再需要 torch / pyannote.audio。

与 pyannote 0.4.7 实现的对应关系：
    _segment          <- Inference(segmentation).slide + Powerset.to_multilabel(soft=False)
    _speaker_count    <- SpeakerDiarizationMixin.speaker_count
    _extract_embeddings <- SpeakerDiarization.get_embeddings（含 embedding_exclude_overlap）
    _cluster          <- AgglomerativeClustering.cluster / assign_embeddings
    _reconstruct      <- SpeakerDiarization.reconstruct + to_diarization + to_annotation
"""
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import DiarizationConfig, DiarizationResult, SpeakerSegment

# --------------------------------------------------------------------------- #
# 常量（须与 30-Export-Diarization-ONNX.py 保持一致）
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 16000
CHUNK_SECONDS = 10.0            # 分割模型训练/推理的分片时长
CHUNK_STEP_SECONDS = 1.0        # = 0.1 * CHUNK_SECONDS
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

FBANK_FRAMES = 998              # 10s 波形 -> 1 + (160000-400)//160
NUM_MEL_BINS = 80

# pyannote 3.1 的默认超参（见 speaker-diarization-3.1/config.yaml）
DEFAULT_ONSET = 0.5
DEFAULT_CLUSTER_THRESHOLD = 0.7045654963945799
DEFAULT_CLUSTER_METHOD = "centroid"
DEFAULT_MIN_CLUSTER_SIZE = 12
MIN_ACTIVE_RATIO = 0.2
DEFAULT_EMBEDDING_BATCH_SIZE = 32

EPSILON = float(np.finfo(np.float32).eps)
_AGG_EPSILON = 1e-12

SEGMENTATION_ONNX = "diarization_segmentation.onnx"
EMBEDDING_ONNX = "diarization_embedding.onnx"

# 设备 -> onnxruntime provider 优先级
_PROVIDER_CANDIDATES: Dict[str, List[str]] = {
    "dml": ["DmlExecutionProvider"],
    "directml": ["DmlExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "gpu": ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
    "auto": [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ],
}


# --------------------------------------------------------------------------- #
# 帧网格 / 滑窗（对齐 pyannote.core.SlidingWindow 的语义）
# --------------------------------------------------------------------------- #
class _FrameGrid:
    """等间隔帧网格：第 i 帧占据 [start + i*step, start + i*step + duration]"""

    __slots__ = ("start", "duration", "step")

    def __init__(self, start: float, duration: float, step: float):
        self.start = float(start)
        self.duration = float(duration)
        self.step = float(step)

    def closest_frame(self, t: float) -> int:
        """最接近时刻 t 的帧索引（与 pyannote 完全一致）"""
        return int(np.rint((t - self.start - 0.5 * self.duration) / self.step))

    def middle(self, i: int) -> float:
        return self.start + i * self.step + 0.5 * self.duration

    def extent(self, num_frames: int) -> Tuple[float, float]:
        """covered Segment(start, end)"""
        return self.start, self.start + (num_frames - 1) * self.step + self.duration


def _aggregate(
    data: np.ndarray,
    num_frames_per_chunk: int,
    chunks_start: float,
    chunks_duration: float,
    chunks_step: float,
    frames: _FrameGrid,
    skip_average: bool = False,
    missing: float = np.nan,
) -> np.ndarray:
    """复刻 pyannote `Inference.aggregate`（hamming=False, warm_up=(0,0), epsilon=1e-12）

    把每个分片的逐帧分数按「分片中心对齐」累加到全局帧网格上（重叠相加）。
    """
    num_chunks = data.shape[0]
    num_classes = data.shape[2]
    # 多出一维（如 (chunks, frames, classes, ...)）时统一压平
    if data.ndim != 3:
        raise ValueError(f"aggregate 仅支持 3 维输入，收到 {data.shape}")

    num_frames = (
        frames.closest_frame(
            chunks_start
            + chunks_duration
            + (num_chunks - 1) * chunks_step
            + 0.5 * frames.duration
        )
        + 1
    )
    num_frames = max(num_frames, num_frames_per_chunk)

    total = np.zeros((num_frames, num_classes), dtype=np.float32)
    weight = np.zeros((num_frames, num_classes), dtype=np.float32)
    seen = np.zeros((num_frames, num_classes), dtype=np.float32)

    for c in range(num_chunks):
        score = data[c].astype(np.float32, copy=True)
        mask = 1.0 - np.isnan(score)
        np.nan_to_num(score, copy=False, nan=0.0)

        chunk_start = chunks_start + c * chunks_step
        start_frame = max(0, frames.closest_frame(chunk_start + 0.5 * frames.duration))
        end_frame = start_frame + num_frames_per_chunk

        # 理论上 num_frames 足够容纳所有分片；越界时安全裁剪，避免索引异常
        if start_frame >= num_frames:
            continue
        take = min(num_frames_per_chunk, num_frames - start_frame)
        total[start_frame:end_frame][:take] += (score * mask)[:take]
        weight[start_frame:end_frame][:take] += mask[:take]
        seen[start_frame:end_frame][:take] = np.maximum(
            seen[start_frame:end_frame][:take], mask[:take]
        )

    if skip_average:
        average = total
    else:
        average = total / np.maximum(weight, _AGG_EPSILON)
    if missing is not None and not (isinstance(missing, float) and np.isnan(missing)):
        average[seen == 0.0] = missing
    return average


# --------------------------------------------------------------------------- #
# fbank（复刻 torchaudio.compliance.kaldi.fbank 在 pyannote 下的调用方式）
# --------------------------------------------------------------------------- #
def _mel_banks(num_bins: int, n_fft: int, sr: int,
               low_freq: float = 20.0, high_freq: float = 0.0) -> np.ndarray:
    """Kaldi 风格三角梅尔滤波器组，返回 (num_bins, n_fft//2)"""
    num_fft_bins = n_fft // 2
    nyquist = 0.5 * sr
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sr / n_fft
    mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
    delta = (mel_high - mel_low) / (num_bins + 1)

    b = np.arange(num_bins).reshape(-1, 1)
    left = mel_low + b * delta
    center = mel_low + (b + 1.0) * delta
    right = mel_low + (b + 2.0) * delta

    freqs = fft_bin_width * np.arange(num_fft_bins)
    mel = (1127.0 * np.log(1.0 + freqs / 700.0)).reshape(1, -1)

    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    return np.maximum(0.0, np.minimum(up, down))


_MEL_BANKS_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def compute_fbank(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    num_mel_bins: int = NUM_MEL_BINS,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
) -> np.ndarray:
    """提取 80 维 fbank（并减去时间维均值），返回 (num_frames, num_mel_bins) float32

    与 `pyannote.audio.models.embedding.wespeaker.WeSpeakerResNet34.compute_fbank` 等价：
    先 ×32768（影响 log 截断位置，必须一致），去直流、预加重、hamming 窗、512 点功率谱、
    Kaldi 梅尔滤波、取对数，最后减去时间均值。
    """
    x = np.asarray(waveform, dtype=np.float32).reshape(-1) * (1 << 15)
    win_shift = int(sample_rate * frame_shift * 0.001)
    win_size = int(sample_rate * frame_length * 0.001)
    n_fft = 1 << (win_size - 1).bit_length()

    if len(x) < win_size:
        x = np.pad(x, (0, win_size - len(x)))
    m = 1 + (len(x) - win_size) // win_shift

    idx = np.arange(win_size)[None, :] + win_shift * np.arange(m)[:, None]
    frames = x[idx].astype(np.float64)

    frames -= frames.mean(axis=1, keepdims=True)              # remove_dc_offset
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)   # replicate pad
    frames -= 0.97 * prev                                     # preemphasis

    n = np.arange(win_size)
    frames *= 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (win_size - 1))  # hamming(periodic=False)

    if n_fft != win_size:
        frames = np.pad(frames, ((0, 0), (0, n_fft - win_size)))

    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    key = (num_mel_bins, n_fft, sample_rate)
    banks = _MEL_BANKS_CACHE.get(key)
    if banks is None:
        banks = np.pad(_mel_banks(num_mel_bins, n_fft, sample_rate), ((0, 0), (0, 1)))
        _MEL_BANKS_CACHE[key] = banks

    mel = spectrum @ banks.T
    mel = np.log(np.maximum(mel, EPSILON))
    features = mel.astype(np.float32)
    return features - features.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------- #
# powerset 解码
# --------------------------------------------------------------------------- #
def build_powerset_mapping(num_classes: int, max_set_size: int) -> np.ndarray:
    """powerset 类别索引 -> 多标签 0/1 向量，形状 (num_powerset_classes, num_classes)

    顺序与 pyannote.utils.powerset.Powerset 一致：先按集合大小升序，再按组合字典序。
    """
    from itertools import combinations

    rows: List[List[float]] = []
    for set_size in range(0, max_set_size + 1):
        for combo in combinations(range(num_classes), set_size):
            row = [0.0] * num_classes
            for k in combo:
                row[k] = 1.0
            rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _nearest_upsample_1d(values: np.ndarray, size: int) -> np.ndarray:
    """复刻 torch.nn.functional.interpolate(mode='nearest') 的一维重采样"""
    n = len(values)
    if n == size:
        return values
    idx = np.floor(np.arange(size) * (n / size)).astype(np.int64)
    np.clip(idx, 0, n - 1, out=idx)
    return values[idx]


# --------------------------------------------------------------------------- #
# ONNX 引擎
# --------------------------------------------------------------------------- #
class OnnxDiarizationEngine:
    """基于 onnxruntime 的说话人分离引擎"""

    def __init__(
        self,
        segmentation_path: str,
        embedding_path: str,
        device: str = "auto",
        segmentation_batch_size: int = 16,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
        cluster_method: str = DEFAULT_CLUSTER_METHOD,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        exclude_overlap: bool = True,
        verbose: bool = False,
    ):
        self.segmentation_path = segmentation_path
        self.embedding_path = embedding_path
        self.device = (device or "auto").lower()
        self.segmentation_batch_size = max(1, int(segmentation_batch_size))
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.cluster_threshold = float(cluster_threshold)
        self.cluster_method = str(cluster_method)
        self.min_cluster_size = int(min_cluster_size)
        self.exclude_overlap = bool(exclude_overlap)
        self.verbose = verbose

        self._seg_session = None
        self._emb_session = None
        self.providers: List[str] = []

        # 分割模型的输出几何信息（运行时从 ONNX 元数据/推理结果推断）
        self.frames_per_chunk: Optional[int] = None
        self.frame_duration: Optional[float] = None
        self.frame_step: Optional[float] = None
        self.num_classes: Optional[int] = None
        self.powerset_max_classes: Optional[int] = None
        self.powerset_mapping: Optional[np.ndarray] = None

        self._min_num_samples: Optional[int] = None

    # ------------------------------------------------------------------ #
    def _log(self, msg: str, on_status: Optional[Callable[[str], None]] = None):
        if self.verbose:
            print(f"[diar/onnx] {msg}")
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    @staticmethod
    def _providers_for(device: str) -> List[str]:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        wanted = _PROVIDER_CANDIDATES.get(device, _PROVIDER_CANDIDATES["auto"])
        chosen = [p for p in wanted if p in available]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        return chosen

    def _make_session(self, path: str):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        if "DmlExecutionProvider" in self.providers:
            # DirectML 上禁用并发/内存复用，避免与其它 onnxruntime 会话争抢
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(path, so, providers=self.providers)

    # ------------------------------------------------------------------ #
    def load(self, on_status: Optional[Callable[[str], None]] = None):
        """加载两个 ONNX 会话（重复调用只加载一次）"""
        if self._seg_session is not None and self._emb_session is not None:
            return

        for path in (self.segmentation_path, self.embedding_path):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"未找到 ONNX 模型: {path}\n"
                    "请先运行: uv run python 30-Export-Diarization-ONNX.py"
                )

        self.providers = self._providers_for(self.device)
        self._log(f"使用执行后端: {self.providers}")

        self._seg_session = self._make_session(self.segmentation_path)
        self._emb_session = self._make_session(self.embedding_path)

        self._read_segmentation_metadata()
        self._log(f"分割模型就绪 (设备 {self.providers[0]})", on_status)

    def _read_segmentation_metadata(self):
        """从 ONNX 图中读取帧率等几何信息（导出脚本写入的 metadata_props 优先）"""
        import onnx

        meta: Dict[str, str] = {}
        try:
            model = onnx.load(self.segmentation_path, load_external_data=False)
            meta = {p.key: p.value for p in model.metadata_props}
        except Exception:
            pass

        def _f(key, default):
            try:
                return float(meta[key])
            except (KeyError, ValueError):
                return default

        def _i(key, default):
            try:
                return int(float(meta[key]))
            except (KeyError, ValueError):
                return default

        self.frames_per_chunk = _i("frames_per_chunk", 589)
        self.frame_duration = _f("frame_duration", 0.0619375)
        self.frame_step = _f("frame_step", 0.016875)
        self.num_classes = _i("num_classes", 3)
        self.powerset_max_classes = _i("powerset_max_classes", 2)
        self.powerset_mapping = build_powerset_mapping(
            self.num_classes, self.powerset_max_classes
        )

        # 嵌入模型元数据
        try:
            emodel = onnx.load(self.embedding_path, load_external_data=False)
            emeta = {p.key: p.value for p in emodel.metadata_props}
            self._min_num_samples = int(float(emeta.get("min_num_samples", 0))) or None
        except Exception:
            self._min_num_samples = None

    # ------------------------------------------------------------------ #
    # 1. 分割
    # ------------------------------------------------------------------ #
    def _chunk_starts(self, num_samples: int) -> Tuple[List[int], bool]:
        """复刻 pyannote Inference.slide 的分片切法（不足一片时尾部补零）"""
        window = CHUNK_SAMPLES
        step = int(round(CHUNK_STEP_SECONDS * SAMPLE_RATE))
        if num_samples >= window:
            num_chunks = (num_samples - window) // step + 1
        else:
            num_chunks = 0
        has_last = (num_samples < window) or ((num_samples - window) % step > 0)
        return [c * step for c in range(num_chunks)], has_last

    def _segment(
        self,
        waveform: np.ndarray,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """滑动窗口分割，返回 (num_chunks, frames_per_chunk, num_classes) 的 0/1 多标签"""
        assert self._seg_session is not None
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        starts, has_last = self._chunk_starts(len(wav))

        chunks: List[np.ndarray] = []
        for s in starts:
            chunks.append(wav[s:s + CHUNK_SAMPLES])
        if has_last:
            tail = wav[len(starts) * int(round(CHUNK_STEP_SECONDS * SAMPLE_RATE)):]
            tail = tail[:CHUNK_SAMPLES]
            chunks.append(np.pad(tail, (0, CHUNK_SAMPLES - len(tail))))

        if not chunks:
            return np.zeros((0, self.frames_per_chunk or 589, self.num_classes or 3), np.float32)

        outputs: List[np.ndarray] = []
        total = len(chunks)
        for i in range(0, total, self.segmentation_batch_size):
            batch = np.stack(chunks[i:i + self.segmentation_batch_size])[:, None, :]
            logits = self._seg_session.run(["powerset"], {"waveform": batch})[0]
            outputs.append(logits)
            if on_status is not None:
                done = min(i + self.segmentation_batch_size, total)
                self._log(f"说话人分割 {done}/{total}", on_status)

        logits = np.concatenate(outputs, axis=0)                  # (C, F, num_powerset)
        # powerset -> 硬多标签（argmax -> 映射表 -> 0/1），等价于 Powerset.to_multilabel(soft=False)
        argmax = np.argmax(logits, axis=-1)                       # (C, F)
        return self.powerset_mapping[argmax].astype(np.float32)    # (C, F, num_classes)

    # ------------------------------------------------------------------ #
    # 2. 说话人计数
    # ------------------------------------------------------------------ #
    def _frame_grid(self) -> _FrameGrid:
        return _FrameGrid(0.0, self.frame_duration, self.frame_step)

    def _speaker_count(self, binarized: np.ndarray, num_chunks: int) -> np.ndarray:
        """逐帧瞬时说话人数，返回 (num_frames, 1) 的 uint8"""
        count = _aggregate(
            np.sum(binarized, axis=-1, keepdims=True),
            num_frames_per_chunk=self.frames_per_chunk,
            chunks_start=0.0,
            chunks_duration=CHUNK_SECONDS,
            chunks_step=CHUNK_STEP_SECONDS,
            frames=self._frame_grid(),
            skip_average=False,
            missing=0.0,
        )
        return np.rint(count).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # 3. 说话人嵌入
    # ------------------------------------------------------------------ #
    def _min_num_frames(self) -> int:
        """由 min_num_samples 换算出的最小有效帧数（与 pyannote 一致）"""
        min_samples = self._min_num_samples
        if not min_samples:
            return 1
        return int(math.ceil(self.frames_per_chunk * min_samples / CHUNK_SAMPLES))

    def _extract_embeddings(
        self,
        waveform: np.ndarray,
        binarized: np.ndarray,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """对每个 (分片, 局部说话人) 提取嵌入，返回 (num_chunks, num_speakers, dim)"""
        assert self._emb_session is not None
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)

        num_chunks, num_frames, num_speakers = binarized.shape
        min_num_frames = self._min_num_frames()

        if self.exclude_overlap:
            clean_frames = 1.0 * (np.sum(binarized, axis=2, keepdims=True) < 2)
            clean = binarized * clean_frames
        else:
            clean = binarized

        # 逐分片准备：(fbank, [每个说话人的权重])；同一分片只做一次 fbank
        dimension = self._emb_dimension()
        embeddings = np.full((num_chunks, num_speakers, dimension), np.nan, dtype=np.float32)

        step = int(round(CHUNK_STEP_SECONDS * SAMPLE_RATE))
        pending_fbank: List[np.ndarray] = []
        pending_weights: List[np.ndarray] = []
        pending_index: List[Tuple[int, int]] = []

        def flush():
            if not pending_fbank:
                return
            feats = np.stack(pending_fbank)
            weights = np.stack(pending_weights)
            out = self._emb_session.run(
                ["embedding"], {"fbank": feats, "weights": weights}
            )[0]
            for row, (c, s) in zip(out, pending_index):
                embeddings[c, s] = row
            pending_fbank.clear()
            pending_weights.clear()
            pending_index.clear()

        for c in range(num_chunks):
            start = c * step
            chunk = wav[start:start + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            fbank = compute_fbank(chunk)                          # (998, 80)
            if fbank.shape[0] != FBANK_FRAMES:
                fbank = fbank[:FBANK_FRAMES]
                if fbank.shape[0] < FBANK_FRAMES:
                    pad = np.zeros((FBANK_FRAMES - fbank.shape[0], fbank.shape[1]), np.float32)
                    fbank = np.concatenate([fbank, pad], axis=0)

            for s in range(num_speakers):
                mask = binarized[c, :, s]
                clean_mask = clean[c, :, s]
                used = clean_mask if float(np.sum(clean_mask)) > min_num_frames else mask
                weights = _nearest_upsample_1d(
                    np.nan_to_num(used, nan=0.0).astype(np.float32), FBANK_FRAMES
                )
                pending_fbank.append(fbank)
                pending_weights.append(weights)
                pending_index.append((c, s))

                if len(pending_index) >= self.embedding_batch_size:
                    flush()
                    if on_status is not None:
                        done = min(c + 1, num_chunks)
                        self._log(f"说话人嵌入 {done}/{num_chunks}", on_status)

        flush()
        return embeddings

    def _emb_dimension(self) -> int:
        if not hasattr(self, "_emb_dim") or self._emb_dim is None:
            out = self._emb_session.get_outputs()[0]
            shape = out.shape
            dim = int(shape[-1]) if isinstance(shape[-1], int) and shape[-1] > 0 else 256
            self._emb_dim = dim
        return self._emb_dim

    # ------------------------------------------------------------------ #
    # 4. 聚类
    # ------------------------------------------------------------------ #
    @staticmethod
    def _set_num_clusters(num_embeddings, num_clusters, min_clusters, max_clusters):
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, int(min_clusters)))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, int(max_clusters)))
        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters({min_clusters}) 不能大于 max_clusters({max_clusters})"
            )
        if min_clusters == max_clusters:
            num_clusters = min_clusters
        return num_clusters, min_clusters, max_clusters

    def _agglomerative_cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: Optional[int],
    ) -> np.ndarray:
        """复刻 pyannote AgglomerativeClustering.cluster"""
        from scipy.cluster.hierarchy import fcluster, linkage

        num_embeddings, _ = embeddings.shape
        min_cluster_size = min(
            self.min_cluster_size, max(1, round(0.1 * num_embeddings))
        )
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        if self.cluster_method in ("centroid", "median", "ward"):
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram = linkage(embeddings, method=self.cluster_method, metric="euclidean")
        else:
            dendrogram = linkage(embeddings, method=self.cluster_method, metric="cosine")

        clusters = fcluster(dendrogram, self.cluster_threshold, criterion="distance") - 1
        cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)

        if num_large_clusters < min_clusters:
            num_clusters = min_clusters
        elif num_large_clusters > max_clusters:
            num_clusters = max_clusters

        if num_clusters is not None and num_large_clusters != num_clusters:
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)
            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1

            for iteration in np.argsort(np.abs(dendrogram[:, 2] - self.cluster_threshold)):
                if _dendrogram[iteration, 3] < min_cluster_size:
                    continue
                clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)
                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters
                if num_large_clusters == num_clusters:
                    break

            if best_num_large_clusters != num_clusters:
                clusters = fcluster(_dendrogram, best_iteration, criterion="distance") - 1
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)

        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters

        large_centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in large_clusters]
        )
        small_centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in small_clusters]
        )
        from scipy.spatial.distance import cdist

        distances = cdist(large_centroids, small_centroids, metric="cosine")
        for i, j in enumerate(np.argmin(distances, axis=0)):
            clusters[clusters == small_clusters[i]] = large_clusters[j]

        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters

    def _filter_embeddings(self, embeddings: np.ndarray, binarized: np.ndarray):
        """复刻 BaseClustering.filter_embeddings（min_active_ratio=0.2）"""
        _, num_frames, _ = binarized.shape
        single_active_mask = np.sum(binarized, axis=2, keepdims=True) == 1
        num_clean_frames = np.sum(binarized * single_active_mask, axis=1)
        active = num_clean_frames >= MIN_ACTIVE_RATIO * num_frames
        valid = ~np.any(np.isnan(embeddings), axis=2)
        chunk_idx, speaker_idx = np.where(active * valid)
        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def _cluster(
        self,
        embeddings: np.ndarray,
        binarized: np.ndarray,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
    ) -> np.ndarray:
        """返回 (num_chunks, num_speakers) 的硬聚类编号"""
        from scipy.spatial.distance import cdist

        train, train_chunk_idx, train_speaker_idx = self._filter_embeddings(
            embeddings, binarized
        )
        num_embeddings = train.shape[0]
        if num_embeddings == 0:
            return np.zeros(embeddings.shape[:2], dtype=np.int8)

        num_clusters, min_clusters, max_clusters = self._set_num_clusters(
            num_embeddings, num_speakers, min_speakers, max_speakers
        )

        if max_clusters < 2:
            num_chunks, num_speakers_local, _ = embeddings.shape
            return np.zeros((num_chunks, num_speakers_local), dtype=np.int8)

        train_clusters = self._agglomerative_cluster(
            train, min_clusters, max_clusters, num_clusters
        )

        n_clusters = int(np.max(train_clusters)) + 1
        centroids = np.vstack(
            [np.mean(train[train_clusters == k], axis=0) for k in range(n_clusters)]
        )

        num_chunks, num_speakers_local, dim = embeddings.shape
        flat = embeddings.reshape(-1, dim)
        distances = cdist(flat, centroids, metric="cosine").reshape(
            num_chunks, num_speakers_local, n_clusters
        )
        soft = 2.0 - distances
        return np.argmax(soft, axis=2).astype(np.int8)

    # ------------------------------------------------------------------ #
    # 5. 时间轴重建
    # ------------------------------------------------------------------ #
    def _reconstruct(
        self,
        segmentations: np.ndarray,
        hard_clusters: np.ndarray,
        count: np.ndarray,
    ) -> np.ndarray:
        """复刻 SpeakerDiarization.reconstruct + to_diarization"""
        num_chunks, num_frames, _ = segmentations.shape
        num_clusters = int(np.max(hard_clusters)) + 1

        clustered = np.full((num_chunks, num_frames, num_clusters), np.nan, dtype=np.float32)
        for c in range(num_chunks):
            cluster = hard_clusters[c]
            for k in np.unique(cluster):
                if k == -2:
                    continue
                clustered[c, :, k] = np.max(segmentations[c][:, cluster == k], axis=1)

        activations = _aggregate(
            clustered,
            num_frames_per_chunk=num_frames,
            chunks_start=0.0,
            chunks_duration=CHUNK_SECONDS,
            chunks_step=CHUNK_STEP_SECONDS,
            frames=self._frame_grid(),
            skip_average=True,
            missing=0.0,
        )

        _, num_speakers = activations.shape
        max_per_frame = int(np.max(count))
        if num_speakers < max_per_frame:
            activations = np.pad(
                activations, ((0, 0), (0, max_per_frame - num_speakers))
            )

        length = min(len(activations), len(count))
        activations = activations[:length]
        count = count[:length]

        # 每帧只保留激活度最高的 count 个说话人，其余置 0
        order = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t in range(length):
            for i in range(int(count[t, 0])):
                binary[t, order[t, i]] = 1.0
        return binary

    def _to_segments(self, binary: np.ndarray, min_duration_on: float = 0.0,
                     min_duration_off: float = 0.0) -> List[Tuple[str, float, float]]:
        """复刻 to_annotation：对每个说话人做迟滞二值化，得到 (label, start, end)"""
        grid = self._frame_grid()
        num_frames, num_speakers = binary.shape
        if num_frames == 0:
            return []

        timestamps = [grid.middle(i) for i in range(num_frames)]
        onset, offset = DEFAULT_ONSET, DEFAULT_ONSET
        out: List[Tuple[str, float, float]] = []

        for k in range(num_speakers):
            scores = binary[:, k]
            start = timestamps[0]
            is_active = bool(scores[0] > onset)
            last_t = timestamps[0]
            for t, y in zip(timestamps[1:], scores[1:]):
                last_t = t
                if is_active:
                    if y < offset:
                        out.append((f"SPEAKER_{k:02d}", start, t))
                        start = t
                        is_active = False
                else:
                    if y > onset:
                        start = t
                        is_active = True
            if is_active:
                out.append((f"SPEAKER_{k:02d}", start, last_t))

        # min_duration_on / min_duration_off 为 0 时无需后处理（与 3.1 配置一致）
        if min_duration_on > 0:
            out = [s for s in out if s[2] - s[1] >= min_duration_on]
        if min_duration_off > 0:
            out = self._fill_gaps(out, min_duration_off)
        return out

    @staticmethod
    def _fill_gaps(spans: List[Tuple[str, float, float]], collar: float):
        """填补同一说话人的短静音（等价 Annotation.support + collar）"""
        by_label: Dict[str, List[Tuple[float, float]]] = {}
        for label, s, e in spans:
            by_label.setdefault(label, []).append((s, e))
        out: List[Tuple[str, float, float]] = []
        for label, runs in by_label.items():
            runs.sort()
            merged: List[List[float]] = []
            for s, e in runs:
                if merged and s - merged[-1][1] <= collar:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            out.extend((label, s, e) for s, e in merged)
        out.sort(key=lambda x: (x[1], x[2]))
        return out

    # ------------------------------------------------------------------ #
    # 对外主入口
    # ------------------------------------------------------------------ #
    def diarize(
        self,
        waveform: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 20,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> DiarizationResult:
        """对整段波形做说话人分离，返回 DiarizationResult（标签已归一化为 spk0/spk1...）"""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"ONNX 说话人分离仅支持 {SAMPLE_RATE}Hz，收到 {sample_rate}Hz")

        self.load(on_status=on_status)
        t0 = time.time()

        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return DiarizationResult(segments=[], labels=[], performance={"total_time": 0.0})

        # 1) 分割
        self._log("说话人识别: 分割 ...", on_status)
        binarized = self._segment(wav, on_status=on_status)
        num_chunks = binarized.shape[0]
        if num_chunks == 0:
            return DiarizationResult(segments=[], labels=[], performance={"total_time": 0.0})

        # 2) 计数（提前退出：整段没有语音）
        count = self._speaker_count(binarized, num_chunks)
        if int(np.max(count)) == 0:
            self._log("未检测到任何语音段", on_status)
            return DiarizationResult(
                segments=[], labels=[], performance={"total_time": time.time() - t0}
            )

        # 说话人数归一化（对齐 pyannote set_num_speakers）
        min_speakers_eff = num_speakers or min_speakers or 1
        max_speakers_eff = num_speakers or max_speakers or 20
        count = np.minimum(count, max_speakers_eff).astype(np.uint8)

        # 3) 嵌入 + 聚类
        self._log("说话人识别: 嵌入 ...", on_status)
        embeddings = self._extract_embeddings(wav, binarized, on_status=on_status)
        self._log("说话人识别: 聚类 ...", on_status)
        hard_clusters = self._cluster(
            embeddings, binarized, num_speakers, min_speakers_eff, max_speakers_eff
        )

        # 4) 非活跃说话人丢到专用簇 -2
        inactive = np.sum(binarized, axis=1) == 0
        hard_clusters = hard_clusters.copy()
        hard_clusters[inactive] = -2

        # 5) 重建独占时间轴（每帧只保留一个说话人，与 pyannote exclusive 输出一致）
        exclusive_count = np.minimum(count, 1).astype(np.uint8)
        binary = self._reconstruct(binarized, hard_clusters, exclusive_count)
        raw_spans = self._to_segments(binary)
        spans = [(label, s, e) for label, s, e in raw_spans if e > s]

        # 6) 归一化标签（按首次出现顺序 -> spk0/spk1...）
        spans.sort(key=lambda x: (x[1], x[2]))
        label_map: Dict[str, str] = {}
        segments: List[SpeakerSegment] = []
        for label, s, e in spans:
            if label not in label_map:
                label_map[label] = f"spk{len(label_map)}"
            segments.append(
                SpeakerSegment(
                    start_time=round(float(s), 3),
                    end_time=round(float(e), 3),
                    speaker=label,
                    label=label_map[label],
                )
            )

        labels = list(dict.fromkeys(seg.label for seg in segments))
        elapsed = time.time() - t0
        self._log(
            f"说话人识别完成: 共 {len(labels)} 位说话人 / {len(segments)} 个语音段，"
            f"耗时 {elapsed:.2f}s",
            on_status,
        )
        return DiarizationResult(
            segments=segments,
            labels=labels,
            performance={
                "total_time": elapsed,
                "device": self.providers[0] if self.providers else "unknown",
                "backend": "onnxruntime",
            },
        )


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def onnx_models_available(model_dir: str) -> bool:
    """检查 ONNX 说话人分离模型是否已导出"""
    base = Path(model_dir)
    return (base / SEGMENTATION_ONNX).exists() and (base / EMBEDDING_ONNX).exists()


def build_onnx_engine(config: DiarizationConfig) -> Optional[OnnxDiarizationEngine]:
    """按配置创建 ONNX 引擎；模型不存在时返回 None"""
    model_dir = getattr(config, "model_dir", None) or str(Path(__file__).parent.parent.parent / "model")
    if not onnx_models_available(model_dir):
        return None
    return OnnxDiarizationEngine(
        segmentation_path=str(Path(model_dir) / SEGMENTATION_ONNX),
        embedding_path=str(Path(model_dir) / EMBEDDING_ONNX),
        device=config.device or "auto",
        exclude_overlap=True,
    )
