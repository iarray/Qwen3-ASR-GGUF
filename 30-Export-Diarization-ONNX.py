# coding=utf-8
"""
30-Export-Diarization-ONNX.py - 把 pyannote 说话人分离流水线的两个神经网络导出为 ONNX

pyannote 原生实现完全基于 PyTorch（CPU/CUDA），本项目希望改走 onnxruntime
（DirectML / CUDA / CPU），因此需要把模型导出一次：

    pyannote/segmentation-3.0                (PyanNet, powerset) -> diarization_segmentation.onnx
    pyannote/wespeaker-voxceleb-resnet34-LM  (ResNet34 + 统计池化) -> diarization_embedding.onnx

导出后推理端（qwen_asr_gguf/inference/diarization_onnx.py）只依赖 onnxruntime + numpy + scipy，
不再需要 torch / pyannote.audio。

用法：
    uv run python 30-Export-Diarization-ONNX.py                   # 导出到 model/
    uv run python 30-Export-Diarization-ONNX.py --out-dir model --verify

注意：
- 首次运行需要联网下载 pyannote 模型到 HuggingFace 缓存（segmentation-3.0 公开，
  wespeaker 嵌入模型公开；speaker-diarization-3.1 需要授权但本脚本不需要它）。
- 导出用的 fbank 特征由 torchaudio 计算，推理端用等价的 numpy 实现，两者已逐位对齐。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJ_DIR = Path(__file__).parent

# 与推理端共享的常量（改动需同步 diarization_onnx.py）
SEG_CHECKPOINT = "pyannote/segmentation-3.0"
EMB_CHECKPOINT = "pyannote/wespeaker-voxceleb-resnet34-LM"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 10.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

SEG_ONNX_NAME = "diarization_segmentation.onnx"
EMB_ONNX_NAME = "diarization_embedding.onnx"


def _write_metadata(path: Path, values: dict):
    """把常量写入 ONNX metadata_props，推理端读它来对齐帧率等几何参数"""
    import onnx

    model = onnx.load(str(path))
    existing = {p.key: p for p in model.metadata_props}
    for key, value in values.items():
        if key in existing:
            existing[key].value = str(value)
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = str(value)
    onnx.save_model(model, str(path))


def _compute_min_num_samples(model, sample_rate: int = SAMPLE_RATE) -> int:
    """复刻 pyannote 的二分搜索，求嵌入模型能接受的最短波形样点数"""
    import torch

    lower, upper = 2, round(0.5 * sample_rate)
    middle = (lower + upper) // 2
    with torch.inference_mode():
        while lower + 1 < upper:
            try:
                _ = model(torch.randn(1, 1, middle))
                upper = middle
            except Exception:
                lower = middle
            middle = (lower + upper) // 2
    return upper


def _log(msg: str):
    print(f"[导出] {msg}", flush=True)


def _patch_asteroid_clamp():
    """修补 asteroid_filterbanks 里的 torch.clamp

    ParamSincFB.filters() 使用 `torch.clamp(x, min_low_hz, sample_rate/2)`，
    其中 scalars 与 tensor 混用，torch.onnx 导出器无法解析该重载（报
    "Cannot cast FakeTensor to number"）。改用 maximum/minimum 语义完全等价。
    """
    import torch
    import asteroid_filterbanks.param_sinc_fb as psf

    def _safe_filters(self):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        raw = low + self.min_band_hz + torch.abs(self.band_hz_)
        lo_bound = torch.full_like(raw, float(self.min_low_hz))
        hi_bound = torch.full_like(raw, float(self.sample_rate) / 2.0)
        high = torch.minimum(torch.maximum(raw, lo_bound), hi_bound)
        return torch.cat(
            [self.make_filters(low, high, "cos"), self.make_filters(low, high, "sin")],
            dim=0,
        )

    psf.ParamSincFB.filters = _safe_filters


def _inline_external_data(path: Path):
    """把 ONNX 的外部权重文件合并回单个 .onnx（torch 导出默认拆成 .onnx + .onnx.data）"""
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    data_file = Path(str(path) + ".data")
    model = onnx.load(str(path))  # 会自动加载外部数据
    if data_file.exists():
        convert_model_from_external_data(model)
        onnx.save_model(model, str(path), save_as_external_data=False)
        data_file.unlink()
        return True
    return False


# --------------------------------------------------------------------------- #
# 分割模型
# --------------------------------------------------------------------------- #
def export_segmentation(out_path: Path, opset: int = 17) -> int:
    import torch
    from pyannote.audio import Model

    _patch_asteroid_clamp()

    _log(f"加载 {SEG_CHECKPOINT} ...")
    model = Model.from_pretrained(SEG_CHECKPOINT)
    model.eval()

    spec = next(iter(model.specifications))
    if not spec.powerset:
        raise RuntimeError(f"{SEG_CHECKPOINT} 不是 powerset 模型，本脚本仅支持 powerset 分割模型")
    _log(
        f"规格: 时长 {spec.duration}s, 类别 {spec.classes}, "
        f"powerset max_classes={spec.powerset_max_classes}"
    )

    dummy = torch.randn(1, 1, CHUNK_SAMPLES)
    t0 = time.time()
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["waveform"],
        output_names=["powerset"],
        opset_version=opset,
        do_constant_folding=True,
        # 用旧版 TorchScript 导出器：新版 dynamo 导出器会忽略 dynamic_axes，
        # 把 batch 维固定成 1（批量推理直接报 Reshape 错误），且耗时高两个数量级。
        dynamo=False,
        dynamic_axes={"waveform": {0: "batch"}, "powerset": {0: "batch"}},
    )
    _log(f"导出完成，耗时 {time.time() - t0:.1f}s")
    _inline_external_data(out_path)
    _log(f"文件大小 {out_path.stat().st_size / 1e6:.1f} MB")

    # 记录帧率信息，供推理端校验
    with torch.no_grad():
        frames = model(dummy).shape[1]
    rf = model.receptive_field
    _write_metadata(
        out_path,
        {
            "checkpoint": SEG_CHECKPOINT,
            "chunk_seconds": spec.duration,
            "frames_per_chunk": int(frames),
            "frame_duration": float(rf.duration),
            "frame_step": float(rf.step),
            "num_classes": len(spec.classes),
            "powerset_max_classes": int(spec.powerset_max_classes),
            "sample_rate": SAMPLE_RATE,
        },
    )
    _log(f"每 10s 分片输出 {frames} 帧（帧时长 {rf.duration:.6f}s / 帧步长 {rf.step:.6f}s）")
    return {
        "checkpoint": SEG_CHECKPOINT,
        "chunk_seconds": spec.duration,
        "frames_per_chunk": int(frames),
        "frame_duration": float(rf.duration),
        "frame_step": float(rf.step),
        "num_classes": len(spec.classes),
        "powerset_max_classes": int(spec.powerset_max_classes),
        "sample_rate": SAMPLE_RATE,
    }


# --------------------------------------------------------------------------- #
# 嵌入模型
# --------------------------------------------------------------------------- #
class _ResNetWrapper:
    """把 WeSpeakerResNet34 包成 (fbank, weights) -> embedding"""

    @staticmethod
    def build(module):
        import torch.nn as nn

        class _W(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, fbank, weights):
                return self.m(fbank, weights=weights)[1]

        return _W(module)


def export_embedding(out_path: Path, opset: int = 17):
    import torch
    from pyannote.audio import Model

    _log(f"加载 {EMB_CHECKPOINT} ...")
    model = Model.from_pretrained(EMB_CHECKPOINT)
    model.eval()
    resnet = model.resnet.eval()

    # fbank 帧数由波形长度决定：1 + (N - 400) // 160
    n_frames = 1 + (CHUNK_SAMPLES - 400) // 160
    _log(f"chunk={CHUNK_SECONDS}s -> fbank {n_frames} 帧 x 80 维")

    wrapper = _ResNetWrapper.build(resnet).eval()
    dummy_fbank = torch.randn(1, n_frames, 80)
    dummy_weights = torch.ones(1, n_frames)

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy_fbank, dummy_weights),
        str(out_path),
        input_names=["fbank", "weights"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "fbank": {0: "batch"},
            "weights": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )
    _log(f"导出完成，耗时 {time.time() - t0:.1f}s")
    _inline_external_data(out_path)
    _log(f"文件大小 {out_path.stat().st_size / 1e6:.1f} MB")

    _log("计算 min_num_samples（二分搜索）...")
    min_num_samples = _compute_min_num_samples(model)

    with torch.no_grad():
        dim = resnet(dummy_fbank, weights=dummy_weights)[1].shape[-1]
    _write_metadata(
        out_path,
        {
            "checkpoint": EMB_CHECKPOINT,
            "chunk_seconds": CHUNK_SECONDS,
            "fbank_frames": int(n_frames),
            "num_mel_bins": 80,
            "dimension": int(dim),
            "min_num_samples": int(min_num_samples),
        },
    )
    _log(f"嵌入维度 {dim}，min_num_samples = {min_num_samples}")
    return {
        "checkpoint": EMB_CHECKPOINT,
        "chunk_seconds": CHUNK_SECONDS,
        "fbank_frames": int(n_frames),
        "num_mel_bins": 80,
        "dimension": int(dim),
        "min_num_samples": int(min_num_samples),
    }


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def verify(seg_path: Path, emb_path: Path):
    """与 PyTorch 输出逐位对比，确认导出无损"""
    import soundfile as sf
    import torch
    import onnxruntime as ort
    from pyannote.audio import Model

    _patch_asteroid_clamp()
    so = ort.SessionOptions()
    so.log_severity_level = 3

    sample = Path(sys.prefix) / "Lib/site-packages/pyannote/audio/sample/sample.wav"
    if not sample.exists():
        import pyannote.audio
        sample = Path(pyannote.audio.__file__).parent / "sample" / "sample.wav"

    wav, _sr = sf.read(str(sample), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav[:CHUNK_SAMPLES] if len(wav) >= CHUNK_SAMPLES else np.pad(wav, (0, CHUNK_SAMPLES - len(wav)))
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    tw = torch.from_numpy(wav)[None, None]

    print()
    _log("=== 数值校验（对比 PyTorch 输出）===")

    seg_model = Model.from_pretrained(SEG_CHECKPOINT).eval()
    sess = ort.InferenceSession(str(seg_path), so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = seg_model(tw).numpy()
    out = sess.run(["powerset"], {"waveform": tw.numpy()})[0]
    agree = float((ref.argmax(-1) == out.argmax(-1)).mean())
    _log(f"分割: 形状 {ref.shape} | max|Δ| = {np.abs(ref - out).max():.3e} | argmax 一致率 = {agree:.4f}")

    emb_model = Model.from_pretrained(EMB_CHECKPOINT).eval()
    resnet = emb_model.resnet.eval()
    n_frames = 1 + (CHUNK_SAMPLES - 400) // 160
    sess2 = ort.InferenceSession(str(emb_path), so, providers=["CPUExecutionProvider"])
    fb = emb_model.compute_fbank(tw)[0].numpy()
    k = np.ones((1, n_frames), dtype=np.float32)
    with torch.no_grad():
        ref2 = resnet(torch.from_numpy(fb)[None], weights=torch.from_numpy(k))[1].numpy()
    out2 = sess2.run(["embedding"], {"fbank": fb[None], "weights": k})[0]
    cos = float(np.dot(ref2[0], out2[0]) / (np.linalg.norm(ref2[0]) * np.linalg.norm(out2[0])))
    _log(f"嵌入: 形状 {ref2.shape} | max|Δ| = {np.abs(ref2 - out2).max():.3e} | 余弦相似 = {cos:.6f}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="把 pyannote 说话人分离模型导出为 ONNX")
    parser.add_argument("--out-dir", default=str(PROJ_DIR / "model"), help="输出目录（默认 model/）")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本（默认 17）")
    parser.add_argument("--verify", action="store_true", help="导出后与 PyTorch 输出做数值校验")
    parser.add_argument("--force", action="store_true", help="已存在时也重新导出")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / SEG_ONNX_NAME
    emb_path = out_dir / EMB_ONNX_NAME

    print("=" * 72)
    print("pyannote 说话人分离 -> ONNX 导出")
    print("=" * 72)

    if seg_path.exists() and emb_path.exists() and not args.force:
        _log("两个 ONNX 模型均已存在，跳过导出（加 --force 可强制重新导出）")
    else:
        if not seg_path.exists() or args.force:
            export_segmentation(seg_path, opset=args.opset)
        if not emb_path.exists() or args.force:
            export_embedding(emb_path, opset=args.opset)

    if args.verify:
        verify(seg_path, emb_path)

    print()
    _log(f"完成：{seg_path}")
    _log(f"完成：{emb_path}")
    print()
    print("提示：导出后运行 `uv run python app.py` 或 `uv run python transcribe.py xxx.wav`，")
    print("      说话人识别会自动优先使用 ONNX 后端（DirectML / CUDA / CPU）。")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
