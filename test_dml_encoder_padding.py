# coding=utf-8
"""
test_dml_encoder_padding.py - 回归测试：DirectML 下补齐分支不得产生 NaN

背景
----
`QwenAudioEncoder._run_backend` 在 DirectML 下会把 hidden_states 补齐到固定长度
(`dml_pad_to` 秒)，以获得固定形状、避免每次重规划。原实现用**零填充**，
而 DirectML 后端收到全零 hidden_states 时内部归一化算子会算出 NaN，
且 NaN 会污染整段输出。

后果：`--provider DML`（默认值）下，只要送进编码器的音频短于 `dml_pad_to` 秒，
编码结果就全是 NaN。ASR 主流程恰好每片都补齐到整 40 秒所以看不出来，
但对齐器传入的是真实语音切片（通常不足 40 秒），于是
`np.argmax` 在 NaN 上恒返回 0 → **所有字幕时间戳变成 0**。

修复：补齐时复制末帧，而不是补零。
本测试确保该行为不再回退。

用法：
    uv run python test_dml_encoder_padding.py
无需 GPU 时也能运行（DML 不可用会自动跳过 DML 部分，仅跑 CPU 一致性检查）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

MODEL_DIR = Path(__file__).parent / "model"
FE = MODEL_DIR / "qwen3_asr_encoder_frontend.int4.onnx"
BE = MODEL_DIR / "qwen3_asr_encoder_backend.int4.onnx"

_passed = 0
_failed = 0


def check(name: str, cond: bool, extra: str = ""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name} {extra}")
    else:
        _failed += 1
        print(f"  [FAIL] {name} {extra}")


def main() -> int:
    if not FE.exists() or not BE.exists():
        print(f"[跳过] 未找到编码器模型：{FE.name} / {BE.name}")
        return 0

    import onnxruntime as ort

    from qwen_asr_gguf.inference.encoder import QwenAudioEncoder

    avail = ort.get_available_providers()
    has_dml = "DmlExecutionProvider" in avail
    print(f"可用执行提供器: {avail}\n")

    # 用确定性伪噪声，避免随机静音触发偶发分支
    rng = np.random.default_rng(0)
    audio_short = (rng.random(int(16000 * 12.0)).astype(np.float32) - 0.5) * 0.2
    audio_full = (rng.random(int(16000 * 40.0)).astype(np.float32) - 0.5) * 0.2

    # ---------------- [1] CPU 基线 ---------------- #
    print("[1] CPU 基线（不触发 DML 补齐分支）")
    enc_cpu = QwenAudioEncoder(str(FE), str(BE), onnx_provider="CPU", dml_pad_to=40, verbose=False)
    out_cpu_short, _ = enc_cpu.encode(audio_short)
    check("CPU 短音频无 NaN", not np.isnan(out_cpu_short).any(),
          f"shape={out_cpu_short.shape}")

    # ---------------- [2] DML 补齐分支 ---------------- #
    if not has_dml:
        print("\n[2] 未检测到 DirectML，跳过 DML 补齐分支检查")
    else:
        print("\n[2] DML 补齐分支（seq_len < h_target_len）")
        enc_dml = QwenAudioEncoder(str(FE), str(BE), onnx_provider="DML", dml_pad_to=40, verbose=False)

        out_dml_short, _ = enc_dml.encode(audio_short)
        check("DML 短音频无 NaN（本次修复点）",
              not np.isnan(out_dml_short).any(),
              f"shape={out_dml_short.shape} nan_ratio={np.isnan(out_dml_short).mean():.3f}")

        check("DML 短音频与 CPU 形状一致",
              out_dml_short.shape == out_cpu_short.shape,
              f"{out_dml_short.shape} vs {out_cpu_short.shape}")

        if not np.isnan(out_dml_short).any():
            delta = np.abs(out_dml_short - out_cpu_short)
            check("DML 与 CPU 数值接近", float(delta.max()) < 5e-3,
                  f"max|Δ|={delta.max():.6f} mean|Δ|={delta.mean():.8f}")

        # 长度恰好等于 h_target_len 时不走补齐分支，也必须正常
        out_dml_full, _ = enc_dml.encode(audio_full)
        check("DML 整 40 秒无 NaN（不走补齐分支）",
              not np.isnan(out_dml_full).any(), f"shape={out_dml_full.shape}")

        # 边界：极短音频（远小于 1 秒的窗口）
        out_dml_tiny, _ = enc_dml.encode(audio_short[:1600])
        check("DML 极短音频（0.1s）无 NaN",
              not np.isnan(out_dml_tiny).any(), f"shape={out_dml_tiny.shape}")

    print("\n" + "=" * 50)
    print(f"通过 {_passed} 项，失败 {_failed} 项")
    print("=" * 50)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
