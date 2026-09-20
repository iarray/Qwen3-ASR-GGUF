# 项目长期记忆 —— Qwen3-ASR-GGUF

## 基本约定
- 运行环境：Windows 11 + Python **3.14** + **uv** 管理依赖（`uv sync` / `uv add`）。
- 模型放在 `model/`：`qwen3_asr_llm.q4_k.gguf`、`qwen3_asr_encoder_{frontend,backend}.int4.onnx`、
  `qwen3_aligner_llm.q4_k.gguf`、`qwen3_aligner_encoder_{frontend,backend}.int4.onnx`。
- **说话人识别模型**也在 `model/`：`diarization_segmentation.onnx`（5.9 MB）、
  `diarization_embedding.onnx`（26.5 MB）。由 `30-Export-Diarization-ONNX.py` 一次性导出，
  运行期不依赖 torch / pyannote / HF Token。`model/` 已被 gitignore。
- llama.cpp 的 DLL 放在 `qwen_asr_gguf/inference/bin/`（已 gitignore）。
- 依赖 ffmpeg 处理 mp4/mkv/m4a 等格式（`qwen_asr_gguf/inference/audio.py` 会走 ffmpeg 分支）。

## 对外能力（按需求.txt 实现）
- **命令行**：`python transcribe.py <文件...> [--speakers N] [--diar-backend onnx|pyannote|auto] [--max-line-duration 6]`
- **图形界面**：`python app.py`（ttkbootstrap），界面配置存 `ui_settings.json`
- **Python API**：`qwen_asr_gguf.pipeline.TranscriptionPipeline`
- **导出说话人 ONNX 模型**：`python 30-Export-Diarization-ONNX.py`（需 torch + pyannote.audio，仅一次）

## 字幕规范（重要，勿随意更改）
- 每行字幕 **尽量 ≤ 6 秒**（`max_line_duration`），超出则按标点强制切分。
- 断句靠标点：句末标点必换行，行过短（< 1s）时允许顺延合并；说话人切换必定换行。
- 每条字幕以 `[spk0]` 开头（`spk` + 首次出现顺序编号）；无说话人识别时不加前缀。
- **英文空格不能丢**：对齐器把词间空格输出成独立的空白项，聚合从句时必须保留，
  拼接时用 `subtitle._smart_join()`（仅两侧都是 ASCII 词字符才补空格），否则英文会粘成一串。
- **UI 布局**：主操作按钮固定放在文件工具条（不要只放窗口最底部），
  且底部条要先 `pack(side=BOTTOM)` 占位，输出区最后 pack，避免窗口变矮时按钮被挤没。
- 输出位置：与源音视频**同目录、同文件名**，扩展名 `.srt`（另有 `.txt` / `.json`）。
- JSON 导出保持**列表**格式（每项 text/start/end[/speaker]），不要改成对象包裹，避免破坏下游脚本。

## 代码结构约定
- `qwen_asr_gguf/inference/`：引擎层（asr / aligner / encoder / diarization(+_onnx) / subtitle / exporters）。
- `qwen_asr_gguf/pipeline.py`：编排层，GUI 与 CLI 都通过它调用。
- 引擎不直接 print 面向用户的进度：统一走 `on_stream / on_chunk / on_status` 回调。
- 说话人识别默认开启，但**失败必须降级**为纯文本字幕，且只报一次错，不阻塞其它文件。

## 说话人识别后端（双后端，2026-09-20 新增）
- **默认 `auto`**：本地有 ONNX 模型就走 `onnx`，否则回退 `pyannote`。
- `onnx` 后端（推荐）：纯 onnxruntime，`DiarizationConfig.device` 支持 `dml` / `cuda` / `cpu`
  （`auto` 时按 DML → CUDA → CPU 顺序探测）。**不需要 HF Token**。
  实现在 `inference/diarization_onnx.py`，出口是 `OnnxDiarizationEngine.diarize()`。
- `pyannote` 后端：原生 PyTorch，模型 gated，**需要 HF Token**。
- 两者输出结构完全一致（`DiarizationResult`，标签归一化为 `spk0/spk1...`），
  实测 30s 样本帧级 IoU = 1.0000、说话人标签一致率 = 1.0000。
- 队列入口统一是 `SpeakerDiarizer.diarize_waveform()/diarize_file()`，内部按 `backend` 分派；
  新增后端只需在门面加一个分支，pipeline / GUI / CLI 无需改动。
- **HF Token 检查必须按后端区分**：`pipeline.diarization_needs_hf_token(config)` 判断，
  ONNX 后端返回 False（GUI 因此不会误弹 Token 警告）。

## 已知限制
- pyannote 后端需要 HuggingFace Token（模型 gated）；ONNX 后端不需要。
- AMD 显卡上 pyannote 默认走 CPU（除非装 torch-directml 且能被 pyannote 接受）；
  想要 GPU 请用 `--diar-backend onnx --spk-device dml`。
- 沙箱环境下 ffmpeg 不在 PATH，非 wav 输入无法在自动化测试中验证。

## 坑位（务必遵守）
- **DirectML 下绝不能给编码器喂全零 hidden_states**：`QwenAudioEncoder._run_backend()`
  的 DML 定长补齐必须**复制末帧**而非零填充，否则 DML 内部归一化算子产出 NaN 并污染整段输出，
  最终表现为**字幕时间戳全部为 0**、说话人前缀也一起消失。
  见 `test_dml_encoder_padding.py`。
- `torch.onnx.export` 必须显式 `dynamo=False`，否则忽略 `dynamic_axes` 且慢约 150 倍。
- numpy 复刻 Kaldi fbank 时 **必须先 `×32768`**，它决定 `log` 的截断位置。
- `build.spec` 的 `hiddenimports` 需手动登记惰性导入的包（`onnxruntime`、`scipy.*`）。

## 测试
- `python test_subtitle_speaker.py`：字幕断句 / 说话人合并的离线单测（无需模型），22 项。
- `python test_dml_encoder_padding.py`：DML 编码器补齐分支 NaN 回归（需模型，无 DML 会自动跳过），6 项。
- 真实语音冒烟可用 `pyannote/audio/sample/sample.wav`（30 秒双人英文），
  期望结果：2 位说话人、13 条字幕、首条 `00:00:06,720 --> 00:00:08,480 [spk1] Hello. Hello.`
