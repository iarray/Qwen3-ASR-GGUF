# coding=utf-8
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# 获取项目根目录 (适配打包环境)
if getattr(sys, 'frozen', False):
    # 打包环境：sys.executable 位于 dist/Project/ 根目录
    PROJ_DIR = Path(sys.executable).parent
else:
    # 源码环境
    PROJ_DIR = Path(__file__).parent


import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from qwen_asr_gguf.pipeline import (
    SUBTITLE_FILE_EXTS,
    PipelineConfig,
    TranscriptionPipeline,
    build_diarizer,
    build_engine,
    build_translator,
    diarization_needs_hf_token,
    find_missing_models,
    find_missing_translation_model,
    is_subtitle_file,
    resolve_diarization_backend,
)
from qwen_asr_gguf.inference.translator import (
    DEFAULT_TRANSLATION_PROMPT,
    describe_backends as describe_translation_backends,
    resolve_translation_backend,
)

app = typer.Typer(help="Qwen3-ASR GGUF 命令行转录工具 (支持说话人识别与 SRT 字幕)", add_completion=False)
console = Console()

# 输入里含字幕文件时的提示（字幕文件不参与识别，只翻译）
_SUBTITLE_EXTS_HINT = "、".join(SUBTITLE_FILE_EXTS)


def check_model_files(config: PipelineConfig, need_asr: bool = True):
    """检查模型文件完整性

    Args:
        need_asr: 为 False 时（输入全是 .srt 字幕）不校验识别模型，只校验翻译模型
    """
    missing_files = []
    if need_asr:
        missing_files += find_missing_models(config.model_dir, config.precision, config.enable_aligner)
    missing_files += find_missing_translation_model(config)

    if missing_files:
        console.print("\n[bold red]错误：找不到以下所需模型文件：[/bold red]")
        for f in missing_files:
            console.print(f"  - {f}")
        console.print("\n[bold yellow]请到以下链接下载模型文件，并解压到 model 目录：[/bold yellow]")
        console.print("[blue]https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models[/blue]\n")
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    files: List[Path] = typer.Argument(
        ...,
        help=f"要处理的文件：音视频（识别+可选翻译），或字幕文件（{_SUBTITLE_EXTS_HINT}，只翻译）",
    ),
    
    # 组 1: 模型与硬件
    model_dir: str = typer.Option(str(PROJ_DIR / "model"), "--model-dir", "-m", help="模型权重根目录", rich_help_panel="模型配置"),
    precision: str = typer.Option("int4", "--prec", help="编码器精度: fp32, fp16, int8, int4", rich_help_panel="模型配置"),
    timestamp: bool = typer.Option(True, "--timestamp/--no-ts", help="是否开启时间戳引擎", rich_help_panel="模型配置"),
    onnx_provider: str = typer.Option("DML", "--provider", help="ONNX 执行后端: CPU, CUDA, DML, TRT", rich_help_panel="模型配置"),
    llm_use_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="LLM 是否使用 GPU 加速", rich_help_panel="模型配置"),
    use_vulkan: bool = typer.Option(True, "--vulkan/--no-vulkan", help="是否开启 Vulkan 加速 (设置 GGML_VULKAN=1)", rich_help_panel="模型配置"),
    n_ctx: int = typer.Option(2048, "--n-ctx", help="LLM 上下文窗口大小", rich_help_panel="模型配置"),
    
    # 组 2: 转录逻辑
    language: Optional[str] = typer.Option(None, "--language", "-l", help="强制指定语种 (例: Chinese, English)", rich_help_panel="转录设置"),
    context: str = typer.Option("", "--context", help="上下文提示词 (Prompt)", rich_help_panel="转录设置"),
    temperature: float = typer.Option(0.4, "--temperature", help="采样温度", rich_help_panel="转录设置"),

    # 组 3: 说话人识别
    speakers: int = typer.Option(0, "--speakers", "-s", help="说话人数；0 表示自动检测", rich_help_panel="说话人识别"),
    max_speakers: int = typer.Option(20, "--max-speakers", help="自动检测时的说话人上限", rich_help_panel="说话人识别"),
    no_diarization: bool = typer.Option(False, "--no-diarization", help="关闭说话人识别", rich_help_panel="说话人识别"),
    diarization_backend: str = typer.Option("auto", "--diar-backend", help="说话人推理后端: auto, onnx, pyannote", rich_help_panel="说话人识别"),
    diarization_device: str = typer.Option("auto", "--spk-device", help="说话人识别设备: auto, cpu, cuda, dml (onnx 后端才支持 dml)", rich_help_panel="说话人识别"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HuggingFace Token (仅 pyannote 后端需要)", rich_help_panel="说话人识别"),

    # 组 4: 字幕
    max_line_duration: float = typer.Option(6.0, "--max-line-duration", help="每行字幕最长时长(秒)", rich_help_panel="字幕设置"),
    max_line_chars: int = typer.Option(48, "--max-line-chars", help="每行字幕最长字数", rich_help_panel="字幕设置"),

    # 组 4.5: 字幕翻译
    translate: bool = typer.Option(False, "--translate/--no-translate", help="开启字幕翻译（基于 Qwen3-8B GGUF）", rich_help_panel="字幕翻译"),
    translate_model: str = typer.Option("Qwen3-8B-Q4_K_M.gguf", "--translate-model", help="翻译模型文件名（放在模型目录下）", rich_help_panel="字幕翻译"),
    translate_backend: str = typer.Option("auto", "--translate-backend", help="翻译加速后端: auto, vulkan, cuda, cpu", rich_help_panel="字幕翻译"),
    translate_gpu_layers: int = typer.Option(-1, "--translate-gpu-layers", help="卸载到 GPU 的层数，-1 = 全部", rich_help_panel="字幕翻译"),
    translate_batch: int = typer.Option(1, "--translate-batch", help="单次请求翻译几段字幕（1 = 逐段，质量最好；调大略快但质量下降）", rich_help_panel="字幕翻译"),
    translate_prompt: Optional[str] = typer.Option(None, "--translate-prompt", help="翻译提示词（含 {text} 占位符）", rich_help_panel="字幕翻译"),
    translate_prompt_file: Optional[Path] = typer.Option(None, "--translate-prompt-file", help="从文件读取翻译提示词（优先级高于 --translate-prompt）", rich_help_panel="字幕翻译"),
    translate_max_line_duration: float = typer.Option(6.0, "--translate-max-line-duration", help="译文每行最长时长(秒)", rich_help_panel="字幕翻译"),
    translate_max_line_chars: int = typer.Option(24, "--translate-max-line-chars", help="译文每行最长字数", rich_help_panel="字幕翻译"),
    translate_suffix: str = typer.Option(".zh", "--translate-suffix", help="译文文件后缀，最终生成 <原名><后缀>.srt", rich_help_panel="字幕翻译"),

    seek_start: float = typer.Option(0.0, "--seek-start", "-ss", help="音频开始位置 (秒)", rich_help_panel="音频切片"),
    duration: Optional[float] = typer.Option(None, "--duration", "-t", help="处理音频的时长 (秒)", rich_help_panel="音频切片"),
    
    # 组 5: 音频裁剪与性能
    chunk_size: float = typer.Option(40.0, "--chunk-size", "-c", help="分段识别时长 (秒)", rich_help_panel="流式配置"),
    memory_num: int = typer.Option(1, "--memory-num", help="记忆的历史片段数量", rich_help_panel="流式配置"),
    
    # 组 6: 其他
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="是否打印详细日志", rich_help_panel="其他选项"),
    yes: bool = typer.Option(False, "--yes", "-y", help="覆盖已存在的输出文件", rich_help_panel="其他选项"),
):
    """
    使用 Qwen3-ASR GGUF 模型对音频进行高精度转录，并可选用说话人分离，
    最终输出带 [bold]\\[spk0][/bold] 说话人标记的 SRT 字幕（每行控制在 6 秒内，按标点语义断句）。

    说话人分离支持两种后端（--diar-backend）：[bold]onnx[/bold]（onnxruntime，可用
    DirectML / CUDA / CPU，不需要 HF Token）与 [bold]pyannote[/bold]（原生 PyTorch，需要 HF Token）。

    加 [bold]--translate[/bold] 可在字幕基础上再翻译一份：不完整的句子会与后续行合并翻译，
    译文按标点拆回多行（每行 ≤ 6 秒），保留 [bold]\\[spk0][/bold] 标记，
    另存为 [bold]<原名>.zh.srt[/bold]，原字幕文件保持不变。

    也可以直接把 [bold].srt[/bold] 字幕文件当输入（配 [bold]--translate[/bold]）：
    此时[bold]跳过解码 / 说话人识别 / 语音识别[/bold]，只在现有字幕上补一份译文，
    原字幕不动，适合字幕已经有了、只想省时间补翻译的场景：

        [cyan]uv run python transcribe.py a.srt --translate[/cyan]
    """
    enable_diarization = not no_diarization

    # 输入分流：音视频走完整识别链路；字幕文件只做翻译
    media_files = [p for p in files if not is_subtitle_file(p)]
    subtitle_files = [p for p in files if is_subtitle_file(p)]
    need_asr = bool(media_files)

    if subtitle_files and not translate:
        console.print(
            f"[yellow]提示：输入里有 {len(subtitle_files)} 个字幕文件"
            f"（{_SUBTITLE_EXTS_HINT}），但未加 [cyan]--translate[/cyan]，"
            "字幕文件只会做翻译，因此将被跳过。[/yellow]"
        )
    if not need_asr and translate:
        console.print(
            "[cyan]检测到输入全部为字幕文件：跳过解码、说话人识别与语音识别，只做翻译。[/cyan]"
        )

    # 翻译提示词：文件优先于字符串
    prompt_text = DEFAULT_TRANSLATION_PROMPT
    if translate_prompt_file is not None:
        try:
            prompt_text = Path(translate_prompt_file).read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[bold red]无法读取提示词文件 {translate_prompt_file}: {e}[/bold red]")
            raise typer.Exit(code=1)
    elif translate_prompt:
        prompt_text = translate_prompt
    if translate and "{text}" not in prompt_text:
        console.print("[yellow]提示词里没有 {text} 占位符，原文会被直接追加在提示词之后。[/yellow]")

    # 1. 环境准备
    if not use_vulkan:
        os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan

    # 2. 构造配置
    config = PipelineConfig(
        model_dir=model_dir,
        precision=precision,
        onnx_provider=onnx_provider,
        llm_use_gpu=llm_use_gpu,
        use_vulkan=use_vulkan,
        n_ctx=n_ctx,
        chunk_size=chunk_size,
        memory_num=memory_num,
        language=language,
        context=context,
        temperature=temperature,
        enable_aligner=timestamp,
        max_line_duration=max_line_duration,
        max_line_chars=max_line_chars,
        enable_diarization=enable_diarization,
        diarization_backend=diarization_backend,
        hf_token=hf_token,
        diarization_device=diarization_device,
        max_speakers=max_speakers,
        export_txt=True,
        export_json=True,
        enable_translation=translate,
        translation_model=translate_model,
        translation_backend=translate_backend,
        translation_prompt=prompt_text,
        translation_batch_size=translate_batch,
        translation_n_gpu_layers=translate_gpu_layers,
        translation_max_line_duration=translate_max_line_duration,
        translation_max_line_chars=translate_max_line_chars,
        translation_output_suffix=translate_suffix,
        verbose=verbose,
    )

    if not timestamp:
        console.print("[yellow]注意：未开启时间戳引擎，将无法生成精确 SRT，仅能按字数估算时间轴。[/yellow]")

    # 3. 打印配置面板
    config_table = Table(show_header=False, box=None)
    config_table.add_row("模型目录", f"[green]{model_dir}[/green]")
    config_table.add_row("编码精度", f"[cyan]{precision}[/cyan]")
    config_table.add_row("加速设备", f"ONNX:{onnx_provider} | LLM-GPU:{'[green]ON[/green]' if llm_use_gpu else '[red]OFF[/red]'} | Vulkan:{'[green]ON[/green]' if use_vulkan else '[red]OFF[/red]'}")
    config_table.add_row("时间戳对齐", f"{'[green]启用[/green]' if timestamp else '[red]禁用[/red]'}")
    config_table.add_row("语言设定", f"{language or '自动识别'}")
    if subtitle_files:
        config_table.add_row(
            "输入分流",
            f"音视频 {len(media_files)} 个（识别）"
            + (f" | 字幕 {len(subtitle_files)} 个（[cyan]只翻译[/cyan]）" if subtitle_files else ""),
        )
    if not need_asr:
        config_table.add_row(
            "识别链路",
            "[dim]输入全是字幕文件，已跳过解码 / 说话人识别 / 语音识别[/dim]",
        )
    elif enable_diarization:
        resolved = resolve_diarization_backend(config)
        if resolved == "onnx":
            spk_desc = f"{speakers} 人" if speakers > 0 else f"自动检测 (上限 {max_speakers})"
            config_table.add_row(
                "说话人识别",
                f"[green]ONNX Runtime[/green] | {spk_desc} | 设备:{diarization_device}",
            )
        elif resolved == "pyannote":
            spk_desc = f"{speakers} 人" if speakers > 0 else f"自动检测 (上限 {max_speakers})"
            config_table.add_row(
                "说话人识别",
                f"[green]pyannote.audio[/green] | {spk_desc} | 设备:{diarization_device}",
            )
            if diarization_needs_hf_token(config):
                config_table.add_row(
                    "",
                    "[yellow]未提供 --hf-token，pyannote 模型为 gated 资源，下载可能失败[/yellow]",
                )
        else:
            config_table.add_row(
                "说话人识别",
                "[red]无可用后端，将跳过（请先导出 ONNX 模型或安装 pyannote.audio）[/red]",
            )
    else:
        config_table.add_row("说话人识别", "[red]关闭[/red]")
    config_table.add_row("字幕断句", f"每行 ≤ {max_line_duration:g} 秒 / {max_line_chars} 字")
    if translate:
        tr_backend, tr_note = resolve_translation_backend(translate_backend)
        config_table.add_row(
            "字幕翻译",
            f"[green]{translate_model}[/green] | 后端:{tr_backend} | 每批 {translate_batch} 段",
        )
        config_table.add_row("", f"[dim]{tr_note}[/dim]")
        config_table.add_row(
            "译文排版",
            f"每行 ≤ {translate_max_line_duration:g} 秒 / {translate_max_line_chars} 字，"
            f"输出 <原名>[cyan]{translate_suffix}[/cyan].srt（原字幕保留）",
        )
    else:
        config_table.add_row("字幕翻译", "[dim]关闭[/dim]")
    
    console.print(Panel(config_table, title="[bold cyan]Qwen3-ASR 配置选项[/bold cyan]", expand=False))

    # 4. 检查模型文件是否存在（纯字幕输入时不需要识别模型）
    check_model_files(config, need_asr=need_asr)

    # 5. 初始化引擎（只有需要识别音视频时才加载，纯字幕输入直接跳过）
    engine = None
    if need_asr:
        with console.status("[bold yellow]正在初始化引擎，请稍候...[/bold yellow]"):
            try:
                t0 = time.time()
                engine = build_engine(config, verbose=verbose)
                init_duration = time.time() - t0
                console.print(f"--- [QwenASR] 引擎初始化耗时: {init_duration:.2f} 秒 ---")
            except Exception as e:
                console.print(f"[bold red]引擎初始化失败:[/bold red]\n{e}")
                console.print(f"[bold yellow]建议解决方案：[/bold yellow]")
                console.print(f"  1. 尝试使用 CPU 后端: 使用 [cyan]--provider CPU --no-gpu[/cyan]")
                console.print(f"  2. 尝试关闭 Vulkan 加速: 使用 [cyan]--no-vulkan[/cyan]")
                console.print(f"  3. 如果问题仍然存在，请在 GitHub 提交 Issue 并附带 [cyan]{PROJ_DIR}\\logs\\latest.log[/cyan] 日志文件。")
                raise typer.Exit(code=1)
    else:
        console.print("--- [QwenASR] 输入全是字幕文件，跳过 ASR 引擎初始化 ---")

    diarizer = build_diarizer(config) if (enable_diarization and need_asr) else None

    # 翻译模型在正式跑之前预加载：一是让用户马上看到实际生效的后端，
    # 二是避免第一段字幕等到模型加载完才开始翻译。
    translator = None
    if translate:
        translator = build_translator(config)
        with console.status("[bold yellow]正在加载翻译模型，请稍候...[/bold yellow]"):
            loaded = translator.load()
        if loaded:
            console.print(
                f"--- [翻译] 模型 {translate_model} 已加载，"
                f"后端 {translator.backend}（{translator.backend_note}）---"
            )
        else:
            console.print(f"[bold red]翻译模型加载失败: {translator.last_error}[/bold red]")
            if need_asr:
                console.print("[yellow]将继续输出原文与说话人字幕，不做翻译。[/yellow]")
            else:
                console.print("[yellow]输入全是字幕文件且翻译不可用，本次没有可做的事。[/yellow]")
            translator = None
            config.enable_translation = False

    pipeline = TranscriptionPipeline(engine, diarizer, config, translator)

    # 事件回调：把流水线进度渲染到控制台
    streaming_state = {"speaker": ""}

    def on_event(kind: str, payload: dict):
        if kind == "text":
            piece = payload.get("text", "")
            if piece:
                rprint(piece, end="", soft_wrap=True)
        elif kind == "chunk":
            spks = payload.get("speakers") or []
            header = f"\n[dim]── 分片 {payload.get('index',0)+1}/{payload.get('total',1)} " \
                     f"[{payload.get('start',0):.1f}s-{payload.get('end',0):.1f}s][/dim]"
            if spks:
                header += " " + " ".join(f"[bold magenta][{s}][/bold magenta]" for s in spks)
            rprint(header)
            if spks:
                rprint(f"[bold magenta][{spks[0]}][/bold magenta] ", end="")
            streaming_state["speaker"] = spks[0] if spks else ""
        elif kind == "warning":
            rprint(f"\n[bold yellow]⚠ {payload.get('message','')}[/bold yellow]")
        elif kind == "stage" and payload.get("stage") == "diarize":
            rprint(f"\n[bold cyan]{payload.get('message','')}[/bold cyan]")
        elif kind == "stage" and payload.get("stage") == "translate":
            rprint(f"\n[bold cyan]🌐 {payload.get('message','')}[/bold cyan]")
        elif kind == "file" and payload.get("kind") == "srt":
            rprint(f"\n[bold green]✅ 已生成字幕文件: {payload.get('path','')}[/bold green]")
        elif kind == "file" and payload.get("kind") == "translated_srt":
            rprint(f"[bold green]✅ 已生成译文文件: {payload.get('path','')}[/bold green]")

    # 6. 循环处理文件
    try:
        for input_path in files:
            if not input_path.exists():
                console.print(f"[yellow]跳过不存在的文件: {input_path}[/yellow]")
                continue

            is_sub = is_subtitle_file(input_path)

            # 字幕文件唯一的处理方式是翻译，未开启翻译直接跳过
            if is_sub and translator is None:
                console.print(
                    f"[yellow]跳过字幕文件 {input_path.name}："
                    f"{'未开启 --translate' if not translate else '翻译不可用'}。[/yellow]"
                )
                continue

            kind_label = "字幕（只翻译）" if is_sub else "音视频（识别）"
            console.print(
                f"\n[bold blue]开始处理（{kind_label}）:[/bold blue] {input_path.name}\n"
            )

            # 检查输出文件冲突
            # 注意：字幕输入时**原文件就是输出之一**，绝不能把它算进可覆盖列表
            base_out = input_path.with_suffix("")
            targets = []
            if not is_sub:
                targets += [f"{base_out}.txt", f"{base_out}.srt"]
            tr_out = f"{base_out}{translate_suffix}.srt" if translate else None
            if tr_out:
                if is_sub and Path(tr_out) == input_path:
                    console.print(
                        f"[bold red]译文路径与输入字幕相同（{tr_out}），"
                        "请用 --translate-suffix 换一个后缀。已跳过。[/bold red]"
                    )
                    continue
                targets.append(tr_out)
            existing = [p for p in targets if Path(p).exists()]
            if existing and not yes:
                if not typer.confirm(f"文件 {', '.join(existing)} 已存在，是否覆盖?"):
                    console.print("[yellow]已跳过。[/yellow]")
                    continue

            if is_sub:
                res = pipeline.translate_subtitle_file(
                    str(input_path), on_event=on_event
                )
            else:
                res = pipeline.run(
                    str(input_path),
                    num_speakers=speakers if speakers > 0 else None,
                    on_event=on_event,
                    with_speaker=enable_diarization,
                    start_second=seek_start,
                    duration=duration,
                )

            console.print()
            if res.speakers:
                console.print(f"[bold cyan]🎤 检测到 {res.speakers.num_speakers} 位说话人: "
                              f"{', '.join(res.speakers.labels)}[/bold cyan]")
            if is_sub:
                labels = []
                for seg in res.segments:
                    if seg.speaker and seg.speaker not in labels:
                        labels.append(seg.speaker)
                if labels:
                    console.print(f"[bold cyan]🎤 字幕中的说话人: {', '.join(labels)}[/bold cyan]")
                console.print(f"[bold green]✅ 输入字幕条目: {len(res.segments)} 条，"
                              f"总耗时 {res.elapsed:.1f} 秒[/bold green]")
            else:
                console.print(f"[bold green]✅ 字幕条目: {len(res.segments)} 条，"
                              f"总耗时 {res.elapsed:.1f} 秒[/bold green]")
            if res.translated_srt_path:
                console.print(f"[bold green]🌐 译文文件: {res.translated_srt_path} "
                              f"（{len(res.translated_segments)} 条）[/bold green]")
            elif is_sub:
                console.print("[bold yellow]⚠ 未生成译文文件。[/bold yellow]")

    finally:
        if engine is not None:
            engine.shutdown()
        if translator is not None:
            translator.release()
        console.print("\n[bold green]所有任务已完成。[/bold green]")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app()
