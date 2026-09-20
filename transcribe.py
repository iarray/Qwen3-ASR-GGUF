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
    PipelineConfig,
    TranscriptionPipeline,
    build_diarizer,
    build_engine,
    diarization_needs_hf_token,
    find_missing_models,
    resolve_diarization_backend,
)

app = typer.Typer(help="Qwen3-ASR GGUF 命令行转录工具 (支持说话人识别与 SRT 字幕)", add_completion=False)
console = Console()


def check_model_files(config: PipelineConfig):
    """检查模型文件完整性"""
    missing_files = find_missing_models(config.model_dir, config.precision, config.enable_aligner)

    if missing_files:
        console.print("\n[bold red]错误：找不到以下所需模型文件：[/bold red]")
        for f in missing_files:
            console.print(f"  - {f}")
        console.print("\n[bold yellow]请到以下链接下载模型文件，并解压到 model 目录：[/bold yellow]")
        console.print("[blue]https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/tag/models[/blue]\n")
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    files: List[Path] = typer.Argument(..., help="要转录的音频文件列表"),
    
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
    """
    enable_diarization = not no_diarization

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
    if enable_diarization:
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
    
    console.print(Panel(config_table, title="[bold cyan]Qwen3-ASR 配置选项[/bold cyan]", expand=False))

    # 4. 检查模型文件是否存在
    check_model_files(config)

    # 5. 初始化引擎
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

    diarizer = build_diarizer(config) if enable_diarization else None
    pipeline = TranscriptionPipeline(engine, diarizer, config)

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
        elif kind == "file" and payload.get("kind") == "srt":
            rprint(f"\n[bold green]✅ 已生成字幕文件: {payload.get('path','')}[/bold green]")

    # 6. 循环处理文件
    try:
        for audio_path in files:
            if not audio_path.exists():
                console.print(f"[yellow]跳过不存在的文件: {audio_path}[/yellow]")
                continue

            console.print(f"\n[bold blue]开始处理:[/bold blue] {audio_path.name}\n")

            # 检查输出文件冲突
            base_out = audio_path.with_suffix("")
            txt_out = f"{base_out}.txt"
            srt_out = f"{base_out}.srt"
            if (Path(txt_out).exists() or Path(srt_out).exists()) and not yes:
                if not typer.confirm(f"文件 {txt_out} / {srt_out} 已存在，是否覆盖?"):
                    console.print("[yellow]已跳过。[/yellow]")
                    continue

            res = pipeline.run(
                str(audio_path),
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
            console.print(f"[bold green]✅ 字幕条目: {len(res.segments)} 条，"
                          f"总耗时 {res.elapsed:.1f} 秒[/bold green]")

    finally:
        engine.shutdown()
        console.print("\n[bold green]所有任务已完成。[/bold green]")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app()
