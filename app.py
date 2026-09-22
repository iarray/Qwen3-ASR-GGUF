# coding=utf-8
"""
app.py - 语音识别 + 说话人识别 + 字幕翻译 桌面端 (ttkbootstrap)

功能：
1. 可配置说话人数（留空 / 0 表示自动检测）
2. 批量添加音频或视频文件，按列表顺序逐个识别并生成字幕
3. 界面实时显示识别内容与说话人，并预览最终 SRT
4. 字幕输出到原音频文件目录，文件名与音频文件一致
5. 可选字幕翻译：在识别出的字幕基础上翻译（Qwen3-8B GGUF，可走 Vulkan / CUDA / CPU），
   不完整的句子与后续行合并后翻译，译文按标点拆回多行，保留说话人标记，
   另存为 <原名>.zh.srt，原字幕文件保持不变
6. 支持直接把 .srt 字幕文件加进列表——**只翻译，不重跑识别**：
   跳过多媒体解码、说话人识别与语音识别，省掉整条识别链路的时间

运行：
    uv run python app.py
"""
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, BOTTOM, END, LEFT, RIGHT, TOP, X, Y

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
if getattr(sys, "frozen", False):
    PROJ_DIR = Path(sys.executable).parent
else:
    PROJ_DIR = Path(__file__).parent

SETTINGS_FILE = PROJ_DIR / "ui_settings.json"

AUDIO_EXTS = (
    ".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".ts", ".m4v",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".3gp",
)

# 字幕文件：直接丢进来只做翻译，跳过解码 / 说话人识别 / 语音识别
# （与 qwen_asr_gguf.pipeline.SUBTITLE_FILE_EXTS 保持一致；这里不 import 是为了
#  避免一启动 GUI 就去加载推理动态库）
SUBTITLE_EXTS = (".srt",)


def _is_subtitle_file(path) -> bool:
    """输入是否是字幕文件（只会走翻译，不做识别）"""
    return Path(str(path)).suffix.lower() in SUBTITLE_EXTS


def _is_media_file(path) -> bool:
    return not _is_subtitle_file(path)


def _speaker_labels(segments) -> list:
    """从字幕片段里收集出现过的说话人标签（保持首次出现顺序）"""
    seen: list = []
    for seg in segments or []:
        spk = getattr(seg, "speaker", "") or ""
        if spk and spk not in seen:
            seen.append(spk)
    return seen

SPEAKER_COLORS = [
    "#0b6ed0", "#d62728", "#1a9850", "#e07b00", "#7b3fbf",
    "#0f9aa8", "#b8459b", "#6b6b6b", "#8a6d00", "#3d5a80",
]

LIGHT_THEME = "bootstrap-light"
DARK_THEME = "bootstrap-dark"


def _default_translation_prompt() -> str:
    """取翻译默认提示词（放在函数里 import，避免启动时就来一轮 DLL 初始化）"""
    try:
        from qwen_asr_gguf.inference.translator import DEFAULT_TRANSLATION_PROMPT

        return DEFAULT_TRANSLATION_PROMPT
    except Exception:
        return (
            "你是专业字幕翻译，只输出译文，不要额外解释。\n"
            "翻译规则：\n"
            "1. 英译中，符合中文口语习惯，不要直译英文句式，去掉欧化长句；\n"
            "2. 保留原句语气，自然流畅，像原生中文台词；\n"
            "3. 短句拆分，适合字幕阅读，不要堆砌长句；\n"
            "4. 专有名词保持统一。\n"
            "原文：\n"
            "{text}"
        )


def default_settings() -> dict:
    return {
        "model_dir": str(PROJ_DIR / "model"),
        "precision": "int4",
        "onnx_provider": "DML",
        "llm_use_gpu": True,
        "use_vulkan": True,
        "n_ctx": 2048,
        "chunk_size": 40.0,
        "memory_num": 1,
        "language": "",
        "context": "",
        "temperature": 0.4,
        "max_line_duration": 6.0,
        "max_line_chars": 48,
        "enable_diarization": True,
        "diarization_backend": "auto",
        "diarization_model_dir": "",
        "diarization_device": "auto",
        "min_speakers": 1,
        "max_speakers": 20,
        "hf_token": "",
        "num_speakers": 0,
        "export_txt": True,
        "export_json": True,
        # ---- 字幕翻译 ----
        "enable_translation": False,
        "translate_model": "Qwen3-8B-Q4_K_M.gguf",
        "translate_backend": "auto",
        "translate_gpu_layers": -1,
        "translate_batch": 1,
        "translate_max_line_duration": 6.0,
        "translate_max_line_chars": 24,
        "translate_suffix": ".zh",
        "translate_prompt": _default_translation_prompt(),
        "dark_mode": False,
        "last_dir": "",
    }


def load_settings() -> dict:
    data = default_settings()
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                data.update(saved)
    except Exception:
        pass
    return data


def save_settings(data: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# 主界面
# --------------------------------------------------------------------------- #
class ASRApp(ttk.Window):

    def __init__(self):
        self.settings = load_settings()
        super().__init__(
            title="Qwen3-ASR GGUF · 语音识别 + 说话人识别",
            themename=DARK_THEME if self.settings.get("dark_mode") else LIGHT_THEME,
        )
        # 自适应屏幕：小屏上也不会把底部状态条顶出可见区域
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = max(960, min(1240, sw - 80))
        h = max(620, min(880, sh - 110))
        x, y = max(0, (sw - w) // 2), max(0, (sh - h) // 2 - 20)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(min(1000, w), min(640, h))

        # 运行时状态
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event: threading.Event | None = None
        self.file_rows: list[dict] = []       # {iid, path, status}
        self.is_running = False
        self._cur_speaker_line = ""
        self._text_buffer: list[str] = []
        self._speaker_tags: dict[str, str] = {}
        self._stat_started_at = 0.0

        self._build_ui()
        self._apply_settings_to_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # 界面搭建
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        # ---------------- 顶部：参数配置 ---------------- #
        top = ttk.Frame(self, padding=(10, 8, 10, 0))
        top.pack(fill=X)

        self._build_model_frame(top).grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_speaker_frame(top).grid(row=0, column=1, sticky="nsew")
        top.columnconfigure(0, weight=1, uniform="cfg")
        top.columnconfigure(1, weight=1, uniform="cfg")

        # ---------------- 中部：文件列表 ---------------- #
        mid = ttk.Labelframe(
            self, text=" 待处理文件（音视频=识别+翻译；.srt 字幕=只翻译，跳过识别） ", padding=8
        )
        mid.pack(fill=X, **pad)

        bar = ttk.Frame(mid)
        bar.pack(fill=X, pady=(0, 6))

        ttk.Button(bar, text="＋ 添加文件", bootstyle="primary", command=self._add_files, width=12).pack(side=LEFT)
        ttk.Button(bar, text="＋ 添加字幕", bootstyle="info", command=self._add_subtitle_files, width=12).pack(side=LEFT, padx=6)
        ttk.Button(bar, text="－ 移除选中", bootstyle="secondary-outline", command=self._remove_selected, width=12).pack(side=LEFT)
        ttk.Button(bar, text="清空列表", bootstyle="secondary-outline", command=self._clear_files, width=10).pack(side=LEFT, padx=6)
        ttk.Button(bar, text="上移", bootstyle="secondary-outline", command=lambda: self._move(-1), width=6).pack(side=LEFT, padx=(2, 2))
        ttk.Button(bar, text="下移", bootstyle="secondary-outline", command=lambda: self._move(1), width=6).pack(side=LEFT)
        ttk.Button(bar, text="上移", bootstyle="secondary-outline", command=lambda: self._move(-1), width=6).pack(side=LEFT, padx=(6, 2))
        ttk.Button(bar, text="下移", bootstyle="secondary-outline", command=lambda: self._move(1), width=6).pack(side=LEFT)

        # 主要操作按钮放在文件工具条上：这块区域紧邻「添加文件」，永远可见，
        # 避免窗口变矮时被下方输出区把按钮挤没。
        self.lbl_count = ttk.Label(bar, text="共 0 个文件", bootstyle="secondary")
        self.lbl_count.pack(side=RIGHT)

        self.btn_stop = ttk.Button(
            bar, text="■  停止", bootstyle="danger-outline", width=10,
            command=self._stop, state="disabled"
        )
        self.btn_stop.pack(side=RIGHT, padx=(6, 12))

        self.btn_start = ttk.Button(
            bar, text="▶  开始识别", bootstyle="success", width=14, command=self._start
        )
        self.btn_start.pack(side=RIGHT)

        table_wrap = ttk.Frame(mid)
        table_wrap.pack(fill=X)

        columns = ("order", "name", "spk", "status")
        self.tree = ttk.Treeview(
            table_wrap, columns=columns, show="headings", height=6, bootstyle="primary"
        )
        self.tree.heading("order", text="#")
        self.tree.heading("name", text="文件")
        self.tree.heading("spk", text="说话人")
        self.tree.heading("status", text="状态")
        self.tree.column("order", width=44, anchor="center", stretch=False)
        self.tree.column("name", width=640, anchor="w")
        self.tree.column("spk", width=90, anchor="center", stretch=False)
        self.tree.column("status", width=220, anchor="w", stretch=False)

        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=LEFT, fill=X, expand=True)
        vsb.pack(side=RIGHT, fill=Y)

        self.tree.tag_configure("done", foreground="#1a9850")
        self.tree.tag_configure("running", foreground="#0b6ed0")
        self.tree.tag_configure("failed", foreground="#d62728")

        # ---------------- 下部：输出区 ---------------- #
        # 这里只创建、不 pack，留到 _build_ui 末尾再填充剩余空间
        out = ttk.Frame(self, padding=(10, 0, 10, 0))

        self.notebook = ttk.Notebook(out, bootstyle="primary")
        self.notebook.pack(fill=BOTH, expand=True)

        # Tab1 实时转录
        tab1 = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab1, text=" 实时转录 ")
        self.txt_live = self._make_text(tab1)

        # Tab2 字幕预览
        tab2 = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab2, text=" 字幕预览 (SRT) ")
        self.txt_srt = self._make_text(tab2)

        # Tab3 译文预览
        tab3 = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(tab3, text=" 译文预览 ")
        self.txt_trans = self._make_text(tab3)

        # Tab4 翻译设置
        tab4 = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab4, text=" 翻译设置 ")
        # 注意：这个构建函数内部已经完成了 Canvas 的 pack，调用方不要再 pack 一次
        self._build_translate_frame(tab4)

        # ---------------- 底部：状态条 ---------------- #
        # 先 pack 底部状态条（side=BOTTOM），再让输出区吃剩余空间，
        # 这样窗口变矮时只会压缩输出区，不会把按钮挤到窗口外。
        bottom = ttk.Frame(self, padding=(10, 6, 10, 10))
        bottom.pack(side=BOTTOM, fill=X)

        ttk.Button(
            bottom, text="打开输出目录", bootstyle="secondary-outline", width=12,
            command=self._open_output_dir
        ).pack(side=LEFT)

        self.btn_theme = ttk.Button(
            bottom, text="🌙 深色", bootstyle="secondary-outline", width=10,
            command=self._toggle_theme
        )
        self.btn_theme.pack(side=RIGHT)

        self.progress = ttk.Progressbar(bottom, bootstyle="success-striped", length=320)
        self.progress.pack(side=RIGHT, padx=12)
        self.progress.configure(maximum=1.0, value=0.0)

        self.lbl_status = ttk.Label(bottom, text="就绪", bootstyle="secondary")
        self.lbl_status.pack(side=RIGHT, padx=(0, 8))

        # 输出区最后 pack，独占所有剩余空间
        out.pack(side=TOP, fill=BOTH, expand=True)

    def _make_text(self, parent) -> tk.Text:
        wrap = ttk.Frame(parent)
        wrap.pack(fill=BOTH, expand=True)
        text = tk.Text(
            wrap, wrap="word", undo=False, relief="flat", borderwidth=0,
            highlightthickness=0, padx=10, pady=8, height=10,
            font=("Microsoft YaHei UI", 11),
        )
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=vsb.set)
        text.pack(side=LEFT, fill=BOTH, expand=True)
        vsb.pack(side=RIGHT, fill=Y)

        text.tag_configure("spk", font=("Microsoft YaHei UI", 11, "bold"))
        text.tag_configure("meta", foreground="#7a7a7a")
        text.tag_configure("err", foreground="#d62728")
        text.tag_configure("ok", foreground="#1a9850")
        text.tag_configure("head", foreground="#7a3fbf", font=("Microsoft YaHei UI", 11, "bold"))
        return text

    def _build_model_frame(self, parent) -> ttk.Labelframe:
        f = ttk.Labelframe(parent, text=" 模型与硬件 ", padding=10)
        f.columnconfigure(1, weight=1)

        def row(r, label, widget, hint=None, col=1):
            ttk.Label(f, text=label, width=10).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=col, sticky="ew", pady=3, padx=(0, 6))
            return widget

        # 模型目录
        self.var_model_dir = tk.StringVar()
        dirbox = ttk.Frame(f)
        ent = ttk.Entry(dirbox, textvariable=self.var_model_dir)
        ent.pack(side=LEFT, fill=X, expand=True)
        ttk.Button(dirbox, text="…", width=3, bootstyle="secondary-outline",
                   command=self._choose_model_dir).pack(side=LEFT, padx=(4, 0))
        row(0, "模型目录", dirbox)

        self.var_precision = tk.StringVar()
        row(1, "编码精度", ttk.Combobox(f, textvariable=self.var_precision,
                                    values=["int4", "int8", "fp16", "fp32"], state="readonly"))

        self.var_provider = tk.StringVar()
        row(2, "ONNX 后端", ttk.Combobox(f, textvariable=self.var_provider,
                                     values=["DML", "CPU", "CUDA", "TRT"], state="readonly"))

        self.var_n_ctx = tk.IntVar()
        row(3, "上下文", ttk.Spinbox(f, textvariable=self.var_n_ctx, from_=512, to=16384,
                                  increment=512))

        self.var_chunk = tk.DoubleVar()
        row(4, "分段时长(s)", ttk.Spinbox(f, textvariable=self.var_chunk, from_=5, to=120,
                                     increment=5))

        self.var_memory = tk.IntVar()
        row(5, "记忆片段", ttk.Spinbox(f, textvariable=self.var_memory, from_=0, to=5,
                                   increment=1))

        self.var_language = tk.StringVar()
        row(6, "语言", ttk.Combobox(f, textvariable=self.var_language,
                                  values=["", "Chinese", "English", "Japanese", "Korean", "Cantonese"],
                                  state="readonly"))

        toggles = ttk.Frame(f)
        self.var_gpu = tk.BooleanVar()
        self.var_vulkan = tk.BooleanVar()
        ttk.Checkbutton(toggles, text="LLM 使用 GPU", variable=self.var_gpu,
                        bootstyle="success-round-toggle").pack(side=LEFT, padx=(0, 16))
        ttk.Checkbutton(toggles, text="Vulkan 加速", variable=self.var_vulkan,
                        bootstyle="success-round-toggle").pack(side=LEFT)
        toggles.grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
        return f

    def _build_speaker_frame(self, parent) -> ttk.Labelframe:
        f = ttk.Labelframe(parent, text=" 说话人识别与字幕 ", padding=10)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        def row(r, c, label, widget):
            ttk.Label(f, text=label, width=11).grid(row=r, column=c * 2, sticky="w", pady=3)
            widget.grid(row=r, column=c * 2 + 1, sticky="ew", pady=3, padx=(0, 10))

        self.var_enable_diar = tk.BooleanVar()
        ttk.Checkbutton(f, text="启用说话人识别", variable=self.var_enable_diar,
                        bootstyle="success-round-toggle").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        self.var_num_spk = tk.IntVar()
        row(1, 0, "说话人数", ttk.Spinbox(f, textvariable=self.var_num_spk, from_=0, to=16,
                                     increment=1))
        ttk.Label(f, text="0 = 自动检测", bootstyle="secondary").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self.var_max_spk = tk.IntVar()
        row(1, 1, "最多说话人", ttk.Spinbox(f, textvariable=self.var_max_spk, from_=1, to=32,
                                       increment=1))

        # 推理后端：onnx 走 onnxruntime（可用 DirectML / CUDA / CPU），pyannote 是原生 PyTorch
        self.var_diar_backend = tk.StringVar()
        cb_backend = ttk.Combobox(
            f, textvariable=self.var_diar_backend,
            values=["auto", "onnx", "pyannote"], state="readonly",
        )
        cb_backend.grid(row=3, column=1, sticky="ew", pady=3, padx=(0, 10))
        cb_backend.bind("<<ComboboxSelected>>", lambda _e: self._refresh_backend_hint())
        ttk.Label(f, text="识别后端", width=11).grid(row=3, column=0, sticky="w", pady=3)

        self.var_diar_device = tk.StringVar()
        cb_device = ttk.Combobox(f, textvariable=self.var_diar_device,
                                 values=["auto", "cpu", "cuda", "dml"], state="readonly")
        cb_device.grid(row=3, column=3, sticky="ew", pady=3, padx=(0, 10))
        cb_device.bind("<<ComboboxSelected>>", lambda _e: self._refresh_backend_hint())
        ttk.Label(f, text="识别设备", width=11).grid(row=3, column=2, sticky="w", pady=3)

        self.var_max_dur = tk.DoubleVar()
        row(4, 0, "每行最长(s)", ttk.Spinbox(f, textvariable=self.var_max_dur, from_=1, to=30,
                                        increment=0.5))
        self.var_max_chars = tk.IntVar()
        row(4, 1, "每行最长字数", ttk.Spinbox(f, textvariable=self.var_max_chars, from_=8, to=200,
                                        increment=4))

        self.var_hf_token = tk.StringVar()
        row(5, 0, "HF Token", ttk.Entry(f, textvariable=self.var_hf_token, show="●"))
        self.var_export_txt = tk.BooleanVar()
        ttk.Checkbutton(f, text="同时导出 txt/json", variable=self.var_export_txt,
                        bootstyle="success-round-toggle").grid(
            row=5, column=2, columnspan=2, sticky="w", pady=3)

        self.lbl_diar_hint = ttk.Label(
            f, text="", bootstyle="secondary", justify="left", wraplength=520,
        )
        self.lbl_diar_hint.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))
        return f

    def _refresh_backend_hint(self):
        """根据本机实际可用的后端刷新提示文案"""
        try:
            from qwen_asr_gguf.inference.diarization import (
                onnx_backend_available,
                onnxruntime_available,
                pyannote_available,
            )
        except Exception as e:  # pragma: no cover
            self.lbl_diar_hint.configure(text=f"无法加载说话人模块：{e}")
            return

        model_dir = self.var_model_dir.get().strip() or str(PROJ_DIR / "model")
        want = self.var_diar_backend.get() or "auto"

        has_ort = onnxruntime_available()
        has_onnx = has_ort and onnx_backend_available(model_dir)
        has_pya = pyannote_available()

        if want == "onnx":
            actual = "onnx" if has_onnx else "不可用"
        elif want == "pyannote":
            actual = "pyannote" if has_pya else "不可用"
        else:
            actual = "onnx" if has_onnx else ("pyannote" if has_pya else "不可用")

        tips = []
        if actual == "onnx":
            # 探测当前会用到哪个执行提供器
            prov = "CPU"
            try:
                import onnxruntime as ort
                avail = ort.get_available_providers()
                if "DmlExecutionProvider" in avail:
                    prov = "DirectML"
                elif "CUDAExecutionProvider" in avail:
                    prov = "CUDA"
            except Exception:
                pass
            tips.append(f"当前使用 ONNX Runtime（{prov}），无需 HuggingFace Token。")
            if self.var_diar_device.get() == "dml" and prov != "DirectML":
                tips.append("已选 dml，但未检测到 DirectML，将回退其它执行提供器。")
        elif actual == "pyannote":
            tips.append("当前使用 pyannote.audio（PyTorch），模型为 gated 资源，需填写 HF Token。")
            if self.var_diar_device.get() == "dml":
                tips.append("pyannote 在 AMD 显卡上通常只能回退 CPU，如需 GPU 推理请把后端改为 onnx。")
        else:
            if not has_ort:
                tips.append("未安装 onnxruntime；")
            elif not has_onnx:
                tips.append(f"未在 {model_dir} 找到 diarization_segmentation.onnx / diarization_embedding.onnx；")
            if not has_pya:
                tips.append("也未安装 pyannote.audio。")
            tips.append("请先执行  uv run python 30-Export-Diarization-ONNX.py  导出模型。")

        self.lbl_diar_hint.configure(text="".join(tips))

    # ------------------------------------------------------------------ #
    # 翻译设置页
    # ------------------------------------------------------------------ #
    def _make_scroll_area(self, parent) -> ttk.Frame:
        """做一个纵向可滚动的内容区，返回用于放内容的内部 Frame

        翻译设置页内容较高，小窗口 / 低分辨率下会被 Notebook 挤掉，
        套一层 Canvas 之后内容始终可达。
        """
        canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)

        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        vsb.pack(side=RIGHT, fill=Y)
        self._trans_canvas = canvas
        return inner

    def _build_translate_frame(self, parent):
        """构建「翻译设置」页；内部已完成 Canvas 的 pack，调用方不要再 pack"""
        wrap = self._make_scroll_area(parent)

        head = ttk.Frame(wrap)
        head.pack(fill=X, pady=(0, 8))
        self.var_enable_translate = tk.BooleanVar()
        ttk.Checkbutton(
            head, text="启用字幕翻译", variable=self.var_enable_translate,
            bootstyle="success-round-toggle", command=self._refresh_translate_hint,
        ).pack(side=LEFT)
        ttk.Label(
            head,
            text="既能在识别出的字幕上翻译，也能直接翻译现成的 .srt（跳过识别）；原字幕保留，译文另存一份",
            bootstyle="secondary",
        ).pack(side=LEFT, padx=(12, 0))

        grid = ttk.Labelframe(wrap, text=" 参数 ", padding=10)
        grid.pack(fill=X, pady=(0, 8))
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        def row(r, c, label, widget, width=12):
            ttk.Label(grid, text=label, width=width).grid(row=r, column=c * 2, sticky="w", pady=3)
            widget.grid(row=r, column=c * 2 + 1, sticky="ew", pady=3, padx=(0, 12))

        self.var_trans_model = tk.StringVar()
        row(0, 0, "翻译模型", ttk.Entry(grid, textvariable=self.var_trans_model))

        self.var_trans_backend = tk.StringVar()
        cb = ttk.Combobox(
            grid, textvariable=self.var_trans_backend,
            values=["auto", "vulkan", "cuda", "cpu"], state="readonly",
        )
        cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_translate_hint())
        row(0, 1, "加速后端", cb)

        self.var_trans_gpu_layers = tk.IntVar()
        row(1, 0, "GPU 层数", ttk.Spinbox(
            grid, textvariable=self.var_trans_gpu_layers, from_=-1, to=128, increment=1))

        self.var_trans_batch = tk.IntVar()
        row(1, 1, "每次翻译段数", ttk.Spinbox(
            grid, textvariable=self.var_trans_batch, from_=1, to=32, increment=1))

        self.var_trans_max_dur = tk.DoubleVar()
        row(2, 0, "译文每行(s)", ttk.Spinbox(
            grid, textvariable=self.var_trans_max_dur, from_=1, to=30, increment=0.5))

        self.var_trans_max_chars = tk.IntVar()
        row(2, 1, "译文每行字数", ttk.Spinbox(
            grid, textvariable=self.var_trans_max_chars, from_=6, to=60, increment=2))

        self.var_trans_suffix = tk.StringVar()
        row(3, 0, "译文后缀", ttk.Entry(grid, textvariable=self.var_trans_suffix))
        ttk.Label(
            grid, text="生成 <文件名><后缀>.srt，原字幕保持原文件名不变", bootstyle="secondary",
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=3)

        self.lbl_trans_hint = ttk.Label(wrap, text="", bootstyle="secondary", justify="left", wraplength=820)
        self.lbl_trans_hint.pack(fill=X, pady=(0, 8))

        pf = ttk.Labelframe(wrap, text=" 提示词（{text} 处会替换为原文） ", padding=8)
        pf.pack(fill=X, pady=(0, 8))

        bar = ttk.Frame(pf)
        bar.pack(fill=X, pady=(0, 6))
        ttk.Button(bar, text="恢复默认", bootstyle="secondary-outline", width=10,
                   command=self._reset_translate_prompt).pack(side=LEFT)
        ttk.Button(bar, text="从文件导入", bootstyle="secondary-outline", width=12,
                   command=self._load_translate_prompt).pack(side=LEFT, padx=6)
        ttk.Label(bar, text="提示词里必须包含 {text} 占位符", bootstyle="secondary").pack(side=LEFT, padx=6)

        box = ttk.Frame(pf)
        box.pack(fill=BOTH, expand=True)
        self.txt_prompt = tk.Text(
            box, wrap="word", height=8, relief="flat", borderwidth=0,
            highlightthickness=0, padx=8, pady=6, font=("Microsoft YaHei UI", 10),
        )
        psb = ttk.Scrollbar(box, orient="vertical", command=self.txt_prompt.yview)
        self.txt_prompt.configure(yscrollcommand=psb.set)
        self.txt_prompt.pack(side=LEFT, fill=BOTH, expand=True)
        psb.pack(side=RIGHT, fill=Y)
        return wrap

    def _refresh_translate_hint(self):
        """刷新翻译可用性提示（模型是否存在、后端会落到哪）"""
        try:
            from qwen_asr_gguf.inference.translator import (
                TranslationConfig,
                resolve_translation_backend,
                translation_model_available,
            )
        except Exception as e:  # pragma: no cover
            self.lbl_trans_hint.configure(text=f"无法加载翻译模块：{e}")
            return

        model_dir = self.var_model_dir.get().strip() or str(PROJ_DIR / "model")
        model_fn = self.var_trans_model.get().strip() or "Qwen3-8B-Q4_K_M.gguf"
        cfg = TranslationConfig(model_dir=model_dir, model_fn=model_fn)
        backend, note = resolve_translation_backend(self.var_trans_backend.get() or "auto")

        tips = []
        if translation_model_available(cfg):
            tips.append(f"模型已就绪（{model_fn}）；")
        else:
            tips.append(f"未找到模型 {os.path.join(model_dir, model_fn)}；")
        tips.append(note + "。")
        if self.var_enable_translate.get():
            tips.append(
                "翻译在识别完成后进行，不完整句子会与后续行合并后再翻译；"
                "也可以直接用「＋ 添加字幕」把 .srt 丢进来——只翻译，跳过解码与识别。"
            )
        else:
            tips.append("开启后，既能在识别出的字幕上翻译，也能直接翻译现成的 .srt 字幕文件。")
        self.lbl_trans_hint.configure(text="".join(tips))

    def _reset_translate_prompt(self):
        self.txt_prompt.delete("1.0", END)
        self.txt_prompt.insert("1.0", _default_translation_prompt())

    def _load_translate_prompt(self):
        path = filedialog.askopenfilename(
            title="选择提示词文件", filetypes=[("文本文件", "*.txt *.md"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return
        self.txt_prompt.delete("1.0", END)
        self.txt_prompt.insert("1.0", content)

    # ------------------------------------------------------------------ #
    # 设置读写
    # ------------------------------------------------------------------ #
    def _apply_settings_to_ui(self):
        s = self.settings
        self.var_model_dir.set(s.get("model_dir", str(PROJ_DIR / "model")))
        self.var_precision.set(s.get("precision", "int4"))
        self.var_provider.set(s.get("onnx_provider", "DML"))
        self.var_n_ctx.set(int(s.get("n_ctx", 2048)))
        self.var_chunk.set(float(s.get("chunk_size", 40.0)))
        self.var_memory.set(int(s.get("memory_num", 1)))
        self.var_language.set(s.get("language", ""))
        self.var_gpu.set(bool(s.get("llm_use_gpu", True)))
        self.var_vulkan.set(bool(s.get("use_vulkan", True)))

        self.var_enable_diar.set(bool(s.get("enable_diarization", True)))
        self.var_num_spk.set(int(s.get("num_speakers", 0)))
        self.var_max_spk.set(int(s.get("max_speakers", 20)))
        self.var_max_dur.set(float(s.get("max_line_duration", 6.0)))
        self.var_max_chars.set(int(s.get("max_line_chars", 48)))
        self.var_diar_backend.set(s.get("diarization_backend", "auto"))
        self.var_diar_device.set(s.get("diarization_device", "auto"))
        self.var_hf_token.set(s.get("hf_token", ""))
        self.var_export_txt.set(bool(s.get("export_txt", True)))

        # ---- 字幕翻译 ----
        self.var_enable_translate.set(bool(s.get("enable_translation", False)))
        self.var_trans_model.set(s.get("translate_model", "Qwen3-8B-Q4_K_M.gguf"))
        self.var_trans_backend.set(s.get("translate_backend", "auto"))
        self.var_trans_gpu_layers.set(int(s.get("translate_gpu_layers", -1)))
        self.var_trans_batch.set(int(s.get("translate_batch", 1)))
        self.var_trans_max_dur.set(float(s.get("translate_max_line_duration", 6.0)))
        self.var_trans_max_chars.set(int(s.get("translate_max_line_chars", 24)))
        self.var_trans_suffix.set(s.get("translate_suffix", ".zh"))
        self.txt_prompt.delete("1.0", END)
        self.txt_prompt.insert("1.0", s.get("translate_prompt") or _default_translation_prompt())

        try:
            self._refresh_backend_hint()
        except Exception:
            pass
        try:
            self._refresh_translate_hint()
        except Exception:
            pass

        if s.get("dark_mode"):
            self.btn_theme.configure(text="☀ 浅色")

    def _collect_settings(self) -> dict:
        s = dict(self.settings)
        s.update({
            "model_dir": self.var_model_dir.get().strip(),
            "precision": self.var_precision.get(),
            "onnx_provider": self.var_provider.get(),
            "n_ctx": int(self.var_n_ctx.get() or 2048),
            "chunk_size": float(self.var_chunk.get() or 40.0),
            "memory_num": int(self.var_memory.get() or 0),
            "language": self.var_language.get().strip(),
            "llm_use_gpu": bool(self.var_gpu.get()),
            "use_vulkan": bool(self.var_vulkan.get()),
            "enable_diarization": bool(self.var_enable_diar.get()),
            "num_speakers": int(self.var_num_spk.get() or 0),
            "max_speakers": int(self.var_max_spk.get() or 20),
            "max_line_duration": float(self.var_max_dur.get() or 6.0),
            "max_line_chars": int(self.var_max_chars.get() or 48),
            "diarization_backend": self.var_diar_backend.get() or "auto",
            "diarization_device": self.var_diar_device.get(),
            "hf_token": self.var_hf_token.get().strip(),
            "export_txt": bool(self.var_export_txt.get()),
            "enable_translation": bool(self.var_enable_translate.get()),
            "translate_model": self.var_trans_model.get().strip() or "Qwen3-8B-Q4_K_M.gguf",
            "translate_backend": self.var_trans_backend.get() or "auto",
            "translate_gpu_layers": int(self.var_trans_gpu_layers.get()),
            "translate_batch": max(1, int(self.var_trans_batch.get() or 1)),
            "translate_max_line_duration": float(self.var_trans_max_dur.get() or 6.0),
            "translate_max_line_chars": int(self.var_trans_max_chars.get() or 24),
            "translate_suffix": self.var_trans_suffix.get().strip() or ".zh",
            "translate_prompt": self.txt_prompt.get("1.0", "end-1c").strip()
            or _default_translation_prompt(),
            "dark_mode": self._current_theme() == DARK_THEME,
        })
        self.settings = s
        return s

    def _current_theme(self) -> str:
        """返回当前主题名（ttkbootstrap 2.x 通过 theme_use() 无参调用获取）"""
        try:
            return self.style.theme_use() or ""
        except Exception:
            return ""

    def _on_close(self):
        self._collect_settings()
        save_settings(self.settings)
        self.destroy()

    def _toggle_theme(self):
        dark = self._current_theme() != DARK_THEME
        self.style.theme_use(DARK_THEME if dark else LIGHT_THEME)
        self.btn_theme.configure(text="☀ 浅色" if dark else "🌙 深色")
        self._apply_text_colors()

    def _apply_text_colors(self):
        dark = self._current_theme() == DARK_THEME
        bg = "#1e1e1e" if dark else "#ffffff"
        fg = "#e6e6e6" if dark else "#1a1a1a"
        for widget in (self.txt_live, self.txt_srt, self.txt_trans):
            widget.configure(bg=bg, fg=fg, insertbackground=fg)
            widget.tag_configure("meta", foreground="#9a9a9a" if dark else "#7a7a7a")
            widget.tag_configure("ok", foreground="#4ec27a" if dark else "#1a9850")
            widget.tag_configure("err", foreground="#ff6b6b" if dark else "#d62728")
            widget.tag_configure("head", foreground="#c08bff" if dark else "#7a3fbf")
        self.txt_prompt.configure(bg=bg, fg=fg, insertbackground=fg)
        canvas = getattr(self, "_trans_canvas", None)
        if canvas is not None:
            canvas.configure(bg=bg)

    # ------------------------------------------------------------------ #
    # 文件列表操作
    # ------------------------------------------------------------------ #
    def _choose_model_dir(self):
        d = filedialog.askdirectory(title="选择模型目录", initialdir=self.var_model_dir.get() or str(PROJ_DIR))
        if d:
            self.var_model_dir.set(os.path.normpath(d))
            self._refresh_backend_hint()
            self._refresh_translate_hint()

    def _add_paths(self, paths):
        """把一批路径追加进列表（自动去重）"""
        if not paths:
            return
        self.settings["last_dir"] = os.path.dirname(paths[0])
        existing = {row["path"] for row in self.file_rows}
        added = 0
        for p in paths:
            p = os.path.normpath(p)
            if p in existing:
                continue
            iid = self.tree.insert("", END, values=(
                len(self.file_rows) + 1, Path(p).name, "-", "等待中"
            ))
            self.file_rows.append({
                "iid": iid, "path": p, "status": "等待中",
                "kind": "subtitle" if _is_subtitle_file(p) else "media",
            })
            existing.add(p)
            added += 1
        self._refresh_order()
        return added

    def _add_files(self):
        """添加音视频文件（也接受 .srt——会自动按「只翻译」处理）"""
        init = self.settings.get("last_dir") or str(PROJ_DIR)
        paths = filedialog.askopenfilenames(
            title="选择音频 / 视频文件（也可直接选 .srt 字幕）",
            initialdir=init,
            filetypes=[
                ("音视频文件", " ".join(f"*{e}" for e in AUDIO_EXTS)),
                ("字幕文件（只翻译）", " ".join(f"*{e}" for e in SUBTITLE_EXTS)),
                ("所有文件", "*.*"),
            ],
        )
        self._add_paths(paths)

    def _add_subtitle_files(self):
        """只添加 .srt 字幕文件：跳过识别，直接在已有字幕上翻译"""
        init = self.settings.get("last_dir") or str(PROJ_DIR)
        paths = filedialog.askopenfilenames(
            title="选择字幕文件（只翻译，不重新识别）",
            initialdir=init,
            filetypes=[
                ("字幕文件", " ".join(f"*{e}" for e in SUBTITLE_EXTS)),
                ("所有文件", "*.*"),
            ],
        )
        if paths and not self.var_enable_translate.get():
            # 字幕文件唯一的用途就是翻译，这里顺手帮用户打开开关，避免"加了没反应"
            if messagebox.askyesno(
                "需要开启翻译",
                "列表里的字幕文件只会做「翻译」这一步。\n\n"
                "当前「启用字幕翻译」是关闭的，是否现在打开？",
            ):
                self.var_enable_translate.set(True)
                self._refresh_translate_hint()
        self._add_paths(paths)


    def _remove_selected(self):
        for iid in list(self.tree.selection()):
            for i, row in enumerate(self.file_rows):
                if row["iid"] == iid:
                    self.file_rows.pop(i)
                    break
            self.tree.delete(iid)
        self._refresh_order()

    def _clear_files(self):
        for row in self.file_rows:
            self.tree.delete(row["iid"])
        self.file_rows = []
        self._refresh_order()

    def _move(self, delta: int):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        idx = next((i for i, r in enumerate(self.file_rows) if r["iid"] == iid), None)
        if idx is None:
            return
        new = idx + delta
        if new < 0 or new >= len(self.file_rows):
            return
        self.file_rows[idx], self.file_rows[new] = self.file_rows[new], self.file_rows[idx]

        for i, row in enumerate(self.file_rows):
            self.tree.move(row["iid"], "", i)
        self._refresh_order()
        self.tree.selection_set(iid)

    def _refresh_order(self):
        for i, row in enumerate(self.file_rows):
            self.tree.set(row["iid"], "order", i + 1)
        self.lbl_count.configure(text=f"共 {len(self.file_rows)} 个文件")

    def _set_row_status(self, index: int, status: str, speaker: str = None, tag: str = ""):
        if index < 0 or index >= len(self.file_rows):
            return
        iid = self.file_rows[index]["iid"]
        self.tree.set(iid, "status", status)
        if speaker is not None:
            self.tree.set(iid, "spk", speaker)
        self.tree.item(iid, tags=(tag,) if tag else ())

    def _open_output_dir(self):
        sel = self.tree.selection()
        target = None
        if sel:
            idx = next((i for i, r in enumerate(self.file_rows) if r["iid"] == sel[0]), None)
            if idx is not None:
                target = os.path.dirname(self.file_rows[idx]["path"])
        if not target:
            target = self.settings.get("last_dir") or str(PROJ_DIR)
        try:
            os.startfile(target)  # noqa: S606  (Windows)
        except Exception as e:
            messagebox.showerror("无法打开目录", str(e))

    # ------------------------------------------------------------------ #
    # 文本输出
    # ------------------------------------------------------------------ #
    def _append(self, widget: tk.Text, text: str, tag: str = None):
        widget.configure(state="normal")
        widget.insert(END, text, tag or ())
        widget.see(END)
        widget.configure(state="disabled")

    def _clear(self, widget: tk.Text):
        widget.configure(state="normal")
        widget.delete("1.0", END)
        widget.configure(state="disabled")

    def _speaker_tag(self, widget: tk.Text, speaker: str) -> str:
        """为说话人分配稳定的颜色 tag

        注意：Tk 的 tag 配置是**每个 Text 控件独立**的，同一个 tag 名在另一个 Text 上
        并不会自动继承样式，所以这里每次都要按控件注册一次。
        """
        if not speaker:
            return "meta"
        key = f"spk-{speaker}"
        if key not in self._speaker_tags:
            try:
                num = int("".join(ch for ch in speaker if ch.isdigit()) or 0)
            except ValueError:
                num = len(self._speaker_tags)
            self._speaker_tags[key] = SPEAKER_COLORS[num % len(SPEAKER_COLORS)]
        if key not in widget.tag_names():
            widget.tag_configure(
                key, foreground=self._speaker_tags[key], font=("Microsoft YaHei UI", 11, "bold")
            )
        return key

    # ------------------------------------------------------------------ #
    # 运行控制
    # ------------------------------------------------------------------ #
    def _start(self):
        if self.is_running:
            return
        if not self.file_rows:
            messagebox.showwarning("提示", "请先添加要处理的音频 / 视频文件，或 .srt 字幕文件。")
            return

        s = self._collect_settings()
        save_settings(s)

        # ---- 先把输入按「音视频 / 字幕」分流 ----
        # 音视频：解码 → 说话人 → 识别（→ 可选翻译）
        # 字幕  ：只翻译，完全不碰 ASR 与说话人识别，省掉整条识别链路的时间
        media_rows = [r for r in self.file_rows if _is_media_file(r["path"])]
        sub_rows = [r for r in self.file_rows if _is_subtitle_file(r["path"])]

        translate_on = bool(s["enable_translation"])
        usable_subs = sub_rows if translate_on else []
        if sub_rows and not translate_on:
            messagebox.showwarning(
                "字幕文件将被跳过",
                f"列表里有 {len(sub_rows)} 个字幕文件，但「启用字幕翻译」是关闭的。\n\n"
                "字幕文件的唯一处理方式是翻译，所以这些文件本次会被跳过。\n"
                "若只想翻译字幕，请到「翻译设置」标签页打开开关（无需加载识别模型）。",
            )
        if not media_rows and not usable_subs:
            messagebox.showwarning(
                "提示",
                "没有可处理的文件。\n\n"
                "· 音频 / 视频文件：需要语音识别模型\n"
                "· .srt 字幕文件：需要先开启「启用字幕翻译」",
            )
            return

        # 翻译前置检查（音视频要翻译、或列表里有字幕文件时都需要）
        translate_backend = ""
        if translate_on:
            from qwen_asr_gguf.inference.translator import (
                TranslationConfig,
                resolve_translation_backend,
                translation_model_available,
            )

            tcfg = TranslationConfig(model_dir=s["model_dir"], model_fn=s["translate_model"])
            if not translation_model_available(tcfg):
                messagebox.showerror(
                    "缺少翻译模型",
                    f"未找到翻译模型：\n{os.path.join(s['model_dir'], s['translate_model'])}\n\n"
                    "请把 Qwen3-8B-Q4_K_M.gguf 放到模型目录下，"
                    "或在「翻译设置」标签页里修改模型文件名。",
                )
                return

            translate_backend, tr_note = resolve_translation_backend(s["translate_backend"])
            want = (s["translate_backend"] or "auto").lower()
            if want in ("cuda", "vulkan") and translate_backend != want:
                if not messagebox.askyesno(
                    "加速后端不可用",
                    f"{tr_note}\n\n是否继续？（将使用 {translate_backend.upper()} 推理）",
                ):
                    return

        # 说话人识别与识别模型只在**有音视频文件**时才需要检查
        resolved_backend = ""
        if media_rows and s["enable_diarization"]:
            from qwen_asr_gguf.pipeline import (
                PipelineConfig,
                diarization_needs_hf_token,
                resolve_diarization_backend,
            )

            probe = PipelineConfig(
                model_dir=s["model_dir"],
                precision=s["precision"],
                enable_diarization=True,
                diarization_backend=s.get("diarization_backend", "auto"),
                diarization_model_dir=s.get("diarization_model_dir") or None,
                hf_token=s.get("hf_token") or None,
            )
            resolved = resolve_diarization_backend(probe)

            if not resolved:
                messagebox.showwarning(
                    "说话人识别不可用",
                    "没有任何可用的说话人识别后端，本次将跳过说话人识别，仅输出纯文本字幕。\n\n"
                    "二选一：\n"
                    "  1) 导出 ONNX 模型（推荐，支持 DirectML）：\n"
                    "     uv run python 30-Export-Diarization-ONNX.py\n"
                    "  2) 安装 pyannote：uv add pyannote.audio",
                )
                s["enable_diarization"] = False
                self.var_enable_diar.set(False)
            elif diarization_needs_hf_token(probe):
                ok = messagebox.askyesno(
                    "缺少 HuggingFace Token",
                    "pyannote 说话人分离模型是 HuggingFace 上的 gated 资源，需要 Token 才能下载。\n\n"
                    "当前未填写 Token，下载大概率失败（会降级为纯文本字幕）。\n"
                    "若使用 ONNX 后端则无需 Token。\n\n是否仍要继续？",
                )
                if not ok:
                    return

            if s["enable_diarization"] and resolved:
                resolved_backend = resolved

        # 检查识别模型（只有需要识别音视频时才校验）
        if media_rows:
            from qwen_asr_gguf.pipeline import find_missing_models

            missing = find_missing_models(s["model_dir"], s["precision"])
            if missing:
                messagebox.showerror(
                    "缺少模型文件",
                    "以下模型文件不存在：\n\n" + "\n".join(missing[:6]) +
                    "\n\n请到项目 Releases 页面下载模型并解压到 model 目录。",
                )
                return

        self._clear(self.txt_live)
        self._clear(self.txt_srt)
        self._clear(self.txt_trans)
        self._speaker_tags.clear()
        self._append(self.txt_live, f"任务开始：{time.strftime('%Y-%m-%d %H:%M:%S')}\n", "meta")
        if media_rows:
            self._append(self.txt_live, f"待识别音视频：{len(media_rows)} 个\n", "meta")
        if usable_subs:
            self._append(
                self.txt_live,
                f"待翻译字幕：{len(usable_subs)} 个（跳过语音识别，直接翻译）\n",
                "meta",
            )
        if resolved_backend:
            self._append(
                self.txt_live,
                f"说话人识别：{resolved_backend}  设备 {s.get('diarization_device', 'auto')}\n",
                "meta",
            )
        if translate_backend:
            self._append(
                self.txt_live,
                f"字幕翻译：{s['translate_model']}  后端 {translate_backend}"
                f"（每批 {s['translate_batch']} 段，译文后缀 {s['translate_suffix']}）\n",
                "meta",
            )

        for i, row in enumerate(self.file_rows):
            if _is_subtitle_file(row["path"]) and not translate_on:
                self._set_row_status(i, "跳过（未开启翻译）", "-", "")
            else:
                self._set_row_status(i, "等待中", "-", "")

        self.cancel_event = threading.Event()
        self._set_running(True)
        self._stat_started_at = time.time()

        self.worker = threading.Thread(
            target=self._worker_main,
            args=(s, [r["path"] for r in self.file_rows], self.cancel_event),
            daemon=True,
        )
        self.worker.start()
        self.after(60, self._drain_events)

    def _stop(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self._append(self.txt_live, "\n[已请求停止，正在收尾 ...]\n", "err")
            self.btn_stop.configure(state="disabled")

    def _set_running(self, running: bool):
        self.is_running = running
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")

    # ------------------------------------------------------------------ #
    # 工作线程
    # ------------------------------------------------------------------ #
    def _put(self, kind: str, **payload):
        self.events.put((kind, payload))

    def _worker_main(self, s: dict, files: list[str], cancel_event: threading.Event):
        engine = None
        translator = None
        try:
            from qwen_asr_gguf.pipeline import (
                PipelineConfig,
                TranscriptionPipeline,
                build_diarizer,
                build_engine,
                build_translator,
            )

            media_files = [p for p in files if _is_media_file(p)]
            sub_files = [p for p in files if _is_subtitle_file(p)]

            pcfg = PipelineConfig(
                model_dir=s["model_dir"],
                precision=s["precision"],
                onnx_provider=s["onnx_provider"],
                llm_use_gpu=s["llm_use_gpu"],
                use_vulkan=s["use_vulkan"],
                n_ctx=s["n_ctx"],
                chunk_size=s["chunk_size"],
                memory_num=s["memory_num"],
                language=s["language"] or None,
                context="",
                temperature=float(s.get("temperature", 0.4)),
                max_line_duration=s["max_line_duration"],
                max_line_chars=s["max_line_chars"],
                enable_diarization=s["enable_diarization"],
                diarization_backend=s.get("diarization_backend", "auto"),
                diarization_model_dir=s.get("diarization_model_dir") or None,
                hf_token=s["hf_token"] or None,
                diarization_device=s["diarization_device"],
                min_speakers=1,
                max_speakers=s["max_speakers"],
                export_txt=s["export_txt"],
                export_json=s["export_txt"],
                enable_translation=s.get("enable_translation", False),
                translation_model=s.get("translate_model", "Qwen3-8B-Q4_K_M.gguf"),
                translation_backend=s.get("translate_backend", "auto"),
                translation_prompt=s.get("translate_prompt") or _default_translation_prompt(),
                translation_batch_size=int(s.get("translate_batch", 1) or 1),
                translation_n_gpu_layers=int(s.get("translate_gpu_layers", -1)),
                translation_max_line_duration=float(s.get("translate_max_line_duration", 6.0)),
                translation_max_line_chars=int(s.get("translate_max_line_chars", 24)),
                translation_output_suffix=s.get("translate_suffix") or ".zh",
                verbose=False,
            )

            # ASR 引擎只在真的需要识别音视频时才初始化：
            # 纯字幕输入时完全不碰它，省掉几秒到几十秒的模型加载。
            if media_files:
                self._put("log", message="正在初始化 ASR 引擎 ...", tag="meta")
                t0 = time.time()
                engine = build_engine(pcfg, verbose=False)
                self._put("log", message=f"引擎初始化完成（{time.time()-t0:.1f} 秒）", tag="ok")
            else:
                self._put(
                    "log",
                    message=f"列表里只有 {len(sub_files)} 个字幕文件，跳过 ASR 引擎初始化。",
                    tag="meta",
                )

            # 翻译模型在整个批处理期间只加载一次
            if pcfg.enable_translation:
                self._put("log", message="正在加载翻译模型 ...", tag="meta")
                t1 = time.time()
                translator = build_translator(pcfg)
                if translator.load(on_status=lambda m: self._put("log", message=m, tag="meta")):
                    self._put(
                        "log",
                        message=f"翻译就绪（{translator.backend_note}，耗时 {time.time()-t1:.1f} 秒）",
                        tag="ok",
                    )
                else:
                    self._put(
                        "warning",
                        message=f"翻译模型加载失败，本次跳过翻译：{translator.last_error}",
                    )
                    translator = None
                    pcfg.enable_translation = False

            diarizer = (
                build_diarizer(pcfg) if (pcfg.enable_diarization and media_files) else None
            )
            pipeline = TranscriptionPipeline(engine, diarizer, pcfg, translator)

            num_speakers = int(s.get("num_speakers") or 0) or None

            for idx, path in enumerate(files):
                if cancel_event.is_set():
                    break

                if _is_subtitle_file(path):
                    if translator is None:
                        self._put(
                            "file_skipped",
                            file_index=idx,
                            path=path,
                            reason="未启用字幕翻译" if not pcfg.enable_translation else "翻译不可用",
                        )
                        continue
                    self._put("file_start", file_index=idx, path=path, subtitle=True)

                    def forward_sub(kind, payload, file_index=idx):
                        data = dict(payload)
                        data["file_index"] = file_index
                        self._put(kind, **data)

                    # 字幕文件：直接解析 + 翻译，跳过解码 / 说话人 / 识别
                    result = pipeline.translate_subtitle_file(
                        path, on_event=forward_sub, cancel_event=cancel_event
                    )
                    self._put("file_done", file_index=idx, result=result)
                    continue

                self._put("file_start", file_index=idx, path=path)

                def forward(kind, payload, file_index=idx):
                    """把流水线事件转发到 UI 线程（file_index 标识当前文件）"""
                    data = dict(payload)
                    data["file_index"] = file_index
                    self._put(kind, **data)

                result = pipeline.run(
                    path,
                    num_speakers=num_speakers,
                    on_event=forward,
                    cancel_event=cancel_event,
                    with_speaker=pcfg.enable_diarization,
                )
                self._put("file_done", file_index=idx, result=result)

            self._put("all_done")
        except Exception:
            self._put("fatal", message=traceback.format_exc())
        finally:
            if engine is not None:
                try:
                    engine.shutdown()
                except Exception:
                    pass
            if translator is not None:
                try:
                    translator.release()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # 事件循环
    # ------------------------------------------------------------------ #
    def _drain_events(self):
        pending_text: list[str] = []
        pending_trans: list[str] = []
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "text":
                    pending_text.append(payload.get("text", ""))
                    continue
                if kind == "translate_text":
                    pending_trans.append(payload.get("text", ""))
                    continue
                # 先把缓存的流式文本刷出去，保证顺序
                if pending_text:
                    self._append(self.txt_live, "".join(pending_text))
                    pending_text = []
                if pending_trans:
                    self._append(self.txt_trans, "".join(pending_trans))
                    pending_trans = []
                self._handle_event(kind, payload)
        except queue.Empty:
            pass

        if pending_text:
            self._append(self.txt_live, "".join(pending_text))
        if pending_trans:
            self._append(self.txt_trans, "".join(pending_trans))

        if self.is_running:
            self.after(60, self._drain_events)

    def _handle_event(self, kind: str, payload: dict):
        index = payload.get("file_index", -1)

        if kind == "log":
            self._append(self.txt_live, payload.get("message", "") + "\n", payload.get("tag", "meta"))

        elif kind == "status":
            self.lbl_status.configure(text=payload.get("message", ""))

        elif kind == "warning":
            self._append(self.txt_live, f"\n[警告] {payload.get('message','')}\n", "err")

        elif kind == "stage":
            stage = payload.get("stage")
            msg = payload.get("message", "")
            if "progress" in payload:
                self.progress.configure(value=float(payload["progress"]))
            if stage == "diarize":
                self.lbl_status.configure(text=msg)
            if stage == "done":
                self._append(self.txt_live, f"\n✔ {msg}\n", "ok")
            elif stage in ("load", "asr", "export", "translate"):
                self.lbl_status.configure(text=msg)

        elif kind == "chunk":
            i, total = payload.get("index", 0), payload.get("total", 1)
            speakers = payload.get("speakers") or []
            start, end = payload.get("start", 0.0), payload.get("end", 0.0)
            header = f"\n\n── 分片 {i+1}/{total}  [{start:.1f}s - {end:.1f}s] "
            if speakers:
                header += "".join(f"[{sp}] " for sp in speakers)
            header += "──\n"
            self._append(self.txt_live, header, "meta")

            if speakers:
                tag = self._speaker_tag(self.txt_live, speakers[0])
                self._append(self.txt_live, f"[{speakers[0]}] ", tag)

        elif kind == "progress":
            self.progress.configure(value=float(payload.get("progress", 0.0)))

        elif kind == "file":
            label = "译文" if payload.get("kind") == "translated_srt" else ""
            self._append(self.txt_live, f"已保存{label}: {payload.get('path','')}\n", "meta")

        elif kind == "file_start":
            name = Path(payload.get("path", "")).name
            only_translate = bool(payload.get("subtitle"))
            tag = "只翻译字幕" if only_translate else "音视频识别"
            self._append(
                self.txt_live,
                f"\n\n{'='*60}\n处理文件（{tag}）: {name}\n{'='*60}\n",
                "head",
            )
            self._set_row_status(
                index, "翻译中 ..." if only_translate else "识别中 ...", None, "running"
            )
            self.lbl_status.configure(
                text=("正在翻译: " if only_translate else "正在处理: ") + name
            )

        elif kind == "file_skipped":
            self._set_row_status(index, f"跳过（{payload.get('reason', '')}）", "-", "")
            self._append(
                self.txt_live,
                f"\n已跳过 {Path(payload.get('path', '')).name or '文件'}："
                f"{payload.get('reason', '')}\n",
                "meta",
            )

        elif kind == "file_done":
            self._on_file_done(index, payload.get("result"))

        elif kind == "fatal":
            self._append(self.txt_live, "\n[致命错误]\n" + payload.get("message", "") + "\n", "err")
            messagebox.showerror("运行出错", payload.get("message", "")[-1500:])

        elif kind == "all_done":
            self._set_running(False)
            self.progress.configure(value=1.0 if not (self.cancel_event and self.cancel_event.is_set()) else 0.0)
            elapsed = time.time() - self._stat_started_at
            self.lbl_status.configure(text=f"全部完成，用时 {elapsed:.1f} 秒")
            self._append(self.txt_live, f"\n任务结束，总用时 {elapsed:.1f} 秒。\n", "head")

    def _on_file_done(self, index: int, result):
        if result is None:
            return
        name = Path(result.audio_path).name
        only_translate = bool(getattr(result, "source_kind", "media") == "subtitle")

        if getattr(result, "cancelled", False):
            self._set_row_status(index, "已中断", "-", "failed")
        elif result.translated_srt_path:
            spk = "-"
            if result.speakers:
                spk = f"{result.speakers.num_speakers} 人"
            elif only_translate:
                labels = _speaker_labels(result.segments)
                if labels:
                    spk = f"{len(labels)} 人"
            if only_translate:
                status = (
                    f"完成 (仅翻译 {len(result.segments)} 条 → "
                    f"译文 {len(result.translated_segments)} 条)"
                )
            else:
                status = f"完成 ({len(result.segments)} 条字幕"
                status += f" / 译文 {len(result.translated_segments)} 条"
                status += ")"
            self._set_row_status(index, status, spk, "done")
        elif result.srt_path and only_translate:
            # 字幕读进来了，但没产生译文（翻译未开 / 失败）
            status = f"未翻译 ({len(result.segments)} 条字幕，未产生译文)"
            self._set_row_status(index, status, "-", "failed")
        elif result.srt_path:
            spk = f"{result.speakers.num_speakers} 人" if result.speakers else "-"
            status = f"完成 ({len(result.segments)} 条字幕"
            if result.translated_srt_path:
                status += f" / 译文 {len(result.translated_segments)} 条"
            status += ")"
            self._set_row_status(index, status, spk, "done")
        else:
            self._set_row_status(index, "失败", "-", "failed")

        # 追加到字幕预览（仅翻译模式下这块展示的是**输入字幕**，不是重新识别出来的）
        self._append(self.txt_srt, f"\n{'='*72}\n", "meta")
        self._append(self.txt_srt, f"文件: {name}\n", "head")
        if result.srt_path:
            label = "来源字幕（未重新识别）: " if only_translate else "字幕: "
            self._append(self.txt_srt, f"{label}{result.srt_path}\n", "meta")
        if result.speakers:
            self._append(
                self.txt_srt,
                f"说话人: {result.speakers.num_speakers} 位 "
                f"({', '.join(result.speakers.labels)})\n",
                "meta",
            )
        elif only_translate:
            labels = _speaker_labels(result.segments)
            if labels:
                self._append(
                    self.txt_srt,
                    f"说话人: {len(labels)} 位（来自字幕标记 {', '.join(labels)}）\n",
                    "meta",
                )
            else:
                self._append(self.txt_srt, "说话人: 字幕中无说话人标记\n", "meta")
        self._append(self.txt_srt, f"{'='*72}\n", "meta")

        for seg in result.segments:
            if seg.speaker:
                tag = self._speaker_tag(self.txt_srt, seg.speaker)
                self._append(self.txt_srt, f"[{seg.speaker}] ", tag)
            self._append(
                self.txt_srt,
                f"{seg.start_time:>8.2f} → {seg.end_time:<8.2f}  {seg.text}\n",
            )

        # 追加到译文预览
        if result.translated_srt_path:
            self._append(self.txt_trans, f"\n{'='*72}\n", "meta")
            self._append(self.txt_trans, f"文件: {name}\n", "head")
            self._append(self.txt_trans, f"译文: {result.translated_srt_path}\n", "meta")
            self._append(self.txt_trans, f"{'='*72}\n", "meta")
            for seg in result.translated_segments:
                if seg.speaker:
                    tag = self._speaker_tag(self.txt_trans, seg.speaker)
                    self._append(self.txt_trans, f"[{seg.speaker}] ", tag)
                self._append(
                    self.txt_trans,
                    f"{seg.start_time:>8.2f} → {seg.end_time:<8.2f}  {seg.text}\n",
                )


def main():
    # Windows 高 DPI 适配（必须在创建窗口前设置）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = ASRApp()
    app._apply_text_colors()
    app.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
