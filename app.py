# coding=utf-8
"""
app.py - 语音识别 + 说话人识别 桌面端 (ttkbootstrap)

功能：
1. 可配置说话人数（留空 / 0 表示自动检测）
2. 批量添加音频或视频文件，按列表顺序逐个识别并生成字幕
3. 界面实时显示识别内容与说话人，并预览最终 SRT
4. 字幕输出到原音频文件目录，文件名与音频文件一致

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

SPEAKER_COLORS = [
    "#0b6ed0", "#d62728", "#1a9850", "#e07b00", "#7b3fbf",
    "#0f9aa8", "#b8459b", "#6b6b6b", "#8a6d00", "#3d5a80",
]

LIGHT_THEME = "bootstrap-light"
DARK_THEME = "bootstrap-dark"


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
        mid = ttk.Labelframe(self, text=" 待处理文件（按列表顺序逐个识别） ", padding=8)
        mid.pack(fill=X, **pad)

        bar = ttk.Frame(mid)
        bar.pack(fill=X, pady=(0, 6))

        ttk.Button(bar, text="＋ 添加文件", bootstyle="primary", command=self._add_files, width=12).pack(side=LEFT)
        ttk.Button(bar, text="－ 移除选中", bootstyle="secondary-outline", command=self._remove_selected, width=12).pack(side=LEFT, padx=6)
        ttk.Button(bar, text="清空列表", bootstyle="secondary-outline", command=self._clear_files, width=10).pack(side=LEFT)
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

        try:
            self._refresh_backend_hint()
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
        for widget in (self.txt_live, self.txt_srt):
            widget.configure(bg=bg, fg=fg, insertbackground=fg)
            widget.tag_configure("meta", foreground="#9a9a9a" if dark else "#7a7a7a")
            widget.tag_configure("ok", foreground="#4ec27a" if dark else "#1a9850")
            widget.tag_configure("err", foreground="#ff6b6b" if dark else "#d62728")
            widget.tag_configure("head", foreground="#c08bff" if dark else "#7a3fbf")

    # ------------------------------------------------------------------ #
    # 文件列表操作
    # ------------------------------------------------------------------ #
    def _choose_model_dir(self):
        d = filedialog.askdirectory(title="选择模型目录", initialdir=self.var_model_dir.get() or str(PROJ_DIR))
        if d:
            self.var_model_dir.set(os.path.normpath(d))
            self._refresh_backend_hint()

    def _add_files(self):
        init = self.settings.get("last_dir") or str(PROJ_DIR)
        paths = filedialog.askopenfilenames(
            title="选择音频 / 视频文件",
            initialdir=init,
            filetypes=[
                ("音视频文件", " ".join(f"*{e}" for e in AUDIO_EXTS)),
                ("所有文件", "*.*"),
            ],
        )
        if not paths:
            return
        self.settings["last_dir"] = os.path.dirname(paths[0])
        existing = {row["path"] for row in self.file_rows}
        for p in paths:
            p = os.path.normpath(p)
            if p in existing:
                continue
            iid = self.tree.insert("", END, values=(
                len(self.file_rows) + 1, Path(p).name, "-", "等待中"
            ))
            self.file_rows.append({"iid": iid, "path": p, "status": "等待中"})
        self._refresh_order()

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
        """为说话人分配稳定的颜色 tag"""
        if not speaker:
            return "meta"
        key = f"spk-{speaker}"
        if key not in self._speaker_tags:
            try:
                num = int("".join(ch for ch in speaker if ch.isdigit()) or 0)
            except ValueError:
                num = len(self._speaker_tags)
            color = SPEAKER_COLORS[num % len(SPEAKER_COLORS)]
            widget.tag_configure(key, foreground=color, font=("Microsoft YaHei UI", 11, "bold"))
            self._speaker_tags[key] = color
        return key

    # ------------------------------------------------------------------ #
    # 运行控制
    # ------------------------------------------------------------------ #
    def _start(self):
        if self.is_running:
            return
        if not self.file_rows:
            messagebox.showwarning("提示", "请先添加要识别的音频或视频文件。")
            return

        s = self._collect_settings()
        save_settings(s)

        # 说话人识别前置检查（按后端区分：ONNX 后端不需要 HF Token）
        resolved_backend = ""
        if s["enable_diarization"]:
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

        # 检查模型
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
        self._speaker_tags.clear()
        self._append(self.txt_live, f"任务开始：{time.strftime('%Y-%m-%d %H:%M:%S')}\n", "meta")
        if resolved_backend:
            self._append(
                self.txt_live,
                f"说话人识别：{resolved_backend}  设备 {s.get('diarization_device', 'auto')}\n",
                "meta",
            )

        for i, row in enumerate(self.file_rows):
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
        try:
            from qwen_asr_gguf.pipeline import (
                PipelineConfig,
                TranscriptionPipeline,
                build_diarizer,
                build_engine,
            )

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
                verbose=False,
            )

            self._put("log", message="正在初始化 ASR 引擎 ...", tag="meta")
            t0 = time.time()
            engine = build_engine(pcfg, verbose=False)
            self._put("log", message=f"引擎初始化完成（{time.time()-t0:.1f} 秒）", tag="ok")

            diarizer = build_diarizer(pcfg) if pcfg.enable_diarization else None
            pipeline = TranscriptionPipeline(engine, diarizer, pcfg)

            num_speakers = int(s.get("num_speakers") or 0) or None

            for idx, path in enumerate(files):
                if cancel_event.is_set():
                    break
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

    # ------------------------------------------------------------------ #
    # 事件循环
    # ------------------------------------------------------------------ #
    def _drain_events(self):
        pending_text: list[str] = []
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "text":
                    pending_text.append(payload.get("text", ""))
                    continue
                # 先把缓存的流式文本刷出去，保证顺序
                if pending_text:
                    self._append(self.txt_live, "".join(pending_text))
                    pending_text = []
                self._handle_event(kind, payload)
        except queue.Empty:
            pass

        if pending_text:
            self._append(self.txt_live, "".join(pending_text))

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
            elif stage in ("load", "asr", "export"):
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
            self._append(self.txt_live, f"已保存: {payload.get('path','')}\n", "meta")

        elif kind == "file_start":
            name = Path(payload.get("path", "")).name
            self._append(self.txt_live, f"\n\n{'='*60}\n处理文件: {name}\n{'='*60}\n", "head")
            self._set_row_status(index, "识别中 ...", "-", "running")
            self.lbl_status.configure(text=f"正在处理: {name}")

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
        if getattr(result, "cancelled", False):
            self._set_row_status(index, "已中断", "-", "failed")
        elif result.srt_path:
            spk = f"{result.speakers.num_speakers} 人" if result.speakers else "-"
            self._set_row_status(index, f"完成 ({len(result.segments)} 条字幕)", spk, "done")
        else:
            self._set_row_status(index, "失败", "-", "failed")

        # 追加到字幕预览
        self._append(self.txt_srt, f"\n{'='*72}\n", "meta")
        self._append(self.txt_srt, f"文件: {name}\n", "head")
        if result.srt_path:
            self._append(self.txt_srt, f"字幕: {result.srt_path}\n", "meta")
        if result.speakers:
            self._append(
                self.txt_srt,
                f"说话人: {result.speakers.num_speakers} 位 "
                f"({', '.join(result.speakers.labels)})\n",
                "meta",
            )
        self._append(self.txt_srt, f"{'='*72}\n", "meta")

        for seg in result.segments:
            if seg.speaker:
                tag = self._speaker_tag(self.txt_srt, seg.speaker)
                self._append(self.txt_srt, f"[{seg.speaker}] ", tag)
            self._append(
                self.txt_srt,
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
