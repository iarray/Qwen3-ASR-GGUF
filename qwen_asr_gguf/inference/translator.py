# coding=utf-8
"""
translator.py - 字幕翻译引擎（Qwen3-8B GGUF + llama.cpp）

把 ASR 产出的字幕片段翻译成目标语言，保持时间轴与说话人标记：

    SubtitleSegment 列表
        -> group_segments_for_translation : 结尾没有句末标点的行与后续行合并
        -> LLM 逐块翻译（可按 batch_size 合并请求，失败自动回退逐条）
        -> build_translated_segments      : 译文按标点拆行、时长按比例分配
        -> 新的 SubtitleSegment 列表（每行仍带 [spk0]）

设计要点：
- 直接复用 `llama.py` 的 llama.cpp 绑定，**运行期不依赖 torch / transformers**；
- 加速后端可选 Vulkan / CUDA / CPU，缺哪个动态库会自动回退并在状态里说明；
- 任何一块翻译失败都回退成原文，不影响其它块与其它文件；
- 翻译请求之间共享同一份已加载的模型，避免每个文件重复加载 4.7 GB 权重。
"""
import codecs
import gc
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from . import llama
from . import subtitle as subtitle_mod
from .schema import SubtitleSegment
from .subtitle import TranslationUnit, build_translated_segments, group_segments_for_translation

# llama.cpp 动态库所在目录（ggml-vulkan.dll / ggml-cuda.dll 都在这里）
BIN_DIR = Path(__file__).resolve().parent / "bin"

DEFAULT_TRANSLATION_MODEL = "Qwen3-8B-Q4_K_M.gguf"

# 用户给出的默认提示词；`{text}` 会被替换成待翻译内容
DEFAULT_TRANSLATION_PROMPT = """你是专业字幕翻译，只输出译文，不要额外解释。
翻译规则：
1. 英译中，符合中文口语习惯，不要直译英文句式，去掉欧化长句；
2. 保留原句语气，自然流畅，像原生中文台词；
3. 短句拆分，适合字幕阅读，不要堆砌长句；
4. 专有名词保持统一。
原文：
{text}"""

# 一次翻译多段时追加的格式约束
_BATCH_INSTRUCTION = (
    "本次共有 {count} 段内容需要翻译。请严格逐段处理，每段输出一行，"
    "格式为「序号. 译文」，序号必须与输入一致；"
    "不要合并、不要遗漏、不要输出原文，也不要输出任何解释。"
)

_NUMBERED_LINE_RE = re.compile(r"^\s*(?:第)?\s*(\d+)\s*(?:段|条|个|句)?\s*[.、．)）:：]\s*(.*)$")

_PREFIX_CLEANUP_RE = re.compile(r"^(?:译文|翻译结果|翻译|translation)\s*[:：]\s*", re.IGNORECASE)

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("「", "」"), ("『", "』"), ("“", "”"), ("‘", "’"))

# 思考块（Qwen3 的 ` thinking` / `<｜end▁of▁thinking｜>` 是词表里的**特殊 token**，
# 不同版本 GGUF 里 `token_to_piece` 的显示可能是 `<think>` 或带零宽空格的写法，
# 所以这里用候选列表动态定位，而不是硬编码 token id。）
_THINK_BEGIN_CANDIDATES = ("<think>", "\u200bthinking")
_THINK_END_CANDIDATES = ("</think>", "\u200b</think>", "\u200bthinking\n\n\u200b")
# 特殊 token 的 id 一定落在词表尾部
_SPECIAL_TOKEN_MIN_ID = 100000


def _pick_special_token(model, candidates: Sequence[str]) -> int:
    """在词表里定位一个特殊 token；找不到返回 -1"""
    for text in candidates:
        try:
            ids = model.tokenize(text)
        except Exception:
            continue
        if len(ids) == 1 and ids[0] >= _SPECIAL_TOKEN_MIN_ID:
            return ids[0]
    return -1


# --------------------------------------------------------------------------- #
# 1. 加速后端探测
# --------------------------------------------------------------------------- #
def _ggml_backend_present(name: str) -> bool:
    """判断 llama.cpp 二进制里是否带了某个 GPU 后端（看动态库是否存在）"""
    if not BIN_DIR.is_dir():
        return False
    for pattern in (f"*{name}*.dll", f"*{name}*.so", f"*{name}*.dylib"):
        try:
            if any(BIN_DIR.glob(pattern)):
                return True
        except OSError:
            continue
    return False


def available_gpu_backends() -> List[str]:
    """返回本机 llama.cpp 实际支持的 GPU 后端列表（cuda 优先）"""
    out: List[str] = []
    if _ggml_backend_present("cuda"):
        out.append("cuda")
    if _ggml_backend_present("vulkan"):
        out.append("vulkan")
    return out


def describe_backends() -> str:
    """给界面用的一句话描述"""
    gpus = available_gpu_backends()
    if not gpus:
        return "未检测到 GPU 后端，将使用 CPU"
    names = {"cuda": "CUDA", "vulkan": "Vulkan"}
    return "可用加速后端：" + " / ".join(names.get(g, g) for g in gpus)


def resolve_translation_backend(backend: str = "auto") -> Tuple[str, str]:
    """把用户选择解析成实际生效的后端

    Returns:
        (实际后端, 说明文案)，实际后端取值 `cuda` / `vulkan` / `cpu`
    """
    want = (backend or "auto").strip().lower()
    alias = {
        "auto": "auto", "": "auto",
        "vulkan": "vulkan", "vk": "vulkan", "amd": "vulkan",
        "cuda": "cuda", "nvidia": "cuda", "n卡": "cuda",
        "cpu": "cpu", "none": "cpu", "off": "cpu",
    }
    want = alias.get(want, want)

    has_cuda = _ggml_backend_present("cuda")
    has_vulkan = _ggml_backend_present("vulkan")

    if want == "cpu":
        return "cpu", "已指定使用 CPU 推理"
    if want == "cuda":
        if has_cuda:
            return "cuda", "使用 CUDA 后端（ggml-cuda）"
        return "cpu", (
            "未找到 ggml-cuda 动态库，已回退 CPU；"
            "如需 CUDA，请下载 llama.cpp 的 CUDA 版本替换 inference/bin/ 下的 ggml*.dll"
        )
    if want == "vulkan":
        if has_vulkan:
            return "vulkan", "使用 Vulkan 后端（ggml-vulkan）"
        return "cpu", "未找到 ggml-vulkan 动态库，已回退 CPU"
    # auto：有 CUDA 用 CUDA，其次 Vulkan，最后 CPU
    if has_cuda:
        return "cuda", "自动选择：CUDA 后端"
    if has_vulkan:
        return "vulkan", "自动选择：Vulkan 后端"
    return "cpu", "未检测到 GPU 后端，使用 CPU 推理"


# --------------------------------------------------------------------------- #
# 2. 提示词构造与结果解析
# --------------------------------------------------------------------------- #
def build_translation_prompt(template: str, items: Sequence[str]) -> Tuple[str, str]:
    """把模板渲染成 (system, user) 两段

    模板中 `{text}` 之前的部分当系统提示，之后的部分（通常为空）并入用户输入。
    一次翻译多段时自动追加编号输出格式要求。
    """
    template = template or DEFAULT_TRANSLATION_PROMPT
    if "{text}" in template:
        head, _, tail = template.partition("{text}")
    else:
        head, tail = template, ""

    system = head.strip("\n").rstrip()

    if len(items) == 1:
        user = f"{items[0]}{tail}".strip()
        return system, user

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))
    extra = _BATCH_INSTRUCTION.format(count=len(items))
    system = f"{system}\n{extra}" if system else extra
    user = f"{numbered}{tail}".strip()
    return system, user


def clean_translation(text: str) -> str:
    """清理模型输出里常见的多余成分（思考块、前缀、包裹引号、markdown 标记）"""
    t = (text or "").strip()
    if not t:
        return ""
    # 模型偶尔会先吐一段思考过程（Enable Thinking 被打开时），把整块去掉
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = t.replace("</think>", "").replace("<think>", "").strip()
    t = _PREFIX_CLEANUP_RE.sub("", t).strip()
    t = t.replace("**", "").strip()
    for a, b in _QUOTE_PAIRS:
        if len(t) >= 2 and t[0] == a and t[-1] == b:
            t = t[1:-1].strip()
    t = re.sub(r"^(?:译文|翻译)\s*[:：]\s*", "", t).strip()
    return t


def parse_numbered_translations(reply: str, count: int) -> Optional[List[str]]:
    """解析「序号. 译文」格式的批量回复；任何一段缺失都返回 None（触发逐条回退）

    单段（count == 1）时更宽松：模型偶尔会直接给出译文而不带编号，这种情况下整段回复
    就当作译文。
    """
    if not reply or count <= 0:
        return None

    slots: List[List[str]] = [[] for _ in range(count)]
    cur = -1
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _NUMBERED_LINE_RE.match(line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < count:
                cur = idx
                body = m.group(2).strip()
                if body:
                    slots[idx].append(body)
                continue
        if cur >= 0:
            slots[cur].append(line)

    if all(slots):
        return [subtitle_mod._join_pieces(lines).strip() for lines in slots]

    if count == 1:
        lines = [ln.strip() for ln in reply.splitlines() if ln.strip()]
        text = subtitle_mod._join_pieces(lines).strip()
        return [text] if text else None

    return None


# --------------------------------------------------------------------------- #
# 3. 引擎
# --------------------------------------------------------------------------- #
@dataclass
class TranslationConfig:
    """翻译引擎配置"""
    model_dir: str = "model"
    model_fn: str = DEFAULT_TRANSLATION_MODEL

    # 加速后端：auto / vulkan / cuda / cpu
    backend: str = "auto"
    n_gpu_layers: int = -1        # -1 = 全部层卸载到 GPU
    n_ctx: int = 4096
    n_batch: int = 1024
    n_ubatch: int = 512
    n_threads: int = 0            # 0 = 自动

    # 采样
    temperature: float = 0.3
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    penalty_last_n: int = 64
    max_new_tokens: int = 512

    # 翻译逻辑
    prompt_template: str = DEFAULT_TRANSLATION_PROMPT
    enable_thinking: bool = False   # Qwen3 思考模式（字幕翻译不需要，默认关）
    # 单次请求里放几段。实测 >1 的提速有限（约 10%），但模型在同时处理多段时
    # 容易丢掉语气词、把专有名词当成普通词，所以默认逐段翻译保证质量。
    batch_size: int = 1

    # 译文排版
    max_line_duration: float = 6.0
    max_line_chars: int = 24
    min_line_duration: float = 0.6
    max_group_duration: float = 30.0   # 单个合并块的时间上限（安全阀）

    output_suffix: str = ".zh"      # 译文文件名后缀：xxx.zh.srt
    enabled: bool = False
    verbose: bool = False


class SubtitleTranslator:
    """字幕翻译器：加载一次 Qwen3-8B，反复翻译多个文件的字幕"""

    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._model = None
        self._ctx = None
        self._backend = ""
        self._backend_note = ""
        self._last_error = ""
        self._load_failed = False

        # 缓存特殊 token
        self.ID_IM_START = -1
        self.ID_IM_END = -1
        self.ID_THINK_BEGIN = -1
        self.ID_THINK_END = -1

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    @property
    def loaded(self) -> bool:
        return self._model is not None and self._ctx is not None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def backend_note(self) -> str:
        return self._backend_note

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def model_path(self) -> str:
        return os.path.join(self.config.model_dir, self.config.model_fn)

    # ------------------------------------------------------------------ #
    # 加载 / 释放
    # ------------------------------------------------------------------ #
    def load(self, on_status: Optional[Callable[[str], None]] = None) -> bool:
        """加载翻译模型；已加载则直接返回 True"""
        if self.loaded:
            return True
        if self._load_failed:
            return False

        path = self.model_path
        if not os.path.exists(path):
            self._load_failed = True
            self._last_error = f"翻译模型不存在: {path}"
            self._log(self._last_error, on_status)
            return False

        cfg = self.config
        backend, note = resolve_translation_backend(cfg.backend)
        self._backend, self._backend_note = backend, note
        self._log(f"翻译后端：{note}", on_status)

        t0 = time.time()
        try:
            kthreads = int(cfg.n_threads) if cfg.n_threads else None
            model = llama.LlamaModel(
                path,
                n_gpu_layers=int(cfg.n_gpu_layers),
                use_gpu=1 if backend != "cpu" else 0,
            )
            ctx = llama.LlamaContext(
                model,
                n_ctx=int(cfg.n_ctx),
                n_batch=int(cfg.n_batch),
                n_ubatch=int(cfg.n_ubatch),
                embeddings=False,
                n_threads=kthreads,
            )
        except Exception as e:
            self._load_failed = True
            self._last_error = f"翻译模型加载失败: {e}"
            self._log(self._last_error, on_status)
            return False

        self._model = model
        self._ctx = ctx
        self.ID_IM_START = model.token_to_id("<|im_start|>")
        self.ID_IM_END = model.token_to_id("<|im_end|>")
        self.ID_THINK_BEGIN = _pick_special_token(model, _THINK_BEGIN_CANDIDATES)
        self.ID_THINK_END = _pick_special_token(model, _THINK_END_CANDIDATES)

        if self.ID_IM_START < 0 or self.ID_IM_END < 0:
            self._load_failed = True
            self._last_error = "该模型不是 ChatML 格式（缺少 <|im_start|> / <|im_end|>），无法用于翻译"
            self._log(self._last_error, on_status)
            self.release()
            return False

        self._log(
            f"翻译模型就绪（{os.path.basename(path)}，耗时 {time.time() - t0:.1f} 秒）",
            on_status,
        )
        return True

    def release(self):
        """释放模型占用的显存 / 内存"""
        self._ctx = None
        self._model = None
        self._load_failed = False
        self.ID_IM_START = -1
        self.ID_IM_END = -1
        self.ID_THINK_BEGIN = -1
        self.ID_THINK_END = -1
        gc.collect()

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #
    def _encode_chat(self, system: str, user: str) -> List[int]:
        """按 Qwen3 的 ChatML 模板构造 token 序列"""
        m = self._model
        ids: List[int] = []
        ids.append(self.ID_IM_START)
        ids += m.tokenize(f"system\n{system}")
        ids.append(self.ID_IM_END)
        ids.append(self.ID_IM_START)
        ids += m.tokenize(f"user\n{user}")
        ids.append(self.ID_IM_END)
        ids.append(self.ID_IM_START)
        ids += m.tokenize("assistant\n")

        if not self.config.enable_thinking and self.ID_THINK_BEGIN >= 0 and self.ID_THINK_END >= 0:
            # 非思考模式：按 Qwen3 官方模板把空思考块整段写进 prompt
            # （` thinking\n\n\n\n`）。缺了这段，模型会自己开一段思考再回答，
            # 那部分内容会混进字幕里。
            ids.append(self.ID_THINK_BEGIN)
            ids += m.tokenize("\n\n")
            ids.append(self.ID_THINK_END)
            ids += m.tokenize("\n\n")
        return ids

    def _generate(
        self,
        system: str,
        user: str,
        on_stream: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        cfg = self.config
        model, ctx = self._model, self._ctx
        if model is None or ctx is None:
            return ""

        tokens = self._encode_chat(system, user)
        if len(tokens) >= int(cfg.n_ctx) - 8:
            raise RuntimeError(f"输入过长（{len(tokens)} tokens），超出上下文 {cfg.n_ctx}")

        batch = llama.LlamaBatch(len(tokens), 0, 1)
        batch.set_tokens(tokens)
        ctx.clear_kv_cache()
        if ctx.decode(batch) != 0:
            raise RuntimeError("LLM prefill 失败")

        sampler = llama.LlamaSampler(
            temperature=float(cfg.temperature),
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
            repeat_penalty=float(cfg.repeat_penalty),
            penalty_last_n=int(cfg.penalty_last_n),
        )
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pieces: List[str] = []
        try:
            tok = sampler.sample(ctx.ptr)
            for _ in range(int(cfg.max_new_tokens)):
                if cancel_event is not None and cancel_event.is_set():
                    break
                if tok < 0 or tok in (model.eos_token, self.ID_IM_END):
                    break
                if ctx.decode_token(tok) != 0:
                    break
                piece = decoder.decode(model.token_to_bytes(tok), final=False)
                if piece:
                    pieces.append(piece)
                    if on_stream is not None:
                        try:
                            on_stream(piece)
                        except Exception:
                            pass
                tok = sampler.sample(ctx.ptr)
            tail = decoder.decode(b"", final=True)
            if tail:
                pieces.append(tail)
        finally:
            sampler.free()

        return clean_translation("".join(pieces))

    def _translate_one(
        self,
        text: str,
        on_stream: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        system, user = build_translation_prompt(self.config.prompt_template, [text])
        try:
            return self._generate(system, user, on_stream=on_stream, cancel_event=cancel_event)
        except Exception as e:
            self._last_error = str(e)
            if self.config.verbose:
                print(f"[翻译] 单段失败: {e}")
            return ""

    def _translate_batch(
        self,
        texts: Sequence[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[List[str]]:
        """一次请求翻译多段；格式不符合预期时返回 None 让上层回退"""
        system, user = build_translation_prompt(self.config.prompt_template, texts)
        try:
            reply = self._generate(system, user, cancel_event=cancel_event)
        except Exception as e:
            self._last_error = str(e)
            return None
        parsed = parse_numbered_translations(reply, len(texts))
        return parsed

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def translate_texts(
        self,
        texts: Sequence[str],
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_stream: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """批量翻译纯文本列表，逐条返回译文（失败时回退为原文）"""
        results: List[str] = list(texts)
        if not texts:
            return results
        if not self.load():
            return results

        batch_size = max(1, int(self.config.batch_size or 1))
        done = 0
        for start in range(0, len(texts), batch_size):
            if cancel_event is not None and cancel_event.is_set():
                break
            chunk = list(texts[start:start + batch_size])

            if len(chunk) == 1:
                out = [self._translate_one(chunk[0], on_stream=on_stream, cancel_event=cancel_event)]
            else:
                out = self._translate_batch(chunk, cancel_event=cancel_event)
                if out is None:
                    # 批量格式没解析成功 → 逐条重来，保证不丢内容
                    out = [
                        self._translate_one(t, on_stream=None, cancel_event=cancel_event)
                        for t in chunk
                    ]

            for i, value in enumerate(out):
                if value:
                    results[start + i] = value
            done += len(chunk)
            if on_progress is not None:
                try:
                    on_progress(min(done, len(texts)), len(texts))
                except Exception:
                    pass

        return results

    def translate_segments(
        self,
        segments: Sequence[SubtitleSegment],
        with_speaker: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_stream: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        on_group: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[SubtitleSegment]:
        """翻译字幕片段列表，返回新的片段列表（时间轴与说话人标记保持不变）

        Args:
            on_group: `(已完成块数, 总块数, 当前块的原文)` 回调，便于界面展示进度
        """
        if not segments:
            return []

        units, translations = self.translate_units(
            segments,
            on_progress=on_progress,
            on_stream=on_stream,
            cancel_event=cancel_event,
            on_group=on_group,
        )
        if not units:
            return []

        return build_translated_segments(
            units,
            translations,
            with_speaker=with_speaker,
            max_duration=self.config.max_line_duration,
            max_chars=self.config.max_line_chars,
            min_duration=self.config.min_line_duration,
        )

    def translate_units(
        self,
        segments: Sequence[SubtitleSegment],
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_stream: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        on_group: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[TranslationUnit], List[str]]:
        """只做「分组 + 翻译」，返回 (翻译单元列表, 逐单元译文)"""
        units = group_segments_for_translation(
            segments, max_group_duration=float(self.config.max_group_duration)
        )
        if not units:
            return [], []

        total = len(units)

        def _progress(cur: int, _tot: int):
            if on_progress is not None:
                try:
                    on_progress(cur, total)
                except Exception:
                    pass

        def _stream(piece: str):
            if on_stream is not None:
                try:
                    on_stream(piece)
                except Exception:
                    pass

        if on_group is not None:
            try:
                on_group(0, total, units[0].text)
            except Exception:
                pass

        translations = self.translate_texts(
            [u.text for u in units],
            on_progress=_progress,
            on_stream=_stream,
            cancel_event=cancel_event,
        )
        return units, translations

    # ------------------------------------------------------------------ #
    def _log(self, message: str, on_status: Optional[Callable[[str], None]] = None):
        if on_status is not None:
            try:
                on_status(message)
                return
            except Exception:
                pass
        if self.config.verbose:
            print(f"[翻译] {message}")


# --------------------------------------------------------------------------- #
# 4. 便捷函数
# --------------------------------------------------------------------------- #
def translation_model_available(config: Optional[TranslationConfig] = None) -> bool:
    """翻译模型文件是否就绪"""
    cfg = config or TranslationConfig()
    return os.path.exists(os.path.join(cfg.model_dir, cfg.model_fn))


def missing_translation_models(config: Optional[TranslationConfig] = None) -> List[str]:
    cfg = config or TranslationConfig()
    path = os.path.join(cfg.model_dir, cfg.model_fn)
    return [] if os.path.exists(path) else [path]


def build_translator(config: Optional[TranslationConfig] = None) -> SubtitleTranslator:
    return SubtitleTranslator(config)
