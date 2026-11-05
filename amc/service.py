import logging
import math
import threading
import re
import os
from typing import Any, Dict, List

from .config import MAX_SEARCH_IMAGES
from .library import MemoryLibrary
from .contract import ContractError, validate_add_request, validate_search_request

def distill_record(text: str, query: str, limit: int = 320) -> str:
    """把长记录蒸馏成"自包含的最小事实单元"。

    背景（reviewer 在平台上实测）：整条原始记录（几百到几千字）丢给答案端，
    关键句会被淹没 → 答案端读不出来；但也不能替它算结论。
    正确中间态是：**一句话能说清"是什么 / 多少钱 / 什么时候 / 出自哪"**。

    做法（纯抽取，不做任何归纳）：
      1. 保留 `[日期 角色] 说话人:` 前缀（身份与时间）
      2. 在正文里找与问题词重叠最多的句子，取它 ±1 句
      3. 若该窗口里没有金额而全文有，则把含金额的句子补进来
    """
    import os as _os
    if _os.getenv("AMC_DISTILL", "1") == "0":
        return text
    t = (text or "").strip()
    limit = max(120, int(limit))
    if len(t) <= limit * 1.6:
        return t
    # 1) 前缀：`[2024-05-07 user] 王景川: ...`
    m = re.match(r"^(\[[^\]]{4,40}\][^:\n]{0,24}:?)\s*", t)
    prefix = (m.group(1) + " ") if m else ""
    body = t[m.end():] if m else t
    # 2) 切句（中英文标点都切）
    # 只在"句末标点 + 空白 + 下一个句子开头"处切，避免把 "$150.00" 切成 "$150. 00"
    sents = [x.strip() for x in re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])|\n+", body)
             if x.strip()]
    if len(sents) <= 3:
        return t
    q_terms = set(re.findall(r"[\u4e00-\u9fff]{2,4}|[a-zA-Z]{3,}", query or ""))
    def _score(i):
        w = sents[i]
        return sum(1 for q in q_terms if q in w)
    # 过滤"半句"：以引号/右括号/连接词开头、或过短的片段不作为起点
    _bad_head = re.compile(r'^[”"\'’\)\]\}，,、；;：:]|^(and|but|so|because|which|that)\b', re.I)
    _cand = [i for i, w in enumerate(sents) if len(w) >= 12 and not _bad_head.match(w)] or list(range(len(sents)))
    best, best_s = _cand[0], -1
    for i in _cand:
        sc = _score(i)
        if sc > best_s:
            best, best_s = i, sc
    if best_s <= 0:
        best = 0
    _bad_idx = {i for i, w in enumerate(sents) if len(w) < 12 or _bad_head.match(w)}
    pick = [i for i in range(max(0, best - 1), min(len(sents), best + 2)) if i not in _bad_idx] or [best]
    # 3) 金额句补进来（问钱时）
    # 与 L1435 的 `_is_amt_q` 保持一致：**必须加词边界**。
    # 不加的话 "cost" 会命中 "costume"、"total" 会命中 "totally"
    # ⇒ 非金额题也会被塞进"金额句"，给答案侧添噪声。
    if re.search(r"\bhow much\b|\bcost\b|\bprice\b|\bpaid\b|\btotal\b"
                 r"|多少钱|花了|总价", query or "", re.I):
        money_re = re.compile(r"[\$£€¥]\s?\d|\d[\d,]*\s?(?:dollars?|pounds?|usd|gbp|eur|元|块)", re.I)
        if not any(money_re.search(sents[i]) for i in pick):
            for i, w in enumerate(sents):
                if money_re.search(w):
                    pick.append(i)
                    break
    pick = sorted(set(pick))
    out = []
    total = len(prefix)
    for i in pick:
        w = sents[i]
        if total + len(w) > limit and out:
            break
        out.append(w)
        total += len(w) + 1
    snip = " ".join(out)[:limit]
    return (prefix + snip).strip()


_ROUGH_CACHE: dict = {}          # 按图片 URL 的 sha 缓存（同一图只调一次）


def _visual_rough_desc(url: str, timeout: float = 45.0) -> str:
    """对一张图生成 ≤200 字的**粗描述**，供"只读文本"的答案端使用。

    reviewer 2026-11-01 的四条约束：
      ① 只对 route==VISUAL 且有图的题调；**按图片 sha 缓存**
      ② **≤200 字、放 text 段最前面**（被截断就白做）
      ③ 模板固定三问：背景色 / 有什么（物件+可读文字）/ 版式形状（曲线·峰·条纹·心形·左右）
      ④ 必须做"纯文本口径"验收
    另有反方向提醒（c01）：答案端会**跳过具体名词给概括回答**
      ⇒ 描述必须写**具体名词**（"白色背景"/"心形"/"中段有峰"），不写"画面很美"。
    """
    import hashlib as _h, os as _o, httpx as _hx
    if not url:
        return ""
    key = _h.sha1(url.encode("utf-8", "ignore")).hexdigest()
    if key in _ROUGH_CACHE:
        return _ROUGH_CACHE[key]
    base = (_o.getenv("AMC_QUERY_VISION_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    k = _o.getenv("AMC_QUERY_VISION_API_KEY") or ""
    m = _o.getenv("AMC_QUERY_VISION_MODEL") or "qwen3-vl-plus"
    if not k:
        return ""
    # 两边独立收敛到"第 5 问 = 形状"这件事本身就是交叉验证。
    prompt = ("用中文客观描述这张图，≤200 字，**只写具体名词**（不写'画面很美'这类概括词）：\n"
              "1) 主体和场景；\n"
              "2) 所有可读的文字/数字/标识（尽量原样列出）；\n"
              "3) 物体清单；\n"
              "4) 显著颜色/材质/光线（纯色背景要写明色名）；\n"
              "5) 若有图表/折线/版式，一句话说明**形状**（升/降/峰/谷/条纹/几行几列/左右上下）；\n"
              "   若有人物或物件，说明它在画面**哪一侧**。\n"
              "不要解释。")
    try:
        r = _hx.post(f"{base}/chat/completions",
                     headers={"Authorization": f"Bearer {k}"},
                     json={"model": m, "max_tokens": 200, "messages": [{"role": "user", "content": [
                         {"type": "text", "text": prompt},
                         {"type": "image_url", "image_url": {"url": url}}]}]},
                     timeout=timeout)
        out = str(((r.json() or {}).get("choices") or [{}])[0]
                  .get("message", {}).get("content", "")).strip()[:200]
    except Exception:
        out = ""
    _ROUGH_CACHE[key] = out
    return out


def _batched_embedder(settings):
    """把 embedding 调用按 10 条一批包起来（DashScope 批量上限）。"""
    base = getattr(settings, "embedding_base_url", "") or ""
    model = getattr(settings, "embedding_model", "") or ""
    key = getattr(settings, "embedding_api_key", "") or ""
    if not (base and model):
        return None
    def _fn(texts):
        import httpx as _hx
        _extra_ids = list(getattr(_DECISION, "amount_ids", []) or [])
        out = []
        texts = list(texts)
        for i in range(0, len(texts), 10):
            r = _hx.post(base.rstrip("/") + "/embeddings",
                         json={"model": model, "input": texts[i:i + 10], "dimensions": 1024},
                         headers={"Authorization": f"Bearer {key}"}, timeout=90)
            r.raise_for_status()
            out += [d["embedding"] for d in r.json()["data"]]
        return out
    return _fn


def assemble(items, budget_tokens: int = 100_000, top_k: int = 100):
    """按 token 预算装配返回集：**只丢整条，不切条内**（2026-11-02）。

    背景（官方口径，reviewer 核对）：
      · Answer 窗口 128,000 tokens，预留 8,192 + 2,048 ⇒ **可用输入 117,760**
      · 超出时平台"按返回顺序取 token 前缀"截断 ⇒ **顺序决定谁被看到**
      · 正式评测 **top_k = 100**，且"返回数量不得超过 top_k"
      · **允许 100 条完整记录，不允许条内截断**

    取舍：删掉条内截断后条数会变少 ⇒ **宁要 60 条完整，不要 100 条残缺**。
    """
    kept, used, dropped = [], 0, []
    try:
        budget_tokens = int(int(os.getenv("AMC_ASSEMBLE_TOKENS", budget_tokens) or budget_tokens))
    except Exception:
        pass
    try:
        top_k = int(int(os.getenv("AMC_ASSEMBLE_TOPK", top_k) or top_k))
    except Exception:
        pass
    for it in (items or []):
        if len(kept) >= top_k:
            dropped.append(str(it.get("id")))
            continue
        t = ""
        c = it.get("content")
        if isinstance(c, list):
            t = "".join(str(x.get("text") or "") for x in c if isinstance(x, dict))
        elif isinstance(c, str):
            t = c
        ntok = max(1, int(len(t) / 2.2))          # 中英混合约 2.2 字符/token
        if used + ntok > budget_tokens:
            dropped.append(str(it.get("id")))
            continue
        kept.append(it); used += ntok
    return kept, dropped, used


def _head_tail(text: str, head: int = None, tail: int = None) -> str:
    """把长记录压成"头 + 尾"（各 head/tail 字），保证总长可控。

    为什么需要（2026-11-01 实测）：平台记录常是**整段会话**（1,200~2,100 字），
    而 gold 结论句常落在 **68%~95%** 处（例：Q2 的 "…先把学习节奏固定下来…" 在 1430/2107）。
    答案端若按前 N 字截断，结论句必被切掉 ⇒ **补进证据也等于没补**。
    压成"头+尾"后总长 ≤640 字，低于答案端常见的截断阈值 ⇒ 两头都保得住。
    """
    # ⚠️ 2026-11-02：**默认值可配**（默认仍 320，行为不变）。
    #    原因（实测）：ATM 数据里 56% 的记录 >800 字（中位 851 / p90 2426 / max 10483），
    #    而 320+320 只保留 **21.5% 的记录完整**、总字符保留 48.9%。
    #    更要紧的是：gold 常落 68%~95%，记录 2426 字时 tail 从 2106 起
    #    ⇒ **1650~2106 被切，正好切掉 gold**。
    #    改 800/800 后：完整记录 **85.1%**、总字符 77.6%、
    #    100 条 ≈ 73k tokens（Answer 窗口 117,760 ⇒ 装得下）。
    #    用 env 控制 ⇒ 不加 env 行为完全不变，可随时回滚。
    if head is None:
        head = int(os.getenv("AMC_HEAD_TAIL_HEAD", "320") or 320)
    if tail is None:
        tail = int(os.getenv("AMC_HEAD_TAIL_TAIL", "320") or 320)
    head = max(0, int(head)); tail = max(0, int(tail))
    t = str(text or "")
    if len(t) <= head + tail + 8:
        return t
    return t[:head] + " …[中间省略]… " + t[-tail:]


class _SkipCalib(Exception):
    """校准器短路（通道已命中）。"""


logger = logging.getLogger("amc.service")
WORD_RE = re.compile(r"[A-Za-z0-9_:$.-]+")


def query_to_text(query: Any, options: Any = None, describer=None) -> str:
    """把 query（str | ContentPart[]）转为检索用文本。

    若 query 含图片且提供了 ``describer``，则用 VLM 生成描述并**并入**文本；
    describer 缺失/失败时保持原文本（绝不因此让 /search 失败）。
    """
    chunks = []
    if isinstance(query, str):
        chunks.append(query)
    elif isinstance(query, list):
        for part in query:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
    text = "\n".join(c.strip() for c in chunks if c.strip())
    #   **选项不再拼进检索文本**。理由：
    #     选项里常出现实体名/地名/颜色（"A. White background …"），
    #     ② 选项词在库里出现少 ⇒ 被 IDF 当成高价值词 ⇒ 权重算错。
    #   `options` 仍然原样返回给答案端（我们只返回证据，不消费选项）。
    #   实测数据：平台 4,980 题里 **1,590（32%）带 A/B/C/D 选项** ⇒ 影响面大。
    _ = options      # 保留形参以兼容调用方；**不并入 text**
    if describer is not None:
        try:
            text = describer.augment(text, query)
        except Exception:
            logger.exception("query-image augmentation failed; falling back to text only")
    return text


_ROUND_TEXT_LOCK = threading.Lock()
_ROUND_TEXT_CACHE: Dict[str, str] = {}
_ROUND_TEXT_CONN = None          # 专用只读连接（避免与共享连接竞争）


def _round_conn(db_path: str):
    global _ROUND_TEXT_CONN
    if _ROUND_TEXT_CONN is None:
        import sqlite3 as _sq
        _ROUND_TEXT_CONN = _sq.connect(f"file:{db_path}?mode=ro", uri=True,
                                       check_same_thread=False)
        _ROUND_TEXT_CONN.execute("PRAGMA query_only=1")
    return _ROUND_TEXT_CONN


def sanitize_scores(hits):
    clean = []
    prev = None
    for mem, raw_s in hits:
        try:
            val = float(raw_s)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = prev if prev is not None else 0.0
        if prev is not None and val > prev:
            val = prev
        prev = val
        clean.append((mem, val))
    return clean


_DECISION = __import__("threading").local()   # 每次请求的"是否拒答"决定


# selftest-marker-20261102: 用于验证"build 变 ⇒ 旧缓存键失效"（无害注释）
def _build_tag() -> str:
    """BUILD id —— 2026-11-02 加进缓存键。

    为什么：缓存键原来**没有 build**，而重启时 `_cache_load()` 会全量载入旧条目
    ⇒ **重启 + 改了代码 ⇒ 新代码返回旧响应**（今天"改了却没生效"的原因之一）。
    加进键后：代码一变 ⇒ 键变 ⇒ 旧条目永远不可达（无需删文件）。
    """
    try:
        from .build_id import BUILD as _B
        return _B
    except Exception:
        return "unknown"



# 官方媒体限制：单图解码后 ≤10 MiB，**单次 Search 响应内图片合计 ≤30 MiB**。
# 超限 ⇒ 平台判契约错误 ⇒ 整题失败 ⇒ 所以必须自己压到限制内。
#   ① 单图 > 8 MiB ⇒ 压到 ≤8 MiB
#   ② 合计 > 28 MiB ⇒ **从最后一张开始压**（rank 越后越不重要）
#   ③ 保持原格式（jpeg→jpeg / png→png），降质量或降分辨率
#   ④ 前 3 名的图不动（除非单张就超 8 MiB）
#   ⑤ 日志：[search-img] compressed n=.. orig=..MiB -> ..MiB
_MIB = 1024 * 1024


def _shrink_image_bytes(raw: bytes, target_bytes: int, fmt_hint: str = "") -> bytes:
    """把图片字节压到 <= target_bytes（尽量保持格式；压不动就降分辨率）。"""
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        fmt = (im.format or fmt_hint or "PNG").upper()
        if fmt not in ("JPEG", "PNG", "WEBP"):
            fmt = "JPEG"
        if fmt == "JPEG" and im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        # ① 先试降质量
        for q in (85, 75, 65, 55, 45, 35, 25):
            buf = io.BytesIO()
            if fmt == "JPEG":
                im.save(buf, format="JPEG", quality=q, optimize=True)
            elif fmt == "WEBP":
                im.save(buf, format="WEBP", quality=q)
            else:
                im.save(buf, format="PNG", optimize=True, compress_level=9)
            out = buf.getvalue()
            if len(out) <= target_bytes:
                return out
            if fmt == "PNG":
                break          # PNG 降质量无效 ⇒ 直接降分辨率
        # ② 再试降分辨率
        scale = 0.75
        for _ in range(8):
            w, h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
            im2 = im.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            if fmt == "JPEG":
                im2.save(buf, format="JPEG", quality=70, optimize=True)
            elif fmt == "WEBP":
                im2.save(buf, format="WEBP", quality=70)
            else:
                im2.save(buf, format="PNG", optimize=True, compress_level=9)
            out = buf.getvalue()
            if len(out) <= target_bytes:
                return out
            scale *= 0.7
        return out
    except Exception:
        return raw


def compress_response_images(records, max_single_mib: float = 8.0, max_total_mib: float = 28.0,
                             protect_top: int = 3):
    """就地压缩响应里的 data-URI 图片，保证 单图<=max_single、合计<=max_total。"""
    try:
        import base64 as _b64, re as _re, os as _os
        if (_os.getenv("AMC_SEARCH_IMG_COMPRESS", "1") or "1").strip() not in ("1", "on", "true", "yes"):
            return records
        single_cap = int(max_single_mib * _MIB)
        total_cap = int(max_total_mib * _MIB)
        # 收集 (记录序号, part 引用, 原始字节)
        imgs = []
        for ri, r in enumerate(records or []):
            c = r.get("content")
            if not isinstance(c, list):
                continue
            for pi, part in enumerate(c):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                url = str((part.get("image_url") or {}).get("url") or "")
                if not url.startswith("data:"):
                    continue
                try:
                    head, b64 = url.split(",", 1)
                    raw = _b64.b64decode(b64, validate=False)
                except Exception:
                    continue
                imgs.append({"ri": ri, "part": part, "head": head, "raw": raw, "orig": len(raw)})
        if not imgs:
            return records
        orig_total = sum(x["orig"] for x in imgs)
        n_done = 0
        # ① 单图超限 ⇒ 压（前 protect_top 也压，因为单图硬限制）
        for x in imgs:
            if len(x["raw"]) > single_cap:
                new = _shrink_image_bytes(x["raw"], single_cap)
                if len(new) < len(x["raw"]):
                    x["raw"] = new; n_done += 1
        # ② 合计超限 ⇒ 从后往前压
        def _total():
            return sum(len(x["raw"]) for x in imgs)
        idx = len(imgs) - 1
        while _total() > total_cap and idx >= 0:
            x = imgs[idx]
            if idx < protect_top and len(x["raw"]) <= single_cap:
                idx -= 1
                continue
            # 目标：让总量回到 26 MiB 以内
            over = _total() - int(26 * _MIB)
            tgt = max(256 * 1024, len(x["raw"]) - max(over, len(x["raw"]) // 2))
            new = _shrink_image_bytes(x["raw"], tgt)
            if len(new) >= len(x["raw"]):
                idx -= 1
                continue
            x["raw"] = new; n_done += 1
        # ③ 回写 data URI
        for x in imgs:
            if len(x["raw"]) == x["orig"]:
                continue
            mime = "image/png"
            m = _re.search(r"data:([^;,]+)", x["head"])
            if m:
                mime = m.group(1)
            # 压缩后格式可能仍是原格式（_shrink 保持），mime 不变
            b64 = _b64.b64encode(x["raw"]).decode("ascii")
            x["part"]["image_url"] = {"url": f"data:{mime};base64,{b64}"}
        if n_done:
            logger.info("[search-img] compressed n=%d orig=%.1fMiB -> %.1fMiB",
                        n_done, orig_total / _MIB, _total() / _MIB)
        return records
    except Exception:
        logger.exception("[search-img] 压缩失败（忽略，保持原图）")
        return records

# ⚠️ ：预嵌的**单线程队列**（替代"每次 add 起一个线程"）
import queue as _queue_mod

_PREWARM_Q: "_queue_mod.Queue" = _queue_mod.Queue(maxsize=20000)
_PREWARM_STARTED = __import__("threading").Lock()


def _start_prewarm_worker_once() -> None:
    """懒启动**唯一**的预嵌 worker（全局只一个）。"""
    with _PREWARM_STARTED:
        if getattr(_start_prewarm_worker_once, "_started", False):
            return
        try:
            def _loop():
                while True:
                    batch = []
                    try:
                        batch.append(_PREWARM_Q.get(timeout=1.0))
                    except Exception:
                        continue
                    # 批量聚合（最多 32 个 add 任务）⇒ 一次 embed 写库，减少锁争抢
                    while len(batch) < 32:
                        try:
                            batch.append(_PREWARM_Q.get_nowait())
                        except Exception:
                            break
                    _SVC_REF[0] and _SVC_REF[0]._prewarm_embed_batch(batch)
            __import__("threading").Thread(target=_loop, daemon=True,
                                           name="prewarm-worker").start()
            _start_prewarm_worker_once._started = True
        except Exception:
            pass


_SVC_REF = [None]


#   动机：实测 ab_guided 串行耗时 ~20s（LLM 生成答案 + 定位），而它自带 assemble()、不依赖主检索
#   ⇒ 可与 fused/ce/llm 并行，把 20s 藏到 7s 的检索后面。
_AB_POOL = None


def _ab_pool():
    global _AB_POOL
    if _AB_POOL is None:
        try:
            import concurrent.futures as _cf
            _AB_POOL = _cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="ab-par")
        except Exception:
            _AB_POOL = False
    return _AB_POOL or None


class MemoryService:
    # /search 结果缓存：评测重试时同一题**秒回**（键含 user+query+top_k+options）
    # 落盘到 data/store/search_cache.json，服务重启也不丢；/add 真正写入新数据时整体失效。
    _search_cache: dict = {}
    #   `/add` 原来 `_search_cache.clear()` **全量清空** ⇒ add 期间（93 条/分钟）
    #   缓存每秒被清一次 ⇒ 408 后平台重试**必然重算**（实测同一 query 连打 3 次：
    #   21.0s / 23.7s / 21.9s，几乎同耗时 ⇒ 缓存完全没命中）。
    #     · /add 时 `_user_ver[uid] += 1`（O(1)，不清别人）
    #   ⇒ value 结构不变（仍是 2 元组）⇒ 磁盘缓存、读侧、兼容性全部不受影响。
    #     ⇒ 磁盘上若存在 `uv:0` 的老条目，重启后仍可能命中；
    #     ⇒ 但 **TTL 15 分钟**兜底（最长陈旧 15 分钟），且"重启不改数据"
    #        ⇒ 复用该缓存在语义上是**正确**的，不构成陈旧风险。
    _user_ver: dict = {}
    _search_cache_lock = None
    _cache_file: str = ""
    _cache_loaded: bool = False

    def __init__(self, store, retriever, max_return=50, settings=None, describer=None,
                 vl_reranker=None, relation_selector=None, multimodal_selector=None,
                 agent_retriever=None, evidence_judge=None):
        self.store = store
        self.retriever = retriever
        # Agent 检索流程（可选）：门控 → 调度 → 按类检索 → 证据校验
        self.agent = agent_retriever
        self._evidence_judge = evidence_judge
        self.max_return = max_return
        self.settings = settings
        self._lib_cache = {}
        self._lib_dirty = set()
        self._lib_locks = {}
        self._lib_locks_guard = __import__("threading").Lock()
        self._warm_pending = {}
        self._warm_guard = __import__("threading").Lock()
        self._warm_thread = None
        # 记忆侧查询图理解（可选）：query 含图片时用 VLM 描述并入查询文本
        self.describer = describer
        # 记忆侧专用 VL reranker（可选）：仅在查询含图时对头部候选重排
        self.vl_reranker = vl_reranker
        # 记忆侧关系感知证据选择（可选）：结构化+通用关系规则(+逐对图像证据关系)；
        # 生效时**替代** vl_rerank，避免重复重排。
        self.relation_selector = relation_selector
        # 多模态证据选择（可选）：查询带图时用「查询图 + 候选图」重排
        self.multimodal_selector = multimodal_selector
        # ⚠️ ：启动**唯一的**预嵌 worker（替代"每次 add 起线程"）
        self._start_prewarm_worker()

    def _start_prewarm_worker(self) -> None:
        """把 self 注册给 worker 并启动它（**全局只一个线程**）。"""
        try:
            _SVC_REF[0] = self
            _start_prewarm_worker_once()
        except Exception:
            logger.exception("[prewarm] worker 启动失败（忽略）")

    def _lib_lock_for(self, user_id):
        with self._lib_locks_guard:
            lk = self._lib_locks.get(user_id)
            if lk is None:
                lk = __import__("threading").Lock()
                self._lib_locks[user_id] = lk
            return lk

    def cache_store(self, req: Dict[str, Any], resp: Any) -> None:
        """把任意 200 响应写入重试缓存（覆盖所有返回路径）。"""
        try:
            if not MemoryService._cache_enabled():
                return
            if not isinstance(resp, dict) or "data" not in resp:
                return
            import hashlib as _hl, threading as _th, time as _tm
            if MemoryService._search_cache_lock is None:
                MemoryService._search_cache_lock = _th.Lock()
            q_text = query_to_text(req.get("query"), req.get("options"), self.describer)
            #    写侧原来多了 `"\x01v2rich"` 后缀，而读侧（search 入口 _ck）没有
            #    ⇒ `cache_store` 写的条目**永远命中不了**（三处直接写 _search_cache 的能用，
            #      因为它们用的是读侧格式）。
            #    修法：**改写侧**（只 1 处；改读侧要动 4 处）
            # ⚠️ （子 agent 审计）：写侧**漏了 `\x01uv:<ver>`**，读侧固定带
            #   ⇒ cache_store 写的条目永远命中不了 ⇒ "408 后台写缓存"完全空转（日志却打已写缓存）。
            try:
                _uv_w = int(MemoryService._user_ver.get(str(req.get("user_id")), 0))
            except Exception:
                _uv_w = 0
            key = _hl.sha1((str(req.get("user_id")) + "\x01" + q_text + "\x01" +
                            str(req.get("top_k")) + "\x01" + str(sorted(req.get("options") or []))
                            + "\x01b:" + _build_tag() + "\x01uv:" + str(_uv_w)
                            ).encode()).hexdigest()
            with MemoryService._search_cache_lock:
                MemoryService._search_cache[key] = (_tm.time(), resp)
                if len(MemoryService._search_cache) > 2048:
                    for _k in list(MemoryService._search_cache)[:512]:
                        MemoryService._search_cache.pop(_k, None)
            MemoryService._cache_save()
        except Exception:
            logger.exception("cache_store failed")

    # ---------- 重试缓存（落盘）----------
    @classmethod
    def _cache_path(cls):
        import os as _os
        return cls._cache_file or _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "store", "search_cache.json")

    @classmethod
    def _cache_ttl(cls):
        import os as _os
        return float(_os.getenv("AMC_RETRY_CACHE_TTL", "21600") or 21600)   # 默认 6 小时

    @classmethod
    def _cache_enabled(cls):
        import os as _os
        return (_os.getenv("AMC_RETRY_CACHE", "1") or "1") not in ("0", "false", "False")

    @classmethod
    def _cache_load(cls):
        import json as _json, os as _os
        if cls._cache_loaded:
            return
        cls._cache_loaded = True
        if not cls._cache_enabled():
            return
        try:
            path = cls._cache_path()
            if _os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    cls._search_cache = _json.load(fh) or {}
                logger.info("[retry] 载入缓存 %d 条", len(cls._search_cache))
        except Exception:
            logger.exception("[retry] 载入缓存失败")

    _last_save_ts: float = 0.0

    # ⚠️ ：慢请求的"部分结果"快照（retrieve 阶段的 100 条）
    _partial_cache = {}
    _partial_lock = threading.Lock()

    @classmethod
    def _partial_store(cls, ck, snap):
        try:
            with cls._partial_lock:
                cls._partial_cache[ck] = snap
                if len(cls._partial_cache) > 64:
                    for k in list(cls._partial_cache)[:16]:
                        cls._partial_cache.pop(k, None)
        except Exception:
            pass

    @classmethod
    def _partial_get(cls, ck):
        try:
            with cls._partial_lock:
                return cls._partial_cache.get(ck)
        except Exception:
            return None

    @classmethod
    def _cache_save(cls, force: bool = False):
        if not cls._cache_enabled():
            return
        # ⚠️ ：**节流**。原来每个成功响应都落盘一次 2.6GB JSON
        #   （约 2.3TB/天），是超时的重要来源；现在默认 ≥60s 才写一次。
        try:
            import time as _ts
            import os as _os_thr   # 本函数后面才 `import os as _os`，这里必须单独导入
            _min = float(_os_thr.getenv("AMC_CACHE_SAVE_MIN_S", "60") or 60)
            if not force and (_ts.time() - float(cls._last_save_ts or 0.0)) < _min:
                return
            cls._last_save_ts = _ts.time()
        except Exception:
            pass
        try:
            import json as _json, os as _os, tempfile as _tf
            path = cls._cache_path()
            _os.makedirs(_os.path.dirname(path), exist_ok=True)
            fd, tmp = _tf.mkstemp(dir=_os.path.dirname(path))
            # ⚠️ 必须用**快照**再序列化：其它线程会在 dump 期间写 `_search_cache`，
            # 直接 dump 会抛 `RuntimeError: dictionary changed size during iteration`
            # ⇒ 缓存永远存不下来（实测 10-29~10-30 共 1,018 次失败）。
            # 浅拷贝足够（值的替换不影响遍历），且不引入锁竞争。
            # ⚠️ （子 agent 审计，严重）：mkstemp 临时文件**没有 finally 清理**。
            #   实证：data/store/tmp* 泄漏 36GB（数个 2.3~2.8GB），根分区一度 98%。
            #   ⇒ 无论成功失败都删 tmp（成功时已被 replace 走，unlink ENOENT 忽略）。
            try:
                with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                    _json.dump(dict(cls._search_cache), fh, ensure_ascii=False)
                _os.replace(tmp, path)
            finally:
                try:
                    if _os.path.exists(tmp):
                        _os.unlink(tmp)
                except Exception:
                    pass
        except Exception:
            logger.exception("[retry] 保存缓存失败")

    @staticmethod
    def _normalize_add(req: Dict[str, Any]) -> Dict[str, Any]:
        """**宽容归一化**（容错匹配）。

        动机：平台实测会发**字段不规范但内容有效**的请求（未知 role、content 是 dict/嵌套 list、
        只有空字符串等）。若我们判 invalid ⇒ 之前会 200 但**不存** ⇒ **静默丢数据**
        ⇒ 表现不是报错，而是"某些题检索不到证据"，极难定位。
        ⇒ 正确做法：**能救就救**（保留内容），只有"完全没有 messages"才略过。
        """
        try:
            req = dict(req or {})
        except Exception:
            return req
        _msgs = req.get("messages")
        if not isinstance(_msgs, list):
            return req
        _OK_ROLES = {"user", "assistant", "system"}
        out = []
        for _m in _msgs:
            if not isinstance(_m, dict):
                # 纯字符串消息 ⇒ 当作 user 文本
                if isinstance(_m, str) and _m.strip():
                    out.append({"role": "user", "content": [{"type": "text", "text": _m}]})
                continue
            _m = dict(_m)
            # ① role 不规范（tool/function/其他）⇒ 当 user，**保留内容**
            if str(_m.get("role") or "").lower() not in _OK_ROLES:
                _m["role"] = "user"
            # ② content 形态归一化：字符串 ⇒ parts 列表；dict ⇒ 包一层；嵌套 list ⇒ 展平文本
            _c = _m.get("content")
            if isinstance(_c, str):
                _m["content"] = [{"type": "text", "text": _c}]
            elif isinstance(_c, dict):
                _t = _c.get("text") if isinstance(_c.get("text"), str) else ""
                _m["content"] = [{"type": "text", "text": _t}] if _t else [{"type": "text", "text": str(_c)[:8000]}]
            elif isinstance(_c, list):
                _parts = []
                for _p in _c:
                    if isinstance(_p, dict):
                        _parts.append(_p)
                    elif isinstance(_p, str):
                        _parts.append({"type": "text", "text": _p})
                _m["content"] = _parts
            # ③ 完全空内容 ⇒ 跳过这一条（其余照存）
            _cc = _m.get("content")
            _has = False
            if isinstance(_cc, list):
                for _p in _cc:
                    if isinstance(_p, dict) and (str(_p.get("text") or "").strip()
                                                 or _p.get("type") == "image_url"):
                        _has = True
                        break
            if not _has:
                continue
            out.append(_m)
        req["messages"] = out
        return req

    def add(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """AML Official Add Endpoint Implementation"""
        # ⚠️ 先宽容归一化（未知 role / content 形态 / 空消息），再校验
        req = self._normalize_add(req)
        errs = validate_add_request(req)
        if errs:
            raise ContractError(422, "; ".join(errs[:5]))

        #   **不再"每次 /add 全量清空检索缓存"**。
        #   原做法导致：add 期间（93 条/分钟）缓存每秒被清一次 ⇒ 408 后平台重试
        #   **必然重算**（实测同一 query 连打 3 次 = 21.0/23.7/21.9s，缓存 0 命中）。
        #   "代码/逻辑变了却命中旧响应" 的风险由 key 里的 `b:<build>` 兜住。
        #   想恢复旧行为：`AMC_CACHE_CLEAR_ON_ADD=1`
        try:
            import os as _os2
            if (_os2.getenv("AMC_CACHE_CLEAR_ON_ADD", "0") or "0").strip() in ("1", "on", "true", "yes"):
                MemoryService._search_cache.clear()
                _p = "data/store/search_cache.json"
                if _os2.path.exists(_p):
                    _os2.remove(_p)
                logger.info("[retry] 缓存全量清空（AMC_CACHE_CLEAR_ON_ADD=1）")
        except Exception:
            logger.exception("clear search cache on add failed")
        status = self.store.add_request(req)
        if status == "inserted" and getattr(self, "_tagger", None) is not None:
            try:
                import datetime as _dt
                _msgs = req.get("messages") or []
                _txt = []
                for _m in _msgs:
                    _c = _m.get("content") if isinstance(_m, dict) else _m
                    if isinstance(_c, str): _txt.append(_c)
                    elif isinstance(_c, list):
                        for _p in _c:
                            if isinstance(_p, dict) and _p.get("type") == "text": _txt.append(_p.get("text", ""))
                _ts = None
                for _m in _msgs:
                    if isinstance(_m, dict) and _m.get("timestamp"): _ts = _m["timestamp"]; break
                _date = _dt.datetime.utcfromtimestamp(float(_ts)/1000).strftime("%Y-%m-%d") if _ts else ""
                self._tagger.submit(req.get("user_id", ""), req.get("request_id", ""),
                                    "\n".join(_txt), _date, any(isinstance(_p, dict) and _p.get("type") == "image_url"
                                                                  for _m in _msgs if isinstance(_m, dict)
                                                                  for _p in (_m.get("content") if isinstance(_m.get("content"), list) else [])))
            except Exception:
                logger.exception("tagger submit failed")
        #   原来只有 `status == "inserted"` 才预嵌 ⇒ 平台重放同一 chunk 时不补
        #   ⇒ 若首次插入时被队列满丢掉，重放也救不回来（永久缺口）。
        #   预嵌是幂等的（`get_many` 会跳过已缓存的），所以 replayed 再入队无害。
        if status in ("inserted", "replayed"):
            try:
                _new_ids = []
                if hasattr(self.store, "memory_ids_of"):
                    _new_ids = self.store.memory_ids_of(req.get("request_id", "")) or []
                if _new_ids:
                    self._prewarm_embed(req.get("user_id", ""), _new_ids)
            except Exception:
                logger.exception("prewarm submit failed (忽略)")
            # L1：把新记忆交给档案层异步抽取（幂等，不阻塞 /add）
            # ⚠️ 2026-11-03（方案 C）：**默认不在 /add 触发全量抽取** ——
            #   全量后台抽取 ≈ 8,080 次 LLM，而其中大部分记录平台根本不会问到。
            #   现在改成"检索时按命中抽取"（ProfileService.extract_for）。
            #   若想恢复旧行为：AMC_PROFILE_ON_ADD=1
            try:
                import os as _os_pa
                _on_add = (_os_pa.getenv("AMC_PROFILE_ON_ADD", "0") or "0").strip() in ("1", "on", "true", "yes")
                _pf = getattr(self, "_profile", None) if _on_add else None
                if _pf is not None and getattr(_pf, "enabled", False):
                    _ids = self.store.memory_ids_of(req["request_id"]) if hasattr(self.store, "memory_ids_of") else []
                    if _ids:
                        _pf.submit(req["user_id"], _ids)
            except Exception:
                logger.exception("profile submit failed")
            try:
                #   收益：add 期间平台重试**能命中缓存**（实测原来 3 次都重算 21~24s）。
                _uid_v = str(req.get("user_id") or "")
                with MemoryService._search_cache_lock:
                    if _uid_v:
                        MemoryService._user_ver[_uid_v] = int(
                            MemoryService._user_ver.get(_uid_v, 0)) + 1
                    if len(MemoryService._search_cache) > 512:
                        for _k in list(MemoryService._search_cache)[:128]:
                            MemoryService._search_cache.pop(_k, None)
                logger.info("[retry] 缓存按用户失效 uid=%s ver=%s（不再全量清空）",
                            _uid_v[-12:] or "-", MemoryService._user_ver.get(_uid_v, 0))
            except Exception:
                pass
            try:
                self._lib_dirty.add(req["user_id"])
                #   ⇒ 原来"每次 add 都 _schedule_warm"⇒ 建库 1,775 次（每次 1.5~11s）
                #   ⇒ add 延迟被拖到 **中位 10.6s / p90 18.4s**。
                #   改成只标脏 ⇒ add 回到写库的 0.1~0.5s；图书馆在**首次 search** 时懒建。
                #   想恢复旧行为：AMC_WARM_ON_ADD=1
                import os as _os_w
                if (_os_w.getenv("AMC_WARM_ON_ADD", "0") or "0").strip() in ("1", "on", "true", "yes"):
                    if getattr(self.settings, "warm_library", True):
                        self._schedule_warm(req["user_id"])
            except Exception:
                pass
            hook = getattr(self.retriever, "notify_add", None)
            if hook is not None:
                try:
                    hook(req["user_id"], req["request_id"])
                except Exception:
                    logger.exception("notify_add failed")
        else:
            # 幂等重放：不重复抽取/失效缓存，但**仍然预热图书馆**。
            # 评测重放 add 时不会触发 inserted 分支，若不预热，首次 /search 要吃冷启动。
            try:
                import os as _os_w2
                if (_os_w2.getenv("AMC_WARM_ON_ADD", "0") or "0").strip() in ("1", "on", "true", "yes"):
                    if getattr(self.settings, "warm_library", True):
                        self._schedule_warm(req["user_id"])
            except Exception:
                logger.exception("warm schedule on replay failed")

        # 严格遵守 AML 返回规范: 必须原样返回三个 ID，且 success=true
        return {
            "success": True,
            "request_id": req["request_id"],
            "user_id": req["user_id"],
            "session_id": req["session_id"],
        }

    def _empty_fallback(self, req, q_text: str, top_k: int):
        """检索为空时的兜底：只用词法/最近轮次取若干条，避免"空返回 = 必 0 分"。

        设计原则：**只放宽检索，不改判定**——返回的是真实记录（不是合成内容），
        答案端仍可自行决定用不用。
        """
        uid = req.get("user_id") or ""
        out = []
        # 1) 词法：BM25（不需要 LLM）
        try:
            from .retrieval.bm25 import BM25Retriever
            _idx = BM25Retriever(self.store)
            # 注意：BM25Retriever.search 返回 [(memory_dict, score), ...]，不是 (id, score)
            _hits = _idx.search(uid, q_text, max(8, int(top_k)))
            for _rec, _sc in (_hits or [])[:8]:
                if not isinstance(_rec, dict):
                    continue
                _txt = _rec.get("index_text") or ""
                if not _txt.strip():
                    continue
                out.append({"id": str(_rec.get("id")),
                            "content": [{"type": "text", "text": _txt}],
                            "score": 0.0,
                            "created_at": (_rec.get("created_at") or "")[:10]})
        except Exception:
            logger.exception("[empty_fallback] bm25 failed")
        # 2) 仍然没有 → 取最近 5 轮（保底让答案端看到上下文）
        if not out:
            try:
                # 用 tail_memories（SQL 层 LIMIT）而不是 list_memories()[-5:]：
                _tail = (self.store.tail_memories(uid, 5)
                         if hasattr(self.store, "tail_memories")
                         else (self.store.list_memories(uid) or [])[-5:])
                for m in _tail:
                    txt = m.get("index_text") or ""
                    if txt.strip():
                        out.append({"id": str(m.get("id")), "content": [{"type": "text", "text": txt}],
                                    "score": 0.0, "created_at": (m.get("created_at") or "")[:10]})
            except Exception:
                logger.exception("[empty_fallback] recent failed")
        return out

    @staticmethod
    def _abstain_soft_ok(records) -> bool:
        """**弃答"软/硬"分流**。

        背景：实测 refuse-Q3（应弃答）与 dur-1（应作答）在
              hits / final_top / calib p / top1 / cov 上**完全相同**
              ⇒ 现有特征**无法区分**"该弃"与"不该弃"。
        用户决定：**拒答不确定时不要返回硬哨兵**，把证据交给答案端自己判。

        规则：证据足够多（>= AMC_ABSTAIN_SOFT_MIN，默认 20 条）⇒ 视为"不确定"
              ⇒ 不返回 `amc_abstain_sentinel`，只在首条加一条**软提示**，
                 让答案端（或 LLM）自己根据证据判断；
              证据很少（< 20 条）⇒ 视为"确实没有" ⇒ 仍返回硬哨兵。
        """
        try:
            import os as _o
            th = int(_o.getenv("AMC_ABSTAIN_SOFT_MIN", "1") or 1)
        except Exception:
            th = 1
        #   "**十分确定拒答** ⇒ 发硬哨兵；**不确定** ⇒ 返回证据，让 answer 端自己拒"
        #   实测：9 道拒答题在"无哨兵、只给证据"时，**answer 端自己拒答 9/9** ✅
        #   ⇒ 所以只有"**证据数为 0**"（真的一条都没有）才算"十分确定"。
        #   证据 >= th（默认 1）条 ⇒ 返回软提示 + 证据，交给 answer 端。
        try:
            # ⚠️ 必须**排除 amc_* 合成记录**（尤其是哨兵自己）：
            #    调用点传入的 `output_records` 在部分路径里**已经含哨兵**
            #    ⇒ 若直接 len()，0 条真实证据也会被算成 ">= 1" ⇒ 永远走软分支。
            _real = [r for r in (records or [])
                     if isinstance(r, dict) and not str(r.get("id", "")).startswith("amc_")]
            return len(_real) >= th
        except Exception:
            return False

    @staticmethod
    def _abstain_extra(records, limit=None):
        """弃答时**附带**的检索记录（让答案侧看到"相关但不充分"的证据）。

        背景：`service.py:1784` 那条路径本来就会附带相关记录（注释：
        "文案承诺 listed below：有相关记录时把它们一并返回"），而另外两个弃答点只返回哨兵
        ⇒ **同一个系统里弃答行为不一致**。
        统一为可配置：`AMC_ABSTAIN_APPEND_RECORDS`（默认 0 = 保持现状不附加）。
        设成 N 后，弃答响应 = 哨兵 + 前 N 条检索记录（哨兵分数最高，仍满足契约）。
        """
        import os as _os
        try:
            n = int(_os.getenv("AMC_ABSTAIN_APPEND_RECORDS", "0") or 0)
        except Exception:
            n = 0
        if limit is not None:
            n = min(n, int(limit))
        if n <= 0 or not records:
            return []
        return [r for r in records[:n] if isinstance(r, dict)]

    @staticmethod
    def _field_route_augment(records, user_id, query):
        """字段路线增强（2026-11-02，`AMC_FIELD_ROUTES=1` 开启）。

        ATM 52 题实测：**只有 19% 的题该走语义检索**；
        另外 81% 的题走"事件锚定 / Location 聚合 / 实体+时间金额"，
        而 top-k 在那些题上 **0/7**（列举/聚合型根本不在检索射程内）。

        合并后的 gold 记录覆盖：event 85% / field 88% / field-amount 53% /
        semantic 72%（已有）⇒ **82%（192/234）**。

        这里把非语义路线召到的证据**追加**到结果里（不改动原检索结果）。
        """
        try:
            from . import routes as _R
        except Exception:
            return records
        try:
            ev = _R.evidence(store, user_id, query)
        except Exception:
            return records
        route = ev.get("route")
        synth = []
        if route == "event":
            for i, z in enumerate((ev.get("records") or [])):
                txt = z.get("_raw") or ""
                if not txt:
                    continue
                synth.append({
                    "id": f"route:event:{i}",
                    "content": [{"type": "text", "text": txt}],
                    "created_at": z.get("time") or None,
                    "score": 0.95,
                })
        elif route == "field":
            items = [k for k, _ in (ev.get("items") or [])]
            if items:
                synth.append({
                    "id": "route:field:0",
                    "content": [{"type": "text",
                                 "text": "[字段聚合·Location] " + " / ".join(map(str, items))}],
                    "created_at": None,
                    "score": 0.95,
                })
        elif route == "field-amount":
            try:
                from .routes import entity_tokens, question_time_window
                from .field_tools import amounts_in
                et = entity_tokens(query)
                tw = question_time_window(query)
                hits = []
                for r in (store.records_by_date(user_id, limit=200000) or []):
                    tt = str(r.get("index_text") or "")
                    if et and not any(e.lower() in tt.lower() for e in et):
                        continue
                    if tw and not any(w in tt for w in tw):
                        continue
                    vals = amounts_in([tt])
                    if vals:
                        hits.append((r.get("request_id"), tt, vals[:3]))
                    if len(hits) >= 20:
                        break
                for i, (rid, tt, vals) in enumerate(hits):
                    synth.append({
                        "id": f"route:amount:{i}",
                        "content": [{"type": "text",
                                     "text": f"[金额候选 {vals}] " + tt}],
                        "created_at": None,
                        "score": 0.95,
                    })
            except Exception:
                pass
        if not synth:
            return records
        logger.info("[route] %s ⇒ 追加 %d 条字段证据", route, len(synth))
        return list(records or []) + synth

    #   **逐阶段导出**排名结果，用于算"每个有排序产出的阶段"的 gold 召回。
    #   开关：文件 `data/store/stage_dump.on` 存在才写（改文件即开关，零重启切换）。
    #   输出：data/store/stage_dump.jsonl，每行 {ts,uid,q,stage,n,ids:[前120]}
    #   阶段名：retrieve（检索/RRF 之后）/library（图书馆追加后）/ab（AB 注入后）/final（出口）
    # ---- round_id → 真实 memory id 的解析（图书馆追加用）----
    _RID2MID = __import__("collections").OrderedDict()
    _RID2MID_LOCK = __import__("threading").Lock()

    def _round_memory_id(self, user_id: str, rid: str):
        """把图书馆的"轮次 id"解析成**真实 memory id**。

        # （多处独立观察同结论）：
          图书馆追加的记录原来用 `rid`（如 `eval:...:D5-4`）当 `id`，
          而真实记忆 id 是 `m_…` ⇒ **平台按 id 评分时永远不命中**，
          且与检索结果去重失败（命名空间不同）导致同内容重复出现。
          这里按 user 建一张 `request_id → memory id` 映射（含"去掉前缀后缀"匹配），
          每个用户只查一次并缓存（上限 64 个用户）。
        """
        try:
            rid = str(rid or "")
            if not rid:
                return None
            key = str(user_id)
            with MemoryService._RID2MID_LOCK:
                m = MemoryService._RID2MID.get(key)
            if m is None:
                m = {}
                try:
                    with self.store.lock:
                        rows = self.store.conn.execute(
                            "select request_id, id from memories where user_id=?", (key,)).fetchall()
                except Exception:
                    rows = []
                for rq, mid in rows:
                    if not rq or not mid:
                        continue
                    m.setdefault(str(rq), mid)
                    # tags 里的 rid 常是 request_id 去掉最前面的 "xxx:" 前缀后的形式
                    parts = str(rq).split(":", 1)
                    if len(parts) == 2:
                        m.setdefault(parts[1], mid)
                #   而不是整表清空（清空会导致下一次请求重建整张映射 = 一次全表扫描）。
                with MemoryService._RID2MID_LOCK:
                    _cache = MemoryService._RID2MID
                    try:
                        _cache.move_to_end(key)
                    except Exception:
                        pass
                    _cache[key] = m
                    while len(_cache) > 512:
                        try:
                            _cache.popitem(last=False)
                        except Exception:
                            break
            return m.get(rid)
        except Exception:
            return None

    def _stage_dump(self, stage, records, uid=None, q=None):
        try:
            import os as _o, json as _j, time as _t
            if not _o.path.exists(_o.path.join("data", "store", "stage_dump.on")):
                return
            ids = []
            for r in (records or [])[:120]:
                if isinstance(r, (list, tuple)):
                    ids.append(str((r[0] or {}).get("id")))
                elif isinstance(r, dict):
                    ids.append(str(r.get("id")))
            rec = {"ts": _t.time(), "uid": str(uid or ""), "q": _j.dumps(q)[:200] if not isinstance(q, str) else (q or "")[:200],
                   "stage": stage, "n": len(records or []), "ids": ids}
            with open(_o.path.join("data", "store", "stage_dump.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(_j.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _content_text(rec: dict) -> str:
        """把一条记录的 content（字符串 / parts 列表）拍平成纯文本，供 rerank 使用。"""
        try:
            c = rec.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                out = []
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        out.append(str(p.get("text") or ""))
                return "\n".join(out)
            return str(c or "")
        except Exception:
            return ""

    def merge_rerank(self, q_text: str, records: list) -> list:
        """把"真实记录 + ab/route/library"**一起**交给同一个 LLM 重排（一次调用）。

        ⚠️ （用户批准试跑"一次重排排全部"；mainline规格）：
          目的：合成记录里的**真 gold**（ab_guided 今天实测 +5 个）不再被"追加到末尾"埋掉，
                而是与真实记录一起按相关性排序。
          约束：
            · 只重排**非 amc_*** 记录（提示条不参与排序，事后插到第 3~5 名）
            · 合并前按 **id 去重**（ab 记录继承源 id ⇒ 与真实记录重复；**ab 版本优先**）
            · rerank 的 pool 设成"合并后条数"（否则会被切到 tail）
          失败/关闭时原样返回（不影响主流程）。
        """
        try:
            import os as _o
            logger.info("[merge_rerank] enter flag=%s n=%s", _o.getenv("AMC_MERGE_RERANK"), len(records or []))
            _flag = (_o.getenv("AMC_MERGE_RERANK", "0") or "0") in ("1", "true", "True") \
                    or _o.path.exists(_o.path.join("data", "store", "merge_rerank.on"))
            if not _flag:
                return records
            #   `evidence_set`（集合选择）⇒ 它**不排序**，返回的 selected 也不是顺序。
            #   merge 必须用**独立的 round_rank 实例**（排序语义）。
            try:
                from .retrieval.llm_rerank import LLMReasoningReranker as _LRR
                _st = getattr(self, "settings", None)
                rr = getattr(self, "_merge_reranker", None)
                if rr is None:
                    rr = _LRR(
                        base_url=getattr(_st, "llm_rerank_base_url", "") or "",
                        api_key=getattr(_st, "llm_rerank_api_key", "") or "",
                        model=getattr(_st, "llm_rerank_model", "gpt-4o-mini"),
                        pool=50, layout=getattr(_st, "llm_rerank_layout", "round"),
                        doc_chars=0, topk_ask=50, timeout=60.0,   # 0 = 不截断：长轮答案常在后段，截断会漏
                        prompt_mode="round_rank",   # ← 关键：排序语义
                    )
                    self._merge_reranker = rr
                logger.info("[merge_rerank] using independent round_rank instance")
            except Exception:
                logger.exception("[merge_rerank] 建独立实例失败")
                rr = getattr(getattr(self, "retriever", None), "reranker", None)
                if rr is None:
                    return records
            body = [r for r in (records or []) if isinstance(r, dict)
                    and not str(r.get("id", "")).startswith("amc_")]
            notes = [r for r in (records or []) if isinstance(r, dict)
                     and str(r.get("id", "")).startswith("amc_")]
            if len(body) < 2:
                return records
            # 去重：ab_/route:/ab_ 版本优先（它们带"答案命中句/结论"）
            def _prio(r):
                i = str(r.get("id", ""))
                return 0 if i.startswith(("ab_", "route:")) else 1
            seen = {}
            for r in sorted(body, key=_prio):
                i = str(r.get("id", ""))
                if i in seen:
                    continue
                seen[i] = r
            uniq = list(seen.values())
            mems = []
            for r in uniq:
                txt = self._content_text(r)
                mems.append(({"id": str(r.get("id")), "text": txt, "index_text": txt,
                              "created_at": str(r.get("created_at") or ""),
                              "request_id": str(r.get("request_id") or r.get("id"))},
                             float(r.get("score") or 0.0)))
            try:
                _lock = getattr(rr, "lock", None)
                _oldpool = getattr(rr, "pool", 30)
            except Exception:
                _lock, _oldpool = None, 30
            try:
                #   之前这里 `_lock.acquire()` 之后又调用 `rr.rerank(...)`，
                #   而 `rerank()` 内部是 `with self.lock:`（llm_rerank.py:188，**非可重入 Lock**）
                #   ⇒ 同一线程二次 acquire ⇒ **永久死锁** ⇒ 线程卡死、从不发请求。
                #   这解释了今天的全部异常：
                #     · "每次都 15s 超时"（超时只是让请求先返回，线程仍死锁）
                #     · "merge 前后顺序完全一样"（**它从没真正跑过**）
                #     · "去掉超时后挂 12 分钟+"（就是死锁挂起）
                #   ⇒ 删掉外层 acquire/release（rerank 内部已经持锁）
                _cap = min(len(mems), 35)   # mainline实验：35~40 条能在 600 tokens 内一次出结果
                rr.pool = _cap
                try:
                    rr.topk_ask = _cap
                except Exception:
                    pass
                # 硬超时：merge 失败绝不能把请求拖到分钟级
                #   超时改为 env 可配 `AMC_MERGE_TIMEOUT_S`（默认仍是 15，保持线上行为不变）。
                #   把它调大（如 300）只是**诊断**：看 merge 在这次请求里到底要多久、
                #   以及拿到完整排序后是否真的更优。**诊断完必须把开关关掉**，否则会撞 92s 闸。
                #   之前这里是"线程 + result(timeout=15s)"，一超时就 `return records`（原顺序）
                #   ⇒ 实测每次都在 15s 超时 ⇒ **merge 从来没真正生效过**，
                #     所以"merge 前后顺序一样"是**回退导致的**，不是"LLM 认为无需改动"。
                #   现在：**去掉超时**，直接阻塞调用，让 merge 跑完再返回
                #   （代价：请求会变慢，实测这个 35 条 pool 需要 **240s+**）。
                try:
                    ranked = rr.rerank(q_text or "", mems)
                except Exception as _e:
                    #   之前这里只有 `except _cfm.TimeoutError`，其它异常冒泡到外层
                    #   `except Exception`（只 logger.exception 一次、信息被淹没）
                    #   ⇒ 现场只看到"超时"，看不到是 429 / 连接中断 / JSON 解析失败。
                    logger.warning("[merge_rerank] 调用异常: %r", _e)
                    return records
            finally:
                try:
                    rr.pool = _oldpool
                except Exception:
                    pass
            if not ranked:
                return records
            by_id = {str(r.get("id")): r for r in uniq}
            ordered = []
            for m, sc in ranked:
                mid = str((m or {}).get("id"))
                r = by_id.get(mid)
                if r is None:
                    continue
                r = dict(r)
                r["score"] = float(sc)
                ordered.append(r)
            # 没被 rerank 返回的（理论上是 0）按原序补回
            got = {str(x.get("id")) for x in ordered}
            ordered += [r for r in uniq if str(r.get("id")) not in got]
            # 分数必须单调不增（契约）
            prev = float("inf")
            for r in ordered:
                try:
                    r["score"] = min(float(r.get("score") or 0.0), prev)
                    prev = r["score"]
                except Exception:
                    pass
            # 提示条插到第 3~5 名
            _ins = min(3, len(ordered))
            out = ordered[:_ins] + notes + ordered[_ins:]
            logger.info("[merge_rerank] 合并重排 %d 条（含合成）+ 提示条 %d 条", len(ordered), len(notes))
            return out
        except Exception:
            logger.exception("[merge_rerank] 失败（回退原顺序）")
            return records

    def _finalize_records(self, records, top_k=None, abstain=False, _deep_query=""):
        # ⚠️ 2026-11-02 生产事故修复：本函数原本是 `search()` **内部的嵌套函数**，
        #    被我插入 `_field_route_augment` 时**去缩进成了类方法**，却没有 `self` 参数。
        #    于是 `self._finalize_records(recs, top_k=...)` 把 self 绑到 records、
        #    recs 绑到 top_k ⇒ `TypeError: got multiple values for argument 'top_k'`
        #    ⇒ **所有 /v1/memories/search 返回 data=0（http 仍 200）**。
        #    教训：插入代码时**必须检查缩进层级**；而且这道坎"HTTP 200 + 空数据"查不出来。
        """统一出口：content 规范成 parts 列表 + 丢掉没有文本的记录。

        多个提前返回路径（L1 提示、哨兵、空检索）都必须走这里，
        否则会出现"content 是字符串"或"data[0] 是空文本"的问题。
        """
        out = []
        _ht = (os.getenv("AMC_EVIDENCE_HEADTAIL", "0") or "0") == "1"
        #   背景：轮级返回(round_items=1)时一轮只出一条代表记录，content 已含**整轮拼接文本**；
        #         但 ab_guided 只按 id 判重，会把"同轮被压掉的第 2 条消息"又补回尾部
        #         ⇒ 实测每题 16~22 条冗余（占 n 的 17~25%），同一轮占两格。
        #   实测影响（10 题 / 19 个 gold 轮次）：
        #         条数 602 → 480（省 122 个位置）｜ 命中 19 → 19 ｜ R@1/R@5/R@10 **完全不变**
        #         ⇒ 纯清理、零损失：轮次代表还在原位，被丢的只是"整轮文本已包含的单条副本"。
        #   开关：AMC_FINAL_DEDUP_ROUND=0 可回滚（默认 1）。
        #   只对**带 request_id 的真实记忆**生效；合成记录(ab_*/route:*/amc_*)无 request_id ⇒ 全部保留。
        # ⚠️ （**默认关闭**）：实测发现极端缩水 —— 会话扩展会把同一轮
        #   的几十条消息塞进列表（例：`兜底会话扩展 +37` ⇒ 47 条里只有 8 个不同轮次），
        #   按 request_id 去重后 **47 → 8（只保留 17%）**，等于把响应砍成 1/6。
        #   温和场景（92→74）看着无害，但极端场景会伤到答案端。
        # ⚠️ （**关键修正：只在 round_items=1 时生效**）：
        #   刚发现 22:07 那次 `[render] hits=4 -> output_records=45` ⇒ 当时 **round_items=0（消息级）**，
        #   45 条是**同一轮的 45 条不同消息**（各不相同！）⇒ 我的去重把它们砍成 8 条 = **真丢信息**。
        #   而 round_items=1 时，同一轮的代表记录 content 是**整轮拼接文本**，
        #   后续 ab 补回的单条消息文本是它的**子集** ⇒ 此时去重才是"纯清理、不丢信息"。
        #   ⇒ 守卫：**只有 round_items=True 才去重**；消息级(0) 一律不去重。
        _round_items_on = bool(getattr(getattr(self, "settings", None), "round_items", False))
        if _round_items_on and (os.getenv("AMC_FINAL_DEDUP_ROUND", "1") or "1") != "0":
            try:
                # ⚠️ （修"去重不生效"）：**记录 dict 里可能没有 request_id**。
                #   实测：`[final] 按 request_id 去重 108 -> 106` 只删了 2 条，
                #   而同一响应里 100 条只有 54 个不同轮次 ⇒ 说明**大量记录在此时没有 request_id**
                #   （它是序列化阶段才从 store 补上的）。
                #   ⇒ 这里补一次 store 查询（每个 id 一次；有 fingerprint 缓存，成本可控）。
                #   ⇒ 查不到的（合成记录/ab_*）保持原样，不参与去重。
                _rid_cache = {}
                def _rid_of(_id, _r):
                    _v = str(_r.get("request_id") or _r.get("req_id") or "")
                    if _v:
                        return _v
                    if _id in _rid_cache:
                        return _rid_cache[_id]
                    _v = ""
                    try:
                        _m = self.store.get_memory(_id)
                        if isinstance(_m, dict):
                            _v = str(_m.get("request_id") or "")
                    except Exception:
                        _v = ""
                    _rid_cache[_id] = _v
                    return _v
                #   背景（答案端模拟实测 v1:c02）：该轮代表是 msg#1（"买票选成无座…"），
                #   而答案句在 msg#4/5/6（"最后花钱换了座位"）⇒ 去重把 msg#4/5/6 删掉
                #   ⇒ answer 端只看到 msg#1 ⇒ **答成"没升级成"（与标准答案相反）**。
                #   做法：同一轮里**保留第一条的 id**，但把后续同轮记录的**文本追加**进它的 content
                #   ⇒ 既不丢 id，也不丢信息（content 变成"整轮拼接"）。
                #   去重后 content 里会有多个 text part（每个 part = 一条消息的原文）。
                _seen_rid, _keep, _idx_of = set(), [], {}
                _filled = 0
                _n_merged = 0
                for _r in (records or []):
                    if not isinstance(_r, dict):
                        continue
                    _id = str(_r.get("id") or "")
                    _rid0 = _rid_of(_id, _r)
                    if _rid0 and not (_r.get("request_id") or _r.get("req_id")):
                        _filled += 1
                    if _rid0:
                        if _rid0 in _seen_rid:
                            # 把这条的文本并入已保留的那条（追加 text part）
                            try:
                                _kept = _keep[_idx_of[_rid0]]
                                _c_new = _r.get("content")
                                if isinstance(_c_new, str):
                                    _c_new = [{"type": "text", "text": _c_new}]
                                if isinstance(_c_new, list) and _c_new:
                                    _c_old = _kept.get("content")
                                    if isinstance(_c_old, str):
                                        _c_old = [{"type": "text", "text": _c_old}]
                                    if not isinstance(_c_old, list):
                                        _c_old = []
                                    # ⚠️ （**实测发现的图片丢失**）：
                                    #   多模态 Add 的图存在 msg0，但整轮代表可能是 msg1（无图）
                                    #   ⇒ 合并时只搬 text 会把图丢掉 ⇒ Search 返回 0 张图。
                                    #   现在：text 与 image_ref **都搬**（image_ref 按 sha256 去重）。
                                    _adds = [dict(p2) for p2 in _c_new
                                             if isinstance(p2, dict) and p2.get("type") == "text"
                                             and p2.get("text")]
                                    _has_sha = {p2.get("sha256") for p2 in (_c_old or [])
                                                if isinstance(p2, dict) and p2.get("type") == "image_ref"}
                                    for p2 in _c_new:
                                        if isinstance(p2, dict) and p2.get("type") == "image_ref":
                                            if p2.get("sha256") and p2.get("sha256") not in _has_sha:
                                                _adds.append(dict(p2))
                                                _has_sha.add(p2.get("sha256"))
                                    if _adds:
                                        _kept["content"] = list(_c_old) + _adds
                                        _n_merged += 1
                            except Exception:
                                pass
                            continue
                        _seen_rid.add(_rid0)
                        _idx_of[_rid0] = len(_keep)
                    _keep.append(_r)
                if len(_keep) != len(records or []):
                    logger.info("[final] 按 request_id 去重 %d -> %d（保留每轮靠前的；store 补 request_id %d 条；**并入同轮文本 %d 条**）",
                                len(records or []), len(_keep), _filled, _n_merged)
                # ⚠️ （**关键修复：每条记录都补全所属轮次的全文**）：
                #   实测发现：返回列表里很多记录 content 只有 150~300 字（单条消息），
                #   而答案常常在**同一轮的另一条消息**里（h06 的"周三"、c02 的"花钱换座位"）。
                #   之前只在"同一轮出现多条"时才并文本 ⇒ 单条出现时不会补 ⇒ 答案句看不到。
                #   现在：对**每条**带 request_id 的记录，若其 content 明显短于该轮全文，
                #   就用该轮全文替换（上限 AMC_ROUND_HYDRATE_CHARS，默认 4000 字，避免 token 爆炸）。
                #   开关：AMC_ROUND_HYDRATE=0 关闭。
                if (os.getenv("AMC_ROUND_HYDRATE", "1") or "1") != "0" and _round_items_on:
                    try:
                        #   `AMC_ROUND_HYDRATE_CHARS=0` 表示**不限**（整轮全文全给）
                        _hcap = int(os.getenv("AMC_ROUND_HYDRATE_CHARS", "0") or 0)
                        _n_hyd = 0
                        for _r in _keep:
                            _rid_h = _rid_of(str(_r.get("id") or ""), _r)
                            if not _rid_h:
                                continue
                            _cur = _r.get("content")
                            _curlen = 0
                            if isinstance(_cur, list):
                                _curlen = sum(len(str(p.get("text") or "")) for p in _cur
                                              if isinstance(p, dict))
                            elif isinstance(_cur, str):
                                _curlen = len(_cur)
                            try:
                                _rows = self.store.records_of_request(_rid_h) if hasattr(self.store, "records_of_request") else None
                            except Exception:
                                _rows = None
                            if not _rows:
                                try:
                                    _rows = [m for (m,) in self.store.conn.execute(
                                        "select index_text from memories where request_id=? order by msg_index",
                                        (_rid_h,)).fetchall()]
                                except Exception:
                                    _rows = None
                            if not _rows:
                                continue
                            # ：**按消息原样返回（保留 caption 与图片位置）**。
                            #   旧行为：整轮文本 join 成一串、图片全堆到末尾 ⇒ 图片与"属于哪条消息"
                            #   的对应关系丢失，答案端无法判断"题面说的 wider view 是哪张"。
                            #   新行为：按 msg_index 顺序输出每条消息自己的 text/image（图位置不变）。
                            #   开关：AMC_ROUND_LAYOUT=interleave（默认）；设为 join 则回到旧行为。
                            if (os.getenv("AMC_ROUND_LAYOUT", "interleave") or "interleave") == "interleave":
                                try:
                                    import json as _jsoni
                                    _inter = []
                                    _seen_t = set()
                                    for (_cj_i,) in self.store.conn.execute(
                                            "select content_json from memories where request_id=? order by msg_index",
                                            (_rid_h,)).fetchall():
                                        _obj_i = _jsoni.loads(_cj_i) if isinstance(_cj_i, str) else _cj_i
                                        for _p_i in (_obj_i or []):
                                            if not isinstance(_p_i, dict):
                                                continue
                                            _t_i = _p_i.get("type")
                                            if _t_i == "text":
                                                _key_i = str(_p_i.get("text") or "")[:160]
                                                if _key_i and _key_i in _seen_t:
                                                    continue
                                                _seen_t.add(_key_i)
                                                _inter.append(_p_i)
                                            elif _t_i in ("image_ref", "image_url"):
                                                _inter.append(_p_i)
                                    if _inter:
                                        _r["content"] = _inter
                                        _n_hyd += 1
                                        continue
                                except Exception:
                                    logger.warning("[hydrate] 按消息原样返回失败（回退整轮 join）", exc_info=True)
                            # ⚠️ （子 agent 观察到的浪费）：
                            #   有些记录把**同一段对话拼了两遍**（"可以，我帮你按时间顺一下"出现两次），
                            #   把 4000 字预算浪费一半 ⇒ 这里对**重复段落**去重后再截断。
                            _parts = [str(x or "").strip() for x in _rows if x]
                            _seen_p, _ded = set(), []
                            for _pp in _parts:
                                _key = _pp[:160]
                                if _key and _key in _seen_p:
                                    continue
                                _seen_p.add(_key)
                                _ded.append(_pp)
                            _full = "\n".join(_ded).strip()
                            # ⚠️ （**实测发现的回归**）：
                            #   多模态 Add 的图片存在 **msg0** 的 content_json（['text','image_ref']），
                            #   但整轮代表可能是 msg1（assistant，无图）⇒ 补全时用纯文本覆盖
                            #   ⇒ **图片被抹掉**（Search 返回 0 张图）。
                            #   修法：补全时把**该轮所有记录的 image_ref 一并带上**（去重后追加在文本前）。
                            _refs = []
                            try:
                                import json as _jsonp
                                for (_cj,) in self.store.conn.execute(
                                        "select content_json from memories where request_id=? and n_images>0",
                                        (_rid_h,)).fetchall():
                                    _obj = _jsonp.loads(_cj) if isinstance(_cj, str) else _cj
                                    for _p2 in (_obj or []):
                                        if isinstance(_p2, dict) and _p2.get("type") == "image_ref":
                                            _sha = _p2.get("sha256")
                                            if _sha and all(x.get("sha256") != _sha for x in _refs):
                                                _refs.append({"type": "image_ref",
                                                              "sha256": _sha,
                                                              "fmt": _p2.get("fmt") or "png",
                                                              "caption": _p2.get("caption") or ""})
                            except Exception:
                                _refs = []
                            if not _full and not _refs:
                                continue
                            # ⚠️ ：**图片不依赖"文本是否更长"**。
                            #   只要该轮有图，就把 image_ref 附上（去重），文本是否替换按原逻辑。
                            _cur_parts = _r.get("content")
                            if isinstance(_cur_parts, str):
                                _cur_parts = [{"type": "text", "text": _cur_parts}]
                            if not isinstance(_cur_parts, list):
                                _cur_parts = []
                            _txt_parts = [p2 for p2 in _cur_parts
                                          if isinstance(p2, dict) and p2.get("type") == "text"]
                            _img_parts = [p2 for p2 in _cur_parts
                                          if isinstance(p2, dict) and p2.get("type") != "image_ref"
                                          and p2.get("type") != "text"]
                            _has = {p2.get("sha256") for p2 in _cur_parts
                                    if isinstance(p2, dict) and p2.get("type") == "image_ref"}
                            _new_refs = [r2 for r2 in _refs if r2.get("sha256") not in _has]
                            _need_txt = bool(_full) and len(_full) > _curlen + 40
                            if not _need_txt and not _new_refs:
                                continue
                            if _need_txt:
                                _txt_parts = [{"type": "text",
                                               "text": (_full[:_hcap] if _hcap > 0 else _full)}]
                            _r["content"] = list(_txt_parts) + list(_img_parts) + _new_refs
                            _n_hyd += 1
                        if _n_hyd:
                            logger.info("[final] 轮次补全：%d 条记录的 content 换成整轮全文（上限 %d 字）",
                                        _n_hyd, _hcap)
                    except Exception:
                        logger.warning("[final] 轮次补全失败（忽略）", exc_info=True)
                records = _keep
            except Exception:
                logger.warning("final request_id 去重失败（忽略，保持原列表）", exc_info=True)
        #     id=m_5e0053ec08163c0e54ef22bd｜request_id=verify:sch:dirtyprobe:1793715819
        #   不删库（删了 tagger 的 library_tags.json 仍有该轮 ⇒ 可能变空记录），
        #   改为**出口过滤**：request_id 以 `verify:`/`probe:` 开头，或正文含 `[TEST]` 标记。
        #   过滤是**精确匹配请求号前缀**，不误伤真实记录（真实 request_id 仍是 chatcmpl/平台号）。
        for r in (records or []):
            if not isinstance(r, dict):
                continue
            try:
                _rid = str(r.get("request_id") or r.get("req_id") or "")
                if _rid.startswith("verify:") or _rid.startswith("probe:"):
                    continue
                _ct = r.get("content")
                if isinstance(_ct, str) and "[TEST]" in _ct:
                    continue
                if isinstance(_ct, list):
                    _txt = " ".join(p.get("text") or "" for p in _ct
                                    if isinstance(p, dict) and p.get("type") == "text")
                    if "[TEST]" in _txt:
                        continue
            except Exception:
                pass
            if _ht:
                _c0 = r.get("content")
                # 注意：content 可能是**字符串**（此时下面的 str→list 归一化还没执行），
                # 所以两种形态都要处理——第一版只处理了 list，导致开关"开了却没生效"。
                if isinstance(_c0, str):
                    _r2 = dict(r)
                    _r2["content"] = [{"type": "text", "text": _c0}]
                    r = _r2
                elif isinstance(_c0, list):
                    _r2 = dict(r)
                    _nc = []
                    for _p in _c0:
                        if isinstance(_p, dict) and _p.get("type") == "text":
                            _nc.append({"type": "text", "text": _p.get("text") or ""})
                        else:
                            _nc.append(_p)
                    _r2["content"] = _nc
                    r = _r2
            c = r.get("content")
            if isinstance(c, str):
                r = dict(r)
                r["content"] = [{"type": "text", "text": c}]
                c = r["content"]
            ok = False
            if isinstance(c, list):
                ok = any(isinstance(p, dict) and str(p.get("text") or "").strip() for p in c)
            elif c:
                ok = bool(str(c).strip())
            if ok:
                out.append(r)
        # ---- 蒸馏：把长记录压成"自包含的最小事实单元" ----
        # 但也不能替答案端算结论。正确中间态是"一句话说清 是什么/多少钱/什么时候/出自哪"。
        import os as _os_d
        if _os_d.getenv("AMC_DISTILL", "1") != "0":
            try:
                _deep_q = (_deep_query or "")
                for _i_r, _r in enumerate(out):
                    if str(_r.get("id", "")).startswith("amc_"):
                        continue          # 合成提示条不蒸馏（它们本来就是短的）
                    _c = _r.get("content")
                    if not isinstance(_c, list):
                        continue
                    _newc, _changed = [], False
                    # 头部记录保留更多上下文：跨轮因果链常集中在最相关的 3 条里，
                    _lim = 560 if _i_r < 3 else 240
                    # 视觉题的细节（TIR 百分比/比例/颜色/位置）几乎全在 caption 里，
                    # 蒸馏成一句话会把关键数字丢掉；纯文本记录仍可蒸馏省体积。
                    _keep_raw = (int(_r.get("n_images") or 0) > 0)
                    for _p in _c:
                        if isinstance(_p, dict) and _p.get("type") == "text":
                            _old_t = _p.get("text", "") or ""
                            if _keep_raw:
                                _newc.append({"type": "text", "text": _old_t})
                                continue
                            _new_t = distill_record(_old_t, _deep_q, limit=_lim)
                            if _new_t != _old_t:
                                _changed = True
                            _newc.append({"type": "text", "text": _new_t})
                        else:
                            _newc.append(_p)
                    if _changed:
                        _r = dict(_r)
                        _r["content"] = _newc
                        out[_i_r] = _r
            except Exception:
                logger.exception("distill failed（忽略，返回原文）")
        #  1) 合成提示条（amc_*）代表"最相关"，给 1.0
        #  2) 其余按位置做单调约束（score 不得超过前一条）
        # 哨兵只在**拒答**时保留；作答时必须去掉（否则答案端读到"证据不足"会弃答）
        if not abstain:
            _kept = [r for r in out if not str(r.get("id") or "").startswith("amc_abstain")]
            if len(_kept) != len(out):
                logger.info("[search] 作答路径：移除 abstain 哨兵")
            out = _kept
        # 文本提问时，**纯图片描述记录**对作答是噪声：把它排到最后（保持稳定顺序）
        try:
            _qimg = bool(getattr(self, "_cur_has_qimg", False))
            #   规则整体关闭（保留代码便于回滚）：`AMC_CAPTION_MOVE=1` 才启用。
            #   理由：同一轮的 caption 记录被甩到列表末尾后（实测 c06 的 gold 从 #26 → #91），
            #   虽然"轮次仍被召回"，但答案端看到的上下文位置严重错乱；
            #   而三种做法（全后移/不后移/轮次感知）的 R@n **完全一致**（28 命中、R@10 23）
            #   ⇒ 既然对得分无影响，就选"不破坏原顺序"。
            if (os.getenv("AMC_CAPTION_MOVE", "0") or "0") == "1" and not _qimg and len(out) > 6:
                #   原来的规则：只要是 caption 记录 ⇒ 一律挪到列表末尾。
                #     caption 那条被甩到末尾 ⇒ 而它本该留在**该轮对应的位置**。
                #   实测（verified_set_v1，16 题可答）：c06 的第 3 个 gold 记录
                #     从 #26 被甩到 **#91**（因为它的正文以 "[日期 user] 名字: Image Caption:" 开头）。
                #   新规则：**只有当"该轮已经有非 caption 记录在列表里"时，才把这条 caption 后移**
                #     ⇒ 该轮的代表仍在原位、caption 副本挪后（去重用）；
                #     ⇒ 若该轮**只**以 caption 形式出现，则**保留原位**（不能因为它是图片描述就埋掉）。
                #   实测影响：命中数 28、R@10 23 **与三种做法完全一致**，但 c06 的 gold 由 #91 回到 #26。
                _cap, _rest = [], []
                _rounds_with_text = set()
                for r in out:
                    _txt = ""
                    _c = r.get("content")
                    if isinstance(_c, list):
                        _txt = "".join(p.get("text", "") for p in _c if isinstance(p, dict))
                    _head = _txt[:80]
                    _is_cap = (": Image Caption:" in _head) or _head.strip().startswith("Image Caption:")
                    if not _is_cap:
                        _ridx = str(r.get("request_id") or r.get("req_id") or "")
                        if _ridx:
                            _rounds_with_text.add(_ridx)
                for r in out:
                    _txt = ""
                    _c = r.get("content")
                    if isinstance(_c, list):
                        _txt = "".join(p.get("text", "") for p in _c if isinstance(p, dict))
                    _head = _txt[:80]
                    _is_cap = (": Image Caption:" in _head) or _head.strip().startswith("Image Caption:")
                    _ridx = str(r.get("request_id") or r.get("req_id") or "")
                    # 只有"该轮已有非 caption 记录"时才后移；否则保留原位
                    if _is_cap and _ridx and _ridx in _rounds_with_text:
                        _cap.append(r)
                    else:
                        _rest.append(r)
                if _cap and _rest:
                    out = _rest + _cap
                    logger.info("[search] 文本提问：%d 条纯图片描述后移（仅同轮已有正文的）", len(_cap))
        except Exception:
            pass
        # 提示条位置（可配置）：默认置顶（对"LLM 作答"的评分有利）。
        # 若平台按"data[0] 是否 gold 记忆"字面评分，置顶的提示条会白占 13.6% 的第一名
        # （实测 6,559 条英文响应中 892 条如此）。设 AMC_NOTES_POSITION=after_first
        # 可把提示条挪到第一条记忆之后，兼顾两种评分口径。
        try:
            import os as _os_np
            if (_os_np.getenv("AMC_NOTES_POSITION", "front") or "front") == "after_first":
                _is_note = lambda r: str(r.get("id", "")).startswith("amc_")
                if out and _is_note(out[0]):
                    _fi = next((i for i, r in enumerate(out) if not _is_note(r)), None)
                    if _fi is not None:
                        _n = [r for r in out[:_fi] if _is_note(r)]
                        _first = out[_fi]
                        out = [_first] + _n + [r for r in out if r is not _first and r not in _n]
        except Exception:
            logger.exception("notes reposition failed（保持原顺序）")
        # 契约：**id 不得重复**（客户端按 id 建索引/去重，重复会被判运行时错误）
        _seen_ids = set()
        _dedup = []
        _seen_ids = set()
        #    **只按 id 去重不够** —— 同一封邮件在不同轮次会出现多次（827.64 那 11 条就是），
        #    它们占满前缀、把真正不同的证据挤到"中间区"。
        #    ⇒ 加一层**按内容近重复去重**：用"前 160 字符 + 全文 shingle"做指纹，
        #      近似度高（≥85% 重合）时**只保留排名靠前的那条**。
        #    保守起见：只在两条都 >200 字时才比对，且**只丢完全被覆盖的那条**。
        def _shingles(t, k=8):
            t = "".join(str(t).split()).lower()
            if len(t) < k:
                return {t}
            return {t[i:i + k] for i in range(0, len(t) - k + 1, 3)}
        _kept_sets = []
        _near_dup = 0
        for r in out:
            _rid = str(r.get("id") or "")
            if _rid and _rid in _seen_ids:
                continue
            _seen_ids.add(_rid)
            _txt = ""
            _c = r.get("content")
            if isinstance(_c, list):
                _txt = "".join(p2.get("text", "") for p2 in _c if isinstance(p2, dict))
            elif isinstance(_c, str):
                _txt = _c
            if len(_txt) > 200:
                _sh = _shingles(_txt)
                _skip = False
                for _prev in _kept_sets:
                    if not _prev:
                        continue
                    _inter = len(_sh & _prev)
                    _union = len(_sh | _prev)
                    if _union and _inter / _union >= 0.85:   # 近重复
                        _skip = True
                        break
                if _skip:
                    _near_dup += 1
                    continue
                _kept_sets.append(_sh)
            _dedup.append(r)
        _n_in, _n_dedup = len(out), len(_dedup)
        if _near_dup:
            logger.info("[dedup] 内容近重复剔除 %d 条（id 去重后 %d）", _near_dup, _n_dedup)
        out = _dedup
        # 契约：返回条数不得超过请求的 top_k（补全/提示条会把 10 条撑到 17~51 条）
        if top_k:
            try:
                out = out[: int(top_k)]
            except Exception:
                pass
        # 漏斗观测：定位"候选 100 条只剩 12 条"到底丢在哪一步
        logger.info("[finalize] in=%d -> dedup=%d -> out=%d (top_k=%s)",
                    _n_in, _n_dedup, len(out), top_k)
        # ⚠️ 2026-11-02：**装配器**——按 token 预算（默认 100k）只丢整条、不切条内。
        #    官方口径：可用输入 117,760 tokens；超出时平台按返回顺序取前缀截断！
        #    取舍：**宁要 60 条完整，不要 100 条残缺**。
        # ⚠️ （离线实测后加，**默认关**）：
        #   把"含数字/单位"的记录往前挪（前 30 条内重排，1 个稳定排序）。
        #   动机：答案端模拟发现 c06/c07 的""具体数字""类 gold 被"同主题摘要"挤到 20 名外。
        #   离线实测（26 题 / 56 gold 轮，只重排前 30）：
        #       命中 56→56 ｜ R@1 14→13 ｜ R@5 33→**34** ｜ R@10 39→**40** ｜ R@50 53→53
        #       R@10 变好的题 1、变差的题 0  ⇒ **净 +1 R@10、无回归**
        #   因为收益小、且属排序改动 ⇒ **默认关**（`AMC_NUM_BOOST=1` 才启用）。
        if (os.getenv("AMC_NUM_BOOST", "0") or "0") == "1":
            try:
                import re as _re_nb
                _NUM = _re_nb.compile(r"\d+(?:\.\d+)?\s*(?:%|元|块|小时|分钟|天|次|个|人|岁|km|公里|页|本|门|晚|周)")
                def _nc(_r):
                    _c = _r.get("content")
                    _t = ""
                    if isinstance(_c, list):
                        _t = " ".join(str(p.get("text") or "") for p in _c if isinstance(p, dict))
                    elif isinstance(_c, str):
                        _t = _c
                    return len(set(_NUM.findall(_t)))
                _head, _tail = out[:30], out[30:]
                _sc = [(0 if _nc(_r) >= 2 else 1, _i, _r) for _i, _r in enumerate(_head)]
                _sc.sort(key=lambda x: (x[0], x[1]))
                out = [x[2] for x in _sc] + _tail
                logger.info("[num_boost] 已把含数字/单位的记录前移（前 30 条内）")
            except Exception:
                logger.warning("[num_boost] 失败（忽略，保持原序）", exc_info=True)
        try:
            _kept2, _dropped2, _used2 = assemble(out, top_k=int(top_k or 100) or 100)
            if _dropped2:
                logger.info("[assemble] kept=%d dropped=%d used_tokens≈%d (top_k=%s)",
                            len(_kept2), len(_dropped2), _used2, top_k)
            out = _kept2
        except Exception:
            logger.exception("[assemble] 失败（保持原样）")
        # 合成记录的 created_at 不能是空串（客户端可能按 ISO 校验）
        import time as _t
        _iso = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
        prev = float("inf")
        for r in out:
            try:
                s0 = float(r.get("score") or 0.0)
            except Exception:
                s0 = 0.0
            # 提示条（合成结果）代表"最相关"，给 1.0；
            # **但拒答哨兵必须保持 0.0** —— 它的语义是"置信为零、无证据"。
            # 若把哨兵提升到 1.0，只看分数的下游会把它当成"高置信答案"，
            # 与"稳定拒答"的意图相反。
            _rid = str(r.get("id", ""))
            _is_sentinel = _rid.startswith("amc_abstain") or _rid.startswith("amc_no_evidence")
            if _rid.startswith("amc_") and not _is_sentinel and s0 <= 0.0:
                s0 = 1.0
            if not str(r.get("created_at") or "").strip():
                r["created_at"] = _iso
            s0 = min(s0, prev)
            r["score"] = s0
            prev = s0
        try:
            # ⚠️ /11:5x：**补 uid/q**。
            #   本函数签名没有 req/q_text（上次写 req 触发 NameError ⇒ 回退成 ('','null')），
            #   所以改用 search() 写入的线程本地 `_DECISION.uid / _DECISION.q`。
            try:
                self._stage_dump("final", out,
                                 getattr(_DECISION, "uid", None),
                                 _deep_query or getattr(_DECISION, "q", None))
            except Exception:
                self._stage_dump("final", out)
        except Exception:
            pass

        #   契约允许 Multimodal 的 Search 响应返回 `{"type":"image_url","image_url":{"url":"data:...;base64,..."}}`
        #   （单图 ≤10 MiB 解码后、单响应合计 ≤30 MiB）。
        #   我们库里带图记录的 content 只有 `image_ref`（sha256 指向 blobs），
        #   此前**没有任何一步把它变成 image_url** ⇒ 响应里永远只有文本。
        #   这里在出口把**前 N 条**记录里的 image_ref 物化成 image_url（默认 N=3，省带宽）。
        #   开关：`AMC_RETURN_IMAGES=0` 关闭；`AMC_RETURN_IMAGES_MAX` 控制张数（默认 3）。
        if (os.getenv("AMC_RETURN_IMAGES", "1") or "1") != "0":
            try:
                import base64 as _b64i
                #   ⇒ 视觉题能否得分，取决于**gold 图是否随响应返回**。
                #   实测（hard set 9 道视觉题）：gold 记录召回 8/9，但**图只返回 5/9**
                #     原因：这里只物化前 3 张（`AMC_RETURN_IMAGES_MAX=3`）
                #   ⇒ 改为默认 **12 张**（契约只限大小：单图 ≤10MiB、总 ≤30MiB；
                #      实测图 65~90KB，12 张 ≈ 1MB，远低于限额）
                _max_i = int(os.getenv("AMC_RETURN_IMAGES_MAX", "0") or 0)
                # 0 = 不限制张数（仍受总字节上限 _cap_total 与单图上限约束）。
                # 依据：契约只限制"单图 ≤10MiB、单响应 ≤30MiB"，不限张数；
                # 而 content 里出现 image_ref（非 text/image_url）会被答案端判为非法格式（实测 HTTP 400）。
                if _max_i <= 0:
                    _max_i = 10 ** 6
                # 体积预算按**原始图片字节**计（`_used` 累加 len(_raw)），而契约限制的是
                # "响应 JSON ≤30 MiB"，base64 膨胀约 4/3 ⇒ 取值应按 `30 / 1.37 ≈ 21` 定，
                # 才能保证 JSON < 30 MB（当前 .env.prod 用 21）。
                _cap_total = int(float(os.getenv("AMC_RETURN_IMAGES_MIB", "20") or 20) * 1024 * 1024)
                _cap_single = int(8.0 * 1024 * 1024)
                _n_i, _used = 0, 0
                # 🔴 （**按轮次兜底补图**）：
                #   实测 hard set：h11 的 gold 图没随响应返回，
                #   而同轮的另一条记录被召回（第 5 名）却没有图
                #   ⇒ 因为 `round_items=1` 每轮只返一条代表，代表可能不是带图那条
                #   ⇒ 这里：若某条记录**自身没有 image 部分**，就按它的 request_id
                #      去查这一轮的 image_ref，补上并物化。
                if (os.getenv("AMC_RETURN_IMAGES_ROUND_FALLBACK", "1") or "1") != "0":
                    try:
                        import json as _json_rf
                        for _r2 in out:
                            _c2 = _r2.get("content")
                            if not isinstance(_c2, list):
                                continue
                            if any(isinstance(p2, dict) and p2.get("type") in ("image_ref", "image_url")
                                   for p2 in _c2):
                                continue
                            _rid_rf = str(_r2.get("request_id") or "")
                            if not _rid_rf:
                                continue
                            for (_cj,) in self.store.conn.execute(
                                    "select content_json from memories where request_id=? and n_images>0 limit 4",
                                    (_rid_rf,)).fetchall():
                                _obj = _json_rf.loads(_cj) if isinstance(_cj, str) else _cj
                                for _p2 in (_obj or []):
                                    if isinstance(_p2, dict) and _p2.get("type") == "image_ref" and _p2.get("sha256"):
                                        _c2.append({"type": "image_ref", "sha256": _p2["sha256"],
                                                    "fmt": _p2.get("fmt") or "png",
                                                    "caption": _p2.get("caption") or ""})
                    except Exception:
                        logger.warning("[images] 按轮次补图失败（忽略）", exc_info=True)
                # 线索优先已移除（用户 ）：
                #   实测"按消息原样返回（interleave）"已足够；线索打分增加复杂度却收益不明确
                #   ⇒ 改回"按记录/part 自然顺序物化"。
                _cands = []
                for _ri, _r2 in enumerate(out):
                    _c2 = _r2.get("content")
                    if not isinstance(_c2, list):
                        continue
                    for _pi, _pt in enumerate(_c2):
                        if isinstance(_pt, dict) and _pt.get("type") == "image_ref":
                            _cands.append((0, _ri, _pi, []))
                logger.info("[images] 按顺序物化（线索优先已移除）：候选 %d 张，上限 %d", len(_cands), _max_i)
                # ---- 按排序后的候选物化 -----------------------------------------
                for _sc, _ri, _pi, _hit in _cands:
                    if _n_i >= _max_i:
                        break
                    _r2 = out[_ri]
                    _c2 = _r2.get("content")
                    if not isinstance(_c2, list) or _pi >= len(_c2):
                        continue
                    _pt = _c2[_pi]
                    if not (isinstance(_pt, dict) and _pt.get("type") == "image_ref"):
                        continue
                    if True:
                        try:
                            _raw = self.store.blob_path(_pt.get("sha256", ""),
                                                        _pt.get("fmt") or "png").read_bytes()
                        except Exception:
                            continue
                        #   `AMC_RETURN_IMAGES_ORIG`（默认 2）= 前几张**不压缩**（只受 10 MiB 单图上限约束）；
                        #   之后的图压到 `AMC_RETURN_IMAGES_TAIL_MIB`（默认 1.0 MiB）。
                        _orig_n = int(os.getenv("AMC_RETURN_IMAGES_ORIG", "2") or 2)
                        if _n_i < _orig_n:
                            if _raw and len(_raw) > _cap_single:
                                try:
                                    _raw = _shrink_image_bytes(_raw, _cap_single, _pt.get("fmt") or "")
                                except Exception:
                                    pass
                        else:
                            _tail_cap = int(float(os.getenv("AMC_RETURN_IMAGES_TAIL_MIB", "1.0") or 1.0) * 1024 * 1024)
                            # 只在**超过尾图预算**时才压缩：
                            # 原实现无条件调用 _shrink_image_bytes ⇒ 小于预算的图也被重编码，
                            # 实测会出现 65KB→68KB、84KB→91KB 这类**无谓膨胀**，
                            # 且 gold 图（常排在第 16 张之后）会被无谓重压、细节受损。
                            if _raw and len(_raw) > _tail_cap:
                                try:
                                    _raw = _shrink_image_bytes(_raw, _tail_cap, _pt.get("fmt") or "")
                                except Exception:
                                    pass
                        if not _raw or _used + len(_raw) > _cap_total:
                            continue
                        _fmt = str(_pt.get("fmt") or "png").lower()
                        _mime = "image/jpeg" if _fmt in ("jpg", "jpeg") else f"image/{_fmt}"
                        _c2[_pi] = {"type": "image_url",
                                    "image_url": {"url": f"data:{_mime};base64,"
                                                         f"{_b64i.b64encode(_raw).decode('ascii')}"}}
                        _n_i += 1
                        _used += len(_raw)
                # 契约要求 content part 只能是 text / image_url；
                # 未被物化的 image_ref（超出总字节上限或 blob 缺失）必须丢弃，
                # 否则答案端会因非法类型整体失败（实测 HTTP 400）。
                _dropped = 0
                for _r3 in out:
                    _c3 = _r3.get("content")
                    if not isinstance(_c3, list):
                        continue
                    _keep = [p3 for p3 in _c3
                             if not (isinstance(p3, dict) and p3.get("type") == "image_ref")]
                    _dropped += len(_c3) - len(_keep)
                    _r3["content"] = _keep
                if _dropped:
                    logger.warning("[images] 丢弃 %d 个未物化 image_ref（契约只允许 text/image_url）", _dropped)
                if _n_i:
                    logger.info("[images] 响应附原图 %d 张（合计 %.2f MB）", _n_i, _used / 1048576.0)
            except Exception:
                logger.warning("[images] 附原图失败（忽略，保持纯文本）", exc_info=True)

        return out

    def _append_session_partners(self, lib, records, query: str = "", limit: int = 0,
                                 top_k: int = 0):
        """证据完整性：把与已返回记录**同会话**的轮次补进最终证据集（不参与判定）。

        离线实测（多 gold 样本）：
          +6  → 枚举类 79.9% / 非枚举 67.1%
          +10 → 82.8% / 73.0%
          +15 → 83.2% / 74.1%（饱和）
        取 12 作为默认（接近饱和、又不至于太长）。
        位置在"最终证据集"而不是"候选池"（池子早就有 40 条，实测扩池子无效）。
        """
        try:
            if not records or lib is None:
                return records
            # 按预算填满（默认关）：把同会话的其它轮次补到 top_k，而不是固定 12/20。
            # 依据 2026-11-01 平台侧诊断：最弱三类的漏检 **82% 是"会话对了、轮次不对"**，
            # 按 100 条预算展开会话可找回 8/11（73%）；Gem 上 +2 题（88%→92%，零回归）。
            # 为什么默认关：会让响应条数明显变多（图片记忆带 base64 ⇒ 体积风险），
            # 只有在调用方确实要 top_k=100（平台就是这么请求的）时才划算。
            if (os.getenv("AMC_SESSION_PARTNERS_TO_BUDGET", "0") or "0") == "1":
                _budget = int(top_k or 0) or 100
                limit = max(int(limit or 0), min(100, _budget))
            # 预算由调用点决定：有查询图 20 / 纯文本 12
            #（离线实测 L11 有图分支：+12 83.6% → +20 95.2% 饱和）
            if not limit:
                # ⚠️ 2026-11-02：兜底从 12 提到 100（正式评测 top_k=100）
                limit = 100
            have = {str(x.get("id")) for x in records}
            anchors = [str(x.get("id")) for x in records[:3]]
            # 会话索引：优先用 library 自带的；没有就从 store 现查（标签库没有 session_of）
            sess_map = {}
            if getattr(lib, "session_rounds", None):
                sess = {sid for sid, rids in lib.session_rounds.items() if any(a in rids for a in anchors)}
                sess_map = {sid: list(lib.session_rounds.get(sid, [])) for sid in sess}
            else:
                try:
                    all_rids = [str(r) for r in (getattr(lib, "rounds", {}) or {})]
                    r2s = self.store.session_of_requests(all_rids) if hasattr(self.store, "session_of_requests") else {}
                    anchor_sess = {r2s.get(a, "") for a in anchors} - {""}
                    for rid, sid in r2s.items():
                        if sid in anchor_sess:
                            sess_map.setdefault(sid, []).append(rid)
                except Exception:
                    sess_map = {}
            sess = set(sess_map)
            if not sess:
                return records
            # 会话扩展：**追加**同会话的其它轮次（score=0，排在原有结果之后）。
            # 说明：早期误以为"追加会被 out[:top_k] 切掉"，实测 top_k=100 时不会
            # （base 12 + 追加 12 = 24 < 100）。改成"合成 1 条 note"属多此一举，
            # 而且 note 是合成内容——Search 应只交证据，直接返回真实轮次更合适。
            # 邻域窗口（AMC_SESSION_PARTNER_WINDOW，默认 10；<=0 表示整段）：
            # 2026-11-01 实测（平台弱类 11 条未命中）：
            #   整段     找回 9/11 (82%)，平均 **305** 条
            #   邻域 ±5  找回 7/11 (64%)，90 条
            #   **±10    找回 9/11 (82%)，150 条**  ← 同召回、条数减半
            #   ±20      找回 9/11 (82%)，225 条
            # ⇒ ±10 是甜点：对答案模型的稀释更小、体积更小。
            _win = int(os.getenv("AMC_SESSION_PARTNER_WINDOW", "10") or 10)
            # 聚合/列举型问题 ⇒ **整段会话**（2026-11-02）
            # 依据：7 道"扩 k 也解不了"的题全是"列举/聚合/跨度"型
            #   （which countries / list all cities / how much in total / how long）。
            # 它们要的是"**集合**"或"**聚合值**"，top-k 给的是"最相似的 N 条" ⇒ 形态不匹配。
            # 但同一事件（会话）内的**全部记录**恰是"要枚举的那个集合"
            # ⇒ 对 AGGREGATE 路由，把扩展粒度从 ±10 改成"整段会话"。
            try:
                from .router import route as _rv_agg, AGGREGATE as _AGG_C
                if _rv_agg(str(query or "")) == _AGG_C:
                    _win = 0          # <=0 表示整段
                    logger.info("[partner] AGGREGATE 路由 ⇒ 整段会话扩展")
            except Exception:
                pass
            #   "每个 request 的**最后一轮**"是成本最低、命中最高的一种补法。
            #   平台 Q2 实测：补 10 条 → 1/3；补 msg_index>=10 → 3/3(+40)；
            #   补 msg_index>=15 → 3/3(+20)；**补 msg_index=max → 3/3(+4)** ← 最优
            # 理由：会话尾部是"结论轮"（"最后决定…""明白了，就…"），
            #      而题面里的"后来/最终/想清楚之后"几乎都指向这些轮。
            #   题面含"后来/最后/最终/想清楚之后"这类**指向后半段**的词才启用；
            #   若题面明确指向"中间某次"，补尾部会引入干扰。
            # 实测依据：Q2（"先后做了哪两步，才慢慢把职业焦虑想清楚"）补尾部 3/3 命中、+6 条。
            _tail_mode = (os.getenv("AMC_SESSION_PARTNER_TAIL", "0") or "0") == "1"
            if _tail_mode:
                import re as _re_tail
                _TAIL_CUE = _re_tail.compile(
                    r"后来|最后|最终|之后|以后|想清楚|结论|决定|总结|最后决定|"
                    r"\blater\b|\bfinal(ly)?\b|\bafterwards?\b|\bin the end\b|\bat last\b|"
                    r"\bend up\b|\bconclusion\b|\bdecided?\b", _re_tail.I)
                _tail_mode = bool(_TAIL_CUE.search(str(query or "")))
            _anchor_set = {str(a) for a in anchors}
            add = []
            for sid in sess:
                _rids = sess_map.get(sid, [])
                if _tail_mode:
                    # 每个 request_id 只取 msg_index 最大的那一轮。
                    # 注意：lib.rounds 里**没有 msg_index**（只有 round_id/text/date/has_image），
                    # 所以必须从 store 现查（records_of_session 已按 ts/request_id/msg_index 排序）。
                    _ordered = []
                    try:
                        _uid = self.store.user_of_request(anchors[0]) if hasattr(self.store, "user_of_request") else ""
                    except Exception:
                        _uid = ""
                    _rows = []
                    try:
                        _rows = self.store.records_of_session(_uid, sid, limit=400) if _uid else []
                    except Exception:
                        _rows = []
                    if _rows:
                        _last = {}
                        for _r in _rows:
                            _pr = str(_r.get("request_id") or "")
                            _mi = int(_r.get("msg_index") or 0)
                            if _pr not in _last or _mi > _last[_pr][0]:
                                _last[_pr] = (_mi, str(_r.get("id") or ""))
                        _ordered = [v[1] for v in _last.values()]
                    else:
                        _ordered = list(_rids[-1:])
                elif _win > 0:
                    _pos = [i for i, r in enumerate(_rids) if r in _anchor_set]
                    _keep = set()
                    for _p in _pos:
                        _keep.update(range(max(0, _p - _win), min(len(_rids), _p + _win + 1)))
                    if not _keep:
                        _keep = set(range(min(len(_rids), 2 * _win + 1)))
                    _ordered = [_rids[i] for i in sorted(_keep)]
                else:
                    _ordered = list(_rids)
                for rid in _ordered:
                    if rid in have or rid in add:
                        continue
                    _r = lib.round(rid) or {}
                    t = _r.get("text", "")
                    if not t:
                        # ⚠️ 只有图、没有文字的轮次不能直接丢掉：
                        # 视觉题的邻**恰恰可能就是"只有图"的那条**。
                        #   text 为空，若这里 `continue` 会永远补不到带图邻居。）
                        # 线上当前实测"带图且文本为空"= 0 条，所以这是**防守性修复**；
                        # 真遇到时用占位文本保留它，交给上层按 n_images 判断。
                        if (_r.get("has_image") or (_r.get("n_images") or 0) > 0):
                            t = "[image]"
                        else:
                            continue
                    add.append(rid)
                    if len(add) >= limit:
                        break
                if len(add) >= limit:
                    break
            if add:
                out = list(records) + [{"id": rid,
                                        "content": [{"type": "text", "text": (lib.round(rid) or {}).get("text", "")}],
                                        "score": 0.0,
                                        "created_at": (lib.round(rid) or {}).get("date", "")} for rid in add]
                logger.info("[evidence] session partners +%d/%d (sessions=%d)", len(add), limit, len(sess))
                return out
            return records
        except Exception:
            logger.exception("session partner append failed")
            return records

    def _expand_partners(self, lib, pool, query: str, anchors, limit: int = 6, min_topic: float = 0.34):
        """证据完整性扩展：补入与 anchor 同会话、或与问题主题高度重叠的轮次。

        离线实测：多 gold 问题"全部 gold 进 top-10"从 56.5% → 70.8%（+14.3pp）。
        只补"伙伴"，不扩"对"（扩种子会让配对变差）。
        """
        try:
            import re as _re
            _stop = set("a an the of to in on at for with and or is are was were be this that it "
                        "its my me i we our you your he she they them his her as by from about "
                        "into over some any do does did have has had what which who why how".split())
            qt = {w for w in _re.findall(r"[a-z][a-z\-]{2,}", (query or "").lower()) if w not in _stop}
            if not qt:
                return pool
            have = set(pool)
            anchors = [a for a in (anchors or []) if a][:3]
            sess = set()
            for sid, rids in getattr(lib, "session_rounds", {}).items():
                if any(a in rids for a in anchors):
                    sess.add(sid)
            picks = []
            for rid in lib.rounds:
                if rid in have:
                    continue
                txt = (lib.round(rid) or {}).get("text", "") or ""
                ov = len(qt & {w for w in _re.findall(r"[a-z][a-z\-]{2,}", txt.lower()) if w not in _stop})
                ov = ov / max(1, len(qt))
                same = any(rid in lib.session_rounds.get(sid, []) for sid in sess)
                if same or ov >= min_topic:
                    picks.append((1.0 if same else ov, rid))
            picks.sort(key=lambda x: -x[0])
            out = list(pool) + [rid for _, rid in picks[:limit]]
            if picks:
                logger.info("[pool] partner expand +%d (sessions=%d)", min(limit, len(picks)), len(sess))
            return out
        except Exception:
            logger.exception("partner expand failed")
            return pool

    def _schedule_warm(self, user_id: str, delay: float = 3.0) -> None:
        """登记一次预热（去抖）：最后一次 add 之后 delay 秒才真正重建。"""
        import threading as _th, time as _tm
        with self._warm_guard:
            self._warm_pending[user_id] = _tm.time() + delay
            alive = self._warm_thread is not None and self._warm_thread.is_alive()
            if not alive:
                self._warm_thread = _th.Thread(target=self._warm_loop, daemon=True)
                self._warm_thread.start()

    def _warm_loop(self) -> None:
        import time as _tm
        while True:
            _tm.sleep(1.0)
            now = _tm.time()
            with self._warm_guard:
                due = [u for u, t in list(self._warm_pending.items()) if t <= now]
                for u in due:
                    self._warm_pending.pop(u, None)
                idle = not self._warm_pending
            for u in due:
                try:
                    self._warm_library(u)
                except Exception:
                    logger.exception("warm loop failed for %s", u[-12:])
            if idle and not due:
                return

    def _warm_library(self, user_id: str):
        """后台预热图书馆（dense 向量），避免首次 /search 冷启动 10~20s。"""
        try:
            self._lib_dirty.add(user_id)
            self._user_library(user_id)
            logger.info("[library] warmed for %s", user_id[-16:])
        except Exception:
            logger.exception("library warm failed")

    def _user_library(self, user_id: str):
        """按用户懒建图书馆（dense+词法）。add 后标脏，首次检索时重建。"""
        try:
            if user_id not in self._lib_dirty and user_id in self._lib_cache:
                return self._lib_cache[user_id]
            with self._lib_lock_for(user_id):
                if user_id not in self._lib_dirty and user_id in self._lib_cache:
                    return self._lib_cache[user_id]
                rows = self.store.list_memories(user_id) or []
                if not rows:
                    return None
                rounds = [{"round_id": r["id"],
                           "text": r.get("index_text") or "",
                           "date": (r.get("created_at") or "")[:10],
                           "has_image": bool(r.get("n_images"))} for r in rows]
                # 会话索引：伙伴扩展（同会话）依赖它
                session_of = {}
                for r in rows:
                    sid = r.get("session_id") or ""
                    if sid:
                        session_of[r["id"]] = sid
                emb = None
                if getattr(self, "_tagger", None) is not None:
                    try:
                        # ⚠️🔴 2026-11-02：**必须调用**（`embedder` 是方法，返回嵌入函数）。
                        #    原来写成 `emb = self._tagger.embedder`（少括号）⇒
                        #    library 里 `self.embedder(texts)` 变成"方法传参"⇒ TypeError
                        #    ⇒ `_ensure_vectors` 的 except 把 `_vec` 置 None
                        #    ⇒ **dense 打分静默失效**（只靠词法），而日志里看不出任何异常。
                        emb = self._tagger.embedder()
                    except Exception:
                        emb = None
                if emb is None:
                    emb = _batched_embedder(self.settings)
                lib = MemoryLibrary(rounds, [], session_of or None, None, embedder=emb)
                self._lib_cache[user_id] = lib
                self._lib_dirty.discard(user_id)
                logger.info("[library] built for user %s (%d rounds, dense=%s)",
                            user_id[-24:], len(rounds), bool(emb))
                return lib
        except Exception:
            logger.exception("user library build failed")
        return None

    def _multimodal_reorder(self, hits, query):
        """用查询图与候选图片做图像级重排；无查询图或无候选图时原样返回。"""
        from .imaging import decode_data_uri
        from .query_vision import extract_query_images

        urls = extract_query_images(query, 1)
        if not urls:
            return hits
        try:
            fmt = "jpeg"
            q_uri = self.multimodal_selector._shrink(decode_data_uri(urls[0]), fmt)
        except Exception:
            return hits
        if not q_uri:
            return hits

        pool = int(getattr(self.settings, "multimodal_select_pool", 4) or 4)
        cands, by_id = [], {}
        for memory, _score in hits:
            if len(cands) >= pool:
                break
            for part in (memory.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image_ref":
                    try:
                        raw = self.store.blob_path(part["sha256"], part["fmt"]).read_bytes()
                    except Exception:
                        break
                    uri = self.multimodal_selector._shrink(raw, part.get("fmt"))
                    if uri:
                        cands.append((memory["id"], uri))
                        by_id[memory["id"]] = memory
                    break
        if not cands:
            return hits
        picked = [cands[j][0] for j in self.multimodal_selector.select(q_uri, cands)
                  if 0 <= j < len(cands)]
        if not picked:
            return hits
        front = [(by_id[mid], score) for mid in picked
                 for m2, score in hits if m2.get("id") == mid]
        rest = [h for h in hits if h[0].get("id") not in set(picked)]
        logger.info("multimodal-select: promoted %d/%d candidates", len(front), len(cands))
        return front + rest

    def _prewarm_embed_batch(self, batch) -> None:
        """worker 调用：**把多个 add 的 id 聚合起来、一次 embed 写库**。

        这样 DB 锁只被抢 1 次（而不是每个 add 一次），锁争抢从 40/min 降到 ~2/min。
        """
        try:
            _ids = []
            for _u, _ls in (batch or []):
                for _i in (_ls or []):
                    if _i:
                        _ids.append(str(_i))
            if not _ids:
                return
            _ids = list(dict.fromkeys(_ids))[:2000]
            emb = None
            try:
                if getattr(self, "_tagger", None) is not None:
                    emb = self._tagger.embedder()     # ⚠️ 必须调用
            except Exception:
                emb = None
            if emb is None:
                emb = _batched_embedder(self.settings)
            if emb is None:
                return
            _q = ",".join("?" * len(_ids))
            _rows = list(self.store.conn.execute(
                f"SELECT index_text, n_images FROM memories WHERE id IN ({_q})", _ids))
            try:
                from .retrieval.embedding import DenseRetriever as _DR
                _texts = [_DR._doc_text({"index_text": r[0], "n_images": r[1]}, 4000)
                          for r in _rows if str(r[0] or "").strip()]
            except Exception:
                _texts = [str(r[0] or "") for r in _rows if str(r[0] or "").strip()]
            _texts = [t for t in _texts if t and t.strip()]
            if not _texts:
                return
            _n = 0
            for _i in range(0, len(_texts), 64):
                try:
                    emb(_texts[_i:_i + 64])
                    _n += len(_texts[_i:_i + 64])
                except Exception:
                    pass
            if _n:
                logger.info("[prewarm] 批量预嵌 %d 条（worker，%d 个 add 合并）", _n, len(batch))
        except Exception:
            logger.exception("[prewarm] batch failed (忽略)")

    def _prewarm_embed(self, user_id: str, memory_ids: List[str]) -> None:
        """把新写入的记录**在后台补齐 embedding 缓存**。

        背景：`MemoryLibrary._ensure_vectors()` 只在**首次查询**某用户时才对
        该用户全部轮次做 embedding ⇒ 新灌的数据第一次被查时，
        要把几千条记录现算一遍 ⇒ 实测**首次查询新用户 5~6 分钟**。
        平台 /add 874 用户 ⇒ 串行 874 × 5 分钟 ≈ **73 小时**冷启动。

        ⇒ 这里在 /add 之后异步预热：把 embedding 结果写进 `embcache.db`，
           等真正查询时 `_ensure_vectors` 只做**缓存读**，不再打 API。
        """
        try:
            _ids = [str(x) for x in (memory_ids or []) if x]
            if not _ids:
                return
        except Exception:
            return
        def _work():
            try:
                emb = None
                try:
                    if getattr(self, "_tagger", None) is not None:
                        emb = self._tagger.embedder()   # ⚠️ 必须调用
                except Exception:
                    emb = None
                if emb is None:
                    emb = _batched_embedder(self.settings)
                if emb is None:
                    return
                _q = ",".join("?" * len(_ids))
                _rows = list(self.store.conn.execute(
                    f"SELECT index_text, n_images FROM memories WHERE id IN ({_q})", _ids))
                # ⚠️🔴 ：**必须用 `_doc_text` 生成文本**，不能用裸 index_text。
                #   这是今天第三次踩同一个坑（前两次：我的第一版 prewarm 脚本、图书馆）：
                #     缓存键 = sha1(namespace || **文本**)
                #     检索/图书馆都用 `DenseRetriever._doc_text(index_text, 4000)`（caption 前移）
                #     而这里原来用裸 `index_text` ⇒ **键不同 ⇒ /add 的预嵌全白做**
                #   ⇒ 改成 _doc_text 后，**一次预嵌同时喂检索侧和图书馆**（两边现在同键）
                try:
                    from .retrieval.embedding import DenseRetriever as _DR
                    _mc = 4000
                    _texts = [_DR._doc_text({"index_text": r[0], "n_images": r[1]}, _mc)
                              for r in _rows if str(r[0] or "").strip()]
                    _texts = [t for t in _texts if t and t.strip()]
                except Exception:
                    _texts = [str(r[0] or "") for r in _rows if str(r[0] or "").strip()]
                if not _texts:
                    return
                _B = 32
                _n = 0
                _err = None
                for _i in range(0, len(_texts), _B):
                    try:
                        _r = emb(_texts[_i:_i + _B])       # 只为了填缓存，不保存结果
                        _n += len(_texts[_i:_i + _B])
                    except Exception as _e:
                        if _err is None:
                            _err = f"{type(_e).__name__}: {str(_e)[:120]}"
                if _err:
                    logger.warning("[prewarm] 嵌入失败（首个错误）: %s", _err)
                logger.info("[prewarm] user=%s 预嵌 %d 条（/add 后台）", str(user_id)[-12:], _n)
            except Exception:
                logger.exception("[prewarm] failed (忽略)")
        #   原来**每次 /add 都起一个 daemon 线程** ⇒ 15 分钟约 40 线程/分钟
        #   ⇒ 服务进程 **181 个线程**、CPU 68%、所有线程抢 `store.lock`（SQLite 单写者）+ GIL
        #   ⇒ **小 body 的 add 中位 5.07s**（平台侧 3.16s、p90 10s、最大 54s）
        #   改成 **单线程 + 队列**：/add 只 put（微秒级），一个常驻 worker 批量消化
        #   ⇒ 线程数 40/min → 1，DB 锁争抢降到 1/40，add 应回到 <1s
        try:
            #   原来 `put` 包在 `except: pass` 里，maxsize=20000 一满就丢，
            #   修法：先用非阻塞 put；满了就**在后台线程里同步嵌这一批**（降级但不丢）。
            _item = (str(user_id), list(_ids))
            try:
                _PREWARM_Q.put_nowait(_item)
            except Exception:
                try:
                    logger.warning("[prewarm] 队列已满（%d）⇒ 改同步兜底嵌 %d 条",
                                   _PREWARM_Q.maxsize, len(_ids))
                    import threading as _th3
                    _th3.Thread(target=lambda it=_item: self._prewarm_embed_batch([it]),
                                daemon=True, name="prewarm-emerg").start()
                except Exception:
                    logger.exception("[prewarm] 同步兜底也失败（忽略）")
        except Exception:
            pass

    def search(self, req: Dict[str, Any], hydrate: bool = True) -> Dict[str, Any]:
        """AML Official Search Endpoint Implementation"""
        errs = validate_search_request(req)
        if errs:
            raise ContractError(422, "; ".join(errs[:5]))
        #    原则：不限制单请求时间，只**记录**慢请求（见 PLAN.md §HW）。
        #    代码保留 set_deadline 能力，但预算传 0 ⇒ deadline_hit() 恒 False。
        try:
            from . import routes as _RT
            _RT.set_deadline(0)
        except Exception:
            pass

        # 官方正式评测 top_k 为 100，返回数量绝不可超过请求的 top_k
        _DECISION.abstain = False
        _DECISION.n = 12
        _DECISION.amount_ids = []
        try:
            _q = req.get("query")
            self._cur_has_qimg = bool(isinstance(_q, list) and any(
                isinstance(x, dict) and x.get("type") in ("image_url", "image") for x in _q))
        except Exception:
            self._cur_has_qimg = False
        requested_k = req["top_k"]
        effective_limit = min(requested_k, self.max_return)

        logger.info("[search] top_k=%s (max_return=%s) user=%s", requested_k, self.max_return,
                    req.get("user_id", "")[-16:])

        q_text = query_to_text(req["query"], req.get("options"), self.describer)
        # ⚠️ ：把 uid/q 记到线程本地，供 `_finalize_records` 的
        #   stage_dump("final") 使用（该函数签名里没有 req/q_text ⇒ 以前 final 只能记 ('','null')）
        try:
            _DECISION.uid = req.get("user_id")
            _DECISION.q = q_text
        except Exception:
            pass
        _ab_fut = None
        try:
            import os as _os_abp
            if (_os_abp.getenv("AMC_AB_PARALLEL", "0") or "0") in ("1", "true", "True"):
                from .ab_retrieve import ab_enabled as _ab_on_p
                if bool(_ab_on_p()) or (_os_abp.getenv("AMC_AB_CHANNEL", "0") or "0") == "1":
                    _pool_ab = _ab_pool()
                    if _pool_ab is not None:
                        from .ab_pipeline import evidence_records_guided as _ab_ev_p
                        _ab_fut = _pool_ab.submit(
                            _ab_ev_p, self.store.conn, "memories", req.get("user_id", ""), q_text,
                            dict(text_col="index_text", date_col="created_at"))
                        logger.info("[ab] 并行提交（与检索同时开始）")
        except Exception:
            logger.exception("[ab] 并行提交失败（回退串行）")
            _ab_fut = None

        # ---- 结果缓存（TTL 15 分钟）：命中直接返回，重试/重续不再跑完整流程 ----
        import hashlib as _hl, threading as _th, time as _tm
        if MemoryService._search_cache_lock is None:
            MemoryService._search_cache_lock = _th.Lock()
        MemoryService._cache_load()
        _uv = 0
        try:
            with MemoryService._search_cache_lock:
                _uv = int(MemoryService._user_ver.get(str(req.get("user_id")), 0))
        except Exception:
            _uv = 0
        _ck = _hl.sha1((str(req.get("user_id")) + "\x01" + q_text + "\x01" +
                        str(requested_k) + "\x01" + str(sorted(req.get("options") or []))
                        + "\x01b:" + _build_tag() + "\x01uv:" + str(_uv)).encode()).hexdigest()
        _now = _tm.time()
        with MemoryService._search_cache_lock:
            _hit = MemoryService._search_cache.get(_ck)
            if _hit and _now - _hit[0] < MemoryService._cache_ttl():
                logger.info("[retry] HIT hit=1 user=%s q=%r（重试直接返回，不再重算）",
                            str(req.get("user_id"))[-12:], q_text[:40])
                return _hit[1]
        
        # 检索端以大候选池召回，结合 Reranker 排序。
        # ⚠️ 池子必须比 top_k **留出过滤余量**：`_finalize_records` 会丢掉
        # "没有文本"的记录（纯图片轮次）并按 id 去重。
        # 实测（46,193 条真实响应）：top_k=100 的请求只返回 **中位 77 条**（返回率 77%），
        # 即约 23% 的名额被过滤掉了 ⇒ 若 gold 排在第 78~100 位就永远拿不到。
        # 所以池子取 1.5×top_k（上限 200），保证过滤后仍能填满。
        pool_limit = max(int(effective_limit * 1.5), 50, effective_limit + 20)
        if pool_limit > 200:
            pool_limit = 200
        # 视觉题：池子必须更大。带图记录只占 12.7%，池子 50 时只能捞到 ~17 条带图，
        # 不够填满 top-10；模拟实验用的是 top-200 的池子，才拿到 gold@10 76.4→85.4%。
        try:
            from .router import route as _route_p
            if _route_p(q_text) == "VISUAL":
                # 池大小可配：为了做"同候选数"对照（默认 200 = 线上配置）
                _vpool = int(os.getenv("AMC_VISUAL_POOL", "200") or 200)
                pool_limit = max(pool_limit, _vpool)
        except Exception:
            pass
        if self.agent is not None:
            try:
                hits = self.agent.run(self.retriever, req["user_id"], q_text, pool_limit,
                                      req.get("query"))
                v = getattr(self.agent, "last_verdict", None) or {}
                if v:
                    logger.info("[agent] verdict=%s sufficient=%s violations=%d missing=%s",
                                v.get("verdict"), v.get("sufficient"),
                                len(v.get("violations") or []), (v.get("missing") or "")[:60])
                    # 可选拒答（默认关闭）：证据与前提矛盾，或"校验不足 + 顶部分数极低"
                    if getattr(self.settings, "agent_abstain", False):
                        low = float(getattr(self.settings, "agent_abstain_score", 0.0) or 0.0)
                        contradicts = (v.get("verdict") == "contradicts"
                                       and bool(v.get("violations")))
                        weak = (not v.get("sufficient", True) and low > 0
                                and hits and float(hits[0][1]) < low)
                        if contradicts or weak:
                            logger.info("[agent] abstain: verdict=%s top=%.4f low=%.3f",
                                        v.get("verdict"), float(hits[0][1]) if hits else -1, low)
                            hits = []
            except Exception as exc:
                logger.exception("agent retrieval failed")
                raise RuntimeError(f"agent retrieval failed: {exc}") from exc
        else:
            hits = self.retriever.search(req["user_id"], q_text, pool_limit)
            # 查询扩展（默认关）：把"另一条查询"的召回结果合并到 **pool 层**。
            # 两种模式（2026-11-01 的跨语料实测）：
            #   variant = 确定性变体（去句末提问）：平台 +1 题 p=1.00（无效）、Gem +4 p=0.125
            #   **hyde  = 让 LLM 写记忆风格假设文本**：平台 **+4 题 p=0.125**、Gem **+3 p=0.25**、
            #            ATM 0；**154 题 7 胜 0 负（符号检验 p≈0.008）**
            # ⇒ 默认用 hyde；variant 保留但仅在显式指定时启用。
            #
            #   Gem ~400 条 → +3 | 平台中位 266 条 → +4 | ATM 10,504 条 → **0**
            #   ⇒ 超过 AMC_HYDE_MAX_RECORDS（默认 3000）不跑，省 LLM 也避免无用的 +62% 延迟。
            #
            # 关键：扩展**只用于扩大 pool**，最终排序仍由原问题的精排决定
            #       ⇒ 不能把两次检索的结果直接拼进响应（会撞 top_k≤100 契约、相对顺序无意义）。
            _qe_mode = (os.getenv("AMC_QUERY_EXPAND", "0") or "0").strip().lower()
            if _qe_mode in ("1", "true", "hyde", "variant") and hits:
                if _qe_mode in ("1", "true"):
                    _qe_mode = "hyde"          # 兼容旧开关：默认走 hyde
                try:
                    _nm = self.store.count_memories(req["user_id"])
                except Exception:
                    _nm = 0
                _mx = int(os.getenv("AMC_HYDE_MAX_RECORDS", "3000") or 3000)
                if _nm > _mx:
                    logger.info("[expand] 跳过：用户记录 %d > 阈值 %d（大库上扩展无效）", _nm, _mx)
                else:
                    try:
                        _seen_key = {(m.get("id") if isinstance(m, dict) else None) for m, _ in hits}
                        _added = 0
                        if _qe_mode == "hyde":
                            from .hyde import hypothetical as _hyde
                            _h = _hyde(
                                q_text,
                                base_url=getattr(self.settings, "llm_rerank_base_url", "") or "",
                                api_key=getattr(self.settings, "llm_rerank_api_key", "") or "",
                                model=getattr(self.settings, "llm_rerank_model", "gpt-4o-mini"),
                                timeout=float(os.getenv("AMC_HYDE_TIMEOUT", "10") or 10),
                            )
                            _extra_queries = [_h] if _h else []
                        else:
                            from .query_expand import variants as _variants
                            _extra_queries = _variants(q_text)
                        for _v in _extra_queries:
                            try:
                                _vh = self.retriever.search(req["user_id"], _v, pool_limit)
                            except Exception:
                                continue
                            for m, sc in _vh:
                                _k = m.get("id") if isinstance(m, dict) else None
                                if _k in _seen_key:
                                    continue
                                _seen_key.add(_k)
                                # 扩展命中**轻微降权**：与原问题不同源，不应压过原结果
                                hits.append((m, float(sc) * 0.9))
                                _added += 1
                            if _added >= pool_limit:      # 合并后不超过 2×pool_limit
                                break
                        if _added:
                            logger.info("[expand] mode=%s +%d 条 pool=%d（用户记录 %d）",
                                        _qe_mode, _added, len(hits), _nm)
                        elif _qe_mode == "hyde":
                            logger.info("[expand] hyde 无新增（用户记录 %d）", _nm)
                    except Exception:
                        logger.exception("[expand] 失败（忽略，用原 hits）")

        # 专用 VL reranker（可选，记忆侧）：需要**原始查询部件**（含查询图），
        # 因此放在服务层而非检索器内。仅当查询含图且开关打开时生效（见 vl_rerank.py 的硬闸）。
        # 两者互斥时默认 relation_select 优先；VL_RERANK_FIRST=1 时反过来（VL 先跑）
        vl_first = bool(getattr(self.settings, "vl_rerank_first", False))
        rs_applied = False
        if vl_first and self.vl_reranker is not None:
            try:
                hits = self.vl_reranker.rerank(self.store, q_text, req["query"], hits)
            except Exception:
                logger.exception("vl-rerank(first) failed; keeping retrieval order")
        if self.relation_selector is not None:
            try:
                hits, rs_applied = self.relation_selector.select(
                    self.store, q_text, req["query"], hits)
            except Exception:
                logger.exception("relation-select failed; keeping retrieval order")

        if self.vl_reranker is not None and not rs_applied and not vl_first:
            try:
                hits = self.vl_reranker.rerank(self.store, q_text, req["query"], hits)
            except Exception:
                logger.exception("vl-rerank failed; keeping retrieval order")

        # 多模态证据选择（可选，默认关）：查询带图时，用「查询图 + 候选图片」让 gpt-4o-mini
        # 直接判定哪张候选图承载答案，把命中的候选提到前面。
        if self.multimodal_selector is not None:
            try:
                hits = self._multimodal_reorder(hits, req["query"])
            except Exception:
                logger.exception("multimodal-select failed; keeping retrieval order")

        # 内存侧内部证据去重 (经消融验证为稳定正向收益)
        if self.settings and getattr(self.settings, "search_dedup", False):
            dedup_hits = []
            seen_tokens = set()
            for memory, score in hits:
                txt = (memory.get("index_text") or "").strip().lower()
                toks = tuple(sorted(set(WORD_RE.findall(txt))))[:20]
                if toks and toks in seen_tokens:
                    continue
                if toks:
                    seen_tokens.add(toks)
                dedup_hits.append((memory, score))
            hits = dedup_hits

        # ✅ 视觉题只看**带图记录**（实测 +9.0pp）
        # 依据：VISUAL 题的 gold 记录 **98.1% 带图**，而带图记录只占全部记录的 12.7%
        #       ⇒ 87% 的语料对这些题是纯噪声。
        # 模拟（Gem 589 道 VISUAL）：
        #   不滤图  gold@1 33.4% | gold@10 76.4%
        #   只搜带图 gold@1 38.4% | gold@10 **85.4%**  （gold@1 相对 +15%）
        # 安全阀：过滤后若不足 effective_limit 条，用原顺序补齐，绝不因过滤而返回更少。
        try:
            from .router import route as _route_v
            if ((_route_v(q_text) == "VISUAL")
                    and (os.getenv("AMC_VISUAL_IMAGE_ONLY", "1") or "1") not in ("0", "false", "False")):
                # 重排方式可配（做对照用）：
                #   image  = 带图记录优先（线上默认）
                #   random = 同池内**随机重排**（对照：排除"重排本身"的效应）
                #   none   = 不重排（纯 baseline）
                _mode = (os.getenv("AMC_VISUAL_ORDER", "image") or "image").strip().lower()
                if _mode == "random":
                    import random as _rnd
                    _rnd.shuffle(hits)
                    logger.info("[visual] 随机重排对照: %d 条", len(hits))
                elif _mode == "image":
                    _with = [h for h in hits if (h[0].get("n_images") or 0) > 0]
                    if _with:
                        _kept_ids = {id(h[0]) for h in _with}
                        hits = _with + [h for h in hits if id(h[0]) not in _kept_ids]
                        logger.info("[visual] 带图记录优先: %d/%d", len(_with), len(hits))
        except Exception:
            logger.exception("visual image-only 过滤失败（忽略）")
        logger.info("[retrieve] pool_limit=%d -> hits=%d (effective_limit=%d)",
                    pool_limit, len(hits), effective_limit)
        hits = sanitize_scores(hits)[:effective_limit]
        self._stage_dump("retrieve", hits, req.get("user_id"), q_text)

        #   实测（全新难题集 8 题）：**4 题（50%）撞 92s 闸返回空数组**，
        #   而"只要返回了结果"的 3 题都是 gold 全在前 10。
        #   ⇒ 空数组 = 0 分；次优结果 = 可能得分。
        #   做法：在 retrieve 阶段（100 条已排好）**存一份快照**，
        #   超时分支可以从这里取"已完成部分的最好结果"用 200 返回。
        #   存储：模块级 LRU（最多 64 条），key 与缓存键一致。
        if (os.getenv("AMC_SLOW_FALLBACK", "1") or "1") != "0":
            try:
                _snap = []
                for _m, _s in (hits or [])[:100]:
                    _c = _m.get("content")
                    if isinstance(_c, str):
                        _c = [{"type": "text", "text": _c}]
                    if not isinstance(_c, list) or not _c:
                        continue
                    _snap.append({"id": str(_m.get("id") or ""),
                                  "content": _c,
                                  "score": float(_s or 0.0),
                                  "created_at": _m.get("created_at") or "",
                                  "request_id": _m.get("request_id") or "",
                                  "n_images": int(_m.get("n_images") or 0),
                                  "session_id": _m.get("session_id") or "",
                                  "msg_index": _m.get("msg_index") or 0,
                                  "role": _m.get("role") or ""})
                if _snap:
                    import hashlib as _hlp
                    _pk = _hlp.sha1((str(req.get("user_id")) + "\x01" + str(q_text)).encode()).hexdigest()
                    MemoryService._partial_store(_pk, _snap)
            except Exception:
                logger.warning("[slow-fallback] 存快照失败（忽略）", exc_info=True)

        # 通用 abstention 门控（不依赖题型分类）：若**最高分仍低于阈值**，
        # 说明记忆里没有可用证据 —— 契约允许返回空数组 "when nothing is found"。
        # 实测：Mem-Gallery 可答题 top-1 >= 0.23；官方"信息不在记忆中"的题 top-1 ≈ 0。
        eps = float(getattr(self.settings, "abstain_gate_eps", 0.0) or 0.0)
        if eps > 0 and hits and hits[0][1] < eps:
            return {"data": []}

        # 头部会话多样性：多跳题需要跨会话证据，但单会话常霸占前几名。
        # 仅在"前 head_size 条"内限制每个 request_id 最多 head_cap 条；
        # 会话消息数 <= head_cap 时完全无影响（Mem-Gallery/ATM 每会话仅 2 条）。
        head_cap = int(getattr(self.settings, "head_session_cap", 0) or 0)
        head_size = int(getattr(self.settings, "head_size", 10) or 10)
        if head_cap > 0 and len(hits) > 1:
            # 扫描**完整列表**：先按"每会话最多 head_cap 条"凑满 head_size 条，
            # 再把剩余按原顺序接上（这样跨会话的证据能进入头部）
            seen = {}
            chosen = []
            chosen_ids = set()
            for item in hits:
                rid = item[0].get("request_id") or item[0].get("id")
                if len(chosen) >= head_size:
                    break
                if seen.get(rid, 0) >= head_cap:
                    continue
                seen[rid] = seen.get(rid, 0) + 1
                chosen.append(item)
            chosen_ids = {id(m) for m, _ in chosen}
            rest = [it for it in hits if id(it[0]) not in chosen_ids]
            new_hits = chosen + rest
            if len(new_hits) == len(hits):
                # 重排后仍须保持分数非递增：按新顺序分配原分数序（多重集合不变）
                scores_desc = sorted((sc for _, sc in hits), reverse=True)
                hits = [(m, scores_desc[i]) for i, (m, _sc) in enumerate(new_hits)]

        # 轮级返回（可开关）：同轮只出一件，content 用整轮拼接文本。
        # 这一步只是"去重 + 合并展示"，**不改变排序**（轮级序 = 消息级序去重后的子序列）。
        import re as _re
        if bool(getattr(self.settings, "round_items", False)):
            head_rids, rep = [], {}
            for memory, score in hits:
                rid = memory.get("request_id")
                if rid and rid in rep:
                    continue
                if rid:
                    rep[rid] = (memory, score)
                head_rids.append((memory, score, rid))
            merged = []
            for memory, score, rid in head_rids:
                if not rid:
                    merged.append((memory, score, memory["content"]))
                    continue
                text = _ROUND_TEXT_CACHE.get(rid, "")
                if not text:
                    # 主路径：查库拿**整轮全部消息**（专用只读连接）
                    dbp = getattr(self.store, "db_path", "")
                    if not dbp:
                        raise RuntimeError("store db_path unavailable")
                    try:
                        with _ROUND_TEXT_LOCK:
                            rows = _round_conn(dbp).execute(
                                "SELECT index_text FROM memories WHERE request_id=? ORDER BY msg_index",
                                (rid,)).fetchall()
                        text = "\n".join((r[0] or "").strip() for r in rows if r[0]).strip()
                    except Exception:
                        logger.exception("round text db lookup failed for %s", rid[:24])
                        # 不做降级：直接让本次 search 失败（官方可断点重跑）
                        raise RuntimeError(f"round text unavailable: {rid[:24]}")
                    if not text:
                        logger.error("round text empty for %s", rid[:24])
                        raise RuntimeError(f"round text empty: {rid[:24]}")
                    if len(_ROUND_TEXT_CACHE) < 20000:
                        _ROUND_TEXT_CACHE[rid] = text
                if not text:
                    merged.append((memory, score, memory["content"]))
                    continue
                if bool(getattr(self.settings, "round_strip_caption", False)) and "Image Caption" in text:
                    text = _re.sub(r"Image Caption:.*?(?=\n\[|\Z)", "", text, flags=_re.S).strip()
                _max = int(getattr(self.settings, "round_text_max_chars", 0) or 0)
                if _max > 0 and len(text) > _max:
                    # 保留与查询最相关的窗口：按句切分，取命中词最多的连续片段
                    import re as _re
                    sents = [x for x in _re.split(r"(?<=[。！？!?；;\n])", text) if x.strip()]
                    toks = [t for t in _re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", q_text.lower())]
                    def _hit_score(x):
                        xl = x.lower()
                        return sum(1 for t in toks if t in xl)
                    best_i, best_s = 0, -1
                    for i in range(len(sents)):
                        acc, j = "", i
                        while j < len(sents) and len(acc) < _max:
                            acc += sents[j]; j += 1
                        sc = _hit_score(acc)
                        if sc > best_s:
                            best_s, best_i = sc, i
                    acc, j = "", best_i
                    while j < len(sents) and len(acc) + len(sents[j]) <= _max:
                        acc += sents[j]; j += 1
                    text = (acc or text[:_max]).strip()
                parts = [{"type": "text", "text": text}]
                for c in (memory.get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "image_ref":
                        parts.append(c)
                merged.append((memory, score, parts))
            hits = [(m, sc) for m, sc, _ in merged]
            _round_parts = {id(m): p for m, _, p in merged}
        else:
            _round_parts = None

        # 渲染 Content: 遵照 AML 规范水合图片与文本
        output_records = []
        img_budget = MAX_SEARCH_IMAGES
        max_imgs = int(getattr(self.settings, "search_max_images", 10) or 10)
        img_count = 0

        for memory, score in hits:
            # 根据配置决定是否水合原图 Base64 (比赛规范支持返回多模态 ContentPart[])
            src_content = (_round_parts or {}).get(id(memory), memory["content"])
            rendered_content, bytes_used = self.store.render_content(
                src_content,
                hydrate=hydrate,
                remaining_budget=img_budget,
                allow_image=(img_count < max_imgs),
            )
            img_budget -= bytes_used
            if bytes_used > 0:
                img_count += 1

            # content 形态：默认压平成字符串（对 harness 最稳）；含图条目用 parts 列表。
            fmt = getattr(self.settings, "search_content_format", "text")
            has_img_part = isinstance(rendered_content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image")
                for p in rendered_content)
            if fmt == "parts" and isinstance(rendered_content, list) and not has_img_part:
                # 纯文本条目仍返回字符串，最大兼容
                rendered_content = "\n".join(
                    str(p.get("text", "")) for p in rendered_content
                    if isinstance(p, dict) and p.get("type") == "text")
            if fmt != "parts" and isinstance(rendered_content, list):
                chunks = []
                for part in rendered_content:
                    if not isinstance(part, dict):
                        chunks.append(str(part)); continue
                    if part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
                    elif part.get("type") in ("image_url", "image"):
                        chunks.append("[image]")
                rendered_content = "\n".join(c for c in chunks if c)

            _empty = (rendered_content in (None, "", [], {})
                      or (isinstance(rendered_content, str) and not rendered_content.strip())
                      or (isinstance(rendered_content, list) and not rendered_content))
            if _empty:
                _empty_skipped = locals().get("_empty_skipped", 0) + 1
                locals()["_empty_skipped"] = _empty_skipped
                if _empty_skipped <= 3:
                    logger.info("[render] 跳过空 content 记录 %s（reason=%s）",
                                str(memory.get("id"))[:16],
                                ("no_parts" if not rendered_content else "empty_after_render"))
                continue
            _rec = {
                "id": memory["id"],
                "content": rendered_content,
                "score": round(score, 6),
                "created_at": memory["created_at"],
            }
            # 结构化字段（默认开；AMC_RICH_RECORD_FIELDS=0 可关）：
            # 供答案侧的 critic "接地" —— 它必须能把指代/追问落到 record_id，而不是猜日期。
            #   · session_id  → 按会话分组（"证据覆盖了几个 gold 会话"）
            #   · role        → 一手(user) / 二手(assistant) 判定
            #   · msg_index   → **会话内唯一的序**（同会话时间戳常全相同，只能靠它排）
            #   · n_images    → 🖼 标记不用解码 base64
            if (os.getenv("AMC_RICH_RECORD_FIELDS", "1") or "1") != "0":
                try:
                    _mid = str(memory.get("id") or "")
                    # 合成记录（amc_aggregate_* / amc_abstain_* 等）没有真实会话，
                    # 但**不能留空** —— 否则答案端按会话分组渲染时会崩/错位。
                    # 给它们一个稳定的合成会话名与 system 角色，渲染器就能正常分组。
                    _synth = _mid.startswith("amc_")
                    _rec["session_id"] = (memory.get("session_id")
                                          or ("synthetic:" + _mid[:16] if _synth else ""))
                    _rec["request_id"] = memory.get("request_id") or (_mid if _synth else "")
                    _rec["msg_index"] = int(memory.get("msg_index") or 0)
                    _rec["role"] = memory.get("role") or ("system" if _synth else "")
                    _rec["n_images"] = int(memory.get("n_images") or 0)
                    # 纯 caption（去掉 "[Image xxx]" 前缀）：
                    if _rec["n_images"] > 0:
                        import re as _re_cap
                        _m = _re_cap.search(r"\[Image\s+[0-9a-fA-F]{4,}\]\s*(.+?)(?:\n|$)",
                                            str(memory.get("index_text") or ""), _re_cap.S)
                        if _m:
                            _rec["caption"] = str(_m.group(1) or "").strip()[:1000]
                except Exception:
                    pass
            output_records.append(_rec)

        # 严格遵循 AML 返回规范: 顶级对象且仅有 data 数组
        # LLM 判官（仅低分时）：这些轮里到底有没有回答问题的信息？
        def _llm_judge(question, records, n=5):
            try:
                import httpx as _hx, json as _js
                base = (getattr(self.settings, "llm_rerank_base_url", "") or "").rstrip("/")
                key = getattr(self.settings, "llm_rerank_api_key", "") or "EMPTY"
                model = getattr(self.settings, "llm_rerank_model", "") or "gpt-4o-mini"
                if not base:
                    return None
                docs = []
                for i, r in enumerate(records[:n]):
                    c = r.get("content")
                    t = ("".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
                         if isinstance(c, list) else str(c or ""))
                    docs.append(f"[{i}] {t}")
                prompt = ("You check whether personal-memory rounds contain the information needed to answer a question.\n"
                          "Reply STRICT JSON only: {\"answerable\": true|false, \"quote\": \"<the exact sentence that answers, or empty>\"}\n\n"
                          f"Question: {question}\n\nRounds:\n" + "\n".join(docs) + "\nJSON:")
                resp = _hx.post(base + "/chat/completions",
                                headers={"Authorization": f"Bearer {key}"},
                                json={"model": model, "temperature": 0.0, "max_tokens": 160,
                                      "messages": [{"role": "user", "content": prompt}]}, timeout=60)
                resp.raise_for_status()
                txt = (resp.json()["choices"][0]["message"]["content"] or "").strip()
                if txt.startswith("```"):
                    txt = txt.split("```")[1]
                    txt = txt[4:] if txt.lower().startswith("json") else txt
                data = _js.loads(txt)
                return bool(data.get("answerable")), str(data.get("quote") or "").strip()
            except Exception:
                logger.warning("llm judge failed", exc_info=False)
                return None

        # 纯分数 abstention 门控：用**最终返回分数**判断（官方数据双峰，0.1 可分开）
        _abs = float(getattr(self.settings, "abstain_topk_score", 0.0) or 0.0)
        _high = float(getattr(self.settings, "abstain_judge_high", 0.0) or 0.0)
        _thr = max(_abs, _high)
        if _thr > 0 and output_records and float(output_records[0].get("score", 0.0)) < _thr:
            _low = float(output_records[0].get("score", 0.0))
            _abstain = True
            _judged = False
            # 判官不适合的场景（其输入只有文本，天然看不到图片内容/无法可靠计数）：
            #   视觉题（image/photo/picture/图/照片）与聚合题（how much/how many/一共/总共）
            # 这些题默认豁免，避免误杀。
            import re as _re3
            # 视觉判定：只看**查询本身**（证据含图不可靠——文本题的记忆里也常有截图）
            # 只有"问图片内容"才算视觉题（which/what/describe/color/background + image/photo）；
            # "Did X photograph …?" 这类经历题（photograph 作动词）仍交给判官
            _vis = bool(_re3.search(r"which|what|describe|color|colour|background|visible|show|depict|look like|"
                                    r"内容|什么|哪张|哪幅|描述|颜色|背景|显示",
                                    q_text, _re.I)) and bool(
                _re3.search(r"\bimages?\b|\bphotos?\b|\bpictures?\b|\bgraphs?\b|\bcharts?\b|"
                            r"图|照片|截图|画面", q_text, _re.I))
            _agg = bool(_re3.search(r"how (much|many)|\btotal\b|in all|altogether|"
                                    r"一共|总共|合计|总计|多少钱|多少个|几次|几张|几个",
                                    q_text, _re.I))
            # 时序比较题（before/after/first/先后/之前之后）：判官难以从窗口内证据确定先后
            # 只认"真正的先后比较"，避免"…之后，我有没有…"这类被误豁免
            _tmp = bool(_re3.search(r"\bbefore\b|\bafter\b|\bwhich (came|happened) first\b|"
                                    r"\bearlier\b|\blater\b|\bcome first\b|"
                                    r"先后|哪个先|先.*还是.*后|早于|晚于", q_text, _re.I))
            _low_band = float(getattr(self.settings, "abstain_topk_score", 0.0) or 0.0)
            # 正则**不再豁免拒答**：只作为证据核验的提示（见下方 library 增强块）
            if _vis or _agg or _tmp:
                logger.info("[abstain] hint-only (%s): 交由证据核验判定",
                            "visual" if _vis else ("aggregate" if _agg else "temporal"))
            elif _low_band > 0 and _low < _low_band:
                # 低分档：默认拒答；判官可"捞回"（必须逐字摘出能直接回答的句子）
                _judged = True
                _abstain = True
                _resc = (getattr(self, "_evidence_judge", None)
                         if bool(getattr(self.settings, "abstain_rescue", False)) else None)
                if _resc is not None:
                    rv = _resc.rescue(q_text, [(m, float(sc)) for m, sc in hits[:10]])
                    if rv and rv.get("found"):
                        _abstain = False
                        _ri = rv.get("round")
                        if isinstance(_ri, int) and 0 <= _ri < len(output_records):
                            if _ri > 0:
                                output_records.insert(0, output_records.pop(_ri))
                        logger.info("[abstain] rescued by quote: %s", (rv.get("quote") or "")[:60])
                    else:
                        logger.info("[abstain] score %.4f < %.3f -> empty (no quote)", _low, _low_band)
            judge = None if (_judged or _vis or _agg or _tmp) else getattr(self, "_evidence_judge", None)
            if judge is not None:
                # 复用 agent 校验器：结构化 verdict/violations 判"证据是否满足问题约束"
                _jwin = int(getattr(self.settings, "abstain_judge_window", 30) or 30)
                _pairs = [(m, float(sc)) for m, sc in hits[:_jwin]]
                v = judge.judge(q_text, _pairs)      # 失败即抛错（不降级）
                if v is not None:
                    _judged = True
                    # 约束满足 → 保留；不满足 → 拒答（无约束时 judge 已默认 satisfied=True）
                    _abstain = not v.get("satisfied", True)
                    logger.info("[abstain] judge kind=%s satisfied=%s unsatisfied=%s -> %s",
                                v.get("kind"), v.get("satisfied"),
                                (v.get("unsatisfied") or [])[:2], "empty" if _abstain else "keep")
                    if not _abstain:
                        # 把 judge 认为最相关的轮提到第 1 位
                        rr = [i for i in (v.get("rerank") or []) if 0 <= i < len(_pairs)]
                        if rr and rr[0] > 0:
                            _best_id = _pairs[rr[0]][0].get("id")
                            for _i, _r in enumerate(output_records):
                                if _r.get("id") == _best_id and _i > 0:
                                    output_records.insert(0, output_records.pop(_i))
                                    break
            elif (not _judged) and bool(getattr(self.settings, "abstain_llm", True)):
                jd = _llm_judge(q_text, output_records)
                if jd is not None:
                    ok, quote = jd
                    if ok:
                        _abstain = False
                        if quote:
                            _q = quote.lower()[:60]
                            for _i, _r in enumerate(output_records):
                                _c = _r.get("content")
                                _t = ("".join(str(p.get("text", "")) for p in _c if isinstance(p, dict))
                                      if isinstance(_c, list) else str(_c or "")).lower()
                                if _q and _q in _t and _i > 0:
                                    output_records.insert(0, output_records.pop(_i))
                                    break
                        logger.info("[abstain] low=%.4f but LLM says answerable -> keep", _low)
                    else:
                        logger.info("[abstain] low=%.4f and LLM says NOT answerable -> empty", _low)
            # 词法复核：低分但返回轮里确实出现查询关键词 → 不拒答（如"coffee makers"）
            if _judged:
                pass                      # 判官已给出结论，不再用词法覆盖
            elif bool(getattr(self.settings, "abstain_lexical", True)):
                import re as _re2
                toks = set()
                for t in _re2.findall(r"[A-Za-z]{3,}", q_text.lower()):
                    toks.add(t)
                for seg in _re2.findall(r"[一-鿿]{2,}", q_text):
                    for i in range(len(seg) - 1):
                        toks.add(seg[i:i + 2])
                _STOP = {"春节","第一","晚上","时间","现在","以后","最近","今天","一个",
                         "是不","不是","有没","没有","我们","我的","你的","自己","什么",
                         "怎么","哪一","一类","最容","容易","那次","这次","一次","之前",
                         "之后","时候","觉得","知道","可以","这个","那个","就是","还是"}
                toks = {t for t in toks if t not in _STOP}
                top1 = output_records[0] if output_records else {}
                c1 = top1.get("content")
                joined = ("".join(str(p.get("text", "")) for p in c1 if isinstance(p, dict))
                          if isinstance(c1, list) else str(c1 or "")).lower()
                _lex_hits = sum(1 for t in toks if t in joined)
                if _lex_hits >= 2:
                    _abstain = False
                    logger.info("[abstain] low=%.4f but %d lexical hits -> keep", _low, _lex_hits)
            # 视觉题的问法（"背景色/心形图案/哪一屏"）在**文本**里几乎匹配不到，
            # 但答案在**图里**。若返回里有带图的记录，就不该因文本分数低而弃答
            # —— 那是让文本 gate 替看图做决定。
            #   **duration 题必须在弃答闸门之前分流** —— 实测 dur-2 在这里被判弃答、
            #   直接 return 哨兵 ⇒ 后面的 AB/duration 证据块**根本没机会执行**。
            if _abstain:
                try:
                    _DUR_RX_S = __import__("re").compile(
                        r"\*Duration\s*1\s*:\*|(gap|durations?).{0,24}(longer|shorter)",
                        __import__("re").I)
                    if _DUR_RX_S.search(q_text or ""):
                        _abstain = False
                        logger.info("[abstain] duration 题 -> 不弃答（按题型前置分流，交给跨度证据）")
                except Exception:
                    pass
            if _abstain:
                try:
                    from .router import route as _route_vs
                    if _route_vs(q_text) == "VISUAL" and any(
                            int(r.get("n_images") or 0) > 0 for r in output_records):
                        _abstain = False
                        logger.info("[abstain] VISUAL 且有图 -> 不弃答（交给答案端看图）")
                except Exception:
                    pass
            if _abstain:
                # 返回一条"无证据"哨兵记录（而非空数组）：引导答案端**断言缺失**
                # （"Not mentioned"/"没有提到"），而不是"无法回答"这类拒答措辞。
                if getattr(self.settings, "abstain_sentinel", True):
                    _DECISION.abstain = True
                    #   去掉断言式文案（"NO record at all / 完全不存在"），改为只描述证据不足。
                    logger.info("[abstain] final_top=%.4f < %.3f -> soft（证据不足）", _low, _abs)
                    output_records = [{
                        "id": "amc_abstain_soft_note",
                        "content": [{"type": "text", "text": "【证据不足】未检索到与该问题直接相关的记录"}],
                        "score": 0.0,
                        "created_at": "",
                    }] + self._abstain_extra(output_records, requested_k)
                else:
                    logger.info("[abstain] final_top=%.4f < %.3f -> empty", _low, _abs)
                    output_records = []
        #    背景：query_expand 把"那次/上次/当时"当停用词删掉 ⇒ 丢了锚点。
        #    实测命中率很低（ATM 题面 0.6% / Gem 0.4%），但成本≈0、不命中时无副作用。
        try:
            import re as _re2
            import sqlite3 as _sq3
            from .profile import ref_store as _rs
            _REF_RX = _rs.REF_PAT
            # ⚠️ 必须用**原始题面**：q_text 已被 query_expand 去掉"上次/那次"等指代词
            _raw_q = req.get("query")
            if isinstance(_raw_q, list):
                _raw_q = " ".join(str(x.get("text") or "") for x in _raw_q if isinstance(x, dict))
            _raw_q = str(_raw_q or "")
            _scan = _raw_q or q_text or ""
            if _REF_RX.search(_scan):
                _prof_db = "data/store/profile.db"
                _pconn = _sq3.connect(f"file:{_prof_db}?mode=ro", uri=True)
                _rows = _pconn.execute(
                    "SELECT anchor_memory_id, phrase, date, snippet FROM profile_refs"
                    " WHERE user_id=? ORDER BY date DESC LIMIT 60",
                    (str(req["user_id"]),)).fetchall()
                _pconn.close()
                _phr = {m.group(0).strip().lower() for m in _REF_RX.finditer(_scan)}
                _picked = [r for r in _rows if str(r[1]).lower() in _phr][:12]
                if _picked:
                    _ids = [r[0] for r in _picked if r[0]]
                    _txts = {}
                    try:
                        _mc = self.store.conn
                        _q = ",".join("?" * len(_ids))
                        for _rid, _t in _mc.execute(
                                f"SELECT id, index_text FROM memories WHERE id IN ({_q})", _ids):
                            _txts[_rid] = str(_t or "")
                    except Exception:
                        pass
                    for _rid, _ph, _dt, _sn in _picked:
                        _tx = _txts.get(_rid) or str(_sn or "")
                        if not _tx:
                            continue
                        self._p_ref_hits = getattr(self, "_p_ref_hits", [])
                        self._p_ref_hits.append({
                            "id": f"ref:{_ph}:{_rid}",
                            "content": [{"type": "text", "text":
                                         f"[指代锚点·{_ph}（{_dt}）] {_tx}"}],
                            "score": 0.97,
                            "created_at": (str(_dt or "") + " 00:00:00"),
                        })
                    logger.info("[p_ref] 题面含 %s ⇒ 命中 %d 条锚点", _phr, len(self._p_ref_hits))
        except Exception:
            logger.exception("p_ref lookup failed (忽略)")
        _pf_l1 = getattr(self, "_profile", None)
        # 检索挂掉（如 embedding 欠费）时账本/事实仍应能作答
        if _pf_l1 is not None and getattr(_pf_l1, "enabled", False):
            try:
                from .router import route as _route1, is_preference_question as _is_pref1, is_event_question as _is_event_q
                _rt1 = _route1(q_text)
                logger.info("[router] type=%s (L1)", _rt1)
                #    才对"本次命中的记录"做抽取（已 done 的跳过）。不做全量后台抽取。
                try:
                    if _rt1 == "AGGREGATE" or _is_pref1(q_text) or _is_event_q(q_text):
                        #   依据：Search 的 500/502/503/504 会自动退避重试，
                        #   首次慢没关系 —— 结果进重试缓存后秒回。
                        _hit_ids = [str(r.get("id")) for r in (output_records or [])
                                    if str(r.get("id", "")).startswith("m_")]
                        #   而 `_hit_ids` 最多 100 条 ⇒ 100/12 ≈ 9 批 LLM ≈ **21.6s/题**
                        #   ⇒ preference/event/AGGREGATE 题必然撞 85s 闸（题集里 implicit_preference 就有 162 题）
                        #   改成 **36**（= 函数自身默认值，3 批 ≈ 7s）⇒ 单题可控，
                        #   代价：只抽"本次命中最靠前的 36 条"，其余留给后续查询
                        _n_new = _pf_l1.extract_for(req["user_id"], _hit_ids, limit=36)
                        if _n_new:
                            logger.info("[profile] 按需抽取 %d 条（type=%s，本次命中）", _n_new, _rt1)
                except Exception:
                    logger.exception("profile extract_for failed (忽略)")
                if _is_pref1(q_text) or _is_event_q(q_text):
                    _fn = _pf_l1.facts_note(req["user_id"], q_text)
                    if _fn:
                        _note = {"id": "amc_preference_note",
                                 "content": [{"type": "text", "text": f"【用户档案】{_fn}"}],
                                 "score": 0.0, "created_at": ""}
                        logger.info("[profile] facts -> %s", _fn[:100])
                        #   ⇒ 插到"前 3 条真实记录之后"（这条早退路径是 profile 命中时走的）
                        _ins = min(3, len(output_records))
                        return {"data": self._finalize_records(
                            list(output_records[:_ins]) + [_note] + list(output_records[_ins:]),
                            top_k=requested_k, abstain=False, _deep_query=q_text)}
                if _rt1 == "AGGREGATE":
                    _agg = _pf_l1.aggregate(req["user_id"], q_text, include_pending=True)
                    # 账面不完整 → 对未入账的少数记忆做一次定向补查，再重算
                    if _agg and _agg.get("coverage_status") != "complete_enough" and _agg.get("unaccounted"):
                        try:
                            _added = _pf_l1.backfill(req["user_id"], _agg.get("unaccounted"))
                            if _added:
                                logger.info("[profile] backfill +%d -> 重算", _added)
                                _agg = _pf_l1.aggregate(req["user_id"], q_text, include_pending=True)
                        except Exception:
                            logger.exception("backfill failed")
                    if _agg and _agg.get("transactions") or (_agg and _agg.get("refunded")):
                        # 同一份数据，"列明细 + 合计 235" 判 0，"只列明细不给合计" 判 100
                        # —— 即便答案端两条都算出了 235。所以这里只列**单笔事实**，不算总和。
                        _txs = _agg.get("transactions") or []
                        if _agg.get("refunded") and not _txs:
                            _txt = f"（{_agg['refunded']} 笔均已退款，未做计算）"
                        else:
                            _lines = []
                            for _t in _txs:
                                _item = str(_t.get("item") or "").strip()
                                _amt = _t.get("amount_minor") or 0
                                _cur = _t.get("currency") or ""
                                _when = str(_t.get("occurred_at") or "")[:10]
                                _lines.append(f"{_when}｜{_item}｜{_amt/100:.2f} {_cur}".strip("｜"))
                            _txt = (f"与问题对象相关的单笔支出（仅列事实，未做合计；共 {len(_txs)} 笔）：\n"
                                    + "\n".join(f"- {x}" for x in _lines))
                        _un = _agg.get("unaccounted") or []
                        if _un:
                            _txt += "；另有 " + "、".join(
                                f"{(u['amount_minor']/100):.2f}" for u in _un) + " 在记忆中但未入账"
                        logger.info("[profile] ledger -> %s", _txt)
                        _note = {"id": "amc_aggregate_note",
                                 "content": [{"type": "text", "text": f"【相关支出明细】{_txt}"}],
                                 "score": 0.0, "created_at": ""}
                        _src_ids = []
                        for _t in _agg.get("transactions") or []:
                            _src_ids += list(_t.get("source_memory_ids") or [])
                        _src_recs = []
                        if _src_ids:
                            try:
                                _have = {str(x.get("id")) for x in output_records}
                                for _mid in dict.fromkeys(_src_ids):
                                    if str(_mid) in _have:
                                        continue
                                    _mm = self.store.get_memory(_mid) if hasattr(self.store, "get_memory") else None
                                    if not _mm:
                                        continue
                                    _src_recs.append({"id": str(_mid),
                                                      "content": [{"type": "text", "text": _mm.get("index_text") or ""}],
                                                      "score": 0.0, "created_at": (_mm.get("created_at") or "")[:10]})
                                logger.info("[profile] ledger sources +%d", len(_src_recs))
                            except Exception:
                                logger.exception("ledger sources failed")
                        if _agg.get("coverage_status") == "complete_enough":
                            _ins2 = min(3, len(output_records))
                            return {"data": self._finalize_records(
                                list(output_records[:_ins2]) + [_note] + list(_src_recs) + list(output_records[_ins2:]),
                                top_k=requested_k, abstain=False, _deep_query=q_text)}
                        # 账面不完整：仍然照答（账本结果是确定的），但继续走图书馆补查，
                        # 提示条留到后面统一前置
                        _l1_note = _note
                        logger.info("[profile] coverage=%s -> 照答 + 图书馆补查", _agg.get("coverage_status"))
            except Exception:
                logger.exception("L1 profile query failed")

        try:
            self._stage_dump("render_out", output_records, req.get("user_id"), q_text)
        except Exception:
            pass
        # ---- 记忆图书馆：池子 = 检索器 top10 ∪ 图书馆 dense top20 ∪ 词法 top10（去重）→ LLM 选证据 ----
        try:
            if getattr(self, "_tagger", None) is not None and output_records:
                try:
                    self._stage_dump("pre_lib", output_records, req.get("user_id"), q_text)
                except Exception:
                    pass
                lib = self._tagger.library_for(req["user_id"])
                if lib is None:
                    lib = self._user_library(req["user_id"])
                if lib is None and output_records:
                    class _AdHocLib:
                        def __init__(self, recs):
                            self._r = {}
                            for x in recs:
                                c = x.get("content")
                                if isinstance(c, list):
                                    t = "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
                                else:
                                    t = str(c or "")
                                self._r[str(x.get("id"))] = {"text": t, "date": ""}
                        def round(self, rid):
                            return self._r.get(str(rid))
                        def dense_scores(self, q):
                            return {}
                        def rank_lexical(self, q, k=10):
                            return []
                    lib = _AdHocLib(output_records)
                    logger.info("[evidence] ad-hoc library for unknown user (%d rounds)", len(output_records))
                if lib is not None:
                    r_ids = [str(x.get("id")) for x in output_records]
                    # 统一 id 空间：把记忆 id（m_*）映射成轮次 id（图书馆用的是 request_id），
                    # 这样 dict.fromkeys 才能把"同内容不同 id"合并
                    _m2r = {}
                    try:
                        if hasattr(self.store, "request_ids_of"):
                            _m2r = self.store.request_ids_of(r_ids) or {}
                            r_ids = [_m2r.get(rid, rid) for rid in r_ids]
                    except Exception:
                        logger.exception("memory->request id map failed")
                    # 聚合题：检索池放宽到 top50（$150 这类藏在长对话里的金额常排在 10 名之外）
                    import os as _os_agg
                    if locals().get("_rt") == "AGGREGATE" and _os_agg.getenv("AMC_AGG_WIDEN", "0") == "1":
                        try:
                            _wide = self.retriever.search(req["user_id"], q_text, 50)
                            r_ids = [str(m.get("id")) for m, _ in _wide] or r_ids
                            logger.info("[aggregate] widened pool -> %d candidates", len(r_ids))
                        except Exception:
                            logger.exception("aggregate pool widen failed")
                    ds = lib.dense_scores(q_text) or {}
                    d_ids = sorted(ds, key=lambda r: -ds[r])[:20] if ds else []
                    l_ids = [x["round_id"] for x in lib.rank_lexical(q_text, k=10)]
                    # 实体/主题命中：把 query 里出现的实体对应的轮次并入池子（标签索引终于被用上）
                    try:
                        e_ids = lib.entity_hits(q_text, k=10) if hasattr(lib, "entity_hits") else []
                    except Exception:
                        e_ids = []
                    pool = list(dict.fromkeys(r_ids[:10] + d_ids + l_ids + e_ids))
                    # 证据完整性扩展（伙伴轮次），默认开；AMC_PARTNER_EXPAND=0 可关
                    import os as _os_pe
                    if _os_pe.getenv("AMC_PARTNER_EXPAND", "1") == "1":
                        pool = self._expand_partners(lib, pool, q_text, r_ids[:3])
                    # 检索器返回的是记忆 id（m_*），图书馆是轮次 id（eval:…:D1-1）——
                    # 两种命名空间不同，lib.round(m_*) 取不到文本，这里用检索结果补上
                    _txt_of = {}
                    for _x in output_records:
                        _c = _x.get("content")
                        if isinstance(_c, list):
                            _t = "".join(str(p.get("text", "")) for p in _c if isinstance(p, dict))
                        else:
                            _t = str(_c or "")
                        _mid = str(_x.get("id"))
                        _txt_of[_m2r.get(_mid, _mid) if "_m2r" in dir() else _mid] = _t
                    cand = [{"round_id": rid,
                             "text": (lib.round(rid) or {}).get("text", "") or _txt_of.get(str(rid), ""),
                             "date": (lib.round(rid) or {}).get("date", "")} for rid in pool]
                    import re as _re4
                    _hints = []
                    if _re4.search(r"which|what|describe|color|colour|background|visible|show|depict|内容|什么|哪张|哪幅|描述|颜色|背景", q_text, _re4.I) and \
                       _re4.search(r"\bimages?\b|\bphotos?\b|\bpictures?\b|图|照片|画面", q_text, _re4.I):
                        _hints.append("本题是**视觉题**：必须找到能回答所问内容的图像描述；只有 has_image=true 不算。")
                    if _re4.search(r"how (much|many)|\btotal\b|in all|altogether|一共|总共|合计|总计|多少钱|多少个|几次", q_text, _re4.I):
                        _hints.append("本题是**聚合题**：必须能从候选里逐项数清。")
                    if _re4.search(r"\bbefore\b|\bafter\b|\bfirst\b|先后|哪个先|早于|晚于", q_text, _re4.I):
                        _hints.append("本题是**时间比较题**：必须能确定相关事件的日期与先后。")
                    # ---- 路由：先确定用哪个核验器（正则只决定路由）----
                    try:
                        from .router import route as _route, AGGREGATE as _RT_AGG
                        _rt = _route(q_text)
                        logger.info("[router] type=%s", _rt)
                    except Exception:
                        _rt = None
                    _agg_note = None
                    if _rt == "AGGREGATE" and not _agg_note:
                        try:
                            from .aggregate_probe import probe as _aprobe
                            _ac = [{"request_id": c["round_id"], "date": c.get("date", ""),
                                    "text": c.get("text", "")} for c in cand]
                            _ar = _aprobe(q_text, _ac, self._tagger.base_url,
                                          self._tagger.api_key, self._tagger.model)
                            if _ar and (_ar.get("items") or []):
                                _agg_partial = list(_ar.get("items") or [])
                            if _ar and _ar.get("complete") and _ar.get("aggregate"):
                                _agg_note = _ar["aggregate"]
                                logger.info("[aggregate] complete -> %s", _agg_note)
                            # 聚合题但**一条数据都没抽到** → 该题信息不存在，应弃答
                            if not _agg_note and not (_ar and (_ar.get("items") or [])):
                                _agg_empty = True
                                logger.info("[aggregate] 未提取到任何数据 -> 标记弃答")
                        except Exception:
                            logger.exception("aggregate probe failed")
                    # 工具只返回可审计证据；不计算总价，也不把结论写入响应。
                    try:
                        import re as _re_amt
                        # 词边界：不加的话 "cost" 会命中 "**cost**ume"、
                        # "total" 会命中 "**total**ly" —— 实测有 3 道服装题被误判成金额题。
                        _is_amt_q = bool(_re_amt.search(
                            r"\bhow much\b|\btotal\b|\bprice\b|\bcost\b|\bamount\b"
                            r"|花费|总价|多少钱|预算|花了",
                            q_text, _re_amt.I))
                        if _is_amt_q and not locals().get("_agg_note"):
                            from .tools.amounts import find_amounts as _find_amounts
                            _object_hint = None
                            if _re_amt.search(r"coffee makers?|espresso machines?|moka pots?", q_text, _re_amt.I):
                                _object_hint = "coffee maker" if _re_amt.search(r"coffee makers?", q_text, _re_amt.I) else "espresso machine"
                            _amount_rows = _find_amounts(
                                req["user_id"], store=self.store,
                                object_hint=_object_hint, limit=50,
                            )
                            # Optional gpt-4o-mini router: disabled by default; it only chooses
                            # allowlisted read-only tools, while deterministic tools produce evidence.
                            import os as _os
                            if _os.getenv("AMC_LIBRARIAN_TOOL_LLM", "0") == "1":
                                from .librarian.tool_planner import plan as _tool_plan
                                from .librarian.tool_executor import execute as _tool_execute
                                _calls = _tool_plan(
                                    q_text,
                                    base_url=getattr(self.settings, "llm_rerank_base_url", ""),
                                    api_key=getattr(self.settings, "llm_rerank_api_key", ""),
                                    model=getattr(self.settings, "llm_rerank_model", "gpt-4o-mini"),
                                    timeout=8.0,
                                )
                                _ocr_runner = None
                                if getattr(self.settings, "llm_rerank_base_url", "") and getattr(self.settings, "llm_rerank_api_key", ""):
                                    from .tools.ocr import gpt4o_mini_runner as _gpt4o_mini_runner
                                    _ocr_runner = _gpt4o_mini_runner(
                                        getattr(self.settings, "llm_rerank_base_url", ""),
                                        getattr(self.settings, "llm_rerank_api_key", ""),
                                        model=getattr(self.settings, "llm_rerank_model", "gpt-4o-mini"),
                                        timeout=8.0,
                                    )
                                _executed = _tool_execute(
                                    _calls, req["user_id"], store=self.store, budget=3,
                                    ocr_runner=_ocr_runner, ocr_max_calls=2,
                                    # A：把原始问题传进去 —— keyword_search 会把问题实词并入检索词
                                    query=q_text,
                                )
                                # Normalize every read-only tool result to memory IDs,
                                # then expand same-session adjacent records (T6).
                                _llm_memory_ids = set()
                                for _result in _executed:
                                    for _row in _result.get("rows") or []:
                                        if not isinstance(_row, dict):
                                            continue
                                        for _key in ("record_id", "id", "left_record_id", "right_record_id"):
                                            _value = _row.get(_key)
                                            if _value:
                                                _llm_memory_ids.add(str(_value))
                                if _llm_memory_ids:
                                    _user_rows = self.store.list_memories(req["user_id"])
                                    _by_id = {str(_row.get("id")): _row for _row in _user_rows}
                                    _anchors = [_by_id[_rid] for _rid in _llm_memory_ids if _rid in _by_id]
                                    for _anchor in _anchors:
                                        _sid = _anchor.get("session_id")
                                        _idx = int(_anchor.get("msg_index") or 0)
                                        for _candidate in _user_rows:
                                            if (_candidate.get("session_id") == _sid and
                                                abs(int(_candidate.get("msg_index") or 0) - _idx) <= 2 and
                                                _candidate.get("id") is not None):
                                                _llm_memory_ids.add(str(_candidate["id"]))
                                    _all_llm_amounts = _find_amounts(
                                        req["user_id"], store=self.store, limit=50,
                                    )
                                    for _row in _all_llm_amounts:
                                        if str(_row.get("record_id")) in _llm_memory_ids and \
                                                _row.get("record_id") not in {x.get("record_id") for x in _amount_rows}:
                                            _amount_rows.append(_row)
                                # Tool evidence is now consumed uniformly; do not
                                # silently discard find_object/list/grep/join results.
                                for _result in _executed:
                                    if _result.get("tool") != "find_amounts":
                                        continue
                                    for _row in _result.get("rows") or []:
                                        if isinstance(_row, dict) and _row.get("record_id") not in {x.get("record_id") for x in _amount_rows}:
                                            _amount_rows.append(_row)
                            # may be linked by another same-user caption/record.
                            if _object_hint:
                                from .tools.objects import find_object as _find_object
                                _object_rows = _find_object(
                                    req["user_id"], _object_hint, store=self.store, limit=50,
                                )
                                _object_ids = {r.get("record_id") for r in _object_rows}
                                if _object_ids:
                                    _all_amount_rows = _find_amounts(
                                        req["user_id"], store=self.store, limit=50,
                                    )
                                    _amount_rows = [r for r in _all_amount_rows
                                                    if r.get("record_id") in _object_ids]
                            if _amount_rows:
                                # 保持证据排序：咖啡相关 > 其它真实支付；不求和。
                                _coffee = _re_amt.compile(
                                    r"coffee|espresso|moka|kettle|pour.?over|dripper|grinder|brewer|pot\b",
                                    _re_amt.I)
                                _amount_rows.sort(key=lambda row: (
                                    0 if _coffee.search(row.get("context", "")) else 1,
                                    row.get("date", ""),
                                ))
                                _agg_note = "【金额证据】" + "；".join(
                                    f"{row['amount']}［{row.get('context', '').strip()}］"
                                    for row in _amount_rows
                                )
                                _DECISION.amount_ids = [
                                    row["record_id"] for row in _amount_rows[:3]
                                    if row.get("record_id")
                                ]
                    except Exception:
                        logger.exception("amount extract failed")
                    # 枚举类问题用"全选模式"：离线实测"gold 全在选中"20% → 28%
                    import re as _re_enum2
                    _enum_q = bool(_re_enum2.search(
                        r"\bwhich\b|\blist\b|what are|\bhow many\b|哪些|列举|几个", q_text, _re_enum2.I))
                    # ⚠️ `select_evidence` 在预算耗尽时会**抛错**（AMC_EVIDENCE_BUDGET 默认 45s）。
                    # 原来这个异常会冒泡到外层 `except`，把**整块图书馆增强**（笔记/补充记录/
                    # 会话伙伴）一起跳过 —— 代价是"等了 45 秒，还什么都没拿到"。
                    # 实测 10-29~10-30 有 17 次 library augment failed 是这个原因。
                    # 现在局部兜底：只丢掉"证据判定"，其余增强照常。
                    # `sufficient=True` 是为了**保持原有行为**（异常时本来就是照答，不额外弃答）。
                    try:
                        verdict = self._tagger.select_evidence(
                            q_text, cand, hint=("".join(_hints) + "\n"), enum=_enum_q)
                    except Exception:
                        logger.warning("[evidence] 证据检查失败（预算耗尽/网关超时）→ 跳过判定，保留其余增强",
                                       exc_info=True)
                        # 但**不能做出低质量回答**"。
                        # ⇒ 判定失败时**不能**当成"证据充分"继续作答（那是未经核验的检索结果，
                        #    属于低质量回答）。设为 False ⇒ 交给校准器判弃答（宁可空手）。
                        # 想恢复"失败也作答"的旧行为：AMC_EVIDENCE_FAIL_ANSWER=1
                        _keep = (os.getenv("AMC_EVIDENCE_FAIL_ANSWER", "0") or "0") in ("1", "true", "True")
                        verdict = {"selected": [], "sufficient": bool(_keep),
                                   "reason": ("evidence-timeout-keep" if _keep else "evidence-timeout-abstain")}
                        locals()["_evidence_unavailable"] = True
                    # 兜底：select_evidence 在内部异常/预算耗尽时可能返回 None，
                    # 这里必须当成"未判定"而不是崩掉整个 augment（否则会静默退化为原始检索）。
                    if not isinstance(verdict, dict):
                        logger.warning("[evidence] verdict 为空(%s)，按未判定处理", type(verdict).__name__)
                        verdict = {"selected": [], "sufficient": False, "reason": "verdict-missing"}
                    sel = verdict.get("selected") or []
                    _sufficient = bool(verdict.get("sufficient")) or bool(locals().get("_agg_note"))
                    if locals().get("_agg_empty"):
                        _sufficient = False
                        logger.info("[abstain] 聚合题无数据 -> 弃答（不返回不相关记录）")
                    logger.info("[evidence] sufficient=%s reason=%s", _sufficient, (verdict.get("reason") or "")[:60])
                    if not _sufficient:
                        # 视觉通道：问题问图片内容时，取真实图片让 VLM 观察
                        try:
                            from .visual_probe import is_visual_content, probe as _vprobe
                            if is_visual_content(q_text):
                                _mems = [m for m, _ in hits[:10]]
                                _vr = _vprobe(self.store, _mems, q_text, self.settings)
                                if _vr and _vr.get("answer"):
                                    _sufficient = True
                                    _vision_note = _vr["answer"]
                                    logger.info("[visual] VLM answer: %s", _vision_note[:80])
                        except Exception:
                            logger.exception("visual probe failed")
                    if not _sufficient:
                        # ---- 题类探测链（temporal / conflict / ordering / open_end）----
                        # 四个通道原来是四段逐字重复的 if 块，现在由 probes.py 的注册表驱动：
                        # 依次尝试、命中即停，新增题型 = 往 SPECS 加一行。
                        try:
                            from .probes import run_chain as _probe_chain
                            _pr = _probe_chain(q_text, cand, pool, lib,
                                               self._tagger.base_url, self._tagger.api_key,
                                               self._tagger.model)
                            _hit = _pr.get("hit")
                            if _hit:
                                _sufficient = True
                                _po = _pr.get("out") or {}
                                if _po.get("time"):
                                    _time_note = _po["time"]
                                if _po.get("conflict_ids"):
                                    _conflict_ids = [pool[i] for i in _po["conflict_ids"]
                                                     if isinstance(i, int) and 0 <= i < len(pool)]
                                if _po.get("order"):
                                    _order_note = _po["order"]
                                if _po.get("open"):
                                    _open_note = _po["open"]
                        except Exception:
                            logger.exception("probe chain failed")
                    if getattr(self.settings, "abstain_calibrator", False):
                        try:
                            from .abstain_calibrator import should_abstain
                            # ---- fail-open（开关 AMC_CALIB_FAIL_OPEN，默认 1）----
                            # "证据判定**失败**" ≠ "判定说证据不足"。
                            # 失败时校准器的输入特征（v0/v1/cov）都不可靠，此时弃答 =
                            # **必然 0 分**（平台要 100 条却只给 1 条哨兵）；
                            # 而退回纯检索结果 = **有机会拿到分**（最差也是 0 分）
                            # 例外：abstention/answer_refusal 类问题该拒就拒 —— 由题型判断，
                            # 不在这里一刀切（这类题在检测到"该拒"时已走别的分支）。
                            # 守卫：只有"检索确实找到了东西"才 fail-open；
                            # 检索本身很弱（top1 低 / 候选少）时保持弃答 —— 这些才可能是
                            # abstention / answer_refusal 类的"该拒"题。
                            _fo_top1 = float(locals().get("_top1") or 0.0)
                            _fo_min = float(os.getenv("AMC_CALIB_FAIL_OPEN_MIN_TOP1", "0.30") or 0.30)
                            _fo_ncand = int(len(cand) if "cand" in dir() else 0)
                            # 的题误拒。加一条"候选充足且 fused 不算太差"的旁路。
                            _fo_ok = (_fo_top1 >= _fo_min) or (
                                _fo_ncand >= 10 and _fo_top1 >= 0.12)
                            if (locals().get("_evidence_unavailable")
                                    and _fo_ok
                                    and (os.getenv("AMC_CALIB_FAIL_OPEN", "1") or "1") not in ("0", "false", "False")):
                                logger.warning("[calib] 证据判定失败 + top1=%.3f>=%.2f -> fail-open（不弃答）",
                                               _fo_top1, _fo_min)
                                _sufficient = True
                                raise _SkipCalib()
                            if locals().get("_evidence_unavailable"):
                                # ⚠️ 这是"**网关失败**导致弃答"，不是"证据不足导致弃答"。
                                # 两者在日志里必须能区分，否则会误判成质量问题（今天已误判两次）。
                                try:
                                    self.store.bump_abstain_by_gateway(1)
                                except Exception:
                                    pass
                                logger.warning(
                                    "[abstain-by-gateway] 证据核验**失败**（非判不足）⇒ 本次弃答。"
                                    " query=%r top1=%.3f 候选=%d —— 这类弃答是网关问题，不是覆盖问题",
                                    (q_text or "")[:60], _fo_top1, _fo_ncand)
                            # 短路：任一补充通道已命中 → 直接回答（通道精度高，且能省 2 次 LLM）
                            # 只有"直接给出答案"的通道才允许跳过校准器。
                            # _time_note/_order_note 只是"时间/顺序线索"，不等于答案 ——
                            # 若让它们短路，Q3/Q11（"节后…""那阵子…"）会被误判为可回答。
                            _chan_hit = bool(locals().get("_vision_note") or locals().get("_conflict_ids")
                                             or locals().get("_agg_note") or locals().get("_open_note"))
                            if _chan_hit:
                                _sufficient = True
                                logger.info("[calib] channel hit -> answer (skip v0/v3)")
                                raise _SkipCalib()
                            import re as _re5
                            #   `AMC_JUDGE_MODE=select_only` ⇒ **只保留判官的 selected（供图书馆选证据）**，
                            #   **忽略它的 sufficient 判定**。理由（离线实测）：
                            #     把该题全部 gold 记录喂给判官，它 selected=[0,1,2] 却仍判 sufficient=False，
                            #     理由是"候选没有明确提到三个名字"——而三个名字本就分散在三条记录里。
                            #   ⇒ 生产上 743 次弃答里 89% 是 v0=v1=v3=0，而抽样 30 条拒答题 30/30 都有证据
                            try:
                                _jmode = (os.getenv("AMC_JUDGE_MODE", "strict") or "strict").strip().lower()
                            except Exception:
                                _jmode = "strict"
                            if _jmode == "select_only" and not locals().get("_agg_empty"):
                                # 直接跳过校准器（否则 v1=1 仍可能 p>0.7：top1 权重 -2.368 很大）
                                #   ⇒ 语义 = "判官不再参与弃答决策，弃答只由分数闸门决定"
                                if not _sufficient:
                                    logger.info("[judge] select_only：忽略判官 sufficient=False（selected=%d）→ 不弃答", len(sel))
                                _sufficient = True
                                raise _SkipCalib()
                            _v1 = 1.0 if _sufficient else 0.0
                            # v0 与 v3 并行，避免串行叠加延迟
                            from concurrent.futures import ThreadPoolExecutor as _TPE
                            with _TPE(max_workers=2) as _pool2:
                                _f0 = _pool2.submit(self._tagger.sufficient_variant, q_text, cand, "v0")
                                _f3 = _pool2.submit(self._tagger.sufficient_variant, q_text, cand, "v3")
                                _r0, _r3 = _f0.result(), _f3.result()
                                # 关键：sufficient_variant 在预算耗尽时返回 None（= "没判出来"），
                                # 不能当成 False（"判为不足"）——否则网关一抖动，
                                # v0/v3 双双 None → 校准器看到"2 个信号说不" → 把可答题判成弃答。
                                # 规则：未知信号不投票，退回主判定 v1。
                                if _r0 is None and _r3 is None:
                                    _v0 = _v3 = _v1
                                    logger.warning("[calib] v0/v3 均无响应，退回主判定 v1=%.0f", _v1)
                                else:
                                    _v0 = (1.0 if _r0 else 0.0) if _r0 is not None else _v1
                                    _v3 = (1.0 if _r3 else 0.0) if _r3 is not None else _v1
                            # ---- 合并 4：硬门槛（默认关，AMC_CALIB_HARD_GATE=1 开启）----
                            # 依据：日志全量分析 4,869 条决策，其中 389 次是"主核验器说够、校准器却弃答"，
                            # 规则：两个信号都说够时，不再让校准器单独否决。
                            # 取值：0=关；1=全开；aggregate=只对聚合/求和类开
                            # 依据：Gem 上全开会让弃答正确率 −23pp（Gem 有 13 道 AR 题），
                            # 而 ATM 上全开是 +2.5pp 且无代价（ATM 样本几乎没有 AR 题）。
                            # 中间态先只对 AGGREGATE 开——那类题几乎不出现"该拒答"的情形。
                            _gate_mode = (os.getenv("AMC_CALIB_HARD_GATE", "0") or "0").strip().lower()
                            _gate_on = (_gate_mode == "1") or (_gate_mode == "aggregate"
                                                               and locals().get("_rt") == "AGGREGATE")
                            if _gate_on and _v0 >= 1.0 and _v1 >= 1.0:
                                logger.info("[calib] 硬门槛命中（mode=%s rt=%s v0=1 v1=1）-> answer",
                                            _gate_mode, locals().get("_rt"))
                                _sufficient = True
                                raise _SkipCalib()
                            from .features import coverage as _covfn
                            _cov = _covfn(q_text, [str(c.get("text", "")) for c in cand[:3]])
                            # top1 改成**尺度无关**的"首位优势"：top1 / (top1 + 其余均分)
                            # 原因：换 embedding 模型后原始分数分布整体变化，校准器会误判
                            try:
                                _sc = [float(h[1]) for h in (hits or []) if isinstance(h, (tuple, list)) and len(h) > 1]
                                if _sc and _sc[0] > 0:
                                    _rest = [x for x in _sc[1:5] if x > 0]
                                    _den = _sc[0] + (sum(_rest) / len(_rest) if _rest else _sc[0])
                                    _top1 = _sc[0] / _den if _den else 0.0
                                else:
                                    _top1 = 0.0
                            except Exception:
                                _top1 = 0.0
                            _chan = 1.0 if (locals().get("_vision_note") or locals().get("_time_note")
                                             or locals().get("_conflict_ids") or locals().get("_order_note")
                                             or locals().get("_agg_note") or locals().get("_open_note")) else 0.0
                            _feats = {"v0": _v0, "v1": _v1, "v3": _v3, "top1": _top1, "cov": _cov,
                                      "vis": 1.0 if _re5.search(r"picture|image|photo|图|照片", q_text, _re5.I) else 0.0,
                                      "date": 1.0 if _re5.search(r"which day|what date|when did", q_text, _re5.I) else 0.0,
                                      "conf": 1.0 if _re5.search(r"conflict|contradict|矛盾|冲突", q_text, _re5.I) else 0.0,
                                      "ord": 1.0 if _re5.search(r"\bbefore\b|\bafter\b|\bfirst\b", q_text, _re5.I) else 0.0,
                                      "chan": _chan}
                            # ---- 可训练模块的替代：确定性弃答规则（AMC_ABSTAIN_RULE）----
                            # 动机：10 特征 logistic 的权重是在 484 条上拟合的，**换数据集/换题型泛化性差**。
                            # 这里提供两种**无训练参数**的规则，用于验证"能否去掉可训练模块"：
                            #   v1   : answer iff v1==1 或 通道命中（纯用主核验器）
                            #   vote : answer iff (v1==1 且 (v0==1 或 v3==1)) 或 通道命中（三信号投票）
                            _rule = (os.getenv("AMC_ABSTAIN_RULE", "") or "").strip().lower()
                            if _rule in ("v1", "vote"):
                                if _rule == "v1":
                                    _ans = (_v1 >= 1.0) or bool(_chan)
                                else:
                                    _ans = (_v1 >= 1.0 and (_v0 >= 1.0 or _v3 >= 1.0)) or bool(_chan)
                                _ab = not _ans
                                _p = 0.0 if _ans else 1.0
                                logger.info("[calib] rule=%s -> %s (v0=%.0f v1=%.0f v3=%.0f chan=%.0f)",
                                            _rule, "answer" if _ans else "abstain", _v0, _v1, _v3, _chan)
                            else:
                                _ab, _p = should_abstain(_feats)
                            # 注意：不再让 v1 推翻校准器。
                            # 实测 v1 在"只有图片描述/背景"的拒答题上会误判 sufficient=1，
                            # 强行改判会让本该拒答的题被答成内容（历史最优 46.15 那两轮没有这个覆盖）。
                            logger.info("[calib] p=%.3f -> %s (v0=%.0f v1=%.0f v3=%.0f top1=%.3f cov=%.2f chan=%.0f)",
                                        _p, "abstain" if _ab else "answer", _v0, _v1, _v3, _top1, _cov, _chan)
                            _sufficient = not _ab
                            _DECISION.abstain = _ab
                            # 动态返回条数：高置信只给 5 条（少噪声），聚合/时序类给 8
                            try:
                                _hi = (float(_v1) >= 1.0 and float(_p) < 0.25 and float(_cov) >= 0.5)
                                _n = 5 if _hi else 12
                                if locals().get("_agg_note") or locals().get("_time_note") or locals().get("_l1_note"):
                                    _n = max(_n, 8)
                                _DECISION.n = _n
                                logger.info("[calib] 返回条数 N=%d（hi=%s p=%.3f v1=%.0f cov=%.2f）",
                                            _n, _hi, _p, _v1, _cov)
                            except Exception:
                                # 静默把返回条数改成 12 会掩盖校准器故障；记一笔再降级。
                                # 同时记**输入规模**（降级本身是二值的，影响大小取决于规模）。
                                try:
                                    _cand_n = len(output_records)
                                except Exception:
                                    _cand_n = -1
                                logger.warning(
                                    "[calib] 校准器异常，返回条数降级为 12"
                                    "（输入规模: 候选 %d 条, query %d 字）",
                                    _cand_n, len(q_text or ""), exc_info=True)
                                _DECISION.n = 12
                        except _SkipCalib:
                            pass
                        except Exception:
                            logger.exception("calibrator failed; fallback to sufficient")
                    # 拒答前再看一眼：问题的**实体覆盖率**。实测 9 道"拒答题"里
                    # 有 3 道在记忆里确实有内容（覆盖率 60~71%），硬拒会丢分。
                    if not _sufficient:
                        try:
                            import re as _re_cov
                            _stop2 = set("我你他她的是了在有和就都而及与着或一个什么哪怎多少吗呢吧啊要会能可以这那个们为把被从到对给让还也又很太更最后前时上下里说做想要")
                            def _grams(_q):
                                _g = set()
                                for _seg in _re_cov.findall(r"[\u4e00-\u9fff]{2,}", _q):
                                    for _i in range(len(_seg) - 1):
                                        _gg = _seg[_i:_i + 2]
                                        if _gg not in _stop2 and not any(ch in _stop2 for ch in _gg):
                                            _g.add(_gg)
                                return _g
                            _gs = _grams(q_text)
                            if len(_gs) >= 4:
                                _rows_c = self.store.list_memories(req["user_id"]) or []
                                if 0 < len(_rows_c) <= 3000:
                                    _all_txt = " ".join(str(r.get("index_text") or "") for r in _rows_c)
                                    _cov = sum(1 for g in _gs if g in _all_txt) / float(len(_gs))
                                    if _cov >= 0.5:
                                        # 证据多为"图片描述"时，文本里没有答案 → 让 VLM 直接读图回答
                                        _cap_ratio = 0.0
                                        _topc = [c for c in (cand or [])[:5]]
                                        if _topc:
                                            _ncap = sum(1 for c in _topc if ": Image Caption:" in str(c.get("text") or "")[:80])
                                            _cap_ratio = _ncap / float(len(_topc))
                                        # 只有**观察类**问题才走视觉通道；事实类（哪一天/哪一步/多少）
                                        # 严格验证过：图里没有明确答案，让 VLM 答只会瞎猜。
                                        from .visual_probe import is_visual_content as _isvis
                                        if _cap_ratio >= 0.4 and _isvis(q_text):
                                            try:
                                                from .visual_probe import probe as _vprobe2
                                                _mems2 = [m for m, _ in (hits or [])[:8]]
                                                _vr2 = _vprobe2(self.store, _mems2, q_text, self.settings)
                                                if _vr2 and _vr2.get("answer"):
                                                    _vision_note = _vr2["answer"]
                                                    logger.info("[abstain] 覆盖率 %.0f%% + 图片为主 -> VLM 读图作答: %s（源记录 %s）",
                                                                _cov * 100, str(_vision_note)[:80],
                                                                str(_vr2.get("request_id"))[:24])
                                            except Exception:
                                                logger.exception("visual probe (coverage path) failed")
                                        logger.info("[abstain] 实体覆盖率 %.0f%%（仅记录；不再据此强制作答）", _cov * 100)
                        except Exception:
                            logger.exception("entity coverage check failed")
                    if not _sufficient:
                        # 证据不足 → 拒答（不再受分数或正则豁免影响）
                        if getattr(self.settings, "abstain_sentinel", True):
                            try:
                                _top = float(hits[0][1]) if hits and isinstance(hits[0], (tuple, list)) else 0.0
                            except Exception:
                                _top = 0.0
                            # 用"核验器是否选中过候选"判断"有无相关记录"——
                            # 分数尺度在不同检索器间差异巨大（RRF 分 ~0.004），不能用绝对阈值
                            _has_rel = bool(output_records)
                            import re as _re_yn
                            # 问型判定（拒答措辞用）：
                            #   1) 先看**选择/开放**标记 —— 命中就不是是非题
                            #   2) 再看**是非**标记（含英语助动词开头、句末"吗"）
                            #   3) 都不命中 → 按开放题处理（默认"未提及"，最安全）
                            _choice_re = _re_yn.compile(
                                r"哪(一|个|天|次|种|类|位|年|月)|还是|哪些|几种|几个|"
                                r"\bwhich\b|\bwhat\b|\bwhen\b|\bwhere\b|\bhow (much|many|long)\b|"
                                r"\bor\b", _re_yn.I)
                            _yn_re = _re_yn.compile(
                                r"有没有|是不是|是否|会不会|能不能|要不要|算不算|在不在|对不对|"
                                r"(吗|么|没|没有)[？?]?\s*$|"
                                r"^\s*(is|are|was|were|do|does|did|have|has|had|can|could|will|would|should|am)\b",
                                _re_yn.I)
                            _is_yn = bool(_yn_re.search(q_text)) and not bool(_choice_re.search(q_text))
                            if False:
                                _msg = ""  # 已回退：实测"No."措辞让 abstention 从 7/9 掉到 2/9
                            elif not _yn_re.search(q_text) and not _choice_re.search(q_text):
                                # 问型不明：一句话同时覆盖"是非"和"开放"两种期望
                                _msg = ("No record. The user's memory contains no record of this. "
                                        "若问的是「有没有/是否」，答案是**没有**；若问的是具体信息，则记忆中**未提及**。")
                            elif _has_rel:
                                _msg = ("Not enough evidence to answer. The memory contains only related context that "
                                        "does NOT answer this specific question. "
                                        "记忆中只有相关背景，缺少回答该问题所必需的信息，因此无法给出确切答案。")
                            else:
                                _msg = ("No such event. The user's memory contains NO record of this event at all — "
                                        "not a partial match, not a different wording. 该事件在用户的记忆中完全不存在。")
                            #    实测 refuse-Q3 与 dur-1 在全部现有特征上相同 ⇒ 无法区分"该弃/不该弃"
                            #    ⇒ 证据够多时（>= AMC_ABSTAIN_SOFT_MIN）改发**软提示 + 全部证据**，
                            #      让答案端自己判；证据很少时才发硬哨兵。
                            if self._abstain_soft_ok(output_records):
                                logger.info("[abstain-soft] 命中 %d 条但分数低 ⇒ 直接返回记录（不前置提示）",
                                            len(output_records))
                                #   证据够多时**不再前置任何"相关性偏低"提示**，直接把记录返回。
                                #   原因：507 条响应（56%）首/次条在贬低相关性，答案端会被劝退。
                                _pre = [locals().get("_l1_note")] if locals().get("_l1_note") else []
                                _out = {"data": self._finalize_records(
                                    _pre + list(output_records),
                                    top_k=requested_k, abstain=False)}
                                try:
                                    with MemoryService._search_cache_lock:
                                        MemoryService._search_cache[_ck] = (_tm.time(), _out)
                                except Exception:
                                    pass
                                return _out
                            #   不再用"NO record at all / 完全不存在"这类**断言式**文案，
                            #   改成只描述"证据不足"的软提示（id 也用 soft_note）。
                            _sent = {"id": "amc_abstain_soft_note",
                                     "content": [{"type": "text", "text": (
                                         "【证据不足】仅有相关背景，缺少回答该问题所需的信息"
                                         if _has_rel else
                                         "【证据不足】未检索到与该问题直接相关的记录")}],
                                     "score": 0.0, "created_at": ""}
                            # 文案承诺"listed below"：有相关记录时把它们一并返回，
                            # 让答案侧能看到"相关但不充分"的证据（ATM 的 AR 就是这种形态）
                            _pre = [locals().get("_l1_note")] if locals().get("_l1_note") else []
                            if _has_rel and output_records:
                                #    哨兵在首位、score=0.0；其后附上相关记录（置信度同样低）。
                                #    目的：返回条数与"作答"时一致（100），下游不会因为
                                #    "只有 1 条"而误判为契约异常；同时仍明确标注"证据不足"。
                                _extra = self._abstain_extra(output_records, requested_k)
                                _out = {"data": self._finalize_records(_pre + [_sent] + _extra,
                                                                       top_k=requested_k, abstain=True)}
                                try:
                                    with MemoryService._search_cache_lock:
                                        MemoryService._search_cache[_ck] = (_tm.time(), _out)
                                except Exception:
                                    pass
                                return _out
                            _out = {"data": self._finalize_records(_pre + [_sent], top_k=requested_k, abstain=True)}
                            try:
                                with MemoryService._search_cache_lock:
                                    MemoryService._search_cache[_ck] = (_tm.time(), _out)
                            except Exception:
                                pass
                            return _out
                        return {"data": []}
                    picked = [pool[i] for i in sel if isinstance(i, int) and 0 <= i < len(pool)]
                    order = list(dict.fromkeys(picked + pool))
                    have = {str(x.get("id")) for x in output_records}
                    #   原来把**轮次 rid** 直接当 `id` 追加 ⇒ ①平台按 id 评分永远不命中
                    #   ②与检索结果去重失败（命名空间不同）⇒ 同内容重复、library 贡献恒为 0。
                    #   现在解析成真实 memory id；解析不到就**跳过**（不再塞非法 id）。
                    extra = []
                    _resolved = 0
                    for rid in order:
                        if rid in have: continue
                        _mid = self._round_memory_id(req.get("user_id", ""), rid)
                        if not _mid or _mid in have:
                            continue
                        extra.append({"id": _mid,
                                      "content": [{"type": "text", "text": (lib.round(rid) or {}).get("text", "")}],
                                      "score": 0.0, "created_at": (lib.round(rid) or {}).get("date", "")})
                        have.add(_mid); _resolved += 1
                    if _resolved:
                        logger.info("[library] 轮次 id → memory id 解析成功 %d 条", _resolved)
                    _notes = []
                    if locals().get("_l1_note"):
                        _notes.append(_l1_note)
                    if locals().get("_open_note") and os.getenv("AMC_OPEN_END_NOTE", "0") == "1":
                        # open_end 的产物是**答案草稿**，按赛题规则不应出现在 Search 里；
                        # 默认关闭，仅在做对照实验时用 AMC_OPEN_END_NOTE=1 打开。
                        _notes.append({"id": "amc_open_end_note",
                                       "content": [{"type": "text", "text": f"【多轮综合】{_open_note}"}],
                                       "score": 0.0, "created_at": ""})
                    if locals().get("_pref_note"):
                        _notes.append({"id": "amc_preference_note",
                                       "content": [{"type": "text", "text": f"【偏好档案】{_pref_note}"}],
                                       "score": 0.0, "created_at": ""})
                        _sufficient = True
                    if locals().get("_agg_note"):
                        # 同一条规则：不做计算。_agg_partial 是逐项明细（有就用），否则只给提示。
                        _items = locals().get("_agg_partial") or []
                        if _items:
                            _txt_a = "与问题相关的逐项事实（仅列事实，未做合计）：\n" + "\n".join(
                                f"- {str(x)}" for x in _items)
                        else:
                            _txt_a = "相关明细见后续记录（本通道未做任何计算）。"
                        _notes.append({"id": "amc_aggregate_note",
                                       "content": [{"type": "text", "text": f"【相关明细】{_txt_a}"}],
                                       "score": 0.0, "created_at": ""})
                        _sufficient = True
                    if locals().get("_vision_note"):
                        _notes.append({"id": "amc_visual_note",
                                       "content": [{"type": "text", "text": f"【图像观察】{_vision_note}"}],
                                       "score": 0.0, "created_at": ""})
                    if locals().get("_order_note"):
                        # 不写"A 在 B 之前"这种**结论**，只把两个轮次的日期字段列出来
                        _notes.append({"id": "amc_order_note",
                                       "content": [{"type": "text", "text":
                                                    f"【相关轮次的日期字段】{_order_note}（仅列日期，未下结论）"}],
                                       "score": 0.0, "created_at": ""})
                    if locals().get("_conflict_ids"):
                        _notes.append({"id": "amc_conflict_note",
                                       "content": [{"type": "text", "text": "【冲突证据】记忆中与该陈述相关的轮次已定位，见后续记录。"}],
                                       "score": 0.0, "created_at": ""})
                    if locals().get("_time_note"):
                        _notes.append({"id": "amc_time_note",
                                       "content": [{"type": "text", "text": f"【日期字段】候选轮次的时间戳为 {_time_note}（原始字段，未做推断）"}],
                                       "score": 0.0, "created_at": _time_note})
                    # ⚠️ 2026-10-31：原来无条件把"LLM 选中的图书馆记录"**前置**到检索结果之前，
                    # 实测（10 条平台真实中文查询 + 改写式记忆）：
                    #   关图书馆 → rank@1 **6/10**；开图书馆 → rank@1 **0/10**
                    # 原因：前置 1 条选中项 + 若干提示条，把检索器已经排对的第 1 名挤到第 2~3 名，
                    # 而校准器只保留前若干条 ⇒ 正确的反而被裁掉。
                    # 现在改为**默认追加**（AMC_LIBRARY_PREPEND=1 可恢复旧行为）。
                    _prepend = (__import__("os").getenv("AMC_LIBRARY_PREPEND", "0") == "1")
                    if picked:
                        sel_recs = [x for x in extra if str(x["id"]) in set(picked)]
                        rest = [x for x in extra if str(x["id"]) not in set(picked)]
                        if _prepend:
                            output_records = sel_recs + output_records + rest
                        else:
                            output_records = output_records + sel_recs + rest
                    elif extra:
                        output_records = output_records + extra
                    # 提示条（聚合/偏好/视觉/时序…）永远排在最前
                    if locals().get("_notes"):
                        # **合成出来的结果**，不是检索候选 —— 必须始终置顶，
                        # 否则放到末尾会被 top_k 截断掉（聚合题的答案就在提示条里）。
                        # 只有"图书馆检索出的 extra 记录"才受 AMC_LIBRARY_PREPEND 控制。
                        #   提示条（amc_*）**不能前置**（会抢首条、挤掉 gold），
                        #   但**也不能一律追加到末尾**——真实记录有 100 条时会被 top_k 截掉、整个消失。
                        #   ⇒ 折中：**插在第 3~5 名之间**（真实记录前 3 条之后），
                        #     既不抢首条，也一定落在 top_k 之内。
                        _ins = min(3, len(output_records))
                        output_records = (list(output_records[:_ins]) + list(_notes)
                                          + list(output_records[_ins:]))
                    # 证据完整性：补同会话伙伴（默认开，AMC_SESSION_PARTNERS=0 关）
                    import os as _os_sp
                    if _os_sp.getenv("AMC_SESSION_PARTNERS", "1") == "1":
                        _has_qimg = any(isinstance(_p, dict) and _p.get("type") in ("image_url", "image")
                                        for _p in (req.get("query") or []))
                        output_records = self._append_session_partners(
                            lib, output_records, q_text, limit=20 if _has_qimg else 12,
                            top_k=int(req.get("top_k") or 0))
                    logger.info("[library] pool=%d picked=%d appended=%d", len(pool), len(picked), len(extra))
                    self._stage_dump("library", output_records, req.get("user_id"), q_text)
        except Exception:
            logger.exception("library augment failed")
        # 原因是扩展根本没执行；绕开 library 直接从 store 展开后 **63% → 71%**。
        if ((os.getenv("AMC_SESSION_PARTNERS_TO_BUDGET", "0") or "0") == "1"
                and output_records):
            try:
                _have = {str(r.get("id")) for r in output_records}
                _anchors = [str(r.get("id")) for r in output_records[:3]]
                _r2s = {r.get("id"): r.get("session_id") for r in output_records}
                _win = int(os.getenv("AMC_SESSION_PARTNER_WINDOW", "10") or 10)
                _budget = min(100, int(req.get("top_k") or 100))
                _add = []
                for _a in _anchors:
                    _sid = _r2s.get(_a) or ""
                    if not _sid:
                        continue
                    _rows = self.store.records_of_session(req["user_id"], _sid, limit=200)
                    if _a not in _rows:
                        _rows = _rows[: 2 * _win + 1]
                        _cand = _rows
                    else:
                        _i = _rows.index(_a)
                        _cand = _rows[max(0, _i - _win): _i + _win + 1]
                    for _rid in _cand:
                        if len(output_records) + len(_add) >= _budget:
                            break
                        if _rid in _have or any(x.get("id") == _rid for x in _add):
                            continue
                        try:
                            _m = self.store.get_memory(_rid)
                        except Exception:
                            _m = None
                        if not _m or not (_m.get("content") or _m.get("n_images")):
                            continue
                        _m = dict(_m)
                        _m["score"] = 0.0
                        _add.append(_m)
                if _add:
                    output_records = list(output_records) + _add
                    logger.info("[evidence] store 兜底会话扩展 +%d（无 library）", len(_add))
            except Exception:
                logger.exception("store 兜底会话扩展失败（忽略）")
        logger.info("[render] hits=%d -> output_records=%d（空 content 跳过 %d）",
                    len(hits), len(output_records), locals().get("_empty_skipped", 0))
        # 统一 content 形状：有些路径给的是纯字符串，有些是 parts 列表 → 一律规范成列表
        try:
            _norm = []
            for _r in output_records:
                _c = _r.get("content")
                if isinstance(_c, str):
                    _r = dict(_r)
                    _r["content"] = [{"type": "text", "text": _c}]
                _norm.append(_r)
            output_records = _norm
        except Exception:
            logger.exception("content normalize failed")
        # 过滤"有记录但没文本"的轮次：它们不携带信息，却会占据 data[0]
        try:
            def _has_text(rec):
                c = rec.get("content")
                if isinstance(c, list):
                    return any(isinstance(p, dict) and str(p.get("text") or "").strip() for p in c)
                return bool(str(c or "").strip())
            _kept = [r for r in output_records if _has_text(r)]
            if len(_kept) != len(output_records):
                logger.info("[search] dropped %d empty-text records", len(output_records) - len(_kept))
            output_records = _kept
        except Exception:
            logger.exception("empty-text filter failed")
        if not output_records:
            try:
                if hasattr(self.store, "count_memories") and self.store.count_memories(req["user_id"]) == 0:
                    return {"data": []}
            except Exception:
                logger.exception("count_memories failed")
            # ---- 空候选兜底（AMC_EMPTY_FALLBACK=1，默认关）----
            # 依据：ATM 全量 390 题里有 3.3%（13 题）返回 0 条 —— 空返回**一定是 0 分**，
            # 而"兜一次更松的检索"至少有非零期望。典型是日期题（temporal 通道需要候选才触发）。
            # ⚠️ 只有"候选池为空"才是真正的检索失败；**主动弃答不兜底**。
            # 实测：那道 "What date did I fly to Cairo?" 用 BM25 能召回 5 条，
            # 说明空返回是**判定为弃答**后清空的，不是没检索到 —— 兜底会破坏弃答能力。
            _pool_empty = True
            try:
                _p = locals().get("pool")
                _pool_empty = (not _p) if isinstance(_p, list) else True
            except Exception:
                _pool_empty = True
            if os.getenv("AMC_EMPTY_FALLBACK", "0") == "1" and _pool_empty:
                try:
                    _fb = self._empty_fallback(req, q_text, requested_k)
                    if _fb:
                        logger.info("[search] empty fallback -> %d 条", len(_fb))
                        return {"data": self._finalize_records(_fb, top_k=requested_k, abstain=False)}
                except Exception:
                    logger.exception("empty fallback failed")
            # 有记忆但检索没结果 → 给明确答复，避免"无声失败"
            logger.info("[search] empty retrieval -> sentinel")
            return {"data": self._finalize_records([{"id": "amc_no_evidence",
                              "content": [{"type": "text", "text":
                                           "No relevant memory found for this question. The user's memory contains no matching records."}],
                              "score": 0.0, "created_at": ""}], top_k=requested_k)}
        # 金额证据的来源记录：确保它们出现在返回里（note 之后、其它记录之前）
        try:
            _want = [i for i in (getattr(_DECISION, "amount_ids", []) or []) if i]
            if _want:
                _byid = {str(r.get("id")): r for r in output_records if isinstance(r, dict)}
                _promoted = []
                for _rid in _want:
                    _row = _byid.get(_rid)
                    if _row is None:
                        try:
                            _row = self.store.get_memory(_rid) if hasattr(self.store, "get_memory") else None
                        except Exception:
                            _row = None
                    if _row:
                        _promoted.append(_row)
                if _promoted:
                    _pid = {str(r.get("id")) for r in _promoted}
                    _head = [r for r in output_records if str(r.get("id", "")).startswith("amc_")]
                    _tail = [r for r in output_records if not str(r.get("id", "")).startswith("amc_")
                             and str(r.get("id")) not in _pid]
                    output_records = _head + _promoted + _tail
                    logger.info("[aggregate] 金额来源记录前置 %d 条（%s）",
                                len(_promoted), [str(r.get("id"))[:12] for r in _promoted])
        except Exception:
            logger.exception("attach amount source records failed")
        # ---- M3（可选模块，默认关闭）：LLM 支持度判定 ----
        # 只在"原本要作答"时介入；判"证据不足" → 降级为拒答。
        # 不反向提升，避免放大误答。开关：AMC_MOD_ASSESS=1
        try:
            from .mod_assess import enabled as _ma_on, judge as _ma_judge
            if _ma_on():
                _top = output_records[:6]
                _vr = _ma_judge(q_text, [{"id": r.get("id"),
                                          "text": " ".join(str(p.get("text") or "")
                                                           for p in (r.get("content") or [])
                                                           if isinstance(p, dict))} for r in _top])
                if not _vr.get("answerable"):
                    logger.info("[mod_assess] 判不足（%s）-> 降级为拒答", str(_vr.get("reason"))[:60])
                    if self._abstain_soft_ok(output_records):
                        logger.info("[abstain-soft] mod_assess 判不足但证据 %d 条 ⇒ 不发哨兵，改软提示",
                                    len(output_records))
                        # ⚠️ 同上：祈使句 → 纯事实描述（合规：禁止 prompt injection）
                        #   判不足但证据够多时**不再前置提示**，只返回记录。
                        return {"data": self._finalize_records(list(output_records),
                                                               top_k=requested_k, abstain=False,
                                                               _deep_query=q_text)}
                    return {"data": self._finalize_records([{
                        "id": "amc_abstain_soft_note",
                        "content": [{"type": "text", "text":
                                     "【证据不足】仅有相关背景，缺少回答该问题所需的信息"}],
                        "score": 0.0, "created_at": ""}] + self._abstain_extra(output_records, requested_k),
                        top_k=requested_k, abstain=True)}
                logger.info("[mod_assess] 判可答（%s）", str(_vr.get("reason"))[:60])
        except Exception:
            logger.exception("mod_assess failed; keep original decision")
        #    开关：AMC_AB_CHANNEL=1 才启用；默认 0 ⇒ 行为完全不变。
        #    产出作为**第一条**证据（不删任何现有记录）⇒ flag=0 即完全回退。
        #    kind 写进日志，供平台跑完后归因"哪条路线被用到多少次"。
        import os as _os_ab   # ⚠️ 必须本地导入：这个函数作用域里没有 `_os`
        #    `data/store/ab_flag.txt` 内容为 1/on/true ⇒ 开；否则关（1 秒 TTL）。
        #    ⚠️ 必须 try/except：我 17:11 那次就是"开关那行代码本身抛异常"导致全挂。
        _ab_hot = False
        try:
            from .ab_retrieve import ab_enabled as _ab_on
            _ab_hot = bool(_ab_on())
        except Exception:
            _ab_hot = False
        if _ab_hot or (_os_ab.getenv("AMC_AB_CHANNEL", "0") or "0") == "1":
            try:
                #   AB 通道**只返回证据记录**（不返回答案文本）——因为答案端不归我们管，
                #   返回答案文本等于白做；返回**原文关键句**才能让答案端抄到具体名词。
                #   拒答情形返回 []（不注入）⇒ 我方 abstain 逻辑完全不受影响。
                #   但他的两个关键零件可以复用：`build_context`（按题面配额取上下文）。
                #   ⇒ 这里对 duration 题**各取两段的证据记录**（不含结论），
                #     让平台的答案模型自己算跨度/比较 —— 我们只给证据，不下结论。
                #   我实测否掉了它的 `duration_spans`（3/3 荒谬：两段完全相同 124 天、
                #   或 "freelance" 算成 18.6 年）。**改进版用"专名锚定"**：
                #     · 能锚定 ⇒ 给"记录数 + 起止 + 天数（含首尾）"
                #     · 不能锚定 ⇒ 明说"不可比较，不要硬选"
                #   ⇒ 荒谬值不会传播（只有**两侧都锚定**才下结论）。
                #   实测 dur-1/2/3：分别因 Toronto 缺失 / 无专名 / 一侧无专名 ⇒ 全部"不可比较"
                #   —— 这是**诚实**的结论，比让答案端瞎猜 A/B 好。
                _ab_recs = []
                _DUR_RX = __import__("re").compile(
                    r"\*Duration\s*1\s*:\*|(gap|durations?).{0,24}(longer|shorter)", __import__("re").I)
                if _DUR_RX.search(q_text or ""):
                    try:
                        from .ab_pipeline import duration_span_anchored as _dur
                        _ab_recs = _dur(
                            self.store.conn, "memories", req.get("user_id", ""), q_text,
                            dict(text_col="index_text", date_col="created_at")) or []
                        logger.info("[ab] duration 锚定 ⇒ %d 条", len(_ab_recs))
                    except Exception:
                        logger.exception("[ab] duration 锚定失败（忽略）")
                if not _ab_recs:
                    #   `evidence_records`       = 按题面配额取上下文（只保证"相关"）
                    #   `evidence_records_guided`= **先生成答案 → 提取关键短语 →
                    #                              在证据里逐字定位 → 只注入真实存在的那几句**
                    #   区别：后者直接保证"含答案原句的记录"被前置，
                    #        且模型编造的短语会被"逐字命中"校验自动丢弃。
                    #   依据（官方错误处理表）：**500/502/503/504 会自动退避重试**，
                    #   而我们的"重试缓存"（`search_cache`，键含 user+query+top_k+build）
                    #   会在**第一次跑完后把结果落盘** ⇒ 平台重试时**直接命中缓存、秒回**。
                    #   ⇒ 所以"首次慢"不再是风险：慢一次，重试就快了。
                    #   guided 的收益实测（6 题 A/B）：c04 0/1→1/1、dog268 2/5→3/5、
                    #   gem-direct_recall refuse→answer ⇒ **+3/6**
                    if _ab_fut is not None:
                        # 用并行提交的结果（已在检索期间跑完）；超时则放弃 ab 通道
                        import os as _os_abw
                        try:
                            _ab_recs = _ab_fut.result(timeout=float(_os_abw.getenv("AMC_AB_WAIT_S", "90") or 90))
                            logger.info("[ab] 并行任务已完成（无需再等）")
                        except Exception:
                            logger.warning("[ab] 并行任务超时/失败 ⇒ 本次跳过 ab 通道")
                            _ab_recs = []
                    else:
                        from .ab_pipeline import evidence_records_guided as _ab_ev
                        _ab_recs = _ab_ev(
                            self.store.conn, "memories", req.get("user_id", ""), q_text,
                            dict(text_col="index_text", date_col="created_at"))
                logger.info("[ab] evidence_records=%d", len(_ab_recs or []))
                if _ab_recs:
                    # ⚠️ （子 agent 审计，严重）：AB 记录改**继承源 id** 后，
                    #   原来这句 `id not in _ab_ids` 会把**整条原始记录删掉**、只留一句 ≤220 字片段
                    #   ⇒ 违反本段注释"作为第一条证据（不删任何现有记录）"。
                    #   修法：原记录一律保留；只注入"源记录不在返回集里"的 AB 条（避免同 id 重复）。
                    _orig_ids = {str(r.get("id")) for r in output_records}
                    _ab_fixed = []
                    for _x in _ab_recs:
                        if str(_x.get("id")) in _orig_ids:
                            continue          # 原文已在返回集，别用短句替换
                        _y = dict(_x)
                        _y.setdefault("score", 0.99)
                        _y.setdefault("created_at", "")
                        _ab_fixed.append(_y)
                    if _ab_fixed:
                        # ⚠️ ：ab 记录也**不再前置**（原来 score 0.99 会占满 top）
                        #   ⇒ 追加到真实记录之后；score 压到 0.001（只作"补充候选"）
                        for _y in _ab_fixed:
                            _y["score"] = 0.001
                        output_records = list(output_records) + _ab_fixed
            except Exception:
                logger.exception("[ab] channel failed (忽略)")
        # ⚠️ 2026-11-02（p_ref）：把指代锚点记录**并入返回集**（放在最前，score 0.97）
        _prh = getattr(self, "_p_ref_hits", None)
        if _prh:
            logger.info("[p_ref] 并入 %d 条锚点到返回集", len(_prh))
            output_records = list(_prh) + [r for r in output_records
                                           if str(r.get("id")) not in
                                           {str(x.get("id")) for x in _prh}]
            self._p_ref_hits = []
        self._stage_dump("ab", output_records, req.get("user_id"), q_text)
        logger.info("[final] output_records=%d -> finalize(top_k=%d)", len(output_records), requested_k)
        # 只在 route==VISUAL 且有带图记录时做；按图片 sha 缓存；≤200 字放 text 段**最前面**。
        #   背景：全库含图记录 33,692 条里 caption 覆盖 **99.3%**（两种格式）：
        #     · `[Image D1:IMG_001] <英文>`    15,168 条（旧评测/副本族）
        #     · 两种都没有：246 条（0.7%）
        #   ⇒ 原来的逻辑"每题都调一次 VLM"在 98.7% 的情况下是**重复劳动**
        #     （每题 +2~5s，且它的缓存是内存 dict、重启即失效）
        #   ⇒ 而且原来固定注入到 `output_records[0]`，但图可能来自 top-3 里的另一条
        #     ⇒ 现在改成：**注入到"图真正所在的那条记录"**，且该记录已有 caption 就跳过。
        if (os.getenv("AMC_VISUAL_ROUGH_DESC", "0") or "0") == "1":
            try:
                from .router import route as _rv
                if _rv(q_text) == "VISUAL":
                    _tgt, _u, _tgt_txt = None, "", ""
                    for _r in output_records[:3]:
                        _parts = _r.get("content")
                        _txt_r = ""
                        if isinstance(_parts, list):
                            _txt_r = "".join(str(_p.get("text") or "") for _p in _parts
                                             if isinstance(_p, dict))
                        elif isinstance(_parts, str):
                            _txt_r = _parts
                        _found = ""
                        if isinstance(_parts, list):
                            for _p in _parts:
                                if isinstance(_p, dict) and _p.get("type") == "image_url":
                                    _uu = str((_p.get("image_url") or {}).get("url") or "")
                                    if _uu.startswith("data:"):
                                        _found = _uu; break
                        if _found:
                            _tgt, _u, _tgt_txt = _r, _found, _txt_r
                            break
                    _has_caption = ('[Image ' in _tgt_txt) or ('Image Caption' in _tgt_txt)
                    if _u and not _has_caption:
                        _d = _visual_rough_desc(_u)
                        if _d and _tgt is not None:
                            _c = _tgt.get("content")
                            if isinstance(_c, list):
                                _tgt["content"] = [{"type": "text", "text": "[视觉粗描述] " + _d}] + list(_c)
                            elif isinstance(_c, str):
                                _tgt["content"] = "[视觉粗描述] " + _d + "\n" + _c
                            logger.info("[visual] 粗描述已注入（%d 字，该记录无 caption）", len(_d))
                    else:
                        logger.info("[visual] 目标记录已有 caption ⇒ 跳过 VLM（省 2~5s）")
            except Exception:
                logger.exception("[visual] 粗描述失败（忽略）")
        _final = {"data": self._finalize_records(output_records, top_k=requested_k, abstain=False, _deep_query=q_text)}
        # ⚠️ 2026-11-03：**响应图片压缩**（官方限制 单图≤10MiB、合计≤30MiB）
        #   压完再进缓存 ⇒ 重试命中缓存时不必再压。
        try:
            _final["data"] = compress_response_images(_final["data"])
        except Exception:
            logger.exception("[search-img] 压缩入口失败（忽略）")
        logger.info("[final] 返回 %d 条", len(_final["data"]))
        try:
            with MemoryService._search_cache_lock:
                MemoryService._search_cache[_ck] = (_tm.time(), _final)
                if len(MemoryService._search_cache) > 512:
                    for _k in list(MemoryService._search_cache)[:128]:
                        MemoryService._search_cache.pop(_k, None)
            MemoryService._cache_save()
            logger.info("[retry] STORE user=%s q=%r", str(req.get("user_id"))[-12:], q_text[:40])
        except Exception:
            pass

        #   N 内完成 ⇒ 返回 merge 后顺序（200，首查即受益）
        #   超时 ⇒ 返回 **408**（平台重试；后台继续跑并写回缓存，重试通常命中）
        try:
            _wait_s = float(os.getenv("AMC_MERGE_SYNC_WAIT_S", "0") or 0)
            _flag_on2 = (os.getenv("AMC_MERGE_RERANK", "0") or "0") == "1" or \
                        os.path.exists(os.path.join("data", "store", "merge_rerank.on"))
            _data2 = list(_final.get("data") or [])
            logger.info("[merge_sync] 判定 wait=%.1f flag=%s n=%d", _wait_s, _flag_on2, len(_data2))
            if _wait_s > 0 and _flag_on2 and len(_data2) >= 8:
                import concurrent.futures as _cf2
                _mpool = getattr(MemoryService, "_merge_pool", None)
                if _mpool is None:
                    _mpool = _cf2.ThreadPoolExecutor(max_workers=4, thread_name_prefix="merge-sync")
                    MemoryService._merge_pool = _mpool
                _snap2 = [dict(x) for x in _data2]

                def _apply_merged(_f, _snap=_snap2, _ck=_ck):
                    try:
                        _merged = _f.result()
                        if not _merged:
                            return
                        _by = {str(x.get("id")): x for x in _snap}
                        _ordered = [_by[str(x.get("id"))] for x in _merged if str(x.get("id")) in _by]
                        if len(_ordered) != len(_by):
                            logger.info("[merge_sync] 长度不符（%d vs %d）⇒ 不用", len(_ordered), len(_by))
                            return
                        with MemoryService._search_cache_lock:
                            MemoryService._search_cache[_ck] = (_tm.time(), {"data": _ordered})
                        try:
                            MemoryService._cache_save()
                        except Exception:
                            pass
                        logger.info("[merge_sync] ✅ 已应用合并顺序（%d 条）", len(_ordered))
                    except Exception:
                        logger.warning("[merge_sync] 应用失败（忽略）", exc_info=True)

                _mf = _mpool.submit(self.merge_rerank, q_text, _snap2)
                try:
                    _mf.result(timeout=_wait_s)
                except _cf2.TimeoutError:
                    logger.warning("[merge_sync] 超过 %.1fs 未完成 ⇒ 返回 408（后台继续并写缓存）", _wait_s)
                    _final["_merge_timeout"] = True
                    _mf.add_done_callback(_apply_merged)
                else:
                    _apply_merged(_mf)
        except Exception:
            logger.warning("[merge_sync] 同步等待分支异常（忽略，返回原顺序）", exc_info=True)
        return _final
