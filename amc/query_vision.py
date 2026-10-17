"""记忆侧：查询图理解（把查询里的图片转成文本，供 text-embedding-v4 使用）。

**为什么必须放在记忆侧**：比赛约束下 embedding 固定为 `text-embedding-v4`（纯文本），
而 answer 侧不可改。T3 归因显示 Mem-Gallery **487/1711 = 28.5%** 的题目在原始 QA 里带
`question_image`，但 harness 从未把它传进查询 → 这些题在评测里成了"the picture"式**悬空指代**
（带图样本 R@1 达成率 26.9% vs 纯文本 79.6%）。

契约依据：AML 的 `/search` 允许 `query` 为 **ContentPart[]**（见 `amc/contract.py`），
即查询本身可以是多模态的；此前 `query_to_text` **只取 text 部件、静默丢弃图像**。

本模块只做一件事：**若查询含图片，用 VLM 生成检索导向的描述并并入查询文本**。
- 不改契约、不改返回结构、不改分数单调性；
- 不触碰 answer 侧；
- VLM 不可用时**优雅退化**为原文本查询（绝不让 /search 失败）。

实测（487 条带查询图样本，仅检索阶段）：
    R@1 0.0720 → 0.2861 (+0.2141)，Hit@1 0.2238 → 0.8152 (+0.5914)，pool@30 0.7806 → 0.9893
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("amc.query_vision")

DEFAULT_PROMPT = (
    "You are given the image that a user attached to their question about their own chat "
    "history. Produce a SHORT retrieval-oriented description that would let a search engine "
    "find the memory this image refers to. Output plain text only:\n"
    "1) one dense sentence on the main subject and scene;\n"
    "2) a comma-separated list of discriminative attributes (object, category, colour, count, "
    "breed, product/brand name, any legible text, location hints).\n"
    "Do not speculate beyond what is visible."
)

_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.S)


def extract_query_images(query: Any, max_images: int = 2) -> List[str]:
    """从 query（str | ContentPart[]）里取出图片 data URI 列表。"""
    urls: List[str] = []
    if not isinstance(query, list):
        return urls
    for part in query:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("image_url", "image", "input_image"):
            continue
        payload = part.get("image_url", part)
        url = ""
        if isinstance(payload, dict):
            url = str(payload.get("url") or "")
        elif isinstance(payload, str):
            url = payload
        if url:
            urls.append(url)
        if len(urls) >= max_images:
            break
    return urls


class QueryImageDescriber:
    """把查询图转成文本描述（带磁盘缓存 + 优雅退化）。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 max_images: int = 2, timeout: float = 60.0, retries: int = 2,
                 cache_path: Optional[str] = None, prompt: str = DEFAULT_PROMPT):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.max_images = max(1, int(max_images))
        self.retries = max(1, int(retries))
        self.prompt = prompt
        self.client = httpx.Client(timeout=timeout)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "ok": 0, "cached": 0, "failed": 0}
        self._cache: Dict[str, str] = {}
        self._db: Optional[sqlite3.Connection] = None
        if cache_path:
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(cache_path, check_same_thread=False)
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS query_image_captions("
                    "key TEXT PRIMARY KEY, caption TEXT NOT NULL)")
                self._db.commit()
            except Exception:
                logger.exception("query-image cache unavailable; running uncached")
                self._db = None

    # ---------- cache ----------
    def _lookup(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self._cache:
                return self._cache[key]
            if self._db is not None:
                row = self._db.execute(
                    "SELECT caption FROM query_image_captions WHERE key=?", (key,)).fetchone()
                if row:
                    self._cache[key] = row[0]
                    return row[0]
        return None

    def _store(self, key: str, caption: str) -> None:
        with self.lock:
            self._cache[key] = caption
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO query_image_captions(key,caption) VALUES(?,?)",
                        (key, caption))
                    self._db.commit()
                except Exception:
                    logger.exception("query-image cache write failed")

    # ---------- inference ----------
    def _caption_data_uri(self, data_uri: str) -> Optional[str]:
        key = hashlib.sha256(data_uri.encode("utf-8")).hexdigest()
        hit = self._lookup(key)
        if hit:
            with self.lock:
                self.stats["cached"] += 1
            return hit
        # 送 VLM 前统一缩图（否则大图会让服务端 prompt 超长 → 400）
        try:
            from .imaging import decode_data_uri, encode_data_uri, vlm_ready_image
            raw, _fmt = vlm_ready_image(decode_data_uri(data_uri))
            data_uri = encode_data_uri(raw, "jpeg")
        except Exception:
            logger.warning("query image shrink failed; sending original")
        body = {"model": self.model, "temperature": 0.0, "max_tokens": 160,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": self.prompt}]}]}
        for attempt in range(1, self.retries + 1):
            try:
                r = self.client.post(self.base_url + "/chat/completions",
                                     headers={"Authorization": f"Bearer {self.api_key}"},
                                     json=body)
                r.raise_for_status()
                text = (r.json()["choices"][0]["message"]["content"] or "").strip()
                if text:
                    self._store(key, text)
                    return text
            except Exception as exc:
                logger.warning("query-image caption attempt %d/%d failed: %s",
                               attempt, self.retries, exc)
        return None

    # ---------- public ----------
    def describe(self, query: Any) -> str:
        """返回可并入查询文本的描述片段（无图或失败时返回空串）。"""
        urls = extract_query_images(query, self.max_images)
        if not urls:
            return ""
        with self.lock:
            self.stats["calls"] += 1
        outs: List[str] = []
        for url in urls:
            if not _DATA_URI_RE.match(url):
                # 仅支持 data URI（服务侧不主动抓外网）；其它形态记一次失败并跳过
                with self.lock:
                    self.stats["failed"] += 1
                continue
            cap = self._caption_data_uri(url)
            if cap:
                outs.append(cap)
            else:
                with self.lock:
                    self.stats["failed"] += 1
        if not outs:
            return ""
        with self.lock:
            self.stats["ok"] += 1
        return "\n".join(outs)

    def augment(self, text: str, query: Any) -> str:
        """把查询图描述并回文本查询。"""
        extra = self.describe(query)
        if not extra:
            return text
        return f"{text}\n[Query image] {extra}" if text else f"[Query image] {extra}"

    def describe_stats(self) -> dict:
        return dict(self.stats)
