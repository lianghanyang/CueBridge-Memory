"""记忆侧：专用 VL reranker（HTTP 调用 vLLM 服务）作为 `/search` 的末段重排。

为什么放在记忆侧：answer 侧不可改，而重排需要**原始查询部件**（文本 + 查询图），
只有 `MemoryService.search` 看得到；检索器接口（`search(user_id, text, limit)`）保持不动。

实测依据（`VT_RERANK_RESULT.md` §10，150 条带查询图样本、候选 top-5、专用 Qwen3-VL-Reranker-8B）：

| 条件 | ΔR@1 vs 不重排 |
|---|---|
| **查询含图** + 候选文本 | **+0.0463** [+0.0080,+0.0885] ✅ |
| **查询含图** + 候选文本 + 候选图 | **+0.0510** [+0.0119,+0.0927] ✅（Hit@1 0.7533→0.8200） |
| **查询不含图**（同一模型） | **−0.0745** [−0.1223,−0.0263] ❌ |

⇒ **硬闸**：只有当查询里确实有图片时才启用重排；纯文本查询一律跳过（`requires_query_image`）。
   这与"没有指代对象时重排只会打乱已经很好的检索序"一致。

服务端要求（见 `VT_RERANK_RESULT.md` §10）：
  * 用官方 `qwen3_vl_reranker.jinja` + `--enforce-eager`；
  * 请求用 `system` / `query` / `document` 三种 role；
  * 分数取末位 next-token 的 `P(yes)/(P(yes)+P(no))`（与官方 `sigmoid((W_yes−W_no)·h_last)` 同源）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("amc.vl_rerank")

SYSTEM_PROMPT = ('Judge whether the Document meets the requirements based on the Query and '
                 'the Instruct provided. Note that the answer can only be "yes" or "no".')

_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.S)
_YES = ("yes", "y")
_NO = ("no", "n")


def extract_query_images(query: Any, max_images: int = 2) -> List[str]:
    """从 query（str | ContentPart[]）取出图片 data URI 列表。"""
    out: List[str] = []
    if not isinstance(query, list):
        return out
    for part in query:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("image_url", "image", "input_image"):
            continue
        payload = part.get("image_url", part)
        url = payload.get("url") if isinstance(payload, dict) else payload
        if isinstance(url, str) and url:
            out.append(url)
        if len(out) >= max_images:
            break
    return out


class VLRerankClient:
    """对 (查询文本+查询图, 候选文本[+候选图]) 做 pointwise 打分并重排。"""

    def __init__(self, base_url: str, model: str, pool: int = 5, timeout: float = 60.0,
                 workers: int = 8, retries: int = 2, cache_path: Optional[str] = None,
                 include_doc_image: bool = False, max_doc_chars: int = 1200,
                 requires_query_image: bool = True, max_query_images: int = 1):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.pool = max(1, int(pool))
        self.retries = max(1, int(retries))
        self.workers = max(1, int(workers))
        self.include_doc_image = bool(include_doc_image)
        self.max_doc_chars = int(max_doc_chars)
        self.requires_query_image = bool(requires_query_image)
        self.max_query_images = max(1, int(max_query_images))
        self.client = httpx.Client(timeout=timeout)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "scored": 0, "cached": 0, "failed": 0, "skipped_no_image": 0}
        self._mem: Dict[str, float] = {}
        self._db: Optional[sqlite3.Connection] = None
        if cache_path:
            try:
                from pathlib import Path
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(cache_path, check_same_thread=False)
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("CREATE TABLE IF NOT EXISTS vl_rerank_scores("
                                 "key TEXT PRIMARY KEY, score REAL NOT NULL)")
                self._db.commit()
            except Exception:
                logger.exception("vl-rerank cache unavailable; running uncached")
                self._db = None

    # ---------------- cache ----------------
    def _get(self, key: str) -> Optional[float]:
        with self.lock:
            if key in self._mem:
                return self._mem[key]
            if self._db is not None:
                row = self._db.execute("SELECT score FROM vl_rerank_scores WHERE key=?",
                                       (key,)).fetchone()
                if row is not None:
                    self._mem[key] = float(row[0])
                    return float(row[0])
        return None

    def _put(self, key: str, score: float) -> None:
        with self.lock:
            self._mem[key] = score
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO vl_rerank_scores(key,score) VALUES(?,?)",
                        (key, float(score)))
                    self._db.commit()
                except Exception:
                    logger.exception("vl-rerank cache write failed")

    # ---------------- scoring ----------------
    @staticmethod
    def _score_from_logprobs(data: Dict[str, Any]) -> Optional[float]:
        try:
            top = data["choices"][0]["logprobs"]["content"][0].get("top_logprobs") or []
            yes = sum(math.exp(t["logprob"]) for t in top
                      if str(t["token"]).strip().lower() in _YES)
            no = sum(math.exp(t["logprob"]) for t in top
                     if str(t["token"]).strip().lower() in _NO)
            if yes + no > 0:
                return float(yes / (yes + no))
        except Exception:
            pass
        return None

    def _one(self, query_text: str, query_images: List[str],
             doc_text: str, doc_image: Optional[str]) -> Optional[float]:
        qc: List[Dict[str, Any]] = [{"type": "text", "text": query_text}]
        qc += [{"type": "image_url", "image_url": {"url": u}} for u in query_images]
        dc: List[Dict[str, Any]] = [{"type": "text", "text": doc_text[: self.max_doc_chars]}]
        if doc_image:
            dc.append({"type": "image_url", "image_url": {"url": doc_image}})
        body = {"model": self.model, "temperature": 0.0, "max_tokens": 1,
                "logprobs": True, "top_logprobs": 20,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "query", "content": qc},
                             {"role": "document", "content": dc}]}
        for _ in range(self.retries):
            try:
                r = self.client.post(self.base_url + "/chat/completions", json=body)
                r.raise_for_status()
                s = self._score_from_logprobs(r.json())
                if s is not None:
                    return s
            except Exception as exc:
                logger.warning("vl-rerank score attempt failed: %s", exc)
        return None

    def _doc_image_uri(self, store, memory: Dict[str, Any]) -> Optional[str]:
        """候选记忆的首张图（仅在 include_doc_image 时取）。缺失 blob 时返回 None。"""
        try:
            from .imaging import encode_data_uri
            for part in (memory.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image_ref":
                    raw = store.blob_path(part["sha256"], part.get("fmt", "jpeg")).read_bytes()
                    return encode_data_uri(raw, part.get("fmt"))
        except Exception:
            return None
        return None

    # ---------------- public ----------------
    def rerank(self, store, query_text: str, query_parts: Any,
               hits: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """只在头部 pool 条内重排；失败/无图时原样返回（绝不改变集合与契约）。"""
        if not hits:
            return hits
        query_images = extract_query_images(query_parts, self.max_query_images)
        if self.requires_query_image and not query_images:
            # 硬闸：纯文本查询下专用 VL reranker 实测为负收益（−0.0745），跳过
            with self.lock:
                self.stats["skipped_no_image"] += 1
            return hits
        head, tail = hits[: self.pool], hits[self.pool:]
        if len(head) < 2:
            return hits
        with self.lock:
            self.stats["calls"] += 1

        def work(item):
            memory = item[0]
            doc_text = memory.get("index_text") or ""
            doc_img = self._doc_image_uri(store, memory) if self.include_doc_image else None
            key = hashlib.sha1(
                json.dumps([self.base_url, self.model, query_text, query_images,
                            memory.get("id"), bool(doc_img)],
                           ensure_ascii=False).encode("utf-8")).hexdigest()
            cached = self._get(key)
            if cached is not None:
                with self.lock:
                    self.stats["cached"] += 1
                return cached
            s = self._one(query_text, query_images, doc_text, doc_img)
            if s is None:
                with self.lock:
                    self.stats["failed"] += 1
                return None
            self._put(key, s)
            with self.lock:
                self.stats["scored"] += 1
            return s

        try:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(head))) as ex:
                scores = list(ex.map(work, head))
        except Exception:
            logger.exception("vl-rerank batch failed; keeping original order")
            return hits
        if all(s is None for s in scores):
            return hits
        base = head[0][1]
        ranked = sorted(
            [(m, s, pos) for pos, ((m, _), s) in enumerate(zip(head, scores))],
            key=lambda x: (-(x[1] if x[1] is not None else -1.0), x[2]))
        out, floor = [], None
        for idx, (m, s, _pos) in enumerate(ranked):
            val = base - 1e-6 * idx            # 保持非递增
            if floor is not None and val > floor:
                val = floor
            out.append((m, val))
            floor = val
        return out + list(tail)

    def describe(self) -> Dict[str, Any]:
        return {"class": self.__class__.__name__, "base_url": self.base_url,
                "model": self.model, "pool": self.pool,
                "include_doc_image": self.include_doc_image,
                "requires_query_image": self.requires_query_image,
                "stats": dict(self.stats)}
