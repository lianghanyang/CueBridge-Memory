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

PROVENANCE_PROMPT = (
    "Decide whether the Document contains or refers to the SAME PICTURE as the QUERY image. "
    "Answer yes only if the Document's image is the one the user is asking about "
    "(the same picture, or the same specific subject and scene). "
    "Topical or thematic similarity is NOT enough."
)

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
                 requires_query_image: bool = True, max_query_images: int = 1,
                 instruct: str = "provenance"):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.pool = max(1, int(pool))
        self.retries = max(1, int(retries))
        self.workers = max(1, int(workers))
        self.include_doc_image = bool(include_doc_image)
        self.max_doc_chars = int(max_doc_chars)
        self.requires_query_image = bool(requires_query_image)
        self.max_query_images = max(1, int(max_query_images))
        # instruct 选择：provenance（默认，判"图中图是否同一张"）| relevance（旧行为）
        # 487 条实测：provenance 版 Hit@1 +6.4pp / R@1 +2.5pp（同一评分函数、同一候选）
        self.instruct = instruct if instruct in ("provenance", "relevance") else "provenance"
        self.client = httpx.Client(timeout=timeout)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "scored": 0, "cached": 0, "failed": 0,
                      "skipped_no_image": 0, "doc_image_used": 0, "round_text_used": 0}
        self._margins: List[float] = []
        self.max_doc_image_side = 384
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
        sys_prompt = PROVENANCE_PROMPT if self.instruct == "provenance" else SYSTEM_PROMPT
        body = {"model": self.model, "temperature": 0.0, "max_tokens": 1,
                "logprobs": True, "top_logprobs": 20,
                "messages": [{"role": "system", "content": sys_prompt},
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
        """候选记忆的首张图（仅在 include_doc_image 时取），**缩放到安全尺寸**。

        教训：直接发原图时单张可到 ~1.2 MB，请求体过大被网关整包拒绝（全部候选 0 分）。
        """
        try:
            from .imaging import encode_data_uri, vlm_ready_image
            for part in (memory.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image_ref":
                    raw = store.blob_path(part["sha256"], part.get("fmt", "jpeg")).read_bytes()
                    small, fmt = vlm_ready_image(raw, part.get("fmt"))
                    side = getattr(self, "max_doc_image_side", 384)
                    try:
                        from PIL import Image
                        import io as _io
                        while len(small) > 200_000 and side >= 128:
                            im = Image.open(_io.BytesIO(small)).convert("RGB")
                            w, h = im.size
                            sc = side / float(max(w, h))
                            im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
                            out = _io.BytesIO(); im.save(out, "JPEG", quality=80)
                            small, fmt = out.getvalue(), "jpeg"
                            side = max(128, side // 2)
                    except Exception:
                        pass
                    with self.lock:
                        self.stats["doc_image_used"] = self.stats.get("doc_image_used", 0) + 1
                    return encode_data_uri(small, fmt)
        except Exception:
            return None
        return None

    def _round_text(self, store, memory: Dict[str, Any], cache: Dict[str, str]) -> str:
        """该记忆**所属整轮**的拼接文本（不是单条消息）。

        依据：22 个混淆案例里 77% 是“同一天/同一轮内多轮次混淆”，
        只喂单条消息会让 reranker 看不到同轮的上下文。
        """
        rid = memory.get("request_id")
        if not rid:
            return (memory.get("index_text") or "")
        if rid in cache:
            return cache[rid]
        text = ""
        try:
            rows = store.conn.execute(
                "SELECT index_text FROM memories WHERE request_id=? ORDER BY msg_index", (rid,)
            ).fetchall()
            text = "\n".join((r[0] or "").strip() for r in rows if r[0]).strip()
        except Exception:
            text = ""
        if not text:
            text = (memory.get("index_text") or "")
        cache[rid] = text
        return text

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
        # pool 的单位是**轮**，不是消息：同一 request_id 只打一次分，得分回填整轮。
        # 注意：输出必须保持**同样的命中集合**（契约），只是顺序按轮分重排。
        deep = lambda m: (m.get("request_id") or m.get("id"))
        reps, picked = [], set()
        for item in hits:
            rid = deep(item[0])
            if rid in picked:
                continue
            if len(reps) >= self.pool:
                break
            picked.add(rid)
            reps.append(item)
        if len(reps) < 2:
            return hits
        with self.lock:
            self.stats["calls"] += 1

        round_cache: Dict[str, str] = {}

        def work(item):
            memory = item[0]
            doc_text = self._round_text(store, memory, round_cache)
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
                self._margins.append(float(s))
                self.stats["scored"] += 1
            return s

        try:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(reps))) as ex:
                scores = list(ex.map(work, reps))
        except Exception:
            logger.exception("vl-rerank batch failed; keeping original order")
            return hits
        if all(s is None for s in scores):
            return hits

        round_score: Dict[str, float] = {}
        for pos, (item, sc) in enumerate(zip(reps, scores)):
            if sc is not None:
                round_score[deep(item[0])] = float(sc)
        if not round_score:
            return hits
        base = hits[0][1]
        # 轮序：有分的按分数降序，无分的排在其后（保持原相对顺序）
        scored_rounds = list(round_score)
        order = sorted(scored_rounds, key=lambda r: -round_score[r])
        pos_of = {rid: i for i, rid in enumerate(order)}
        fallback = len(order)
        indexed = list(enumerate(hits))
        indexed.sort(key=lambda t: (pos_of.get(deep(t[1][0]), fallback), t[0]))
        out, floor = [], None
        for idx, (_orig, (m, _score)) in enumerate(indexed):
            val = base - 1e-6 * idx            # 保持非递增
            if floor is not None and val > floor:
                val = floor
            out.append((m, val))
            floor = val
        return out

    def describe(self) -> Dict[str, Any]:
        return {"class": self.__class__.__name__, "base_url": self.base_url,
                "model": self.model, "pool": self.pool,
                "include_doc_image": self.include_doc_image,
                "requires_query_image": self.requires_query_image,
                "stats": dict(self.stats)}
