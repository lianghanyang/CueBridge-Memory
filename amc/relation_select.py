"""记忆侧：**关系感知证据选择**（结构化表示 + 通用关系规则 + 图↔图证据关系）。

来源（`STRUCTURED_MEMORY_PROBE.md` §8）：在全 image 路 487 条上与"只上 4B"对照——

| 臂 | R@1 | Hit@1 | R@n_gold |
|---|---|---|---|
| 4B-only（原 route-optimal） | 0.2658 | 0.7454 | 0.6547 |
| **R1** 结构化 + 通用关系规则 + 输出全部证据 | 0.2917 | 0.8111 | 0.6841 |
| **R2** = R1 + 图↔图证据关系重排 | **0.2969** | **0.8378** | **0.7161** |
| 现管线 L11（被替代） | 0.2234 | 0.6838 | 0.5515 |

R2 vs 4B-only：ΔR@1 **+0.0311 [+0.0118,+0.0518] ✅**、ΔHit@1 +0.0924 ✅、ΔR@n_gold +0.0614 ✅。
全库换算（image→R2, text→+LLM）：R@1 **0.4396 → 0.4630**，Hit@1 0.7584 → 0.8075。

**作用域（硬闸）**：本档只在**查询含图**时启用（text 路未验证，且 §3.2 表明结构对 L2b 无效）。
命中时它**替代** `vl_rerank`（实测 R2 ≥ L11），避免重复重排。

结构化卡片的字段来源全部是**服务侧已有的数据**：
  * 身份：`session_id / request_id / msg_index / role / created_at`（store 列）
  * 承载关系：`n_images`（哪一轮带图）+ 图像描述（content 里 image_ref 之后的 text 部件）
  * 会话邻接：同 session 的前后 request
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

logger = logging.getLogger("amc.relation_select")

RULES = ("Rules for selecting evidence:\n"
         "  R1. Evidence = the turns the ANSWER is grounded in (the turns that state/contain "
         "what the question asks for), not the turns that merely share the topic or wording.\n"
         "  R2. A neighbouring turn in the SAME session is not evidence just because it is "
         "adjacent or on-topic — adjacency is not evidence.\n"
         "  R3. If the question refers to the attached picture, the evidence must carry the "
         "matching image (has_image: YES, description consistent with the query image); a "
         "text-only turn of the same session is not a substitute.\n"
         "  R4. Several turns can be evidence — output ALL of them.\n"
         "  R5. Use the record fields: date/speaker/turn index decide WHO said it and WHEN; "
         "the image description says what a turn actually shows.\n"
         "Output ONE line: RANKING = [i, j, k, ...] with every supporting candidate, most "
         "relevant first.")

PAIR_PROMPT = """You are comparing two images from a user's chat history.

Image A = the picture the user attached to their question.
Image B = an image stored in one candidate memory.

Question the user asked: {question}

Decide the relation between the two DEPICTED INSTANCES (not the topic):
  SAME_INSTANCE = B shows the very same object/plant/animal/product (same individual).
  SAME_CATEGORY = B shows a different individual of the same kind.
  DIFFERENT     = unrelated subject matter.
Also decide whether B is actually the picture the user is referring to:
  EVIDENCE = YES if B is the referred picture, else NO.

Answer in exactly two lines:
RELATION=<SAME_INSTANCE|SAME_CATEGORY|DIFFERENT>
EVIDENCE=<YES|NO>"""

_DATA_URI_RE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$", re.S)


def extract_query_images(query: Any, max_images: int = 1) -> List[str]:
    out: List[str] = []
    if not isinstance(query, list):
        return out
    for part in query:
        if not isinstance(part, dict) or part.get("type") not in (
                "image_url", "image", "input_image"):
            continue
        payload = part.get("image_url", part)
        url = payload.get("url") if isinstance(payload, dict) else payload
        if isinstance(url, str) and url:
            out.append(url)
        if len(out) >= max_images:
            break
    return out


def parse_ranking(text: str, n: int) -> List[int]:
    """容错解析 `RANKING = [i, j, ...]`（允许 `CAND 3` 这类前缀）。"""
    if not text:
        return []
    m = re.search(r"RANKING\s*=\s*\[([^\]]*)\]", text)
    seg = m.group(1) if m else None
    if seg is None:
        cands = re.findall(r"\[([^\]]*)\]", text)
        seg = cands[-1] if cands else ""
    nums = [int(x) for x in re.findall(r"\d+", seg)]
    seen, out = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            out.append(x - 1)
    return out


def parse_rel(text: str) -> Tuple[Optional[str], Optional[bool]]:
    if not text:
        return None, None
    m = re.search(r"RELATION\s*=\s*(SAME_INSTANCE|SAME_CATEGORY|DIFFERENT)", text, re.I)
    e = re.search(r"EVIDENCE\s*=\s*(YES|NO)", text, re.I)
    rel = m.group(1).upper() if m else None
    ev = (e.group(1).upper() == "YES") if e else None
    if rel is None:
        up = text.upper()
        for cand in ("SAME_INSTANCE", "SAME_CATEGORY", "DIFFERENT"):
            if cand in up:
                rel = cand
                break
    return rel, ev


def rel_score(rel: Optional[str], ev: Optional[bool]) -> Optional[float]:
    base = {"SAME_INSTANCE": 2.0, "SAME_CATEGORY": 1.0, "DIFFERENT": 0.0}.get(rel)
    if base is None:
        return None
    return base - (0.5 if ev is False else 0.0)


class RelationSelectClient:
    """结构化证据选择 + 可选的逐对图像证据关系重排。"""

    def __init__(self, base_url: str, model: str, pool: int = 10, timeout: float = 90.0,
                 workers: int = 8, retries: int = 2, cache_path: Optional[str] = None,
                 pairwise: bool = True, requires_query_image: bool = True,
                 text_chars: int = 600, neighbor_chars: int = 120, max_tokens: int = 700,
                 api_key: str = "EMPTY"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"      # 此前缺这个 → 对网关 401
        self.model = model
        self.pool = max(2, int(pool))
        self.timeout = timeout
        self.workers = max(1, int(workers))
        self.retries = max(1, int(retries))
        self.pairwise = bool(pairwise)
        self.requires_query_image = bool(requires_query_image)
        self.text_chars = int(text_chars)
        self.neighbor_chars = int(neighbor_chars)
        self.max_tokens = int(max_tokens)
        self.client = httpx.Client(timeout=timeout)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "applied": 0, "skipped_no_image": 0, "failed": 0,
                      "pairs": 0, "pair_failed": 0, "cached": 0}
        self._mem: Dict[str, Any] = {}
        self._db: Optional[sqlite3.Connection] = None
        self._local = threading.local()
        if cache_path:
            try:
                from pathlib import Path
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(cache_path, check_same_thread=False)
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("CREATE TABLE IF NOT EXISTS rel_select("
                                 "key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                self._db.commit()
            except Exception:
                logger.exception("relation-select cache unavailable; running uncached")
                self._db = None

    # ---------------- cache ----------------
    def _get(self, key: str):
        with self.lock:
            if key in self._mem:
                return self._mem[key]
            if self._db is not None:
                row = self._db.execute("SELECT payload FROM rel_select WHERE key=?",
                                       (key,)).fetchone()
                if row:
                    val = json.loads(row[0])
                    self._mem[key] = val
                    return val
        return None

    def _put(self, key: str, val) -> None:
        with self.lock:
            self._mem[key] = val
            if self._db is not None:
                try:
                    self._db.execute("INSERT OR REPLACE INTO rel_select(key,payload) "
                                     "VALUES(?,?)", (key, json.dumps(val)))
                    self._db.commit()
                except Exception:
                    logger.exception("relation-select cache write failed")

    # ---------------- store access ----------------
    def _conn(self, store):
        """只读游标（每线程一个；store 的 conn 不保证线程安全）。"""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(f"file:{store.db_path}?mode=ro", uri=True)
            self._local.conn = c
        return c

    def _turns_of(self, store, request_id: str) -> List[Dict[str, Any]]:
        rows = self._conn(store).execute(
            "SELECT id,msg_index,role,created_at,index_text,n_images,content_json "
            "FROM memories WHERE request_id=? ORDER BY msg_index", (request_id,)).fetchall()
        out = []
        for (mid, mi, role, created, it, ni, cj) in rows:
            caption = ""
            try:
                parts = json.loads(cj or "[]")
                for pos, p in enumerate(parts):
                    if isinstance(p, dict) and p.get("type") == "image_ref":
                        nxt = parts[pos + 1] if pos + 1 < len(parts) else {}
                        if isinstance(nxt, dict) and nxt.get("type") == "text":
                            caption = str(nxt.get("text") or "")
                            break
            except Exception:
                caption = ""
            out.append({"mid": mid, "mi": mi, "role": role, "created": created or "",
                        "text": (it or "").strip(), "n_images": ni or 0,
                        "caption": caption})
        return out

    def _req_image_ref(self, store, request_id: str):
        """该请求下第一条含图记忆的 image_ref（直接查库，不受 head 窗口限制）。"""
        try:
            row = self._conn(store).execute(
                "SELECT content_json FROM memories WHERE request_id=? AND n_images>0 "
                "ORDER BY msg_index LIMIT 1", (request_id,)).fetchone()
        except Exception:
            return None
        if not row:
            return None
        try:
            for part in json.loads(row[0] or "[]"):
                if isinstance(part, dict) and part.get("type") == "image_ref":
                    return part
        except Exception:
            return None
        return None

    def _neighbors(self, store, session_id: str, request_id: str, limit: int = 1):
        rows = self._conn(store).execute(
            "SELECT request_id, MIN(created_at) FROM memories WHERE session_id=? "
            "GROUP BY request_id ORDER BY MIN(created_at)", (session_id,)).fetchall()
        rids = [r[0] for r in rows]
        if request_id not in rids:
            return []
        i = rids.index(request_id)
        return [rids[j] for j in range(max(0, i - limit), min(len(rids), i + limit + 1))
                if j != i]

    def _card(self, store, request_id: str) -> str:
        turns = self._turns_of(store, request_id)
        if not turns:
            return f"  request: {request_id}   (no turns)"
        sess = ""
        # session 从 store 取（turns_of 未带，这里补一次）
        row = self._conn(store).execute("SELECT session_id FROM memories WHERE request_id=? "
                                        "LIMIT 1", (request_id,)).fetchone()
        sess = (row[0] if row else "") or ""
        date = (turns[0]["created"] or "?")[:10]
        L = [f"  request: {request_id.split(':')[-1]}   date: {date}   "
             f"session: {sess.split(':')[-1]}"]
        for t in turns:
            L.append(f"  turn[{t['role']}]: {t['text'][: self.text_chars]}")
            if t["n_images"]:
                L.append(f"    ├ has_image: YES ({t['n_images']})")
                if t["caption"]:
                    L.append(f"    └ image.caption: {t['caption'][:320]}")
                else:
                    L.append("    └ image.caption: (unavailable)")
            else:
                L.append("    └ has_image: NO")
        for nb in self._neighbors(store, sess, request_id, 1):
            nts = self._turns_of(store, nb)
            date_nb = (nts[0]["created"] or "?")[:10] if nts else "?"
            txt = " | ".join(f"{t['role']}: {t['text'][: self.neighbor_chars]}"
                             for t in nts)
            L.append(f"  session_neighbor {nb.split(':')[-1]} ({date_nb}): {txt}")
        return "\n".join(L)

    # ---------------- calls ----------------
    def _chat(self, messages, max_tokens=None):
        body = {"model": self.model, "temperature": 0.0,
                "max_tokens": max_tokens or self.max_tokens,
                "messages": messages}
        for _ in range(self.retries):
            try:
                r = self.client.post(
                    self.base_url + "/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as exc:
                logger.warning("relation-select call failed: %s", exc)
        return ""

    @staticmethod
    def _data_uri(raw: bytes, fmt: str) -> str:
        """转 data URI；超大图先按契约缩放（否则服务端可能 400）。"""
        import base64
        try:
            from .imaging import vlm_ready_image
            raw, fmt = vlm_ready_image(raw, fmt)
        except Exception:
            logger.warning("image shrink failed; sending raw (%d bytes)", len(raw))
        return f"data:image/{fmt or 'jpeg'};base64," + base64.b64encode(raw).decode()

    # ---------------- public ----------------
    def select(self, store, query_text: str, query_parts: Any,
               hits: List[Tuple[Dict[str, Any], float]]
               ) -> Tuple[List[Tuple[Dict[str, Any], float]], bool]:
        """返回 (重排后的 hits, 是否真正生效)。失败/越界一律原样返回。"""
        if len(hits) < 3:
            return hits, False
        qimgs = extract_query_images(query_parts, 1)
        if self.requires_query_image and not qimgs:
            with self.lock:
                self.stats["skipped_no_image"] += 1
            return hits, False
        head, tail = hits[: self.pool], hits[self.pool:]
        req_of = {m["id"]: m.get("request_id") for m, _ in head}
        cards, order = [], []
        for m, _s in head:
            rid = m.get("request_id")
            if not rid or rid in order:
                continue
            order.append(rid)
            try:
                cards.append(self._card(store, rid))
            except Exception:
                logger.exception("card build failed for %s", rid)
                return hits, False
        if len(cards) < 2:
            return hits, False

        qtext = query_text.strip()
        prompt = [f"Question: {qtext}"]
        if qimgs:
            prompt.append("The user attached a picture to this question.")
        prompt.append("")
        prompt.append("Candidate memories (structured records):")
        for i, c in enumerate(cards, 1):
            prompt.append(f"[CAND {i}]\n{c}")
        prompt.append("")
        prompt.append(RULES)
        key = "sel|" + hashlib.sha1(
            json.dumps([self.base_url, self.model, qtext, len(cards), cards],
                       ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            self.stats["calls"] += 1
        cached = self._get(key)
        if cached is not None:
            with self.lock:
                self.stats["cached"] += 1
            ranking = cached.get("ranking") or []
        else:
            ranking = parse_ranking(self._chat([{"role": "user",
                                                 "content": "\n".join(prompt)}]), len(cards))
            self._put(key, {"ranking": ranking})
        if not ranking:
            with self.lock:
                self.stats["failed"] += 1
            return hits, False

        # 依 request 序：被选中的（按 LLM 顺序）在前，其余保持原相对序
        picked_rids = [order[j] for j in ranking if j < len(order)]
        picked = [rid for rid in picked_rids]
        rest = [rid for rid in order if rid not in set(picked)]
        seq = picked + rest

        # 可选：对**被选中的含图候选**做逐对证据关系重排
        if self.pairwise and qimgs:
            # 逐对候选 = 被选中的、且该**请求**下确有含图记忆的轮（查库，不看 head 窗口）
            _ref_cache = {}
            for rid in picked:
                _ref_cache[rid] = self._req_image_ref(store, rid)
            img_rids = [rid for rid in picked if _ref_cache.get(rid)]
            scored: Dict[str, Optional[float]] = {}
            if len(img_rids) >= 2:
                quri = qimgs[0]

                def work(rid):
                    ref = _ref_cache.get(rid)
                    if not ref:
                        return rid, None
                    try:
                        raw = store.blob_path(ref["sha256"],
                                              ref.get("fmt", "jpeg")).read_bytes()
                    except OSError:
                        return rid, None
                    pkey = "pair|" + hashlib.sha1(
                        json.dumps([self.base_url, self.model, qtext, rid,
                                    ref.get("sha256"), len(raw)],
                                   ensure_ascii=False).encode()).hexdigest()
                    hit = self._get(pkey)
                    if hit is not None:
                        with self.lock:
                            self.stats["cached"] += 1
                        return rid, rel_score(hit.get("rel"), hit.get("ev"))
                    body = [{"type": "image_url", "image_url": {"url": quri}},
                            {"type": "text", "text": "Image A (attached to the question) above.\n"},
                            {"type": "image_url",
                             "image_url": {"url": self._data_uri(raw, ref.get("fmt"))}},
                            {"type": "text", "text": "Image B (candidate memory) above.\n\n"
                             + PAIR_PROMPT.replace("{question}", qtext)}]
                    txt = self._chat([{"role": "user", "content": body}], max_tokens=260)
                    rel, ev = parse_rel(txt)
                    self._put(pkey, {"rel": rel, "ev": ev})
                    with self.lock:
                        self.stats["pairs"] += 1
                    return rid, rel_score(rel, ev)

                try:
                    with ThreadPoolExecutor(max_workers=min(self.workers,
                                                            len(img_rids))) as ex:
                        for rid, sc in ex.map(work, img_rids):
                            scored[rid] = sc
                except Exception:
                    logger.exception("pairwise relation failed; keeping LLM order")
                    scored = {}
            if scored and any(v is not None for v in scored.values()):
                pos = {rid: i for i, rid in enumerate(seq)}
                img_sorted = sorted([r for r in img_rids if r in pos],
                                    key=lambda r: (-(scored.get(r) if scored.get(r) is not None
                                                     else -1.0), pos[r]))
                rest_seq = [r for r in seq if r not in set(img_sorted)]
                seq = img_sorted + rest_seq

        rank_pos = {rid: i for i, rid in enumerate(seq)}
        head_sorted = sorted(head, key=lambda x: rank_pos.get(x[0].get("request_id"), 10 ** 6))
        if not head_sorted:
            return hits, False
        base = head[0][1]
        out, floor = [], None
        for idx, (mem, _s) in enumerate(head_sorted):
            val = base - 1e-6 * idx
            if floor is not None and val > floor:
                val = floor
            out.append((mem, val))
            floor = val
        with self.lock:
            self.stats["applied"] += 1
        return out + list(tail), True

    def describe(self) -> Dict[str, Any]:
        return {"class": self.__class__.__name__, "base_url": self.base_url,
                "model": self.model, "pool": self.pool, "pairwise": self.pairwise,
                "requires_query_image": self.requires_query_image,
                "stats": dict(self.stats)}
