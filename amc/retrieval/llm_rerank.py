"""记忆服务内的 LLM 推理式重排（把已实测有效的「候选表示结构化」落进生产路径）。

背景（本轮全部实测，见 result/）：
  * 4B cross-encoder 初筛 → LLM 推理式重排，双数据集显著有效：
      Mem-Gallery +0.016~+0.018，ATM-Bench +0.035~+0.036（配对 CI 均不含 0）。
  * **候选表示的结构化**进一步带来最大单点收益：给 LLM 看候选时**补上同一轮的另一条消息**
    （因为评测单元是「一轮」，此前只给了半轮）：
      Mem-Gallery R@1 0.3835 → 0.4009（+0.0174，CI [+0.0070,+0.0286]），目标格 VS +0.0444，
      R@20 非劣（+0.0007）；机制验收门判定 PASS（见 result/mechanism-gate__*）。
  * 但"加更多结构上下文"（session 邻近轮）反而 −0.0118 → **粒度对齐才是关键**。
  * 证据长度是头号实现细节：ATM 截断到 420 字符会 −0.044。

因此本模块默认 `layout="round"`、`doc_chars=1400`。

⚠️ **唯一硬门槛**：这是每次 /search 一次 LLM 调用（15~30 条 × 1400 字符 ≈ 6~12k input token）。
   必须确认比赛的延迟/成本预算；默认 `AMC_LLM_RERANK=0` 关闭，确认后再开。
   契约安全：只重排**原始记忆**并保证分数非递增，不合成任何答案。
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading

import httpx

logger = logging.getLogger(__name__)

PROMPT = """You are the evidence selector of a personal-memory search engine.

The user asked the question below. A first-stage retriever returned candidate memory
excerpts. Each candidate may include "<same-round other turn>" — the other half of the
same conversational round; judge the ROUND as a unit, not the fragment alone.

Rank the candidates by how likely each is to contain the evidence needed to answer the
question. Sharing topic words is NOT enough: reason about entities, dates and relations.

Return STRICT JSON only:
{"ranking": [<candidate numbers, most relevant first, at most __K__>]}

Question: __Q__

Candidates:
__C__
JSON:"""


EVIDENCE_SET_PROMPT = """You are the evidence selector of a personal-memory search engine.

The user asked the question below. Each candidate is a whole conversational ROUND
(both halves of the exchange are shown together) — judge the ROUND as a unit.

Several rounds may be needed as evidence: list them ALL, not only the single best one.
Sharing topic words is NOT enough; reason about entities, dates and relations.

Question: __Q__

Candidates:
__C__

Return STRICT JSON only (candidate numbers are 0-based, as shown in brackets):
{"selected": [<every candidate that contains evidence, most relevant first>]}
JSON:"""


def _parse_selected(text, n):
    """解析多选输出 {"selected": [...]}（0-based）。

    模型常给出**不合法 JSON**（尾随逗号/夹杂说明/被截断），因此：
    先试严格 JSON，失败再用正则从文本里抽数字列表 —— 绝不因为格式问题丢掉整档收益。
    """
    if not text:
        raise ValueError("empty")
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    raw = None
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):
        try:
            data = json.loads(candidate)
            raw = data.get("selected") or data.get("ranking") or []
            break
        except Exception:
            continue
    if raw is None:
        import re as _re
        m = _re.search(r'"?selected"?\s*:\s*\[([^\]]*)\]', text)
        seg = m.group(1) if m else None
        if seg is None:                       # 被截断/无右括号
            m2 = _re.search(r'"?selected"?\s*:\s*\[?([0-9][0-9,\s]*)', text)
            seg = m2.group(1) if m2 else None
        if seg is None:
            lists = _re.findall(r"\[([^\]]*)\]", text)
            if lists:
                seg = lists[-1]
            else:
                # 截断场景（模型输出被 max_tokens 截掉右括号）：取最后一个 "[" 之后的数字
                tail = text.rsplit("[", 1)[-1] if "[" in text else ""
                seg = tail
        raw = [int(x) for x in re.findall(r"\d+", seg)]
    out, seen = [], set()
    for item in raw:
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise ValueError("empty selection")
    return out


def _parse_ranking(text, n):
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    data = json.loads(text)
    raw = data.get("ranking") or data.get("order") or []
    out, seen = [], set()
    for item in raw:
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise ValueError("empty ranking")
    return out


class LLMReasoningReranker:
    """在 4B cross-encoder 之后的推理式重排（可选、opt-in）。"""

    def __init__(self, base_url, api_key, model, pool=30, layout="round",
                 doc_chars=1400, topk_ask=10, timeout=60.0, retries=2,
                 workers=1, mate_chars=600, prompt_mode="round_rank"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.pool = int(pool)
        self.layout = layout
        self.doc_chars = int(doc_chars)
        self.topk_ask = int(topk_ask)
        self.mate_chars = int(mate_chars)
        # round_rank=产线原提示（排序）| evidence_set=集合级多选（S3 实测更优，需配合门控）
        self.prompt_mode = (prompt_mode or "round_rank").lower()
        self.retries = max(1, int(retries))
        self.client = httpx.Client(timeout=timeout)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "ok": 0, "fallback": 0}
        self._mate_cache = {}

    # ---------- candidate rendering ----------
    def _render(self, memory, mate_of):
        text = (memory.get("index_text") or "").strip()
        if self.layout == "round" and mate_of:
            mate = mate_of.get(memory.get("id"))
            if mate:
                return (text + "\n<same-round other turn> " + mate)[: self.doc_chars]
        return text[: self.doc_chars]

    # ---------- LLM ----------
    def _call(self, prompt, n):
        maxtok = 400 if self.prompt_mode == "evidence_set" else 200
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": maxtok,
                   "messages": [{"role": "user", "content": prompt}]}
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.client.post(self.base_url + "/chat/completions",
                                     headers={"Authorization": f"Bearer {self.api_key}"},
                                     json=payload)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                if self.prompt_mode == "evidence_set":
                    return _parse_selected(content, n)
                return _parse_ranking(content, n)
            except Exception as exc:      # 任何失败都退化为原排序，绝不让 /search 失败
                last = exc
                logger.warning("[llm_rerank] attempt %d/%d failed: %s", attempt, self.retries, exc)
        raise RuntimeError(f"llm rerank failed: {last}")

    # ---------- pipeline interface ----------
    def notify_add(self, user_id, request_id):
        hook = getattr(self.base, "notify_add", None)
        if hook is not None:
            hook(user_id, request_id)

    def rerank(self, query, hits, mate_of=None):
        """hits: [(memory, score)] 已按 4B 分数降序。返回同集合、分数非递增。"""
        with self.lock:
            self.stats["calls"] += 1
        if not hits:
            return hits
        head, tail = hits[: self.pool], hits[self.pool:]
        listing = "\n".join(
            f"[{j}] {self._render(m, mate_of)}" for j, (m, _) in enumerate(head))
        if self.prompt_mode == "evidence_set":
            prompt = (EVIDENCE_SET_PROMPT.replace("__Q__", query)
                      .replace("__C__", listing))
        else:
            prompt = (PROMPT.replace("__K__", str(min(self.topk_ask, len(head))))
                      .replace("__Q__", query).replace("__C__", listing))
        try:
            order = self._call(prompt, len(head))
            with self.lock:
                self.stats["ok"] += 1
        except Exception:
            with self.lock:
                self.stats["fallback"] += 1
            return hits                      # 退化：保持 4B 排序
        ranked = [head[j] for j in order] + [h for j, h in enumerate(head)
                                            if j not in set(order)]
        span = (ranked[0][1] - ranked[-1][1]) if len(ranked) > 1 else 1.0
        out, floor = [], None
        for pos, (m, s) in enumerate(ranked):
            value = max(s, ranked[0][1]) - 1e-6 * pos    # 保持非递增且不抬高基线
            if floor is not None and value > floor:
                value = floor
            out.append((m, value))
            floor = value
        return out + list(tail)

    def describe(self):
        return {"class": self.__class__.__name__, "model": self.model,
                "pool": self.pool, "layout": self.layout,
                "doc_chars": self.doc_chars, **self.stats}


def _dedup_by_request(hits):
    """按 request_id 去重：保留每个 request 的**首个**（= 排序最靠前）条目。

    返回 (uniq, extras)：uniq 为去重后的列表（原相对顺序），
    extras 为 request_id -> 被去掉的副本列表（原相对顺序）。
    """
    uniq, extras, seen = [], {}, set()
    for m, sc in hits:
        rid = m.get("request_id")
        if rid and rid in seen:
            extras.setdefault(rid, []).append((m, sc))
            continue
        if rid:
            seen.add(rid)
        uniq.append((m, sc))
    return uniq, extras


def _reexpand(ranked, extras):
    """把被去重的副本**插回其代表之后**，并保持分数非递增。"""
    if not extras:
        return ranked
    left = dict(extras)
    out, floor = [], None
    for m, sc in ranked:
        sc = float(sc) if floor is None else min(float(sc), floor)
        out.append((m, sc)); floor = sc
        for m2, _sc2 in left.pop(m.get("request_id"), []):
            out.append((m2, floor))       # 与代表同分，且保证非递增
    for copies in left.values():          # 代表没进 ranked 的极端情况
        for m2, sc2 in copies:
            sc2 = float(sc2) if floor is None else min(float(sc2), floor)
            out.append((m2, sc2)); floor = sc2
    return out


class LLMReasoningRerankPipeline:
    """把 LLMReasoningReranker 接到检索器尾部（与 RerankPipeline 同接口）。

    负责按 user 构建/缓存「同轮另一条」的文本映射（按 store.version 失效），
    这样候选表示的「轮结构」才是完整一轮 —— 这是实测 +0.0174 的来源。

    ``gate_margin``：内层（cross-encoder）**给不出区分度**时 top1≈top2 → margin 小，
    该子集恰是 LLM 集合级选择能加分的地方（低 margin 半区 ΔHit@1 +0.1154、高 margin 半区 −0.0250）。
    注意：margin 与「多证据」**正相关但非等价**——AUC 0.6816 [0.6496,0.7125]、富集仅 1.25×、且非单调
    （τ=0.193 时门控内 329 多证据 / 191 单证据）。所以门控筛的是「内层无区分度」，不是「多证据题」。
    诊断落盘见 ``result/gate-vs-set__memgallery__text-route-1040__margin-vs-ngold.json``。
    """

    def __init__(self, base, reranker, store, max_users=32, mate_chars=600,
                 gate_margin=0.0, skip_marker="", dedup_rounds=False):
        import collections
        self.base = base
        self.reranker = reranker
        self.store = store
        self.mate_chars = int(mate_chars)
        self.gate_margin = float(gate_margin or 0.0)
        self.skip_marker = skip_marker or ""
        # 按 request 去重后再喂 LLM（见下）：round_emit=adjacent 会让同一轮的两条都进池，
        # 而 round 单位打分给它们**完全相同的分数** → 池的窗口里一半槽位是重复副本，
        # 多选型提示会把预算浪费在重复对上（实测前 12 槽只有 ~6 个不同轮、相邻 tie 6.23 对）。
        self.dedup_rounds = bool(dedup_rounds)
        self.max_users = max(1, int(max_users))
        self._cache = collections.OrderedDict()
        self._lock = threading.RLock()

    def _mate_of(self, user_id):
        version = self.store.version[user_id]
        with self._lock:
            hit = self._cache.get(user_id)
            if hit and hit["version"] == version:
                self._cache.move_to_end(user_id)
                return hit["mate"]
        groups = {}
        for m in self.store.list_memories(user_id):
            groups.setdefault(m.get("request_id"), []).append(m)
        mate = {}
        for _rid, ms in groups.items():
            for m in ms:
                text = "\n".join((o.get("index_text") or "") for o in ms
                                  if o.get("id") != m.get("id"))
                mate[m.get("id")] = text[: self.mate_chars]
        with self._lock:
            self._cache[user_id] = {"version": version, "mate": mate}
            self._cache.move_to_end(user_id)
            while len(self._cache) > self.max_users:
                self._cache.popitem(last=False)
        return mate

    def notify_add(self, user_id, request_id):
        hook = getattr(self.base, "notify_add", None)
        if hook is not None:
            hook(user_id, request_id)

    def search(self, user_id, query, limit):
        hits = self.base.search(user_id, query, max(self.reranker.pool, limit))
        if not hits:
            return []
        # 路径硬闸：命中标记（如 image 路的 "[Query image]"）则完全不走 LLM 档
        if self.skip_marker and self.skip_marker in (query or ""):
            return hits[:limit]
        # 门控：内层（cross-encoder）在多证据题上无区分度（top1≈top2，margin 小），
        # 那正是 LLM 集合级选择能加分的地方；margin 大 = 内层已能唯一确定证据 → 不调 LLM。
        gate = float(getattr(self, "gate_margin", 0.0) or 0.0)
        if gate > 0.0 and len(hits) >= 2:
            # **必须按 request 去重后再算 margin**：round 单位下同一轮的两条消息得分相同，
            # 直接取 hits[0]-hits[1] 会恒等于 0（门控永不触发）。
            try:
                seen_rid, req_scores = set(), []
                for _m, _s in hits:
                    _rid = _m.get("request_id")
                    if _rid in seen_rid:
                        continue
                    seen_rid.add(_rid)
                    req_scores.append(float(_s))
                    if len(req_scores) >= 2:
                        break
                margin = (req_scores[0] - req_scores[1]) if len(req_scores) >= 2 else -1.0
            except Exception:
                margin = -1.0
            if margin > gate:
                with self.reranker.lock:
                    self.reranker.stats["gated"] = self.reranker.stats.get("gated", 0) + 1
                return hits[:limit]
        try:
            mate = self._mate_of(user_id)
        except Exception:
            logger.exception("[llm_rerank] mate build failed; falling back to raw layout")
            mate = {}
        if not self.dedup_rounds:
            return self.reranker.rerank(query, hits, mate)[:limit]
        # 按 request 去重后再喂 LLM；排完把副本插回（LLM 只看到"不同轮"）
        uniq, extras = _dedup_by_request(hits)
        with self.reranker.lock:
            self.reranker.stats["dedup_dropped"] = (
                self.reranker.stats.get("dedup_dropped", 0) + (len(hits) - len(uniq)))
            self.reranker.stats["dedup_uniq_window"] = len(uniq[: self.reranker.pool])
        ranked = self.reranker.rerank(query, uniq, mate)
        return _reexpand(ranked, extras)[:limit]

    def describe(self):
        info = {"class": self.__class__.__name__}
        for name, obj in (("base", self.base), ("reranker", self.reranker)):
            fn = getattr(obj, "describe", None)
            info[name] = fn() if callable(fn) else {"class": obj.__class__.__name__}
        return info
