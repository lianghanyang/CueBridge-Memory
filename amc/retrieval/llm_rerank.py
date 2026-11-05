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
import os
import random
import time
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


import re as _re2
_FACTOR_TOK = _re2.compile(r"[A-Za-z][A-Za-z\-]+|\d{2,4}")
_FACTOR_STOP = set("""a an the is are was were be been being of to in on at for with about and or but if then than
that this these those it its as by from into over under after before between during within without which who
whom whose what when where why how do does did not no yes can could would should may might must also more most
some any all each other such only own same so too very just""".split())


def _query_factors(query: str):
    out = []
    for t in _FACTOR_TOK.findall(query or ""):
        t = t.lower()
        if len(t) >= 3 and t not in _FACTOR_STOP and t not in out:
            out.append(t)
    return out


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
            if not (isinstance(data, dict) and ("selected" in data or "ranking" in data)):
                continue                  # 没有字段 = 解析失败
            raw = data.get("selected") or data.get("ranking") or []
            break
        except Exception:
            continue
    if raw is None:
        # 没有任何 selected/ranking 结构 = 解析失败（不是"无证据"）
        if not re.search(r'"?selected"?|"?ranking"?', text, re.I):
            raise ValueError("no selection field")
        import re as _re
        m = _re.search(r'"?selected"?\s*:\s*\[([^\]]*)\]', text)
        seg = m.group(1) if m else None
        if seg is None:                       # 被截断/无右括号
            m2 = _re.search(r'"?selected"?\s*:\s*\[?([0-9][0-9,\s]*)', text)
            seg = m2.group(1) if m2 else None
        if seg is None:
            lists = _re.findall(r"\[([^\]]*)\]", text)
            seg = lists[-1] if lists else ""
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
        return []                      # LLM 明确表示"没有可作证据的候选"
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
        return []                       # 模型判定"无相关候选"（合法结果，不是失败）
    return out


class LLMReasoningReranker:
    """在 4B cross-encoder 之后的推理式重排（可选、opt-in）。"""

    def __init__(self, base_url, api_key, model, pool=30, layout="round",
                 doc_chars=1400, topk_ask=10, timeout=60.0, retries=3,
                 workers=1, mate_chars=600, prompt_mode="round_rank"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        # 网关池：主用 + 备用；失败立即切换，整轮失败后轮转重试
        from amc.llm_endpoints import (endpoints as _eps, endpoint_tag as _etag,
                                       pick as _pick, demote as _demote, promote as _promote)
        self.endpoints = _eps(base_url or "", api_key or "", model or "")
        self._ep = 0
        self._etag = _etag
        self._pick, self._demote, self._promote = _pick, _demote, _promote
        self.pool = int(pool)
        self.layout = layout
        self.doc_chars = int(doc_chars)
        self.topk_ask = int(topk_ask)
        self.mate_chars = int(mate_chars)
        # round_rank=产线原提示（排序）| evidence_set=集合级多选（S3 实测更优，需配合门控）
        self.prompt_mode = (prompt_mode or "round_rank").lower()
        self.retries = max(1, int(retries))
        self._fail_streak = 0
        self._open_until = 0.0
        self.breaker_n = int(os.getenv("AMC_LLM_RERANK_BREAKER_N", "5") or 5)
        self.breaker_secs = float(os.getenv("AMC_LLM_RERANK_BREAKER_SECS", "600") or 600)
        # 单次 /search 的重排预算：默认 75s（够扛 2~4 次退避，又不至于让评测端超时）
        self.total_budget = float(os.getenv("AMC_LLM_RERANK_TOTAL_BUDGET", "600") or 600)
        self.client = httpx.Client(timeout=timeout)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "ok": 0, "fallback": 0}
        self._mate_cache = {}

    # ---------- candidate rendering ----------
    def _render(self, memory, mate_of):
        text = (memory.get("index_text") or "").strip()
        # doc_chars <= 0 ⇒ 不截断（整轮全文交给 LLM）。
        # 依据：长轮（3~5k 字）里答案常在后段（实测第 17/18 条），
        # 1400 字截断会把答案切掉，导致重排把 gold 排到 10 名以后（h03：截断 #10~#12、不截断 #2）。
        _cap = int(getattr(self, "doc_chars", 0) or 0)
        if self.layout == "round" and mate_of:
            mate = mate_of.get(memory.get("id"))
            if mate:
                full = text + "\n<same-round other turn> " + mate
                return full if _cap <= 0 else full[:_cap]
        return text if _cap <= 0 else text[:_cap]

    # ---------- LLM ----------
    def _call(self, prompt, n, mode=None):
        mode = mode or self.prompt_mode
        #   200 tokens 会被截断（模型常附带解释）⇒ JSON 解析失败 ⇒ 重试 3 次 ⇒ 260s 空等。
        #   round_rank 给 600；集合选择类给 400；其余 200。
        if mode == "round_rank":
            maxtok = 600
        elif mode in ("evidence_set", "evidence_set_factors", "evidence_set_missing"):
            maxtok = 400
        else:
            maxtok = 200
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": maxtok,
                   "messages": [{"role": "user", "content": prompt}]}
        last = None
        deadline = time.time() + self.total_budget
        attempt = 0
        _cycle = 0
        while True:
            attempt += 1
            if time.time() > deadline:
                break
            _ep = self._pick(self.endpoints)
            try:
                _payload = dict(payload); _payload["model"] = _ep["model"]
                try:
                    from amc.llm_stream import enabled as _ks_on, chat_text as _ks_chat
                    _use_stream = _ks_on()
                except Exception:
                    _use_stream = False
                if _use_stream:
                    content = _ks_chat(_ep["base"], _ep["key"], _ep["model"],
                                       _payload["messages"], max_tokens=_payload["max_tokens"],
                                       temperature=_payload.get("temperature", 0.0),
                                       timeout=float(getattr(self, "timeout", 60.0))).strip()
                else:
                    r = self.client.post(_ep["base"] + "/chat/completions",
                                         headers={"Authorization": f"Bearer {_ep['key']}"},
                                         json=_payload)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"].strip()
                self._promote(_ep)
                if mode in ("evidence_set", "evidence_set_factors", "evidence_set_missing"):
                    sel = _parse_selected(content, n)
                    return sel              # 可能是 []，表示"无证据"（不算失败）
                return _parse_ranking(content, n)   # 可能为 []（合法：都无关）
            except Exception as exc:
                last = exc
                # 关键：**下一个网关立刻试**，不等待；一整轮都失败后才退避
                self._demote(_ep)
                if len(self.endpoints) > 1 and _cycle < len(self.endpoints) - 1:
                    _cycle += 1
                    logger.warning("[llm_rerank] %s 失败(%s)，立即切换 → %s",
                                   self._etag(_ep), str(exc)[:60],
                                   self._etag(self.endpoints[self._ep]))
                    continue
                _cycle = 0
                wait = min(8.0, 0.8 * attempt)
                logger.warning("[llm_rerank] attempt %d 全部网关失败 (%s); %.1fs 后轮转重试 (budget left %.0fs)",
                               attempt, str(exc)[:60], wait, max(0.0, deadline - time.time()))
                time.sleep(wait)
        raise RuntimeError(f"llm rerank failed after {attempt-1} attempts / {self.total_budget:.0f}s: {last}")

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
        # ---- 熔断器：连续失败 N 次后，M 秒内跳过 LLM（保持 4B 顺序，避免全站 500/空）----
        if self._open_until and time.time() < self._open_until:
            with self.lock:
                self.stats["breaker_skip"] = self.stats.get("breaker_skip", 0) + 1
            return hits
        head, tail = hits[: self.pool], hits[self.pool:]
        if self.prompt_mode in ("evidence_set_factors", "evidence_set_missing"):
            qf = _query_factors(query)
            lines = []
            for j, (m, _) in enumerate(head):
                rendered = self._render(m, mate_of)
                toks = set(t.lower() for t in _FACTOR_TOK.findall(rendered))
                hit = [f for f in qf if f in toks]
                miss = [f for f in qf if f not in toks]
                line = f"[{j}] {rendered}"
                if self.prompt_mode == "evidence_set_factors":
                    if hit:
                        line += "\n    matched factors: " + ", ".join(hit[:8])
                    if miss:
                        line += "\n    missing factors: " + ", ".join(miss[:8])
                else:
                    if miss:
                        line += "\n    missing factors: " + ", ".join(miss[:8])
                lines.append(line)
            listing = "\n".join(lines)
        else:
            listing = "\n".join(
                f"[{j}] {self._render(m, mate_of)}" for j, (m, _) in enumerate(head))
        if self.prompt_mode in ("evidence_set", "evidence_set_factors", "evidence_set_missing"):
            prompt = (EVIDENCE_SET_PROMPT.replace("__Q__", query)
                      .replace("__C__", listing))
        else:
            prompt = (PROMPT.replace("__K__", str(min(self.topk_ask, len(head))))
                      .replace("__Q__", query).replace("__C__", listing))
        try:
            order = self._call(prompt, len(head))
            # LLM 只负责"筛选"：选中项按 **4B 分数降序**排列（同分才用 LLM 顺序），
            # 避免 evidence_set 的多选顺序把低分项排到高分项前面。
            # 对**前 K 名**（默认 3，即"领先组"）再用 round_rank 做一次显式排序：
            # evidence_set 是多选题，其顺序无依据；只对领先组重排，尽量不影响整体 R@gold。
            _K = int(getattr(self, "topk_rank", 3))
            if (_K > 0 and order and len(order) >= 2
                    and self.prompt_mode in ("evidence_set", "evidence_set_factors",
                                             "evidence_set_missing")):
                _grp = order[:_K]
                if len(_grp) >= 2:
                    _lines = [f"[{i}] {self._render(head[j][0], mate_of)}"
                              for i, j in enumerate(_grp)]
                    _p = (PROMPT.replace("__Q__", query)
                          .replace("__K__", str(len(_grp)))
                          .replace("__C__", "\n".join(_lines)))
                    _o2 = self._call(_p, len(_grp), mode="round_rank")
                    if _o2:
                        order[:_K] = [_grp[i] for i in _o2]
                    with self.lock:
                        self.stats["topk_rank"] = self.stats.get("topk_rank", 0) + 1
        except Exception as exc:
            with self.lock:
                self.stats["fallback"] += 1
            raise RuntimeError(f"llm rerank failed after {self.retries} attempts: {exc}") from exc
        with self.lock:
            self._fail_streak = 0
            self._open_until = 0.0
        ranked = [head[j] for j in order] + [h for j, h in enumerate(head)
                                            if j not in set(order)]
        # 分数语义：**暴露的 top-1 = 重排前的真实最高分**（"池内最佳匹配有多强"），
        # 其余按 LLM 名次微降 —— 既保证非递增，又让上层能据此判断"有没有证据"
        # （用于通用 abstention 门控），而不会像 max()/多重集那样把 top 抬到与名次无关。
        top_raw = max(float(s) for _, s in ranked) if ranked else 0.0
        # ⚠️ 衰减步长必须是 **1e-3**（原来是 1e-6），理由有两条：
        #     原来步长 1e-6 ⇒ 相邻分数差恰好贴容差边界，浮点/取整都可能越界。
        #  ② 步长 1e-6 意味着所有记录的分数只差最后一位 —— 平台（或答题端）
        #     只要把分数 round 到 4~5 位，就会**全部变成同一个数**，
        #     一旦按分数排序就可能打乱我们已经排好的顺序。
        # 1e-3 既保留"非递增"，又让分数在 3 位小数上可区分；top1 仍等于真实最高分。
        # 用**几何衰减**而不是线性衰减：
        #   · 线性 `top - 1e-3*i`：top 小时（如 0.005）50 条里只有 6 个可区分值，
        #     其余全被 clamp 到 0 ⇒ 名次信息丢失（实测见 PENDING_CHANGES 三点十四）。
        #   · 几何 `top * 0.98**i`：永远 >0、永远严格递减、4 位小数下基本都可区分
        #     （top=0.005、n=50 时仍有 32 个不同值）。top1 仍等于真实最高分。
        out = [(m, top_raw * (0.98 ** i)) for i, (m, _s) in enumerate(ranked)]
        _final = out + list(tail)
        #   导出 **LLM rerank 之后** 的排序（本地 4B 分 + LLM 选序的结果）。
        try:
            from amc import stage_dump as _sd
            _sd.dump("llm", _final, uid=getattr(self, "_last_uid", None), q=query)
        except Exception:
            pass
        return _final

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
        # 运行时覆盖：`AMC_LLM_RERANK_GATE_MARGIN_FILE` 指向一个只含数字的文件，
        # 每次调用都重新读 ⇒ **改阈值不需要重启**，可以在线扫 {0, 0.05, 0.1, 0.2}
        # 看"跳过多少次精排 / hit1 变多少"，扫完写回 0 或删文件即恢复。
        # 目的：这是目前**唯一能减少每查询 LLM 调用**的杠杆（省一次 ≈ 33%），
        # 而它的最优阈值必须在当代配置下实测，不能凭旧管线的数字。
        try:
            import os as _os_g
            _gf = _os_g.getenv("AMC_LLM_RERANK_GATE_MARGIN_FILE", "")
            if _gf:
                with open(_gf) as _fh:
                    gate = float((_fh.read() or "0").strip() or 0)
        except Exception:
            pass
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
        #   以便它导出 llm 阶段时能把同一次请求的各级结果归并到一起。
        try:
            self.reranker._last_uid = user_id
        except Exception:
            pass
        try:
            mate = self._mate_of(user_id)
        except Exception:
            logger.exception("[llm_rerank] mate build failed; falling back to raw layout")
            mate = {}
        # LLM 重排失败（超时/网关挂）**不能**让整个检索失败：
        # 退化到重排前的顺序，仍然交证据。这是"合同优先于质量"的兜底。
        if not self.dedup_rounds:
            try:
                return self.reranker.rerank(query, hits, mate)[:limit]
            except Exception as exc:
                logger.warning("[llm_rerank] 重排失败，退回原顺序: %s", str(exc)[:120])
                with self.reranker.lock:
                    self.reranker.stats["fallback"] = self.reranker.stats.get("fallback", 0) + 1
                return hits[:limit]
        # 按 request 去重后再喂 LLM；排完把副本插回（LLM 只看到"不同轮"）
        uniq, extras = _dedup_by_request(hits)
        with self.reranker.lock:
            self.reranker.stats["dedup_dropped"] = (
                self.reranker.stats.get("dedup_dropped", 0) + (len(hits) - len(uniq)))
            self.reranker.stats["dedup_uniq_window"] = len(uniq[: self.reranker.pool])
        try:
            ranked = self.reranker.rerank(query, uniq, mate)
            return _reexpand(ranked, extras)[:limit]
        except Exception as exc:
            logger.warning("[llm_rerank] 重排失败（dedup 路径），退回原顺序: %s", str(exc)[:120])
            with self.reranker.lock:
                self.reranker.stats["fallback"] = self.reranker.stats.get("fallback", 0) + 1
            return hits[:limit]

    def describe(self):
        info = {"class": self.__class__.__name__}
        for name, obj in (("base", self.base), ("reranker", self.reranker)):
            fn = getattr(obj, "describe", None)
            info[name] = fn() if callable(fn) else {"class": obj.__class__.__name__}
        return info
