"""L1 服务：异步提取 + 查询路由（聚合类问题优先查账本）。

设计（对应需求）：
- `/add` 只做**入队**（幂等：user_id+memory_id），不阻塞
- 后台线程按用户批量抽取（12 条/批），带图片描述上下文
- 抽取失败 → profile_ingest 标记 failed，可重试；原始记忆不受影响
- 查询：`aggregate(user_id, question)` → 类别识别 → SQL 求和 → 带 coverage_status
"""
from __future__ import annotations

import logging
import os
import queue
import re
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from .extractor import extract_batch
from .entity_resolver import resolve, match_vague_item
from .fact_store import FactStore

logger = logging.getLogger("amc.profile")

# 批大小与并发数：平台全量（42 万条）下用默认值会严重超时 ——
# 离线审计实测：35,290 次 LLM 调用 ÷ 7h = 0.714s/次，而实际 2-5s/次 ⇒ 需要 3-7 倍并发。
# 生产可用 AMC_PROFILE_BATCH / AMC_PROFILE_WORKERS 调大。
import os as _os_profile
_BATCH = max(1, int(_os_profile.getenv("AMC_PROFILE_BATCH", "12")))
_WORKERS = max(1, int(_os_profile.getenv("AMC_PROFILE_WORKERS", "1")))
_MAX_CAPS = 60


class ProfileService:
    def __init__(self, profile_db: str, memory_db: str,
                 base_url: str = "", api_key: str = "", model: str = "gpt-4o-mini",
                 enabled: bool = True):
        self.store = FactStore(profile_db)
        self.memory_db = memory_db
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.enabled = bool(enabled and base_url and model)
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop = False
        if self.enabled:
            for _wi in range(_WORKERS):
                threading.Thread(target=self._loop, name=f"profile-worker-{_wi}", daemon=True).start()
            logger.info("[profile] enabled model=%s db=%s", model, profile_db)
        else:
            logger.info("[profile] disabled")

    # ---------- 入队 ----------
    def submit(self, user_id: str, memory_ids: List[str]) -> None:
        if not self.enabled or not user_id or not memory_ids:
            return
        for mid in memory_ids:
            self.store.mark_ingest(user_id, str(mid), "pending")
        self.q.put(user_id)

    # ---------- 后台循环 ----------
    def _loop(self) -> None:
        while not self._stop:
            try:
                user_id = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process_user(user_id)
            except Exception:
                logger.exception("[profile] process failed for %s", user_id)

    def _memories(self, user_id: str, memory_ids: List[str]) -> List[Dict[str, Any]]:
        if not memory_ids:
            return []
        conn = sqlite3.connect(self.memory_db)
        conn.row_factory = sqlite3.Row
        out = []
        try:
            for i in range(0, len(memory_ids), 200):
                chunk = memory_ids[i:i + 200]
                q = ("SELECT id,request_id,index_text,created_at FROM memories WHERE id IN (%s)"
                     % ",".join("?" * len(chunk)))
                out.extend(dict(r) for r in conn.execute(q, chunk).fetchall())
        finally:
            conn.close()
        return out

    def _image_captions(self, user_id: str) -> List[str]:
        conn = sqlite3.connect(self.memory_db)
        conn.row_factory = sqlite3.Row
        caps = []
        try:
            for r in conn.execute("SELECT index_text FROM memories WHERE user_id=?", (user_id,)):
                txt = r["index_text"] or ""
                # 我们生成的描述：[Image xxxx] ...
                m = re.search(r"\[Image[^\]]*\](.*)", txt, re.S)
                if m:
                    c = m.group(1).strip().split("\n")[0][:200]
                    if c:
                        caps.append(c)
                # 官方 caption（英文原句，常常是唯一能对上"英文物品名"的线索）
                for mo in re.finditer(r"Image Caption:\s*([^\n\[]{4,120})", txt):
                    c = mo.group(1).strip()
                    if c and c not in caps:
                        caps.insert(0, c)     # 官方 caption 放前面，避免被截断
        finally:
            conn.close()
        return caps[:_MAX_CAPS]

    def extract_for(self, user_id: str, memory_ids, limit: int = 36) -> int:
        """**按需抽取**：只抽"本次查询命中的记录"，已 done 的跳过。

        动机（2026-11-03 设计约束方案 C）：全量后台抽取是 96,957 条 / 约 8,080 次 LLM，
        而其中绝大多数是"平台根本不会问到"的记录 ⇒ 改成"被问到才抽"：
          · 每题最多 limit 条（12 条/批 ⇒ 36 条 = 3 批）
          · profile_ingest 已 done 的直接跳过（同一记录只抽一次）
          · 不做再排队（requeue=False）⇒ 后台 worker 不需要跑
        """
        if not self.enabled or not user_id:
            return 0
        ids = [str(x) for x in (memory_ids or []) if str(x).startswith("m_")]
        #   理由：平台 Search 的 500/502/503/504 会**自动退避重试**，
        #   而我们的"重试缓存"会在首次跑完后落盘 ⇒ 重试秒回 ⇒ 慢一次可接受。
        #   `limit <= 0` 表示不截断（保留参数以兼容旧调用）。
        if limit and int(limit) > 0:
            ids = ids[:int(limit)]
        if not ids:
            return 0
        try:
            st = self.store.ingest_status(user_id, ids)
        except Exception:
            logger.exception("ingest_status failed")
            st = {}
        todo = [i for i in ids if st.get(i) != "done"]
        if not todo:
            return 0
        try:
            self._process_user(user_id, only_ids=todo)
        except Exception:
            logger.exception("extract_for failed")
            return 0
        return len(todo)

    def _process_user(self, user_id: str, only_ids=None) -> None:
        if only_ids:
            pending = [str(x) for x in only_ids]
        else:
            conn = sqlite3.connect(self.store.path)
            rows = conn.execute(
                "SELECT memory_id FROM profile_ingest WHERE user_id=? AND status='pending' LIMIT 200",
                (user_id,)).fetchall()
            conn.close()
            pending = [r[0] for r in rows]
        if not pending:
            return
        mems = self._memories(user_id, pending)
        if not mems:
            for mid in pending:
                self.store.mark_ingest(user_id, mid, "failed")
            return
        caps = self._image_captions(user_id)
        done = 0
        for k in range(0, len(mems), _BATCH):
            chunk = mems[k:k + _BATCH]
            items = [{"i": j, "text": m["index_text"] or "", "date": (m["created_at"] or "")[:10],
                      "memory_id": m["id"], "request_id": m["request_id"]} for j, m in enumerate(chunk)]
            try:
                _cats = self.store.list_categories(user_id)
            except Exception:
                _cats = []
            _raw = extract_batch(items, self.base_url, self.api_key, self.model,
                                 image_captions=caps, categories=_cats)
            if not _raw:
                # ⚠️ 原来 `or {}` 后照常把整批标 done —— 抽取失败会被**静默当成"没有交易"**，
                # 证据永久丢失而 coverage 仍报 complete_enough。
                # 现在标 failed（可观测、可重试），不再标 done。
                for it in items:
                    self.store.mark_ingest(user_id, it["memory_id"], "failed")
                logger.warning("[profile] user=%s 批次抽取失败，%d 条标 failed（可重试）",
                               user_id[-12:], len(items))
                continue
            out = _raw
            for t in out.get("transactions") or []:
                src = items[t["i"]]
                cat = resolve(t.get("item", ""), t.get("category", ""))
                status = t.get("status", "pending")
                note = ""
                if cat == "other" or cat.startswith("other_"):
                    hit = match_vague_item(t.get("item", ""), t.get("quote", ""), caps)
                    if hit:
                        cat, status, note = hit[1], "pending", "image_match"
                # 登记类别（LLM 可复用或新建），供后续"对象级"聚合匹配
                try:
                    self.store.upsert_category(
                        user_id, cat,
                        description=str(t.get("category_desc") or ""),
                        parent=str(t.get("parent") or "") or (cat if cat in
                            ("coffee_maker","coffee_accessory","kitchen_appliance","travel","food","electronics") else ""))
                except Exception:
                    logger.exception("upsert_category failed")

                self.store.upsert_transaction(
                    user_id, category=cat, amount_minor=t["amount_minor"],
                    currency=t.get("currency") or "USD", item=t.get("item", ""),
                    occurred_at=t.get("occurred_at") or src["date"], status=status,
                    confidence=0.9 if status == "confirmed" else 0.6,
                    sources=[{"memory_id": src["memory_id"], "request_id": src["request_id"],
                              "quote": (t.get("quote") or "") + (f" [{note}]" if note else "")}])
            for f in out.get("facts") or []:
                src = items[f["i"]]
                self.store.add_fact(user_id, fact_type=f.get("fact_type", "preference"),
                                    predicate=f.get("predicate", ""), obj=f.get("object", ""),
                                    polarity=f.get("polarity", "positive"), confidence=0.8,
                                    source_memory_ids=[src["memory_id"]],
                                    valid_from=f.get("occurred_at") or f.get("due_at") or "")
            for it in items:
                self.store.mark_ingest(user_id, it["memory_id"], "done")
            done += len(items)
        logger.info("[profile] user=%s ingested=%d", user_id[-12:], done)
        # 按需模式（only_ids）不再排队：否则会把剩余积压全抽，违背"被问到才抽"
        if only_ids:
            return
        try:
            left = self.store.conn.execute(
                "SELECT COUNT(*) FROM profile_ingest WHERE user_id=? AND status='pending'",
                (user_id,)).fetchone()[0]
            if left:
                logger.info("[profile] user=%s 还有 %d 条待处理，继续排队", user_id[-12:], left)
                self.q.put(user_id)
        except Exception:
            logger.exception("requeue check failed")

    # ---------- 残留检查：发现"未入账"的金额 ----------
    _AMT = None

    def residual_check(self, user_id: str, limit: int = 400) -> Dict[str, Any]:
        """扫描该用户记忆里出现过的金额，找出**账本里没有**的那些。

        这是 coverage 的第二重含义：不仅"入库任务完成"，还要"账面没漏"。
        """
        import re as _re
        amt_re = _re.compile(r"[\$£€]\s?(\d[\d,]*(?:\.\d{1,2})?)|(\d[\d,]*(?:\.\d{1,2})?)\s?(?:dollars|bucks|USD|EUR|GBP)")
        conn = sqlite3.connect(self.memory_db)
        conn.row_factory = sqlite3.Row
        mem_amounts = []
        try:
            for r in conn.execute(
                    "SELECT id,index_text FROM memories WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                    (user_id, int(limit))):
                t = r["index_text"] or ""
                for m in amt_re.finditer(t):
                    #   ① 第一人称购买动词（I bought / I paid / it cost me / hit my account …）
                    #   ② 且金额前紧邻不能是 worth/under/over/budget/price/saving（那是讨论，不是支出）
                    _ctx = t[max(0, m.start() - 110):m.end() + 60].lower()
                    _fp = re.search(
                        r"\b(i (?:bought|paid|picked up|grabbed|ordered|purchased|spent)|"
                        r"(?:cost|charged|set) me|hit my account|out of pocket|"
                        r"\bi (?:got|grabbed) (?:it|this|that)\b)|"
                        r"(?:^|[.!?]\s+|\balso\s+|\band\s+|\bthen\s+)"
                        r"(?:bought|paid|picked up|grabbed|ordered|purchased)\b", _ctx)
                    if not _fp:
                        continue
                    _pre = t[max(0, m.start() - 18):m.start()].lower()
                    if re.search(r"(worth|under|over|below|budget|price|saving|save|rrp|retail|msrp)\s*[:~]?\s*[\$£€]?\s*$", _pre):
                        continue
                    raw = (m.group(1) or m.group(2) or "").replace(",", "")
                    try:
                        v = int(round(float(raw) * 100))
                    except Exception:
                        continue
                    if v <= 0:
                        continue
                    mem_amounts.append((v, r["id"], t[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")))
        finally:
            conn.close()
        known = set()
        for r in self.store.conn.execute(
                "SELECT amount_minor FROM profile_transactions WHERE user_id=?", (user_id,)):
            known.add(int(r[0]))
        unaccounted = [(v, mid, snip) for (v, mid, snip) in mem_amounts if v not in known]
        # 去重（同一金额多次提及只报一次）
        seen = set()
        uniq = []
        for v, mid, snip in unaccounted:
            if v in seen:
                continue
            seen.add(v)
            uniq.append({"amount_minor": v, "memory_id": mid, "snippet": snip.strip()[:120]})
        return {"unaccounted": uniq[:10], "n": len(uniq)}

    def store_mem_conn(self):
        if getattr(self, "_mem_conn", None) is None:
            self._mem_conn = sqlite3.connect(self.memory_db, check_same_thread=False)
        return self._mem_conn

    # ---------- 定向补查：对"未入账"的记忆重新抽取 ----------
    def backfill(self, user_id: str, unaccounted: List[Dict[str, Any]], max_items: int = 3) -> int:
        """只对未入账金额所在的那几条记忆重抽一次（补查，不是全量重跑）。

        返回新增交易数。失败不影响已有账本。
        """
        ids = []
        for u in (unaccounted or [])[:max_items]:
            mid = str(u.get("memory_id") or "")
            if mid and mid not in ids:
                ids.append(mid)
        if not ids:
            return 0
        mems = self._memories(user_id, ids)
        if not mems:
            return 0
        caps = self._image_captions(user_id)
        items = [{"i": j, "text": m["index_text"] or "", "date": (m["created_at"] or "")[:10],
                  "memory_id": m["id"], "request_id": m["request_id"]} for j, m in enumerate(mems)]
        out = extract_batch(items, self.base_url, self.api_key, self.model, image_captions=caps) or {}
        added = 0
        for t in out.get("transactions") or []:
            src = items[t["i"]]
            cat = resolve(t.get("item", ""), t.get("category", ""))
            status = t.get("status", "pending")
            if cat == "other":
                hit = match_vague_item(t.get("item", ""), t.get("quote", ""), caps)
                if hit:
                    cat, status = hit[1], "pending"
            before = self.store.conn.execute(
                "SELECT COUNT(*) FROM profile_transactions WHERE user_id=? AND category=? AND amount_minor=?",
                (user_id, cat, t["amount_minor"])).fetchone()[0]
            self.store.upsert_transaction(
                user_id, category=cat, amount_minor=t["amount_minor"],
                currency=t.get("currency") or "USD", item=t.get("item", ""),
                occurred_at=t.get("occurred_at") or src["date"], status=status,
                confidence=0.85 if status == "confirmed" else 0.6,
                sources=[{"memory_id": src["memory_id"], "request_id": src["request_id"],
                          "quote": (t.get("quote") or "") + " [backfill]"}])
            if not before:
                added += 1
        if added:
            logger.info("[profile] backfill user=%s memories=%d added=%d", user_id[-10:], len(items), added)
        return added

    # ---------- 查询 ----------
    def aggregate(self, user_id: str, question: str, include_pending: bool = True) -> Optional[Dict[str, Any]]:
        """聚合类问题：从账本求和。返回 None 表示账本里没有可用信息。"""
        # **实体门槛**：问题里出现句中大写专有名词（Aberdeen / Waitrose / BRITA / Cairo）
        # 说明问的是**具体对象**，而账本只按类别记账 → 归并会算错，直接不掺和。
        # 纯类别问题（"how much on coffee makers"）没有专有名词 → 正常走账本。
        _mid_caps = [w for i, w in enumerate(question.split())
                     if i > 0 and w[:1].isupper() and not w.isupper()]
        if _mid_caps:
            # 实体级问题：先看**可增长类别表**里有没有对应对象（如 hotel_aberdeen）
            _matched = self._match_category(self.store, user_id, question)
            if not _matched:
                return None
            cat = _matched
        else:
            cat = self._question_category(question)
        terms = self._question_object_terms(question)
        # 回到**类别聚合**（这是账本原本的正确用法：类别级问题）。
        # 实体级问题已在上面的"专有名词门槛"里被挡掉。
        # 那是错答案而不是"兜底答案"。
        if not cat:
            return None
        res = self.store.sum_spending(user_id, category=cat)
        if not res["transactions"]:
            conn = sqlite3.connect(self.store.path)
            try:
                nref = conn.execute(
                    "SELECT COUNT(*) FROM profile_transactions WHERE user_id=? AND category=? AND status='refunded'",
                    (user_id, cat)).fetchone()[0]
            finally:
                conn.close()
            if nref:
                cov = self.store.coverage(user_id)
                return {"category": cat, "transactions": [],
                        "refunded": nref, "coverage_status": cov["coverage_status"], "coverage": cov}
            return None
        rows = list(res["transactions"])
        if include_pending:
            conn = sqlite3.connect(self.store.path)
            try:
                more = conn.execute(
                    "SELECT item,amount_minor,currency,occurred_at,status FROM profile_transactions "
                    "WHERE user_id=? AND category=? AND status='pending' ORDER BY occurred_at",
                    (user_id, cat)).fetchall()
            finally:
                conn.close()
            rows = rows + [{"item": r[0], "amount_minor": r[1], "currency": r[2],
                            "occurred_at": r[3], "status": r[4]} for r in more]
        by_cur: Dict[str, int] = {}
        for r in rows:
            by_cur[r["currency"] or "USD"] = by_cur.get(r["currency"] or "USD", 0) + int(r["amount_minor"])
        # 附上每条交易的原始证据（memory_id），供 API 层把原文一起返回
        try:
            tids = [r.get("transaction_id") for r in rows if r.get("transaction_id")]
            src = {}
            if tids:
                q = ("SELECT transaction_id, memory_id FROM profile_transaction_sources WHERE transaction_id IN (%s)"
                     % ",".join("?" * len(tids)))
                for tid, mid in self.store.conn.execute(q, tids):
                    src.setdefault(tid, []).append(mid)
            # 来源排序：优先"原文里真的提到该商品"的记忆（避免把恰好含相同金额的闲聊当来源）
            def _rank_sources(tid, item):
                ids = src.get(tid, [])
                if len(ids) <= 1:
                    return ids
                import re as _re
                toks = {w for w in _re.findall(r"[a-z]{3,}", (item or "").lower())}
                scored = []
                try:
                    for mid in ids:
                        row = self.store_mem_conn().execute(
                            "SELECT index_text FROM memories WHERE id=?", (mid,)).fetchone()
                        t = (row[0] if row else "") or ""
                        tl = t.lower()
                        sc = sum(1 for w in toks if w in tl)
                        scored.append((sc, mid))
                except Exception:
                    return ids
                scored.sort(key=lambda x: -x[0])
                _hit = [m for sc, m in scored if sc > 0]
                return _hit or [m for _, m in scored]
            for r in rows:
                r["source_memory_ids"] = _rank_sources(r.get("transaction_id"), r.get("item") or "")
        except Exception:
            logger.exception("aggregate sources failed")
        cov = self.store.coverage(user_id)
        resid = self.residual_check(user_id)
        status = cov["coverage_status"]
        if status == "complete_enough" and resid.get("n"):
            status = "partial"          # 账面外还有金额 → 不能声称"已核清"
        #    原因：赛题硬规则是"Search 不得替答案端做计算"，所以上层**只列单笔、不算和**；
        #    `grep totals` 全仓 **0 处消费** ⇒ 留着只会让未来有人误用（给出低估的确定数字）。
        #    保留 `by_cur` 的计算（下面这几行仍会跑）但**不对外暴露**。
        _ = by_cur
        return {"category": cat, "transactions": rows,
                "coverage_status": status, "coverage": cov,
                "unaccounted": resid.get("unaccounted") or []}

    def facts_note(self, user_id: str, question: str) -> Optional[str]:
        """偏好/事件/计划类问题：从 profile_facts 取活动事实，拼成一句提示。"""
        import re as _re
        _STOP = {"what", "which", "like", "love", "prefer", "favourite", "favorite", "into",
                 "interested", "does", "did", "the", "and", "for", "you", "your", "have", "has",
                 "when", "where", "who", "why", "how", "now", "move", "moved", "live", "living",
                 "planning", "plan", "doing", "get", "got"}
        kws = [w for w in _re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}|[\u4e00-\u9fff]{2,}", question or "")
               if w.lower() not in _STOP][:8]
        rows = self.store.query_facts(user_id, active_only=True, limit=50)
        if not rows:
            return None
        # 关键词过滤（词边界；无命中则全部）
        picked = []
        for r in rows:
            blob = (str(r.get("object") or "") + " " + str(r.get("predicate") or "")).lower()
            if any((_re.search(r"\b%s\b" % _re.escape(k.lower()), blob) if k.isascii() else (k in blob))
                   for k in kws):
                picked.append(r)
        rows = picked or rows[:6]
        parts = []
        for r in rows[:6]:
            ft = r.get("fact_type") or "preference"
            pol = r.get("polarity") or "positive"
            obj = r.get("object") or ""
            if ft == "event":
                parts.append(f"{r.get('predicate') or '发生'} {obj}" + (f"（{r.get('valid_from')}）" if r.get("valid_from") else ""))
            elif ft == "commitment":
                parts.append(f"计划：{obj}")
            else:
                parts.append(f"{'喜欢' if pol == 'positive' else '不喜欢'}{obj}")
        n_neg = sum(1 for r in rows if (r.get("polarity") or "positive") != "positive"
                    and (r.get("fact_type") or "preference") == "preference")
        note = "；".join(parts)
        #    理由：facts 覆盖率只有 0.7%（998/137,694）⇒ 若题问"雇主/住址"，
        #    返回一条旧值而不说"可能不完整"，就是"确定但可能错"（C 类危险输出）。
        #    但 `return None` 会让答案端完全失去线索 ⇒ 所以"标注"而不是"丢弃"。
        try:
            _cov = self.store.coverage(user_id)
            _st = _cov.get("coverage_status")
            if _st != "complete_enough":
                _annot = (f"（档案覆盖不足：status={_st}，"
                          f"pending={_cov.get('pending')}，"
                          f"已抽事实 {_cov.get('transactions')} 笔；以下内容可能不完整）\n")
            else:
                _annot = ""
        except Exception:
            _annot = ""
        _body = note + (f"（其中 {n_neg} 条为否定偏好）" if n_neg else "")
        return (_annot + _body) if _annot else _body

    # 问句里的"功能词"，不属于消费对象
    _QSTOP = set("""how much many total altogether spent spend cost costs costed paid pay paying
        did do does was were is are my me i the a an for on in at of to and or what which
        price prices worth charge charged fee fees bill bills receipt total overall during
        have has had from most recent last order ordered stay stayed night nights
        remember seeing old very really just about there was were this that these those
        花 花费 花了 一共 总共 合计 多少 钱 我 的 了 是 在 给""".split())

    @classmethod
    def _question_object_terms(cls, question: str) -> set:
        """把问题里的"消费对象"抽成匹配词（实体或类别都一样处理）。

        例：
          "How much did I spend on the hotel in Aberdeen?"  -> {hotel, aberdeen}
          "How much total on coffee makers?"                -> {coffee, maker}
        """
        q = (question or "").lower()
        q = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", q)
        toks = {t for t in q.split() if len(t) >= 2 and t not in cls._QSTOP}
        # 词形归一（makers -> maker）
        norm = set()
        for t in toks:
            norm.add(t)
            if len(t) > 3 and t.endswith("s"):
                norm.add(t[:-1])
        return norm

    @classmethod
    def _match_category(cls, store, user_id: str, question: str) -> Optional[str]:
        """用问题里的"对象词"去匹配**可增长类别表**（如 hotel_aberdeen）。

        比规则表更细：规则只有 6 个桶，这里能命中 LLM 在 /add 时新建的细粒度类别。
        """
        try:
            cats = store.list_categories(user_id)
        except Exception:
            # ⚠️ 静默返回 None 会让"DB 异常"伪装成"没有匹配类别" ⇒ 聚合题静默降级。
            logger.exception(
                "[profile] list_categories 失败 → 类别匹配降级为 None"
                "（输入规模: query %d 字, user %s）", len(question or ""), user_id[-12:])
            return None
        if not cats:
            return None
        terms = cls._question_object_terms(question)
        if not terms:
            return None
        # ⚠️ 关键修正（2026-10-31）：问题用的是**通用类别词**时（"how much did I spend on hotels"），
        # 必须用**粗类**（travel）做聚合，而不是匹配到某一笔细粒度类别（hotel_conrad_cairo）。
        # 否则"在开罗住酒店一共花多少"会退化成"康莱德那一家花多少"，
        # 而且 service.py 里 coverage=complete_enough 会**短路整个图书馆检索** ⇒ 用一笔冒充一组。
        # 判据：**细粒度类别的所有词是否都被问题覆盖**。
        #   "hotels in Cairo"       问题词 {hotel, cairo} ⊅ {hotel, conrad, cairo} ⇒ 用粗类 travel ✓
        #   "the Conrad Cairo hotel" 问题词 {conrad, cairo, hotel} ⊇ 类别词     ⇒ 用细类 hotel_conrad_cairo ✓
        # 这条比"有没有专有名词"准：Cairo 也是专有名词，但它是**聚合限定词**不是实体名。
        _coarse = resolve(question)
        best, best_score, best_cover = None, 0, False
        for c in cats:
            name = str(c.get("name") or "").lower()
            desc = str(c.get("description") or "").lower()
            toks = set(re.findall(r"[a-z0-9]+", name.replace("_", " "))) | set(re.findall(r"[a-z0-9]+", desc))
            score = len(terms & toks)
            _covered = bool(toks) and toks <= terms
            if score > best_score or (_covered and not best_cover):
                best, best_score, best_cover = name, score, _covered
        # 细粒度类别被问题**完全覆盖** ⇒ 问的是这个具体类别，用它
        if best and best_cover:
            return best
        # 否则优先粗类（通用类别词 → 按大类聚合）—— 但**必须先证明粗类下真有交易**。
        # ⚠️ `sum_spending` 是 `category=?` **精确匹配、不做父子展开**，
        # 而细粒度类别（hotel_conrad_cairo / flight_xxx）的 parent 多数为空。
        # 若不检查就用粗类，会从"错但非零"退化成"零"（更糟）。
        if _coarse and _coarse != "other":
            try:
                _probe = store.sum_spending(user_id, category=_coarse)
                if (_probe or {}).get("transactions"):
                    return _coarse
            except Exception:
                pass
        # 至少命中 2 个对象词，或唯一候选命中 1 个
        if best and (best_score >= 2 or (best_score == 1 and len(cats) == 1)):
            return best
        return None

    @staticmethod
    def _question_category(question: str) -> Optional[str]:
        cat = resolve(question)
        return None if cat == "other" else cat
