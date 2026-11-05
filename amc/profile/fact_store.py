"""L1 事实库读写：交易账本 + 事实表 + 覆盖状态。

只做**确定性**的事：校验、去重、聚合。
事实抽取（LLM）在 extractor.py，本模块不调用任何模型。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional

from .schema import ensure_schema


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def tx_id(user_id: str, category: str, amount_minor: int, currency: str, occurred_at: str,
          item: str = "") -> str:
    """稳定 ID：同一(用户, 类别, 金额, 币种, 日期, 商品) 视为同一笔交易。

    ⚠️ 2026-10-31 修正：原来只用 `(user, category, amount, currency)`，
    导致**同日同额的多笔消费被合并成一笔**。离线复现（见 AUDIT_PROFILE.md）：
    3 笔 £3.50 的咖啡 → 只剩 1 行，sum=350（应为 1050），**少算 67%**。
    现在纳入 **日期 + 商品名**：
      · 同一笔在小票/邮件里重复出现 → 日期与商品一致 → 仍然合并 ✓
      · 不同日期的同额消费 → 日期不同 → 分开 ✓
    代价：同一笔若被解析出两个不同日期，会产生两行（重复计数）。
    权衡：重复计数会**高估**，合并会**低估**；金额题里低估更致命，故倾向前者。
    """
    _item = re.sub(r"[^a-z0-9]+", "", (item or "").lower())[:24]
    _day = (occurred_at or "")[:10]
    raw = f"{user_id}|{category}|{amount_minor}|{currency}|{_day}|{_item}"
    return "tx_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class FactStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        ensure_schema(self.conn)
        self.lock = __import__("threading").Lock()

    _STOP = frozenset("the of a an and for with to in on at".split())

    @classmethod
    def _tokens(cls, s: str) -> set:
        import re as _re
        return {w for w in _re.findall(r"[a-z]{3,}", (s or "").lower()) if w not in cls._STOP}

    @classmethod
    def _item_overlap(cls, a: str, b: str) -> int:
        return len(cls._tokens(a) & cls._tokens(b))

    _CAT_RANK = {"coffee_maker": 5, "coffee_accessory": 4, "kitchen_appliance": 3,
                 "electronics": 2, "travel": 2, "food": 1, "other": 0}

    def _merge_similar(self, user_id: str, category: str, amount_minor: int,
                       currency: str, item: str, occurred_at: str = "") -> Optional[str]:
        """同一(用户,金额,币种)且商品词面有重叠 → 视为同一笔，返回已有 transaction_id。

        避免"同一笔消费被抽成不同类别"造成重复计数（如 $85 同时是 coffee_maker 与 kitchen_appliance）。
        """
        rows = self.conn.execute(
            "SELECT transaction_id, category, item, status, occurred_at FROM profile_transactions "
            # 必须**同一天**才可能是同一笔：不同日期的同额消费是不同的交易
            # （3 笔 £3.50 的咖啡曾被合并成 1 笔，少算 67%）。
            "WHERE user_id=? AND amount_minor=? AND currency=? AND occurred_at=?",
            (user_id, int(amount_minor), (currency or "").upper(), occurred_at or "")).fetchall()
        best = None
        for r in rows:
            # ⚠️ 原来只要"有 1 个共同词"就合并，导致
            #   「£85 咖啡机」+「£85 咖啡豆」→ 合并成咖啡机，food 汇总变 0。
            # 改成 Jaccard >= 0.5（共同词占全部词的过半），才认为是同一商品。
            _ta = self._tokens(item); _tb = self._tokens(r["item"])
            _j = (len(_ta & _tb) / len(_ta | _tb)) if (_ta | _tb) else 0.0
            if _j >= 0.5:
                rank = self._CAT_RANK.get(category, 0)
                if best is None or rank > best[1]:
                    best = (r["transaction_id"], rank, r["status"])
        if best is None:
            return None
        tid, rank, status = best
        if self._CAT_RANK.get(category, 0) > self._CAT_RANK.get(
                self.conn.execute("SELECT category FROM profile_transactions WHERE transaction_id=?",
                                  (tid,)).fetchone()[0], 0):
            self.conn.execute(
                "UPDATE profile_transactions SET category=?, updated_at=? WHERE transaction_id=?",
                (category, _now(), tid))
        self.conn.commit()
        return tid

    # ---------- 交易 ----------
    def upsert_transaction(self, user_id: str, *, category: str, amount_minor: int,
                           currency: str, item: str = "", merchant: str = "",
                           occurred_at: str = "", status: str = "pending",
                           confidence: float = 0.0, extractor: str = "v1",
                           sources: Iterable[Dict[str, str]] = ()) -> str:
        """写入/合并一笔交易，并把证据挂上。返回 transaction_id。"""
        amount_minor = int(amount_minor)
        currency = (currency or "").upper()
        occurred_at = occurred_at or ""
        tid = tx_id(user_id, category, amount_minor, currency, occurred_at, item)
        with self.lock:
            now = _now()
            _merged = self._merge_similar(user_id, category, amount_minor, currency, item, occurred_at)
            if _merged and _merged != tid:
                tid = _merged
            if status == "refunded":
                self.conn.execute(
                    "UPDATE profile_transactions SET status='refunded', updated_at=? "
                    "WHERE user_id=? AND category=? AND amount_minor=? AND currency=? "
                    "AND occurred_at=? "
                    "AND status IN ('confirmed','pending')",
                    (now, user_id, category, amount_minor, currency, occurred_at))
                self.conn.execute(
                    "INSERT OR REPLACE INTO profile_transactions(transaction_id,user_id,occurred_at,merchant,item,"
                    "category,amount_minor,currency,status,confidence,extractor,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid + "_refund", user_id, occurred_at, merchant, item, category, amount_minor,
                     currency, "refunded", float(confidence), extractor, now, now))
                self.conn.commit()
                return tid + "_refund"
            row = self.conn.execute(
                "SELECT status FROM profile_transactions WHERE transaction_id=?", (tid,)).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO profile_transactions(transaction_id,user_id,occurred_at,merchant,item,"
                    "category,amount_minor,currency,status,confidence,extractor,created_at,updated_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, user_id, occurred_at, merchant, item, category, amount_minor,
                     currency, status, float(confidence), extractor, now, now))
            else:
                # 已有：状态只升不降（pending→confirmed），金额不变（金额不同会被算作新交易）
                if row["status"] == "pending" and status == "confirmed":
                    self.conn.execute(
                        "UPDATE profile_transactions SET status='confirmed', updated_at=? WHERE transaction_id=?",
                        (now, tid))
            for s in sources:
                mid = str(s.get("memory_id") or "")
                if not mid:
                    continue
                self.conn.execute(
                    "INSERT OR IGNORE INTO profile_transaction_sources"
                    "(transaction_id,memory_id,request_id,quote,created_at) VALUES(?,?,?,?,?)",
                    (tid, mid, str(s.get("request_id") or ""), str(s.get("quote") or "")[:400], now))
            self.conn.commit()
        return tid

    def sum_spending(self, user_id: str, category: Optional[str] = None,
                     currency: Optional[str] = None) -> Dict[str, Any]:
        """确定性聚合：按币种分组求和 + 笔数 + 每条来源。"""
        q = ("SELECT currency, SUM(amount_minor) AS total_minor, COUNT(*) AS n "
             "FROM profile_transactions WHERE user_id=? AND status='confirmed'")
        args: List[Any] = [user_id]
        if category:
            q += " AND category=?"
            args.append(category)
        if currency:
            q += " AND currency=?"
            args.append(currency.upper())
        q += " GROUP BY currency"
        rows = self.conn.execute(q, args).fetchall()
        detail_q = ("SELECT transaction_id,item,amount_minor,currency,occurred_at,status "
                    "FROM profile_transactions WHERE user_id=? AND status='confirmed'")
        dargs: List[Any] = [user_id]
        if category:
            detail_q += " AND category=?"; dargs.append(category)
        detail_q += " ORDER BY occurred_at"
        detail = [dict(r) for r in self.conn.execute(detail_q, dargs).fetchall()]
        return {"by_currency": [dict(r) for r in rows], "transactions": detail}

    # ---------- 类别表 ----------
    def upsert_category(self, user_id: str, name: str, description: str = "",
                        parent: str = "") -> None:
        name = (name or "").strip()
        if not name:
            return
        with self.lock:
            now = _now()
            self.conn.execute(
                "INSERT INTO profile_categories(user_id,name,description,parent,n_tx,created_at,updated_at) "
                "VALUES(?,?,?,?,1,?,?) ON CONFLICT(user_id,name) DO UPDATE SET "
                "n_tx=profile_categories.n_tx+1, updated_at=excluded.updated_at, "
                "description=COALESCE(NULLIF(excluded.description,''), profile_categories.description)",
                (user_id, name, description or "", parent or "", now, now))
            self.conn.commit()

    def list_categories(self, user_id: str, limit: int = 80) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT name,description,parent,n_tx FROM profile_categories "
                "WHERE user_id=? ORDER BY n_tx DESC, name LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ---------- 事实 ----------
    def add_fact(self, user_id: str, *, fact_type: str, predicate: str, obj: str,
                 polarity: str = "positive", confidence: float = 0.0,
                 evidence_type: str = "explicit_statement",
                 source_memory_ids: Optional[List[str]] = None,
                 valid_from: str = "") -> str:
        raw = f"{user_id}|{fact_type}|{predicate}|{obj}|{valid_from}"
        fid = "ft_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
        now = _now()
        with self.lock:
            # 版本管理：同一对象出现**相反极性**的新事实 → 旧事实置为 superseded（保留历史）
            opposite = "negative" if (polarity or "positive") == "positive" else "positive"
            self.conn.execute(
                "UPDATE profile_facts SET status='superseded', valid_to=?, updated_at=? "
                "WHERE user_id=? AND object=? AND polarity=? AND status='active'",
                ((valid_from or now)[:10], now, user_id, obj, opposite))
            self.conn.execute(
                "INSERT OR REPLACE INTO profile_facts(fact_id,user_id,fact_type,subject,predicate,object,"
                "polarity,confidence,evidence_type,source_memory_ids,valid_from,valid_to,status,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, user_id, fact_type, "user", predicate, obj, polarity, float(confidence),
                 evidence_type, json.dumps(source_memory_ids or [], ensure_ascii=False),
                 valid_from, None, "active", now, now))
            self.conn.commit()
        return fid

    def query_facts(self, user_id: str, fact_type: Optional[str] = None,
                    keyword: str = "", active_only: bool = True, limit: int = 20) -> List[Dict[str, Any]]:
        q = "SELECT * FROM profile_facts WHERE user_id=?"
        args: List[Any] = [user_id]
        if fact_type:
            q += " AND fact_type=?"; args.append(fact_type)
        if active_only:
            q += " AND status='active'"
        q += " ORDER BY updated_at DESC LIMIT ?"; args.append(int(limit))
        rows = [dict(r) for r in self.conn.execute(q, args).fetchall()]
        if keyword:
            kw = keyword.lower()
            rows = [r for r in rows if kw in (str(r.get("object") or "") + " " + str(r.get("predicate") or "")).lower()]
        return rows

    # ---------- 覆盖状态 ----------
    def coverage(self, user_id: str) -> Dict[str, Any]:
        """coverage_status：complete_enough / partial / unknown。

        规则（确定性）：
          - 有未处理/失败的入库任务 → partial
          - 一条交易都没有 → unknown
          - 其余 → complete_enough（在"已入库记忆"范围内）
        """
        row = self.conn.execute(
            "SELECT SUM(status='pending') AS p, SUM(status='failed') AS f FROM profile_ingest WHERE user_id=?",
            (user_id,)).fetchone()
        pending = int(row["p"] or 0) if row else 0
        failed = int(row["f"] or 0) if row else 0
        n_tx = self.conn.execute(
            "SELECT COUNT(*) FROM profile_transactions WHERE user_id=?", (user_id,)).fetchone()[0]
        if pending or failed:
            state = "partial"
        elif n_tx == 0:
            state = "unknown"
        else:
            state = "complete_enough"
        return {"coverage_status": state, "pending": pending, "failed": failed, "transactions": n_tx}

    def ingest_status(self, user_id: str, memory_ids) -> Dict[str, str]:
        """批量查抽取状态（按需抽取用：已 done 的不再重复抽）。"""
        ids = [str(x) for x in (memory_ids or []) if x]
        if not ids:
            return {}
        out: Dict[str, str] = {}
        with self.lock:
            for i in range(0, len(ids), 200):
                chunk = ids[i:i + 200]
                q = ("SELECT memory_id,status FROM profile_ingest "
                     "WHERE user_id=? AND memory_id IN (%s)" % ",".join("?" * len(chunk)))
                for mid, st in self.conn.execute(q, (user_id, *chunk)).fetchall():
                    out[str(mid)] = str(st)
        return out

    def mark_ingest(self, user_id: str, memory_id: str, status: str, extractor: str = "v1") -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO profile_ingest(user_id,memory_id,status,extractor,updated_at)"
                " VALUES(?,?,?,?,?)", (user_id, memory_id, status, extractor, _now()))
            self.conn.commit()
