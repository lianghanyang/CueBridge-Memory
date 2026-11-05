"""指代锚点存储（p_ref）—— 2026-11-02

问题：`query_expand` 把"那次/上次/当时"当停用词**删掉**了，
      所以 relational_association(1019) / implicit_preference(122) 这类题
      （"那次图书馆自习"、"她推荐的那家"）没有锚点 ⇒ 只能靠词面 + 时间窗猜。

做法（reviewer 的 p_ref 最小版，只依赖 5 个字段）：
  ingest 时扫**用户消息**里的指代短语，记下 (phrase, date, anchor_rid, snippet)。
  检索时若题面含指代短语 ⇒ 直接用它的 anchor 记录，而不是删掉它。
"""
from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

REF_PAT = re.compile(
    r"(那次|那次那晚|上次|那天|那回|那时候|那阵子|后来|之后|以前|当时|这次|"
    r"\bthat time\b|\blast time\b|\bback then\b|\bthe other day\b|\bpreviously\b)",
    re.I,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_refs (
    user_id          TEXT NOT NULL,
    phrase           TEXT NOT NULL,
    date             TEXT,
    anchor_memory_id TEXT,
    session_id       TEXT,
    snippet          TEXT,
    score            REAL DEFAULT 1.0,
    created_at       TEXT,
    PRIMARY KEY (user_id, anchor_memory_id, phrase)
);
CREATE INDEX IF NOT EXISTS idx_refs_user_phrase ON profile_refs(user_id, phrase);
"""


def ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def extract_refs(text: str, role: str) -> list[tuple[str, str]]:
    """返回 [(phrase, snippet)]；只抽 user 角色的消息。"""
    if str(role) not in ("user", "human", "User"):
        return []
    if not text:
        return []
    out = []
    for m in REF_PAT.finditer(text):
        phrase = m.group(0).strip()
        snippet = text[m.start(): m.start() + 160].strip()
        # ⚠️ 2026-11-02：**统一小写**。否则库里会同时有 'Previously' 和 'previously'
        #    两份，而查询侧用小写 ⇒ 命中率被腰斩（实测 40 vs 9）。
        out.append((str(phrase).strip().lower(), snippet))
    return out


def record(conn: sqlite3.Connection, user_id: str, memory_id: str, role: str,
           text: str, date: str = "", session_id: str = "") -> int:
    """把一条记忆里的指代短语写入 profile_refs。返回写入条数。"""
    rows = extract_refs(text, role)
    if not rows:
        return 0
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for phrase, snippet in rows:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO profile_refs"
                "(user_id,phrase,date,anchor_memory_id,session_id,snippet,score,created_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (str(user_id), phrase, str(date)[:10], str(memory_id),
                 str(session_id), snippet, 1.0, now))
            n += 1
        except Exception:
            pass
    if n:
        conn.commit()
    return n


def lookup(conn: sqlite3.Connection, user_id: str, question: str,
           limit: int = 20) -> list[dict[str, Any]]:
    """题面里的指代短语 ⇒ 对应的锚点记录（按日期新→旧）。"""
    if not question:
        return []
    phrases = [m.group(0).strip() for m in REF_PAT.finditer(question)]
    if not phrases:
        return []
    out, seen = [], set()
    for ph in phrases[:3]:
        try:
            rows = conn.execute(
                "SELECT anchor_memory_id,date,session_id,snippet FROM profile_refs"
                " WHERE user_id=? AND phrase=? COLLATE NOCASE ORDER BY date DESC LIMIT ?",
                (str(user_id), ph, int(limit))).fetchall()
        except Exception:
            continue
        for rid, date, sid, snip in rows:
            if rid in seen:
                continue
            seen.add(rid)
            out.append({"phrase": ph, "anchor_memory_id": rid, "date": date,
                        "session_id": sid, "snippet": snip})
    return out
