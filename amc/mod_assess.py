"""M3：LLM 支持度判定（来自 MEG / astra 框架的可选模块）。

目的：**不靠相关性分数**判断"证据够不够"。分数高 ≠ 能回答；
        应拒答与非拒答样本的相似度分布实测严重重叠。

行为：
  - 仅当 `AMC_MOD_ASSESS=1` 时启用（默认关闭 → 完全等于现有行为）
  - 只在"原本判定要作答"时介入：若模型判"证据不足以回答" → 降级为拒答
  - **不反向提升**（不把原本要拒答的改成作答），避免放大误答
  - 快速失败：超时/异常一律放行（保持原判定）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from typing import Any

import httpx

logger = logging.getLogger("amc.mod_assess")

_cache: dict[str, dict] = {}
_lock = threading.Lock()
_db: sqlite3.Connection | None = None

PROMPT = """判断下面的记忆证据**能否回答**这个问题。

问题：{q}

证据：
{evidence}

规则：
- 只有当证据里**明确包含**回答问题所需的信息时，answerable 才为 true；
- 只是话题相关、部分相关、或需要外部知识推断 → false；
- "总量/总计"类问题，需要证据里能看到**每一笔**相关金额才算 true；
- "有没有/是不是"类问题，证据必须能确定是或否（只沾边不算）。

只输出 JSON：{{"answerable": true/false, "reason": "一句话"}}"""


def enabled() -> bool:
    return os.getenv("AMC_MOD_ASSESS", "0") not in ("0", "", "false", "False")


def _cache_path() -> str:
    return os.getenv("AMC_MOD_ASSESS_CACHE", "./data/store/mod_assess_cache.db")


def _conn() -> sqlite3.Connection | None:
    global _db
    if _db is None:
        try:
            _db = sqlite3.connect(_cache_path(), check_same_thread=False)
            _db.execute("CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, v TEXT)")
            _db.commit()
        except Exception:
            logger.exception("mod_assess cache init failed")
            _db = None
    return _db


def _key(q: str, ids: list[str]) -> str:
    return hashlib.sha256((q + "\x00" + ",".join(sorted(ids))).encode("utf-8")).hexdigest()


def _get(k: str) -> dict | None:
    with _lock:
        if k in _cache:
            return _cache[k]
    db = _conn()
    if db is not None:
        try:
            row = db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
            if row:
                val = json.loads(row[0])
                with _lock:
                    _cache[k] = val
                return val
        except Exception:
            pass
    return None


def _put(k: str, v: dict) -> None:
    with _lock:
        _cache[k] = v
        if len(_cache) > 4096:
            _cache.clear()
    db = _conn()
    if db is not None:
        try:
            db.execute("INSERT OR REPLACE INTO cache(k,v) VALUES(?,?)",
                       (k, json.dumps(v, ensure_ascii=False)))
            db.commit()
        except Exception:
            pass


def judge(question: str, records: list[dict[str, Any]], max_records: int = 0) -> dict:
    """返回 {'answerable': bool, 'reason': str}；异常一律 answerable=True（放行）。"""
    from .llm_endpoints import demote, pick, endpoints as _eps

    # ⚠️ 2026-11-02：max_records=0 ⇒ **不按条数截断**（原来默认 6）。
    #    head/tail 800/800 后单条≈1650 字 ⇒ 6 条=9900 > 6000
    #    ⇒ 裁判实际只看约 3.6 条 ⇒ 有证据也判"证据不足" ⇒ 误拒答。
    picked = records[:max_records] if max_records and max_records > 0 else list(records)
    k = _key(question, [str(r.get("id")) for r in picked])
    hit = _get(k)
    if hit is not None:
        return hit

    ev = "\n\n".join(f"[{i}] {str(r.get('text'))}" for i, r in enumerate(picked, 1))
    if not ev.strip():
        return {"answerable": False, "reason": "no-evidence"}

    eps = _eps(os.getenv("AMC_LLM_RERANK_BASE_URL", ""), os.getenv("AMC_LLM_RERANK_API_KEY", ""),
               os.getenv("AMC_LLM_RERANK_MODEL", "gpt-4o-mini"))
    if not eps:
        return {"answerable": True, "reason": "no-endpoint"}
    ep = pick(eps)
    if not ep:
        return {"answerable": True, "reason": "no-endpoint"}
    body = {"model": ep.get("model") or "gpt-4o-mini", "temperature": 0.0, "max_tokens": 120,
            "messages": [{"role": "user", "content": PROMPT.format(q=question[:400], evidence=ev[:40000])}]}
    try:
        r = httpx.post(ep["base"] + "/chat/completions",
                       headers={"Authorization": f"Bearer {ep['key']}"},
                       json=body, timeout=15)          # 快速失败：评测端超时不可接受
        r.raise_for_status()
        raw = (r.json()["choices"][0]["message"]["content"] or "").strip()
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)
        out = json.loads(m.group(0)) if m else {"answerable": True, "reason": raw[:80]}
        out["answerable"] = bool(out.get("answerable"))
        _put(k, out)
        return out
    except Exception as exc:
        demote(ep)
        logger.warning("[mod_assess] 判定失败放行: %s", str(exc)[:100])
        return {"answerable": True, "reason": "failed"}
