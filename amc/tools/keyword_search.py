"""Read-only 全用户关键词/近义词检索（A+B 版）。

为什么这样写（2026-10-31 的轨迹分析结论）
-----------------------------------------
原评测脚本里的 `keyword_search` 有两个结构性缺陷，实测在 120 道 Gem 题上
`tool_any` 只有 **61.7%**：

```
① 只用语料 planner 给的 terms，而 planner 常给**抽象短语**
   （如 "beethoven's symphonic structure"、"academic decision"）
   记忆原文里没有这个字符串 ⇒ 子串匹配必失败。
   实测：未命中的 24 条里，**11 条只要改用问题自己的实词就能命中**。
② 命中后 `ids[:limit]` 按**时间顺序**截断（records 是按 ts 排的）
   ⇒ 最相关的记录如果在时间上靠后，会被截掉。
```

两个改动（A+B），实测同一批 120 题：
```
基线（只 planner terms + 时间序截断）   61.7%
A  并入问题的实词                       80.8%   (+19.1pp)
A+B 再按"命中词数"降序排序               88.3%   (+26.6pp)
```

设计约束
--------
· **纯本地、只读**：不调 LLM，不写库，不做统计/求和不给答案（只交证据）
· **可解释**：返回每条命中多少词，便于上层排序与调试
· **不改变"是否有证据"的判定**：它只负责"把候选扩大"，最终取舍仍由上层决定
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

# 问题里高频但无区分度的词（做 A 的并集时必须剔除，否则会把全库都召回来）
_STOP = {
    "what", "which", "when", "where", "whose", "whom", "why", "how",
    "would", "could", "should", "there", "their", "about", "after", "before",
    "please", "return", "conversation", "according", "mention", "mentions",
    "follow", "following", "information", "based", "first", "last", "next",
    "image", "images", "shown", "show", "picture", "pictures", "photo",
    "does", "did", "have", "has", "had", "was", "were", "been", "being",
    "from", "with", "that", "this", "these", "those", "your", "you", "the",
    "and", "but", "for", "not", "are", "its", "his", "her", "our", "they",
    "can", "will", "one", "two", "more", "most", "also", "than", "then",
    "list", "name", "names", "kind", "type", "types", "give", "tell",
}

_WORD = re.compile(r"[a-z][a-z0-9'’-]{2,}")


def _tokens(text: str) -> set[str]:
    return {w.strip("'’-") for w in _WORD.findall((text or "").lower())}


def _query_terms(query: str, limit: int = 24) -> list[str]:
    """从问题里抽**实词**（A 的那一半）。"""
    toks = [t for t in _WORD.findall((query or "").lower()) if t not in _STOP and len(t) >= 3]
    # 去重但保留首次出现顺序
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:limit]


def _flatten_terms(raw: Any, cap: int) -> list[str]:
    out = []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, (list, tuple)):
        for x in raw[:cap]:
            if isinstance(x, str) and x.strip():
                out.append(x.strip().lower()[:80])
    return out


def keyword_search(user_id: str, terms: Any = None, *, synonyms: Any = None,
                   query: str = "", store: Any = None,
                   records: Optional[Iterable[dict]] = None,
                   limit: int = 100) -> list[dict]:
    """在**该用户的全部记忆**里做关键词/近义词检索。

    参数
    ----
    terms      : planner 给的词条（可为多词短语）
    synonyms   : planner 给的同义词组，形如 [["suggested","advised"], ...]
    query      : 原始问题。**这是 A 的关键** —— 会把问题里的实词并入检索词，
                 解决"planner 给抽象短语、原文没有该字符串"导致的漏检。
    limit      : 最终返回条数上限（按匹配度截断，**不是按时间**）

    返回
    ----
    [{"request_id","memory_id","matched_terms","score"}, ...]，按 score 降序。
    """
    if records is None:
        if store is None:
            raise ValueError("store or records is required")
        records = store.list_memories(user_id)
    # 兼容 dict / sqlite3.Row（生产里两种都会出现）
    def _g(r, key, default=None):
        # 支持 dict 与 sqlite3.Row；**不支持 tuple**（调用方忘设 row_factory 时
        # 以前会静默返回 0 条命中，极难排查 —— 这里改成显式报错）
        if hasattr(r, "keys"):
            return r[key] if key in r.keys() else default
        if hasattr(r, "get"):
            return r.get(key, default)
        raise TypeError(
            "keyword_search 需要 dict 或 sqlite3.Row 行；"
            "若用 sqlite3 请先设置 conn.row_factory = sqlite3.Row")
    rows = [r for r in records if str(_g(r, "user_id", user_id)) == str(user_id)]

    planner_terms = _flatten_terms(terms, 12)
    # A：并入问题实词（去重，planner 词优先，因为它们通常更具体）
    merged: list[str] = []
    seen = set()
    for t in planner_terms + _query_terms(query):
        t = t.strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        merged.append(t)

    # 同义词组：组内任一命中即算命中（保持 planner 的语义意图）
    groups: list[list[str]] = []
    if isinstance(synonyms, (list, tuple)):
        for g in synonyms[:8]:
            if isinstance(g, (list, tuple)):
                gl = [str(x).strip().lower()[:80] for x in g[:6] if isinstance(x, str) and x.strip()]
                if gl:
                    groups.append(gl)

    if not merged and not groups:
        return []

    scored: list[tuple[float, dict]] = []
    for r in rows:
        text = str(_g(r, "index_text") or _g(r, "text") or "").lower()
        if not text:
            continue
        hits = [t for t in merged if t and t in text]
        ghit = 0
        for g in groups:
            if any(t in text for t in g):
                ghit += 1
        if not hits and not ghit:
            continue
        # B：按命中词数打分（长词权重略高，抵消 "the" 类短词噪声；已剔停用词）
        score = sum(1.0 + min(len(t), 12) / 12.0 for t in hits) + 0.5 * ghit
        scored.append((score, r, hits))

    # B：**先排序再截断**（原版按时间序截断，会把相关但靠后的记录丢掉）
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, r, hits in scored[: max(1, min(200, int(limit or 100)))]:
        out.append({
            "request_id": _g(r, "request_id"),
            "memory_id": _g(r, "id"),
            "matched_terms": hits[:8],
            "score": round(float(score), 3),
        })
    return out
