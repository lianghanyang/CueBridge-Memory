"""CD 用的陈述对构建器：把"单条相关性"换成"陈述对的相关性 + 主题一致性"。

确定性实现（不调 LLM）：
  pair_score = rel(a) + rel(b) + λ1·overlap(a,b) + λ2·topic_hit + λ3·session_bonus
其中 rel 来自既有重排分（4B），overlap 是两段文本的实词重叠（同一话题），
session_bonus 在"跨会话"计划下加分（CD 要求跨会话矛盾）。
"""
from __future__ import annotations

import itertools
import re
from typing import Dict, List, Sequence, Tuple

_STOP = set("a an the of to in on at for with and or is are was were be this that it its "
            "my me i we our you your he she they them his her as by from about into over "
            "some any do does did have has had what which who whom why how when where "
            "there mention mentioned".split())


def toks(s: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z\-]{2,}", (s or "").lower()) if w not in _STOP}


def build_pairs(query: str, cands: Sequence[Dict], top_m: int = 12,
                rel_key: str = "score", lam_overlap: float = 0.6,
                lam_topic: float = 0.8, cross_session_bonus: float = 0.3) -> List[Dict]:
    """cands: [{rid, text, date, session, score}] → 排序后的陈述对列表"""
    qt = toks(query)
    if not qt or len(cands) < 2:
        return []
    scored = []
    for c in cands:
        ct = toks(c.get("text") or "")
        topic_hit = len(qt & ct) / max(1, len(qt))
        rel = float(c.get(rel_key) or 0.0)
        scored.append((rel + lam_topic * topic_hit, c, ct))
    scored.sort(key=lambda x: -x[0])
    pool = scored[:top_m]
    out = []
    for (sa, a, ta), (sb, b, tb) in itertools.combinations(pool, 2):
        ov = len(ta & tb) / max(1, min(len(ta), len(tb)) or 1)
        sess = 1.0 if (a.get("session") and b.get("session") and a["session"] != b["session"]) else 0.0
        s = sa + sb + lam_overlap * ov + cross_session_bonus * sess
        out.append({"a": a, "b": b, "score": s, "overlap": ov, "cross_session": sess})
    out.sort(key=lambda x: -x["score"])
    return out


def unit_contains(pair: Dict, gold_rids: set) -> bool:
    return (pair["a"].get("rid") in gold_rids) or (pair["b"].get("rid") in gold_rids)
