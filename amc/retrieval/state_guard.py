"""三条件护栏的状态更新重排（借鉴 FlowGrid AML Retriever v1.1 的 guarded supersession）。

FlowGrid 的设计（docs/EVAL.md 附录 D/E）：
  无保护的时间重排会把「同一话题但在更晚时间点的**无关提及**」误判为覆写，导致
  knowledge_update 类掉分、总 MRR 倒退（L8：目标格 +0.112，总量 −0.031 ~ −0.039）。
  因此 v1.1 只在**三个条件同时成立**时做软重排：
    ① 查询具备时间意图
    ② 两条证据话题高度重合（成对冗余）
    ③ 较新的一条显式表达更新/更正/推迟/作废语义
  只抬新、轻降旧；**旧证据永不删除，仍可检索**。

本模块是这套护栏在本项目的确定性实现（无模型调用）。它只调整**排名**，不改变返回集合，
因此对召回的影响应当是有界的；若实测触发率过低，则本数据集无法验证该机制，
必须如实标注 unknown。
"""
from __future__ import annotations

import math
import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+|\b\d{4}\b")

TEMPORAL_INTENT = (
    "when", "what date", "which date", "how long", "how often", "latest",
    "most recent", "recent", "now", "currently", "current", "still",
    "these days", "by then", "at present", "up to date", "已", "现在", "最近",
)
UPDATE_CUE = (
    "moved to", "moved from", "changed to", "updated to", "revised to",
    "switched to", "rescheduled", "postponed", "delayed", "pushed back",
    "pushed to", "no longer", "instead", "corrected", "is now", "are now",
    "was now", "cancelled", "canceled", "replaced by", "superseded",
)
# 数值/日期状态意图：只在查询确实在问"当前值"时才允许覆写重排
VALUE_INTENT = ("what is", "what's", "which", "how many", "how much", "current",
                "latest", "now", "date", "time", "deadline", "price", "cost")

STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was", "were",
    "be", "been", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "that", "this", "it", "its", "they", "them", "i", "my", "me", "you", "your",
    "what", "when", "where", "which", "who", "why", "how", "do", "does", "did",
}


def _tokens(text):
    return {t.lower() for t in _WORD_RE.findall(text or "")} - STOP


def has_temporal_intent(query):
    q = (query or "").lower()
    return any(m in q for m in TEMPORAL_INTENT)


def has_value_intent(query):
    q = (query or "").lower()
    return any(m in q for m in VALUE_INTENT)


def has_update_cue(text):
    t = (text or "").lower()
    return any(m in t for m in UPDATE_CUE)


def topic_overlap(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _text_of(memory):
    parts = []
    for part in memory.get("content", []) or []:
        if isinstance(part, dict) and part.get("type") == "text":
            v = str(part.get("text", "")).strip()
            if v:
                parts.append(v)
    return "\n".join(parts) or (memory.get("index_text") or "")


def _ts(memory):
    return memory.get("ts") or memory.get("timestamp") or 0


class StateUpdateGuard:
    """三条件护栏的软重排。只在条件全部成立时生效，否则原样返回。"""

    def __init__(self, topk=30, min_overlap=0.22, boost=0.35, demote=0.25,
                 require_value_intent=True):
        self.topk = topk
        self.min_overlap = min_overlap
        self.boost = boost
        self.demote = demote
        self.require_value_intent = require_value_intent
        self.stats = {"calls": 0, "temporal": 0, "pairs": 0, "applied": 0}

    def apply(self, query, hits):
        """hits: [(memory, score)] —— 分数非递增。返回同长度的新列表。"""
        self.stats["calls"] += 1
        if len(hits) < 2:
            return hits
        if not has_temporal_intent(query):
            return hits
        self.stats["temporal"] += 1
        if self.require_value_intent and not has_value_intent(query):
            return hits

        head = hits[: self.topk]
        span = head[0][1] - head[-1][1] if head[0][1] > head[-1][1] else 1.0
        adjust = [0.0] * len(head)
        for i in range(len(head)):
            for j in range(i + 1, len(head)):
                mi, mj = head[i][0], head[j][0]
                newer, older = (mi, mj) if _ts(mi) >= _ts(mj) else (mj, mi)
                if _ts(newer) <= _ts(older):
                    continue
                ov = topic_overlap(_text_of(newer), _text_of(older))
                if ov < self.min_overlap:
                    continue
                if not has_update_cue(_text_of(newer)):
                    continue
                self.stats["pairs"] += 1
                idx_new = i if newer is mi else j
                idx_old = j if newer is mi else i
                adjust[idx_new] += self.boost * span
                adjust[idx_old] -= self.demote * span
        if not any(adjust):
            return hits
        self.stats["applied"] += 1
        merged = [(m, s + a) for (m, s), a in zip(head, adjust)]
        merged.sort(key=lambda x: -x[1])
        out = merged + list(hits[self.topk:])
        # 契约：分数必须非递增
        fixed, floor = [], None
        for m, s in out:
            if floor is not None and s > floor:
                s = floor
            fixed.append((m, s))
            floor = s
        return fixed

    def describe(self):
        return {"class": self.__class__.__name__, "topk": self.topk,
                "min_overlap": self.min_overlap, "boost": self.boost,
                "demote": self.demote, **self.stats}
