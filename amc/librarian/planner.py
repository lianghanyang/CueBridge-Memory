"""Query Planner：把问题解析成"需要什么证据结构"的计划。

只做**结构判断**，不做检索、不做回答。规则优先，不确定时退回 SINGLE。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

EvidenceKind = Literal["single", "pair", "timeline", "multimodal", "set"]

_CONFLICT = re.compile(r"contradict|conflict|inconsisten|disagree|矛盾|冲突|不一致|前后不一", re.I)
_TEMPORAL = re.compile(r"\bbefore\b|\bafter\b|\bearlier\b|\blater\b|first|then|"
                       r"at the end|most recent|which came|之前|之后|先后|最早|最晚|最后", re.I)
_VISUAL = re.compile(r"\bimage\b|\bpicture\b|\bphoto\b|attached|shown|图|照片|截图", re.I)
_SET = re.compile(r"\bhow many\b|\bhow much\b|\btotal\b|\ball\b|\blist\b|哪些|几个|总共|一共", re.I)
_ENTITY = re.compile(r"\b(?:of|by|about|from|did)\s+([A-Z][a-z]{2,})\b")
_DECISION = re.compile(r"choose|chose|chosen|settle|decide|decided|final|pick|选定|决定|最终", re.I)


@dataclass
class EvidencePlan:
    task: str                      # conflict / temporal / visual / set / single
    kind: EvidenceKind             # single / pair / timeline / multimodal / set
    entity: str = ""
    topic: str = ""
    scope: str = "any"             # session | across_sessions | any
    needs_visual: bool = False
    needs_time: bool = False
    needs_decision: bool = False   # "最终选择"类
    n_units: int = 1               # 需要几个证据单元
    stop_when: str = "enough"      # enough / all / conflict_found


def plan(query: str, has_query_image: bool = False) -> EvidencePlan:
    q = query or ""
    visual = bool(_VISUAL.search(q)) or has_query_image
    if _CONFLICT.search(q):
        m = _ENTITY.search(q)
        topic_m = re.search(r"(?:definition|meaning|claim|statement)s?\s+of\s+([a-z ]{3,40})", q, re.I)
        return EvidencePlan(task="conflict", kind="pair",
                            entity=(m.group(1) if m else ""),
                            topic=(topic_m.group(1).strip() if topic_m else ""),
                            scope="across_sessions", needs_visual=visual, n_units=1,
                            stop_when="conflict_found")
    if _TEMPORAL.search(q):
        return EvidencePlan(task="temporal", kind="timeline",
                            scope="across_sessions", needs_time=True,
                            needs_decision=bool(_DECISION.search(q)),
                            n_units=2, stop_when="enough")
    if visual:
        return EvidencePlan(task="visual", kind="multimodal", needs_visual=True, n_units=1)
    if _SET.search(q):
        return EvidencePlan(task="set", kind="set", n_units=3, stop_when="all")
    return EvidencePlan(task="single", kind="single", n_units=1, needs_decision=bool(_DECISION.search(q)))
