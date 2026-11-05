"""时序比较通道：必须同时定位**两个不同轮次**及其日期，才能回答"先后"。

防止把"…after dark"（只有一个事件）误当先后题。
"""
from __future__ import annotations
import json, logging, re, time

logger = logging.getLogger("amc.ordering_probe")

_ROUTE = re.compile(
    r"^\s*(did|does|has|have|was|were|is|are)\b[^?]{0,200}?\b(before|after|first|earlier|later)\b|"
    r"which .*(came|happened) first|先后|哪个先|早于|晚于|更早|更晚|"
    r"先.{0,24}还是|先.{0,24}(再|后)|之前.{0,12}之后|更早|更晚", re.I)

PROMPT = """问题在比较两个事件的**先后**。请从候选轮次里找出这两个事件各自的轮次与**日期**，再判断先后。

问题：{q}

候选（编号 / 日期 / 正文）：
{body}

严格要求（缺一不可）：
1) 事件 A 能定位到某一轮（给编号与日期）
2) 事件 B 能定位到**另一轮**（给编号与日期；不能与 A 是同一轮）
3) 两个日期都能确定时才给结论
若任一事件定位不到、或两个事件落在同一轮 → located=false

只输出 JSON：{{"located":true|false,"a_id":0,"a_date":"YYYY-MM-DD","b_id":1,"b_date":"YYYY-MM-DD","answer":"yes|no","why":"一句话"}}
"""


def is_ordering_question(q: str) -> bool:
    return bool(_ROUTE.search(q or ""))


def probe(query: str, cands: list, base_url: str, api_key: str, model: str) -> dict | None:
    if not cands:
        return None
    body = "\n\n".join("[%d] date=%s | %s" % (i, c.get("date", ""), str(c.get("text", ""))[:700])
                       for i, c in enumerate(cands[:20]))
    from .llm_json import call_json
    obj = call_json(base_url, api_key, model, PROMPT.format(q=query[:400], body=body),
                    timeout=90, retries=3, max_tokens=300)
    if not obj:
        return None
    if not obj.get("located"):
        return None
    ai, bi = obj.get("a_id"), obj.get("b_id")
    if not (isinstance(ai, int) and isinstance(bi, int)) or ai == bi:
        logger.info("[ordering] reject: same/missing rounds a=%s b=%s", ai, bi)
        return None
    ans = str(obj.get("answer", "")).lower()
    if ans not in ("yes", "no"):
        return None
    return {"answer": ans, "a_id": ai, "b_id": bi,
            "a_date": obj.get("a_date"), "b_date": obj.get("b_date")}
