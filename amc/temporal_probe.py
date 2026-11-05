"""时间核验通道：日期型问题（which day / when / what date）允许用轮次的 date 字段作答。

只在证据核验判定"不足"时触发；不改变检索结果，只前置一条"时间证据"记录。
"""
from __future__ import annotations
import json, logging, re, time

logger = logging.getLogger("amc.temporal_probe")

_DATE_Q = re.compile(
    r"which day|what date|on which date|when did|when was|when does|"
    r"哪天|哪一天|什么时候|什么日期|什么时间|几月几号|几号|几月|哪年|几年", re.I)
_DATE_OUT = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

PROMPT = """问题在问某个事件的**日期/时间**。请从候选轮次里找出**确实对应问题所述事件**的那一轮，
答案就是该轮的 **date 字段**（不需要正文里写出日期）。

问题：{q}

候选（编号 / 日期 / 正文）：
{body}

要求：
- 必须先确认这一轮讲的**就是问题所说的那件事**（同一事件、同一对象），不是相近事件
- 找不到对应事件 → found=false
- 只输出日期（YYYY-MM-DD）

只输出 JSON：{{"found":true|false,"event_id":0,"date":"YYYY-MM-DD","why":"一句话"}}
"""


def is_date_question(q: str) -> bool:
    return bool(_DATE_Q.search(q or ""))


def probe(query: str, cands: list, base_url: str, api_key: str, model: str,
          timeout: float = 90.0, retries: int = 3) -> dict | None:
    if not cands or not (base_url and model):
        return None
    body = "\n\n".join(
        "[%d] date=%s | %s" % (i, c.get("date", ""), str(c.get("text", ""))[:700])
        for i, c in enumerate(cands[:20]))
    prompt = PROMPT.format(q=query[:400], body=body)
    from .llm_json import call_json
    obj = call_json(base_url, api_key, model, prompt,
                    timeout=timeout, retries=retries, max_tokens=250)
    if not obj:
        return None
    if not obj.get("found"):
        return None
    date = str(obj.get("date") or "").strip()
    idx = obj.get("event_id")
    if not _DATE_OUT.search(date):
        return None
    rid = None
    if isinstance(idx, int) and 0 <= idx < len(cands):
        rid = cands[idx].get("request_id") or cands[idx].get("round_id")
    return {"date": _DATE_OUT.search(date).group(0), "request_id": rid,
            "why": str(obj.get("why") or "")[:160]}
