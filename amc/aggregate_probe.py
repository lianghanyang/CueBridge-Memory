"""集合级核验器：聚合题（总和/数量/次数）。

**不是**"找到就答"，而是要求：
  1) 逐项列出候选里与问题相关的**所有**条目（含数值）
  2) 判断集合是否**完整**（有没有明显缺失的组成部分）
  3) 只有完整时才给出聚合值
"""
from __future__ import annotations
import json, logging, re, time

logger = logging.getLogger("amc.aggregate_probe")

PROMPT = """问题要求一个**聚合结果**（总和 / 数量 / 次数）。请从候选记忆里把相关条目**逐项列出**，
再给出聚合值。

问题：{q}

候选（编号 / 日期 / 正文）：
{body}

要求：
1) items：列出所有与问题相关的条目（每条给 id、该项的数值/名称、以及它是否计入聚合）
2) complete：候选是否**覆盖了全部**相关条目？
   - 如果问题问"一共/总共/所有"，只有当你确信没有遗漏时才 true
   - 只要有明显可能遗漏（例如同一类消费只找到一笔但问题暗示有多笔）→ false
3) aggregate：仅当 complete=true 时给出聚合值（带单位）
4) 若候选里根本没有相关条目 → complete=false, aggregate=null

只输出 JSON：{{"items":[{{"id":0,"what":"...","value":"..."}}],"complete":true|false,
"aggregate":"...|null","why":"一句话"}}
"""


def build_prompt(q: str, cands: list, chars: int = 900) -> str:
    body = "\n\n".join("[%d] (%s) %s" % (i, c.get("date", ""), str(c.get("text", ""))[:chars])
                       for i, c in enumerate(cands))
    return PROMPT.format(q=(q or "")[:400], body=body)


def probe(query: str, cands: list, base_url: str, api_key: str, model: str,
          timeout: float = 90.0, retries: int = 3) -> dict | None:
    """返回 {"complete": bool, "aggregate": str|None, "items": [...], "why": str} 或 None。"""
    if not cands or not (base_url and model):
        return None
    prompt = build_prompt(query, cands)
    from .llm_json import call_json
    obj = call_json(base_url, api_key, model, prompt,
                    timeout=timeout, retries=retries, max_tokens=600)
    if not obj:
        return None
    return {"complete": bool(obj.get("complete")),
            "aggregate": obj.get("aggregate"),
            "items": obj.get("items") or [],
            "why": str(obj.get("why") or "")[:160]}
