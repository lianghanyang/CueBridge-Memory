"""open_end 合成核验器：允许"从多轮综合"，按"论点是否有依据"逐条核验。

与事实核验的区别：不要求某一轮**单独**包含答案，而是要求**每个必要论点**都能在候选里找到依据。
"""
from __future__ import annotations
import json, logging, re, time

logger = logging.getLogger("amc.open_end_probe")

_ROUTE = re.compile(
    r"\bwhy\b|\bhow\b|\bimpact\b|\binfluence\b|\bexplain\b|\bsummar|\bdescribe\b|"
    r"\bwhat kind\b|\bwhat are the\b|\bdiscuss|\bcompare\b|\bopinion\b|\bview\b|"
    r"为什么|怎么|如何|影响|总结|概括|描述|哪一类|哪些方面|看法|观点", re.I)
_NOT = re.compile(r"^\s*(did|does|has|have|is|are|was|were)\b[^?]{0,120}\b(mention|say|state|deny|buy|bought|visit)\b", re.I)

PROMPT = """这是一道**开放题**，答案需要**综合多轮记忆**，不需要某一轮单独写出完整答案。

问题：{q}

候选（编号 / 日期 / 正文）：
{body}

请：
1) 把问题拆成 2~4 个**必须回答的论点**（claims）
2) 对每个论点，在候选里找依据：给编号 + 原文片段；找不到 → supported=false
3) 判断：**核心论点**是否都有依据？只有都能找到依据时才 synthesized=true
4) 若能综合，写出 1~2 句 answer_draft（只基于这些依据，不要编造）

只输出 JSON：
{{"claims":[{{"claim":"...","supported":true|false,"ids":[0],"quote":"..."}}],
 "synthesized":true|false,"answer_draft":"...","missing":"...","why":"一句话"}}
"""


def is_open_end(q: str) -> bool:
    q = q or ""
    if _NOT.search(q):
        return False
    return bool(_ROUTE.search(q))


def probe(query: str, cands: list, base_url: str, api_key: str, model: str,
          timeout: float = 90.0, retries: int = 3) -> dict | None:
    """返回 {"synthesized": bool, "answer_draft": str, "claims": [...]} 或 None。"""
    if not cands or not (base_url and model):
        return None
    body = "\n\n".join("[%d] (%s) %s" % (i, c.get("date", ""), str(c.get("text", ""))[:900])
                       for i, c in enumerate(cands))
    prompt = PROMPT.format(q=(query or "")[:400], body=body)
    from .llm_json import call_json
    obj = call_json(base_url, api_key, model, prompt,
                    timeout=timeout, retries=retries, max_tokens=700)
    if not obj:
        return None
    claims = obj.get("claims") or []
    n_sup = sum(1 for c in claims if c.get("supported"))
    all_ok = bool(claims) and len(claims) >= 1 and n_sup == len(claims)
    # 分层决策第 3 条：开放题允许"部分回答"——只要有 ≥1 个论点有依据
    partial_ok = n_sup >= 1
    return {"synthesized": bool(obj.get("synthesized")) and all_ok,
            "partial_ok": partial_ok,
            "n_supported": n_sup,
            "answer_draft": str(obj.get("answer_draft") or "")[:400],
            "claims": claims,
            "all_supported": all_ok,
            "missing": str(obj.get("missing") or "")[:160]}
