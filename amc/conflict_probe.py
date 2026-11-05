"""冲突核验通道：抽陈述 S → 逐轮匹配 → 独立核验。

用于"某陈述是否与记忆冲突 / 是否存在矛盾说法"这类问题：
任务目标是**定位承载冲突的那一轮（或那一对）**，不是仲裁谁对。
"""
from __future__ import annotations
import json, logging, re, time

logger = logging.getLogger("amc.conflict_probe")

_ROUTE = re.compile(r"conflict|contradict|inconsistent|does not match|"
                    r"矛盾|冲突|不一致|是否与.*不符", re.I)

EXTRACT = """从问题里抽出它要对照的**那条陈述**。

问题：{q}

只输出 JSON：{{"subject":"...","predicate":"...","value":"...","polarity":"+|-","is_claim":true|false}}
- is_claim=false 表示问题里没有待核实的陈述
"""

FIND = """问题里有一条陈述 S。请在候选记忆中找出**谈论同一主体、同一谓词，但取值不同（或极性相反）**的轮次。

S：subject={s} predicate={p} value={v} polarity={pol}

候选（编号 / 日期 / 正文）：
{body}

只输出 JSON：{{"rounds":[{{"id":0,"verdict":"SAME_VALUE|DIFFERENT_VALUE|DENIES|UNRELATED|UNCLEAR","quote":"原文片段"}}],"conflict_ids":[0],"why":"一句话"}}
"""

VERIFY = """你是独立核验员。下面是从候选记忆里摘出的"冲突证据"，请逐条核实。

S：subject={s} predicate={p} value={v} polarity={pol}

引用：
{quotes}

对每条只检查两件事：
1) 摘录是否真实出现在对应轮次
2) 摘录是否在**同一主体 + 同一谓词**上给出**不同取值**或**相反极性**
（**不要**要求摘录里出现 S 的取值原文；"用 Excel 做分析" vs "用 R 做分析" 属于冲突）

只输出 JSON：{{"verified_ids":[0],"rejected":[{{"id":0,"why":"..."}}],"any_conflict":true|false}}
"""


def is_conflict_question(q: str) -> bool:
    return bool(_ROUTE.search(q or ""))


def _chat(base_url, api_key, model, prompt, timeout=90.0, retries=3):
    """薄封装：共享 amc.llm_json.call_json（去重 + 统一重试策略）。"""
    from .llm_json import call_json
    return call_json(base_url, api_key, model, prompt,
                     timeout=timeout, retries=retries)


def probe(query: str, cands: list, base_url: str, api_key: str, model: str) -> dict | None:
    """返回 {"ids":[候选下标...], "why": str} 或 None。"""
    if not cands:
        return None
    s = _chat(base_url, api_key, model, EXTRACT.format(q=query[:400]))
    logger.info("[conflict] stage1 S=%s", bool(s))
    if not s or not s.get("is_claim", True):
        return None
    body = "\n\n".join("[%d] (%s) %s" % (i, c.get("date", ""), str(c.get("text", ""))[:900])
                       for i, c in enumerate(cands[:20]))
    f = _chat(base_url, api_key, model,
              FIND.format(s=s.get("subject", ""), p=s.get("predicate", ""),
                          v=s.get("value", ""), pol=s.get("polarity", "+"), body=body))
    logger.info("[conflict] stage2 ids=%s", (f or {}).get("conflict_ids"))
    if not f:
        return None
    ids = [x for x in (f.get("conflict_ids") or []) if isinstance(x, int) and 0 <= x < len(cands)]
    if not ids:
        return None
    quotes = "\n\n".join("[%d] %s" % (i, str(cands[i].get("text", ""))[:400]) for i in ids)
    v = _chat(base_url, api_key, model,
              VERIFY.format(s=s.get("subject", ""), p=s.get("predicate", ""),
                            v=s.get("value", ""), pol=s.get("polarity", "+"), quotes=quotes))
    logger.info("[conflict] stage3 any=%s verified=%s", (v or {}).get("any_conflict"), (v or {}).get("verified_ids"))
    ok = [x for x in ((v or {}).get("verified_ids") or []) if isinstance(x, int) and 0 <= x < len(cands)]
    # 以 verified_ids 为准：模型有时给出 verified_ids 却把 any_conflict 写成 false
    if not ok:
        return None
    return {"ids": ok, "why": str(s.get("subject") or "")[:60]}
