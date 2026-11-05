"""CD 第二步：只在"属性已对齐"的 claim 组内判断是否真的冲突。"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger("amc.librarian.claimjudge")

PROMPT = """下面这些主张**都关于同一个主体的同一个属性**，请判断它们是否**实质冲突**。

判定标准（严格执行）：
- 冲突：对**同一属性**给出**互不相容**的取值（如"是 A" vs "不是 A"、"住柏林" vs "住巴黎"）
- 不冲突：一方更具体；程度/时间不同；一个是打算一个是事实；措辞不同但意思一致

问题背景：{q}
主语：{subject}    属性：{property}

主张：
{items}

只输出 JSON：
{{"conflict": true/false, "why": "一句话理由"}}
"""


def judge_group(query: str, group: Dict, base_url: str, api_key: str, model: str,
                timeout: float = 90.0, retries: int = 3) -> Optional[Dict]:
    import httpx
    items = "\n".join(f"- [{m.get('rid')}] {m['value']}" for m in group["members"][:6])
    prompt = (PROMPT.replace("{q}", query or "")
              .replace("{subject}", group.get("subject", ""))
              .replace("{property}", group.get("property", ""))
              .replace("{items}", items))
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                           headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                           json={"model": model, "temperature": 0.0, "max_tokens": 300,
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=timeout)
            r.raise_for_status()
            t = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if t.startswith("```"):
                t = t.split("```")[1]
                t = t[4:] if t.lower().startswith("json") else t
            d = json.loads(t)
            return {"conflict": bool(d.get("conflict")), "why": str(d.get("why") or "")[:200]}
        except Exception as exc:
            st = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(st, int) and 400 <= st < 500:
                return None
            logger.warning("[claimjudge] attempt %d failed: %s", attempt, exc)
            time.sleep(1.2 * attempt)
    return None


def judge_question(query: str, groups: List[Dict], base_url: str, api_key: str, model: str,
                   max_groups: int = 3) -> Dict:
    """任一组判为冲突 → 整体判冲突。返回 {conflict, why, group}。"""
    for g in groups[:max_groups]:
        r = judge_group(query, g, base_url, api_key, model)
        if r and r.get("conflict"):
            return {"conflict": True, "why": r.get("why", ""), "group": g}
    return {"conflict": False, "why": "", "group": None}
