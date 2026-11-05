"""CD 第一步：把候选陈述抽成结构化 claim，并按 (主语, 属性) 对齐。

为什么要这一步：
  直接把 6 条候选丢给 LLM 判"有没有矛盾" → 49.4%（接近抛硬币），
  因为模型在"不同话题/不同方面/程度差异"里乱找。先抽 claim + 对齐属性，
  把判定范围从"任意两条"缩小到"同一属性上的两个取值"。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger("amc.librarian.claim")

PROMPT = """从下面的候选陈述中抽取**关于「{entity}」的事实主张（claims）**。

要求：
- 只抽**陈述句**，且是**关于 {entity}** 的；提问、设想、闲聊不要
- subject: 主张的主体（通常是人物/事物名）
- property: 被描述的**属性**，用简短英文单词，如 definition / price / location / job / pet_breed / purchase / opinion
- value: 该属性上的**具体取值或说法**（照抄原文关键短语）
- polarity: positive / negative（用于"是/不是"类主张）
- 每条必须标注来自哪个候选（i）

输出 JSON 数组：
[{{"i":0,"subject":"Julian","property":"definition","value":"AI is a system that...","polarity":"positive"}}]

候选：
{items}

只输出 JSON 数组，不要解释。
"""


def _chat(prompt: str, base_url: str, api_key: str, model: str,
          timeout: float = 90.0, retries: int = 3) -> Optional[List[Dict]]:
    import httpx
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                           headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                           json={"model": model, "temperature": 0.0, "max_tokens": 900,
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=timeout)
            r.raise_for_status()
            t = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if t.startswith("```"):
                t = t.split("```")[1]
                t = t[4:] if t.lower().startswith("json") else t
            d = json.loads(t)
            return d if isinstance(d, list) else d.get("claims") or []
        except Exception as exc:
            st = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(st, int) and 400 <= st < 500:
                return None
            logger.warning("[claim] attempt %d failed: %s", attempt, exc)
            time.sleep(1.2 * attempt)
    return None


def extract_claims(query: str, cands: List[Dict], base_url: str, api_key: str, model: str,
                   entity: str = "", top_k: int = 8) -> List[Dict]:
    """cands: [{rid, text}] → [{rid, subject, property, value, polarity}]"""
    items = "\n".join(f"[{j}] {c.get('text','')[:420]}" for j, c in enumerate(cands[:top_k]))
    ent = entity or "用户/相关人物"
    out = _chat(PROMPT.replace("{entity}", ent).replace("{items}", items),
                base_url, api_key, model) or []
    claims = []
    for c in out:
        try:
            i = int(c.get("i"))
        except Exception:
            continue
        if not (0 <= i < len(cands)):
            continue
        prop = str(c.get("property") or "").strip().lower()
        val = str(c.get("value") or "").strip()
        if not prop or not val:
            continue
        claims.append({"rid": cands[i].get("rid"), "i": i,
                       "subject": str(c.get("subject") or ent).strip(),
                       "property": re.sub(r"[^a-z_]", "", prop)[:24],
                       "value": val[:200],
                       "polarity": str(c.get("polarity") or "positive").lower()})
    return claims


def align(claims: List[Dict]) -> Dict[tuple, List[Dict]]:
    """按 (主语小写, 属性) 分组。"""
    groups: Dict[tuple, List[Dict]] = {}
    for c in claims:
        key = ((c.get("subject") or "").strip().lower(), c.get("property") or "")
        groups.setdefault(key, []).append(c)
    return groups


def conflicting_groups(groups: Dict[tuple, List[Dict]], min_members: int = 2) -> List[Dict]:
    """筛出"同一属性上有 ≥2 条不同取值"的组（真正值得判定的候选）。"""
    out = []
    for key, members in groups.items():
        if len(members) < min_members:
            continue
        vals = {m["value"].strip().lower()[:80] for m in members}
        if len(vals) >= 2:
            out.append({"subject": key[0], "property": key[1], "members": members})
    return out
