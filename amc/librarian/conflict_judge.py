"""CD 冲突判定：给定问题 + 少量候选陈述，判断是否存在实质矛盾，并指出是哪两条。

与 pair_builder 的分工：
  pair_builder   → 检索阶段的证据对（实测无效，已弃用）
  conflict_judge → 判定阶段：证据已在手，判断两条陈述是否真的冲突
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger("amc.librarian.conflict")

PROMPT = """判断下面这些陈述中，是否存在**实质矛盾**（同一主题、两个说法互不相容）。

注意：
- "措辞不同"不算矛盾；"一方更详细"不算矛盾
- 必须两条陈述对**同一对象/同一问题**给出**不相容**的说法
- 如果只是话题相同但讲的是不同方面，不算冲突

问题：{q}

候选陈述：
{items}

只输出 JSON：
{{"conflict": true/false, "pair": [编号a, 编号b], "why": "一句话理由", "topic": "冲突主题"}}
若没有冲突：{{"conflict": false, "pair": [], "why": "...", "topic": ""}}
"""


def judge(query: str, cands: List[Dict], base_url: str, api_key: str, model: str,
          top_k: int = 6, timeout: float = 90.0, retries: int = 3) -> Optional[Dict]:
    """cands: [{rid, text}] → {conflict, pair:[rid_a, rid_b], why, topic}"""
    import httpx
    items = "\n".join(f"[{j}] {c.get('text','')[:400]}" for j, c in enumerate(cands[:top_k]))
    prompt = PROMPT.replace("{q}", query or "").replace("{items}", items)
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                           headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                           json={"model": model, "temperature": 0.0, "max_tokens": 400,
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=timeout)
            r.raise_for_status()
            t = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if t.startswith("```"):
                t = t.split("```")[1]
                t = t[4:] if t.lower().startswith("json") else t
            d = json.loads(t)
            pair = d.get("pair") or []
            rids = []
            for j in pair[:2]:
                if isinstance(j, int) and 0 <= j < len(cands):
                    rids.append(cands[j].get("rid"))
            return {"conflict": bool(d.get("conflict")), "pair": rids,
                    "why": str(d.get("why") or "")[:200], "topic": str(d.get("topic") or "")[:80]}
        except Exception as exc:
            st = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(st, int) and 400 <= st < 500:
                return None
            logger.warning("[conflict] attempt %d failed: %s", attempt, exc)
            time.sleep(1.2 * attempt)
    return None
