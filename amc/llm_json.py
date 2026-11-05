"""共享的「调 LLM 拿 JSON」助手。

问题：6 个探测模块（temporal / ordering / open_end / conflict / aggregate / visual）
各自复制了一份几乎相同的样板：

    import httpx
    for attempt in range(1, retries + 1):
        r = httpx.post(base_url + "/chat/completions", headers=..., json=..., timeout=...)
        r.raise_for_status()
        t = r.json()["choices"][0]["message"]["content"].strip()
        if t.startswith("```"):            # 去 code fence
            t = t.split("```")[1]
            t = t[4:] if t.lower().startswith("json") else t
        return json.loads(t)

每份约 15 行 × 6 处 ≈ 90 行，且**重试策略不一致**：
有的模块对 4xx 直接返回 None，有的会重试到超时——这是真实的一致性风险。

本模块把它们收敛成一处，行为与原实现保持一致：
  * 4xx（除 429）→ 立即返回 None，不重试
  * 其它异常 → 指数退避重试
  * 去掉 ```json 包裹
  * 解析失败按异常处理（重试）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            t = parts[1]
            if t.lower().startswith("json"):
                t = t[4:]
    return t.strip()


def call_json(base_url: str,
              api_key: str,
              model: str,
              prompt: str,
              *,
              timeout: float = 90.0,
              retries: int = 3,
              max_tokens: int = 600,
              temperature: float = 0.0) -> Optional[Dict[str, Any]]:
    """返回解析后的 JSON dict；失败返回 None。"""
    if not (base_url and model):
        return None
    import httpx
    from .llm_endpoints import endpoints, pick, promote, demote
    eps = endpoints(base_url, api_key, model)
    if not eps:
        return None
    for attempt in range(1, max(1, int(retries)) + 1):
        ep = pick(eps)
        try:
            url = ep["base"].rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {ep.get('key') or 'EMPTY'}"}
            body = {"model": ep.get("model") or model, "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            r = httpx.post(url, headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            promote(ep)
            content = r.json()["choices"][0]["message"]["content"] or ""
            return json.loads(_strip_fence(content))
        except Exception as exc:                      # noqa: BLE001
            demote(ep)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                logger.warning("[llm_json] 4xx(%s) 直接放弃", status)
                return None
            if attempt >= retries:
                logger.warning("[llm_json] 重试 %d 次仍失败: %s", attempt, str(exc)[:80])
                return None
            time.sleep(1.2 * attempt)
    return None
