"""LLM 网关池：主用 + 备用，失败立即切换、全部失败后再轮转重试。

配置（env）：`AMC_LLM_ENDPOINTS="base|key|model;base|key|model;..."`
不配置时退化为单网关（base/key/model 来自 AMC_LLM_RERANK_*）。
"""
from __future__ import annotations

import logging
import os
from typing import List, Dict

logger = logging.getLogger("amc.llm_endpoints")


def endpoints(default_base: str = "", default_key: str = "",
              default_model: str = "") -> List[Dict[str, str]]:
    raw = (os.getenv("AMC_LLM_ENDPOINTS") or "").strip()
    out: List[Dict[str, str]] = []
    if raw:
        for spec in raw.split(";"):
            parts = [p.strip() for p in spec.split("|")]
            if len(parts) >= 3 and parts[0] and parts[1] and parts[2]:
                out.append({"base": parts[0].rstrip("/"), "key": parts[1], "model": parts[2]})
    if not out and default_base:
        out.append({"base": default_base.rstrip("/"), "key": default_key, "model": default_model})
    return out


def endpoint_tag(ep: Dict[str, str]) -> str:
    base = ep.get("base", "")
    try:
        from urllib.parse import urlparse
        host = urlparse(base).netloc or base
    except Exception:
        host = base
    return f"{host}/{ep.get('model', '')}"


# ---------- 动态优先级：失败→排到队尾，成功→排到队首 ----------
import time as _time

_ORDER: List[str] = []                 # 网关优先级（base_url 顺序）
_DOWN: Dict[str, float] = {}           # 冷却到期时间
_DOWN_SECS = float(os.getenv("AMC_LLM_ENDPOINT_COOLDOWN", "60") or 60)


def _ensure_order(eps: List[Dict[str, str]]) -> None:
    for e in eps:
        if e["base"] not in _ORDER:
            _ORDER.append(e["base"])


def pick(eps: List[Dict[str, str]], cur_base: str = "") -> Dict[str, str]:
    """按动态优先级选一个不在冷却期的网关；全部冷却时按优先级返回第一个可用。"""
    if not eps:
        return {}
    _ensure_order(eps)
    by_base = {e["base"]: e for e in eps}
    now = _time.time()
    for b in _ORDER:
        ep = by_base.get(b)
        if ep is not None and _DOWN.get(b, 0.0) <= now:
            return ep
    for b in _ORDER:
        if b in by_base:
            return by_base[b]
    return eps[0]


def demote(ep: Dict[str, str], secs: float = None) -> None:
    """失败：排到队尾 + 冷却。"""
    try:
        b = ep["base"]
        _DOWN[b] = _time.time() + (secs if secs is not None else _DOWN_SECS)
        if b in _ORDER:
            _ORDER.remove(b)
        _ORDER.append(b)
        logger.warning("[endpoints] %s 降权到队尾并冷却 %.0fs（当前优先：%s）",
                       endpoint_tag(ep), secs or _DOWN_SECS,
                       endpoint_tag(pick([{"base": x, "model": ""} for x in _ORDER])) or "-")
    except Exception:
        pass


def promote(ep: Dict[str, str]) -> None:
    """成功：排到队首 + 解除冷却。"""
    try:
        b = ep["base"]
        _DOWN.pop(b, None)
        if b in _ORDER:
            _ORDER.remove(b)
        _ORDER.insert(0, b)
    except Exception:
        pass


def order() -> List[str]:
    return list(_ORDER)


def health() -> Dict[str, float]:
    now = _time.time()
    return {k: max(0.0, v - now) for k, v in _DOWN.items()}
