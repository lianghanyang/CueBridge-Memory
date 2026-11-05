"""HyDE：让 LLM 写一段"记忆风格"的假设文本，用它去检索。

为什么是它（而不是"截断式变体"）
--------------------------------
平台 58 题的实测（2026-11-01）：

```
截断式变体（去句末提问）：+1 题，**p=1.0000**   ← 无效
HyDE                  ：+4 题，**p=0.1250**，零回归
"HyDE + 变体" == "HyDE 单独"（都是 37/58）      ← 变体在 HyDE 之上没有增量
```

根因是用词鸿沟：p045 原问题检索 gold 命中 **0/3**，用 gold 原话检索 **2/3**。
截断变体只是把原问题剪短，**不引入新词**；HyDE 生成的是记忆风格文本，**真的引入新词**。

实现约束
--------
· 只用于**扩大 pool**，最终排序仍由原问题的精排决定
· 失败/超时一律**静默降级**（返回 ""，调用方用原结果）
· 输出长度限制在 300 字，避免把检索拉偏
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("amc.hyde")

_PROMPT = (
    "你在帮一个人检索他自己的历史记忆。请**假装你就是他**，写一段 2-3 句的回忆，"
    "描述这段记忆里可能出现的**具体细节**（时间、地点、人名、物件、金额、动作、感受）。"
    '不要回答问题，只写"记忆里会怎么记这件事"。只输出 JSON：{"memory":"..."}\n\n'
    "他的问题：{q}"
)


def hypothetical(query: str, *, base_url: str = "", api_key: str = "",
                 model: str = "gpt-4o-mini", timeout: float = 8.0,
                 max_chars: int = 300) -> str:
    """生成假设记忆文本；失败返回 ""（调用方据此降级）。"""
    q = (query or "").strip()
    if len(q) < 4:
        return ""
    try:
        from .llm_json import call_json
        obj = call_json(base_url or "http://configured-endpoint-pool.invalid/v1",
                        api_key, model, _PROMPT.format(q=q[:400]),
                        timeout=timeout, retries=1, max_tokens=260)
        if not isinstance(obj, dict):
            return ""
        text = str(obj.get("memory") or "").strip()
        return text[:max_chars]
    except Exception:
        logger.debug("hyde 生成失败（降级为不扩展）", exc_info=True)
        return ""
