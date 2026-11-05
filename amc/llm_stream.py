"""OpenAI 兼容接口的 **SSE 流式** 调用工具。

# provider-b 的**非流式**限 10 次/分钟（流式不受此限）
   而**流式 16 并发全部 200** ⇒ 要让 rerank/judge/ab 也能用 provider-b 分担压力，
   必须走 stream=true 并把 SSE chunk 拼回完整文本。

本模块只做"把流拼成文本"，语义与 `choices[0].message.content` 等价；
失败时可按需回退到非流式（默认开启回退）。
"""
from __future__ import annotations
import json
import logging
import os

logger = logging.getLogger("amc.llm_stream")

#   上线后 10 分钟即可判断"流式到底稳不稳"，比跑几次对照更有说服力。
import threading as _thr
_CNT = {"ok": 0, "truncated": 0, "fallback": 0, "error": 0}
_CNT_LOCK = _thr.Lock()


def _bump(k: str) -> None:
    try:
        with _CNT_LOCK:
            _CNT[k] = _CNT.get(k, 0) + 1
            _n = sum(_CNT.values())
            if _n % 20 == 0:
                logger.info("[stream] ok=%d truncated=%d fallback=%d error=%d",
                            _CNT["ok"], _CNT["truncated"], _CNT["fallback"], _CNT["error"])
    except Exception:
        pass


def stats() -> dict:
    try:
        with _CNT_LOCK:
            return dict(_CNT)
    except Exception:
        return {}


def chat_text(base: str, key: str, model: str, messages, max_tokens: int = 400,
              temperature: float = 0.0, timeout: float = 60.0,
              fallback: bool = True) -> str:
    """返回完整文本（与一次性响应的 message.content 等价）。"""
    base = (base or "").rstrip("/")
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "messages": messages, "stream": True}
    try:
        import httpx
        buf = []
        got = False
        _done = False
        with httpx.stream("POST", base + "/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                body = r.read()[:200]
                raise RuntimeError(f"http {r.status_code}: {body!r}")
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(":"):          # SSE 注释/心跳
                    continue
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    _done = True
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                try:
                    delta = obj["choices"][0].get("delta") or {}
                    piece = delta.get("content")
                except Exception:
                    piece = None
                if piece:
                    buf.append(piece)
                    got = True
        if got:
            if not _done:
                # 没有收到 [DONE] ⇒ 可能被网关提前断流（内容可能不完整）
                _bump("truncated")
            else:
                _bump("ok")
            return "".join(buf)
        raise RuntimeError("stream empty")
    except Exception as exc:
        _bump("fallback")
        if not fallback:
            raise
        logger.warning("[llm_stream] %s 流式失败(%s) ⇒ 回退非流式", base, str(exc)[:60])
        import httpx
        payload.pop("stream", None)
        r = httpx.post(base + "/chat/completions",
                       headers={"Authorization": f"Bearer {key}"},
                       json=payload, timeout=timeout)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"] or "").strip()


def enabled() -> bool:
    return (os.getenv("AMC_LLM_STREAM", "0") or "0") in ("1", "true", "True")
