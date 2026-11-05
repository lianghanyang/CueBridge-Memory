"""Bounded OCR/VLM adapter with safe OpenAI-compatible gpt-4o-mini support.

The adapter is read-only: it accepts image bytes and returns model evidence text only.
The HTTP runner is deliberately injected/configured by the caller so secrets never live
in this module, logs, reports, or source files.
"""
from __future__ import annotations
import base64
import hashlib
from typing import Callable, Optional


class OCRTool:
    def __init__(self, runner: Optional[Callable[[bytes, str], str]] = None, max_calls: int = 2):
        self.runner = runner
        self.max_calls = max(1, int(max_calls))
        self.calls = 0
        self.cache = {}

    def ocr(self, record_id: str, question: str, *, image_bytes: bytes = b""):
        if not image_bytes or self.runner is None or self.calls >= self.max_calls:
            return None
        key = hashlib.sha256(image_bytes + b"\0" + question.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        try:
            answer = self.runner(image_bytes, question)
        except Exception:
            return None
        if answer:
            self.cache[key] = str(answer)
            return self.cache[key]
        return None


def gpt4o_mini_runner(base_url: str, api_key: str, *, model: str = "gpt-4o-mini",
                      timeout: float = 8.0) -> Callable[[bytes, str], str]:
    """Build a single-call OpenAI-compatible vision runner.

    The returned callable never logs the key or image.  It has no retry loop so the
    caller's OCR call budget remains authoritative.  ``base_url`` may be either the
    API root or a full ``/chat/completions`` URL.
    """
    import httpx
    from ..llm_endpoints import endpoints, pick, promote, demote

    configured = endpoints(base_url, api_key, model)
    if not configured:
        raise ValueError("base_url is required")

    def run(image_bytes: bytes, question: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": (
                    "Extract only visual evidence relevant to this question. "
                    "Do not calculate totals, infer an answer, or provide a conclusion. "
                    f"Question: {question[:800]}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ]}],
        }
        ep = pick(configured)
        url = ep["base"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {ep['key'] or 'EMPTY'}"}
        body["model"] = ep.get("model") or model
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
            response.raise_for_status()
            promote(ep)
        except Exception:
            demote(ep)
            raise
        content = response.json()["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        return str(content or "")[:4000]

    return run


def ocr(record_id: str, question: str, *, image_bytes: bytes = b"", runner=None,
        max_calls: int = 2):
    return OCRTool(runner, max_calls=max_calls).ocr(
        record_id, question, image_bytes=image_bytes
    )


__all__ = ["OCRTool", "gpt4o_mini_runner", "ocr"]
