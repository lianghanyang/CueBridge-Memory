"""多模态证据选择：用「查询图 + 候选图片」让 gpt-4o-mini 直接判定哪张候选图承载答案。

用途：查询带图时，在候选层面做**图像级**重排（与文本侧的 evidence_set 档互补）。
默认关闭（AMC_MULTIMODAL_SELECT=1 才启用）。合规：LLM 组件为 gpt-4o-mini；
图片不进入向量层（本模块只做候选重排，不产生 embedding）。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

PROMPT = """The first image is the user's QUERY image. Then come numbered CANDIDATE images.

Which candidate images match the query image (same subject / same scene / same meaning)?

Return STRICT JSON only: {"match": [<candidate numbers, best first>]}
JSON:"""


class MultimodalSelector:
    """把查询图与候选图一起送给 VLM，返回候选重排顺序。"""

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini",
                 max_candidates: int = 4, max_side: int = 512, max_bytes: int = 200_000,
                 timeout: float = 90.0, retries: int = 2):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.max_candidates = max(1, int(max_candidates))
        self.max_side = int(max_side)
        self.max_bytes = int(max_bytes)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "ok": 0, "empty": 0, "failed": 0}

    # ---------- 图片准备 ----------
    def _shrink(self, raw: bytes, fmt: Optional[str]) -> Optional[str]:
        """缩到 max_side / max_bytes 以内的 data URI（避免请求体过大被网关拒绝）。"""
        try:
            from .imaging import encode_data_uri, vlm_ready_image

            small, f2 = vlm_ready_image(raw, fmt)          # 先按统一规则缩一次
            side = self.max_side
            try:
                from PIL import Image
                import io as _io

                while side >= 128:
                    im = Image.open(_io.BytesIO(small)).convert("RGB")
                    w, h = im.size
                    if max(w, h) <= side and len(small) <= self.max_bytes:
                        break
                    sc = side / float(max(w, h))
                    im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
                    out = _io.BytesIO()
                    im.save(out, "JPEG", quality=80)
                    small, f2 = out.getvalue(), "jpeg"
                    side = max(128, side // 2)
            except Exception:
                pass
            return encode_data_uri(small, f2)
        except Exception as exc:
            logger.warning("multimodal-select shrink failed: %s", exc)
            return None

    def select(self, query_image: str, candidates: List[Tuple[str, str]]) -> List[int]:
        """query_image: data URI；candidates: [(key, data URI)]，返回匹配的候选下标（有序）。"""
        cands = [(k, u) for k, u in candidates if u][: self.max_candidates]
        if not query_image or not cands:
            return []
        content = [{"type": "text", "text": PROMPT},
                   {"type": "image_url", "image_url": {"url": query_image}}]
        for j, (_k, u) in enumerate(cands):
            content.append({"type": "text", "text": f"candidate [{j}]"})
            content.append({"type": "image_url", "image_url": {"url": u}})
        body = {"model": self.model, "temperature": 0.0, "max_tokens": 200,
                "messages": [{"role": "user", "content": content}]}
        with self.lock:
            self.stats["calls"] += 1
        for _ in range(max(1, self.retries)):
            try:
                r = httpx.post(self.base_url + "/chat/completions", json=body,
                               headers={"Authorization": f"Bearer {self.api_key}"},
                               timeout=self.timeout)
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"] or ""
                idx = self._parse(raw, len(cands))
                with self.lock:
                    self.stats["ok" if idx else "empty"] += 1
                return idx
            except Exception as exc:
                logger.warning("multimodal-select call failed: %s", exc)
        with self.lock:
            self.stats["failed"] += 1
        return []

    @staticmethod
    def _parse(text: str, n: int) -> List[int]:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            t = t[4:] if t.lower().startswith("json") else t
        try:
            data = json.loads(t)
            raw = data.get("match") or data.get("ranking") or []
        except Exception:
            raw = re.findall(r"\d+", t)
        out, seen = [], set()
        for item in raw:
            try:
                i = int(str(item).strip())
            except (TypeError, ValueError):
                continue
            if 0 <= i < n and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def describe_stats(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.stats)
