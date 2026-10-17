import logging
import math
import re
from typing import Any, Dict, List

from .config import MAX_SEARCH_IMAGES
from .contract import ContractError, validate_add_request, validate_search_request

logger = logging.getLogger("amc.service")
WORD_RE = re.compile(r"[A-Za-z0-9_:$.-]+")


def query_to_text(query: Any, options: Any = None, describer=None) -> str:
    """把 query（str | ContentPart[]）转为检索用文本。

    若 query 含图片且提供了 ``describer``，则用 VLM 生成描述并**并入**文本；
    describer 缺失/失败时保持原文本（绝不因此让 /search 失败）。
    """
    chunks = []
    if isinstance(query, str):
        chunks.append(query)
    elif isinstance(query, list):
        for part in query:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
    text = "\n".join(c.strip() for c in chunks if c.strip())
    if options and isinstance(options, list):
        text = "\n".join([text] + [str(o) for o in options]).strip()
    if describer is not None:
        try:
            text = describer.augment(text, query)
        except Exception:
            logger.exception("query-image augmentation failed; falling back to text only")
    return text


def sanitize_scores(hits):
    clean = []
    prev = None
    for mem, raw_s in hits:
        try:
            val = float(raw_s)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = prev if prev is not None else 0.0
        if prev is not None and val > prev:
            val = prev
        prev = val
        clean.append((mem, val))
    return clean


class MemoryService:
    def __init__(self, store, retriever, max_return=50, settings=None, describer=None,
                 vl_reranker=None, relation_selector=None):
        self.store = store
        self.retriever = retriever
        self.max_return = max_return
        self.settings = settings
        # 记忆侧查询图理解（可选）：query 含图片时用 VLM 描述并入查询文本
        self.describer = describer
        # 记忆侧专用 VL reranker（可选）：仅在查询含图时对头部候选重排
        self.vl_reranker = vl_reranker
        # 记忆侧关系感知证据选择（可选）：结构化+通用关系规则(+逐对图像证据关系)；
        # 生效时**替代** vl_rerank，避免重复重排。
        self.relation_selector = relation_selector

    def add(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """AML Official Add Endpoint Implementation"""
        errs = validate_add_request(req)
        if errs:
            raise ContractError(422, "; ".join(errs[:5]))

        status = self.store.add_request(req)
        if status == "inserted":
            hook = getattr(self.retriever, "notify_add", None)
            if hook is not None:
                try:
                    hook(req["user_id"], req["request_id"])
                except Exception:
                    logger.exception("notify_add failed")

        # 严格遵守 AML 返回规范: 必须原样返回三个 ID，且 success=true
        return {
            "success": True,
            "request_id": req["request_id"],
            "user_id": req["user_id"],
            "session_id": req["session_id"],
        }

    def search(self, req: Dict[str, Any], hydrate: bool = True) -> Dict[str, Any]:
        """AML Official Search Endpoint Implementation"""
        errs = validate_search_request(req)
        if errs:
            raise ContractError(422, "; ".join(errs[:5]))

        # 官方正式评测 top_k 为 100，返回数量绝不可超过请求的 top_k
        requested_k = req["top_k"]
        effective_limit = min(requested_k, self.max_return)

        q_text = query_to_text(req["query"], req.get("options"), self.describer)
        
        # 检索端以大候选池召回，结合 Reranker 排序
        pool_limit = max(effective_limit, 50)
        hits = self.retriever.search(req["user_id"], q_text, pool_limit)

        # 专用 VL reranker（可选，记忆侧）：需要**原始查询部件**（含查询图），
        # 因此放在服务层而非检索器内。仅当查询含图且开关打开时生效（见 vl_rerank.py 的硬闸）。
        rs_applied = False
        if self.relation_selector is not None:
            try:
                hits, rs_applied = self.relation_selector.select(
                    self.store, q_text, req["query"], hits)
            except Exception:
                logger.exception("relation-select failed; keeping retrieval order")

        if self.vl_reranker is not None and not rs_applied:
            try:
                hits = self.vl_reranker.rerank(self.store, q_text, req["query"], hits)
            except Exception:
                logger.exception("vl-rerank failed; keeping retrieval order")

        # 内存侧内部证据去重 (经消融验证为稳定正向收益)
        if self.settings and getattr(self.settings, "search_dedup", False):
            dedup_hits = []
            seen_tokens = set()
            for memory, score in hits:
                txt = (memory.get("index_text") or "").strip().lower()
                toks = tuple(sorted(set(WORD_RE.findall(txt))))[:20]
                if toks and toks in seen_tokens:
                    continue
                if toks:
                    seen_tokens.add(toks)
                dedup_hits.append((memory, score))
            hits = dedup_hits

        hits = sanitize_scores(hits)[:effective_limit]

        # 渲染 Content: 遵照 AML 规范水合图片与文本
        output_records = []
        img_budget = MAX_SEARCH_IMAGES

        for memory, score in hits:
            # 根据配置决定是否水合原图 Base64 (比赛规范支持返回多模态 ContentPart[])
            rendered_content, bytes_used = self.store.render_content(
                memory["content"],
                hydrate=hydrate,
                remaining_budget=img_budget,
            )
            img_budget -= bytes_used

            # content 形态：默认压平成字符串（对 harness 最稳；图片以占位文本表示）
            fmt = getattr(self.settings, "search_content_format", "text")
            if fmt != "parts" and isinstance(rendered_content, list):
                chunks = []
                for part in rendered_content:
                    if not isinstance(part, dict):
                        chunks.append(str(part)); continue
                    if part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
                    elif part.get("type") in ("image_url", "image"):
                        chunks.append("[image]")
                rendered_content = "\n".join(c for c in chunks if c)

            output_records.append({
                "id": memory["id"],
                "content": rendered_content,
                "score": round(score, 6),
                "created_at": memory["created_at"],
            })

        # 严格遵循 AML 返回规范: 顶级对象且仅有 data 数组
        return {"data": output_records}
