import logging

from .anchor import AnchorTimeRetriever
from .bm25 import BM25Retriever
from .tfidf import TfidfRetriever

logger = logging.getLogger(__name__)


def _sparse(name, store, min_score, use_relative_time):
    if name == "tfidf":
        return TfidfRetriever(store, min_score=min_score)
    if name == "bm25":
        return BM25Retriever(store, min_score=min_score)
    base = BM25Retriever(store, min_score=min_score)
    return AnchorTimeRetriever(base, min_score=min_score, use_relative_time=use_relative_time)


def build_retriever(name, store, min_score=0.0, use_relative_time=False,
                    settings=None, **_ignored):
    """name: tfidf | bm25 | bm25_anchor | dense | hybrid | hybrid_round  [+ '_rerank']"""
    normalized = (name or "bm25_anchor").lower().replace("+", "_").replace("-", "_")
    wants_rerank = normalized.endswith("_rerank")
    if wants_rerank:
        normalized = normalized[:-len("_rerank")]
    normalized = {"anchor": "bm25_anchor", "embedding": "dense",
                  "hybrid_rrf": "hybrid"}.get(normalized, normalized)

    # 关键修复：如果 settings 中指定了有效 reranker，无论名字有无 _rerank 后缀都强制启用
    if settings is not None and getattr(settings, "reranker", "none") not in ("none", ""):
        wants_rerank = True

    if normalized in {"tfidf", "bm25", "bm25_anchor"} and not wants_rerank:
        logger.info("[retriever] sparse=%s", normalized)
        return _sparse(normalized, store, min_score, use_relative_time)

    if settings is None:
        raise ValueError(f"retriever '{name}' requires Settings")

    from .embedding import build_dense
    from .hybrid import RRFHybridRetriever, RerankPipeline
    from .reranker import build_reranker

    if normalized == "dense":
        base = build_dense(store, settings)
    elif normalized in {"hybrid", "hybrid_round"}:
        sparse = _sparse("bm25_anchor", store, min_score, use_relative_time)
        dense = build_dense(store, settings)
        use_round = normalized == "hybrid_round" or bool(
            getattr(settings, "round_leg", False)
        )
        if use_round:
            from .round_fusion import RoundDenseIndex, RoundFusionRetriever

            rounds = RoundDenseIndex(
                store, dense.provider,
                cache_path=settings.embedding_cache_path,
                max_chars=getattr(settings, "round_max_chars", 2000),
                max_users=settings.index_cache_users,
            )
            fusion = getattr(settings, "fusion", "blend")
            if fusion == "znorm":
                weights = (getattr(settings, "dense_leg_weight", 1.0),
                           getattr(settings, "round_weight", 1.0),
                           getattr(settings, "sparse_leg_weight", 0.45))
            else:
                weights = (settings.dense_weight, getattr(settings, "round_weight", 1.0),
                           settings.sparse_weight)
            # 发射策略自动选择（与数据集无关）：
            #   有 reranker → adjacent：池里保留整轮两条消息（含 assistant 的答案
            #     文本），reranker 才有足够信息打分；实测 Mem-Gallery
            #     adjacent+rerank=0.3459 vs best_first+rerank=0.3261。
            #   无 reranker → best_first：固定条数覆盖尽可能多的不同 request。
            emit = getattr(settings, "round_emit", "auto")
            if emit not in {"best_first", "adjacent"}:
                emit = "adjacent" if wants_rerank else "best_first"
            view_index = None
            if getattr(settings, "view_window", False) or getattr(settings, "view_segment", False):
                common = dict(
                    max_chars=getattr(settings, "view_max_chars", 2000),
                    max_users=settings.index_cache_users,
                    window_size=getattr(settings, "view_window_size", 3),
                    window_overlap=getattr(settings, "view_window_overlap", 1),
                    segment_max_messages=getattr(settings, "view_segment_max_messages", 12),
                    segment_max_gap_seconds=getattr(settings, "view_segment_max_gap_seconds", 1800),
                    enable_window=getattr(settings, "view_window", False),
                    enable_segment=getattr(settings, "view_segment", False),
                )
                if getattr(settings, "view_mode", "lexical") == "lexical":
                    from .views import ViewSparseIndex
                    view_index = ViewSparseIndex(store, **common)
                else:
                    from .views import ViewDenseIndex
                    view_index = ViewDenseIndex(store, dense.provider,
                                                cache_path=settings.embedding_cache_path,
                                                **common)
                logger.info("[retriever] +views mode=%s window=%s(size=%d,ov=%d) "
                            "segment=%s(max=%d,gap=%ds)",
                            getattr(settings, "view_mode", "lexical"),
                            settings.view_window, settings.view_window_size,
                            settings.view_window_overlap, settings.view_segment,
                            settings.view_segment_max_messages,
                            settings.view_segment_max_gap_seconds)
            base = RoundFusionRetriever(
                sparse=sparse, dense=dense, rounds=rounds, views=view_index,
                view_weight=getattr(settings, "view_weight", 1.0),
                message_slot_ratio=getattr(settings, "message_slot_ratio", 0.5),
                max_candidates=getattr(settings, "max_candidates", 400),
                max_query_tokens=getattr(settings, "max_query_tokens", 64),
                rrf_k=getattr(settings, "round_rrf_k", 20),
                limit=settings.retrieval_limit,
                sparse_weight=weights[2],
                dense_weight=weights[0],
                round_weight=weights[1],
                fusion=fusion,
                emit=emit,
                fusion_pool=getattr(settings, "fusion_pool", "topk"),
                fusion_topk=settings.retrieval_limit,
                blend_beta=getattr(settings, "blend_beta", 0.5),
                rrf_weight_feature=getattr(settings, "rrf_weight_feature", 1.0),
                rrf_weight_lexical=getattr(settings, "rrf_weight_lexical", 1.0),
            )
            logger.info("[retriever] hybrid_round fusion=%s emit=%s k=%d "
                        "weights=(%.2f,%.2f,%.2f) rrf_w=(f%.2f,l%.2f)",
                        base.fusion, base.emit, base.rrf_k, base.dense_weight,
                        base.round_weight, base.sparse_weight,
                        base.rrf_weight_feature, base.rrf_weight_lexical)
        else:
            base = RRFHybridRetriever(
                sparse=sparse,
                dense=dense,
                limit=settings.retrieval_limit,
                rrf_k=settings.rrf_k,
                sparse_weight=settings.sparse_weight,
                dense_weight=settings.dense_weight,
            )
    else:
        base = _sparse(normalized, store, min_score, use_relative_time)

    # LLM 推理式重排与 cross-encoder 相互独立，两条路径都要能挂上（opt-in）
    if not wants_rerank:
        logger.info("[retriever] %s (no rerank)", normalized)
        return _maybe_wrap_llm(base, store, settings)

    reranker = build_reranker(settings)
    if reranker is None:
        logger.warning("[retriever] '%s' but AMC_RERANKER=none -> no rerank", name)
        return _maybe_wrap_llm(base, store, settings)
    logger.info("[retriever] %s + rerank=%s alpha=%.2f",
                normalized, settings.reranker, settings.rerank_alpha)
    pipeline = RerankPipeline(
        base, reranker,
        candidates=settings.reranker_candidates,
        alpha=settings.rerank_alpha,
        prior_mode=getattr(settings, "rerank_prior", "rank"),
        store=store,
        unit=getattr(settings, "rerank_unit", "turn"),
    )
    logger.info("[retriever] rerank_unit=%s", getattr(settings, "rerank_unit", "turn"))
    return _maybe_wrap_llm(pipeline, store, settings)


def _maybe_wrap_llm(inner, store, settings):
    """在检索链尾部追加 LLM 推理式重排（opt-in，默认关闭）。"""
    if not getattr(settings, "llm_rerank", False):
        return inner
    from .llm_rerank import LLMReasoningReranker, LLMReasoningRerankPipeline

    if not getattr(settings, "llm_rerank_api_key", ""):
        logger.warning("[retriever] llm_rerank enabled but no API key -> skipped")
        return inner
    reranker = LLMReasoningReranker(
        base_url=getattr(settings, "llm_rerank_base_url", ""),
        api_key=getattr(settings, "llm_rerank_api_key", ""),
        model=getattr(settings, "llm_rerank_model", "qwen-flash"),
        pool=getattr(settings, "llm_rerank_pool", 30),
        layout=getattr(settings, "llm_rerank_layout", "round"),
        doc_chars=getattr(settings, "llm_rerank_doc_chars", 1400),
        topk_ask=getattr(settings, "llm_rerank_topk_ask", 10),
        prompt_mode=getattr(settings, "llm_rerank_prompt", "round_rank"),
        timeout=float(getattr(settings, "llm_rerank_timeout", 20.0)),
    )
    gate = float(getattr(settings, "llm_rerank_gate_margin", 0.0) or 0.0)
    logger.info("[retriever] +llm_rerank model=%s pool=%d layout=%s doc_chars=%d "
                "gate_margin=%.4f prompt=%s dedup_rounds=%s",
                reranker.model, reranker.pool, reranker.layout, reranker.doc_chars, gate,
                reranker.prompt_mode, getattr(settings, "llm_rerank_dedup_rounds", False))
    return LLMReasoningRerankPipeline(inner, reranker, store,
                                      max_users=settings.index_cache_users,
                                      gate_margin=gate,
                                      skip_marker=getattr(settings,
                                                          "llm_rerank_skip_marker", ""),
                                      dedup_rounds=getattr(settings,
                                                           "llm_rerank_dedup_rounds", False))
