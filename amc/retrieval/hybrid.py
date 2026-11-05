"""RRF fusion (sparse + dense) and robust rerank pipeline."""
import logging
import math
import os
import sys

logger = logging.getLogger(__name__)


class RRFHybridRetriever:
    def __init__(
        self, sparse, dense, limit=120, rrf_k=60, sparse_weight=1.0, dense_weight=1.0
    ):
        self.sparse = sparse
        self.dense = dense
        self.limit = limit
        self.rrf_k = rrf_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def notify_add(self, user_id, request_id):
        hook = getattr(self.dense, "notify_add", None)
        if hook is not None:
            hook(user_id, request_id)

    def _safe(self, retriever, user_id, query, label):
        """**单腿失败让整题失败**：不降级，走重试。

         用户决策：
          降级（单腿返空）会**静默降低证据质量**，比 503 更糟
          ⇒ 回滚成 re-raise，改为**在 embedding 层加重试**：
             `AMC_EMBED_TIMEOUT` 15s→60s，`AMC_EMBED_RETRIES` 2→5。
        """
        try:
            return retriever.search(user_id, query, self.limit)
        except Exception:
            logger.exception("[hybrid] %s leg search failed", label)
            raise

    def _sparse_weight_for(self, user_id):
        """中文用户下调词法权重。

        离线实测（10 条平台真实中文查询 + 改写式中文记忆，见 scripts/zh_retrieval_probe.py）：
            dense only          rank@1 **10/10**
            dense + BM25(w=1.0) rank@1  **7/10**   ← 融合**拉低**了中文排序
            w=0.1 → 10/10 | w=0.2~0.5 → 9/10 | w≥0.7 → 7/10
        原因：中文里字符 n-gram 的区分度远低于英文词，BM25 的"弱但不为零"信号
        会把 dense 已经正确的第一名挤下去。
        ⇒ 语料 CJK 占比 >20% 时把 sparse 权重乘 `AMC_SPARSE_WEIGHT_CJK`（**默认 0 = 纯 dense**）。
        """
        w = float(self.sparse_weight)
        # ⚠️ `self.sparse` 可能被 AnchorTimeRetriever 等包装器包住（它没有 `_index`），
        # 原来直接调 `self.sparse._index` 抛异常后被 except 吞掉 ⇒ **权重永远不生效**。
        # 这里沿 `base` 链向下找真正带 `_index` 的检索器。
        obj, ratio = self.sparse, 0.0
        for _ in range(6):
            if obj is None:
                break
            fn = getattr(obj, "_index", None)
            if fn is not None:
                try:
                    ratio = float((fn(user_id) or {}).get("cjk_ratio") or 0.0)
                except Exception:
                    ratio = 0.0
                break
            obj = getattr(obj, "base", None)
        if ratio > 0.2:
            return w * float(os.getenv("AMC_SPARSE_WEIGHT_CJK", "0.0") or 0.0)
        return w

    def search(self, user_id, query, limit):
        #   把本次的 uid/query 记在实例上，供下游 RerankPipeline / llm_rerank 的
        #   stage_dump 关联同一次请求（否则 ce/llm 两级导出缺 uid，无法归并）。
        self._last_uid, self._last_q = user_id, query
        fused, objects = {}, {}
        _sw = self._sparse_weight_for(user_id)
        for weight, hits in (
            (_sw, self._safe(self.sparse, user_id, query, "sparse")),
            (self.dense_weight, self._safe(self.dense, user_id, query, "dense")),
        ):
            for rank, (memory, _) in enumerate(hits, start=1):
                key = memory["id"]
                objects[key] = memory
                fused[key] = fused.get(key, 0.0) + weight / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        _out = [(objects[key], float(score)) for key, score in ranked[:limit]]
        #   导出 **RRF 融合后、本地 cross-encoder 之前** 的排序。
        try:
            from amc import stage_dump as _sd
            _sd.dump("fused", _out, uid=user_id, q=query)
        except Exception:
            pass
        return _out

    def describe(self):
        return {
            "class": self.__class__.__name__,
            "limit": self.limit,
            "rrf_k": self.rrf_k,
            "sparse_weight": self.sparse_weight,
            "dense_weight": self.dense_weight,
        }


class RerankPipeline:
    """Rerank the base retriever's top-`candidates` pool.

    ``prior_mode`` controls the base-score prior used when ``alpha < 1``:

    ``rank``   the historical ``1 - i/n`` rank prior — it throws away every
               magnitude difference between the base scores.
    ``score``  min-max normalised base scores.  Meaningful only once the base
               emits calibrated scores; with the round-fusion retriever the
               exported score separates gold from non-gold inside the pool with
               AUC 0.79 (Mem-Gallery) / 0.92 (ATM-Bench) versus 0.71 / 0.89 for
               the old RRF scores, so the prior can actually carry signal.
    """

    def __init__(self, base, reranker, candidates=30, alpha=1.0, prior_mode="rank",
                 store=None, unit="turn"):
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.alpha = alpha
        self.prior_mode = (prior_mode or "rank").lower()
        # unit="round"：把同一 request（一整轮对话）的全部消息文本拼起来再交给 cross-encoder。
        # 理由：证据常在同一轮的另一条里，而 turn 级打分看不到它（S2 实测 Hit@1 +0.0269 ✅）。
        self.store = store
        self.unit = (unit or "turn").lower()

    def _round_texts(self, head):
        """request_id -> 该轮全部消息的 index_text 拼接（带 per-call 缓存）。"""
        cache = {}
        if self.store is None:
            return cache
        try:
            conn = self.store.conn
        except Exception:
            return cache
        for m, _s in head:
            rid = m.get("request_id")
            if not rid or rid in cache:
                continue
            try:
                rows = conn.execute(
                    "SELECT index_text FROM memories WHERE request_id=? ORDER BY msg_index",
                    (rid,)).fetchall()
            except Exception:
                rows = []
            cache[rid] = "\n".join((r[0] or "").strip() for r in rows if r[0]).strip()
        return cache

    def notify_add(self, user_id, request_id):
        hook = getattr(self.base, "notify_add", None)
        if hook is not None:
            hook(user_id, request_id)

    def search(self, user_id, query, limit):
        pool = self.base.search(user_id, query, max(self.candidates, limit))
        if not pool or self.reranker is None:
            return pool[:limit]
        head, tail = pool[:self.candidates], pool[self.candidates:]

        head_for_ce = head
        if self.unit == "round" and self.store is not None:
            try:
                rtext = self._round_texts(head)
                head_for_ce = []
                for m, sc_ in head:
                    rt = rtext.get(m.get("request_id")) or ""
                    if rt:
                        mm = dict(m)
                        mm["index_text"] = rt
                        head_for_ce.append((mm, sc_))
                    else:
                        head_for_ce.append((m, sc_))
            except Exception:
                head_for_ce = head
        try:
            reranked = self.reranker.rerank(query, head_for_ce, top_k=len(head_for_ce))
            try:
                from amc import stage_dump as _sd
                # ⚠️ ：**ce 阶段要导出"重排后的完整列表"**。
                #   原来只导出 reranked（= head 的 30 条）⇒ R@100 被当成"只有 30 条"
                #   ⇒ 表格里 ce 的 R@100=68% 是假低（实为 R@30）。
                #   这里把 reranked 与其后的 tail（未进 CE 的候选）按原顺序拼接，
                #   让 R@n 的 n 与最终链路可比。
                _ce_ids = [(m, s) for m, s in reranked]
                try:
                    _seen_ce = {str(m.get("id")) for m, _ in _ce_ids}
                    _tail_ce = [(m, s) for m, s in (tail or []) if str(m.get("id")) not in _seen_ce]
                except Exception:
                    _tail_ce = []
                _sd.dump("ce", _ce_ids + _tail_ce,
                         uid=getattr(self.base, "_last_uid", None), q=query)
            except Exception:
                pass
        except Exception as exc:
            # 严格遵循“重新部署而非退化”原则：
            # 当 GPU 发生致命错误时，绝不吞掉异常默默退化为 base 检索结果，
            # 而是主动退出当前进程，让 supervisor 守护程序重新拉起部署 Memory 端！
            logger.critical(
                "[rerank] GPU fatal error encountered: %s. "
                "Aborting process to trigger supervisor re-deployment without degradation!",
                exc,
                exc_info=True,
            )
            # 使用 os._exit(1) 强制退出，让 CUDA 脏上下文被系统清理
            os._exit(1)

        # round 单位下 reranked 里是**打分为轮文本的副本**：按 id 还原为原始记忆，
        # 避免把"整轮拼接文本"泄漏到下游（去重/水合/返回内容）。
        if head_for_ce is not head:
            by_id = {m.get("id"): m for m, _ in head}
            reranked = [(by_id.get(mm.get("id"), mm), sc_) for mm, sc_ in reranked]

        n = max(len(head), 1)
        if self.prior_mode == "score":
            raw = [float(s) for _, s in head]
            lo, hi = min(raw), max(raw)
            span = hi - lo
            if span > 1e-12:
                prior = {memory["id"]: (float(score) - lo) / span
                         for memory, score in head}
            else:
                prior = {memory["id"]: 0.5 for memory, _ in head}
        else:
            prior = {memory["id"]: 1.0 - i / n for i, (memory, _) in enumerate(head)}
        merged = []
        for memory, score in reranked:
            value = float(score)
            if not math.isfinite(value):
                value = -1.0
            if self.alpha < 1.0:
                value = self.alpha * value + (1.0 - self.alpha) * prior.get(
                    memory["id"], 0.0
                )
            merged.append((memory, value))
        merged.sort(key=lambda item: item[1], reverse=True)

        output = merged
        if len(output) < limit and tail:
            floor = output[-1][1] if output else 0.0
            for offset, (memory, _) in enumerate(tail[: limit - len(output)], start=1):
                output.append((memory, floor - 1e-4 * offset))
        return output[:limit]

    def describe(self):
        return {
            "class": self.__class__.__name__,
            "candidates": self.candidates,
            "alpha": self.alpha,
            "prior_mode": self.prior_mode,
        }
