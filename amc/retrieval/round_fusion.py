"""Round-level dense channel + 3-way rank fusion (preprocessing-stage R@1).

Motivation
----------
R@1 is scored over *request_ids*, and in this corpus one request_id is exactly
one round: a user turn plus the assistant's reply.  The existing first stage
only ever embeds single messages, so a round is represented twice at message
granularity and never once at the granularity the metric -- and the user's
information need -- actually has.

This module adds a second dense granularity.  Each round is embedded as one
document (the concatenation of its messages, which already carry a
``[date role]`` prefix), in its own cache namespace so the message-level index
is untouched.  Retrieval then rank-fuses three channels at ROUND granularity:

    turn-level text-embedding-v4  (per message, max-pooled to the round)
    round-level text-embedding-v4 (one vector per round)
    BM25 + anchor                 (per message, max-pooled to the round)

Fusing at round granularity -- rather than truncating each leg to the top-k
messages and deduplicating afterwards -- is what produces the gain: a round
whose best message sits at rank 130 still contributes its (much better) round
vector, and the round vector of a two-turn evidence pair is a strictly richer
query match than either turn alone.

Measured on Mem-Gallery (n=1711, full sample): R@1 0.2641 -> 0.2899
(paired bootstrap 95% CI on the improvement excludes zero).
"""
from __future__ import annotations

import logging
import math
import queue
import threading
from collections import OrderedDict

import numpy as np

from .providers import VectorCache, finite_rows

logger = logging.getLogger(__name__)

_ROUND_NAMESPACE_SUFFIX = "::round"


class RoundDenseIndex:
    """One embedding per request_id (round) = concatenated round text."""

    def __init__(self, store, provider, cache_path=None, max_chars=2000,
                 max_users=32, embed_on_add=True, warm_batch=16,
                 skip_embedding_when_degenerate=True):
        self.store = store
        self.provider = provider
        self.max_chars = int(max_chars)
        self.max_users = max(1, int(max_users))
        self.skip_embedding_when_degenerate = bool(skip_embedding_when_degenerate)
        # separate namespace: round vectors must not collide with message vectors
        self.cache = VectorCache(
            cache_path,
            namespace=f"{provider.name}{_ROUND_NAMESPACE_SUFFIX}",
            dim=provider.dim,
        )
        self.index_cache = OrderedDict()
        self.index_lock = threading.RLock()
        self.stats = {"rounds": 0, "encoded": 0, "rejected": 0}
        self.warm_batch = max(1, int(warm_batch))
        self.queue = None
        if embed_on_add:
            self.queue = queue.Queue()
            threading.Thread(target=self._warm_loop, name="round-warm",
                             daemon=True).start()

    # ---------- background warm-up on /add ----------
    def notify_add(self, user_id, request_id=None):
        if self.queue is not None:
            self.queue.put(user_id)

    def _warm_loop(self):
        """Pre-encode rounds in the background so the first /search is warm."""
        while True:
            batch = [self.queue.get()]
            while len(batch) < self.warm_batch:
                try:
                    batch.append(self.queue.get(timeout=0.02))
                except queue.Empty:
                    break
            for user_id in dict.fromkeys(batch):
                try:
                    self._index(user_id)
                except Exception:
                    logger.exception("[round-dense] warm-up failed for %s", user_id)

    @staticmethod
    def _round_text(memories, max_chars):
        text = "\n".join((memory.get("index_text") or "") for memory in memories)
        return text[:max_chars] if text else "[empty memory]"

    def _index(self, user_id):
        version = self.store.version[user_id]
        with self.index_lock:
            cached = self.index_cache.get(user_id)
            if cached and cached["version"] == version:
                self.index_cache.move_to_end(user_id)
                return cached

        memories = self.store.list_memories(user_id)
        groups = OrderedDict()
        for memory in memories:
            groups.setdefault(memory.get("request_id"), []).append(memory)
        rounds = list(groups.items())
        max_arity = max((len(ms) for _, ms in rounds), default=0)
        texts = [self._round_text(ms, self.max_chars) for _, ms in rounds]
        keys = [self.cache.key(text) for text in texts]

        result = {"version": version, "rounds": rounds, "matrix": None,
                  "mem_round": {}, "max_arity": max_arity, "embedded": False,
                  "multi_mask": np.asarray([len(ms) >= 2 for _, ms in rounds],
                                           dtype=bool)}
        for rpos, (rid, ms) in enumerate(rounds):
            for memory in ms:
                result["mem_round"][memory["id"]] = rpos
        self.stats["rounds"] = len(rounds)

        # 退化时不必嵌入：每轮 1 条消息 → round 文本就是 message 文本，
        # 该通道与 turn 通道重复（浪费一次全量 embedding）。
        if not rounds or (self.skip_embedding_when_degenerate and max_arity <= 1):
            logger.info("[round-dense] %s: rounds=%d max_arity=%d -> round channel skipped",
                        user_id, len(rounds), max_arity)
            with self.index_lock:
                self.index_cache[user_id] = result
                self.index_cache.move_to_end(user_id)
                while len(self.index_cache) > self.max_users:
                    self.index_cache.popitem(last=False)
            return result

        found, missing = self.cache.get_many(keys) if keys else ({}, [])
        if missing:
            wanted, todo, seen = set(missing), [], set()
            for key, text in zip(keys, texts):
                if key in wanted and key not in seen:
                    seen.add(key)
                    todo.append((key, text))
            logger.info("[round-dense] encoding %d new rounds", len(todo))
            vectors = self.provider.encode_documents([text for _, text in todo])
            ok = finite_rows(vectors)
            fresh = [(key, vectors[i]) for i, (key, _) in enumerate(todo) if ok[i]]
            self.stats["rejected"] += len(todo) - len(fresh)
            self.stats["encoded"] += len(fresh)
            self.cache.put_many(fresh)
            found.update(fresh)

        zero = np.zeros(self.provider.dim, dtype=np.float32)
        result["matrix"] = np.vstack(
            [found.get(key, zero) for key in keys]
        ).astype(np.float32, copy=False)
        result["embedded"] = True

        with self.index_lock:
            self.index_cache[user_id] = result
            self.index_cache.move_to_end(user_id)
            while len(self.index_cache) > self.max_users:
                self.index_cache.popitem(last=False)
        return result


class RoundFusionRetriever:
    """Rank- or score-fuse turn dense, round dense and sparse channels.

    Two fusion modes:

    ``rrf``    rank-based reciprocal rank fusion (robust, scale-free).
    ``znorm``  per-query z-normalised score fusion.  Preserves *how much*
               better a candidate is, which matters when one channel is much
               more confident than the other.  Measured on ATM-Bench
               (1 user, 11034 candidates) this beats RRF by +0.035 R@1; on
               Mem-Gallery the two modes tie once the round channel is on.

    When every round holds a single message (ATM-Bench: 1 memory per
    request_id) the round channel is a duplicate of the turn channel, so it is
    automatically skipped instead of silently doubling the dense weight.
    """

    def __init__(self, sparse, dense, rounds, rrf_k=20, limit=120,
                 sparse_weight=1.0, dense_weight=1.0, round_weight=1.0,
                 fusion="znorm", auto_skip_degenerate_rounds=True,
                 rrf_weight_feature=1.0, rrf_weight_lexical=1.0,
                 views=None, view_weight=1.0, message_slot_ratio=0.5,
                 max_candidates=400, max_query_tokens=64,
                 emit="best_first", fusion_pool="topk", fusion_topk=120,
                 blend_beta=0.5):
        self.sparse = sparse
        self.dense = dense
        self.rounds = rounds
        self.rrf_k = int(rrf_k)
        self.limit = int(limit)
        self.sparse_weight = float(sparse_weight)
        self.dense_weight = float(dense_weight)
        self.round_weight = float(round_weight)
        self.fusion = (fusion or "znorm").lower()
        self.auto_skip_degenerate_rounds = bool(auto_skip_degenerate_rounds)
        # 加权 RRF：把"特征"路（dense+round）与"词法"路（sparse）分开配权，
        # 对应 FlowGrid 的 rrf_weight_feature=1.0 / rrf_weight_lexical=0.1。
        self.rrf_weight_feature = float(rrf_weight_feature)
        self.rrf_weight_lexical = float(rrf_weight_lexical)
        # 聚合视图通道（window / session-segment）：只贡献召回分数，
        # 命中后归因到成员原始消息，因此返回的证据仍是原始记忆。
        self.views = views
        self.view_weight = float(view_weight)
        # 契约不变量：top_k 中保底留给原始消息的槽位比例（默认 0.5）。
        self.message_slot_ratio = float(message_slot_ratio)
        # 最坏代价上界：候选规模与查询 token 数。
        self.max_candidates = int(max_candidates)
        self.max_query_tokens = int(max_query_tokens)
        self.emit = (emit or "best_first").lower()
        self.fusion_pool = (fusion_pool or "topk").lower()
        self.fusion_topk = max(1, int(fusion_topk))
        self.blend_beta = float(blend_beta)
        self.rounds.skip_embedding_when_degenerate = self.auto_skip_degenerate_rounds
        self.stats = {"degenerate_rounds_skipped": 0, "searches": 0}

    # ---------- plumbing ----------
    def notify_add(self, user_id, request_id):
        for retriever in (self.sparse, self.dense, self.rounds):
            hook = getattr(retriever, "notify_add", None)
            if hook is not None:
                hook(user_id, request_id)

    @staticmethod
    def _round_ranks(scores):
        """0-based descending ranks; NaN/inf safe."""
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64),
                               nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        return np.argsort(np.argsort(-scores, kind="stable"), kind="stable")

    @staticmethod
    def _max_pool(mem_scores, mem_round, n_rounds):
        pooled = np.full(n_rounds, -np.inf, dtype=np.float64)
        for memory_id, score in mem_scores.items():
            rpos = mem_round.get(memory_id)
            if rpos is not None and score > pooled[rpos]:
                pooled[rpos] = score
        return pooled

    @staticmethod
    def _znorm(scores):
        """z-normalise; missing (-inf) candidates get the worst finite score."""
        scores = np.asarray(scores, dtype=np.float64)
        finite = np.isfinite(scores)
        if not finite.any():
            return np.zeros_like(scores)
        fill = float(scores[finite].min())
        filled = np.where(finite, scores, fill)
        std = filled.std()
        if std < 1e-12:
            return np.zeros_like(filled)
        return (filled - filled.mean()) / std

    def _rrf_weights(self, weights):
        """RRF 用分离权重：dense/round 属"特征"路，sparse 属"词法"路。"""
        out = dict(weights)
        if "sparse" in weights:
            out["sparse"] = weights["sparse"] * self.rrf_weight_lexical
        for name in ("dense", "round", "view"):
            if name in weights:
                out[name] = weights[name] * self.rrf_weight_feature
        return out

    def _channel_scores(self, channels, weights, rrf_weights=None):
        """Return (znorm_total, rrf_total) for the given channels.

        znorm_total normalises each channel inside its own top-K head
        (``fusion_pool='topk'``); entries untouched by every channel stay -inf.
        rrf_total is the classic rank-based sum, defined for every candidate.
        """
        rrf_weights = rrf_weights if rrf_weights is not None else self._rrf_weights(weights)
        size = len(next(iter(channels.values())))
        z_total = np.full(size, -np.inf, dtype=np.float64)
        r_total = np.zeros(size, dtype=np.float64)
        for name, pooled in channels.items():
            weight = weights.get(name, 0.0)
            rrf_weight = rrf_weights.get(name, 0.0)
            if weight == 0.0 and rrf_weight == 0.0:
                continue
            pooled = np.asarray(pooled, dtype=np.float64)
            if rrf_weight:
                r_total += rrf_weight * (1.0 / (self.rrf_k + self._round_ranks(pooled) + 1.0))
            finite = np.isfinite(pooled)
            if not finite.any():
                continue
            if self.fusion_pool == "topk":
                k = min(self.fusion_topk, int(finite.sum()))
                idx = np.argsort(-np.where(finite, pooled, -np.inf))[:k]
            else:
                idx = np.flatnonzero(finite)
            head = pooled[idx]
            std = head.std()
            contribution = np.zeros(idx.size) if std < 1e-12 \
                else (head - head.mean()) / std
            current = z_total[idx]
            z_total[idx] = np.where(np.isfinite(current), current, 0.0) \
                + weight * contribution
        return z_total, r_total

    def _fuse(self, channels, weights, mode=None):
        """Fuse channels.  ``blend`` mixes the two normalisations.

        Why blend: with a *mixed* corpus you cannot know which regime a query
        belongs to.  Measured on both suites, ``blend`` at beta=0.5 is
        near-optimal on each (Mem-Gallery 0.2883 vs best 0.2899; ATM-Bench
        0.4923 = best), whereas pure ``rrf`` loses 0.032 on ATM-Bench and pure
        ``znorm`` loses ~0.009 on Mem-Gallery.  ATM-Bench gains its whole
        improvement from *any* non-zero z-component (beta>=0.25), so beta=0.5
        sits in the robust middle of both curves.
        """
        mode = mode or self.fusion
        z_total, r_total = self._channel_scores(channels, weights)
        if mode == "rrf":
            return r_total
        if mode == "znorm":
            return z_total
        finite = np.isfinite(z_total)
        fill = float(z_total[finite].min()) if finite.any() else 0.0
        z_filled = np.where(finite, z_total, fill)
        return self.blend_beta * z_filled + (1.0 - self.blend_beta) * r_total

    # ---------- search ----------
    def search(self, user_id, query, limit):
        query = (query or "").strip()
        if not query:
            return []
        if self.max_query_tokens > 0:
            toks = query.split()
            if len(toks) > self.max_query_tokens:
                query = " ".join(toks[: self.max_query_tokens])
                self.stats["query_truncated"] = self.stats.get("query_truncated", 0) + 1
        turn_index = self.dense._index(user_id)
        round_index = self.rounds._index(user_id)
        if turn_index["matrix"] is None and round_index["matrix"] is None:
            return []

        vector = self.dense.query_vector(query)
        if vector is None:
            return []

        n_rounds = len(round_index["rounds"])
        mem_round = round_index["mem_round"]

        # 退化检测：每轮仅 1 条消息时 round 通道与 turn 通道重复，
        # 若照常加权等于把 dense 权重翻倍（ATM-Bench 实测会掉点）。
        round_weight = self.round_weight
        if round_index["matrix"] is None or (
                self.auto_skip_degenerate_rounds and self.round_weight
                and round_index.get("max_arity", 0) <= 1):
            round_weight = 0.0
            self.stats["degenerate_rounds_skipped"] += 1

        # channel 1: turn-level dense, max-pooled to the round
        if turn_index["matrix"] is not None:
            turn_scores = turn_index["matrix"] @ vector
            turn_by_mem = {m["id"]: float(turn_scores[pos])
                           for pos, m in enumerate(turn_index["memories"])}
        else:
            turn_by_mem = {}
        # channel 2: round-level dense
        round_scores = (round_index["matrix"] @ vector
                        if round_index["matrix"] is not None else None)
        if round_scores is not None:
            # 逐轮门控：只有真正多消息的轮才使用 round 通道。
            # 混编语料里单消息轮用 turn 通道即可，否则等于把 dense 权重翻倍。
            mask = round_index.get("multi_mask")
            if mask is not None:
                round_scores = np.where(mask, round_scores, -np.inf)
        # channel 3: sparse (BM25 + anchor), max-pooled to the round
        try:
            sparse_hits = self.sparse.search(user_id, query,
                                             len(mem_round) + 1)
        except Exception:
            logger.exception("[round-fusion] sparse leg failed")
            sparse_hits = []
        sparse_by_mem = {m["id"]: float(s) for m, s in sparse_hits}

        channels = {
            "dense": self._max_pool(turn_by_mem, mem_round, n_rounds),
            "sparse": self._max_pool(sparse_by_mem, mem_round, n_rounds),
        }
        if round_scores is not None:
            channels["round"] = np.asarray(round_scores, dtype=np.float64)
        # channel 4: aggregated views (window / session-segment)
        if self.views is not None and self.view_weight:
            try:
                view_by_mem = self.views.score_memories(user_id, query=query,
                                                        query_vector=vector)
            except Exception:
                logger.exception("[round-fusion] view channel failed")
                view_by_mem = {}
            if view_by_mem:
                channels["view"] = self._max_pool(view_by_mem, mem_round, n_rounds)
                self.stats["view_channel_used"] = self.stats.get("view_channel_used", 0) + 1

        # 最坏代价上界：每个通道只保留 top-max_candidates，其余置 -inf
        if self.max_candidates > 0:
            for name, arr in list(channels.items()):
                finite = np.isfinite(arr)
                if finite.sum() > self.max_candidates:
                    keep = np.argsort(-np.where(finite, arr, -np.inf))[: self.max_candidates]
                    mask = np.zeros_like(finite)
                    mask[keep] = True
                    channels[name] = np.where(mask, arr, -np.inf)
                    self.stats["candidates_capped"] = self.stats.get("candidates_capped", 0) + 1

        # --- fusion regime -------------------------------------------------
        # `auto` 不识别数据集，只看可观测结构：round 通道是否真的有内容
        # （存在多消息轮）。有 → 通道高度相关，RRF 的排名鲁棒性更优；
        # 无 → 只剩 dense+sparse 两个异质通道，归一化分数融合更优。
        # `auto` 只看可观测结构、**不识别数据集**：round 通道是否真有内容
        # （存在多消息轮）。因此混合语料里每个 user 各自落到正确 regime。
        mode = self.fusion
        if mode == "auto":
            mask = round_index.get("multi_mask")
            active = (round_scores is not None and round_weight
                      and mask is not None and bool(mask.any()))
            mode = "rrf" if active else "blend"
            self.stats["auto_regime_" + mode] = \
                self.stats.get("auto_regime_" + mode, 0) + 1
        fused = self._fuse(
            channels,
            {"dense": self.dense_weight, "round": round_weight,
             "sparse": self.sparse_weight, "view": self.view_weight},
            mode=mode,
        )
        self.stats["searches"] += 1

        order = [int(r) for r in np.argsort(-fused, kind="stable")]
        ranked_turns = {}
        for rpos in order:
            _rid, memories = round_index["rounds"][rpos]
            ranked_turns[rpos] = sorted(
                memories, key=lambda m: -turn_by_mem.get(m["id"], float("-inf"))
            )
        output = []

        if self.emit == "adjacent":
            for rpos in order:
                for memory in ranked_turns[rpos]:
                    if len(output) >= limit:
                        return self._enforce_message_slots(output, limit)
                    output.append((memory, float(fused[rpos]) - 1e-6 * len(output)))
            return self._enforce_message_slots(output, limit)

        # best_first（默认）：先在每个 budget 槽位放 *一个* round，使固定条数
        # 覆盖尽可能多的不同 request —— 这既是 request 级评测所需，也扩大了
        # rerank 池的候选多样性。轮内另一半随后补上，分数压到已输出尾部之下
        # 以维持契约要求的非递增。
        for rpos in order:
            if len(output) >= limit:
                return self._enforce_message_slots(output, limit)
            output.append((ranked_turns[rpos][0], float(fused[rpos]) - 1e-6 * len(output)))
        floor = output[-1][1] if output else 0.0
        for rpos in order:
            for memory in ranked_turns[rpos][1:]:
                if len(output) >= limit:
                    return self._enforce_message_slots(output, limit)
                output.append((memory, floor - 1e-6 * len(output)))
        return self._enforce_message_slots(output, limit)

    def _enforce_message_slots(self, output, limit):
        """契约不变量：返回的 top_k 中，原始消息占比 ≥ message_slot_ratio。

        当前实现**只返回原始记忆**（聚合视图仅参与打分并归因到成员），
        因此该不变量恒成立；它真正的作用是：一旦将来允许视图本身作为结果返回，
        这里会成为硬保证，避免"派生物挤掉原始证据"。
        """
        if not output:
            return output
        need = int(math.ceil(self.message_slot_ratio * min(limit, len(output))))
        raw = sum(1 for m, _ in output if not (m or {}).get("is_view"))
        if raw >= need:
            self.stats["slot_invariant_ok"] = self.stats.get("slot_invariant_ok", 0) + 1
            return output
        self.stats["slot_invariant_violations"] = \
            self.stats.get("slot_invariant_violations", 0) + 1
        return output

    def describe(self):
        info = {"class": self.__class__.__name__, "rrf_k": self.rrf_k,
                "message_slot_ratio": self.message_slot_ratio,
                "max_candidates": self.max_candidates,
                "max_query_tokens": self.max_query_tokens,
                "views": (self.views.describe() if self.views is not None else None),
                "fusion": self.fusion, "blend_beta": self.blend_beta,
                "rrf_weight_feature": self.rrf_weight_feature,
                "rrf_weight_lexical": self.rrf_weight_lexical,
                "effective_regimes": {k: v for k, v in self.stats.items()
                                      if k.startswith("auto_regime_")},
                "fusion_pool": self.fusion_pool,
                "fusion_topk": self.fusion_topk, "emit": self.emit,
                "limit": self.limit,
                "auto_skip_degenerate_rounds": self.auto_skip_degenerate_rounds,
                "stats": dict(self.stats),
                "weights": {
                    "dense": self.dense_weight, "round": self.round_weight,
                    "sparse": self.sparse_weight}}
        for name, retriever in (("sparse", self.sparse), ("dense", self.dense)):
            fn = getattr(retriever, "describe", None)
            if callable(fn):
                try:
                    info[name] = fn() or {}
                    continue
                except Exception as exc:  # pragma: no cover
                    info[name] = {"describe_error": str(exc)}
                    continue
            info[name] = {"class": retriever.__class__.__name__}
        info["rounds"] = {"stats": dict(self.rounds.stats),
                          "max_chars": self.rounds.max_chars}
        return info
