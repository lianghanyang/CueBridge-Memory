"""聚合检索视图（window / session-segment）—— 借鉴 FlowGrid AML Retriever。

FlowGrid 的做法（`views.py` + `docs/EVAL.md` 的 L1 档）：
  * `window`          : 会话内按序的确定性滑动窗口，聚合 N 条连续消息
  * `session-segment` : 按会话边界（消息数上限 / 时间间隙）聚合成段
  * 每条聚合视图必须携带 `source_message_ids` 可回指原始消息（provenance）
  * `*_starts` / `segment_boundaries` 支持增量重建，且结果与全量扫描一致
  * `RetrieverConfig.message_slot_ratio = 0.5`：top_k 中保底 ≥50% 槽位留给**原始消息**

本项目的落地方式（与契约一致）：
  视图**只用于召回打分**，命中后按 `source_message_ids` 把分数归因到成员**原始消息**
  （取成员上的最大视图分），因此 `/search` 返回的仍然只有原始记忆。
  这样"视图帮助召回、原始消息才是证据"这条契约在实现层就被满足了。

参数直接采用 FlowGrid 的默认：window size=3 / overlap=1；segment ≤12 条 / ≤1800 秒。
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from .providers import VectorCache

logger = logging.getLogger(__name__)

_VIEW_NAMESPACE_SUFFIX = "::view"


def window_starts(total: int, size: int, overlap: int) -> list[int]:
    """确定性滑窗起点（与 FlowGrid 同语义：步长 = size - overlap）。"""
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    step = max(1, size - overlap)
    return list(range(0, max(0, total - size + 1), step)) or ([0] if total else [])


def segment_boundaries(messages, max_messages=12, max_gap_seconds=1800):
    """返回 [(start, end_exclusive)]：按条数上限或时间间隙切段。"""
    max_messages = max(1, int(max_messages))
    bounds, start, prev = [], 0, None
    for pos, m in enumerate(messages):
        gap = None
        if prev is not None:
            t0, t1 = (prev.get("ts") or 0), (m.get("ts") or 0)
            if t0 and t1:
                gap = abs(t1 - t0)
        if prev is not None and ((pos - start) >= max_messages
                                 or (gap is not None and gap > max_gap_seconds)):
            bounds.append((start, pos))
            start = pos
        prev = m
    if messages:
        bounds.append((start, len(messages)))
    return bounds


def _order(memories):
    return sorted(memories, key=lambda m: (m.get("session_id") or "",
                                           m.get("ts") or 0,
                                           m.get("msg_index") or 0))


def build_views(memories, window_size=3, window_overlap=1,
                segment_max_messages=12, segment_max_gap_seconds=1800,
                enable_window=True, enable_segment=True, max_chars=2000):
    """构造 (视图列表, 元数据)。每条视图带 member_ids 以回指原始消息。"""
    memories = list(memories or [])
    views, meta = [], {"window_starts": {}, "segment_boundaries": {}}
    by_session = OrderedDict()
    for m in _order(memories):
        by_session.setdefault(m.get("session_id") or "", []).append(m)

    for sid, ms in by_session.items():
        if enable_window:
            starts = window_starts(len(ms), window_size, window_overlap)
            meta["window_starts"][sid] = starts
            for start in starts:
                chunk = ms[start:start + max(1, int(window_size))]
                if not chunk:
                    continue
                views.append({
                    "view_id": f"w::{sid}::{start}",
                    "view_type": "window",
                    "member_ids": [c["id"] for c in chunk],
                    "text": "\n".join((c.get("index_text") or "") for c in chunk)[:max_chars],
                })
        if enable_segment:
            bounds = segment_boundaries(ms, segment_max_messages, segment_max_gap_seconds)
            meta["segment_boundaries"][sid] = bounds
            for idx, (a, b) in enumerate(bounds):
                chunk = ms[a:b]
                if not chunk:
                    continue
                views.append({
                    "view_id": f"s::{sid}::{idx}",
                    "view_type": "session-segment",
                    "member_ids": [c["id"] for c in chunk],
                    "text": "\n".join((c.get("index_text") or "") for c in chunk)[:max_chars],
                })
    return views, meta


class ViewDenseIndex:
    """聚合视图的 dense 索引；分数按成员归因到原始消息（成员取最大视图分）。"""

    def __init__(self, store, provider, cache_path=None, max_chars=2000,
                 max_users=32, window_size=3, window_overlap=1,
                 segment_max_messages=12, segment_max_gap_seconds=1800,
                 enable_window=True, enable_segment=True):
        self.store = store
        self.provider = provider
        self.max_chars = int(max_chars)
        self.max_users = max(1, int(max_users))
        self.window_size = int(window_size)
        self.window_overlap = int(window_overlap)
        self.segment_max_messages = int(segment_max_messages)
        self.segment_max_gap_seconds = int(segment_max_gap_seconds)
        self.enable_window = bool(enable_window)
        self.enable_segment = bool(enable_segment)
        self.cache = VectorCache(cache_path, namespace=f"{provider.name}{_VIEW_NAMESPACE_SUFFIX}",
                                 dim=provider.dim)
        self.index_cache = OrderedDict()
        self.index_lock = threading.RLock()
        self.stats = {"views": 0, "encoded": 0}

    def _index(self, user_id):
        version = self.store.version[user_id]
        with self.index_lock:
            hit = self.index_cache.get(user_id)
            if hit and hit["version"] == version:
                self.index_cache.move_to_end(user_id)
                return hit
        memories = self.store.list_memories(user_id)
        views, meta = build_views(
            memories, window_size=self.window_size, window_overlap=self.window_overlap,
            segment_max_messages=self.segment_max_messages,
            segment_max_gap_seconds=self.segment_max_gap_seconds,
            enable_window=self.enable_window, enable_segment=self.enable_segment,
            max_chars=self.max_chars)
        mem_pos = {m["id"]: pos for pos, m in enumerate(memories)}
        result = {"version": version, "views": views, "matrix": None,
                  "mem_pos": mem_pos, "meta": meta, "embedded": False,
                  "n_memories": len(memories)}
        self.stats["views"] = len(views)
        if not views:
            with self.index_lock:
                self.index_cache[user_id] = result
                self.index_cache.move_to_end(user_id)
            return result
        texts = [v["text"] or "[empty view]" for v in views]
        keys = [self.cache.key(t) for t in texts]
        found, missing = self.cache.get_many(keys)
        if missing:
            from .providers import finite_rows
            wanted, todo, seen = set(missing), [], set()
            for key, text in zip(keys, texts):
                if key in wanted and key not in seen:
                    seen.add(key)
                    todo.append((key, text))
            logger.info("[views] encoding %d new views (%d cached)", len(todo), len(found))
            vectors = self.provider.encode_documents([t for _, t in todo])
            ok = finite_rows(vectors)
            fresh = [(key, vectors[i]) for i, (key, _) in enumerate(todo) if ok[i]]
            self.cache.put_many(fresh)
            found.update(fresh)
            self.stats["encoded"] += len(fresh)
        rows = [found.get(k) for k in keys]
        if any(r is None for r in rows):
            logger.warning("[views] %d view vectors unavailable; channel disabled",
                           sum(1 for r in rows if r is None))
            result["matrix"] = None
        else:
            result["matrix"] = np.asarray(rows, dtype=np.float32)
        result["embedded"] = True
        self.stats["encoded"] = len(views)
        with self.index_lock:
            self.index_cache[user_id] = result
            self.index_cache.move_to_end(user_id)
            while len(self.index_cache) > self.max_users:
                self.index_cache.popitem(last=False)
        return result

    def score_memories(self, user_id, query=None, query_vector=None):
        """返回 {memory_id: 视图分}（成员取最大视图分）；无视图时返回 {}。"""
        index = self._index(user_id)
        if index["matrix"] is None or query_vector is None:
            return {}
        view_scores = index["matrix"] @ query_vector
        out = {}
        for pos, view in enumerate(index["views"]):
            s = float(view_scores[pos])
            for mid in view["member_ids"]:
                if s > out.get(mid, -np.inf):
                    out[mid] = s
        return out

    def describe(self):
        return {"class": self.__class__.__name__,
                "window": {"size": self.window_size, "overlap": self.window_overlap,
                           "enabled": self.enable_window},
                "segment": {"max_messages": self.segment_max_messages,
                            "max_gap_seconds": self.segment_max_gap_seconds,
                            "enabled": self.enable_segment},
                "stats": dict(self.stats)}


class _ViewStore:
    """把「聚合视图」伪装成记忆，从而**原样复用**既有的词法检索器（BM25+anchor）。

    只暴露 BM25Retriever / AnchorTimeRetriever 真正用到的两个接口：
    ``version[user_id]`` 与 ``list_memories(user_id)``。
    这就是 FlowGrid 的做法（他们的视图由 SQLite FTS5 索引，零 embedding 成本）。
    """

    def __init__(self, inner, views, inner_version):
        self._inner_version = inner_version
        self._inner = inner
        self._by_user = {}
        for view in views:
            for mid in view["member_ids"]:
                self._by_user.setdefault(mid, view)
        self._views = views
        self.version = _VersionMap(inner, views)

    def list_memories(self, user_id):          # pragma: no cover - 由 _VersionMap 驱动
        return self._views


class _VersionMap(dict):
    """让视图索引随底层 store 的版本一起失效。"""

    def __init__(self, inner, views):
        super().__init__()
        self._inner = inner
        self._views = views

    def __getitem__(self, user_id):
        views = self._views if not callable(self._views) else self._views(user_id)
        return (self._inner.version[user_id], len(views), id(views))


class ViewSparseIndex:
    """聚合视图的**词法**索引（BM25 + anchor），分数按成员归因到原始消息。

    与 ViewDenseIndex 的区别：不需要任何 embedding，成本≈0 —— 这正是 FlowGrid
    用 FTS5 索引视图的原因（他们的 L1 档"视图"因此很便宜）。
    """

    def __init__(self, store, kind="bm25_anchor", min_score=0.0,
                 use_relative_time=False, max_users=32, window_size=3,
                 window_overlap=1, segment_max_messages=12,
                 segment_max_gap_seconds=1800, enable_window=True,
                 enable_segment=True, max_chars=2000):
        self.store = store
        self.kind = kind
        self.min_score = min_score
        self.use_relative_time = use_relative_time
        self.max_users = max(1, int(max_users))
        self.window_size = int(window_size)
        self.window_overlap = int(window_overlap)
        self.segment_max_messages = int(segment_max_messages)
        self.segment_max_gap_seconds = int(segment_max_gap_seconds)
        self.enable_window = bool(enable_window)
        self.enable_segment = bool(enable_segment)
        self.max_chars = int(max_chars)
        self.index_cache = OrderedDict()
        self.index_lock = threading.RLock()
        self.stats = {"views": 0, "users": 0}

    def _pseudo(self, user_id):
        with self.index_lock:
            hit = self.index_cache.get(user_id)
            if hit and hit["version"] == self.store.version[user_id]:
                self.index_cache.move_to_end(user_id)
                return hit
        memories = self.store.list_memories(user_id)
        views, meta = build_views(
            memories, window_size=self.window_size, window_overlap=self.window_overlap,
            segment_max_messages=self.segment_max_messages,
            segment_max_gap_seconds=self.segment_max_gap_seconds,
            enable_window=self.enable_window, enable_segment=self.enable_segment,
            max_chars=self.max_chars)
        by_id = {m["id"]: m for m in memories}
        pseudo = []
        for view in views:
            # 继承首个成员的**全部字段**（anchor/time 逻辑依赖 created_at/ts/role 等），
            # 再覆盖为视图自身的标识与文本。
            base = dict(by_id.get(view["member_ids"][0]) or {})
            base.update({
                "id": view["view_id"],
                "request_id": view["view_id"],
                "index_text": view["text"],
                "is_view": True,
                "view_type": view["view_type"],
                "member_ids": list(view["member_ids"]),
            })
            pseudo.append(base)
        from . import _sparse as _build_sparse   # 延迟导入，避免循环依赖
        adapter = _PseudoStore(self.store, pseudo)
        retriever = _build_sparse(self.kind, adapter, self.min_score,
                                  self.use_relative_time)
        result = {"version": self.store.version[user_id], "views": views,
                  "pseudo": pseudo, "retriever": retriever, "meta": meta}
        self.stats["views"] = len(views)
        self.stats["users"] += 1
        with self.index_lock:
            self.index_cache[user_id] = result
            self.index_cache.move_to_end(user_id)
            while len(self.index_cache) > self.max_users:
                self.index_cache.popitem(last=False)
        return result

    def score_memories(self, user_id, query=None, query_vector=None):
        """返回 {memory_id: 视图分}（成员取最大视图分）；无视图时返回 {}。"""
        if not query:
            return {}
        idx = self._pseudo(user_id)
        if not idx["views"]:
            return {}
        try:
            hits = idx["retriever"].search(user_id, query, len(idx["views"]))
        except Exception:
            logger.exception("[views] lexical view search failed")
            return {}
        by_view = {}
        for memory, score in hits:
            by_view[memory["id"]] = max(by_view.get(memory["id"], -np.inf), float(score))
        out = {}
        for view in idx["views"]:
            s = by_view.get(view["view_id"])
            if s is None:
                continue
            for mid in view["member_ids"]:
                if s > out.get(mid, -np.inf):
                    out[mid] = s
        return out

    def describe(self):
        return {"class": self.__class__.__name__, "mode": "lexical", "kind": self.kind,
                "window": {"size": self.window_size, "overlap": self.window_overlap,
                           "enabled": self.enable_window},
                "segment": {"max_messages": self.segment_max_messages,
                            "max_gap_seconds": self.segment_max_gap_seconds,
                            "enabled": self.enable_segment},
                "stats": dict(self.stats)}


class _PseudoStore:
    """给词法检索器用的最小 store：version + list_memories。"""

    def __init__(self, inner, pseudo):
        self._inner = inner
        self._pseudo = pseudo
        self.version = _ConstVersion(inner)

    def list_memories(self, user_id):
        return self._pseudo


class _ConstVersion(dict):
    def __init__(self, inner):
        super().__init__()
        self._inner = inner

    def __getitem__(self, user_id):
        return (self._inner.version[user_id],)
