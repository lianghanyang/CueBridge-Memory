"""Dense retriever: persistent text-hash cache, background encode-on-add,
LRU per-user index, NaN-safe scoring."""
import json
import logging
import queue
import threading
from collections import OrderedDict

import numpy as np

from .providers import VectorCache, build_provider, finite_rows

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, store, provider, cache_path=None, min_score=-1.0,
                 max_chars=4000, embed_on_add=True, max_users=32, warm_batch=64):
        self.store = store
        self.provider = provider
        self.min_score = min_score
        self.max_chars = max_chars
        self.max_users = max(1, max_users)
        self.warm_batch = max(1, warm_batch)
        self.cache = VectorCache(cache_path, namespace=provider.name, dim=provider.dim)
        self.index_cache = OrderedDict()
        self._idx_stats = {"build_total": 0, "build_by_version": 0, "hit": 0, "skipped_by_debounce": 0}
        self.index_lock = threading.RLock()
        self.stats = {"encoded": 0, "rejected": 0, "warm_batches": 0}
        self._query_memo = OrderedDict()
        self._query_lock = threading.RLock()
        self.query_memo_size = 32
        self.queue = None
        if embed_on_add:
            self.queue = queue.Queue()
            threading.Thread(target=self._warm_loop, name="dense-warm", daemon=True).start()

    # ---------- text ----------
    @staticmethod
    def _doc_text(memory, max_chars):
        text = memory.get("index_text") or ""
        if not text:
            text = "\n".join(
                part.get("text", "")
                for part in memory.get("content", [])
                if isinstance(part, dict) and part.get("type") == "text"
            )
        # ⚠️ caption 必须在**开头**：embedding 侧会按 max_chars 截断，
        # 而 caption 原本是追加在文本尾部的 —— 长记录里它会被截掉，
        # 导致"向量里看不见图"，而这类记录恰好是图像题（TTL）的全部证据。
        # 实测（ATM）：caption 追加位置中位 1331 字，P90 更高；截断 4000 时相当一部分被砍。
        # 这里在**构索引时**把 [Image ...] 段提到最前，不需要改存储格式，对所有历史记录同样生效。
        if memory.get("n_images"):
            import re as _re
            # **只在会截断时才重排** —— 否则所有含图记录的 key 都会变，
            # 导致全量重新 embedding（Gem 的 caption 实验也会被污染）。
            # 判据：caption 段的结束位置是否超过 max_chars。
            _caps = _re.findall(r"\[Image [^\]]+\][^\n]*", text)
            if _caps:
                _last_end = max(text.rfind(_c) + len(_c) for _c in _caps)
                if _last_end > min(max_chars, 10000):
                    _rest = text
                    for _c in _caps:
                        _rest = _rest.replace(_c, "", 1)
                    text = "\n".join(_caps) + "\n" + _rest.strip()
            text = f"{text}\n[contains {memory['n_images']} image(s)]"
        # 截断策略：
        #   max_chars > 0 → 按字符截断（旧行为）
        #   max_chars == 0 → **不截断**（只保留一个防病态输入的硬上限）
        # 为什么允许不截断：ATM 的记录最长 10,483 字符，而 text-embedding-v4 上限 8192 token
        # （中文≈1字/token，英文≈0.25 token/字）⇒ 10k 字符远未超限。
        # 而截断会让尾部信息（尤其是追加的图 caption）丢失，实测 ATM 有 26% 受影响。
        if not text:
            return "[empty memory]"
        _cap = int(max_chars) if max_chars and int(max_chars) > 0 else 32000
        return text[:_cap]

    # ---------- encode with cache ----------
    def _ensure(self, texts, reason="index"):
        keys = [self.cache.key(text) for text in texts]
        found, missing = self.cache.get_many(keys)
        if missing:
            wanted = set(missing)
            todo, seen = [], set()
            for key, text in zip(keys, texts):
                if key in wanted and key not in seen:
                    seen.add(key)
                    todo.append((key, text))
            level = logging.INFO if reason == "index" else logging.DEBUG
            logger.log(level, "[dense] encoding %d new / %d texts (%s)",
                       len(todo), len(texts), reason)
            vectors = self.provider.encode_documents([text for _, text in todo])
            ok = finite_rows(vectors)
            fresh = [(key, vectors[i]) for i, (key, _) in enumerate(todo) if ok[i]]
            rejected = len(todo) - len(fresh)
            if rejected:
                self.stats["rejected"] += rejected
                logger.warning("[dense] %d vectors rejected (non-finite)", rejected)
            self.stats["encoded"] += len(fresh)
            self.cache.put_many(fresh)
            found.update(fresh)
        return keys, found

    # ---------- background warm-up on /add ----------
    def notify_add(self, user_id, request_id):
        if self.queue is not None:
            self.queue.put((user_id, request_id))

    def _request_texts(self, user_id, request_id):
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT index_text, n_images, content_json FROM memories "
                "WHERE user_id=? AND request_id=? ORDER BY msg_index",
                (user_id, request_id),
            ).fetchall()
        return [
            self._doc_text(
                {
                    "index_text": row["index_text"],
                    "n_images": row["n_images"],
                    "content": json.loads(row["content_json"]),
                },
                self.max_chars,
            )
            for row in rows
        ]

    def _warm_loop(self):
        while True:
            batch = [self.queue.get()]
            while len(batch) < self.warm_batch:
                try:
                    batch.append(self.queue.get(timeout=0.02))
                except queue.Empty:
                    break
            try:
                texts = []
                for user_id, request_id in batch:
                    texts.extend(self._request_texts(user_id, request_id))
                if texts:
                    self._ensure(texts, reason="warm")
                    self.stats["warm_batches"] += 1
            except Exception:
                logger.exception("[dense] warm-up failed")

    # ---------- index ----------
    def _index(self, user_id):
        version = self.store.version[user_id]
        with self.index_lock:
            cached = self.index_cache.get(user_id)
            if cached and cached["version"] == version:
                self.index_cache.move_to_end(user_id)
                self._idx_stats["hit"] = self._idx_stats.get("hit", 0) + 1
                return cached

        # ---- debounce（⚠️ 已验证无用，默认关闭；见 data/reports/library/OPERATIONS.md）----
        # 稠密索引的重建成本远高于 BM25（要重新算 embedding），所以这里收益更大。
        import os as _os, time as _t
        if cached:
            _db = float(_os.getenv("AMC_INDEX_DEBOUNCE_SEC", "0") or "0")
            _maxd = int(_os.getenv("AMC_INDEX_DEBOUNCE_MAX_DELTA", "50") or "50")
            if _db > 0:
                _age = _t.time() - float(cached.get("built_at") or 0)
                _cur = self.store.count_memories(user_id)
                _delta = _cur - len(cached.get("memories") or [])
                if _age < _db and 0 <= _delta <= _maxd:
                    self._idx_stats["skipped_by_debounce"] = self._idx_stats.get("skipped_by_debounce", 0) + 1
                    return cached

        memories = self.store.list_memories(user_id)
        self._idx_stats["build_total"] = self._idx_stats.get("build_total", 0) + 1
        if cached:
            self._idx_stats["build_by_version"] = self._idx_stats.get("build_by_version", 0) + 1
        result = {"version": version, "memories": memories, "matrix": None, "built_at": _t.time()}
        if memories:
            texts = [self._doc_text(memory, self.max_chars) for memory in memories]
            keys, found = self._ensure(texts, reason="index")
            zero = np.zeros(self.provider.dim, dtype=np.float32)
            result["matrix"] = np.vstack(
                [found.get(key, zero) for key in keys]
            ).astype(np.float32, copy=False)
            # ⚠️ 缺失 embedding 的记录被补成**零向量**：它们对稠密检索永久不可见
            # （score 恒为 0，被 min_score 过滤），但不会报错 —— 典型的静默召回损失。
            # 这里把"有多少行是零向量"记下来并暴露到 /health，
            # 让"零向量"这种隐形降级可被发现（对应 OPERATIONS.md 的可观测性清单）。
            _missing = sum(1 for k in keys if k not in found)
            result["zero_rows"] = int(_missing)
            self._idx_stats["zero_rows"] = int(_missing)
            if _missing:
                logger.warning("[dense] 用户 %s 缺 embedding %d/%d 条（零向量，对 dense 不可见）",
                               user_id, _missing, len(keys))
                self._idx_stats.setdefault("zero_by_user", {})[user_id] = int(_missing)

        with self.index_lock:
            self.index_cache[user_id] = result
            self.index_cache.move_to_end(user_id)
            while len(self.index_cache) > self.max_users:
                self.index_cache.popitem(last=False)
        return result

    def query_vector(self, query):
        """Encode (and memoize) a query vector. Returns None if non-finite.

        Memoizing means a single /search only pays one embedding API call even
        when several granularity channels need the same query vector.
        """
        query = (query or "").strip()
        if not query:
            return None
        with self._query_lock:
            cached = self._query_memo.get(query)
            if cached is not None:
                self._query_memo.move_to_end(query)
                return cached
        # ⚠️ `max_chars == 0` 表示"不截断"（见 _doc_text）。这里必须同样处理：
        # 否则 `query[:0]` 会得到**空串** → 查询向量变成空串的嵌入 → dense 全乱。
        # （2026-10-31 自查发现：我把 AMC_EMBEDDING_MAX_CHARS 设成 0 后，
        #   离线端到端测试的 rank@1 从 10/10 掉到 1/10，就是这个原因。）
        _qm = int(self.max_chars) if self.max_chars and int(self.max_chars) > 0 else 32000
        vector = np.asarray(
            self.provider.encode_query(query[:_qm]), dtype=np.float32
        ).ravel()
        if not finite_rows(vector)[0]:
            logger.warning("[dense] query vector non-finite, dense leg skipped")
            return None
        with self._query_lock:
            self._query_memo[query] = vector
            self._query_memo.move_to_end(query)
            while len(self._query_memo) > self.query_memo_size:
                self._query_memo.popitem(last=False)
        return vector

    def search(self, user_id, query, limit):
        query = (query or "").strip()
        if not query:
            return []
        index = self._index(user_id)
        if index["matrix"] is None:
            return []
        vector = self.query_vector(query)
        if vector is None:
            return []
        scores = index["matrix"] @ vector
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            (index["memories"][position], float(scores[position]))
            for position in order
            if scores[position] > self.min_score
        ]

    def describe(self):
        info = dict(self.provider.describe())
        info.update(
            cached_users=len(self.index_cache),
            pending_warm=self.queue.qsize() if self.queue is not None else 0,
            cache_rejected=self.cache.rejected,
            **self.stats,
        )
        return info


def build_dense(store, settings):
    return DenseRetriever(
        store,
        build_provider(settings),
        cache_path=settings.embedding_cache_path,
        min_score=-1.0,
        max_chars=settings.embedding_max_chars,
        embed_on_add=settings.embed_on_add,
        max_users=settings.index_cache_users,
    )
