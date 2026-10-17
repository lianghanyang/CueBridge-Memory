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
        if memory.get("n_images"):
            text = f"{text}\n[contains {memory['n_images']} image(s)]"
        return text[:min(max_chars, 10000)] if text else "[empty memory]"

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
                return cached

        memories = self.store.list_memories(user_id)
        result = {"version": version, "memories": memories, "matrix": None}
        if memories:
            texts = [self._doc_text(memory, self.max_chars) for memory in memories]
            keys, found = self._ensure(texts, reason="index")
            zero = np.zeros(self.provider.dim, dtype=np.float32)
            result["matrix"] = np.vstack(
                [found.get(key, zero) for key in keys]
            ).astype(np.float32, copy=False)

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
        vector = np.asarray(
            self.provider.encode_query(query[:self.max_chars]), dtype=np.float32
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
