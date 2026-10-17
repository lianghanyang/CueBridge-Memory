"""Reusable cross-encoder score cache: (query, memory_id) -> rerank score.

Why this exists
---------------
Reranking is *pointwise*: the score of a candidate does not depend on the rest of
the pool, and it is a function of (query, document) only.  So a single GPU pass
over the candidate union can be persisted as a mapping table, after which **any**
change to the preprocessing stage (fusion mode, emission policy, pool breadth,
new channel, ...) can be evaluated end-to-end *without touching the GPU*:
re-run the retriever on CPU, look the pool up in this table, and replay the
rerank stage.

The cache is incremental and content-addressed, so extending coverage (a bigger
pool, a new model, more samples) only pays for the pairs that are actually new.
Keying by query text means repeated/cross-suite queries are free too.

Note on production value: production queries are unseen, so this is primarily an
experiment accelerator.  It does help in production for repeated queries and for
the second `/search` of the same question (e.g. answer-stage re-query).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class RerankScoreCache:
    def __init__(self, path, namespace):
        self.namespace = namespace
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS rerank_scores("
                "ns TEXT NOT NULL, qkey TEXT NOT NULL, mem_id TEXT NOT NULL, "
                "score REAL NOT NULL, PRIMARY KEY(ns, qkey, mem_id))"
            )

    def qkey(self, query):
        return hashlib.sha1(f"{self.namespace}||{query}".encode("utf-8")).hexdigest()

    def get_many(self, query, mem_ids):
        key = self.qkey(query)
        found = {}
        ids = list(mem_ids)
        with self.lock:
            for start in range(0, len(ids), 400):
                chunk = ids[start:start + 400]
                marks = ",".join("?" * len(chunk))
                rows = self.conn.execute(
                    f"SELECT mem_id, score FROM rerank_scores WHERE ns=? AND qkey=? "
                    f"AND mem_id IN ({marks})",
                    (self.namespace, key, *chunk),
                ).fetchall()
                for mem_id, score in rows:
                    found[mem_id] = float(score)
        return found

    def put_many(self, query, pairs):
        if not pairs:
            return
        key = self.qkey(query)
        with self.lock, self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO rerank_scores(ns, qkey, mem_id, score) "
                "VALUES (?,?,?,?)",
                [(self.namespace, key, mem_id, float(score)) for mem_id, score in pairs],
            )

    def fill(self, query, candidates, reranker, batch_doc_chars=1500):
        """Score only the missing (query, memory) pairs; returns {mem_id: score}.

        ``candidates`` is the retriever's ``[(memory, base_score), ...]``.
        Candidates are submitted longest-document-last so that the padding waste
        inside each batch stays small.
        """
        from .reranker import doc_text

        mem_ids = [m.get("id") for m, _ in candidates if m.get("id")]
        found = self.get_many(query, mem_ids)
        self.hits += len(found)
        missing = [(m, s) for m, s in candidates
                   if m.get("id") and m["id"] not in found]
        if not missing:
            return found
        self.misses += len(missing)
        missing.sort(key=lambda item: len(doc_text(item[0], batch_doc_chars)))
        scored = reranker.rerank(query, missing, top_k=len(missing))
        fresh = [(m["id"], float(s)) for m, s in scored if m.get("id")]
        self.put_many(query, fresh)
        found.update(dict(fresh))
        return found

    def stats(self):
        total = self.conn.execute(
            "SELECT COUNT(*) FROM rerank_scores WHERE ns=?", (self.namespace,)
        ).fetchone()[0]
        return {"namespace": self.namespace, "rows": int(total),
                "session_hits": self.hits, "session_misses": self.misses}

    def describe(self):
        return {"class": self.__class__.__name__, **self.stats()}
