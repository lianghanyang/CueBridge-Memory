import logging
from datetime import datetime

import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def day_ordinal(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").toordinal()
    except (TypeError, ValueError):
        return -1


class BM25Retriever:
    def __init__(self, store, min_score=0.0, k1=1.2, b=0.75):
        self.store = store
        self.min_score = min_score
        self.k1 = k1
        self.b = b
        self.cache = {}
        self._stats = {"build_total": 0, "build_by_version": 0, "hit": 0, "skipped_by_debounce": 0}

    #       而 pickle 反序列化只要 **0.74s**（实测）⇒ 值得落盘。
    # 只存 vectorizer + 稀疏矩阵（不含 memories / lowercase / dates，那些从库里重建更省）。
    # 失效判定：fingerprint = (记录数, max(created_at), max(ts), 最后一条 id) 的 sha1。
    #   add-only 场景下它单调变化 ⇒ 语料一变就失效，不会读到旧索引。
    _LIBIDX_DIR = None

    def _libidx_path(self, user_id):
        import os as _os, hashlib as _h
        if BM25Retriever._LIBIDX_DIR is None:
            BM25Retriever._LIBIDX_DIR = _os.getenv("AMC_LIBIDX_DIR", "data/store/libidx")
        _os.makedirs(BM25Retriever._LIBIDX_DIR, exist_ok=True)
        h = _h.sha1(str(user_id).encode("utf-8")).hexdigest()[:16]
        return _os.path.join(BM25Retriever._LIBIDX_DIR, h + ".pkl")

    def _fingerprint(self, user_id):
        import hashlib as _h
        try:
            row = self.store.conn.execute(
                "SELECT COUNT(*), MAX(created_at), MAX(ts), MAX(id) FROM memories WHERE user_id=?",
                (user_id,)).fetchone()
            raw = "|".join(str(x) for x in (row or ()))
        except Exception:
            return None
        return _h.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _try_load_from_disk(self, user_id, fp):
        import os as _os, pickle as _pk, time as _t
        if (_os.getenv("AMC_LIBIDX_DISK", "1") or "1").strip() not in ("1", "on", "true", "yes"):
            return None
        path = self._libidx_path(user_id)
        if not _os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                blob = _pk.load(fh)
            if blob.get("fp") != fp:
                return None
            memories = self.store.list_memories(user_id)
            if not memories or len(memories) != blob["matrix"].shape[0]:
                return None
            texts = [memory["index_text"] or "" for memory in memories]
            dates = [memory["created_at"][:10] for memory in memories]
            self._stats["disk_hit"] = self._stats.get("disk_hit", 0) + 1
            return {
                "version": self.store.version[user_id],
                "memories": memories,
                "vectorizer": blob["vectorizer"],
                "matrix": blob["matrix"],
                "dates": dates,
                "days": np.array([day_ordinal(v) for v in dates]),
                "lowercase": [t.lower() for t in texts],
                "cjk_ratio": blob.get("cjk_ratio", 0.0),
                "built_at": _t.time(),
                "_from_disk": True,
            }
        except Exception:
            return None

    def _save_to_disk(self, user_id, fp, result):
        import os as _os, pickle as _pk, tempfile as _tf
        if (_os.getenv("AMC_LIBIDX_DISK", "1") or "1").strip() not in ("1", "on", "true", "yes"):
            return
        try:
            n = len(result.get("memories") or [])
            if n < int(_os.getenv("AMC_LIBIDX_MIN_ROUNDS", "500") or 500):
                return
            if result.get("matrix") is None or result.get("vectorizer") is None:
                return
            # 磁盘配额：超过 AMC_LIBIDX_MAX_GB 就不再写（避免无限增长）
            try:
                import glob as _g
                _mx = float(_os.getenv("AMC_LIBIDX_MAX_GB", "20") or 20)
                _dir = _os.path.dirname(self._libidx_path(user_id))
                _tot = sum(_os.path.getsize(f) for f in _g.glob(_os.path.join(_dir, "*.pkl")))
                if _tot > _mx * 1024 ** 3:
                    self._stats["disk_quota_skip"] = self._stats.get("disk_quota_skip", 0) + 1
                    return
            except Exception:
                pass
            path = self._libidx_path(user_id)
            blob = {"fp": fp, "vectorizer": result["vectorizer"],
                    "matrix": result["matrix"], "cjk_ratio": result.get("cjk_ratio", 0.0)}
            d = _os.path.dirname(path)
            fd, tmp = _tf.mkstemp(dir=d)
            with _os.fdopen(fd, "wb") as fh:
                _pk.dump(blob, fh, protocol=4)
            _os.replace(tmp, path)
            self._stats["disk_save"] = self._stats.get("disk_save", 0) + 1
        except Exception:
            pass

    def _index(self, user_id):
        version = self.store.version[user_id]
        cached = self.cache.get(user_id)
        if cached and cached["version"] == version:
            self._stats["hit"] = self._stats.get("hit", 0) + 1
            return cached
        # 内存未命中 ⇒ 先试磁盘（同一 fingerprint 才用）
        _fp = self._fingerprint(user_id)
        if _fp:
            _loaded = self._try_load_from_disk(user_id, _fp)
            if _loaded is not None:
                self.cache[user_id] = _loaded
                return _loaded
        # ---- debounce（⚠️ 已验证无用，默认关闭；见 data/reports/library/OPERATIONS.md）----
        # 实测平台是"灌完再查"（add 与 search 不重叠），所以重建 churn 不存在。
        # 这个开关只在"边灌边查"的假设下有意义，而那个假设后来被证明是我自己
        # 的 clean_eval 重放 add 造成的假象。保留代码仅作保险，**不要默认开启**。
        # 而重建成本随记录数线性增长（1 万条 ≈ 4s）。边灌边查时会反复重建 → 长尾。
        # 用旧索引的代价是"最近 N 秒的新记录暂时查不到"，用
        # AMC_INDEX_DEBOUNCE_MAX_DELTA 给漏掉的数量设上界。
        import os as _os, time as _t
        if cached:
            _db = float(_os.getenv("AMC_INDEX_DEBOUNCE_SEC", "0") or "0")
            _maxd = int(_os.getenv("AMC_INDEX_DEBOUNCE_MAX_DELTA", "50") or "50")
            if _db > 0:
                _age = _t.time() - float(cached.get("built_at") or 0)
                _cur = self.store.count_memories(user_id)
                _delta = _cur - len(cached.get("memories") or [])
                if _age < _db and 0 <= _delta <= _maxd:
                    self._stats["skipped_by_debounce"] = self._stats.get("skipped_by_debounce", 0) + 1
                    return cached

        import time as _t2
        memories = self.store.list_memories(user_id)
        self._stats["build_total"] = self._stats.get("build_total", 0) + 1
        if cached:
            self._stats["build_by_version"] = self._stats.get("build_by_version", 0) + 1
        result = {
            "version": version,
            "memories": memories,
            "vectorizer": None,
            "built_at": _t2.time(),
        }

        if not memories:
            self.cache[user_id] = result
            return result

        texts = [memory["index_text"] or "" for memory in memories]
        # ⚠️ 语言自适应分词（2026-10-31）：原来的 `\b\w\w+\b` 对中文**完全不切词**
        # —— CJK 没有空格，`\b` 只在标点处断，一整句（逗号分隔的一个分句）会变成**一个 token**。
        # 实测两句话只切出 4 个 token 且零重叠 ⇒ 中文查询的 BM25 分数恒为 0（等于没有词法检索）。
        # 平台的真实查询是中文第一人称问句（"我这次给孩子买秋冬衣服时，先做了哪两步准备？"），
        _cjk = sum(1 for t in texts[:200] for ch in t[:200] if "\u4e00" <= ch <= "\u9fff")
        _all = sum(len(t[:200]) for t in texts[:200]) or 1
        _cjk_ratio = _cjk / _all
        result["cjk_ratio"] = _cjk_ratio
        if _cjk_ratio > 0.2:
            vectorizer = CountVectorizer(
                lowercase=True, analyzer="char", ngram_range=(2, 3), dtype=np.float32,
            )
            logger.info("[bm25] 用户 %s CJK 占比 %.0f%% → 字符 n-gram 分词", user_id[-12:], _cjk_ratio * 100)
        else:
            vectorizer = CountVectorizer(
                lowercase=True,
                stop_words="english",
                token_pattern=r"(?u)\b\w\w+\b",
                dtype=np.float32,
            )

        try:
            counts = vectorizer.fit_transform(texts).tocoo()
        except ValueError:
            self.cache[user_id] = result
            return result

        document_count = counts.shape[0]
        lengths = np.asarray(counts.tocsr().sum(axis=1)).ravel()
        average_length = max(float(lengths.mean()), 1e-6)
        document_frequency = np.bincount(
            counts.col,
            minlength=counts.shape[1],
        )
        idf = np.log1p(
            (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)

        weighted = (
            idf[counts.col]
            * counts.data
            * (self.k1 + 1)
            / (
                counts.data
                + self.k1
                * (
                    1
                    - self.b
                    + self.b * lengths[counts.row] / average_length
                )
            )
        )

        matrix = sparse.csr_matrix(
            (weighted, (counts.row, counts.col)),
            shape=counts.shape,
        )

        dates = [memory["created_at"][:10] for memory in memories]
        result.update(
            vectorizer=vectorizer,
            matrix=matrix,
            dates=dates,
            days=np.array([day_ordinal(value) for value in dates]),
            lowercase=[text.lower() for text in texts],
        )
        try:
            self._save_to_disk(user_id, _fp, result)
        except Exception:
            pass
        self.cache[user_id] = result
        return result

    def score_all(self, user_id, query):
        index = self._index(user_id)
        if index["vectorizer"] is None or not query.strip():
            return index, np.zeros(len(index["memories"]), dtype=np.float32)

        query_vector = index["vectorizer"].transform([query])
        query_vector.data[:] = 1.0
        scores = (index["matrix"] @ query_vector.T).toarray().ravel()
        return index, scores

    def search(self, user_id, query, limit):
        index, scores = self.score_all(user_id, query)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            (index["memories"][position], float(scores[position]))
            for position in order
            if scores[position] > self.min_score
        ]
