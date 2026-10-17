from datetime import datetime

import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer


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

    def _index(self, user_id):
        version = self.store.version[user_id]
        cached = self.cache.get(user_id)
        if cached and cached["version"] == version:
            return cached

        memories = self.store.list_memories(user_id)
        result = {
            "version": version,
            "memories": memories,
            "vectorizer": None,
        }

        if not memories:
            self.cache[user_id] = result
            return result

        texts = [memory["index_text"] or "" for memory in memories]
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
