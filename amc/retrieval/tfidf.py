import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfRetriever:
    def __init__(self, store, min_score=0.0):
        self.store = store
        self.min_score = min_score
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
            "matrix": None,
        }

        if memories:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
            )
            try:
                matrix = vectorizer.fit_transform(
                    [memory["index_text"] or "" for memory in memories]
                )
                result.update(vectorizer=vectorizer, matrix=matrix)
            except ValueError:
                pass

        self.cache[user_id] = result
        return result

    def search(self, user_id, query, limit):
        index = self._index(user_id)
        if index["vectorizer"] is None or not query.strip():
            return []

        scores = linear_kernel(
            index["matrix"],
            index["vectorizer"].transform([query]),
        ).ravel()

        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            (index["memories"][position], float(scores[position]))
            for position in order
            if scores[position] > self.min_score
        ]
