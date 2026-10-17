"""Pre-rerank pipeline v5 — paper-inspired stages.

Changes vs v4:
  - zscore_normalize is now ALWAYS-ON pre-stage (not a testable stage).
  - conflict_aware_rerank: sim threshold 0.4 -> 0.25; added antonym-pair detection.
  - query_decomposition: sub-query IDF coverage with MIN aggregation.
  - evidence_compression: evidentiality density (ECoRAG-inspired).
"""
from __future__ import annotations
import hashlib, json, math, re
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

Hit = Tuple[dict, float]
Stage = Tuple[str, Callable]

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+|\b\d{4}\b")
_CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NEG_RE = re.compile(
    r"\b(no|not|never|without|cannot|can't|doesn't|don't|didn't|"
    r"isn't|wasn't|aren't|won't|wouldn't|instead|but|however|"
    r"although|though|only|actually|rather)\b", re.I)

_STOP = {
    "the","a","an","and","or","but","if","then","is","are","was","were","be",
    "been","being","do","does","did","have","has","had","of","to","in","on",
    "at","for","with","by","from","as","that","this","these","those","it","its",
    "they","them","their","what","when","where","which","who","why","how","my",
    "your","our","his","her","there","here","still","just","also","very",
}

# Antonym pairs for value-substitution detection (SAIL-Embedding + Conflict-Aware Memory)
_ANTONYM_PAIRS = [
    ("giant","miniature"), ("big","small"), ("large","small"),
    ("high","low"), ("high-maintenance","low-maintenance"),
    ("before","after"), ("alone","together"), ("like","dislike"),
    ("love","hate"), ("enabled","disabled"), ("increase","decrease"),
    ("more","less"), ("always","never"), ("all","none"),
    ("possible","impossible"), ("able","unable"), ("open","closed"),
    ("first","last"), ("early","late"), ("fast","slow"),
]

def _tokens(text): return [t.lower() for t in _WORD_RE.findall(text or "")]
def _content_tokens(text):
    return [t for t in _tokens(text) if t not in _STOP and len(t) >= 3]

def _text_of(memory):
    chunks = []
    for part in memory.get("content", []) or []:
        if isinstance(part, dict) and part.get("type") == "text":
            v = str(part.get("text", "")).strip()
            if v: chunks.append(v)
    return "\n".join(chunks)

def _idf(term, df, n):
    return math.log((n + 1.0) / (df.get(term, 0) + 1.0)) + 1.0

def _score_span(hits):
    scores = [s for _, s in hits]
    return max(scores) - min(scores) if len(scores) > 1 else 1e-6

# ---------------------------------------------------------------------------
# ALWAYS-ON pre-stage: z-score normalization
# ---------------------------------------------------------------------------
def zscore_normalize(hits):
    """Always-on: standardize scores to z-scores."""
    if not hits:
        return hits
    scores = np.array([s for _, s in hits], dtype=np.float64)
    mu, sigma = scores.mean(), scores.std()
    if sigma < 1e-9:
        return [(m, 0.0) for m, _ in hits]
    return [(m, float((s - mu) / sigma)) for m, s in hits]

# ---------------------------------------------------------------------------
# Stage 1: query_decomposition (ACL 2025 SRW)
# ---------------------------------------------------------------------------
def _decompose_query(query):
    q = query.strip()
    parts = re.split(r"\s*(?:;|\?|\.\s+(?=[A-Z]))\s*", q)
    parts = [p.strip() for p in parts if len(p.strip()) > 5]
    if len(parts) <= 1:
        parts = re.split(r"\s+(?:and|or)\s+", q)
        parts = [p.strip() for p in parts if len(p.strip()) > 5]
    return parts[:3] if len(parts) > 1 else [q]

def stage_query_decomposition(user_id, query, hits, ctx):
    if not hits: return hits
    subqs = _decompose_query(query)
    if len(subqs) <= 1: return hits
    span = _score_span(hits); boost = 0.15 * span
    docs = [_content_tokens(_text_of(m)) for m, _ in hits]
    n = max(1, len(docs))
    df = defaultdict(int)
    for ts in docs:
        for t in set(ts): df[t] += 1
    out = []
    for (memory, score), tokens in zip(hits, docs):
        ts_set = set(tokens); coverages = []
        for subq in subqs:
            sq = set(_content_tokens(subq))
            if not sq: continue
            total = sum(_idf(t, df, n) for t in sq) or 1.0
            cov = sum(_idf(t, df, n) for t in sq if t in ts_set) / total
            coverages.append(min(1.0, max(0.0, cov)))
        min_cov = min(coverages) if coverages else 0.0
        out.append((memory, score + boost * min_cov))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------------------------------------------------------------------------
# Stage 2: conflict_aware_rerank (IEEE 2025 + SAIL-Embedding)
# ---------------------------------------------------------------------------
def _token_set(text): return set(_content_tokens(text))
def _jaccard(a, b):
    if not a or not b: return 0.0
    u = len(a | b); return len(a & b) / u if u else 0.0
def _has_negation(text): return bool(_NEG_RE.search(text or ""))

def _has_antonym_conflict(t1, t2):
    """Check if two texts contain antonym pairs."""
    t1_l = t1.lower(); t2_l = t2.lower()
    for a, b in _ANTONYM_PAIRS:
        if (a in t1_l and b in t2_l) or (b in t1_l and a in t2_l):
            return True
    return False

def stage_conflict_aware_rerank(user_id, query, hits, ctx):
    if len(hits) < 2: return hits
    span = _score_span(hits); penalty = 0.5 * span
    texts = [_text_of(m) for m, _ in hits]
    tokens = [_token_set(t) for t in texts]
    negs = [_has_negation(t) for t in texts]
    scores = [s for _, s in hits]
    demoted = set()
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            if i in demoted or j in demoted: continue
            sim = _jaccard(tokens[i], tokens[j])
            # Lower threshold (0.25) + antonym detection
            conflict = False
            if sim >= 0.25 and negs[i] != negs[j]:
                conflict = True
            elif _has_antonym_conflict(texts[i], texts[j]):
                conflict = True
            if conflict:
                if scores[i] < scores[j]: demoted.add(i)
                else: demoted.add(j)
    if not demoted: return hits
    out = []
    for idx, (memory, score) in enumerate(hits):
        out.append((memory, score - penalty if idx in demoted else score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------------------------------------------------------------------------
# Stage 3: query_coverage (IDF-weighted, scale-aware)
# ---------------------------------------------------------------------------
def stage_query_coverage(user_id, query, hits, ctx):
    if not hits: return hits
    q_tokens = set(_content_tokens(query))
    if not q_tokens: return hits
    docs = [_content_tokens(_text_of(m)) for m, _ in hits]
    n = max(1, len(docs)); df = defaultdict(int)
    for ts in docs:
        for t in set(ts): df[t] += 1
    total_idf = sum(_idf(t, df, n) for t in q_tokens) or 1.0
    span = _score_span(hits); boost = 0.20 * span
    out = []
    for (memory, score), tokens in zip(hits, docs):
        ts_set = set(tokens)
        cov = sum(_idf(t, df, n) for t in q_tokens if t in ts_set) / total_idf
        out.append((memory, score + boost * min(1.0, max(0.0, cov))))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------------------------------------------------------------------------
# Stage 4: evidence_compression (ECoRAG ACL 2025 Findings)
# ---------------------------------------------------------------------------
def stage_evidence_compression(user_id, query, hits, ctx):
    if not hits: return hits
    q_tokens = set(_content_tokens(query))
    if not q_tokens: return hits
    span = _score_span(hits); boost = 0.12 * span
    out = []
    for memory, score in hits:
        text = _text_of(memory); tokens = _content_tokens(text)
        if not tokens: out.append((memory, score)); continue
        token_set = set(tokens)
        hits_count = sum(1 for t in tokens if t in q_tokens)
        density = hits_count / max(len(tokens), 1)
        length_penalty = 1.0 if len(tokens) < 200 else max(0.5, 200 / len(tokens))
        out.append((memory, score + boost * density * length_penalty))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------------------------------------------------------------------------
# Stage 5: entity_overlap
# ---------------------------------------------------------------------------
def _entities(text): return {t.lower() for t in _CAP_RE.findall(text or "")}

def stage_entity_overlap(user_id, query, hits, ctx):
    if not hits: return hits
    q_ents = _entities(query)
    if not q_ents: return hits
    span = _score_span(hits); boost = 0.12 * span
    out = []
    for memory, score in hits:
        ov = len(q_ents & _entities(_text_of(memory)))
        out.append((memory, score + (boost * min(ov, 3) if ov else 0.0)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

# ---------------------------------------------------------------------------
# Stage 6: session_expand (conservative)
# ---------------------------------------------------------------------------
def _session_neighbors(user_id, memory, ctx):
    store = ctx.get("store")
    if store is None: return []
    sid = memory.get("session_id"); midx = memory.get("msg_index")
    if sid is None or midx is None: return []
    try:
        with store.lock:
            rows = store.conn.execute(
                "SELECT id FROM memories WHERE user_id=? AND session_id=? "
                "AND msg_index BETWEEN ? AND ? AND id != ?",
                (user_id, sid, int(midx) - 2, int(midx) + 2, memory["id"]),
            ).fetchall()
    except Exception: return []
    return [{"id": row["id"]} for row in rows]

def stage_session_expand(user_id, query, hits, ctx):
    if not hits: return hits
    present_ids = {m["id"] for m, _ in hits}
    boosts = defaultdict(float)
    for memory, _ in hits:
        for nb in _session_neighbors(user_id, memory, ctx):
            nb_id = nb.get("id")
            if nb_id and nb_id in present_ids:
                boosts[nb_id] += 0.02; boosts[memory["id"]] += 0.02
    if not boosts: return hits
    span = _score_span(hits)
    out = [(m, s + span * boosts.get(m["id"], 0.0)) for m, s in hits]
    out.sort(key=lambda x: x[1], reverse=True)
    return out

ALL_STAGES = {
    "query_decomposition":   stage_query_decomposition,
    "conflict_aware_rerank": stage_conflict_aware_rerank,
    "query_coverage":        stage_query_coverage,
    "evidence_compression":  stage_evidence_compression,
    "entity_overlap":        stage_entity_overlap,
    "session_expand":        stage_session_expand,
}
DEFAULT_ORDER = list(ALL_STAGES.keys())

class PreRerankPipeline:
    def __init__(self, base, stage_names=None, store=None):
        self.base = base; self.store = store
        self.stage_names = [s for s in (stage_names or []) if s in ALL_STAGES]
        self.stages = [(n, ALL_STAGES[n]) for n in self.stage_names]

    def notify_add(self, user_id, request_id):
        hook = getattr(self.base, "notify_add", None)
        if hook is not None: hook(user_id, request_id)

    def search(self, user_id, query, limit):
        hits = self.base.search(user_id, query, limit)
        if not hits: return []
        hits = zscore_normalize(hits)  # ALWAYS-ON
        ctx = {"store": self.store, "user_id": user_id}
        for _n, fn in self.stages:
            hits = fn(user_id, query, hits, ctx)
        return hits[:limit]

    def describe(self):
        base_desc = {}
        fn = getattr(self.base, "describe", None)
        if callable(fn):
            try: base_desc = fn() or {}
            except Exception as exc: base_desc = {"describe_error": str(exc)}
        return {"class": self.__class__.__name__,
                "stages": list(self.stage_names), "base": base_desc,
                "always_on": ["zscore_normalize"]}
