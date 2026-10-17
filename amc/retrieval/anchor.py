import math
import re
from datetime import date

import numpy as np

MONTHS = {
    "january": "01", "jan": "01", "february": "02", "feb": "02",
    "march": "03", "mar": "03", "april": "04", "apr": "04",
    "may": "05", "june": "06", "jun": "06", "july": "07",
    "jul": "07", "august": "08", "aug": "08", "september": "09",
    "sept": "09", "sep": "09", "october": "10", "oct": "10",
    "november": "11", "nov": "11", "december": "12", "dec": "12",
}
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
YEAR_RE = re.compile(r"\b20[12]\d\b")
STOP = {
    "what", "when", "where", "which", "who", "why", "how", "show",
    "find", "tell", "list", "give", "the", "this", "that", "my",
    "with", "from", "during", "about", "please",
}


def explicit_dates(query):
    result = set(re.findall(r"\b20\d{2}-\d{2}(?:-\d{2})?\b", query))
    pattern = (
        r"\b(" + MONTH_ALT + r")\.?\s+"
        r"(?:(\d{1,2})(?:st|nd|rd|th)?,?\s+)?(20\d{2})\b"
    )
    for match in re.finditer(pattern, query, re.I):
        month = MONTHS[match.group(1).lower()]
        if match.group(2):
            result.add(
                f"{match.group(3)}-{month}-{int(match.group(2)):02d}"
            )
        else:
            result.add(f"{match.group(3)}-{month}")
    return result


def anchor_terms(query):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'.-]*|\b20[12]\d\b", query)
    result = []
    for index, token in enumerate(tokens):
        lower = token.lower().strip(".'")
        if lower in STOP:
            continue
        if YEAR_RE.fullmatch(token):
            result.append(token)
        elif token[0].isupper() and (index > 0 or token.isupper()):
            result.append(lower)
    return list(dict.fromkeys(result))


def densest_cluster(days, scores, gap_days):
    order = np.argsort(days)
    days = days[order]
    scores = scores[order]
    clusters = []
    start = 0

    for index in range(1, len(days) + 1):
        if index == len(days) or days[index] - days[index - 1] > gap_days:
            clusters.append((start, index))
            start = index

    left, right = max(
        clusters,
        key=lambda pair: (
            pair[1] - pair[0],
            float(scores[pair[0]:pair[1]].sum()),
        ),
    )
    return int(days[left]), int(days[right - 1])


class AnchorTimeRetriever:
    def __init__(
        self,
        base,
        min_score=0.0,
        seed_k=30,
        gap_days=4,
        pad_days=2,
        boost=0.35,
        explicit_boost=0.35,
        use_relative_time=False,
    ):
        self.base = base
        self.min_score = min_score
        self.seed_k = seed_k
        self.gap_days = gap_days
        self.pad_days = pad_days
        self.boost = boost
        self.explicit_boost = explicit_boost
        self.use_relative_time = use_relative_time
        self.last = {}

    def search(self, user_id, query, limit):
        index, scores = self.base.score_all(user_id, query)
        if (
            index.get("vectorizer") is None
            or not len(scores)
            or float(scores.max()) <= 0
        ):
            self.last = {}
            return []

        top_score = float(scores.max())
        bonus = np.zeros_like(scores)
        info = {}

        prefixes = explicit_dates(query)
        if prefixes:
            matched = np.fromiter(
                (
                    any(value.startswith(prefix) for prefix in prefixes)
                    for value in index["dates"]
                ),
                dtype=bool,
                count=len(scores),
            )
            bonus += self.explicit_boost * matched
            info["explicit_dates"] = sorted(prefixes)

        terms = anchor_terms(query)
        event_terms = [
            term for term in terms
            if not YEAR_RE.fullmatch(term) and term not in MONTHS
        ]

        if event_terms:
            required = (
                len(terms)
                if len(terms) <= 2
                else int(math.ceil(0.6 * len(terms)))
            )
            seeds = [
                position
                for position in np.argsort(-scores)[:self.seed_k]
                if scores[position] > 0
                and index["days"][position] >= 0
                and sum(
                    term in index["lowercase"][position]
                    for term in terms
                ) >= required
            ]

            if seeds:
                low, high = densest_cluster(
                    index["days"][seeds],
                    scores[seeds],
                    self.gap_days,
                )
                in_window = (
                    (index["days"] >= low - self.pad_days)
                    & (index["days"] <= high + self.pad_days)
                )
                bonus += self.boost * in_window
                info["window"] = [
                    str(date.fromordinal(low)),
                    str(date.fromordinal(high)),
                ]
                info["terms"] = terms

        adjusted = scores + top_score * bonus * (scores > 0)
        self.last = info

        order = np.argsort(-adjusted, kind="stable")[:limit]
        return [
            (index["memories"][position], float(adjusted[position]))
            for position in order
            if adjusted[position] > self.min_score
        ]
