"""Deterministic, read-only monetary evidence extraction.

This module deliberately returns evidence rows, never an aggregate or answer.
The caller may use the rows to choose evidence, but presentation remains
subject to the librarian's evidence-only contract.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

# 货币符号：$ ￥ ¥ + **€ £**（2026-10-30 补）。
# 依据：ATM 的真实 add 文本里有 **1,399 个 £ 金额、452 个 € 金额**，
# 而原来的字符类不含这两个符号 ⇒ **1,851 个金额完全识别不到**（金额题共 352 道）。
# 货币标记（预筛用，纯子串判断）
_CURRENCY_HINT = ("$", "£", "€", "¥", "￥", "元", "dollar", "buck", "euro", "pound", "usd")

_SYMBOL_AMOUNT = re.compile(r"(?P<currency>[$￥¥€£])\s*(?P<number>\d[\d,]*(?:\.\d{1,2})?)")
_WORD_AMOUNT = re.compile(
    r"(?P<number>\d[\d,]*(?:\.\d{1,2})?)\s*(?P<currency>dollars?|bucks|usd|euros?|pounds?|元)\b",
    re.I,
)
_NUM_WORD = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred"
)
# ⚠️ 原实现 `[a-z]+(?:[- ][a-z]+)?` 会在**每个字母位置**尝试匹配，再回退找 "bucks|dollars"
# —— 离线实测：10,504 条文本上单次 find_amounts 的 13.44s 里 **9.41s 全耗在这条正则**。
# 改成"先用数字词表限定开头"，交替项固定，复杂度降到线性。
_NW_ALT = "(?:" + "|".join(_NUM_WORD.split()) + ")"
_WORD_AMOUNT_WORDS = re.compile(
    r"\b(?P<number>" + _NW_ALT + r"(?:[- ]" + _NW_ALT + r")?)\s+(?P<currency>bucks|dollars?)\b",
    re.I,
)
_WORD_VALUES = {
    w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_WORD_VALUES.update({w: v for w, v in zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))})

_PURCHASE = re.compile(
    r"\b(?:bought|purchase[ds]?|picked up|grabbed|paid|spent|cost me|hit my account|charged|purchased)\b|买|花了|付了",
    re.I,
)
_NOISE = re.compile(
    r"expect to spend|budget|price range|range|target|between|sweet spot|typically|"
    r"list around|on average|red flag|let['’]s say|for example|e\.g\.|say the total|"
    r"set aside|per month|a month|savings",
    re.I,
)
_OBJECT = re.compile(
    r"\b(?:coffee|espresso|moka|kettle|pour.?over|dripper|grinder|brewer|pot|machine|kit|"
    r"maker|contraption|water|toaster|blender|router|nespresso)\b", re.I,
)
_OBJECT_ALIASES = {
    "coffee maker": ("coffee", "espresso", "nespresso", "moka", "pour-over", "pour over", "dripper", "brewer"),
    "coffee makers": ("coffee", "espresso", "nespresso", "moka", "pour-over", "pour over", "dripper", "brewer"),
    "espresso machine": ("espresso", "nespresso", "moka"),
    "coffee machine": ("coffee", "espresso", "nespresso", "moka"),
}


def _word_value(raw: str) -> Optional[float]:
    words = re.split(r"[ -]+", raw.lower().strip())
    if not words or any(w not in _WORD_VALUES and w != "hundred" for w in words):
        return None
    if "hundred" in words:
        before = words[:words.index("hundred")]
        if len(before) != 1 or before[0] not in _WORD_VALUES:
            return None
        value = _WORD_VALUES[before[0]] * 100
        rest = words[words.index("hundred") + 1:]
        if rest:
            tail = _word_value(" ".join(rest))
            if tail is None:
                return None
            value += tail
        return float(value)
    # The supported corpus uses simple cardinal forms (twenty-five, etc.).
    return float(sum(_WORD_VALUES[w] for w in words))


# 符号 → 币种。原来只用 `"$" ? USD : CNY` 二分，导致 €50 / £30 被标成 CNY。
_SYMBOL_CODE = {"$": "USD", "￥": "CNY", "¥": "CNY", "€": "EUR", "£": "GBP"}
# 词形 → 币种（euros/pounds 之前被归到 USD）
_WORD_CODE = {"元": "CNY", "euro": "EUR", "euros": "EUR", "pound": "GBP", "pounds": "GBP"}


def _iter_amounts(text: str):
    spans = []
    for m in _SYMBOL_AMOUNT.finditer(text):
        spans.append((m.start(), m.end(), m.group("number"),
                      _SYMBOL_CODE.get(m.group("currency"), "USD")))
    for m in _WORD_AMOUNT.finditer(text):
        word = m.group("currency").lower()
        spans.append((m.start(), m.end(), m.group("number"), _WORD_CODE.get(word, "USD")))
    for m in _WORD_AMOUNT_WORDS.finditer(text):
        value = _word_value(m.group("number"))
        if value is not None:
            spans.append((m.start(), m.end(), str(value), "USD"))
    return sorted(spans)


def _record_role(record: Dict[str, Any]) -> str:
    return str(record.get("role") or "").lower()


def _object_guess(context: str) -> str:
    hit = _OBJECT.search(context)
    return hit.group(0) if hit else ""


def find_amounts(
    user_id: str,
    *,
    store: Any = None,
    records: Optional[Iterable[Dict[str, Any]]] = None,
    object_hint: Optional[str] = None,
    limit: int = 50,
) -> list[Dict[str, Any]]:
    """Find likely user-paid amounts as auditable evidence rows.

    ``store`` is optional to keep the tool testable; production callers pass the
    existing MemoryStore. The function performs no writes and never computes a
    total. ``object_hint`` is a soft filter: rows without a matching object are
    retained when no row matches, avoiding silent evidence loss.
    """
    # 请求级缓存：金额题一次查询会调 2~3 次 find_amounts，
    # 同一 user 结果相同 ⇒ 用 store.version[user] 做键，add 后自动失效。
    _cache = getattr(find_amounts, "_memo", None)
    if _cache is None:
        _cache = find_amounts._memo = {}
    _ck = None
    if records is None and store is not None:
        try:
            _ck = (user_id, int(getattr(store, "version", {}).get(user_id, 0)), limit,
                   object_hint or "")
        except Exception:
            _ck = None
        if _ck is not None and _ck in _cache:
            return _cache[_ck]
    if records is None:
        if store is None:
            raise ValueError("store or records is required")
        records = store.list_memories(user_id)
    rows = []
    for record in records:
        if str(record.get("user_id") or user_id) != user_id or _record_role(record) != "user":
            continue
        text = str(record.get("index_text") or "")
        # ⚡ 预筛：没有货币标记的记录直接跳过，不跑三条金额正则。
        # 离线实测（10,504 条 ATM 文本 / 13.6M 字符）：三条金额正则合计 ~7.5s，
        # 而真正含货币标记的只有 670 条（6.4%）⇒ 预筛把正则工作量降 15 倍。
        # 用 `in` 做子串判断（C 实现，比正则快两个量级）。
        _tl = text.lower()
        if not any(_h in _tl for _h in _CURRENCY_HINT):
            continue
        # The stored prefix is authoritative, even if a test fixture omits role.
        for start, end, raw_number, currency in _iter_amounts(text):
            context = text[max(0, start - 90): min(len(text), end + 90)].replace("\n", " ").strip()
            before = text[max(0, start - 180):end + 120]
            if not _PURCHASE.search(before) or _NOISE.search(context):
                continue
            # Quoted prices and examples are usually advice, not a transaction.
            if text.rfind('"', 0, start) > text.rfind('"', 0, end):
                continue
            try:
                amount = float(raw_number.replace(",", ""))
            except ValueError:
                continue
            rows.append({
                "amount": f"{amount:.2f}",
                "currency": currency,
                "date": str(record.get("created_at") or "")[:10],
                "verb": (_PURCHASE.search(before).group(0) if _PURCHASE.search(before) else ""),
                "role": "user",
                "record_id": record.get("id"),
                "context": context,
                "object_guess": _object_guess(context),
                "session_id": record.get("session_id"),
                "request_id": record.get("request_id"),
            })
    if object_hint:
        hint = re.sub(r"\s+", " ", object_hint.lower().strip())
        aliases = _OBJECT_ALIASES.get(hint, (hint,))
        matched = [
            r for r in rows
            if any(alias in (r["context"] + " " + r["object_guess"]).lower()
                   for alias in aliases)
        ]
        if matched:
            rows = matched
    # Deterministic de-duplication: same record and amount is one evidence row.
    unique = []
    seen = set()
    for row in rows:
        key = (row["record_id"], row["amount"], row["currency"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= max(1, int(limit)):
            break
    if _ck is not None:
        try:
            if len(_cache) > 64:
                _cache.clear()
            _cache[_ck] = unique
        except Exception:
            pass
    return unique
