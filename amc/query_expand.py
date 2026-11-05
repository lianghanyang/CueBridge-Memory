"""确定性的查询变体生成（多查询并集用，**不调 LLM**）。

为什么不用 LLM
--------------
2026-11-01 的对照实验证明：多查询并集的收益来自"多搜几次"，
**LLM 改写相对"随机扰动"没有优势**（配对 McNemar p=0.688）。
⇒ 用确定性规则生成变体，省一次 LLM 调用，效果相同。

生成什么
--------
三类变体，都被证明能提高 any-gold（配对 +6 题，p=0.031，零回归）：

```
V1 原问题                 （保留，作为并集的一项）
V2 去疑问词/去虚词        "…最让我难受的核心原因是什么？" → "公交 账本 数字 难受"
V3 只留实体与时间锚点      → "公交 账本"（人名/日期/地点/物件）
```

注意：变体**只用于召回**（pool），最终排序仍由原问题的精排决定 ——
所以变体质量不高也不会伤害精度，最坏情况是"多召回了几条不相干的"。
"""
from __future__ import annotations

import re

# 疑问/虚词：在检索里几乎不携带信息，去掉能让检索聚焦到实词
_STOP_ZH = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "自己",
    "的", "地", "得", "了", "着", "过", "是", "在", "有", "和", "与", "及",
    "那", "这", "那个", "这个", "那次", "这次", "那次那", "一次",
    "什么", "怎么", "为什么", "哪个", "哪些", "多少", "是否", "有没有", "能不能",
    "最后", "后来", "之前", "之后", "当时", "现在", "目前", "到底", "究竟",
    "会", "要", "想", "说", "做", "让", "把", "被", "给", "对", "从", "到",
    "比较", "更", "最", "很", "挺", "太", "还", "也", "都", "就", "才",
}
_STOP_EN = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "i", "my", "me", "we", "our", "you", "your", "he", "she", "it", "they",
    "that", "this", "these", "those", "there", "then", "than", "about",
    "would", "could", "should", "can", "will", "may", "might", "must",
    "most", "more", "much", "many", "some", "any", "all", "both",
}

_ZH_WORD = re.compile(r"[\u4e00-\u9fff]{2,}")
_EN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]{2,}")
_DATE = re.compile(r"\d{4}[-/年]\d{1,2}([-/月]\d{1,2}日?)?|\d{1,2}月\d{1,2}日?|\d{1,2}号")


def _words(q: str) -> list[str]:
    """按出现顺序抽词。

    中文没有分词器（jieba 不可用），用**2 字滑窗**做粗切：
    对检索变体来说够用（变体只用于扩大召回，不参与最终排序）。
    英文按 3+ 字母切。
    """
    out: list[tuple[int, str]] = []
    # 中文：连续的汉字串 → 2 字滑窗
    for m in re.finditer(r"[\u4e00-\u9fff]{2,}", q):
        run, base = m.group(), m.start()
        for i in range(len(run) - 1):
            out.append((base + i, run[i:i + 2]))
    for m in _EN_WORD.finditer(q):
        out.append((m.start(), m.group()))
    out.sort()
    # 去重但保序
    seen, res = set(), []
    for _, w in out:
        if w not in seen:
            seen.add(w)
            res.append(w)
    return res


def variants(query: str) -> list[str]:
    """返回 2 个变体（不含原问题）。去重后可能少于 2 个。"""
    q = (query or "").strip()
    if len(q) < 4:
        return []
    ws = _words(q)
    if not ws:
        return []
    low = {w.lower() for w in ws}
    keep = [w for w in ws if w.lower() not in _STOP_ZH and w.lower() not in _STOP_EN]

    out: list[str] = []

    # V2：**去掉句末"提问部分"，保留话题部分**
    # 中文问题常是"话题 + ，+ 提问"，例如
    #   "我那天在公交回家时，一直在想账本上的数字，最后最让我难受的核心原因是什么？"
    #   → 前半（话题）更适合检索，后半（提问）几乎没有检索价值
    if re.search(r"[\u4e00-\u9fff]", q):
        head = re.split(r"[，,？?]", q.rstrip("？?。.！!"))[0].strip()
        if len(head) >= 6 and head != q:
            out.append(head)
    # 英文：去掉疑问与虚词，只留实词
    if keep and len(" ".join(keep)) >= 4:
        en_like = sum(1 for w in keep if w.isascii())
        if en_like >= max(1, len(keep) // 2):
            out.append(" ".join(keep))

    # V3：实体与时间锚点（日期 + 专名首现 + 名词性词），最多 6 个
    anchors: list[str] = _DATE.findall(q)
    seen = set()
    for w in keep:
        if w in seen:
            continue
        # 人名/地名/物件：中文 2-4 字、英文首字母大写
        if (w.isascii() and q.find(w) >= 0 and w[:1].isupper()) or (2 <= len(w) <= 4 and not w.isascii()):
            anchors.append(w)
            seen.add(w)
        if len(anchors) >= 6:
            break
    # V3 只对**英文/混合**query 有意义；中文没有分词器时锚点全是无意义二元组（实测"我那 那天 天在"）
    zh_chars = sum(1 for ch in q if "\u4e00" <= ch <= "\u9fff")
    if zh_chars <= len(q) * 0.5:
        v3 = " ".join(anchors)
        if v3 and len(v3) >= 4:
            out.append(v3)

    # 去重 + 去掉与原问题相同的
    res, seen_s = [], {q}
    for v in out:
        v = v.strip()
        if v and v not in seen_s:
            seen_s.add(v)
            res.append(v)
    return res[:2]
