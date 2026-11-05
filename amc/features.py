"""校准器特征提取（离线与线上共用）。"""
from __future__ import annotations
import re

STOP = set("the a an of to in on at for with and or is are was were be been being do does did "
           "what which who whom whose when where why how much many this that these those it its "
           "i me my we our you your he she they them his her their as by from about into over "
           "there here have has had can could would should will shall may might must not no yes "
           "if then than so such more most less least very just also only own same other "
           "did does user julian please return format based conversation according "
           "什么 哪个 哪些 怎么 如何 是否 有没有 一下 这个 那个 我们 他们".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")
_CJK = re.compile(r"[\u4e00-\u9fff]{2,}")


def content_tokens(text: str) -> list:
    """英文词（≥3 字母）+ 中文二元组，去掉停用词。"""
    t = (text or "").lower()
    toks = [w for w in _WORD.findall(t) if w not in STOP]
    for seg in _CJK.findall(text or ""):
        toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


def coverage(question: str, texts: list) -> float:
    """问题实词在给定文本里的覆盖率（0~1）。"""
    qt = set(content_tokens(question))
    if not qt:
        return 0.0
    blob = " ".join(str(t or "") for t in texts).lower()
    hit = sum(1 for w in qt if w in blob)
    return hit / len(qt)
