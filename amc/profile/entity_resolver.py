"""商品名 → 归一化类别（确定性规则优先，LLM 只提供兜底候选）。

设计：规则表按"从具体到宽泛"排序，命中即返回。
`coffee_maker` 与 `coffee_accessory` 必须分开——
买手冲滤杯不等于买咖啡机（Q5 的 gold 只算 150+85，不算 25 的手冲套装）。
"""
from __future__ import annotations

import re
from typing import Optional

# (正则, 类别) —— 顺序即优先级
RULES = [
    # 咖啡机（真·制作设备）
    (r"\b(espresso machines?|espresso makers?|coffee machines?|coffee makers?|nespressos?|"
     r"drip coffee makers?|stovetop coffee makers?|moka pots?|moka|coffee pots?|coffee pot|percolators?|"
     r"french presses?|cafeti[eè]res?|aeropresses?|bean to cup)\b", "coffee_maker"),
    # 咖啡器具（耗材/辅助，不是"咖啡机"）
    (r"\b(pour[- ]overs?( kits?| drippers?| cones?)?|drippers?|v60|chemex|coffee grinders?|grinders?|"
     r"gooseneck|kettles?|filters?|paper filters?|tampers?|portafilters?|scales?)\b", "coffee_accessory"),
    # 其他厨房家电
    (r"\b(blender|toaster|air fryer|microwave|oven|mixer|rice cooker|slow cooker|"
     r"kettle|food processor|juicer|dishwasher)\b", "kitchen_appliance"),
    # 电子
    (r"\b(iphone|ipad|tablet|phone|laptop|macbook|router|monitor|headphone|earbuds|"
     r"speaker|camera|switch|console|electronics|gadget)\b", "electronics"),
    # 旅行
    (r"\b(flight|hotel|train ticket|airbnb|tour|hostel|travel|trip|journey|transport)\b", "travel"),
    # 餐饮
    (r"\b(dinner|lunch|breakfast|coffee shop|restaurant|groceries|grocery|takeaway|food|meal)\b", "food"),
]


_PLURAL = re.compile(r"\b([a-z]{3,}?)(ies|es|s)\b")


def _singularize(text: str) -> str:
    """把英文复数还原成单数，用于规则匹配。

    为什么需要：RULES 里全是单数形式（`hotel`/`flight`/`restaurant`/`dinner`/`laptop`/`trip`），
    而真实商品名常是复数（"hotels in Cairo"、"3 flights"、"dinners"）。
    `\bhotel\b` 匹配不到 "hotels" ⇒ 这些交易全部落 other ⇒ aggregate 返回 None ⇒
    L1 整整一条路径被跳过。离线审计实测：真实库 349 笔里 **58.5% 不可达规则类别**。
    """
    def _fix(m):
        w, suf = m.group(1), m.group(2)
        if suf == "ies":
            return w + "y"                 # groceries → grocery
        if suf == "es":
            return w                       # boxes → box（粗略）
        return w                           # hotels → hotel
    return _PLURAL.sub(_fix, text)


def resolve(item: str, llm_category: str = "") -> str:
    """返回归一化类别；规则不命中时用 LLM 给的类别（若有效），否则 other。"""
    t = (item or "").lower()
    if t:
        # 先原文匹配，再单数化匹配（顺序不能反：`scales` 应优先落 coffee_accessory）
        for pat, cat in RULES:
            if re.search(pat, t):
                return cat
        _t1 = _singularize(t)
        if _t1 != t:
            for pat, cat in RULES:
                if re.search(pat, _t1):
                    return cat
    lc = (llm_category or "").strip().lower()
    if lc and lc not in ("other", ""):
        # LLM 的类别可能是自由文本，做一次规则再映射
        for pat, cat in RULES:
            if re.search(pat, lc):
                return cat
        return lc
    return "other"


def is_coffee_maker(item: str, llm_category: str = "") -> bool:
    return resolve(item, llm_category) == "coffee_maker"


# ---------- 模糊物品 → 图片描述确定性匹配 ----------
_STOP = set("a an the of to in on at for with and or is are was were be this that it its "
            "my me i we our you your he she they them his her as by from about into over "
            "some really very just like got get bought buy grabbed thing something stuff".split())


def _toks(s: str):
    # 连字符拆开：stainless-steel → stainless, steel
    t = (s or "").lower().replace("-", " ").replace("/", " ")
    return {w for w in re.findall(r"[a-z]{2,}", t) if w not in _STOP}


def match_vague_item(item: str, quote: str, captions: list, min_score: int = 2):
    """把"模糊物品"匹配到某条图片描述。

    返回 (caption, category, score) 或 None。**纯词面证据**，不做语义推断：
      - 物品词 与 描述词 的重叠 ≥ min_score 才算命中
      - 场景词（market/flea/...）额外加分
    """
    it = _toks(item) | _toks(quote)
    if not it:
        return None
    # 模糊物品（"stainless-steel contraption" / thingamajig / gadget）：
    # 只要有一个**材质疑词**命中具体物品描述，就接受（放宽到 1 个词）
    _vague = re.search(r"thing|thingamajig|contraption|gadget|gizmo|whatchamacallit|stuff", item or "", re.I)
    if _vague:
        min_score = 1
    best = None
    for cap in (captions or []):
        ct = _toks(cap)
        overlap = it & ct
        if len(overlap) < min_score:
            continue
        # 只接受"具体物品"类别（避免把模糊物品匹配到收据/笔记之类）
        cat = resolve(cap)
        if cat in ("other", ""):
            continue
        score = len(overlap)
        if {"market", "flea", "stall", "shop"} & (it & ct):
            score += 2
        # 归类优先级：咖啡机 > 咖啡器具（避免"stainless steel"共享词把摩卡壶判成手冲套装）
        if cat == "coffee_maker":
            score += 1
        if best is None or score > best[2]:
            best = (cap, cat, score)
    return best
