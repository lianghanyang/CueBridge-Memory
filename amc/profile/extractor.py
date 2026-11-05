"""L1 事实抽取：LLM 只负责"理解语义并输出结构化候选"，不做任何写库决定。"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

logger = logging.getLogger("amc.profile.extractor")

PROMPT = """从下面这批对话片段中抽取**结构化事实**。只输出 JSON。

规则：
- 只抽取**用户本人的**、有原文依据的事实；不要推断
- 交易（transaction）：必须有**明确金额**，且是**用户实际花钱**（不是预算、不是别人花钱、不是标价讨论）
  · **账单/收据语境也算**：出现 bill / invoice / receipt / charge / fare / fee / total /
    accommodation / dental / taxi / hotel 等词 + 金额 → 视为已发生的消费
    （例如 "Hotel invoice: €842.97"、"Dental bill: £190"、"taxi fare: €62.8"）
  · amount: 数字（不要带货币符号）；currency: USD/GBP/EUR/SGD/CNY 等
  · category: 见下方【已有类别】。规则（很重要）：
    - 若片段里出现了**具体地点 / 商家 / 品牌 / 型号**（如 Aberdeen、Waitrose、Brita、MX500、Conrad Cairo），
      类别名**必须包含该实体**：`hotel_aberdeen`、`grocery_waitrose`、`water_filter_brita`、`ssd_mx500`
      —— 不要退化用 travel / food 这种泛类别
    - 只有当物品确实没有可区分实体时（如泛指的 "coffee makers"、"groceries"），才复用已有泛类别
    - 新建类别用「小写下划线」命名，并在 category_desc 写一句话描述
  · parent: 粗粒度分组（coffee_maker / coffee_accessory / kitchen_appliance / travel / food / electronics / other）
  · occurred_at: YYYY-MM-DD（能从上下文推断就填，否则空）
  · status: confirmed（明确说已付款/已扣款）| pending（只是提到买过，金额不确定）
           | refunded（设计约束退货/退款/取消订单，且金额与之前某笔一致）
- 偏好（preference）：用户表达**倾向/态度/喜好**（**不必是"明确说喜欢"**）
  → predicate=enjoys|dislikes|prefers|recommends|values
  · 只要出现 "I like / I love / I prefer / I'd rather / I hate / 我更喜欢 / 我不太喜欢 /
    我就爱 / 我受不了 / 我觉得X更好 / X对我来说更重要" ⇒ 都要抽
  · object **照抄原文里的具体名词**（地点/物品/活动/风格），不要概括成"旅行""美食"
  · polarity=positive|negative
- 事件（event）：用户经历的重大事件（搬家/换工作/毕业/结婚/生病/旅行）→ predicate=moved|started_job|graduated|married|illness|traveled, object=地名/职位/事件, occurred_at=YYYY-MM-DD
- 意愿（intent）：用户表达**想做但还没做**的事（愿望/打算/兴趣）
  → predicate=wants_to_visit|wants_to_buy|wants_to_try|intends_to|hopes_to
  · 触发表达："I want to / I'd love to / I plan to / I'm going to / I hope to /
    I'm thinking about / 我想去 / 我打算 / 我准备 / 我希望 / 我下次要 / 有机会想去"
  · object **照抄原文里的具体名词**（城市/景点/物品/活动），**不要**写成"旅行""出去玩"
  · due_at=YYYY-MM-DD（可空）
- 计划（commitment）：用户**明确承诺**要做的事（有对象/时间更具体）
  → predicate=plans, object=要做的事, due_at=YYYY-MM-DD（可空）
- **偏好/意愿/计划都必须抽**（这是旅行/生活类用户的主要事实来源）：
  例：`I definitely want to visit the Louvre` ⇒ intent: wants_to_visit / "the Louvre"
  例：`I'd rather stay in a quiet hotel`        ⇒ preference: prefers / "a quiet hotel"
  例：`我下次想去京都看红叶`                      ⇒ intent: wants_to_visit / "京都 红叶"
- **不算交易（重要）**：假设/意愿/比较语气不是购买——
  "I'd pay / I would pay / I could afford / if I had to pick / I'm willing to spend /
   it's worth / costs around / a budget of" 这类**没有实际发生**的表述，**不要**列成交易
- **不能漏**：只要片段里出现「金额 + 购买动词」（bought / picked up / grabbed / paid / ordered），
  就必须列出一笔交易；金额是 "about $85"、"around $150" 这种**约数也要列出**
- 物品名**照抄原文**，不要改写、不要猜测（若原文模糊，就保留模糊原文）
- 只要购买、不要因为买了就推断喜欢
- 每条事实都要给 source: 片段编号 i

输出 JSON：
{{"transactions":[{{"i":0,"amount":150,"currency":"USD","item":"Nespresso machine","category":"coffee_maker","occurred_at":"2024-05-01","status":"confirmed","quote":"..."}}],
 "facts":[{{"i":0,"fact_type":"preference","predicate":"enjoys","object":"making espresso at home","polarity":"positive","quote":"..."}},
          {{"i":1,"fact_type":"event","predicate":"moved","object":"Berlin","occurred_at":"2024-04-01","quote":"..."}},
          {{"i":2,"fact_type":"commitment","predicate":"plans","object":"buy paper filters","due_at":"","quote":"..."}},
          {{"i":3,"fact_type":"intent","predicate":"wants_to_visit","object":"the Louvre","due_at":"","quote":"..."}}]}}

【已有类别】（可复用；没有合适的就新建）
__CATS__

片段：
__ITEMS__

__IMGCTX__"""
_PURE_DATE_RX = re.compile(
    r"^(?:"
    r"\d{4}-\d{2}-\d{2}"                                  # 2025-04-22
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s*\d{4}"                                  # April 22, 2025
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}"                                             # 22 April 2025
    r"|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"          # 2025年4月22日
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s*\d{4}"  # Apr 22, 2025
    r")$", re.I)

IMG_HINT = """【可用的图片描述】用户记忆里的图片被描述成下面这些内容，可用来判断某个模糊物品到底是什么：
__CAPS__"""

_CAT_HINT = [
    ("coffee|espresso|moka|pour-over|grinder|nespresso", "coffee_maker"),
    ("blender|toaster|air fryer|kettle|mixer|oven|microwave", "kitchen_appliance"),
    ("flight|hotel|train|trip|tour", "travel"),
    ("menu|dinner|lunch|restaurant|grocer", "food"),
    ("phone|laptop|tablet|router|monitor|headphone", "electronics"),
]


def guess_category(item: str) -> str:
    """按商品名猜类别（**仅作 LLM 未给类别时的兜底**）。

    ⚠️ 必须用词边界：不用的话 "trip" 会命中 "**trip**le monitor stand"、
    "train" 会命中 "**train**ing course"、 "tour" 会命中 "**tour**nament entry"
    —— 实测这些都曾被判成 travel。这里把每个模式包进
    `(?<![a-z])...(?![a-z])`，中文模式不受影响（无词边界概念，但当前列表里没有中文）。
    """
    t = (item or "").lower()
    for pat, cat in _CAT_HINT:
        if re.search(r"(?<![a-z])(?:" + pat + r")(?![a-z])", t):
            return cat
    return "other"


def _chat(prompt: str, base_url: str, api_key: str, model: str, timeout: float = 90.0,
          retries: int = 3) -> Dict[str, Any] | None:
    import httpx
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(base_url.rstrip("/") + "/chat/completions",
                           headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                           json={"model": model, "temperature": 0.0, "max_tokens": 900,
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=timeout)
            r.raise_for_status()
            t = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if t.startswith("```"):
                t = t.split("```")[1]
                t = t[4:] if t.lower().startswith("json") else t
            return json.loads(t)
        except Exception as exc:
            st = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(st, int) and 400 <= st < 500:
                return None
            logger.warning("[extract] attempt %d failed: %s", attempt, exc)
            time.sleep(1.2 * attempt)
    return None


def extract_batch(items: List[Dict[str, str]], base_url: str, api_key: str, model: str,
                  categories: List[Dict[str, Any]] | None = None,
                  image_captions: List[str] | None = None) -> Dict[str, Any]:
    """items: [{i, text, date, memory_id, request_id}] → {transactions, facts}

    image_captions: 该用户图片的描述（用于把"某个模糊物品"映射到具体类别）
    """
    body = "\n\n".join(f"[{it['i']}] (date={it.get('date','')}) {it['text'][:1200]}" for it in items)
    imgctx = ""
    caps = [c for c in (image_captions or []) if c][:15]
    if caps:
        imgctx = IMG_HINT.replace("__CAPS__", "\n".join(f"- {c[:160]}" for c in caps))
    _cats = ""
    if categories:
        _cats = "\n".join(f"- {c.get('name')}：{c.get('description') or ''}" for c in categories[:60])
    out = _chat(PROMPT.replace("__ITEMS__", body).replace("__IMGCTX__", imgctx).replace("__CATS__", _cats or "（暂无，请新建）"),
                base_url, api_key, model) or {}
    txs, facts = [], []
    for t in (out.get("transactions") or []):
        i = t.get("i")
        if not (isinstance(i, int) and 0 <= i < len(items)):
            continue
        # 金额：LLM 偶尔给出 item 却漏 amount（"about $85" 这类约数最常漏）
        # → 从 quote/item 里用正则兜底抓一次
        _amt_raw = t.get("amount")
        if _amt_raw in (None, "", 0):
            _blob = f"{t.get('quote') or ''} {t.get('item') or ''}"
            _m = re.search(r"[\$£€¥]\s?(\d[\d,]*(?:\.\d+)?)"
                           r"|(\d[\d,]*(?:\.\d+)?)\s?(?:dollars?|bucks|pounds?|usd|元|块)", _blob, re.I)
            if _m:
                _amt_raw = (_m.group(1) or _m.group(2)).replace(",", "")
                if not t.get("currency"):
                    # ⚠️ 原来只判 "$"/"dollar"，其余一律 GBP ——
                    # ⇒ €50 被标成 GBP、¥1200 / "100元" 也被标成 GBP。
                    # 用符号/词形精确映射（与 amc/tools/amounts.py 保持一致）。
                    _hit_txt = _m.group(0)
                    _low = _hit_txt.lower()
                    if "$" in _hit_txt or "dollar" in _low or "usd" in _low:
                        t["currency"] = "USD"
                    elif "€" in _hit_txt or "euro" in _low:
                        t["currency"] = "EUR"
                    elif "£" in _hit_txt or "pound" in _low:
                        t["currency"] = "GBP"
                    elif "¥" in _hit_txt or "￥" in _hit_txt or "元" in _hit_txt or "块" in _hit_txt:
                        t["currency"] = "CNY"
                    else:
                        t["currency"] = "USD"
        try:
            amount = float(_amt_raw)
        except Exception:
            continue
        if amount <= 0:
            continue
        _hypo = re.search(r"\b(i'?d pay|i would pay|i could (pay|afford)|i'?m willing to|"
                          r"if i had to|it'?s worth|costs? around|a budget of|would cost|"
                          r"i might (buy|get|spend))\b",
                          (str(t.get("quote") or "") + " " + str(t.get("item") or "")).lower())
        if _hypo:
            continue
        txs.append({"i": i, "amount_minor": int(round(amount * 100)),
                    "currency": (t.get("currency") or "").upper(),
                    "item": str(t.get("item") or ""), "category": str(t.get("category") or guess_category(t.get("item") or "")),
                    "occurred_at": str(t.get("occurred_at") or ""),
                    "status": (lambda v: v if v in ("confirmed", "refunded") else "pending")(str(t.get("status")).lower()),
                    "quote": str(t.get("quote") or "")[:300]})
    for f in (out.get("facts") or []):
        i = f.get("i")
        if not (isinstance(i, int) and 0 <= i < len(items)):
            continue
        #   实测噪音："event/traveled **April 22, 2025**" —— 把日期当成了事件对象，
        #   而 facts_note 会把它拼进 data[] 给平台 answer 模型看 ⇒ 纯噪音。
        _obj_s = str(f.get("object") or "").strip()
        if _PURE_DATE_RX.match(_obj_s):
            continue
        facts.append({"i": i, "fact_type": str(f.get("fact_type") or "preference"),
                      "predicate": str(f.get("predicate") or ""), "object": str(f.get("object") or ""),
                      "polarity": str(f.get("polarity") or "positive"),
                      "occurred_at": str(f.get("occurred_at") or ""),
                      "due_at": str(f.get("due_at") or ""),
                      "quote": str(f.get("quote") or "")[:300]})
    return {"transactions": txs, "facts": facts}
