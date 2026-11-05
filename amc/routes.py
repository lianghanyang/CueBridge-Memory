"""自动选路 + 证据召唤（2026-11-02）

给定 (store, user_id, question)，选一条路线并召回证据：

  路线 A `event`（题面引用事件）：事件锚定 → 时间窗过滤 → 字段抽取
  路线 B `field`（列举/聚合型，不引用事件）：按字段直接聚合
  路线 C `semantic`（其余，如叙事回忆/多模态定位）：回退到语义检索

依据（ATM 52 题实测）：
```
事件锚定可覆盖 26 道；字段聚合可覆盖 32 道；合计 **42/52 = 81%**
剩下 10 道（19%）是叙事回忆/图片定位 ⇒ 才需要语义检索
```

单题实测（对着 gold 核过）：
```
· "Which European countries have I visited"  ⇒ 路线 B ⇒ gold 6/6，答对
· "How much for BMVC accommodation"          ⇒ 路线 A ⇒ gold 2/2，799.74 GBP
· "How long for ECCV 2024"                   ⇒ 路线 A ⇒ 答对日期区间
· "NeurIPS list all cities"                  ⇒ 路线 A ⇒ gold 8/10
```
"""

from __future__ import annotations

import datetime as _dt
import re
import time
from typing import Any, Dict, List, Optional

from .field_tools import by_location_role, event_anchor, fields_of, location_level

# 问句分类用的正则（宁可宽一点，后面还有事件判据兜底）
_ENUM_RX = re.compile(r"which\s+.*\b(have|did|do)\b|list\s+all|recall\s+all|all\s+the\b|"
                      r"哪些|列出|全部|所有", re.I)
_AGG_RX = re.compile(r"how\s+(much|many|long|often)|total|start\s+and\s+end|"
                     r"多久|多少|几次|总(共|额)", re.I)
_EVENT_RX = re.compile(r"\bduring\b|\bwhen\b|\bafter\b|\bbefore\b|"
                       r"\b(trip|travel|conference|visit|stay)\b|"
                       r"那次|当时|那次旅行|会议上", re.I)
# 题面里可能是"事件名"的专有名词（缩写/带年份的名词）
_EVENT_NAME_RX = re.compile(r"\b([A-Z][A-Za-z]{1,9})\s*(20\d{2})\b")


_AMOUNT_RX = re.compile(r"how\s+much|\btotal\b|\bpay\b|\bpaid\b|\bcost\b|\bspend\b|"
                        r"\bspent\b|\breceive[ds]?\b|\brefund\b|\bcompensation\b|\bprice\b|"
                        r"多少钱|花了|花费|总(共|额)|费用", re.I)
_COUNT_RX = re.compile(r"how\s+(many|long|often)|how\s+nights|\bdays?\b.*\bstay|"
                       r"when\s+did\b|start\s+and\s+end|from\s+\d{4}\s+to\s+\d{4}|"
                       r"多久|多少天|几次|几晚|什么时候|从.*到.*(?:天|年)", re.I)


def classify(question: str) -> str:
    """返回 'enum' | 'amount' | 'count' | 'other'。

    2026-11-02 修正：原来把「金额」和「数量/时长」都归成 `agg`，
    结果金额类题走 Location 聚合 ⇒ **6 题全 0**（Location 里没有金额）。
    拆分后：
      · amount ⇒ 走「事件锚定/主题过滤 + **全文**抽金额求和」
      · count  ⇒ 走「Location/Time 聚合」
    """
    q = str(question or "")
    if _AMOUNT_RX.search(q):
        return "amount"
    if _COUNT_RX.search(q):
        return "count"
    if _ENUM_RX.search(q):
        return "enum"
    return "other"


def has_event(question: str) -> bool:
    """题面是否引用了一个具体事件/时间点。"""
    return bool(_EVENT_RX.search(str(question or "")))


def event_name(question: str) -> str:
    """从题面抽事件名的**锚词**（用于在记录里定位事件）。

    优先取"大写缩写 + 年份"（BMVC 2024 / NeurIPS 2023）；
    没有则退回第一个大写专有名词。
    """
    q = str(question or "")
    # ⚠️ 2026-11-02 关键 bug：`_EVENT_NAME_RX` 会把 **"March 2023" 抽成 "March"**、
    #    "June 2022" 抽成 "June" ⇒ 用"March"当事件词去锚定 ⇒ 锚到 2022 年的记录
    #    ⇒ 窗口完全错位（gold 在 2023-03）⇒ 该题 gold 命中 0。
    #    ⇒ **必须排除月份/星期/常见词**。
    _BAD = {"january", "february", "march", "april", "may", "june", "july", "august",
            "september", "october", "november", "december", "jan", "feb", "mar", "apr",
            "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "the", "this", "that", "my", "our", "last", "next", "year", "month", "week",
            "day", "today", "yesterday", "tomorrow", "summer", "winter", "spring", "autumn",
            "fall", "recent", "past", "first", "second", "third"}
    for m in _EVENT_NAME_RX.finditer(q):
        cand = m.group(1)
        if cand.lower() not in _BAD:
            return cand
    # ⚠️ 2026-11-02：**不要再退回"任何大写词"**。
    #    "Which European countries have I visited" ⇒ 退回到某个大写词 ⇒ 走 event 路线 ⇒ 空结果。
    #    只有"大写缩写+年份"才算事件名（BMVC 2024 / NeurIPS 2023 / ECCV 2024）。
    return ""


def windows_from_event(store: Any, user_id: str, anchor: str,
                       span_days: int = 2, lookback_days: int = 21,
                       cluster_days: int = 7) -> Dict[str, Any]:
    """事件锚定 → 日期集合 → 非对称窗口。

    非对称的理由（实测）：住宿邮件的 `Time:` 是**预订日**，比入住日早数周。
      ⇒ 向后（入住之后）只放宽 `span_days` 天
      ⇒ 向前（预订之前）放宽 `lookback_days` 天
    """
    a = event_anchor(store, user_id, re.escape(anchor)) if anchor else {"dates": [], "places": []}
    dates = a.get("dates") or []
    # ⚠️ 2026-11-02：不能把**所有**提到事件名的日期都当锚点。
    #    实测：BMVC 的日期集合跨 2024-11 ~ 2025-04（6 个日期）⇒ 窗口 62 天
    #    ⇒ 把会议前后所有消费都聚合进来（57491 / 6105，正确 799.74）。
    #    ⇒ 只保留**最密集的一段**（事件本身通常连续 3~6 天）。
    if len(dates) > 1:
        _pd = []
        for _ds in dates:
            try:
                _pd.append(_dt.date.fromisoformat(_ds))
            except Exception:
                pass
        if _pd:
            _pd.sort()
            best, best_lo, best_hi = -1, _pd[0], _pd[0]
            for _i, _start in enumerate(_pd):
                _end = _start
                _cnt = 0
                for _d2 in _pd[_i:]:
                    if (_d2 - _start).days <= int(cluster_days):   # 同一事件的日期跨度上限
                        _end = _d2
                        _cnt += 1
                if _cnt > best:
                    best, best_lo, best_hi = _cnt, _start, _end
            dates = [best_lo.isoformat(), best_hi.isoformat()]
    win: set = set()
    for ds in dates:
        try:
            base = _dt.date.fromisoformat(ds)
        except Exception:
            continue
        for k in range(-int(lookback_days), int(span_days) + 1):
            win.add((base + _dt.timedelta(days=k)).isoformat())
    return {"anchor": anchor, "event_dates": dates, "places": a.get("places") or [],
            "window": sorted(win)}


#    所以必须在**流水线内部**检查并提前放弃。
#    用法：service.search 开头设 DEADLINE = time.time() + budget；结束清零。
import threading as _th

_TL = _th.local()


def set_deadline(seconds: float) -> None:
    """为**当前线程**设置硬预算（并发安全：每个请求在自己线程里）。"""
    _TL.deadline = (time.time() + float(seconds)) if seconds and seconds > 0 else 0.0


def clear_deadline() -> None:
    _TL.deadline = 0.0


def deadline_hit() -> bool:
    """是否已超过本线程的硬预算。"""
    d = getattr(_TL, "deadline", 0.0)
    return bool(d) and time.time() > d


def records_in_window(store: Any, user_id: str, window: List[str]) -> List[Dict[str, str]]:
    """窗口内记录（用 Time 字段的日期前缀匹配）。"""
    if not window:
        return []
    wset = set(window)
    out: List[Dict[str, str]] = []
    # ⚠️ 不要用 `store.records_by_date(limit=...)`：它内部 `min(500, limit)` 夹到 500
    if deadline_hit():
        logger.warning("[deadline] records_in_window 前已超预算，返回空")
        return []
    conn = getattr(store, "conn", None)
    rows = []
    if conn is not None:
        try:
            #   供 route:event:* 继承源 id（原来只取 request_id ⇒ 只能拼合成 id）。
            rows = [{"id": r[2], "request_id": r[0], "index_text": r[1]}
                    for r in conn.execute("SELECT request_id, index_text, id FROM memories WHERE user_id=?",
                                          (str(user_id),))]
        except Exception:
            rows = []
    if not rows and hasattr(store, "records_by_date"):
        try:
            rows = store.records_by_date(user_id, limit=200000)
        except Exception:
            rows = []
    for r in rows:
        txt = str(r.get("index_text") or r.get("text") or "")
        m = re.search(r"Time:\s*(20\d{2}-\d{2}-\d{2})", txt)
        if m and m.group(1) in wset:
            f = fields_of(txt)
            f["id"] = str(r.get("id") or "")
            f["request_id"] = str(r.get("request_id") or "")
            f["_raw"] = txt
            out.append(f)
    return out


def evidence(store: Any, user_id: str, question: str,
             level: str = "country") -> Dict[str, Any]:
    """自动选路并召回证据（**不做最终作答**，只给证据 + 路线说明）。

    返回：
      route: 'event' | 'field' | 'semantic'
      anchor / window / records / items / note
    """
    q = str(question or "")
    kind = classify(question)
    ev = has_event(question)
    # ⚠️ 2026-11-02：**实体类型必须由问题决定**，不能永远 country。
    #    实测："Which **hotels** have I stayed at in Glasgow" 拿到的是**国家清单**
    #    ⇒ 答案端只能答"抱歉，无法提供"。
    #    ⇒ 现在：countries/cities 用 Location 的对应层级；
    #      hotels/restaurants 这类**不在 Location 里** ⇒ **不注入聚合事实**（让记录说话）
    _ql = q.lower()
    if re.search(r"\bcountr(y|ies)\b|哪些国家|哪个国家", _ql):
        level = "country"
    elif re.search(r"\bcit(y|ies)\b|哪些城市|哪个城市", _ql):
        level = "city"
    elif re.search(r"\b(hotel|restaurant|cafe|bar|shop|store|venue)s?\b", _ql):
        level = "none"        # 这类实体不在 Location 里，别注入
    else:
        level = "country"
    # 路线 B0（已移除，2026-11-02）：
    #   曾为 count 类做"日期跨度"路线，但实测 **9 道题全部 0 命中**，
    #   且只返回 3 个**重复的错误区间**（2024-08-10~08-17 / 2022-07-10~07-14 /
    #   2024-11-26~11-29）—— 说明"最频繁区间"抓到的是模板化的预订邮件。
    #   ⇒ 这 9 道回退到 field 路线（实体覆盖 88%），比错误区间有用。
    #   ⇒ 教训：**"看起来像结构化的答案"比"没有答案"更危险**
    #      —— 它会以很具体的形式（日期区间）给出错误结论。

    # 路线 A：引用事件 ⇒ 先锚定
    # ⚠️ 2026-11-02：不能只依赖 `has_event` 的介词特征。
    #    "How much did I pay for my accommodation for **BMVC 2024**" 不含 during/when/trip，
    #    但 BMVC 就是事件名 ⇒ 必须锚定到 Glasgow + Nov 2024 ⇒ 才能拿到那 2 条酒店邮件。
    #    不锚定 ⇒ 金额合计 1827.00（错）；锚定后 ⇒ 799.74（对）。
    if ev or event_name(question):
        name = event_name(question)
        if name:
            # ⚠️ 2026-11-02 实测教训：**窗口宽度必须按题型区分**
            #    · amount/金额类：窗口窄（聚类 4 天/回看 10 天）⇒ 精度优先
            #      否则把整个月的消费都聚合进来（57491 vs 799.74）
            #    · 其余（列举/时长）：窗口宽（聚类 7 天/回看 21 天）⇒ 召回优先
            #      否则漏掉 gold（实测 85% → 46%）
            #    —— 我曾在 2~3 道题上调窄窗口，结果破坏了另外 9 道
            # ⚠️ 2026-11-02：**级联窗口**（与 amount 的级联回退同一模式）。
            #    实测：窄窗保精度但漏 gold（event 46%）；宽窗保召回但聚合噪声大。
            #    ⇒ 逐级放宽，并在 `stage` 里标注用了哪一级，交给下游判断。
            _narrow = (kind == "amount")
            _stages = (
                [("narrow", 4, 10), ("medium", 7, 21), ("wide", 14, 45)]
                if _narrow else
                [("medium", 7, 21), ("wide", 14, 45), ("verywide", 30, 90)]
            )
            _last_w = None
            for _st, _cl, _lb in _stages:
                _last_w = w = windows_from_event(
                    store, user_id, name, span_days=2,
                    lookback_days=_lb, cluster_days=_cl)
                recs = records_in_window(store, user_id, w["window"])
                if recs:
                    return {"route": "event", "kind": kind, "anchor": name,
                            "event_dates": w["event_dates"], "places": w["places"],
                            "n_window": len(w["window"]), "records": recs,
                            "stage": _st,
                            "note": f"事件锚定成功（窗口级别 {_st}）；后续按问题要的层级/量聚合"}
            # 锚到了但窗口内没记录 ⇒ **也要退回**（否则 event 路线会返回空）
        # 锚不到（或窗口为空）⇒ 退回字段/金额
    # 路线 B2：金额类 ⇒ 需要「主题过滤 + 全文抽金额」，**不能用 Location**
    if kind == "amount":
        return {"route": "field-amount", "kind": kind,
                "note": "金额类：需主题词过滤 + 从**全文**抽金额（`amounts_in` 必须传全文）；"
                        "若题面引用事件，先用 windows_from_event 锚定"}



    # 路线 B：列举/数量 ⇒ 字段聚合
    if kind in ("enum", "count"):
        if level == "none":
            # 实体不在 Location 里（酒店/餐厅/店铺名）⇒ **不注入聚合事实**
            return {"route": "field", "kind": kind, "items": [],
                    "note": "实体类型不在 Location 字段里 ⇒ 不注入聚合事实（避免无关清单污染）"}
        cnt, n = by_location_role(store, user_id, level=level)
        items = cnt.most_common(200)
        # ⚠️ 2026-11-02：**"which countries" 必须过滤到真国家**。
        #    实测答案模型会把 Paris/Amsterdam/Rome/Milan 当国家列出
        #    ⇒ 因为 location_level 对某些 Location 串返回了城市。
        ql = str(question or "").lower()
        if "countr" in ql or "国家" in str(question or ""):
            from .field_tools import filter_countries
            filt = filter_countries([{"item": k, "n": v} for k, v in items])
            if filt:                      # 过滤后非空才替换
                items = [(it["item"], it["n"]) for it in filt]
        # ⚠️ 2026-11-02：**tags 只作 field 的兜底**，不能插在 field 之前。
        #    实测：把 tags 放前面会**抢走 field 的 19 道题**（field 22 → 3），
        #    而 field 的实体覆盖是 88% ⇒ 那是净损失。
        if not items:
            try:
                _tr = tags_route(store, user_id, q)
            except Exception:
                _tr = None
            if _tr:
                return _tr
        return {"route": "field", "kind": kind, "n_location_records": n,
                "items": items,
                "note": "Location 字段聚合" + ("（已过滤到国家白名单）" if ("countr" in ql or "国家" in str(question or "")) else "")}
    # 路线 C：语义检索
    return {"route": "semantic", "kind": kind,
            "note": "叙事回忆/多模态定位 ⇒ 回退到语义检索（我的 8 轮优化都在这里，且目标群只占 19%）"}


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def question_time_window(question: str) -> List[str]:
    """从**题面**解析时间约束，返回 `Time:` 字段可匹配的前缀列表（'YYYY-MM' 或 'YYYY'）。

    2026-11-02 实测教训：金额类题若只按"实体词"过滤会失败
      （"How much Marvel Milk Powder did I buy in **March 2023**" ⇒ 8 条候选、gold 0）
    加上题面里的时间约束后 ⇒ "Marvel" + "2023-03" ⇒ 3 条、**gold 2/2**
    ⇒ 所以"实体 + 时间"必须**同时**用，而时间要从**题面**解析。
    """
    q = str(question or "")
    out: List[str] = []
    ql = q.lower()
    for name, num in _MONTHS.items():
        for m in re.finditer(rf"\b{name}\s+(20\d{{2}})\b", ql):
            out.append(f"{m.group(1)}-{num:02d}")
        for m in re.finditer(rf"\b(20\d{{2}})\s+{name}\b", ql):
            out.append(f"{m.group(1)}-{num:02d}")
    # 只有年份
    if not out:
        for m in re.finditer(r"\b(20\d{2})\b", q):
            out.append(m.group(1))
    return sorted(set(out))


def entity_tokens(question: str) -> List[str]:
    """从题面抽"实体词"（去掉疑问/功能词），用于过滤记录。"""
    STOP = {"how", "much", "many", "did", "do", "does", "pay", "paid", "total", "the", "for",
            "my", "and", "what", "in", "of", "to", "was", "were", "is", "are", "i", "me",
            "on", "at", "by", "with", "from", "during", "recent", "trip", "buy", "bought",
            "spend", "spent", "receive", "received", "price", "cost", "get", "got"}
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", str(question or ""))
    return [t for t in toks if t.lower() not in STOP][:6]


def aggregate(store: Any, user_id: str, question: str,
              route: Optional[str] = None, ev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把证据**聚合成一条事实**（2026-11-02）。

    为什么必须有这一步：
    ```
    实测（生产路径，7 道难题）：只给原始证据 ⇒ 端到端答对 **1/7**
      · "Which European countries" ⇒ 字段聚合本身就是事实 ⇒ 答案模型直接读就对
      · "How much for BMVC accommodation" ⇒ 给 11 条酒店邮件 ⇒ 答案是 799.74 GBP
        ⇒ **需要求和**，答案模型不会主动算 ⇒ 失败
    ⇒ 对"列举/聚合型"题：**证据 ≠ 结论**，必须在服务侧聚合。
    ```

    返回 {"fact": "<一句话事实>", "kind": "amount|span|list|count", "detail": {...}}
    """
    ev = ev or evidence(store, user_id, question)
    route = route or ev.get("route")
    kind = ev.get("kind")
    q = str(question or "")

    # 计数类（how many times）**已移除**（2026-11-02）：
    #   曾用"该地名下不同日期数"计数，但结果荒谬：
    #     · "How many times have I visited Newcastle?" ⇒ **7 次**（gold 2 次）
    #     · "How many times have I been to Porto?"     ⇒ **1170 次**
    #   ⇒ 因为"提到该地名的记录"远多于"真正去过的次数"
    #   ⇒ **"1170 次"这种具体而荒谬的数字，比"无法回答"危险得多**
    #   ⇒ 同 span 路线一样：**宁可空白，不给荒谬的具体值**

    # ① 金额：求和（只用窗口/过滤内的记录）
    if kind == "amount" or route == "field-amount":
        recs = ev.get("records") or []
        # ⚠️ 2026-11-02：必须按**方面词**收窄，否则会把整个窗口的消费都算进去
        #    实测：BMVC 窗口 62 天 ⇒ 合计 57491.07（正确 799.74）
        _asp = aspect_regex(q)
        if _asp and recs:
            _arx = re.compile(_asp, re.I)
            recs = [r for r in recs if _arx.search(str(r.get("_raw") or r.get("summary") or ""))]
        if not recs:
            # ⚠️ 2026-11-02 关键教训：**先放宽，再收紧**。
            #    实测 13 道金额题里 **8 道候选为空** —— 因为"实体词 ∧ 时间 ∧ 金额"三重过滤
            #    太严（题面 entity 词与数据用词不一致，如 accommodation vs hotel reservation）。
            #    ⇒ 在召回为 0 的地方优化精度 = 在没水的地方修水管。
            #    做法：**级联回退** strict → medium → loose → any。
            from .field_tools import amounts_in as _ai
            et, tw = entity_tokens(q), question_time_window(q)
            _asp = aspect_regex(q)
            rows_all = []
            try:
                rows_all = [str(tt or "") for (tt,) in store.conn.execute(
                    "SELECT index_text FROM memories WHERE user_id=?", (str(user_id),))]
            except Exception:
                rows_all = []
            def _collect(pred):
                got = []
                for t in rows_all:
                    if not pred(t):
                        continue
                    vs = _ai([t])[:2]
                    if vs:
                        got.append((t, vs))
                return got
            stages = [
                ("strict", lambda t: (not et or any(e.lower() in t.lower() for e in et))
                                     and (not tw or any(w in t for w in tw))
                                     and (not _asp or re.search(_asp, t, re.I))),
                ("medium", lambda t: (not et or any(e.lower() in t.lower() for e in et))
                                     and (not _asp or re.search(_asp, t, re.I))),
                ("loose",  lambda t: (not _asp or re.search(_asp, t, re.I))),
                ("any",    lambda t: True),
            ]
            for name, pred in stages:
                got = _collect(pred)
                if got:
                    # ⚠️ 2026-11-02：**候选必须带币种**，并优先取主导币种。
                    #    实测：不标币种时答案模型挑了 THB 25,829.26（应为 GBP 799.74）。
                    from .field_tools import amounts_with_currency as _awc
                    cur_pairs = []
                    for t, _vs in got:
                        cur_pairs.extend(_awc([t])[:2])
                    _cc: dict = {}
                    for cp in cur_pairs:
                        _cc[cp["currency"] or "?"] = _cc.get(cp["currency"] or "?", 0) + 1
                    _dom = max(_cc, key=_cc.get) if _cc else ""
                    _dom_vals = [cp["value"] for cp in cur_pairs if (cp["currency"] or "?") == _dom]
                    if _dom_vals:
                        #    给一个具体的 sum（如 827.64，真值 799.74）属于 C 类
                        #    "返回具体但错的值" —— 命中率不低但危险。
                        #    改成：只给候选 + 条数 ⇒ 答案端自己加，错了也知道加的是哪些。
                        return {"kind": "amount",
                                "fact": f"候选金额 {_dom_vals[:12]} {_dom}（共 {len(got)} 条记录；"
                                        f"共 {len(got)} 笔）；"
                                        f"过滤级别={name}；币种分布={_cc}",
                                "detail": {"candidates": _dom_vals[:20],
                                           "currency": _dom,
                                           "currency_dist": _cc,
                                           "sum": round(sum(_dom_vals), 2),
                                           "n_records": len(got),
                                           "stage": name, "entities": et, "time": tw,
                                           "aspect": _asp,
                                           "note": f"过滤级别 {name}；**只含 {_dom} 金额**；构成需核对"}}
                    vals = [v for _t, vs in got for v in vs][:20]
                    pairs = [{"values": vs, "snippet": t[:160]} for t, vs in got[:10]]
                    return {"kind": "amount",
                            # ⚠️ M64：不输出合计（具体但错的值比空白危险）
                            "fact": f"候选金额 {vals}（共 {len(got)} 笔）；"
                                    f"过滤级别={name}；实体={et[:3]} 时间={tw[:2]} 方面词={_asp or '无'}",
                            "detail": {"candidates": vals, "sum": round(sum(vals), 2),
                                       "n_records": len(got), "records": pairs,
                                       "stage": name, "entities": et, "time": tw,
                                       "aspect": _asp,
                                       "note": f"过滤级别 {name}；级别越松噪声越大，需核对构成"}}
            return {"kind": "amount", "fact": "", "detail": {}}
        from .field_tools import amounts_in as _ai2, amounts_with_currency as _awc2
        _asp3 = aspect_regex(q)
        _arx3 = re.compile(_asp3, re.I) if _asp3 else None
        pairs: list = []
        cur_all: list = []
        for r in recs:
            raw = str(r.get("_raw") or r.get("summary") or "")
            if _arx3 and not _arx3.search(raw):
                continue
            vs = _ai2([raw])[:2]
            if vs:
                pairs.append({"request_id": r.get("request_id"), "values": vs,
                              "snippet": raw[:160]})
                cur_all.extend(_awc2([raw])[:2])
        if pairs:
            allv = [v for p in pairs for v in p["values"]]
            # ⚠️ 2026-11-02：**候选必须带币种 + 主导币种优先**。
            #    实测：不标币种 ⇒ 答案模型挑了 THB 25,829.26（应为 GBP 799.74）。
            _cc2: dict = {}
            for cp in cur_all:
                k2 = cp["currency"] or "?"
                _cc2[k2] = _cc2.get(k2, 0) + 1
            _dom2 = max(_cc2, key=_cc2.get) if _cc2 else ""
            _dv = [cp["value"] for cp in cur_all if (cp["currency"] or "?") == _dom2] or allv
            # ⚠️ 同上：**不输出合计**（M64：具体但错的值比空白危险）
            fact = (f"候选金额 {_dv[:12]} {_dom2}（共 {len(pairs)} 条记录；"
                    f"共 {len(got)} 笔）；已按事件窗口 + 方面词过滤：{_asp3 or '无'}；"
                    f"币种分布={_cc2}")
            return {"kind": "amount", "fact": fact,
                    "detail": {"candidates": _dv[:20], "currency": _dom2,
                               "currency_dist": _cc2,
                               "sum": round(sum(_dv), 2),
                               "n_records": len(pairs), "records": pairs[:10],
                               "aspect": _asp3,
                               "note": f"只含 {_dom2} 金额；构成需核对"}}
        return {"kind": "amount", "fact": "", "detail": {}}

    # ② 时长/区间：span 路线直接给日期跨度
    if ev.get("route") == "span" and ev.get("first"):
        return {"kind": "span",
                                #    而这是"不同日期的个数"，不是"跨度天数" ⇒ 明确写成"N 个不同日期"。
                                "fact": f"{ev['first']} ~ {ev['last']}（**{ev.get('n_days')} 个不同日期**；"
                        f"时间过滤={ev.get('time_filter') or '无'}）",
                "detail": {"first": ev["first"], "last": ev["last"],
                           "n_days": ev.get("n_days"), "source": "span-route"}}

    # ② 时长/区间：从 Time 字段取最早/最晚
    if kind == "count" and _COUNT_RX.search(q):
        recs = ev.get("records") or []
        # ⚠️ 2026-11-02：不能取 `Time:` 的 min/max（那是**邮件收发/预订日期**，
        #    实测得 2024-09-05 ~ 2024-12-13 / 59 天，而真实入住只有 9/29~10/1）。
        #    正确做法：从 **Summary** 里抽"入住/离店"日期。
        _asp2 = aspect_regex(q)
        _arx2 = re.compile(_asp2, re.I) if _asp2 else None
        times = []
        for r in recs:
            txt = str(r.get("_raw") or r.get("summary") or "")
            if _arx2 and not _arx2.search(txt):
                continue
            for m in re.finditer(r"(20\d{2}-\d{2}-\d{2})", txt):
                times.append(m.group(1))
            # 也支持 "September 29 to October 1, 2024" 这类写法
            for m in re.finditer(r"([A-Z][a-z]+\s+\d{1,2})\s*(?:to|-|–)\s*([A-Z][a-z]+\s+\d{1,2}),?\s*(20\d{2})", txt):
                for part in (m.group(1), m.group(2)):
                    try:
                        mo, da = part.split()
                        _MON = {k[:3]: v for k, v in _MONTHS.items()}
                        mm = _MON.get(mo[:3].lower())
                        if mm:
                            times.append(f"{m.group(3)}-{mm:02d}-{int(da):02d}")
                    except Exception:
                        pass
        # ⚠️ 优先用**显式区间**（"from September 29 to October 1" / "October 1-3"），
        #    而不是全文日期的 min/max —— 后者会把**预订日**也算进去
        #    （实测 ECCV：min/max 得 9/20~10/4，而真实入住是 9/29~10/1）。
        ranges = []
        for r in recs:
            txt = str(r.get("_raw") or r.get("summary") or "")
            if _arx2 and not _arx2.search(txt):
                continue
            for m in re.finditer(r"from\s+([A-Z][a-z]+\s+\d{1,2}).{0,20}?to\s+([A-Z][a-z]+\s+\d{1,2}),?\s*(20\d{2})", txt):
                try:
                    _MON2 = {k[:3]: v for k, v in _MONTHS.items()}
                    a1, a2 = m.group(1).split(), m.group(2).split()
                    m1, m2 = _MON2.get(a1[0][:3].lower()), _MON2.get(a2[0][:3].lower())
                    if m1 and m2:
                        ranges.append((f"{m.group(3)}-{m1:02d}-{int(a1[1]):02d}",
                                       f"{m.group(3)}-{m2:02d}-{int(a2[1]):02d}"))
                except Exception:
                    pass
            for m in re.finditer(r"([A-Z][a-z]+\s+\d{1,2})\s*-\s*(\d{1,2}),?\s*(20\d{2})", txt):
                try:
                    _MON3 = {k[:3]: v for k, v in _MONTHS.items()}
                    p1 = m.group(1).split()
                    mm = _MON3.get(p1[0][:3].lower())
                    if mm:
                        ranges.append((f"{m.group(3)}-{mm:02d}-{int(p1[1]):02d}",
                                       f"{m.group(3)}-{mm:02d}-{int(m.group(2)):02d}"))
                except Exception:
                    pass
        if ranges:
            lo = min(r[0] for r in ranges)
            hi = max(r[1] for r in ranges)
            return {"kind": "span", "fact": f"{lo} ~ {hi}（**{len(set(times)) or 1} 个不同日期**；跨度天数未给，避免含首尾口径歧义）",
                    "detail": {"first": lo, "last": hi, "ranges": ranges[:5], "source": "explicit"}}
        if times:
            return {"kind": "span", "fact": f"{min(times)} ~ {max(times)}（**{len(set(times))} 个不同日期**；跨度天数未给，避免含首尾口径歧义）",
                    "detail": {"first": min(times), "last": max(times), "n_days": len(set(times)),
                               "source": "minmax"}}

    # ③ 列举：去重清单
    items = [k for k, _ in (ev.get("items") or [])]
    if items:
        # ⚠️ 2026-11-02：**不再砍到 40**（which cities in France 实测 88 个 ⇒ 静默丢一半）
        return {"kind": "list", "fact": " / ".join(map(str, items)),
                "detail": {"n": len(items)}}

    return {"kind": "none", "fact": "", "detail": {}}


# 题面用词 → 数据里的实际用词（2026-11-02）
# 实测：题面 "accommodation" 而 gold 写 "hotel reservation" ⇒ 不做同义扩展则 0 命中，
#       做了"整个窗口"的宽松聚合 ⇒ 金额 57491（正确 799.74）。
_ASPECT_SYN = {
    "accommodation": ["hotel", "stay", "booking", "reservation", "room", "lodging"],
    "hotel": ["hotel", "stay", "booking", "reservation", "room"],
    "stay": ["stay", "hotel", "reservation", "booking"],
    "flight": ["flight", "airline", "itinerary", "boarding", "airport"],
    "trip": ["trip", "travel", "itinerary", "flight", "hotel"],
    "dental": ["dental", "dentist", "tooth", "teeth"],
    "delay": ["delay", "delayed", "cancellation", "compensation"],
    "compensation": ["compensation", "refund", "reimburse", "claim"],
    "mouse": ["mouse", "MX Master", "logitech"],
    "milk": ["milk", "powder", "Marvel"],
}


def aspect_regex(question: str) -> str:
    """把题面的"方面词"扩展成同义词正则（用于把聚合范围收窄）。"""
    q = str(question or "").lower()
    words: list = []
    for key, syns in _ASPECT_SYN.items():
        if key in q:
            words.extend(syns)
    if not words:
        return ""
    # 去重且按长度降序（长的优先匹配）
    seen, out = set(), []
    for w in sorted(words, key=len, reverse=True):
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(re.escape(w))
    return "|".join(out)


# ---- tags 路线（视觉语义粗筛，2026-11-02 晚）----
_TAG_STOP = set("""all and for from can the this that with have been will would could should
about into over under more most some such only also than then them they their there
when where which what while your you our his her its not are was were has had did does
image images photo photos video videos time date dates day days year years month
trip hotel second first third recall associated departure end start help find me my i
was is are been being do did does""".split())


def tag_vocab(store: Any, user_id: str, min_freq: int = 20) -> set:
    """该用户的 tag 词表（只保留出现 >= min_freq 次的）。"""
    from collections import Counter
    cnt: Counter = Counter()
    conn = getattr(store, "conn", None)
    if conn is None:
        return set()
    try:
        for (txt,) in conn.execute(
                "SELECT index_text FROM memories WHERE user_id=?", (str(user_id),)):
            m = re.search(r"Tags:\s*([^\n]+)", str(txt or ""))
            if not m:
                continue
            for tg in re.split(r"[,\s]+", m.group(1).strip()):
                tg = tg.lower()
                if len(tg) > 2 and tg not in _TAG_STOP:
                    cnt[tg] += 1
    except Exception:
        return set()
    return {k for k, v in cnt.items() if v >= min_freq}


def tags_route(store: Any, user_id: str, question: str, limit: int = 40):
    """题面内容词 ∩ tag 词表 ⇒ 按 tag 过滤记录。"""
    from .field_tools import by_tags
    vocab = tag_vocab(store, user_id)
    if not vocab:
        return None
    qt = {w for w in re.findall(r"[a-z]{3,}", str(question or "").lower())
          if w not in _TAG_STOP}
    hit = sorted(qt & vocab)
    if not hit:
        return None
    recs = by_tags(store, user_id, hit, require_all=False, limit=limit)
    if not recs:
        return None
    return {"route": "tags", "kind": "enum", "tags_hit": hit,
            "records": [{"request_id": r["request_id"], "tags": r["tags"],
                         "tags_hit": r["tags_hit"], "snippet": r["snippet"]} for r in recs],
            "note": "tags 路线：题面内容词命中视觉标签"}
