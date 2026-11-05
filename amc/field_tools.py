"""字段感知抽取工具（2026-11-02）

来源：ATM 那 7 道"扩 k 也解不了"的题，top-k 命中 0/7，而本模块的两种方法拿到：
  · 列举型（which/where）：Q1 "Which European countries" ⇒ **6/6 答对**
  · 聚合型（how much/long）：Q3 住宿费 ⇒ **gold 2/2**；Q5 停留时长 ⇒ 直接给出日期区间

为什么有效（核心判断）：
```
答案不在"语义相似度"里，而在"字段值"或"时间/地点锚点"里。
  · 照片族记录：`[Image ...] Time: ... Location: ...` ⇒ "去过哪" = Location
  · 邮件族记录：`[Email ...] Time: ... Summary: ...`  ⇒ "花了多少/待多久" = Summary
⇒ 这些记录**几乎没有叙事文本** ⇒ 向量检索必然失败
```

三种"gold 与题目的关联方式"（决定了该用哪个工具）：
```
① **词面关联**   ⇒ 语义检索（传统 search）
② **字段关联**   ⇒ by_location_role / by_time_range（本模块）
③ **时间·地点间接关联** ⇒ event_anchor + by_anchor（本模块）
   —— 例："BMVC 2024 住宿花了多少"：酒店邮件里**不含 BMVC**，
      只能靠"Nov 2024 + Glasgow"关联
```
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 记录里的结构化字段（ATM 数据实际存在的）
_FIELD_RE = {
    "time": re.compile(r"Time:\s*([^\n]+)"),
    "location": re.compile(r"Location:\s*([^\n]+)"),
    "summary": re.compile(r"Summary:\s*([^\n]+)"),
    "amount": re.compile(r"(?:Amount|Price|Total|Cost):\s*([^\n]+)"),
}

_AMOUNT_RE = re.compile(r"(?:USD|EUR|GBP|CNY|RMB|JPY|HK\$|US\$|[$€£¥])\s*([0-9][0-9,]*\.?[0-9]*)|"
                        r"([0-9][0-9,]*\.?[0-9]*)\s*(?:USD|EUR|GBP|CNY|RMB|dollars?|euros?|pounds?)",
                        re.I)


def fields_of(text: str) -> Dict[str, str]:
    """从一条记录的 index_text 里抽出结构化字段。"""
    out: Dict[str, str] = {}
    t = str(text or "")
    for name, rx in _FIELD_RE.items():
        m = rx.search(t)
        if m:
            out[name] = m.group(1).strip()
    return out


def _iter_texts(store: Any, user_id: str) -> Iterable[Tuple[str, str]]:
    """(request_id, index_text)，**全量**遍历该用户的所有记录。

    ⚠️ 不要改用 `store.records_by_date(user_id, limit=...)`：
       它内部把 limit **夹到 500**（`min(500, limit)`），
       而 ATM 用户有 10,504 条 ⇒ 只扫到 4.8%，字段聚合会严重不足。
       （2026-11-02 实测：走 records_by_date 时 `by_location_role` 只拿到几个地名）
    ⇒ 因此这里**直连** `store.conn` 做全量查询。
    """
    conn = getattr(store, "conn", None)
    if conn is not None:
        try:
            cur = conn.execute("SELECT request_id, index_text FROM memories WHERE user_id=?",
                               (str(user_id),))
            for rid, txt in cur:
                yield str(rid), str(txt or "")
            return
        except Exception:
            pass
    # 退化：调用方传了带 `records_by_date` 的适配器（测试用）
    rowfn = getattr(store, "records_by_date", None)
    if rowfn is not None:
        try:
            for r in rowfn(user_id, limit=200000):
                yield str(r.get("request_id") or ""), str(r.get("index_text") or "")
        except Exception:
            return


def location_level(loc: str, level: str = "country") -> str:
    """从一条 Location（逗号分隔的地址串）里取**指定层级**的地名。

    2026-11-02 实测：同一份 Location 字符串，不同问题要**不同层**：
      · "which **countries**"      ⇒ 末段（United Kingdom / Italy / Thailand）
      · "which **cities**"         ⇒ 城市段（Boston / New Orleans / Mississauga）
      · "which **conferences**"    ⇒ **首段**（Allianz MiCo / CentralWorld…场馆名）
    ⇒ 所以层级必须由**问句**决定，不能写死。
    """
    parts = [p.strip() for p in str(loc or "").split(",") if p.strip()]
    if not parts:
        return ""
    lv = (level or "country").lower()
    if lv == "venue":                       # 场馆/机构名：首段
        return parts[0]
    if lv == "country":                     # 国家：末段（跳过邮编）
        for p in reversed(parts):
            if not re.fullmatch(r"[A-Z]{1,2}\d[\w\s]*|\d{4,6}(-\d{4})?", p):
                return p
        return parts[-1]
    if lv == "city":
        # 城市：从后往前找"不是邮编/不是省级/不是国家"的那段
        # 经验：末段=国家，末2常=邮编，末3=省/州，末4~5=城市
        for p in reversed(parts[:-1]):
            if re.fullmatch(r"[A-Z]{1,2}\d[\w\s]*|\d{4,6}(-\d{4})?", p):
                continue
            return p
        return parts[-2] if len(parts) >= 2 else parts[0]
    return parts[-1]


def by_location_role(store: Any, user_id: str, keep_outermost: bool = True,
                     level: str = "country"
                     ) -> Tuple[collections.Counter, int]:
    """列举型：抽所有 `Location:`，可选只取最外层地名（逗号分隔最后一段）。

    返回 (计数器, 带 Location 的记录数)。调用方再把计数器的 key 交给 LLM
    判"是否属于某类地理实体（国家/城市/大洲）"。

    实测：Q1 "Which European countries have I visited" ⇒ 4,206 条 Location
          → 18 个最外层地名 → LLM 判出 8 国 ⇒ **gold 6/6**
    """
    cnt: collections.Counter = collections.Counter()
    n = 0
    for _rid, txt in _iter_texts(store, user_id):
        f = fields_of(txt)
        loc = f.get("location")
        if not loc:
            continue
        n += 1
        if keep_outermost:
            key = location_level(loc, level or "country")
            if key:
                cnt[key] += 1
        else:
            cnt[loc] += 1
    return cnt, n


def event_anchor(store: Any, user_id: str, event_re: str) -> Dict[str, Any]:
    """事件锚定：从提到该事件的记录里抽 **日期集合 + 地点集合**。

    这是"按题面词过滤会漏 gold"的解法 —— 因为 gold 往往用"时间+地点"关联事件。

    实测：BMVC 的锚点 = {2024-11-25..28} + {Glasgow}；
          用该锚点过滤住宿邮件 ⇒ 11 条，**gold 2/2**（而按"BMVC"过滤 ⇒ 0/2）
    """
    rx = re.compile(event_re, re.I)
    dates: set = set()
    places: set = set()
    n = 0
    for _rid, txt in _iter_texts(store, user_id):
        if not rx.search(txt):
            continue
        n += 1
        for d in re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", txt):
            dates.add(d)
        f = fields_of(txt)
        loc = f.get("location") or ""
        for seg in [s.strip() for s in loc.split(",") if s.strip()]:
            # 只保留"像地名"的段（首字母大写、长度>2）
            if seg[:1].isupper() and len(seg) > 2:
                places.add(seg)
    return {"n_event_records": n, "dates": sorted(dates), "places": sorted(places)}


def by_anchor(store: Any, user_id: str, *, aspect_re: str = "",
              place: str = "", date_prefixes: Optional[List[str]] = None,
              limit: int = 200) -> List[Dict[str, str]]:
    """按"方面词 ∧ 地点 ∧ 时间前缀"过滤记录，返回其字段。

    `aspect_re` 例：`hotel|reservation|stay|booking`（住宿类）
    `place`      例：`Glasgow`
    `date_prefixes` 例：`["2024-11", "2024-12"]`
    """
    arx = re.compile(aspect_re, re.I) if aspect_re else None
    out: List[Dict[str, str]] = []
    for rid, txt in _iter_texts(store, user_id):
        if place and place not in txt:
            continue
        if arx and not arx.search(txt):
            continue
        f = fields_of(txt)
        if date_prefixes:
            tm = f.get("time") or ""
            if not any(tm.startswith(p) for p in date_prefixes):
                continue
        f["request_id"] = rid
        out.append(f)
        if len(out) >= max(1, int(limit)):
            break
    return out


def amounts_in(summaries: Iterable[str]) -> List[float]:
    """从文本里抽金额（用于"花了多少"类聚合）。

    ⚠️ **必须传全文，不能只传 Summary**（2026-11-02 实测）：
      gold 的住宿邮件 `Summary:` 里**没有金额**，金额只在正文
      （"The total cost is **655.59 GBP** per room"）。
      只传 Summary ⇒ 抽出 0 个金额。
    """
    vals: List[float] = []
    for s in summaries:
        for m in _AMOUNT_RE.finditer(str(s or "")):
            raw = (m.group(1) or m.group(2) or "").replace(",", "")
            try:
                vals.append(float(raw))
            except Exception:
                continue
    return vals


def by_location_with_sources(store: Any, user_id: str, level: str = "country",
                             top_n: int = 40) -> List[Dict[str, Any]]:
    """列举型：返回**实体 + 其来源记录**（不只实体名）。

    为什么必须带来源（2026-11-02）：
    ```
    · 只给"United Kingdom / France / Italy …" ⇒ 答案端知道"有哪些"
      但**不知道依据在哪**、也**无法核对**；而且无法用 request_id 度量覆盖
    · 给"实体 + 来源 request_id 列表" ⇒ 答案端可核对、可引用；也可度量
    ⇒ 这与"聚合要给构成、校验要给引用"是同一原则：**结论必须带依据**
    ```
    """
    conn = getattr(store, "conn", None)
    if conn is None:
        return []
    bucketed: Dict[str, List[str]] = {}
    for rid, txt in conn.execute(
            "SELECT request_id, index_text FROM memories WHERE user_id=?", (str(user_id),)):
        m = re.search(r"Location:\s*([^\n]+)", str(txt or ""))
        if not m:
            continue
        key = location_level(m.group(1).strip(), level or "country")
        if not key:
            continue
        bucket = bucketed.setdefault(key, [])
        # ⚠️ 2026-11-02：**不要在这里cap**。
        #    原来 `if len(bucket) < 30` ⇒ 每实体最多 30 条 ⇒
        #    gold 里排在 31+ 位的记录被静默丢掉（实测漏 2/6）。
        #    ⇒ 分桶阶段**全收**，截断只在输出阶段按预算做。
        bucket.append(str(rid))
    # ⚠️ 2026-11-02：来源**不能只给高频实体的前 N 条**。
    #    实测："Which European countries" 的 gold 是 **Budapest(3 条)/Compostela(93)** 这类低频记录，
    #    而 top 8 实体各取 30 条 = 240 条，**gold 6 条一条都没覆盖**。
    #    ⇒ 策略：**每个实体都给全部来源**（单实体上限放宽），并且**稀有实体优先分配预算**。
    items = sorted(bucketed.items(), key=lambda kv: len(kv[1]))     # **稀有优先**
    got, out = 0, []
    budget = max(100, int(top_n) * 40)                              # 总来源预算
    for k, v in items:
        if got >= budget:
            break
        take = v[:max(5, budget - got)]                             # 每个实体至少给 5 条
        got += len(take)
        out.append({"item": k, "n": len(v), "sources": take})
    out.sort(key=lambda d: -d["n"])                                 # 输出仍按频次降序
    return out[:max(1, int(top_n))] if top_n else out


# 常见国家/地区英文名（用于"which countries"类问题的**层级过滤**）
# 2026-11-02：实测答案模型会把 Paris/Amsterdam/Rome/Milan 当国家列出
#   ⇒ 因为 `location_level(country)` 对某些 Location 串返回了城市
#   ⇒ 修法：**用国家白名单做最后一道过滤**
_COUNTRIES = {
    "united kingdom", "uk", "england", "scotland", "wales", "northern ireland",
    "ireland", "france", "germany", "italy", "spain", "portugal", "netherlands",
    "belgium", "luxembourg", "switzerland", "austria", "denmark", "sweden",
    "norway", "finland", "iceland", "poland", "czech republic", "czechia",
    "slovakia", "hungary", "romania", "bulgaria", "greece", "croatia", "slovenia",
    "serbia", "bosnia and herzegovina", "montenegro", "albania", "north macedonia",
    "turkey", "russia", "ukraine", "estonia", "latvia", "lithuania", "malta",
    "cyprus", "united states", "usa", "united states of america", "canada",
    "mexico", "brazil", "argentina", "chile", "peru", "colombia", "china",
    "japan", "south korea", "korea", "india", "thailand", "vietnam", "singapore",
    "malaysia", "indonesia", "philippines", "australia", "new zealand",
    "egypt", "morocco", "south africa", "jordan", "israel", "united arab emirates",
    "uae", "qatar", "saudi arabia", "lebanon", "georgia", "armenia", "azerbaijan",
}


def is_country(name: str) -> bool:
    n = str(name or "").strip().lower()
    if not n:
        return False
    if n in _COUNTRIES:
        return True
    return False


def filter_countries(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把实体清单过滤到**只保留真正的国家**（用于 which countries 类问题）。"""
    return [it for it in (items or []) if is_country(it.get("item") or "")]


_AMOUNT_CUR_RE = re.compile(
    r"(?:(USD|EUR|GBP|CNY|RMB|JPY|HKD|AUD|CAD|THB|SGD)\s*([0-9][0-9,]*\.?[0-9]*))"
    r"|(?:([0-9][0-9,]*\.?[0-9]*)\s*(USD|EUR|GBP|CNY|RMB|JPY|HKD|AUD|CAD|THB|SGD))"
    r"|(?:([$€£¥])\s*([0-9][0-9,]*\.?[0-9]*))", re.I)
_SYM2CUR = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY/CNY"}


def amounts_with_currency(texts: Iterable[str]) -> List[Dict[str, Any]]:
    """抽金额**并标币种**（2026-11-02）。

    为什么必须带币种：
    ```
    实测：BMVC 住宿费的正解是 **GBP 799.74**
    但答案模型看到候选 [655.59, 144.15, 16.0, 11.9]（无币种）
      ⇒ 从同一批记录里挑了个 **THB 25,829.26** 作答
    ⇒ **候选值不带单位 ⇒ 答案端必然挑错**
    ```
    """
    out: List[Dict[str, Any]] = []
    for t in texts:
        for m in _AMOUNT_CUR_RE.finditer(str(t or "")):
            cur = m.group(1) or m.group(4) or _SYM2CUR.get(m.group(5) or "", "")
            raw = m.group(2) or m.group(3) or m.group(6) or ""
            try:
                out.append({"value": float(raw.replace(",", "")), "currency": (cur or "").upper()})
            except Exception:
                continue
    return out


def by_keywords(store: Any, user_id: str, keywords: List[str],
                require_all: bool = False, limit: int = 30) -> List[Dict[str, Any]]:
    """**关键词精确过滤**：返回记录中同时/任一命中给定关键词的那些记录。

    为什么需要（2026-11-02）：
    ```
    "I remember several moments when Grace was hiding in stairs" ⇒ 要"哪几张照片"
    实测：top-20 召回里**确实有 gold**（候选召回 91%），
          但答案模型引用的 4 个 ID **一个都没对上**（引用精度 0/4）
    ⇒ 因为 top-20 里有大量"相似但并不对"的照片
    ⇒ 需要的是**按"人+动作+地点"精确过滤**，而不是"相似度前 20"
    ```
    """
    kws = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not kws:
        return []
    conn = getattr(store, "conn", None)
    if conn is None:
        return []
    out: List[Dict[str, Any]] = []
    try:
        for rid, txt in conn.execute(
                "SELECT request_id, index_text FROM memories WHERE user_id=?", (str(user_id),)):
            t = str(txt or "")
            tl = t.lower()
            hit = [k for k in kws if k.lower() in tl]
            ok = (len(hit) == len(kws)) if require_all else bool(hit)
            if not ok:
                continue
            out.append({"request_id": str(rid), "hits": hit,
                        "snippet": t[:200],
                        "n_hits": len(hit)})
            if len(out) >= max(1, int(limit)):
                break
    except Exception:
        return out
    # 命中越多越靠前
    out.sort(key=lambda d: -d["n_hits"])
    return out


def by_tags(store: Any, user_id: str, want_tags: List[str],
            require_all: bool = False, limit: int = 40) -> List[Dict[str, Any]]:
    """**按视觉标签过滤**（`Tags:` 字段）。

    为什么重要（2026-11-02 重大发现）：
    ```
    "I remember several moments when Grace was hiding in strange places…"
    ⇒ gold 记录的全文里**没有 Grace 也没有 stairs**
    ⇒ 但它是**猫的照片**：`Tags: cat, long-haired, tail, bed, carpet, bedroom, sleeping…`
    ⇒ 即 "Grace" 是猫名，题目问的是"猫躲藏的瞬间"
    ⇒ **这类题的答案在 `Tags:` 字段里，不在文本里**
    ⇒ ATM 里 **4,205 条记录带 Tags，7,076 个不同标签**（cat 454 次、under-bed 等）
    ```
    """
    wants = [str(t).strip().lower() for t in (want_tags or []) if str(t).strip()]
    if not wants:
        return []
    conn = getattr(store, "conn", None)
    if conn is None:
        return []
    out: List[Dict[str, Any]] = []
    try:
        for rid, txt in conn.execute(
                "SELECT request_id, index_text FROM memories WHERE user_id=?", (str(user_id),)):
            t = str(txt or "")
            m = re.search(r"Tags:\s*([^\n]+)", t)
            if not m:
                continue
            have = {x.lower() for x in re.split(r"[,\s]+", m.group(1).strip()) if x}
            hit = [w for w in wants if w in have]
            ok = (len(hit) == len(wants)) if require_all else bool(hit)
            if not ok:
                continue
            out.append({"request_id": str(rid), "tags_hit": hit,
                        "tags": sorted(have)[:12],
                        "snippet": t[:160]})
            if len(out) >= max(1, int(limit)):
                break
    except Exception:
        return out
    out.sort(key=lambda d: -len(d["tags_hit"]))
    return out


# 记录里出现"某类实体"的**上下文提示词**（用于从文本抽实体名）
_ENTITY_CUES = {
    "hotel": ["hotel", "reservation", "stay at", "checked in", "booking"],
    "restaurant": ["restaurant", "dined", "dinner at", "lunch at", "reservation at"],
    "shop": ["shop", "store", "order", "purchased", "bought"],
    "conference": ["conference", "venue", "convention", "symposium", "workshop"],
}


def which_x_in_y(store: Any, user_id: str, entity_type: str, place: str = "",
                 limit: int = 25) -> List[Dict[str, Any]]:
    """**泛化的"which X in Y"**：在"提到 Y 的记录"里抽"X 类实体名"。

    这是把已经验证过的两类能力（"which hotels in Glasgow" ✅、"which countries" ✅）
    **泛化成一个工具**（2026-11-02）。

    做法：
      ① 过滤：记录文本里必须出现 `place`
      ② 提示：记录里必须出现该实体类的**上下文词**（hotel/reservation/…）
      ③ 抽取：从这些记录里抽**首字母大写的短语**（2~4 词），排除地名
      ④ 聚合：按出现次数排序
    """
    et = str(entity_type or "").lower()
    cues = _ENTITY_CUES.get(et, [et])
    conn = getattr(store, "conn", None)
    if conn is None:
        return []
    from collections import Counter
    cnt: Counter = Counter()
    src: dict = {}
    _dates: dict = {}          # name -> set(YYYY-MM-DD)，用于 distinct(date) 计数
    try:
        for rid, txt in conn.execute(
                "SELECT request_id, index_text FROM memories WHERE user_id=?", (str(user_id),)):
            t = str(txt or "")
            if place and place.lower() not in t.lower():
                continue
            tl = t.lower()
            if not any(c in tl for c in cues):
                continue
            # ⚠️ 2026-11-02：试过"剔除 Tags/OCR 行 + 折叠换行"，但**召回反而下降**
            #    （"Hilton Glasgow" 从 n=9 掉到不出现）——
            #    因为折叠后邻近窗口变小，更多实体过不了 ±120 字符的邻近性检查。
            #    ⇒ **保留原始文本**，只在抽取时拒绝含换行的实体名（下面处理）
            # 抽"连续 2~4 个首字母大写的词"，且**必须邻近上下文词**（±120 字符）
            #  —— 否则会抽出 "Glasgow Taxis" / "Customer Support" / "Jurassic World" 这类噪声
            cue_pos = [tl.find(c) for c in cues if c in tl]
            for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){1,3})\b", t):
                name = m.group(1).strip()
                if "\n" in name or "\r" in name:
                    continue          # 跨行实体名一律丢弃（"United Kingdom\nTags" 这类）
                n = len(name.split())
                if n < 2 or n > 4:
                    continue
                if is_country(name) or is_country(name.split()[0]):
                    continue
                if any(w.lower() in {"the", "this", "that", "and", "for", "from", "content",
                                     "support", "account", "world", "your"} for w in name.split()):
                    continue
                # 邻近性：实体名附近 ±120 字符内必须出现某个上下文词
                st0 = max(0, m.start() - 120)
                ctx = t[st0:m.end() + 120].lower()
                if not any(c in ctx for c in cues):
                    continue
                cnt[name] += 1
                src.setdefault(name, []).append(str(rid))
                #    一次入住可能产生 10 封邮件 ⇒ 记录数被"单次停留"放大。
                #    真正的偏好指标是 **distinct(date)**（去过几次）。
                _dm = re.search(r"Time:\s*(20\d{2}-\d{2}-\d{2})", t)
                if _dm:
                    _dates.setdefault(name, set()).add(_dm.group(1))
    except Exception:
        return []
    #    实测 "Hilton Glasgow" / "Hilton Glasgow Central" / "DOUBLETREE BY HILTON GLASGOW"
    #    是**同一家或同品牌同城**的不同写法 ⇒ 不聚类的话下游会以为有 3 家 ⇒ 答错。
    #    规则：① 一个名字包含另一个（"Hilton Glasgow" ⊂ "Hilton Glasgow Central"）
    #          ② 或共享 ≥2 个显著 token（排除 the/by/and/in/at 等）
    _STOP = {"the", "by", "and", "of", "in", "at", "for", "a", "an", "hotel", "restaurant",
             "glasgow", "london", "porto", "paris", "city", "central"}
    def _sig(name: str):
        toks = [w.lower() for w in re.split(r"[^0-9A-Za-z]+", name) if len(w) > 2]
        return [w for w in toks if w not in _STOP]
    items = cnt.most_common(max(1, int(limit)) * 4)     # 先多取，聚类后再截断
    clusters = []      # [{"canonical":str,"aliases":[...],"n":int,"sources":[...]}]
    for name, n in items:
        toks = set(_sig(name))
        placed = False
        for cl in clusters:
            ctoks = set(_sig(cl["canonical"])) | {t for a in cl["aliases"] for t in _sig(a)}
            a, b = name.lower(), cl["canonical"].lower()
            if (a in b or b in a) or len(toks & ctoks) >= 2:
                cl["aliases"].append(name); cl["n"] += n
                cl["sources"].extend(src.get(name, [])[:3])
                placed = True; break
        if not placed:
            clusters.append({"canonical": name, "aliases": [], "n": n,
                             "sources": list(src.get(name, [])[:5])})
    clusters.sort(key=lambda c: -c["n"])
    out = []
    for c in clusters[:max(1, int(limit))]:
        # 簇内所有写法/变体的日期并集 ⇒ **distinct(date)**（"去过几次"，不是"提及几次"）
        _ds = set()
        for _nm in [c["canonical"]] + list(c["aliases"]):
            _ds |= _dates.get(_nm, set())
        out.append({"entity": c["canonical"], "n": c["n"],
                    "n_dates": len(_ds),            # **偏好应用这个数**
                    "dates": sorted(_ds)[:12],
                    "aliases": c["aliases"], "n_variants": 1 + len(c["aliases"]),
                    "sources": c["sources"], "entity_type": et, "place": place})
    return out


_NATIONALITY = {"portuguese": "Portugal", "italian": "Italy", "french": "France",
                "spanish": "Spain", "german": "Germany", "dutch": "Netherlands",
                "british": "United Kingdom", "english": "England", "scottish": "Scotland",
                "chinese": "China", "japanese": "Japan", "thai": "Thailand",
                "egyptian": "Egypt", "greek": "Greece", "turkish": "Turkey",
                "american": "United States", "canadian": "Canada", "australian": "Australia"}
