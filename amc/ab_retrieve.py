#!/usr/bin/env python3
"""A/B 穷举召回（离线原型）：不依赖向量库，只用 SQL LIKE + jieba。

两路并集：
  A 时间窗：题面含显式日期 ⇒ 该日 ±2 天全部记录（不排序、不截断）
  B 主题子集：jieba 分词 → DF 过滤（1 ≤ df ≤ 0.6n）→ **每个词配额取 4 条**
合并后按 IDF 加权分排序，在字符预算内**全量**给出（不按 top_k 截断）。

用法：
    from ab_retrieve import build_context
    ctx, recs = build_context(conn, table, user_id, question, budget_chars=42000)
"""
from __future__ import annotations
import os as _os
import re, math

# ---- 热开关（**不需要重启**）：改文件内容即可 ----
_FLAG_FILE = _os.getenv("AMC_AB_FLAG_FILE", "data/store/ab_flag.txt")
_mt_cache = {"t": 0.0, "v": False}
import time as _time_ab


def _lk(conn):
    """AB 通道用 store 的锁（同一把）——防止 16 个慢线程并发用同一条 sqlite 连接。"""
    try:
        from .store import lock_for
        return lock_for(conn)
    except Exception:
        import threading
        return threading.RLock()


def _q_all(conn, sql, args=()):
    with _lk(conn):
        return conn.execute(sql, args).fetchall()


def _q_one(conn, sql, args=()):
    with _lk(conn):
        return conn.execute(sql, args).fetchone()


def ab_enabled() -> bool:
    """逐请求读开关文件；mtime 未变时用缓存（最多 2s 延迟）。

    为什么不用 env：**/add 阶段不能重启**，env 只能启动时读；
    而文件开关可以在平台灌数据期间随时打开/关闭（零窗口）。
    文件内容：`1` / `on` / `true` ⇒ 开；其它或文件不存在 ⇒ 关。
    """
    now = _time_ab.time()
    if now - _mt_cache["t"] < 1.0:      # 1 秒内复用（避免每请求都读盘）
        return _mt_cache["v"]
    _mt_cache["t"] = now
    try:
        with open(_FLAG_FILE, "r", encoding="utf-8", errors="ignore") as fh:
            raw = (fh.read() or "").strip().lower()
        _mt_cache["v"] = raw in ("1", "on", "true", "yes")
    except Exception:
        _mt_cache["v"] = False
    return _mt_cache["v"]

try:                      # mainline运行环境（vllm env）没有 jieba ⇒ 退化为"中文 2-gram + 英文整词"
    import jieba
    def _tok(s):
        return list(jieba.cut(s))
except Exception:
    def _tok(s):
        out = []
        for chunk, is_cn in re.findall(r'([\u4e00-\u9fff]+)|([A-Za-z]+)', s):
            if chunk:
                out += [chunk[i:i+2] for i in range(len(chunk)-1)] or [chunk]
            else:
                out.append(is_cn)
        return out

STOPW = set("什么 怎么 最后 现在 当时 主要 自己 我们 他们 这个 那个 一次 一直 已经 "
            "可以 应该 问题 处理 事情 时候 哪些 哪个 多少 为什么".split())


ENT_STOP = set("Does Do Did Is Are Was Were The This That What Which Who When Where Why How Can Could "
               "Should Would Will If In On At For From With And But Or Not No Yes".split())


def n_entities(q: str) -> int:
    """题面里的实体个数（多实体题要放大配额）：英文大写词 + 中文书名号/引号内专名。"""
    ents = set()
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9-]{1,}\b", q):
        w = m.group(0)
        if w in ENT_STOP:
            continue
        if m.start() == 0:          # 句首大写通常是普通词
            continue
        ents.add(w)
    for m in re.finditer(r"[「『\"《]([^」』\"》]{2,12})[」』\"》]", q):
        ents.add(m.group(1))
    return len(ents)


def terms_of(q: str) -> list[str]:
    ts = []
    for w in _tok(q):
        w = w.strip()
        if len(w) < 2 or w in STOPW:
            continue
        if re.fullmatch(r"[\d\W]+", w):
            continue
        ts.append(w)
    return list(dict.fromkeys(ts))


def build_context(conn, table, user_id, question, budget_chars=42000,
                  text_col="text", id_col="id", date_col="created_at", role_col="role"):
    n = _q_one(conn, f"select count(*) from {table} where user_id=?", (user_id,))[0]
    if not n:
        return "", []
    terms = terms_of(question)
    df = {}
    for t in terms:
        d = _q_one(conn, f"select count(*) from {table} where user_id=? and {text_col} like ?",
                   (user_id, f"%{t}%"))[0]
        if 1 <= d <= max(3, n * 0.6):
            df[t] = d
    if not df:
        return "", []
    idf = {t: math.log(n / max(1, d)) for t, d in df.items()}
    # 配额词：IDF 最高的 12 个 ∪ **所有低频/实体词**（df ≤ max(5, 5%n)）
    # 后者很关键：像 "Egypt"(df=261/10504) 这种中频实体，按纯 IDF 排会被挤出 top-8，
    # 结果整题查不到 —— 2026-11-02 实测踩到。
    # **自适应**：多实体题才用"宽配额"（IDF前12 ∪ 低频实体词，配额 2×实体数）；
    # 单实体/叙事题用"窄配额"（IDF前8，配额 4）—— 实测宽配额会稀释叙事题的上下文，
    # 使 c05/c07/c09 从"对"退化到"错"（2026-11-02）。
    _ne = max(1, n_entities(question))
    if _ne >= 2:
        rare = {t for t, d in df.items() if d <= max(5, n * 0.05)}
        top = list(dict.fromkeys(sorted(idf, key=lambda t: -idf[t])[:12] + sorted(rare, key=lambda t: -idf[t])))
    else:
        top = sorted(idf, key=lambda t: -idf[t])[:8]

    # **配额随实体数增长**：多实体题（Lena/Amy/Lucy）每词 4 条取不全 ⇒ max(4, 2×实体数)
    quota = max(4, 2 * _ne) if _ne >= 2 else 4
    picked = {}
    for t in top:                      # 逐词配额：保证每个 facet 都有代表
        rows = _q_all(conn,
            f"select {id_col},{date_col},{role_col},{text_col} from {table} "
            f"where user_id=? and {text_col} like ? order by {date_col} desc limit 400",
            (user_id, f"%{t}%"))
        scored = sorted(
            ((sum(idf[x] for x in top if x in (txt or "")), cr, role, rid, txt or "")
             for rid, cr, role, txt in rows), key=lambda z: (-z[0], z[1]))[:quota]
        for s, cr, role, rid, txt in scored:
            picked[rid] = (s, cr, role, txt)

    rows = _q_all(conn,
        f"select {id_col},{date_col},{role_col},{text_col} from {table} "
        f"where user_id=? order by {date_col} desc limit 3000", (user_id,))
    for rid, cr, role, txt in rows:    # 再补全局高分
        s = sum(idf[x] for x in top if x in (txt or ""))
        if s > 0 and rid not in picked:
            picked[rid] = (s, cr, role, txt or "")

    recs = [{"id": rid, "date": (cr or "")[:10], "role": role, "text": txt, "score": round(s, 2)}
            for rid, (s, cr, role, txt) in picked.items()]
    recs.sort(key=lambda r: (-r["score"], r["date"]))
    got, used = [], 0
    for r in recs:
        if used + len(r["text"]) > budget_chars:
            continue
        got.append(r); used += len(r["text"])
    ctx = "\n\n".join(f"[{r['date']}] {r['text'][:800]}" for r in got)
    return ctx, got


def dense_pool(conn, table, user_id, question, budget_chars=20000,
               text_col="text", id_col="id", date_col="created_at", role_col="role"):
    """无配额的"密集候选池"：所有命中 top 词的记录按 IDF 加权分排序。

    为什么需要它（2026-11-02 实测）：配额保证"每个 facet 有代表"，但会**挤掉**
    只在单一词上高频命中的记录（如"工商"只出现在 2 条里）⇒ 叙事题会漏。
    配额池 + 密集池 取并集，才能同时保证"覆盖"与"深度"。
    """
    n = _q_one(conn, f"select count(*) from {table} where user_id=?", (user_id,))[0]
    if not n: return [], []
    terms = [t for t in terms_of(question)]
    df = {}
    for t in terms:
        d = _q_one(conn, f"select count(*) from {table} where user_id=? and {text_col} like ?",
                   (user_id, f"%{t}%"))[0]
        if 1 <= d <= max(3, n * 0.6): df[t] = d
    if not df: return [], []
    idf = {t: math.log(n / max(1, d)) for t, d in df.items()}
    top = sorted(idf, key=lambda t: -idf[t])[:10]
    cond = " or ".join([f"{text_col} like ?"] * len(top))
    rows = _q_all(conn,
        f"select {id_col},{date_col},{role_col},{text_col} from {table} "
        f"where user_id=? and ({cond}) order by {date_col} desc limit 4000",
        (user_id, *[f"%{t}%" for t in top]))
    recs = []
    for rid, cr, role, txt in rows:
        sc = sum(idf[t] for t in top if t in (txt or ""))
        if sc <= 0: continue
        recs.append({"id": rid, "date": (cr or "")[:10], "role": role,
                     "text": txt or "", "score": round(sc, 2), "via": "dense"})
    recs.sort(key=lambda r: (-r["score"], r["date"]))
    got, used = [], 0
    for r in recs:
        if used + len(r["text"]) > budget_chars: continue
        got.append(r); used += len(r["text"])
    return got, got


def date_window(conn, table, user_id, question, budget_chars=20000,
                text_col="text", id_col="id", date_col="created_at", role_col="role"):
    """A 路线：时间窗召回 —— 题面有日期线索，或能由实体锚定出日期区间。

    规则（按优先级）：
      1) 显式年月：2024-06 / June 2024 / 2024年6月 / 6月2日 ⇒ 该月 ± 7 天
      2) 实体锚定：题面实体（Egypt / Porto / 婚礼…）在库内出现的**日期区间** ± 3 天
         （"recent trip to Egypt" 这种，先靠实体找到行程，再用行程的起止取窗口）
      3) 都没有 ⇒ 返回空（交给 B/C）
    """
    months = re.findall(r"(20\d{2})[-/年](\d{1,2})", question)
    md = re.findall(r"(\d{1,2})月(\d{1,2})日", question)
    lo = hi = None
    if months:
        y, m = int(months[0][0]), int(months[0][1])
        lo = f"{y:04d}-{m:02d}-01"; hi = f"{y:04d}-{m:02d}-31"
    elif md:
        m, d = int(md[0][0]), int(md[0][1])
        rows = _q_one(conn,
            f"select min({date_col}), max({date_col}) from {table} "
            f"where user_id=? and {date_col} like ?", (user_id, f"%-{m:02d}-{d:02d}%"))
        if rows and rows[0]:
            lo, hi = rows[0][:10], rows[1][:10]
    if lo is None:
        ents = entities_of(question)
        spans = []
        for e in ents:
            r = _q_one(conn,
                f"select min({date_col}), max({date_col}) from {table} "
                f"where user_id=? and {text_col} like ?", (user_id, f"%{e}%"))
            if r and r[0]:
                spans.append((r[0][:10], r[1][:10]))
        if spans:
            lo = min(a for a, _ in spans); hi = max(b for _, b in spans)
    if lo is None:
        return [], []
    from datetime import date, timedelta
    def shift(d, n):
        y, m, dd = (int(x) for x in d.split("-"))
        return (date(y, m, dd) + timedelta(days=n)).isoformat()
    lo, hi = shift(lo, -3), shift(hi, 3)
    # 守卫：实体锚定的窗口如果跨度过大（>21 天），说明这个实体在库里到处出现，
    # 锚不住"某一次事件" ⇒ 不要用时间窗（否则会把大量无关记录灌进上下文）。
    _span = (date.fromisoformat(hi) - date.fromisoformat(lo)).days
    if _span > 21 and not (months or md):
        return [], []
    rows = _q_all(conn,
        f"select {id_col},{date_col},{role_col},{text_col} from {table} "
        f"where user_id=? and {date_col}>=? and {date_col}<=? order by {date_col}",
        (user_id, lo, hi))
    recs = [{"id": r[0], "date": (r[1] or "")[:10], "role": r[2], "text": r[3] or "",
             "score": 1.0, "via": "window"} for r in rows]
    got, used = [], 0
    for r in recs:
        if used + len(r["text"]) > budget_chars:
            continue
        got.append(r); used += len(r["text"])
    return got, got


def session_expand(conn, table, user_id, question, budget_chars=20000,
                   text_col="text", id_col="id", date_col="created_at", role_col="role"):
    """C 路线：题面实体 → 该实体的所有会话 → 会话内全部记录。

    为什么需要它：多实体/跨会话题（"Lena/Amy/Lucy 的宠物各叫什么"）里，
    gold 记录正文全是 "she/her"，人名只出现在**别的轮次**；纯词面匹配永远召回不到。
    """
    ents = entities_of(question)
    if not ents:
        return [], []
    sess = set()
    for e in ents:
        for (sid,) in _q_all(conn,
                f"select distinct session_id from {table} where user_id=? and {text_col} like ?",
                (user_id, f"%{e}%")):
            sess.add(sid)
    if not sess:
        return [], []
    recs = []
    for sid in sess:
        for rid, cr, role, txt in _q_all(conn,
                f"select {id_col},{date_col},{role_col},{text_col} from {table} "
                f"where user_id=? and session_id=?", (user_id, sid)):
            recs.append({"id": rid, "date": (cr or "")[:10], "role": role,
                         "text": txt or "", "score": 0.0, "via": "session"})
    recs.sort(key=lambda r: r["date"])
    got, used = [], 0
    for r in recs:
        if used + len(r["text"]) > budget_chars:
            continue
        got.append(r); used += len(r["text"])
    return got, got


def entities_of(q: str) -> list[str]:
    ents = []
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9-]{1,}\b", q):
        w = m.group(0)
        if w in ENT_STOP or m.start() == 0:
            continue
        ents.append(w)
    for m in re.finditer(r"[「『\"《]([^」』\"》]{2,12})[」』\"》]", q):
        ents.append(m.group(1))
    return list(dict.fromkeys(ents))


PROMPT = """下面是我过去的记忆（按相关度排序）。

【记录】
{ctx}

问题：{question}

作答要求（四条都要遵守）：
1) 只依据记忆；记忆里没有就直说“记忆里没有”。
2) **不要概括**：答案里必须出现记忆原文里的具体名词（对象/地点/商家/菜品/物品/银行/时间/数字）。
3) **逐条列细节**：把记忆里与问题相关的**所有具体细节**都列出来
   （环境类：声音/光线/座位/温度/气味；动机类：责任/划算/实际需要这类原话理由）。
4) 最后一行用一句话给结论。
"""
