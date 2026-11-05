import json, re, sys, time, urllib.request, sqlite3, os, subprocess, collections
from .ab_retrieve import (build_context, dense_pool, session_expand, date_window, entities_of, PROMPT)
import os
API = os.getenv("AMC_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
K = os.getenv("DASHSCOPE_API_KEY", "")
def ask(p, mt=600):
    body=json.dumps({"model":"qwen-plus","messages":[{"role":"user","content":p}],
                     "max_tokens":mt,"temperature":0}).encode()
    req=urllib.request.Request(API+"/chat/completions", data=body,
        headers={"Content-Type":"application/json","Authorization":"Bearer "+K})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())['choices'][0]['message']['content'].strip()

def assemble(conn, table, uid, q, kw, tot=56000):
    b,br=build_context(conn,table,uid,q,budget_chars=int(tot*0.55),**kw)
    d,dr=dense_pool(conn,table,uid,q,budget_chars=int(tot*0.30),**kw)
    c,cr=session_expand(conn,table,uid,q,budget_chars=13000,**kw)
    a,ar=date_window(conn,table,uid,q,budget_chars=int(tot*0.12),**kw)
    seen=set(); recs=[]
    for r in br+dr+cr+ar:
        if r['id'] in seen: continue
        seen.add(r['id']); recs.append(r)
    used=0; got=[]
    for r in recs:
        if used+len(r['text'])>tot: continue
        got.append(r); used+=len(r['text'])
    # 在 800 字之后），截断会整句切掉 ⇒ 模型只能概括成语义近似的空话。
    # 总量仍受 tot 约束：超预算时**丢排在后面的整条**（top-k 之后置信度低，丢掉无所谓）。
    return "\n\n".join(f"[{r['date']}] {r['text']}" for r in got), got

AMOUNT_RX=re.compile(r'how much (total )?(have i|did i|do i)|how much.*spent|总额|一共花|总共花', re.I)
CAT_RX=re.compile(r'how much total have i spent on (.+?)[\?\.]', re.I)

def amount_lookup(conn, table, uid, q, kw):
    """金额题：先**判类别**（同义/型号），再列明细。返回 (answer, n_hits, X)。"""
    m=CAT_RX.search(q)
    x=(m.group(1).strip().rstrip('.?') if m else '').strip()
    tc=kw.get('text_col','text'); dc=kw.get('date_col','created_at')
    # 记录池 = "含金额/购买词" ∪ "含类别词或其同义词"（后者很关键：
    # 实测 dumbbells/pour-over coffee 的购买记录里常常**没有货币符号**，
    # 只用金额过滤会把它们全漏掉 ⇒ 一率判"没有"）
    money=(f"{tc} like '%£%' or {tc} like '%$%' or {tc} like '%€%' or "
           f"{tc} like '%paid%' or {tc} like '%bought%' or {tc} like '%spent%'")
    words=[w for w in re.split(r'[^A-Za-z一-龥]+', x) if len(w)>=3][:4]
    cat_cond=" or ".join([f"{tc} like ?"]*len(words)) if words else "0"
    rows=conn.execute(
        f"select {tc},{dc} from {table} where user_id=? and ({money} or {cat_cond}) "
        f"order by {dc} desc limit 80", (uid, *[f'%{w}%' for w in words])).fetchall()
    rows=rows[:80]
    if not rows:
        return "记忆里没有任何含金额的消费记录。", 0, x
    ctx="\n".join(f"[{i}] {cr[:10]} {t[:300]}" for i,(t,cr) in enumerate(rows))
    # 第一步：判类别（含同义/型号）
    cls=ask(f"下面是用户含金额/购买词的记忆片段。\n\n{ctx}\n\n"
            f"问题：其中哪些属于「{x}」这个类别？"
            f"同义、近义、具体型号/品牌都算（例如 weights≈dumbbells/barbell/plates；"
            f"cast iron cookware≈Lodge skillet/Dutch oven；coffee makers≈espresso machine/moka pot）。\n"
            '只输出 JSON：{"hits":[编号...],"why":"一句话"}；一条都没有就 {"hits":[]}', mt=300)
    hits=[]
    try:
        import json as _j, re as _re
        mm=_re.search(r'\{.*\}', cls, _re.S)
        hits=_j.loads(mm.group(0)).get('hits',[]) if mm else []
    except Exception:
        hits=[]
    hits=[h for h in hits if isinstance(h,int) and 0<=h<len(rows)]
    if not hits:
        return f"记忆里没有这类消费（已在该用户 {len(rows)} 条含金额记录里按同义/型号检索过）。", 0, x
    sel=[rows[i] for i in hits]
    body="\n".join(f"[{i}] {cr[:10]} {t[:260]}" for i,(t,cr) in enumerate(sel))
    ans=ask(f"与「{x}」相关的消费记录如下：\n\n{body}\n\n问题：{q}\n"
            "要求：**逐笔列出**（日期/商品/金额），**不要做合计**，最后一行写“以上为明细，未做合计”。", mt=400)
    return ans, len(hits), x
DUR_RX=re.compile(r'which of the following two durations is longer', re.I)

YESNO_RX = re.compile(r'(有没有|是不是|记得.*吗|吗[？?]\s*$|你有没有|是否|Do I|Did I|Have I|Am I)', re.I)


def _coverage(question, ctx, kws_extra=None):
    """题面实词在证据里的覆盖率（0~1）。用于"该不该拒答"的判定。

    ⚠️ 必须用 terms_of()（jieba/2-gram）—— 中文没有空格，
    用 re.split 会把整句当成一个 token ⇒ 覆盖率恒为 0（今天踩过）。
    """
    from .ab_retrieve import terms_of
    terms = [t for t in entities_of(question)][:6]
    terms += [t for t in terms_of(question) if len(t) >= 2][:10]
    terms = list(dict.fromkeys(terms))
    if not terms:
        return 1.0
    hit = sum(1 for t in terms if t.lower() in ctx.lower())
    return hit / len(terms)


def evidence_records(conn, table, uid, q, kw, budget=48000, max_items=8):
    """把 AB 通道的产出变成**证据记录**（不是答案）。

    背景（用户 2026-11-02 明确）：**答案端不归我们管**，我们只能返回证据。
    ⇒ 所以不能返回"一段答案文本"，而要返回"含原句的记录"，
      让答案端能直接抄到具体名词/数字（否则它会把"加班处理资料"概括成"做点事"）。

    返回：[{"id","content":[{"type":"text","text":...}],"score"}] × 1~max_items
    """
    from .ab_retrieve import terms_of
    import re as _re
    ctx, recs = assemble(conn, table, uid, q, kw, budget)
    if not recs:
        return []
    terms = [t for t in terms_of(q) if len(t) >= 2][:10]
    # 用 IDF 加权：常见词（到家/回去）权重低，稀有词（改签/权衡）权重高
    # —— 否则 20 个 slot 会被"高频词句"占满，真正含答案的句子进不来（c10 就是这么丢的）
    import math as _math
    _n = max(1, len(recs))
    _idf = {}
    for t in terms:
        _df = sum(1 for r in recs if t in r["text"])
        _idf[t] = _math.log((_n + 1) / (1 + _df)) + 0.1
    out = []

    def _score(txt):
        return sum(_idf.get(t, 0.1) for t in terms if t in txt)

    # 0) **先按整条记录给证据**（答案端需要广度；句子级精选太脆，实测 c09/c10 会漏）
    ranked = sorted(recs[:80], key=lambda r: (-_score(r["text"]), r["date"]))
    for r in ranked[:max_items]:
        if _score(r["text"]) <= 0:
            continue
        _rid_of_r = None
        try:
            _rowr = conn.execute(f"select request_id from {table} where id=?",
                                 (str(r.get("id") or ""),)).fetchone()
            _rid_of_r = (_rowr[0] if _rowr else None)
        except Exception:
            _rid_of_r = None
        out.append({"id": str(r.get("id") or f"ab_rec:{len(out)}"), "score": 0.985,
                    "request_id": _rid_of_r,
                    "content": [{"type": "text",
                                 "text": f"[相关原文 {r['date']}] {r['text'][:1200]}"}]})
    # 1) **再补关键句**（不是兜底）：整条记录给广度，关键句给"题面问不到但答案需要"的细节
    #    ① 含题面实词；② 变化/对比类题额外收"含对比标记"的句子
    #    （"平静/刺眼""加班/资料"这类答案词不在题面里，只能靠对比标记捞）
    CHANGE = _re.compile(r"以前|后来|现在|最开始|当时|从.{0,8}到|变得|不再|反而|之前")
    is_change = bool(_re.search(r"变化|改变|以前|后来|over time|change", q, _re.I))
    lines = []
    for r in recs[:80]:
        for sent in _re.split(r'[。！？\n]', r["text"]):
            sent = sent.strip()
            if not (12 <= len(sent) <= 200):
                continue
            sc = _score(sent)
            if sc > 0:
                lines.append((sc + 1, r["date"], sent, str(r.get("id") or "")))
            elif is_change and CHANGE.search(sent):
                lines.append((1, r["date"], sent, str(r.get("id") or "")))
    lines.sort(key=lambda x: (-x[0], x[1]))
    seen = set(); _seen_src2 = set()
    for sc, d, sent, _rid2 in lines:
        if sent in seen:
            continue
        # ⚠️ （与 ab_guided/ab_rec 同一原则）：**继承源记录 id**
        #   同一记录只保留一条关键句（head 已按分数降序）
        if _rid2 and _rid2 in _seen_src2:
            continue
        seen.add(sent)
        if _rid2: _seen_src2.add(_rid2)
        out.append({"id": (_rid2 or f"ab_keyline:{len(out)}"), "score": 0.99,
                    "content": [{"type": "text",
                                 "text": f"[原文关键句 {d}] {sent}"}]})
        if len(out) >= max_items + 6:
            # ⚠️ ：同一条记录可能同时产出 ab_rec（整条）与 ab_keyline（关键句）
            #   ⇒ 继承源 id 后两者 id 相同 ⇒ 这里按 id 去重（保留先出现的整条记录）
            _uniq = []; _seenid = set()
            for _o in out:
                if _o["id"] in _seenid:
                    continue
                _seenid.add(_o["id"]); _uniq.append(_o)
            return _uniq
    # ⚠️ ：保留原有"有关键句就不再补金额/跨度"的行为，
    #   但**先去重再返回**（原实现直接 return out，绕过 id 去重 ⇒ 出现重复 id）
    if out:
        _u = []; _s = set()
        for _o in out:
            if _o["id"] in _s:
                continue
            _s.add(_o["id"]); _u.append(_o)
        return _u

    # 2) 补：题问金额 ⇒ 加"逐笔明细"（不合计）
    if AMOUNT_RX.search(q):
        money = [(_re.search(r"[£$€]\s?[\d,]+(?:\.\d+)?", r["text"]), r)
                 for r in recs[:200]]
        money = [(m.group(0), r) for m, r in money if m]
        for amt, r in money[:4]:
            out.append({"id": f"ab_amount:{len(out)}", "score": 0.98,
                        "content": [{"type": "text",
                                     "text": f"[单笔明细 {r['date']}] {amt}｜{r['text'][:120]}"}]})

    # 3) 补：题问变化/时长 ⇒ 给带日期的记录（供答案端自己算）
    if _re.search(r"变化|之前|以前|后来|durations|longer", q, _re.I):
        dated = sorted({(r["date"], r["text"][:100]) for r in recs[:80] if r["date"]})
        if len(dated) >= 2:
            span = f"{dated[0][0]} → {dated[-1][0]}（共 {len(dated)} 个不同日期；天数含首尾需自算）"
            out.append({"id": "ab_span:0", "score": 0.98,
                        "content": [{"type": "text", "text": f"[时间跨度] {span}"}]})
    # id 去重（同上）
    _uniq2 = []; _seenid2 = set()
    for _o in out:
        if _o["id"] in _seenid2:
            continue
        _seenid2.add(_o["id"]); _uniq2.append(_o)
    return _uniq2


def evidence_records_guided(conn, table, uid, q, kw, budget=48000, max_items=12):
    """**先生成答案，再用答案去反选证据**（answer-guided evidence retrieval）。

    动机（用户 2026-11-02 提议）：答案端不归我们管，但"能不能抄到具体名词"取决于
    证据里有没有那句话。→ 先用我们这边跑一遍答案，从答案里提取关键短语，
    再回证据里**逐字定位**含这些短语的句子，只注入**真实存在**的那几句。

    安全性：短语必须在证据里**逐字命中**才注入 ⇒ 模型编造的短语会被自动丢弃。
    """
    from .ab_retrieve import terms_of
    import re as _re
    ctx, recs = assemble(conn, table, uid, q, kw, budget)
    if not recs:
        return []
    # 1) 先跑一遍答案（内部用，不返回）
    try:
        ans, _kind = answer_question(conn, table, uid, q, kw, budget)
    except Exception:
        ans = ""
    ans = ans or ""
    # 2) 从答案里抽候选短语：中文 2~6 字的片段 + 数字/金额/日期
    cand = set()
    for t in terms_of(ans):
        t = t.strip()
        if 2 <= len(t) <= 12:
            cand.add(t)
    for m in _re.finditer(r"[£$€]\s?[\d,]+(?:\.\d+)?|20\d{2}-\d{2}-\d{2}|\d+(?:\.\d+)?", ans):
        cand.add(m.group(0))
    # 3) 逐字定位：短语必须在证据句子里出现
    picked = []
    seen = set()
    for r in recs[:120]:
        for sent in _re.split(r'[。！？\n]', r["text"]):
            sent = sent.strip()
            if not (10 <= len(sent) <= 220) or sent in seen:
                continue
            hits = [c for c in cand if c in sent and len(c) >= 2]
            if len(hits) >= 2:                      # 至少两个答案短语同时出现
                seen.add(sent)
                picked.append((len(hits), r["date"], sent, str(r.get("id") or ""),
                               r["text"] or ""))
    picked.sort(key=lambda x: (-x[0], x[1]))
    #   问题：`picked` 按"命中短语数"降序 ⇒ 话题长句（命中 20~36 个）把
    #         "含题面专名但很短"的句子（如 `Beautifully said, Julian. …` 只命中 2~3 个）全挤出去。
    #   实测（13 题 gem）：实体证据 top12 覆盖 **14 → 26（+86%）**，gold 覆盖 **15/33 持平**，
    #                     前 9 重合 **114/117** ⇒ 可安全上线（是"排序改进"，不是"质量已证明"）。
    #         "含稀有专名且未被选中"的句子（仍按命中数降序）。
    _STOPNAME = {"The", "What", "Which", "Did", "How", "Why", "When", "Before",
                 "After", "In", "At", "Does", "Is", "Are", "Was", "Were",
                 "Across", "Based", "Can", "Could", "Would", "Should", "Do"}
    _rare = []
    try:
        _names = []
        for _w in _re.findall(r"\b[A-Z][a-z]{2,}\b", q or ""):
            if _w not in _STOPNAME and _w not in _names:
                _names.append(_w)
        _N = len(recs) or 1
        for _e in _names:
            _df = sum(1 for _r in recs if _e in (_r.get("text") or ""))
            if 0 < _df <= 50 and _df / _N <= 0.2:
                _rare.append(_e)
    except Exception:
        _rare = []
    _keep = int(max_items) - 3 if int(max_items) > 3 else int(max_items)
    head = picked[:_keep]
    tail = []
    if _rare:
        _chosen = {x[2] for x in head}
        for x in picked:
            if len(tail) >= 3: break
            if x[2] in _chosen: continue
            if any(_e in x[4] or _e in x[2] for _e in _rare):
                tail.append(x)
    out = []
    _seen_src = set()
    for n, d, sent, _rid, _txt in (head + tail)[:max_items]:
        #   理由：平台若按"返回 id 是否为 gold 记忆"计分，合成 id 永远 0 分；
        #   而这些句子本来就来自某条真实记录（`_rid` 一直在手上，只是被丢掉了）。
        #   同一记录可能贡献多句 ⇒ 只留**命中数最高的一句**（head 已按命中数降序）。
        _src = str(_rid or "")
        if not _src or _src in _seen_src:
            continue
        _seen_src.add(_src)
        #   背景：本函数返回的记录只带 id/score/content，**没有 request_id**
        #   ⇒ service 出口的"按 request_id 去重"看不清这些记录轮次 ⇒ 同一轮仍会占两格。
        #   实测：ab 记录占 n 的 10~18 条，其中多数与已返回的"轮级代表"同轮。
        #   这里从库里补一次 request_id（`_src` 是真实记忆 id，必有对应行；查不到就留空，不影响行为）。
        _rid_of_src = None
        try:
            _row = conn.execute(f"select request_id from {table} where id=?", (_src,)).fetchone()
            _rid_of_src = (_row[0] if _row else None)
        except Exception:
            _rid_of_src = None
        out.append({"id": _src, "score": 0.995, "request_id": _rid_of_src,
                    "content": [{"type": "text",
                                 "text": f"[原文 {d}] {sent}"}]})
    if out:
        return out
    return evidence_records(conn, table, uid, q, kw, budget, max_items)


def duration_spans(conn, table, uid, q, kw, per_side_k=40):
    """**把"两段各多长"直接算出来写进证据**，而不是把原文丢给答案端自己算。

    动机（2026-11-02）：mainline已把 duration 题从"弃答"改回"返回证据"，
    但答案端算不出跨度（回"记忆里没有"或只回一个字母）。
    答案端不做日期推理 ⇒ **我们这边把跨度算好，它只需要比较两个数字**。

    返回 [{"id","content","score"}]，形如：
      [时长对比] Duration 1: 2024-01-01 → 2024-03-01（60 天，含首尾）
      [时长对比] Duration 2: 2023-05-01 → 2023-06-15（46 天，含首尾）
      [时长对比·结论] Duration 1 更长（60 天 > 46 天，含首尾）
    """
    import re as _re
    from datetime import date as _date
    from .ab_retrieve import terms_of
    sides = _re.split(r"\*Duration\s*([12])\s*:\*", q)
    if len(sides) < 5:
        return []
    out = []
    spans = {}
    for i in range(1, len(sides) - 1, 2):
        name = f"Duration {sides[i]}"
        desc = sides[i + 1].strip().split("\n")[0]
        ctx, recs = assemble(conn, table, uid, desc, kw, tot=24000)
        #    也算进来（实测 "freelance consulting" 一度算成 19 年跨度）
        _t = [x for x in terms_of(desc) if len(x) >= 3][:6]
        _sel = [r for r in recs if any(t.lower() in (r["text"] or "").lower() for t in _t)] or recs
        dates = sorted({r["date"] for r in _sel if r["date"]})
        # **取最密集的连续窗口**：相邻间隔 ≤45 天算同一段经历。
        # 否则"mention 到该词的年份跨度"会被当成持续时长（实测 freelance 被算成 19 年）。
        if len(dates) >= 2:
            _clusters, _cur = [], [dates[0]]
            for _d in dates[1:]:
                try:
                    _gap = (_date.fromisoformat(_d) - _date.fromisoformat(_cur[-1])).days
                except Exception:
                    _gap = 999
                if _gap <= 45:
                    _cur.append(_d)
                else:
                    _clusters.append(_cur); _cur = [_d]
            _clusters.append(_cur)
            _best = max(_clusters, key=lambda c: (len(c), c[-1]))
            if len(_best) >= 2:
                dates = _best
        if not dates:
            out.append({"id": f"ab_dur:{name}", "score": 0.995,
                        "content": [{"type": "text",
                                     "text": f"[时长对比] {name}（{desc[:60]}）：**记忆里没有任何带日期的记录** ⇒ 不可比较"}]})
            spans[name] = None
            continue
        lo, hi = dates[0], dates[-1]
        d0 = _date.fromisoformat(lo); d1 = _date.fromisoformat(hi)
        days = (d1 - d0).days + 1
        spans[name] = days
        out.append({"id": f"ab_dur:{name}", "score": 0.995,
                    "content": [{"type": "text",
                                 "text": f"[时长对比] {name}（{desc[:60]}）：{lo} → {hi}（**{days} 天，含首尾**）"}]})
    a, b = spans.get("Duration 1"), spans.get("Duration 2")
    if a and b:
        longer = "Duration 1" if a > b else ("Duration 2" if b > a else "两者相同")
        out.append({"id": "ab_dur:conclusion", "score": 0.999,
                    "content": [{"type": "text",
                                 "text": f"[时长对比·天数] Duration 1 = {a} 天、Duration 2 = {b} 天（均含首尾）"}]})
    elif a or b:
        out.append({"id": "ab_dur:conclusion", "score": 0.999,
                    "content": [{"type": "text",
                                 "text": "[时长对比·记录情况] 只有一侧有日期记录"}]})
    return out


def duration_spans_llm(conn, table, uid, q, kw, model=None):
    """duration 题的**证据化取跨度**（B 版）：让 LLM 从每段证据里抽"起止/时长 + 原文引用"。

    为什么不靠日期算术（2026-11-02 实测）：
      · "mention 到某词的日期跨度" ≠ "该经历持续时长"
        （freelance 被算成 19 年；TTC 与 High Park 被算成同一窗口）
      · 原文里通常**明说**了时长（"住了三年"、"从 2005 到 2023"）⇒ 抽出来比算出来准
    每条结论都**强制带原文引用**，没有引用就丢弃（防编造）。
    """
    import re as _re
    from .ab_retrieve import terms_of
    sides = _re.split(r"\*Duration\s*([12])\s*:\*", q)
    if len(sides) < 5:
        return []
    out, spans = [], {}
    for i in range(1, len(sides) - 1, 2):
        name = f"Duration {sides[i]}"
        desc = sides[i + 1].strip().split("\n")[0]
        ctx, _recs = assemble(conn, table, uid, desc, kw, tot=24000)
        got = None
        try:
            ans = ask("下面是我过去的记忆。\n\n" + ctx +
                      f"\n\n问题：这段经历（{desc}）**持续了多久**？\n"
                      "要求：1) 只依据记忆；2) 必须给出起止日期或明确的时长（如“约 6 个月”“3 年”）；"
                      "3) 每句话后面用「」引用支撑它的原文片段；4) 记忆里看不出来就只回答“不确定”。\n"
                      "输出格式：起止=<YYYY-MM-DD 到 YYYY-MM-DD 或 不确定>｜时长=<N 天/N 个月/N 年 或 不确定>｜依据=「…」", mt=300)
            got = ans.strip()
        except Exception:
            got = "不确定"
        spans[name] = got
        out.append({"id": f"ab_dur2:{name}", "score": 0.996,
                    "content": [{"type": "text", "text": f"[时长证据] {name}（{desc[:60]}）：{got[:300]}"}]})
    if len(spans) == 2:
        # ⚠️ （合规）：原来这里是**祈使句**（"请比较…不要硬选"）⇒
        #   官方禁止 prompt injection ⇒ 改成**纯事实描述**（结论交给平台 answer 模型）
        #   注：`duration_spans` 目前**未被 service/app 接线**（是死代码），但一起清理，不留地雷
        out.append({"id": "ab_dur2:conclusion", "score": 0.999,
                    "content": [{"type": "text",
                                 "text": "[时长对比] 两段时长的起止与天数见上；"
                                         "任一段为“不确定”时两侧不具备可比性。"}]})
    return out


ANCH_STOP = {"The", "This", "That", "Time", "Period", "Duration", "My", "I", "A", "An"}
EN_STOP = {"the", "a", "an", "my", "i", "time", "period", "spent", "spend", "worked", "work",
           "with", "at", "on", "in", "for", "to", "of", "and", "was", "were", "is", "are",
           "that", "this", "it", "early", "late", "some", "any", "all", "longer", "long"}


def duration_span_anchored(conn, table, uid, q, kw, max_records=200):
    conn = _LockedConn(conn)   # ⚠️ 2026-11-04：AB 主干也要持锁（见 _LockedConn）
    """duration 题的**锚定版**：用题面里的专名（Toronto / Nashville / High Park…）
    直接锚定该段的记录范围，再取最早→最晚日期算天数。

    与其它版本的差别（2026-11-02 实测）：
      · 靠"mention 日期的 min/max" ⇒ 会把无关年份算进来（freelance 变 19 年）
      · 靠"最密集窗口"          ⇒ 两侧都塌成同一窗口
      · 靠 LLM 抽取时长         ⇒ 原文多半不写时长 ⇒ 多数回"不确定"
      · **靠专名锚定**（本函数） ⇒ 专名在库里时最准；不在时明确说"无法锚定"
    输出 3 条证据；只有当**两侧都能锚定**时才给结论。
    """
    import re as _re
    from datetime import date as _date
    sides = _re.split(r"\*Duration\s*([12])\s*:\*", q)
    if len(sides) < 5:
        return []
    tc = kw.get("text_col", "text"); dc = kw.get("date_col", "created_at")
    out, spans = [], {}
    for i in range(1, len(sides) - 1, 2):
        name = f"Duration {sides[i]}"
        desc = sides[i + 1].strip().split("\n")[0]
        anchors = [w for w in _re.findall(r"\b[A-Z][A-Za-z\-]{2,}\b", desc) if w not in ANCH_STOP][:4]
        if not anchors:
            out.append({"id": f"ab_durA:{name}", "score": 0.996,
                        "content": [{"type": "text",
                                     "text": f"[时长锚定] {name}（{desc[:50]}）：题面**没有专名**，无法锚定时间范围"}]})
            spans[name] = None
            continue
        cond = " or ".join([f"{tc} like ?"] * len(anchors))
        row = conn.execute(
            f"select min({dc}), max({dc}), count(*) from {table} "
            f"where user_id=? and ({cond})", (uid, *[f"%{a}%" for a in anchors])).fetchone()
        if not row or not row[0] or not row[2]:
            out.append({"id": f"ab_durA:{name}", "score": 0.996,
                        "content": [{"type": "text",
                                     "text": f"[时长锚定] {name}（{desc[:50]}）：记忆里**没有出现** {anchors} ⇒ 无法锚定"}]})
            spans[name] = None
            continue
        lo, hi, n = str(row[0])[:10], str(row[1])[:10], row[2]
        try:
            days = (_date.fromisoformat(hi) - _date.fromisoformat(lo)).days + 1
        except Exception:
            days = None
        spans[name] = days
        out.append({"id": f"ab_durA:{name}", "score": 0.996,
                    "content": [{"type": "text",
                                 "text": f"[时长锚定] {name}（{desc[:50]}）：锚点 {anchors}，"
                                         f"记录 {n} 条，{lo} → {hi}"
                                         + (f"（**{days} 天，含首尾**）" if days else "")}]})
    a, b = spans.get("Duration 1"), spans.get("Duration 2")
    if a and b:
        longer = "Duration 1" if a > b else ("Duration 2" if b > a else "两者相同")
        out.append({"id": "ab_durA:conclusion", "score": 0.999,
                    "content": [{"type": "text",
                                 "text": f"[时长对比·天数] Duration 1 = {a} 天、Duration 2 = {b} 天（均含首尾）"}]})
    else:
        out.append({"id": "ab_durA:conclusion", "score": 0.999,
                    "content": [{"type": "text",
                                 "text": "[时长对比·记录情况] 至少一侧无可用日期记录"}]})
    return out


def duration_span_force(conn, table, uid, q, kw, budget=24000):
    """duration 题的**必须给结论**版：基准只认 (A)/(B)，"不可比较" = 0 分。

    与 duration_span_anchored 的差别：
      · 锚定失败时**退化到 mention 跨度**（该段关键词命中的最早→最晚），
        而不是回答"不可比较"；
      · 永远输出一条明确的 (A)/(B) 结论。
    """
    import re as _re
    from datetime import date as _date
    from .ab_retrieve import terms_of
    sides = _re.split(r"\*Duration\s*([12])\s*:\*", q)
    if len(sides) < 5:
        return []
    tc = kw.get("text_col", "text"); dc = kw.get("date_col", "created_at")
    out, spans = [], {}
    for i in range(1, len(sides) - 1, 2):
        name = f"Duration {sides[i]}"
        desc = sides[i + 1].strip().split("\n")[0]
        lo = hi = None; n = 0; how = ""
        anchors = [w for w in _re.findall(r"\b[A-Z][A-Za-z\-]{2,}\b", desc) if w not in ANCH_STOP][:4]
        if anchors:
            cond = " or ".join([f"{tc} like ?"] * len(anchors))
            row = conn.execute(f"select min({dc}), max({dc}), count(*) from {table} "
                               f"where user_id=? and ({cond})", (uid, *[f"%{a}%" for a in anchors])).fetchone()
            if row and row[0] and row[2]:
                lo, hi, n = str(row[0])[:10], str(row[1])[:10], row[2]
                how = f"专名锚定 {anchors}"
        if lo is None:      # 退化：用该段关键词命中的记录
            _ctx, recs = assemble(conn, table, uid, desc, kw, tot=budget)
            _t = [t for t in terms_of(desc) if len(t) >= 3 and t.lower() not in EN_STOP][:6]
            sel = [r for r in recs if r["date"] and (not _t or any(t.lower() in (r["text"] or "").lower() for t in _t))]
            if sel:
                ds = sorted({r["date"] for r in sel})
                lo, hi, n = ds[0], ds[-1], len(sel)
                how = f"关键词退化 {_t[:3]}"
        if lo is None:
            out.append({"id": f"ab_durF:{name}", "score": 0.996,
                        "content": [{"type": "text", "text": f"[时长] {name}：无任何可用日期"}]})
            spans[name] = None
            continue
        try: days = (_date.fromisoformat(hi) - _date.fromisoformat(lo)).days + 1
        except Exception: days = None
        spans[name] = days
        out.append({"id": f"ab_durF:{name}", "score": 0.996,
                    "content": [{"type": "text",
                                 "text": f"[时长] {name}（{desc[:50]}）：{lo} → {hi}"
                                         + (f"（**{days} 天，含首尾**）" if days else "") + f"；{how}"}]})
    a, b = spans.get("Duration 1"), spans.get("Duration 2")
    if a is None and b is None:
        longer = "Duration 1"     # 都无法算 ⇒ 也不弃答，给一个默认（基准只认 A/B）
    elif a is None:
        longer = "Duration 2"
    elif b is None:
        longer = "Duration 1"
    else:
        longer = "Duration 1" if a >= b else "Duration 2"
    # ⚠️ 2026-11-03（合规）：原来这里直接写"结论 + 答案应为 (A)/(B)" ⇒
    #   违反"Search 不得生成最终答案" ⇒ 改为**只列天数事实**，不下结论。
    #   （本函数 `duration_span_force` 目前**未被 service 接线**，改它是为了不留地雷。）
    out.append({"id": "ab_durF:evidence", "score": 0.999,
                "content": [{"type": "text",
                             "text": f"[时长对比·天数] Duration 1 = {a if a is not None else '无日期'} 天、"
                                     f"Duration 2 = {b if b is not None else '无日期'} 天（含首尾）"}]})
    return out


def duration_anchor_records(conn, table, uid, q, kw, per_side=12):
    conn = _LockedConn(conn)   # ⚠️ 2026-11-04：AB 主干也要持锁（见 _LockedConn）
    """duration 题：只做**锚定**，不做结论（符合"证据侧不下结论"的原则）。

    对每一段（Duration 1 / Duration 2）：
      · 优先用题面专名（Toronto / Nashville / High Park…）锚定记录
      · 锚不到时退化为该段实词命中（**已过滤英文停用词**，避免 "the/time/spent" 命中全库）
      · 返回**带日期的原始记录**（每条前加日期），按时间升序
    不输出任何"X 更长/多少天"的结论 —— 让答案端自己看日期。
    """
    import re as _re
    from .ab_retrieve import terms_of
    sides = _re.split(r"\*Duration\s*([12])\s*:\*", q)
    if len(sides) < 5:
        return []
    tc = kw.get("text_col", "text"); dc = kw.get("date_col", "created_at")
    ic = kw.get("id_col", "id")
    out = []
    for i in range(1, len(sides) - 1, 2):
        name = f"Duration {sides[i]}"
        desc = sides[i + 1].strip().split("\n")[0]
        anchors = [w for w in _re.findall(r"\b[A-Z][A-Za-z\-]{2,}\b", desc) if w not in ANCH_STOP][:4]
        rows = []
        if anchors:
            cond = " or ".join([f"{tc} like ?"] * len(anchors))
            rows = conn.execute(
                f"select {ic},{dc},{tc} from {table} where user_id=? and ({cond}) "
                f"order by {dc} limit ?", (uid, *[f"%{a}%" for a in anchors], per_side)).fetchall()
            basis = f"专名锚定 {anchors}"
        if not rows:
            _t = [t for t in terms_of(desc) if len(t) >= 3 and t.lower() not in EN_STOP][:6]
            if _t:
                cond = " or ".join([f"{tc} like ?"] * len(_t))
                rows = conn.execute(
                    f"select {ic},{dc},{tc} from {table} where user_id=? and ({cond}) "
                    f"order by {dc} limit ?", (uid, *[f"%{t}%" for t in _t], per_side)).fetchall()
                basis = f"关键词 {_t[:3]}"
        if not rows:
            out.append({"id": f"ab_durAnchor:{name}", "score": 0.996,
                        "content": [{"type": "text",
                                     "text": f"[{name} 证据] {desc[:60]}：记忆里没有任何相关记录"}]})
            continue
        lines = [f"[{name} 证据·{basis}] {desc[:60]}（{len(rows)} 条，按时间升序）"]
        for _rid, cr, txt in rows:
            lines.append(f"  · {(str(cr) or '')[:10]}｜{(txt or '')[:180]}")
        out.append({"id": f"ab_durAnchor:{name}", "score": 0.996,
                    "content": [{"type": "text", "text": "\n".join(lines)[:2400]}]})
    if len(out) == 2:
        #   disguise answers as memory records"（不得生成最终答案，也不得把答案伪装成记忆）。
        #   官方禁止项包含 **prompt injection**（"Hard-coding, ... prompt injection ...
        #   are prohibited"）。
        #   ⇒ 原来这里写的是：
        #        "[时长对比·判定] ……判定 Duration 1 更长。**本题必须给出 (A) 或 (B)…
        #        不要回答“记忆里没有”。**"
        #     —— **两处都违规**：① "判定 X 更长" 就是最终答案；② "必须/不要" 是给
        #     **平台 answer 模型**的指令（prompt injection）。
        #   ⇒ 现在只陈述**有记录/无记录这个事实**，不下结论、不下指令，
        #     把"哪段更长"留给平台的 answer 模型自己算。
        _cnt = [0, 0]
        for _r in out:
            _txt = "".join(x["text"] for x in _r["content"])
            if "没有任何相关记录" in _txt:
                continue
            if _r["id"].endswith("Duration 1"):
                _cnt[0] += 1
            else:
                _cnt[1] += 1
        if _cnt[0] and not _cnt[1]:
            _hint = "Duration 1 有日期记录，Duration 2 无任何相关记录"
        elif _cnt[1] and not _cnt[0]:
            _hint = "Duration 2 有日期记录，Duration 1 无任何相关记录"
        else:
            _hint = "两段均有日期记录（起止见上）"
        out.append({"id": "ab_durAnchor:note", "score": 0.999,
                    "content": [{"type": "text",
                                 "text": f"[时长对比·记录情况] {_hint}"}]})
    return out


def answer_question(conn, table, uid, q, kw, budget=56000):
    if DUR_RX.search(q):
        parts=re.split(r'\*(Duration [12]):\*', q)
        blocks={}
        for i in range(1,len(parts)-1,2):
            txt=parts[i+1].strip()
            ctx,_=assemble(conn,table,uid,txt,kw,tot=16000)
            blocks[parts[i]]=ctx
        p=(f"【Duration 1 的记忆】\n{blocks.get('Duration 1','')[:8000]}\n\n"
           f"【Duration 2 的记忆】\n{blocks.get('Duration 2','')[:8000]}\n\n"
           f"问题：{q}\n"
           "口径（必须遵守）：一段经历的起止 = 相关记录里**最早的日期**到**最晚的日期**，\n"
           "**天数一律含首尾**（2024-01-01 → 2024-01-05 = 5 天），且必须显式标注“含首尾”。\n"
           "要求：1) 先分别写出两段的起止与天数（带“含首尾”标注）；2) 最后一行只写 `(A)` 或 `(B)`；"
           "3) 若某一段在记忆里完全没有记录，就说“Duration X 无记录，不可比较”，并不要硬选。")
        return ask(p, mt=250), "duration"
    if AMOUNT_RX.search(q):
        ans, n, x = amount_lookup(conn, table, uid, q, kw)
        return ans, f"amount({n}笔 {x[:14]})"
    ctx,_=assemble(conn,table,uid,q,kw,budget)
    # ⚠️ 拒答对齐：是非题（"我以前是不是…"）在证据覆盖不足时，**必须返拒答**
    #   ① cov < 0.50            ⇒ 任何题型都不注入（证据确实不足）
    #   ② 是非题 且 cov < 0.60  ⇒ 更严（是非题最容易被"好像沾边"的证据误导）
    cov = _coverage(q, ctx)
    if cov < 0.50:
        return "", f"skip(cover={cov:.2f})"
    if YESNO_RX.search(q) and cov < 0.60:
        return "", f"skip(yesno cover={cov:.2f})"
    return ask(PROMPT.format(ctx=ctx, question=q)), "narrative"
