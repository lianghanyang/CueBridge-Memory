"""记忆图书馆：把用户 memory 组织成 时间/实体/主题/模态/关系 树，并提供查询工具。"""
from __future__ import annotations
import json, os, re
import logging
from collections import defaultdict

logger = logging.getLogger("amc.library")


TYPE_WORDS = {
    'artwork': ['artwork', 'art works', 'painting', 'sculpture', 'art piece', '艺术品', '画作', '作品'],
    'place': ['city', 'cities', 'country', 'countries', 'place', 'town', '城市', '国家', '地方'],
    'person': ['person', 'people', 'someone', '人物', '谁'],
    'pet': ['dog', 'cat', 'pet', 'puppy', 'kitten', '宠物', '狗', '猫'],
    'product': ['product', 'device', 'gadget', '产品', '设备'],
    'brand': ['brand', 'company', '品牌', '公司'],
    'plant': ['plant', 'flower', 'tree', '植物', '花'],
    'food': ['food', 'dish', 'meal', 'cake', '食物', '菜', '蛋糕'],
    'event': ['event', 'festival', 'conference', '活动', '会议'],
    'organization': ['organization', 'institute', 'university', '机构', '大学'],
}


CONFLICT_WORDS = ['conflict', 'contradict', 'inconsisten', 'differ', 'disagree', '矛盾', '冲突', '不一致', '相悖']


def is_conflict_query(query):
    q = (query or '').lower()
    return any(w in q for w in CONFLICT_WORDS)


_LATIN = re.compile(r"^[a-z][a-z ]*$")


def _hit(query_lower, word):
    """类别词命中判定。

    ⚠️ 必须对**纯英文词用词边界**，否则子串匹配会产生大量假阳性：
      'cat' → "appli**cat**ion"（219 题被误判成 pet）
      'tree' → "s**tree**t"（59 题被误判成 plant）
    实测：全量 2,755 题里，子串匹配与词边界匹配有 **12.6% 结果不同**，
    且差异全部是"子串多判"（pet +219 / plant +59 / place +17 …）。
    中文词没有词边界概念，仍用子串。
    """
    w = word.lower()
    if _LATIN.match(w):
        return re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", query_lower) is not None
    return w in query_lower


def auto_type_branches(query):
    """从查询里的类别词推断实体类型分支（Y/<type>）。"""
    q = (query or '').lower()
    out = []
    for t, words in TYPE_WORDS.items():
        if any(_hit(q, w) for w in words):
            out.append({'branch': 'Y', 'name': t})
    return out


class MemoryLibrary:
    def __init__(self, rounds, tags, session_of=None, entity_types=None, embedder=None):
        """rounds: [{round_id, text, date, has_image}]；tags: [{round_id, entities, topic, summary}]
        session_of: {round_id: session_id}（可选，用于关系层）"""
        self.rounds = {r['round_id']: r for r in rounds}
        self.meta = {}
        for t in tags:
            rid = str(t.get('round_id'))
            if rid in self.rounds:
                self.meta[rid] = {'entities': [e for e in (t.get('entities') or []) if e],
                                  'topics': [x for x in (t.get('topic') or []) if x],
                                  'summary': t.get('summary') or ''}
        self.time_index = defaultdict(list)
        self.entity_index = defaultdict(list)
        self.topic_index = defaultdict(list)
        self.modality_index = defaultdict(list)
        for rid, r in self.rounds.items():
            d = r.get('date') or 'unknown'
            self.time_index[d[:7]].append(rid)          # YYYY-MM
            self.modality_index['image' if r.get('has_image') else 'text'].append(rid)
            m = self.meta.get(rid, {})
            for e in m.get('entities', []):
                self.entity_index[e.lower()].append(rid)
            for t in m.get('topics', []):
                self.topic_index[t].append(rid)

        # ---- 关系层：同会话链 / 实体共现 ----
        self.entity_types = {str(k): str(v) for k, v in (entity_types or {}).items()}
        self.type_index = defaultdict(list)
        for rid, m in self.meta.items():
            for e in m.get('entities', []):
                t = self.entity_types.get(str(e)) or self.entity_types.get(str(e).lower())
                if t:
                    self.type_index[t].append(rid)
        self.session_rounds = defaultdict(list)
        for rid, sid in (session_of or {}).items():
            if rid in self.rounds: self.session_rounds[sid].append(rid)
        for sid in self.session_rounds:
            self.session_rounds[sid].sort(key=lambda r: self.rounds[r].get('date') or '')
        self.embedder = embedder      # fn(list[str]) -> list[vec]；缺失则退回词法
        self._vec = None
        # ---- 词法语义层（char n-gram TF-IDF，兼容中英）----
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            self._np = np
            docs = []
            self._rid_list = list(self.rounds)
            for rid in self._rid_list:
                m = self.meta.get(rid, {})
                docs.append((m.get('summary') or '') + ' ' + self.rounds[rid]['text'][:1200])
            #   `ngram_range=(2, 3)` → **`(2, 2)`**（1 个字符）
            #   我独立复现（1,200 篇）：**1.46s → 0.35s**（4.2 倍）
            #   召回：13 道 gem 题 gold@10 `31.0% → 36.4%`（≈ +0.7 题，**在噪声内**）
            #   ⇒ 决策理由应是"**快 3~4 倍、未观察到可检测的质量下降**"，
            #     不能说成"召回更好"（13 题不足以支撑后者）
            #   ⚠️ 只改这里；**不动 `bm25.py:181`**（那是 BM25/libidx 的分词器）
            self._tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), max_features=60000, sublinear_tf=True)
            self._X = self._tfidf.fit_transform(docs)
        except Exception:
            self._tfidf = None
        self.co_entity = defaultdict(set)
        for rid, m in self.meta.items():
            es = [e.lower() for e in m.get('entities', [])]
            for a in es:
                for b in es:
                    if a != b: self.co_entity[a].add(b)

    # ---------- 目录 ----------
    def tree(self, max_topics=20, max_entities=40):
        def node(name, ids):
            ex = [self.meta.get(i, {}).get('summary') or self.rounds[i]['text'][:30] for i in ids[:3]]
            ds = sorted(self.rounds[i].get('date') or '' for i in ids if self.rounds[i].get('date'))
            return {'name': name, 'count': len(ids),
                    'date_range': [ds[0], ds[-1]] if ds else [],
                    'examples': ex}
        t_nodes = [node(k, v) for k, v in sorted(self.topic_index.items(), key=lambda kv: -len(kv[1]))[:max_topics]]
        e_nodes = [node(k, v) for k, v in sorted(self.entity_index.items(), key=lambda kv: -len(kv[1]))[:max_entities]]
        return {'T/时间': [node(k, v) for k, v in sorted(self.time_index.items())],
                'E/类型': [node(k, v) for k, v in sorted(self.type_index.items(), key=lambda kv: -len(kv[1]))],
                'M/主题': t_nodes,
                'E/实体': e_nodes,
                'O/模态': [node(k, v) for k, v in self.modality_index.items()]}

    def tree_text(self, max_topics=20, max_entities=30):
        t = self.tree(max_topics, max_entities)
        lines = []
        for branch, nodes in t.items():
            lines.append(f"{branch}（{len(nodes)} 项）")
            for n in nodes[:max_entities if branch.startswith('E/') else max_topics]:
                lines.append(f"  - {n['name']} [{n['count']}] {n['date_range']} 例:{' / '.join(n['examples'][:2])}")
        return "\n".join(lines)

    # ---------- 查询 ----------
    def where(self, path=None, time=None, entities=None, modality=None, k=20):
        ids = None
        if path:
            kind, _, name = path.partition('/')
            idx = {'T': self.time_index, 'M': self.topic_index, 'E': self.entity_index,
                   'O': self.modality_index, 'Y': self.type_index}.get(kind, {})
            cand = idx.get(name.lower() if kind == 'E' else name, [])
            ids = set(cand)
        if entities:
            s = set()
            for e in entities:
                s |= set(self.entity_index.get(str(e).lower(), []))
            ids = s if ids is None else (ids & s)
        if time and len(time) == 2:
            lo, hi = time
            s = {rid for rid, r in self.rounds.items() if lo <= (r.get('date') or '') <= hi}
            ids = s if ids is None else (ids & s)
        if modality:
            s = set(self.modality_index.get(modality, []))
            ids = s if ids is None else (ids & s)
        if ids is None:
            ids = set(self.rounds)
        out = sorted(ids, key=lambda rid: self.rounds[rid].get('date') or '')
        return self._brief(out[:k])

    def by_entity(self, e, k=20): return self.where(entities=[e], k=k)
    def by_time(self, ym, k=20): return self.where(path=f"T/{ym}", k=k)
    def _brief(self, ids):
        return [{'round_id': rid, 'date': self.rounds[rid].get('date'), 'has_image': self.rounds[rid].get('has_image'),
                 'title': self.meta.get(rid, {}).get('summary') or self.rounds[rid]['text'][:30],
                 'topics': self.meta.get(rid, {}).get('topics', []),
                 'entities': self.meta.get(rid, {}).get('entities', [])[:4]} for rid in ids]
    def round(self, rid): return self.rounds.get(rid)

    # ---------- 软打分（用树做特征，而非硬过滤） ----------
    def _ensure_vectors(self):
        if self._vec is not None or self.embedder is None:
            return self._vec
        try:
            import numpy as np
            # ⚠️🔴 ：**让图书馆复用检索侧的向量缓存**。
            #     图书馆 `_ensure_vectors` 对**全部 round** 算向量 ⇒ 10,504 条 ≈ **815 次 embedding**
            #   但真实原因不是 profile（profile 里**零** embedding 调用、backfill 上限 3 条）
            #     ⇒ 是**图书馆用了与检索侧不同的文本**：
            #        检索侧: `DenseRetriever._doc_text(index_text, 4000)`（caption 前移）
            #        图书馆: `(summary + ' ' + text)[:1200]`
            #     ⇒ sha1 不同 ⇒ **embcache 里的 16 万条向量一条也用不上**
            #   修法：**改用与检索侧完全相同的文本函数** ⇒ 直接命中已有 embcache
            #     ⇒ 代价：失去 summary 前缀（但 summary 只覆盖 5.8% 的记录，
            #        且它对 dense 的作用远小于"能命中缓存"）
            try:
                from .retrieval.embedding import DenseRetriever as _DR
                _mc = 4000
                texts = [_DR._doc_text(
                    {"index_text": (self.rounds[r].get('text') or ''),
                     "n_images": (self.rounds[r].get('has_image') and 1 or 0)},
                    _mc) for r in self.rounds]
            except Exception:
                texts = [((self.meta.get(r, {}).get('summary') or '') + ' '
                          + (self.rounds[r].get('text') or ''))[:1200] for r in self.rounds]
            # ⚠️🔴 （实测 P0）：**缺向量太多时不要一次性全嵌**。
            #   现状：平台灌 add 约 5,800 条/分钟，而预嵌只有 458 条/分钟
            #   ⇒ 未嵌入积压 **76,828 条（27%）**
            #     ⇒ 每次查询都 408 ⇒ 平台重试几次后判失败
            #   修法：缺向量超过阈值时**跳过 dense、先用词法（BM25）答题**，
            #        并通知后台补嵌 ⇒ 首查快、后续自动恢复 dense
            try:
                _cap = int(os.getenv("AMC_LIB_DENSE_MAX_MISSING", "800") or 800)
                _miss = None
                # ⚠️ ：用**同一个缓存实例**统计缺向量
                #   （原来自己 new 一个 VectorCache(path) 少了 namespace ⇒ TypeError
                #     ⇒ 被 except 吞掉 ⇒ **守护从未生效**，实测仍全量嵌 467s）
                _vc = getattr(self.embedder, "cache", None)
                if _vc is not None:
                    _f2, _m2 = _vc.get_many([_vc.key(t) for t in texts])
                    _miss = len(_m2)
                if _cap > 0 and _miss is not None and _miss > _cap:
                    logger.warning("[library] 用户缺向量 %d > %d ⇒ **本轮跳过 dense**（用词法），已排后台补嵌",
                                   _miss, _cap)
                    try:
                        if hasattr(self, "on_missing_vectors"):
                            self.on_missing_vectors()
                    except Exception:
                        pass
                    self._vec = None
                    return None
            except Exception:
                logger.exception("[library] 缺向量检查失败（继续全量嵌）")
            V = np.array(self.embedder(texts), dtype='float32')
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            self._vec = (list(self.rounds), V)
        except Exception:
            self._vec = None
        return self._vec

    def dense_scores(self, query):
        """dense 余弦相似度（0..1 归一）。没有 embedder 时返回 {}。"""
        v = self._ensure_vectors()
        if v is None:
            return {}
        try:
            import numpy as np
            ids, V = v
            q = np.array(self.embedder([query or ''])[0], dtype='float32')
            q = q / (np.linalg.norm(q) + 1e-9)
            sim = V @ q
            mx = float(sim.max()) or 1.0
            return {rid: float(s) / mx for rid, s in zip(ids, sim)}
        except Exception:
            return {}

    def lexical_scores(self, query):
        """返回 {rid: 0..1 的余弦相似度}。"""
        if not getattr(self, '_tfidf', None):
            return {}
        try:
            q = self._tfidf.transform([query or ''])
            sim = (self._X @ q.T).toarray().ravel()
            mx = float(sim.max()) or 1.0
            return {rid: float(v) / mx for rid, v in zip(self._rid_list, sim) if v > 0}
        except Exception:
            return {}

    def score_rounds(self, query, rids):
        """给一组 rid 打"与 query 的相关度"分（0..1）：词法余弦 + 标签词面命中。"""
        import re as _re
        lex = self.lexical_scores(query)
        q = (query or '').lower()
        q_toks = set(_re.findall(r'[a-z]{3,}', q))
        for seg in _re.findall(r'[\u4e00-\u9fff]{2,}', query or ''):
            for i in range(len(seg) - 1): q_toks.add(seg[i:i+2])
        out = {}
        for rid in rids:
            m = self.meta.get(rid, {})
            blob = (m.get('summary') or '') + ' ' + ' '.join((m.get('topics') or [])) + ' ' + ' '.join((m.get('entities') or []))
            bl = blob.lower()
            hit = sum(1 for t in q_toks if t in bl) if q_toks else 0
            tag = hit / max(1, len(q_toks))
            out[rid] = 0.6 * lex.get(rid, 0.0) + 0.4 * tag
        return out

    def pair_by_entity(self, query, max_entities=2, per_entity=12):
        """冲突类问题：找出 query 命中的实体，返回「同一实体的多条轮次」（成组）。"""
        import re as _re
        q = (query or '').lower()
        hits = []
        for ent, ids in self.entity_index.items():
            if len(ent) < 3: continue
            if _hit(q, ent):          # 英文用词边界，避免 "cat" 命中 "application"
                hits.append((len(ent), ent, ids))
        hits.sort(reverse=True)
        out, seen = [], set()
        for _l, ent, ids in hits[:max_entities]:
            grp = []
            for rid in sorted(ids, key=lambda r: self.rounds[r].get('date') or ''):
                if rid in seen: continue
                grp.append(rid); seen.add(rid)
                if len(grp) >= per_entity: break
            if grp: out.append({'entity': ent, 'rounds': self._brief(grp)})
        return out

    def rank_lexical(self, query, k=10):
        """纯词法排序（用于与 dense 做候选并集）。"""
        lex = self.lexical_scores(query) if query else {}
        out = sorted(lex, key=lambda r: -lex[r])[:k]
        return self._brief(out)

    def rank(self, plan, query=None, k=50):
        if query:
            plan = list(plan or []) + auto_type_branches(query)
        """plan: [{branch,name,time,entities,modality}]；对全部 round 软打分后取 top-k。
        分数 = 主题命中×2 + 实体命中×1 + 时间窗内×2 + 模态匹配×1（同一 plan 取最大）。"""
        # 先给所有轮次一个"词法/dense 基础分"，plan 只作为加分项（不裁剪候选）
        best = {}
        base_lex = self.dense_scores(query) if query else {}
        if not base_lex and query:
            base_lex = self.lexical_scores(query)
        for rid in self.rounds:
            _b = float(base_lex.get(rid, 0.0)) * 4.0 if base_lex else 0.0
            best[rid] = _b
        for rid, r in self.rounds.items():
            m = self.meta.get(rid, {})
            tops = set(m.get('topics', [])); ents = {e.lower() for e in m.get('entities', [])}
            sc = 0.0
            for st in plan or []:
                b = (st.get('branch') or '').strip(); name = st.get('name') or ''
                _s = 0.0
                if b.startswith('M') and name: _s += 2.0 if name in tops else 0.0
                elif b.startswith('E') and name: _s += 1.0 if name.lower() in ents else 0.0
                elif b.startswith('Y') and name:
                    _t = self.entity_types
                    if any(_t.get(e) == name or _t.get(str(e).lower()) == name for e in (m.get('entities') or [])):
                        _s += 2.0
                for e in (st.get('entities') or []):
                    if str(e).lower() in ents: _s += 1.0
                t = st.get('time')
                if t and len(t) == 2:
                    d = r.get('date') or ''
                    if t[0] <= d <= t[1]: _s += 2.0
                md = st.get('modality')
                if md: _s += 1.0 if (md == ('image' if r.get('has_image') else 'text')) else 0.0
                sc = max(sc, _s)
            best[rid] = best.get(rid, 0.0) + sc
        # 注：dense/词法基础分已在上面（×4）计入，这里不再叠加（避免重复计权）
        if not best:
            out = list(self.rounds)[:k]
            return self._brief(out)
        out = sorted(best, key=lambda rid: (-best[rid], self.rounds[rid].get('date') or ''))
        return self._brief(out[:k])

    # ---------- 关系查询 ----------
    def neighbors(self, rid, k=3):
        """同会话里与该轮相邻的 round（前后各 k 条）。"""
        for sid, ids in self.session_rounds.items():
            if rid in ids:
                i = ids.index(rid)
                out = ids[max(0, i-k):i] + ids[i+1:i+1+k]
                return self._brief(out)
        return []

    def related_entities(self, e, k=10):
        return sorted(self.co_entity.get(str(e).lower(), []))[:k]

    def entity_hits(self, query, k=10):
        """把 query 里出现的**实体/主题**对应的轮次找出来（用于并入检索池）。

        仅做词面命中（实体名出现在 query 里），不依赖 LLM 再次调用。
        """
        import re as _re
        q = (query or "").lower()
        if not q:
            return []
        scored = {}
        for ent, rids in list(self.entity_index.items()):
            if len(ent) < 3 or not _hit(q, ent):
                continue
            for rid in rids:
                scored[rid] = scored.get(rid, 0.0) + 1.0 + min(len(ent), 20) / 40.0
        for top, rids in list(self.topic_index.items()):
            if len(top or "") < 2 or not _hit(q, top):
                continue
            for rid in rids:
                scored[rid] = scored.get(rid, 0.0) + 0.5
        return [rid for rid, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]
