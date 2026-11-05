"""后台打标：新 round 入库后异步打标签，供"记忆图书馆"补召回使用（失败不影响 /add 与 /search）。"""
from __future__ import annotations
import json, logging, os, queue, random, threading, time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TOPICS = ["工作学习","旅行","家庭","健康运动","饮食","消费理财","兴趣娱乐","科技信息","社交","其他"]
PROMPT = """你是记忆图书馆的编目员。给定一批对话轮次（每轮有编号和文本），为每一轮输出标签：
- entities: 具体实体（人物/品牌/产品/地点/组织/事件），最多 5 个，用原文词
- topic: 从这些类别选 1-2 个：__TOPICS__
- summary: ≤20 字标题
只输出 JSON 数组：{"i":<编号>,"entities":[...],"topic":[...],"summary":"..."}
轮次：
__ITEMS__
"""


class BackgroundTagger:
    def __init__(self, path: str, model: str = "gpt-4o-mini", base_url: str = "", api_key: str = "",
                 timeout: float = 30.0, workers: int = 1, model_name: str = None,
                 embed_base_url: str = "", embed_api_key: str = "", embed_model: str = "",
                 embed_dim: int = 0):
        self.embed_base_url = embed_base_url
        self.embed_api_key = embed_api_key
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self._embedder = None
        self.path = path
        #   `library_tags.json` 已长到 **61.5 MB**，而 `_save()` 每批都全量重写
        #   ⇒ 实测 **36 次/分钟** × 1.10s（dumps 0.66 + 写盘 0.43）= **39.6 秒/分钟的 CPU+IO**
        #   ⇒ 且 `_save()` 在 `with self.lock:` 里 ⇒ 16 个 tagger worker + /add 的 submit 全排队
        #   ⇒ **add 延迟飙到 5~10s** ⇒ 平台 19:35:42 报 `ADD_RUNTIME_ERROR`（它侧超时，我们其实都回了 200）
        #   修法：**脏标记 + 20s 节流 + dump 移出锁**
        self._dirty = False
        self._last_save = 0.0
        self._save_min_interval = float(__import__("os").getenv("AMC_TAG_SAVE_MIN_S", "20") or 20)
        self.model = model
        try:
            from amc.llm_endpoints import (endpoints as _eps, endpoint_tag as _etag,
                                           pick as _pick, demote as _demote, promote as _promote)
            self.endpoints = _eps(base_url or "", api_key or "", model or "")
            #   tagger 是**最大的 LLM 消费者**（实测 37 次/分钟），把它的流量从
            #   provider-a 分走一部分到 provider-b（实测流式 16 并发全 200，无 429；
            #   非流式则被限 **10 次/分钟**，所以必须走流式）。
            #   用**独立 env** `AMC_TAGGER_ENDPOINTS`，避免影响 search/evidence 的网关选择。
            _tg_raw = (__import__("os").getenv("AMC_TAGGER_ENDPOINTS") or "").strip()
            if _tg_raw:
                _tg = []
                for _spec in _tg_raw.split(";"):
                    _p = [x.strip() for x in _spec.split("|")]
                    if len(_p) >= 3 and _p[0] and _p[1] and _p[2]:
                        _tg.append({"base": _p[0].rstrip("/"), "key": _p[1], "model": _p[2]})
                if _tg:
                    self.endpoints = _tg
            self._ep_idx = 0
            self._etag = _etag
            self._pick, self._demote, self._promote = _pick, _demote, _promote
        except Exception:
            self.endpoints = [{"base": (base_url or "").rstrip("/"), "key": api_key or "", "model": model or ""}]
            self._ep_idx = 0
            self._etag = lambda e: (e.get("base") or "") + "/" + (e.get("model") or "")
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.q: queue.Queue = queue.Queue()
        self.data: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        # ⚠️ 图书馆缓存（2026-10-31 加）：`library_for()` 原来是**每次调用都重建**
        # MemoryLibrary —— 实测 540 轮 2.0s、10,490 轮 **11.75s**，
        # 而它在每个 /search 里都会被调用（service.py:1270）⇒ 单题 12~19s 的主因。
        # 用 per-user 版本号做缓存键：标签/轮次变化时 rev+1，缓存自然失效。
        self._lib_rev: Dict[str, int] = {}
        # ⚠️ 必须限量：单个 MemoryLibrary 实测 ~260MB（TF-IDF 矩阵 + dim=1024 向量），
        # AMC_LIB_CACHE_USERS 调整。命中时 move_to_end，超限淘汰最久未用的。
        import collections as _coll
        self._lib_cache: "OrderedDict[str, Any]" = _coll.OrderedDict()
        self._lib_cache_max = max(1, int(__import__("os").getenv("AMC_LIB_CACHE_USERS", "48") or 48))
        self._load()
        self.workers = [threading.Thread(target=self._loop, daemon=True) for _ in range(max(1, workers))]
        for w in self.workers: w.start()
        # ⚠️ ：**唯一的落盘线程**（每 5s 检查一次，脏且超间隔才写）
        #   ⇒ 落盘从 36 次/分钟降到 3 次/分钟 ⇒ 省掉 ~40 秒/分钟的 CPU+IO
        try:
            def _saver_loop():
                while True:
                    try:
                        self._save_if_due()
                    except Exception:
                        pass
                    __import__("time").sleep(5.0)
            threading.Thread(target=_saver_loop, daemon=True, name="tagger-saver").start()
        except Exception:
            logger.exception('tagger saver thread failed')

    # ---------- 持久化 ----------
    def _load(self):
        try:
            if os.path.exists(self.path):
                self.data = json.load(open(self.path))
        except Exception:
            self.data = {}

    def _save(self):
        """**节流落盘**：每批只置 `_dirty`，真正 dump 由 `_save_if_due()` 按间隔做。

        原来每批都全量重写 61.5 MB ⇒ 36 次/分钟 ⇒ 40 秒/分钟白烧，且占锁。
        """
        try:
            self._dirty = True
        except Exception:
            pass

    def _save_if_due(self):
        """真正的落盘：脏 + 距上次 ≥ `min_interval` 才写；**不持 self.lock**（dump 外面做）。"""
        try:
            import time as _t
            now = _t.time()
            if not getattr(self, "_dirty", False):
                return
            if (now - getattr(self, "_last_save", 0.0)) < getattr(self, "_save_min_interval", 20.0):
                return
            self._dirty = False
            self._last_save = now
            # 快照后 dump（避免与 worker 并发改 dict）
            try:
                with self.lock:
                    snap = json.dumps(self.data, ensure_ascii=False, default=str)
            except Exception:
                snap = None
            if snap is None:
                return
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            tmp = self.path + '.tmp'
            open(tmp, 'w').write(snap)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception('tagger save_if_due failed')

    # ---------- 入队 ----------
    def submit(self, user_id: str, round_id: str, text: str, date: str = '', has_image: bool = False):
        if not user_id or not round_id:
            return
        with self.lock:
            u = self.data.setdefault(user_id, {})
            if round_id in u:
                return
            u[round_id] = {'date': date, 'has_image': has_image, 'text': text[:1200], 'ready': False}
            self._lib_rev[user_id] = self._lib_rev.get(user_id, 0) + 1
        self.q.put((user_id, round_id))

    # ---------- 打标循环 ----------
    def _loop(self):
        while True:
            batch = []
            try:
                item = self.q.get(timeout=1.0)
                batch.append(item)
                while len(batch) < 8:
                    batch.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if batch:
                self._tag_batch(batch)

    def _tag_batch(self, batch):
        body = "\n".join(f"[{i}] {self.data.get(u, {}).get(r, {}).get('text','')[:400]}"
                         for i, (u, r) in enumerate(batch))
        prompt = PROMPT.replace('__TOPICS__', '、'.join(TOPICS)).replace('__ITEMS__', body)
        tags = None
        # ⚠️ （方案 B）：**多端点轮转 + 流式优先**。
        #   为什么要流式：provider-b 的**非流式**被限 10 次/分钟（实测 429），
        #   而**流式 16 并发全部 200**（实测）⇒ tagger 用它必须走 stream。
        #   失败处理：`demote` 把它排到队尾并冷却，再试下一个端点（原有池机制）。
        for attempt in range(1, 5):
            try:
                import httpx
                _ep = self._pick(self.endpoints, "") if self.endpoints else {
                    "base": self.base_url, "key": self.api_key, "model": self.model}
                _base = (_ep.get("base") or self.base_url).rstrip("/")
                _key = _ep.get("key") or self.api_key
                _mdl = _ep.get("model") or self.model
                _payload = {"model": _mdl, "temperature": 0.0, "max_tokens": 1200,
                            "messages": [{"role": "user", "content": prompt}]}
                txt = ""
                # 先流式（provider-b 只有流式不受限；provider-a 也支持 stream）
                _stream_ok = False
                try:
                    with httpx.stream("POST", _base + "/chat/completions",
                                      headers={"Authorization": f"Bearer {_key}"},
                                      json=dict(_payload, stream=True),
                                      timeout=self.timeout) as _r:
                        if _r.status_code == 200:
                            for _line in _r.iter_lines():
                                if not _line or not _line.startswith("data:"):
                                    continue
                                _d = _line[5:].strip()
                                if _d == "[DONE]":
                                    break
                                try:
                                    _j = json.loads(_d)
                                    txt += ((_j["choices"][0].get("delta") or {}).get("content") or "")
                                except Exception:
                                    pass
                            _stream_ok = bool(txt.strip())
                            if _stream_ok and not getattr(self, "_stream_logged", False):
                                self._stream_logged = True
                                logger.info("[tagger] stream OK（首次）endpoint=%s chars=%d",
                                            self._etag(_ep), len(txt))
                        else:
                            _r.read()
                except Exception:
                    _stream_ok = False
                if not _stream_ok:
                    # 流式不可用 ⇒ 退回非流式（provider-b 会受 10/分钟限制，故优先选别的端点）
                    logger.warning("[tagger] stream 失败，退非流式 endpoint=%s", self._etag(_ep))
                    resp = httpx.post(_base + '/chat/completions',
                                      headers={'Authorization': f'Bearer {_key}'},
                                      json=_payload, timeout=self.timeout)
                    resp.raise_for_status()
                    txt = (resp.json()['choices'][0]['message']['content'] or '')
                txt = txt.strip()
                if txt.startswith('```'):
                    txt = txt.split('```')[1]; txt = txt[4:] if txt.lower().startswith('json') else txt
                tags = json.loads(txt)
                try:
                    self._promote(_ep)
                except Exception:
                    pass
                break
            except Exception:
                try:
                    self._demote(_ep, 60)
                except Exception:
                    pass
                time.sleep(1.0 * attempt)
        with self.lock:
            for i, (u, r) in enumerate(batch):
                rec = self.data.get(u, {}).get(r)
                if rec is None: continue
                got = None
                if tags:
                    for t in tags:
                        try:
                            if int(t.get('i', -1)) == i: got = t; break
                        except Exception: continue
                rec.update({'entities': (got or {}).get('entities', [])[:5],
                            'topics': (got or {}).get('topic', []),
                            'summary': (got or {}).get('summary', ''), 'ready': bool(got)})
                # ⚠️ 必须 bump 版本：打标完成后 `ready`/entities 变了，
                # 否则图书馆缓存会永远停留在"未打标"的版本（实体/主题层为空）。
                self._lib_rev[u] = self._lib_rev.get(u, 0) + 1
            self._save()

    def import_gem_tags(self, tagdir: str, prefix: str = 'pf-frozen-cd:'):
        """把 data/reports/library/tags/*.json（批量打标结果）导入，round_id 加生产前缀。"""
        import glob
        n = 0
        for f in sorted(glob.glob(os.path.join(tagdir, '*.json'))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            u = d.get('user_id', '')
            recs = self.data.setdefault(u, {})
            tags = {str(t.get('round_id')): t for t in d.get('tags', [])}
            for it in d.get('items', []):
                rid = str(it.get('rid', ''))
                if not rid:
                    continue
                t = tags.get(rid) or {}
                self._lib_rev[u] = self._lib_rev.get(u, 0) + 1
                recs[prefix + rid] = {'date': it.get('date', ''), 'has_image': bool(it.get('has_image')),
                                      'text': it.get('text', '')[:1200],
                                      'entities': (t.get('entities') or [])[:5],
                                      'topics': t.get('topic') or [],
                                      'summary': t.get('summary') or '',
                                      'ready': bool(t)}
                n += 1
        self._save()
        return n

    def plan(self, lib, query: str, tries: int = 2):
        """LLM 看目录后给出查询计划（分支+约束）。失败返回 []（回退到纯词法）。"""
        if lib is None: return []
        prompt = ("你是记忆图书馆的检索规划员。目录：\n" + lib.tree_text(max_topics=15, max_entities=25) +
                  "\n\n问题：" + query +
                  '\n分支含义：M=主题、E=具体实体、Y=实体类型（artwork/place/person/pet/product/...）、T=时间、O=模态。'
                  '\n只输出 JSON：{"plan":[{"branch":"M 或 E 或 Y 或 T 或 O","name":"分支名",'
                  '"time":["YYYY-MM-DD","YYYY-MM-DD"] 或 null,"entities":["..."] 或 null,'
                  '"modality":"image 或 text 或 null"}]}')
        for a in range(1, tries + 1):
            try:
                import httpx
                r = httpx.post(self.base_url + '/chat/completions',
                               headers={'Authorization': f'Bearer {self.api_key}'},
                               json={'model': self.model, 'temperature': 0.0, 'max_tokens': 400,
                                     'messages': [{'role': 'user', 'content': prompt}]},
                               timeout=self.timeout)
                r.raise_for_status()
                self._promote(_ep)
                t = (r.json()['choices'][0]['message']['content'] or '').strip()
                if t.startswith('```'):
                    t = t.split('```')[1]; t = t[4:] if t.lower().startswith('json') else t
                return json.loads(t).get('plan') or []
            except Exception:
                time.sleep(0.8 * a)
        return []

    # ---------- LLM 证据选择（10 分钟预算）----------
    def select_evidence(self, query: str, cand: list, budget: float = None, hint: str = "",
                        enum: bool = False) -> dict:
        """从候选池中选出回答问题所需的轮次，并判定**证据是否充分**。

        返回 {"selected": [下标...], "sufficient": bool, "reason": str}。
        失败（预算耗尽）才抛错。
        """
        #   目的：`AMC_EVIDENCE_BUDGET` / `AMC_TAG_MAX_ATTEMPTS` 是**启动时固定在进程 env** 的，
        #   一旦 search 阶段发现"预算过长导致队列污染"，改 env 必须重启（~15s）。
        #   这里加**文件覆盖**（每次调用时读）⇒ 改文件**即时生效、零重启**：
        #     data/store/evidence_budget.txt    （如 `20`）
        #     data/store/tag_max_attempts.txt   （如 `3`）
        #     ① 文件不存在 ⇒ 行为与原来完全一致（读 env）
        #     ② 文件存在但内容非法（空/非数字/≤0）⇒ **忽略并回退 env**，绝不抛错
        #     ③ 每次调用都读 ⇒ 改文件即时生效
        def _file_num(fname, default):
            try:
                _fp = os.path.join("data", "store", fname)
                if os.path.exists(_fp):
                    with open(_fp, "r", encoding="utf-8") as _fh:
                        _raw = (_fh.read() or "").strip()
                    if _raw:
                        _v = float(_raw)
                        if _v > 0:
                            return _v
            except Exception:
                pass
            return default

        if budget is None:
            # 证据核验的总时间预算（代码默认 45s；env 里是 600 ⇒ 会让单题最多挂 10 分钟）
            budget = _file_num("evidence_budget.txt",
                               float(os.getenv("AMC_EVIDENCE_BUDGET", "45") or 45))
        if not cand:
            return {"selected": [], "sufficient": False, "reason": "no candidates"}
        body = "\n\n".join(
            "[%d] (date=%s has_image=%s) %s" % (i, c.get('date',''), c.get('has_image'), str(c.get('text',''))[:900])
            for i, c in enumerate(cand))
        prompt = (
            "你是记忆检索的**证据核验器**。判断候选记忆是否足以回答问题，并选出承载答案的轮次。\n"
            f"{hint}"
            f"问题：{query[:400]}\n\n候选：\n{body}\n\n"
            "规则：\n"
            "- 只有候选里**确实包含回答问题所需的信息**时 sufficient=true，并给出承载该信息的轮次编号\n"
            "- 仅主题相关、只提到相关词、或只有 has_image=true 而没有内容描述 → sufficient=false\n"
            "- **补充规则（重要）**：如果候选只是**谈到了相关话题**，但没有给出问题所问的**那一项**"
            "（具体属性/数值/名称/物种/日期/关系/主体），必须判 false。例如：\n"
            "  · 问「花的学名」，候选只描述外形 → false\n"
            "  · 问「硬盘的价格」，候选只说买了硬盘 → false\n"
            "  · 问「Lena 哥哥的狗的名字」，候选讲的是 Lena 自己的狗 → false\n"
            "- 视觉题：必须有能回答所问内容的图像描述；问物种/名称时必须出现该名称\n"
            "- 聚合题：必须能逐项数清\n"
            "- 时间比较题：必须能确定两个事件的日期\n"
            "_SYN_HINT_"
            "_ENUM_HINT_"
            '只输出 JSON：{"selected":[<编号>...],"sufficient":true|false,"reason":"一句话"}')
        _enum_hint = ("- **枚举模式（本题需要多项）**：把候选里**所有**能作为答案项之一的轮次都选出来，"
                      "宁多勿漏；不要只选最像的 1~3 条。\n") if enum else ""
        prompt = prompt.replace("_ENUM_HINT_", _enum_hint)
        # 同义桥接（默认关）：候选与问题用**不同措辞**说同一件事时，
        # 严格字面判定会把"房贷 vs 还款压力""花销 vs 支出"判成"信息不足"。
        # 开启后允许把同义表述视为包含所需信息。实测对比后再决定是否默认开。
        _syn_hint = ("- **同义桥接**：候选里的说法与问题不同但指同一件事（如 房贷≈还款、"
                     "花销≈支出、住处≈酒店），视为包含所需信息；仅当确实无关才判 false。\n"
                     ) if os.getenv("AMC_EVIDENCE_SYNONYM", "0") == "1" else ""
        prompt = prompt.replace("_SYN_HINT_", _syn_hint)
        deadline = time.time() + budget
        attempt = 0
        _cyc = 0
        last = None
        _max_attempts = int(_file_num("tag_max_attempts.txt",
                                      float(os.getenv("AMC_TAG_MAX_ATTEMPTS", "6") or 6)))
        while time.time() < deadline:
            attempt += 1
            if attempt > _max_attempts:
                logger.warning("[evidence] 放弃（尝试 %d 次仍未拿到可用响应）", attempt - 1)
                return None
            _ep = self._pick(self.endpoints)
            try:
                import httpx
                r = httpx.post(_ep["base"] + '/chat/completions',
                               headers={'Authorization': f"Bearer {_ep['key']}"},
                               json={'model': _ep["model"], 'temperature': 0.0, 'max_tokens': 500,
                                     'messages': [{'role': 'user', 'content': prompt}]},
                               timeout=self.timeout)
                r.raise_for_status()
                self._promote(_ep)
                t = (r.json()['choices'][0]['message']['content'] or '').strip()
                if t.startswith('```'):
                    t = t.split('```')[1]; t = t[4:] if t.lower().startswith('json') else t
                obj = json.loads(t)
                sel = [x for x in (obj.get('selected') or []) if isinstance(x, int) and 0 <= x < len(cand)]
                return {"selected": sel, "sufficient": bool(obj.get("sufficient")),
                        "reason": str(obj.get("reason") or "")[:200]}
            except Exception as exc:
                last = exc
                # 4xx（鉴权/余额/参数）是永久错误：立即失败，不要耗尽预算
                _status = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(_status, int) and 400 <= _status < 500 and _status != 429:
                    raise RuntimeError(f"evidence check permanent failure ({_status}): {exc}") from exc
                # 有备用网关 → 立刻切；整轮失败才退避
                # ⚠️ 退避必须在**每次失败**后生效，不能只在"切完端点"之后。
                # 原来切换分支是 `continue`（不退避）⇒ 快速失败（连接拒绝/503）时会
                # 瞬间烧光尝试次数，退避形同虚设（实测 5 次尝试只用了 4 秒）。
                _backoff = min(30.0, float(2 ** min(attempt - 1, 5)))
                if len(self.endpoints) > 1 and _cyc < len(self.endpoints) - 1:
                    _cyc += 1
                    logger.warning("[evidence] %s 失败(%s)，切换 → %s，退避 %.1fs",
                                   self._etag(_ep), str(exc)[:50],
                                   self._etag(self._pick(self.endpoints)), _backoff)
                    time.sleep(_backoff)
                    continue
                _cyc = 0
                # 原来是 `min(8.0, 0.8*attempt)`（线性、封顶 8s）⇒ 网关排队时退避不够，
                # 重试会不断挤回队尾。改成**指数退避**：1,2,4,8,16,30,30…（封顶 30s）
                logger.info("[evidence] 第 %d 次失败(%s)，退避 %.1fs 后重试",
                            attempt, str(exc)[:50], _backoff)
                time.sleep(_backoff)
        raise RuntimeError(f"evidence check failed after {attempt} attempts / {budget:.0f}s: {last}")


    def sufficient_variant(self, query: str, cand: list, variant: str = "v0",
                           budget: float = 300.0) -> bool:
        """额外的 sufficient 判定（用于校准器的多信号特征）。"""
        if not cand:
            return False
        body = "\n\n".join(
            "[%d] (date=%s has_image=%s) %s" % (i, c.get("date", ""), c.get("has_image"), str(c.get("text", ""))[:900])
            for i, c in enumerate(cand))
        if variant == "v0":
            prompt = ("你是记忆检索的证据核验器。判断候选记忆是否足以回答问题。\n"
                      f"问题：{query[:400]}\n\n候选：\n{body}\n\n"
                      "规则：只有确实包含回答问题所需的信息时 sufficient=true；"
                      "仅主题相关、只提相关词、只有 has_image 无描述 → false。\n"
                      '只输出 JSON：{"sufficient":true|false}')
            obj = self._call_json(prompt, budget)
            return bool(obj and obj.get("sufficient"))
        # v3：两段判定
        pa = ("第一步：主体与条件匹配。\n问题：%s\n\n候选：\n%s\n\n"
              "判断候选里是否有记忆在说同一主体且满足问题的限定条件。\n"
              '只输出 JSON：{"subject_match":true|false,"ids":[0]}') % (query[:400], body)
        a = self._call_json(pa, budget)
        if not (a and a.get("subject_match")):
            return False
        ids = [x for x in (a.get("ids") or []) if isinstance(x, int) and 0 <= x < len(cand)]
        sub = [cand[i] for i in ids] or cand[:5]
        body2 = "\n\n".join("[%d] (date=%s) %s" % (i, c.get("date", ""), str(c.get("text", ""))[:900])
                              for i, c in enumerate(sub))
        pb = ("第二步：属性/值是否存在。\n问题：%s\n\n候选（主体相关轮）：\n%s\n\n"
              "判断这一轮是否明确给出了问题所问的那一项（属性值/名称/数量/日期/物种/关系）。\n"
              '只输出 JSON：{"attribute_present":true|false}') % (query[:400], body2)
        b = self._call_json(pb, budget)
        return bool(b and b.get("attribute_present"))

    def _call_json(self, prompt: str, budget: float = 300.0):
        deadline = time.time() + budget
        attempt = 0
        _cyc = 0
        _max_attempts = int(os.getenv("AMC_TAG_MAX_ATTEMPTS", "6") or 6)
        while time.time() < deadline:
            attempt += 1
            if attempt > _max_attempts:
                logger.warning("[evidence] 放弃（尝试 %d 次仍未拿到可用响应）", attempt - 1)
                return None
            _ep = self._pick(self.endpoints)
            try:
                import httpx
                try:
                    from amc.llm_stream import enabled as _ks_on, chat_text as _ks_chat
                    _use_stream = _ks_on()
                except Exception:
                    _use_stream = False
                _msgs = [{"role": "user", "content": prompt}]
                if _use_stream:
                    t = (_ks_chat(_ep["base"], _ep["key"], _ep["model"], _msgs,
                                  max_tokens=400, temperature=0.0,
                                  timeout=float(self.timeout)) or "").strip()
                else:
                    r = httpx.post(_ep["base"] + "/chat/completions",
                                   headers={"Authorization": f"Bearer {_ep['key']}"},
                                   json={"model": _ep["model"], "temperature": 0.0, "max_tokens": 400,
                                         "messages": _msgs},
                                   timeout=self.timeout)
                    r.raise_for_status()
                    t = (r.json()["choices"][0]["message"]["content"] or "").strip()
                if t.startswith("```"):
                    t = t.split("```")[1]
                    t = t[4:] if t.lower().startswith("json") else t
                return json.loads(t)
            except Exception as _exc:
                _st = getattr(getattr(_exc, "response", None), "status_code", None)
                if isinstance(_st, int) and 400 <= _st < 500 and _st != 429:
                    return None
                self._demote(_ep)
                # 还有备用网关 → 立刻切，不等待
                if len(self.endpoints) > 1 and _cyc < len(self.endpoints) - 1:
                    _cyc += 1
                    logger.warning("[evidence] %s 失败(%s)，立即切换 → %s",
                                   self._etag(_ep), str(_exc)[:60], self._etag(self._pick(self.endpoints)))
                    continue
                _cyc = 0
                # 上游偶发 5xx/超时：**不降级**，指数退避重试（2,4,8,16,32,60… 秒 + 抖动）
                _wait = 1.2 * attempt
                logger.warning("[evidence] LLM 调用失败（%s），%.1fs 后重试（剩余预算 %.0fs）",
                               _exc, _wait, max(0.0, deadline - time.time()))
                time.sleep(_wait)
        return None

    # ---------- dense 嵌入（图书馆排序用）----------
    def embedder(self):
        if self._embedder is not None:
            return self._embedder
        if not (self.embed_base_url and self.embed_model):
            return None
        try:
            from amc.retrieval.providers import OpenAIEmbeddingProvider, VectorCache
            prov = OpenAIEmbeddingProvider(self.embed_model, self.embed_base_url,
                                           api_key=self.embed_api_key, dim=self.embed_dim,
                                           timeout=60, retries=3)
            # 关键：图书馆构建也要走**持久化向量缓存**。
            # 否则每次 /add 让库变脏、下次检索就把全部轮次重嵌一遍
            # （实测 2,868 轮 → 15 分钟，是单查询卡死的根因）。
            _cache = None
            try:
                _cache = VectorCache(os.getenv("AMC_EMBED_CACHE_PATH", "./data/store/embcache.db"),
                                     namespace=prov.name, dim=prov.dim)
            except Exception:
                logger.warning("[library] vector cache 不可用，回退为直连 embedding")

            def _fn(texts):
                texts = list(texts)
                if _cache is None:
                    arr = prov._call(texts)
                    return [list(map(float, row)) for row in arr]
                keys = [_cache.key(t) for t in texts]
                found, missing = _cache.get_many(keys)
                miss_idx = [i for i, k in enumerate(keys) if k not in found]
                if miss_idx:
                    arr = prov._call([texts[i] for i in miss_idx])
                    rows = [list(map(float, row)) for row in arr]
                    _cache.put_many([(keys[i], list(map(float, arr[j]))) for j, i in enumerate(miss_idx)])
                    for j, i in enumerate(miss_idx):
                        found[keys[i]] = rows[j]
                return [found[k] for k in keys]
            self._embedder = _fn
        except Exception:
            logger.exception("library embedder init failed")
            self._embedder = False
        return self._embedder or None

    def refresh_round_text(self, rid_text_pairs) -> int:
        """把**已变更的轮次文本**刷新进 tagger.data，并让图书馆缓存失效。

        为什么需要：`library_for()` 用的是 `self.data[user]`（打标时的快照），
        而 caption 是**异步**写到 `store.index_text` 的 ⇒ 两边不通。
        不刷新的话，异步补上的 caption 在图书馆的召回增强里永远看不见。

        入参：{rid: (user_id, new_text)} —— store 的 text_changed_hook 直接传这个结构。
        返回：实际刷新的轮次数。
        """
        n = 0
        touched_users = set()
        with self.lock:
            for rid, pair in (rid_text_pairs or {}).items():
                try:
                    uid, ntext = pair
                except Exception:
                    continue
                for key in (uid, uid.split(':', 1)[1] if ':' in uid else uid):
                    u = self.data.get(key)
                    if not u or rid not in u:
                        continue
                    u[rid]['text'] = ntext
                    # 文本变了 ⇒ 旧的 summary/entities 也可能过期，但这里只保证"检索文本最新"；
                    # 重新打标交给后台（submit），避免在 caption 线程里再发一次 LLM 调用。
                    touched_users.add(key)
                    n += 1
                    break
            for k in touched_users:
                self._lib_rev[k] = self._lib_rev.get(k, 0) + 1
        if n:
            logger.info("[caption] 图书馆文本已刷新 %d 轮（失效 %d 个用户缓存）", n, len(touched_users))
        return n

    # ---------- 查询 ----------
    def library_for(self, user_id: str):
        # 先看缓存：key = user + 版本号（版本号在任何数据变更时递增）
        with self.lock:
            _rev = self._lib_rev.get(user_id, 0)
            _hit = self._lib_cache.get(user_id)
            if _hit is not None and _hit[0] == _rev:
                self._lib_cache.move_to_end(user_id)
                return _hit[1]
            u = self.data.get(user_id) or {}
            if not u and ':' in user_id:
                # 生产 store 的 user_id 带 "pf-frozen-cd:" 前缀，标签按无前缀存 → 回退匹配
                alt = user_id.split(':', 1)[1]
                u = self.data.get(alt) or {}
            ready = {rid: rec for rid, rec in u.items() if rec.get('ready')}
        if not ready:
            try:
                _fn.cache = _cache            # ⚠️ ：暴露缓存给 library 统计缺向量
            except Exception:
                pass
            return None
        rounds = [{'round_id': rid, 'text': rec.get('text',''), 'date': rec.get('date',''),
                   'has_image': rec.get('has_image', False)} for rid, rec in ready.items()]
        tags = [{'round_id': rid, 'entities': rec.get('entities', []), 'topic': rec.get('topics', []),
                 'summary': rec.get('summary','')} for rid, rec in ready.items()]
        try:
            from amc.library import MemoryLibrary
            import time as _t
            _t0 = _t.time()
            lib = MemoryLibrary(rounds, tags, embedder=self.embedder())
            with self.lock:
                self._lib_cache[user_id] = (_rev, lib)
                self._lib_cache.move_to_end(user_id)
                while len(self._lib_cache) > self._lib_cache_max:
                    self._lib_cache.popitem(last=False)
            logger.info("[library] built user=%s rounds=%d %.2fs (rev=%d)",
                        user_id[-16:], len(rounds), _t.time() - _t0, _rev)
            return lib
        except Exception:
            logger.exception('library build failed')
            return None
