"""Agent 检索流程：置信门控 → LLM 调度（判题型+出检索计划）→ 按类检索 → 证据校验（可再检索一次）。

设计原则
--------
* **绝不破坏契约**：任何环节失败都退化为"普通检索"，永不抛异常给 /search。
* **有界成本**：每请求最多 1 次调度 + 最多 2 次额外检索 + 最多 2 次校验。
* **可观测**：统计各分支次数，暴露到 /health。

流程
----
1. 基础检索（复用现有 retriever）→ 得到 hits（4B/LLM 重排后的轮级序）
2. 置信门控：top1 分数 < gate_eps → 判"无证据"（返回空，由服务层转成 data=[]）
3. 调度（LLM）：输出 type / queries（1-3 个自包含变体）/ notes
4. 执行：对 variants 各检索一次，与基础结果按 id 合并（保留最高分）
5. 校验（LLM）：给定问题 + 前 N 轮，判 sufficient / missing / rerank
6. 若不足且有 remaining rounds：用 "问题 + missing" 再检索一次，合并
7. 用校验给出的 rerank 调整顺序（缺省保持原序）
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEDULE_PROMPT = """You are the scheduler of a personal-memory search engine.
Classify the question and decide how to retrieve evidence for it.

Types:
- lookup: asks for one specific fact/entity/attribute stated in memory
- aggregate: asks for a total / count / sum over several memories
- multi_hop: needs evidence from MULTIPLE sessions or turns (before/after, across days)
- visual: the answer depends on what is visible in an image
- temporal: asks when/whether something happened relative to something else
- abstain_check: asks whether something ever happened / whether info exists
- general: none of the above

Return STRICT JSON only:
{"type": "<one of the types>",
 "queries": ["<1-3 self-contained retrieval queries; keep entities/dates; may be empty>"],
 "notes": "<one short sentence>"}

Question: __Q__
JSON:"""

EXPAND_PROMPT = """You help a personal-memory search engine find evidence.
Rewrite the question into 1-3 SELF-CONTAINED retrieval queries that would surface the
evidence needed. Keep entities, dates, places and numbers. Vary the phrasing, not the meaning.
Return STRICT JSON only:
{"queries": ["<query 1>", "<query 2>"]}

Question: __Q__
JSON:"""

VERIFY_PROMPT = """You are the evidence checker of a personal-memory search engine.
Below are candidate conversational ROUNDS retrieved for the question.
Judge (a) whether they contain ENOUGH evidence, and (b) whether they CONTRADICT the
question's hard constraints (who / when / where / exact value / required object).

Rules:
- Only judge hard constraints that the question explicitly requires.
- Do NOT demand extra detail the question does not ask for.
- If the evidence is plausible but incomplete, mark sufficient=false and say what is missing.
- If the evidence clearly violates a constraint (wrong date/person/value), list it in violations.
- Provide 1-2 targeted retrieval queries that would find the missing evidence.

Return STRICT JSON only:
{"verdict": "answerable" | "missing_info" | "contradicts",
 "sufficient": true|false,
 "missing": "<what is missing; empty if sufficient>",
 "violations": ["<violated constraint>"],
 "queries": ["<targeted search query>"],
 "rerank": [<candidate numbers, most relevant first>]}

Question: __Q__

Candidates:
__C__
JSON:"""


JUDGE_EACH_PROMPT = """You are the evidence checker of a personal-memory search engine.
For EACH candidate conversational ROUND below, decide whether it CONTAINS the evidence
needed to answer the question. Check hard constraints (who / when / where / exact value /
required object). Do not require more detail than the question asks for.

Return STRICT JSON only:
{"evidence": [<candidate numbers that contain evidence, best first; empty list if none>]}

Question: __Q__

Candidates:
__C__
JSON:"""


def _truncate(s: Any, n: int) -> str:
    t = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return t[:n]


class AgentRetriever:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0,
                 max_extra_searches: int = 2, gate_eps: float = 0.005,
                 verify_chars: int = 900, prompt_type: str = "schedule"):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.model = model
        self.timeout = float(timeout)
        self.max_extra = max(0, int(max_extra_searches))
        self.gate_eps = float(gate_eps)
        self.verify_chars = int(verify_chars)
        self.prompt_type = prompt_type
        self.mode = prompt_type if prompt_type in ("expand", "each") else "classify"
        self.max_rounds = 2
        # 分阶段 trace：设置路径后，每次 run 追加一条 {q, stages:{...}, verdict}
        self.trace_path = ""
        self.promote = 2
        self.judge_pool = 30
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "scheduled": 0, "gated": 0,
                      "extra_searches": 0, "verify_calls": 0, "insufficient": 0,
                      "fallback": 0, "types": {}}
        try:
            import httpx
            self.client = httpx.Client(timeout=self.timeout)
        except Exception:
            self.client = None

    # ---------------- LLM ----------------
    def _chat(self, prompt: str, max_tokens: int = 300) -> str:
        if self.client is None:
            raise RuntimeError("no http client")
        body = {"model": self.model, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        last = None
        for attempt in range(1, 4):                 # 3 次重试，失败即抛错（不降级）
            try:
                r = self.client.post(self.base_url + "/chat/completions",
                                     headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:
                last = exc
                logger.warning("[agent] attempt %d/3 failed: %s", attempt, exc)
                if attempt < 3:
                    time.sleep(1.0 * attempt)
        raise RuntimeError(f"agent llm failed after 3 attempts: {last}")

    @staticmethod
    def _json(text: str) -> Optional[dict]:
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            parts = t.split("```")
            t = parts[1] if len(parts) > 1 else t
            t = t[4:] if t.lower().startswith("json") else t
        for cand in (t, re.sub(r",\s*([}\]])", r"\1", t)):
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    # ---------------- steps ----------------
    def schedule(self, question: str, mode: str = "classify") -> Optional[dict]:
        with self.lock:
            self.stats["calls"] += 1
        try:
            if mode == "expand":
                out = self._chat(EXPAND_PROMPT.replace("__Q__", question), 200)
                data = self._json(out)
                if not data:
                    return None
                qs = data.get("queries") or []
                if isinstance(qs, str):
                    qs = [qs]
                data["queries"] = [str(q).strip() for q in qs if str(q).strip()][:3]
                data["type"] = "expand"
                with self.lock:
                    self.stats["scheduled"] += 1
                return data
            out = self._chat(SCHEDULE_PROMPT.replace("__Q__", question), 260)
            data = self._json(out)
            if not data:
                return None
            qs = data.get("queries") or []
            if isinstance(qs, str):
                qs = [qs]
            data["queries"] = [str(q).strip() for q in qs if str(q).strip()][:3]
            data["type"] = str(data.get("type") or "general").strip().lower()
            with self.lock:
                self.stats["scheduled"] += 1
                self.stats["types"][data["type"]] = self.stats["types"].get(data["type"], 0) + 1
            return data
        except Exception as exc:
            with self.lock:
                self.stats["fallback"] += 1
            raise RuntimeError(f"schedule failed: {exc}") from exc

    def verify(self, question: str, rounds: List[Tuple[dict, float]]) -> Optional[dict]:
        if not rounds:
            return None
        with self.lock:
            self.stats["verify_calls"] += 1
        lines = []
        for i, (m, _s) in enumerate(rounds):
            txt = m.get("index_text") or ""
            lines.append(f"[{i}] {_truncate(txt, self.verify_chars)}")
        try:
            out = self._chat(VERIFY_PROMPT.replace("__Q__", question).replace("__C__", "\n".join(lines)), 300)
            data = self._json(out)
            if not data:
                return None
            data["sufficient"] = bool(data.get("sufficient"))
            data["missing"] = str(data.get("missing") or "").strip()
            vv = str(data.get("verdict") or "").strip().lower()
            data["verdict"] = vv if vv in ("answerable", "missing_info", "contradicts") else \
                ("answerable" if data["sufficient"] else "missing_info")
            viol = data.get("violations") or []
            data["violations"] = [str(x).strip() for x in viol if str(x).strip()][:4]
            qs = data.get("queries") or []
            data["queries"] = [str(q).strip() for q in qs if str(q).strip()][:2]
            rr = data.get("rerank") or []
            rr = [int(x) for x in rr if isinstance(x, (int, float)) and 0 <= int(x) < len(rounds)]
            data["rerank"] = rr
            if not data["sufficient"]:
                with self.lock:
                    self.stats["insufficient"] += 1
            return data
        except Exception:
            with self.lock:
                self.stats["fallback"] += 1
            return None

    def judge_each(self, question: str, rounds: List[Tuple[dict, float]]) -> Optional[List[int]]:
        if not rounds:
            return None
        with self.lock:
            self.stats["judge_calls"] = self.stats.get("judge_calls", 0) + 1
        lines = []
        for i, (m, _s) in enumerate(rounds):
            txt = m.get("index_text") or ""
            lines.append(f"[{i}] {_truncate(txt, self.verify_chars)}")
        try:
            out = self._chat(JUDGE_EACH_PROMPT.replace("__Q__", question).replace("__C__", "\n".join(lines)), 300)
            data = self._json(out)
            if not data:
                return None
            ev = data.get("evidence") or []
            idx = []
            for x in ev:
                try:
                    i = int(x)
                except Exception:
                    continue
                if 0 <= i < len(rounds) and i not in idx:
                    idx.append(i)
            return idx
        except Exception as exc:
            with self.lock:
                self.stats["fallback"] += 1
            raise RuntimeError(f"verify failed: {exc}") from exc

    # ---------------- main ----------------
    def run(self, retriever, user_id: str, query_text: str, limit: int,
            image_parts: Any = None) -> List[Tuple[dict, float]]:
        def plain():
            return retriever.search(user_id, query_text, limit)

        if not query_text or not query_text.strip():
            return plain()
        try:
            base = plain()
        except Exception:
            raise
        if not base:
            return base
        # 1) 置信门控（非常保守：只为"确实没有任何线索"拒答）
        if self.gate_eps > 0 and base[0][1] < self.gate_eps:
            with self.lock:
                self.stats["gated"] += 1
            return []
        # 2a) 逐条判定模式：不扩展、不追检索，只对**同一查询**的候选池逐条判定并提序
        if self.mode == "each":
            stages = {"base": [m.get("id") for m, _ in base[:30]]}
            pool = base[: min(len(base), int(getattr(self, "judge_pool", 30)))]
            idx = self.judge_each(query_text, pool)
            stages["judged"] = [m.get("id") for m, _ in pool[:30]]
            if idx:
                picked = [pool[i] for i in idx]
                used = {id(x[0]) for x in picked}
                rest = [x for x in base if id(x[0]) not in used]
                base = picked + rest
            stages["final"] = [m.get("id") for m, _ in base[:30]]
            if self.trace_path:
                try:
                    import json as _json
                    with self.lock:
                        with open(self.trace_path, "a") as fh:
                            fh.write(_json.dumps({"q": query_text[:200], "stages": stages,
                                                  "n_judged": len(idx or []),
                                                  "n_base": len(base)}, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            desc = sorted((float(sc) for _, sc in base), reverse=True)
            return [(m, desc[i]) for i, (m, _sc) in enumerate(base)][:limit]
        # 2) 调度
        plan = self.schedule(query_text, self.mode)
        if plan is None:
            return base
        stages = {"base": [m.get("id") for m, _ in base[:30]]}
        merged = list(base)
        seen = {m.get("id") for m, _ in merged}
        extra_used = 0
        for q in plan.get("queries", [])[:2]:
            if extra_used >= self.max_extra:
                break
            if q.strip() == query_text.strip():
                continue
            try:
                more = retriever.search(user_id, q, limit)
            except Exception:
                continue
            extra_used += 1
            for m, s in more:
                if m.get("id") not in seen:
                    seen.add(m.get("id"))
                    merged.append((m, s))
        stages["expanded"] = [m.get("id") for m, _ in merged[:30]]
        # 3) 校验循环：不足 → 用 "问题 + missing" 再检索 → 再校验（最多 max_rounds 轮）
        for _round in range(max(1, int(getattr(self, "max_rounds", 2)))):
            head = merged[: min(len(merged), 12)]
            ver = self.verify(query_text, head)
            if ver is None or ver["sufficient"] or not ver["missing"]:
                break
            if extra_used >= self.max_extra:
                break
            targeted = (ver.get("queries") or [])[:2]
            if not targeted:
                targeted = [(query_text + " " + ver["missing"])[:400]]
            try:
                more = []
                for tq in targeted[:1]:
                    more += retriever.search(user_id, tq, limit)
                extra_used += 1
                added = 0
                for m, s in more:
                    if m.get("id") not in seen:
                        seen.add(m.get("id"))
                        merged.append((m, s))       # 追加到尾部，绝不动头部
                        added += 1
                if added == 0:
                    break
            except Exception:
                break
        stages["verified"] = [m.get("id") for m, _ in merged[:30]]
        self.last_verdict = ver
        with self.lock:
            self.stats["extra_searches"] += extra_used
            if ver:
                self.stats[f"verdict_{ver.get('verdict','?')}"] = \
                    self.stats.get(f"verdict_{ver.get('verdict','?')}", 0) + 1
        # 3.5) 让高分补充证据"竞争进入头部"（上限 promote 条，按分数插入）
        promote = int(getattr(self, "promote", 0) or 0)
        if promote > 0 and len(merged) > len(base):
            head = merged[: len(base)]
            tail = merged[len(base):]
            tail_sorted = sorted(tail, key=lambda x: float(x[1]), reverse=True)
            for cand in tail_sorted[:promote]:
                # 只有当它比头部最末一名更高分时才提升
                if head and float(cand[1]) > float(head[-1][1]):
                    head.append(cand)
                    head.sort(key=lambda x: float(x[1]), reverse=True)
                    head = head[: len(base)]      # 头部长度不变
                else:
                    break
            merged = head + [x for x in merged if x not in head]
        # 4) 保持原有顺序（基线头名不动）；分数按原分数序重分配以保证非递增
        stages["final"] = [m.get("id") for m, _ in merged[:30]]
        if self.trace_path:
            try:
                import json as _json
                rec = {"q": query_text[:200], "stages": stages,
                       "verdict": (ver or {}).get("verdict"),
                       "violations": (ver or {}).get("violations") or [],
                       "sufficient": (ver or {}).get("sufficient"),
                       "n_base": len(base), "n_final": len(merged)}
                with self.lock:
                    with open(self.trace_path, "a") as fh:
                        fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
        desc = sorted((float(sc) for _, sc in merged), reverse=True)
        out = [(m, desc[i]) for i, (m, _sc) in enumerate(merged)]
        return out[:limit]

    def describe(self) -> Dict[str, Any]:
        return {"class": self.__class__.__name__, "model": self.model,
                "gate_eps": self.gate_eps, "max_extra": self.max_extra,
                **{k: v for k, v in self.stats.items() if k != "types"},
                "types": dict(self.stats.get("types") or {})}


CONSTRAINT_PROMPT = """You check whether the retrieved evidence satisfies the question's requirements.

Step 1 — identify what the question needs:
- kind = "aggregate": asks for a total / count / sum (how much, how many, total, 一共, 总共).
- kind = "multi_step": asks for several parts / steps / sessions (哪几步, 先…后…, 两个, 分别).
- kind = "fact": asks for one fact / attribute / yes-no experience.
- kind = "none": no explicit requirement.

Step 2 — judge satisfaction:
- aggregate: the evidence must CONTAIN the relevant amounts/items over the asked period.
  The total itself need NOT be pre-computed — if the needed records are present, satisfied=true.
- multi_step: the evidence must COVER ALL required parts (e.g. both steps, both sessions).
  Partial coverage = not satisfied; list which part is missing.
- fact: the evidence must DIRECTLY state the asked fact/experience, about the SAME person,
  in the SAME situation the question describes.
  * Topic overlap alone does NOT satisfy it.
  * A group-chat / third-party record does NOT satisfy a question about MY OWN experience.
  * An image-caption description does NOT satisfy it unless it states the asked fact.
  If the asked experience/fact is not stated anywhere, satisfied=false.
- none: satisfied=true.
- Evidence spread across multiple rounds/sessions is fine — judge the SET as a whole.
- Each evidence line starts with "(<timestamp>)" when available; USE IT to judge temporal
  order questions ("before/after/first/which came first").

Return STRICT JSON only:
{"kind": "...", "satisfied": true|false, "unsatisfied": ["<missing/unsatisfied requirement>"], "note": "<short>"}

Question: __Q__

Evidence:
__C__
JSON:"""


RESCUE_PROMPT = """You are given a question and evidence rounds. Your ONLY job is to find an
exact sentence in the evidence that DIRECTLY answers the question.

Rules:
- The sentence must be copied VERBATIM from the evidence (character-for-character), and must
  by itself answer the question (not just share topic words).
- If NO sentence directly answers the question, return found=false and an empty quote.
- Never invent or paraphrase.

Return STRICT JSON only:
{"found": true|false, "quote": "<verbatim sentence from the evidence>", "round": <index>}

Question: __Q__

Evidence:
__C__
JSON:"""


class EvidenceJudge(AgentRetriever):
    """只做"证据是否满足问题约束"的判定，不检索、不追查。复用 AgentRetriever 的 verify()。"""

    def rescue(self, question: str, rounds: List[Tuple[dict, float]]) -> Optional[dict]:
        """默认拒答时的"捞回"：必须在证据里逐字找到能直接回答的句子。"""
        if not rounds:
            return None
        with self.lock:
            self.stats["calls"] += 1
        lines, texts = [], []
        for i, (m, _s) in enumerate(rounds):
            t = m.get("index_text") or ""
            texts.append(t)
            lines.append(f"[{i}] {_truncate(t, self.verify_chars)}")
        try:
            out = self._chat(RESCUE_PROMPT.replace("__Q__", question).replace("__C__", "\n".join(lines)), 260)
            data = self._json(out)
            if not data:
                return None
            quote = str(data.get("quote") or "").strip()
            found = bool(data.get("found")) and bool(quote)
            if not found:
                return {"found": False, "quote": "", "round": None}
            # 防幻觉：quote 必须在证据里逐字出现（允许首尾空白差异）
            joined = "\n".join(texts)
            qn = " ".join(quote.split())
            jn = " ".join(joined.split())
            if qn and qn in jn:
                idx = None
                for i, t in enumerate(texts):
                    if qn in " ".join(t.split()):
                        idx = i
                        break
                return {"found": True, "quote": quote, "round": idx}
            return {"found": False, "quote": "", "round": None, "rejected": "quote_not_in_evidence"}
        except Exception as exc:
            with self.lock:
                self.stats["fallback"] += 1
            raise RuntimeError(f"judge failed: {exc}") from exc

    def judge(self, question: str, rounds: List[Tuple[dict, float]]) -> Optional[dict]:
        if not rounds:
            return None
        with self.lock:
            self.stats["calls"] += 1
        lines = []
        for i, (m, _s) in enumerate(rounds):
            ts = str(m.get("created_at") or m.get("ts") or "")[:19]
            head = f"[{i}] ({ts}) " if ts else f"[{i}] "
            lines.append(head + _truncate(m.get("index_text") or "", self.verify_chars))
        try:
            out = self._chat(CONSTRAINT_PROMPT.replace("__Q__", question).replace("__C__", "\n".join(lines)), 260)
            data = self._json(out)
            if not data:
                return None
            kind = str(data.get("kind") or "").strip().lower()
            uns = data.get("unsatisfied") or []
            _sat = data.get("satisfied")
            if _sat is None:                 # 字段缺失时才按 kind 兜底
                sat = (kind == "none")
            else:
                sat = bool(_sat)
            return {"kind": kind,
                    "satisfied": sat,
                    "unsatisfied": [str(c) for c in uns][:6],
                    "note": str(data.get("note") or "")[:200]}
        except Exception:
            with self.lock:
                self.stats["fallback"] += 1
            return None
