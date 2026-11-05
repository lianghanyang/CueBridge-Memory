"""探测通道注册表：把"同构的题类探测"收成一张表。

合并前：`service.py` 里有 4 段几乎逐字重复的

    if not _sufficient:
        try:
            from .X_probe import is_X_question, probe as _xprobe
            if is_X_question(q_text):
                _xc = [...造候选...]
                _xr = _xprobe(q_text, _xc, url, key, model)
                if _xr:
                    _sufficient = True
                    _x_note = ...
        except Exception:
            logger.exception("X probe failed")

合并后：`run_chain()` 一张表驱动，新增题型 = 加一个 `Spec`。

设计约束（保持与原实现**逐位一致**）：
  * 按 `SPECS` 顺序依次尝试，**命中即停**（等价于原来的 `if not _sufficient` 链）
  * 每个通道的开关、候选来源、成功判据、note 构造都由 `Spec` 描述
  * 任何异常只影响该通道，不影响其余（等价于原来的 per-block try/except）
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 通道定义
@dataclass
class Spec:
    name: str
    detect: Callable[[str], bool]           # 题类判定
    run: Callable[..., Optional[Dict]]      # probe(query, cands, base_url, api_key, model)
    on_hit: Callable[[Dict], Dict[str, Any]]  # 命中时的产出（写入哪几个 note 变量）
    env: str = ""                           # 开关名；为空表示无开关
    cands: str = "pool"                     # "pool"（id/date/text）或 "cand"（round_id/date/text）


def _load():
    """延迟导入，避免与 service 的加载顺序耦合。"""
    from .temporal_probe import is_date_question, probe as temporal_probe
    from .conflict_probe import is_conflict_question, probe as conflict_probe
    from .ordering_probe import is_ordering_question, probe as ordering_probe
    from .open_end_probe import is_open_end, probe as open_end_probe
    return (is_date_question, temporal_probe,
            is_conflict_question, conflict_probe,
            is_ordering_question, ordering_probe,
            is_open_end, open_end_probe)


def build_specs() -> List[Spec]:
    (is_date, temporal, is_conflict, conflict,
     is_ordering, ordering, is_open, open_end) = _load()

    specs = [
        Spec(name="temporal", detect=is_date, run=temporal, env="AMC_TEMPORAL_PROBE",
             cands="pool",
             on_hit=lambda r: {"time": r.get("date")} if r.get("date") else {}),
        Spec(name="conflict", detect=is_conflict, run=conflict, env="",
             cands="pool",
             on_hit=lambda r: {"conflict_ids": r.get("ids")} if r.get("ids") else {}),
        Spec(name="ordering", detect=is_ordering, run=ordering, env="",
             cands="pool",
             on_hit=lambda r: ({"order": f"{r['answer']}（{r.get('a_date')} vs {r.get('b_date')}）"}
                               if r.get("answer") else {})),
        Spec(name="open_end", detect=is_open, run=open_end, env="AMC_OPEN_END",
             cands="cand",
             on_hit=lambda r: ({"open": r.get("answer_draft") or "（已从多轮综合出答案，部分）"}
                               if (r.get("synthesized") or r.get("partial_ok")) else {})),
    ]
    return specs


# ---------------------------------------------------------------- 执行
def _enabled(spec: Spec) -> bool:
    if not spec.env:
        return True
    return os.getenv(spec.env, "1" if spec.env == "AMC_TEMPORAL_PROBE" else "0") not in ("0", "false", "False")


def _build_cands(spec: Spec, pool: List[str], cand: List[Dict], lib) -> List[Dict[str, str]]:
    if spec.cands == "cand":
        return [{"request_id": c.get("round_id"), "date": c.get("date", ""), "text": c.get("text", "")}
                for c in cand]
    return [{"request_id": rid,
             "date": (lib.round(rid) or {}).get("date", ""),
             "text": (lib.round(rid) or {}).get("text", "")}
            for rid in pool[:20]]


def run_chain(q_text: str, cand: List[Dict], pool: List[str], lib,
              base_url: str, api_key: str, model: str) -> Dict[str, Any]:
    """依次尝试各探测通道，**命中即停**。

    返回 {"hit": 通道名 | None, "out": {note 变量...}}
    """
    out: Dict[str, Any] = {}
    for spec in build_specs():
        if not _enabled(spec):
            continue
        try:
            if not spec.detect(q_text):
                continue
            cs = _build_cands(spec, pool, cand, lib)
            if not cs:
                continue
            res = spec.run(q_text, cs, base_url, api_key, model)
            if not res:
                continue
            got = spec.on_hit(res) or {}
            if not got:
                continue
            out.update(got)
            logger.info("[probe:%s] hit -> %s", spec.name,
                        str(list(got.values())[0])[:60])
            return {"hit": spec.name, "out": out}
        except Exception:
            logger.exception("probe %s failed", spec.name)
    return {"hit": None, "out": {}}
