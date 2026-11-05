"""级联阶段导出（只读、零副作用）：把"有排序产出"的每一级结果落盘，用于逐阶段 gold 召回。

开关：文件 `data/store/stage_dump.on` 存在才写。删除该文件即关闭。
输出：`data/store/stage_dump.jsonl`，每行 {ts, sid, uid, q, stage, n, ids}
  sid = 每次请求的短 id（同一个 sid 的多个 stage 可拼成一次级联）
阶段（按执行顺序）：
  fused      BM25+dense → RRF 融合后（未 rerank）
  ce        本地 cross-encoder rerank 后（Qwen3-Reranker-4B，candidates=30）
  llm       LLM rerank 后（evidence_set + round_rank）
  retrieve  服务层后处理（relation/vl/multimodal/dedup/视觉带图优先/截断）后
  library   记忆图书馆追加后
  ab        AB guided 注入后（含 duration/p_ref 前置）
  final     finalize 出口（= 平台看到的顺序）
"""
from __future__ import annotations
import json, os, time

_ON = os.path.join("data", "store", "stage_dump.on")
_OUT = os.path.join("data", "store", "stage_dump.jsonl")
#   data/store/stage_dump.stages 内容如 `fused,ce,llm,retrieve,library,ab,final`
#   · 文件不存在 ⇒ 全导（默认，什么都不变）
#   · 文件存在 ⇒ 只导列出的阶段（未知名字忽略）
#   · 每次写之前读 ⇒ 改文件即生效，不用重启
_STAGES = os.path.join("data", "store", "stage_dump.stages")


def enabled() -> bool:
    try:
        return os.path.exists(_ON)
    except Exception:
        return False


def _stage_allowed(stage: str) -> bool:
    try:
        if not os.path.exists(_STAGES):
            return True
        raw = ""
        with open(_STAGES, "r", encoding="utf-8") as fh:
            raw = fh.read() or ""
        allow = {x.strip().lower() for x in raw.replace("\n", ",").split(",") if x.strip()}
        if not allow:
            return True
        return stage.lower() in allow
    except Exception:
        return True


def dump(stage: str, records, uid=None, q=None, sid=None) -> None:
    if not enabled() or not _stage_allowed(stage):
        return
    try:
        #   120→100 省 17%；每个 id ≈28 字节 ⇒ 单阶段 ~2.8KB、7 级 ~20KB/请求。
        ids = []
        for r in (records or [])[:100]:
            if isinstance(r, (list, tuple)):
                m = r[0] if r else {}
                ids.append(str((m or {}).get("id")))
            elif isinstance(r, dict):
                ids.append(str(r.get("id")))
        rec = {"ts": round(time.time(), 3), "sid": sid or "", "uid": str(uid or ""),
               "q": (q if isinstance(q, str) else json.dumps(q, ensure_ascii=False))[:200],
               "stage": stage, "n": len(records or []), "ids": ids}
        with open(_OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
