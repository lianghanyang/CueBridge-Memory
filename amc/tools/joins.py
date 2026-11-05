"""Bounded same-session adjacency joins; never writes to the memory DB."""
from __future__ import annotations
from typing import Any, Iterable, Optional

def join_by_time(user_id: str, t1: str = "", t2: str = "", window_days: int = 2, *,
                 store: Any = None, records: Optional[Iterable[dict]] = None,
                 limit: int = 100) -> list[dict]:
    # Missing anchors are a normal incomplete-plan case; do not raise or guess.
    if not isinstance(t1, str) or not isinstance(t2, str) or not t1.strip() or not t2.strip():
        return []
    t1, t2 = t1[:200].strip().lower(), t2[:200].strip().lower()
    window_days = min(30, max(1, int(window_days)))
    limit = min(200, max(1, int(limit)))
    if records is None:
        if store is None: raise ValueError("store or records is required")
        records = store.list_memories(user_id)
    rows=[r for r in records if r.get("user_id",user_id)==user_id]
    left=[r for r in rows if t1.lower() in str(r.get("index_text") or "").lower()]
    right=[r for r in rows if t2.lower() in str(r.get("index_text") or "").lower()]
    out=[]
    for a in left:
        for b in right:
            if a.get("id")==b.get("id"): continue
            if a.get("session_id") != b.get("session_id"): continue
            if abs(int(a.get("msg_index",0))-int(b.get("msg_index",0))) > max(1,window_days*10): continue
            out.append({"left_record_id":a.get("id"),"right_record_id":b.get("id"),
                        "session_id":a.get("session_id"),"left_context":a.get("index_text",""),
                        "right_context":b.get("index_text","")})
            if len(out)>=max(1,int(limit)): return out
    return out
