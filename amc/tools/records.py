"""Bounded read-only record enumeration and text matching tools."""
from __future__ import annotations
import re
from typing import Any, Iterable, Optional

def list_records(user_id: str, *, store: Any = None, records: Optional[Iterable[dict]] = None,
                 date_from: str = "", date_to: str = "", role: str = "",
                 has_image: Optional[bool] = None, limit: int = 200) -> list[dict]:
    if records is None:
        if store is None: raise ValueError("store or records is required")
        records = store.list_memories(user_id)
    out=[]
    for r in records:
        if r.get("user_id", user_id) != user_id: continue
        if role and r.get("role") != role: continue
        date=str(r.get("created_at") or "")[:10]
        if date_from and date < date_from or date_to and date > date_to: continue
        if has_image is not None and bool(r.get("n_images", 0)) != has_image: continue
        out.append(r)
        if len(out) >= max(1,int(limit)): break
    return out

def grep_records(user_id: str, pattern: str = "", *, store: Any = None,
                 records: Optional[Iterable[dict]] = None, limit: int = 200) -> list[dict]:
    # Incomplete model plans are safe no-ops; regex is bounded and never interpreted
    # as SQL or executable code.
    if not isinstance(pattern, str) or not pattern.strip():
        return []
    pattern = pattern[:200]
    limit = min(200, max(1, int(limit)))
    rx=re.compile(pattern, re.I)
    return [r for r in list_records(user_id, store=store, records=records, limit=limit*4)
            if rx.search(str(r.get("index_text") or ""))][:max(1,int(limit))]
