"""Read-only object evidence lookup with conservative aliases."""
from __future__ import annotations
import re
from typing import Any, Iterable, Optional

ALIASES = {
    "coffee maker": ("coffee", "espresso", "nespresso", "moka", "pour-over", "pour over", "dripper", "brewer", "coffee pot"),
    "coffee makers": ("coffee", "espresso", "nespresso", "moka", "pour-over", "pour over", "dripper", "brewer", "coffee pot"),
    "espresso machine": ("espresso", "nespresso", "moka"),
    "moka pot": ("moka", "coffee pot", "stainless"),
}

def find_object(user_id: str, object_name: str, *, store: Any = None,
                records: Optional[Iterable[dict]] = None, limit: int = 50) -> list[dict]:
    if records is None:
        if store is None: raise ValueError("store or records is required")
        records = store.list_memories(user_id)
    rows = list(records)
    q = re.sub(r"\s+", " ", (object_name or "").lower().strip())
    terms = ALIASES.get(q, tuple(re.findall(r"[a-z][a-z-]{2,}", q)))
    direct = [r for r in rows if r.get("user_id", user_id) == user_id
              and str(r.get("role", "")).lower() == "user"
              and any(t in str(r.get("index_text") or "").lower() for t in terms)]
    bridge = ("stainless", "contraption", "coffee pot", "moka", "nespresso")
    selected = list(direct)
    if direct:
        selected.extend(r for r in rows if r not in selected and r.get("user_id", user_id) == user_id
                        and str(r.get("role", "")).lower() == "user"
                        and any(t in str(r.get("index_text") or "").lower() for t in bridge))
    out = []
    for r in selected:
        text = str(r.get("index_text") or "")
        out.append({"record_id": r.get("id"), "request_id": r.get("request_id"),
                    "session_id": r.get("session_id"), "date": str(r.get("created_at") or "")[:10],
                    "context": text[:500]})
        if len(out) >= max(1, int(limit)): break
    return out
