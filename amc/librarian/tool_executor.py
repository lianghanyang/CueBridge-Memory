"""Execute validated librarian tool plans with one user scope and a hard budget."""
from __future__ import annotations
from typing import Any
from .tool_planner import ALLOWED_TOOLS
from ..tools import find_amounts, find_object, list_records, grep_records, join_by_time, ocr, OCRTool
from ..tools.keyword_search import keyword_search as _keyword_search

_IMPL = {"find_amounts": find_amounts, "find_object": find_object,
         "list_records": list_records, "grep_records": grep_records,
         "join_by_time": join_by_time, "ocr": ocr,
         "keyword_search": _keyword_search}

# LLM arguments are data, not an extension point.  In particular, never allow a
# model to smuggle SQL, a store, arbitrary records, or an answer-shaped field.
_ARG_KEYS = {
    "find_amounts": {"object_hint", "limit"},
    "find_object": {"object_name", "limit"},
    "list_records": {"date_from", "date_to", "role", "has_image", "limit"},
    "grep_records": {"pattern", "limit"},
    "join_by_time": {"t1", "t2", "window_days", "limit"},
    "ocr": {"record_id", "question"},
    # keyword_search：terms/synonyms 来自 planner，query 由 execute() 注入（A 需要问题实词）
    "keyword_search": {"terms", "synonyms", "limit", "query"},
}


def _bounded_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def execute(calls: list[dict], user_id: str, *, store: Any = None, records=None,
            budget: int = 5, ocr_runner=None, ocr_max_calls: int = 2,
            query: str = "") -> list[dict]:
    out = []
    # The product budget is at most three planner calls, even if a caller passes 5.
    for call in calls[:min(3, max(0, int(budget)))]:
        if not isinstance(call, dict) or call.get("tool") not in ALLOWED_TOOLS:
            continue
        name = call["tool"]
        raw_args = call.get("arguments")
        args = raw_args if isinstance(raw_args, dict) else {}
        args = {k: v for k, v in args.items() if k in _ARG_KEYS[name]}
        if name == "join_by_time":
            # LLMs often choose this tool without supplying both text anchors. Make
            # that a harmless no-op rather than a TypeError.
            if not isinstance(args.get("t1"), str) or not isinstance(args.get("t2"), str):
                args["t1"] = args["t2"] = ""
            args["window_days"] = _bounded_int(args.get("window_days", 2), 2, 1, 30)
            args["limit"] = _bounded_int(args.get("limit", 100), 100, 1, 200)
        elif name in {"list_records", "grep_records", "find_amounts", "find_object"}:
            args["limit"] = _bounded_int(args.get("limit", 50), 50, 1, 200)
        if name == "grep_records" and not isinstance(args.get("pattern"), str):
            args["pattern"] = ""
        if name == "find_object" and not isinstance(args.get("object_name"), str):
            args["object_name"] = ""
        if name == "find_amounts" and args.get("object_hint") is not None and not isinstance(args["object_hint"], str):
            args.pop("object_hint", None)
        if name == "keyword_search":
            # A：把原始问题传进工具，工具会把问题实词并入检索词
            args["query"] = query or ""
        if name != "ocr":
            args["user_id"] = user_id
        if name != "ocr" and records is not None:
            args["records"] = records
        elif name != "ocr" and store is not None:
            args["store"] = store
        if name == "ocr":
            args["runner"] = ocr_runner
            args["max_calls"] = min(max(1, int(args.get("max_calls", ocr_max_calls))), ocr_max_calls)
            if not args.get("image_bytes") and store is not None and args.get("record_id"):
                # Resolve only a same-user memory and its local blob; never accept a
                # filesystem path from the model.
                try:
                    candidates = store.list_memories(user_id)
                    target = str(args["record_id"])
                    memory = next((r for r in candidates if str(r.get("id")) == target or
                                   str(r.get("request_id")) == target), None)
                    for part in (memory or {}).get("content", []):
                        if part.get("type") == "image_ref" and part.get("sha256"):
                            args["image_bytes"] = store.blob_path(
                                part["sha256"], part.get("fmt", "jpeg")).read_bytes()
                            break
                except Exception:
                    pass
        try:
            out.append({"tool": name, "rows": _IMPL[name](**args)})
        except Exception as exc:
            # One malformed tool call must not turn a search into a 500.
            out.append({"tool": name, "rows": [], "error": type(exc).__name__})
    return out
