"""Bounded LLM router for librarian evidence tools.

The model chooses tools and arguments; it never receives SQL access and never
returns an answer or aggregate. Missing/invalid LLM configuration falls back to
small deterministic routing rules.
"""
from __future__ import annotations
import json, re
from typing import Any, Dict, Optional

ALLOWED_TOOLS = {"list_records", "grep_records", "find_amounts", "find_object",
                 "join_by_time", "ocr", "keyword_search"}
PROMPT = """You are a librarian tool router. Do not answer the question. Choose at most 3 read-only tools.
Return JSON only: {{\"calls\":[{{\"tool\":\"find_amounts\",\"arguments\":{{...}}}}]}}
Allowed tools: list_records, grep_records, find_amounts, find_object, join_by_time, ocr, keyword_search.
Use keyword_search for a **broad, user-wide lexical lookup** when the question is phrased abstractly or you are unsure which record holds the answer: keyword_search({"terms":["<literal words from the question>"],"limit":100}).
The user_id is already supplied by the host; never include SQL, file paths, totals, or final answers.
Use find_amounts for spending/price/total questions; find_object for synonyms or cross-record object identity;
use join_by_time only for temporal/session linking, and include BOTH non-empty string arguments t1 and t2;
use grep_records only with a non-empty literal pattern; use ocr only when a candidate image needs a specific visual check.
Arguments: find_amounts={{object_hint?,limit?}}; find_object={{object_name,limit?}};
list_records={{date_from?,date_to?,role?,has_image?,limit?}}; grep_records={{pattern,limit?}};
join_by_time={{t1,t2,window_days?,limit?}}; ocr={{record_id,question}}.
Limits are 1..200. If required evidence anchors are unknown, omit that tool instead of guessing.
Question: {question}
"""

def _fallback(question: str) -> list[dict]:
    q = question or ""; calls=[]
    if re.search(r"how much|total|price|cost|paid|spent|amount|多少钱|总价|花费", q, re.I):
        hint = None
        if re.search(r"coffee makers?|espresso|moka|pour.?over", q, re.I): hint = "coffee maker"
        args = {"limit": 50}
        if hint: args["object_hint"] = hint
        calls.append({"tool": "find_amounts", "arguments": args})
    if re.search(r"coffee|espresso|moka|pour.?over|same object|same item|那个|同一个", q, re.I):
        calls.append({"tool": "find_object", "arguments": {"object_name": "coffee maker", "limit": 50}})
    if re.search(r"before|after|earlier|later|same session|之前|之后|先后", q, re.I):
        calls.append({"tool": "join_by_time", "arguments": {"window_days": 2, "limit": 100}})
    return calls[:3]

def plan(question: str, *, base_url: str = "", api_key: str = "", model: str = "gpt-4o-mini",
         timeout: float = 8.0) -> list[dict]:
    """Return validated tool calls. LLM failure is a deterministic fallback."""
    if not (base_url and model): return _fallback(question)
    from ..llm_json import call_json
    obj = call_json(base_url, api_key, model, PROMPT.format(question=(question or "")[:1200]),
                    timeout=timeout, retries=1, max_tokens=350)
    calls = obj.get("calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list): return _fallback(question)
    out=[]
    for call in calls[:3]:
        if not isinstance(call, dict) or call.get("tool") not in ALLOWED_TOOLS: continue
        args=call.get("arguments")
        if not isinstance(args, dict): args={}
        args={k:v for k,v in args.items() if k not in {"sql","query_result","answer","total"}}
        out.append({"tool":call["tool"],"arguments":args})
    return out or _fallback(question)
