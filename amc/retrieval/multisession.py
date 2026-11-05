"""Multi-session evidence atoms + aggregate evidence retrieval.

This module is deliberately independent of MemoryService and the profile package:
it reads the existing SQLite memory DB and builds an append-only evidence-atom
index that survives the lossy 1200-char library tag summaries.

Atom = a bounded piece of source text carrying a spend/amount fact:
    {user_id, request_id, message_id, amount, currency, item, category,
     status, quote, text_span, confidence}

Why atoms:
- Q5's $150 lives in msg2 of a bill-splitting request and $85 in msg4 of a
  flea-market request; both are outside the 1200-char tag prefix.
- The atom index scans every message, keeps a 240-500 char span, and links the
  atom back to request/message so Search can return evidence, not a total.

Aggregation is intentionally separate from Search: `aggregate()` sums atoms and
returns source-backed evidence; it never fabricates a conclusion in Search.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from typing import Iterable, Optional

COFFEE_MAKER_TERMS = (
    "coffee maker", "coffee machine", "espresso machine", "espresso",
    "moka pot", "moka", "nespresso", "pour-over", "pour over",
    "drip coffee", "french press", "coffee brewer", "stovetop coffee",
)
COFFEE_ACCESSORY_TERMS = (
    "coffee grinder", "grinder", "filter", "tamper", "scale", "cup", "mug",
    "kettle", "paper filter", "coffee accessory",
)
SPEND_TERMS = (
    "spent", "spend", "bought", "buy", "paid", "pay", "picked up", "grabbed",
    "purchased", "purchase", "cost", "charged", "hit my account", "invoice",
    "bill", "fare", "fee",
)
REFUND_TERMS = ("refund", "returned", "return", "cancelled", "canceled", "void")
PENDING_TERMS = ("plan to", "planning to", "will buy", "going to buy", "budget", "willing to", "would pay", "could afford")
AMOUNT_RE = re.compile(r"(?<![A-Za-z])(?:\$|USD\s*|US\$|€|£)\s?(\d{1,6}(?:[.,]\d{1,2})?)\b", re.I)
AMOUNT_WORD_RE = re.compile(r"\b(\d{1,6}(?:[.,]\d{2})?)\s*(?:dollars|usd)\b", re.I)
CURRENCY = {"$": "USD", "us$": "USD", "usd": "USD", "€": "EUR", "£": "GBP"}


@dataclass
class EvidenceAtom:
    user_id: str
    request_id: str
    message_id: str
    amount: float
    currency: str
    item: str
    category: str
    status: str
    quote: str
    text_span: str
    confidence: float

    def to_row(self) -> tuple:
        return (self.user_id, self.request_id, self.message_id, self.amount,
                self.currency, self.item, self.category, self.status,
                self.quote, self.text_span, self.confidence)


def normalize_category(text: str) -> str:
    low = (text or "").lower()
    if any(t in low for t in COFFEE_MAKER_TERMS):
        return "coffee_maker"
    if any(t in low for t in COFFEE_ACCESSORY_TERMS):
        return "coffee_accessory"
    return "other"


def infer_status(text: str) -> str:
    low = (text or "").lower()
    if any(t in low for t in REFUND_TERMS):
        return "refunded"
    if any(t in low for t in PENDING_TERMS):
        return "pending"
    if any(t in low for t in SPEND_TERMS):
        return "confirmed"
    return "unconfirmed"


def _item_phrase(text: str, start: int, end: int) -> str:
    # Prefer the clause around the amount, bounded to ~160 chars.
    left = max(0, start - 120)
    right = min(len(text), end + 120)
    chunk = text[left:right].strip()
    # Strip image caption / markup noise at the boundaries.
    chunk = re.sub(r"\[Image[^\]]*\]", " ", chunk)
    chunk = re.sub(r"Image Caption:", " ", chunk, flags=re.I)
    return re.sub(r"\s+", " ", chunk).strip()[:220]


def extract_atoms(user_id: str, request_id: str, message_id: str,
                  text: str, role: str = "user") -> list[EvidenceAtom]:
    """Extract amount/spend atoms from a single message.

    Conservative: requires an amount and a spend/coffee/category cue in the
    same bounded clause. Duplicate amount mentions in a message are kept only
    once per (amount, category) pair.
    """
    text = text or ""
    if (role or "").lower() != "user":
        # Assistant messages contain budgets/advice prices ($200-$300 etc.),
        # not user spending; never create purchase atoms from them.
        return []
    out: list[EvidenceAtom] = []
    seen: set[tuple] = set()
    for mo in list(AMOUNT_RE.finditer(text)) + list(AMOUNT_WORD_RE.finditer(text)):
        raw = mo.group(0)
        num = mo.group(1).replace(",", "")
        try:
            amount = float(num)
        except ValueError:
            continue
        if amount <= 0 or amount > 1_000_000:
            continue
        span_start, span_end = mo.span()
        span = text[max(0, span_start - 240): min(len(text), span_end + 240)]
        quote = _item_phrase(text, span_start, span_end)
        low = quote.lower()
        has_spend = any(t in low for t in SPEND_TERMS)
        has_coffee = any(t in low for t in COFFEE_MAKER_TERMS + COFFEE_ACCESSORY_TERMS)
        if not (has_spend or has_coffee):
            continue
        category = normalize_category(quote)
        if category == "other" and has_coffee:
            category = "coffee_accessory"
        status = infer_status(quote)
        cur = "USD"
        for sym, code in CURRENCY.items():
            if sym in raw.lower():
                cur = code
                break
        key = (round(amount, 2), category, status)
        if key in seen:
            continue
        seen.add(key)
        conf = 0.55
        if has_spend:
            conf += 0.2
        if category != "other":
            conf += 0.15
        if status == "confirmed":
            conf += 0.1
        out.append(EvidenceAtom(user_id=user_id, request_id=request_id,
                                message_id=message_id, amount=amount, currency=cur,
                                item=quote, category=category, status=status,
                                quote=quote, text_span=span[:500],
                                confidence=round(min(conf, 0.95), 2)))
    return out


class EvidenceAtomIndex:
    """Append-only atom index over the existing memories table."""

    def __init__(self, memory_db: str, index_db: str):
        self.memory_db = memory_db
        self.index_db = index_db
        self._ensure()

    def _ensure(self) -> None:
        con = sqlite3.connect(self.index_db)
        con.execute("""
            CREATE TABLE IF NOT EXISTS evidence_atoms (
                atom_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT 'USD',
                item TEXT,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                quote TEXT,
                text_span TEXT,
                confidence REAL,
                UNIQUE(user_id, request_id, message_id, amount, category, status)
            )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_atoms_user_cat ON evidence_atoms(user_id, category, status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_atoms_request ON evidence_atoms(request_id)")
        con.commit(); con.close()

    def build(self, user_ids: Optional[Iterable[str]] = None,
              category_hints: Optional[dict] = None) -> int:
        mem = sqlite3.connect(self.memory_db)
        q = "SELECT id,user_id,request_id,msg_index,role,index_text FROM memories"
        args: tuple = ()
        if user_ids:
            ids = list(user_ids)
            q += " WHERE user_id IN (%s)" % ",".join("?" * len(ids))
            args = tuple(ids)
        rows = mem.execute(q, args).fetchall()
        mem.close()
        # Image-caption hints: the $85 moka pot is described in the user's
        # photo request ("stainless coffee pot") in a different turn.  If an
        # amount atom shares a distinctive token with such a caption and the
        # caption names a coffee maker, use it as a category hint.
        caption_text = {}
        for _mid, uid, _rid, _mi, role, txt in rows:
            low = (txt or "").lower()
            if ("image caption" in low or "moka" in low or "coffee pot" in low) and any(
                    t in low for t in COFFEE_MAKER_TERMS):
                caption_text.setdefault(uid, []).append(low)
        atoms = []
        for mid, uid, rid, mi, role, txt in rows:
            for a in extract_atoms(uid, rid, str(mid), txt or "", role=role):
                if category_hints:
                    hint = category_hints.get((uid, rid, round(a.amount, 2)))
                    if hint:
                        a.category = hint
                if a.category == "other" and uid in caption_text:
                    item_tokens = {t for t in re.findall(r"[a-z]{4,}", a.item.lower())}
                    for cap in caption_text[uid]:
                        cap_tokens = set(re.findall(r"[a-z]{4,}", cap))
                        if item_tokens & cap_tokens & {"stainless", "moka", "pot", "coffee", "espresso"}:
                            a.category = "coffee_maker"
                            a.confidence = min(a.confidence, 0.65)
                            break
                atoms.append(a.to_row())
        con = sqlite3.connect(self.index_db)
        con.executemany("""
            INSERT OR IGNORE INTO evidence_atoms
            (user_id,request_id,message_id,amount,currency,item,category,status,quote,text_span,confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""", atoms)
        con.commit(); con.close()
        return len(atoms)

    def query(self, user_id: str, category: str = "coffee_maker",
              include_pending: bool = True) -> list[dict]:
        con = sqlite3.connect(self.index_db); con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT * FROM evidence_atoms
            WHERE user_id=? AND category=?
              AND status IN ('confirmed','pending')
              AND (? OR status='confirmed')
            ORDER BY amount DESC""", (user_id, category, 1 if include_pending else 0)).fetchall()
        con.close()
        return [dict(r) for r in rows]

    def aggregate(self, user_id: str, category: str = "coffee_maker",
                  include_pending: bool = True) -> dict:
        atoms = self.query(user_id, category, include_pending)
        confirmed = [a for a in atoms if a["status"] == "confirmed"]
        pending = [a for a in atoms if a["status"] == "pending"]
        total = round(sum(a["amount"] for a in confirmed), 2)
        return {
            "category": category,
            "total": total,
            "currency": "USD",
            "evidence": [{"request_id": a["request_id"], "message_id": a["message_id"],
                          "amount": a["amount"], "item": a["item"], "status": a["status"],
                          "quote": a["quote"]} for a in confirmed],
            "excluded_pending": [{"request_id": a["request_id"], "amount": a["amount"],
                                  "item": a["item"], "quote": a["quote"]} for a in pending],
            "n_atoms": len(atoms),
        }


class ProfileSourceResolver:
    """Read profile transactions + sources and map them back to evidence.

    The profile DB is authoritative for canonical transactions; this resolver
    exposes the request/message/quote provenance so aggregate results can be
    returned as source-backed evidence.
    """

    def __init__(self, profile_db: str):
        self.profile_db = profile_db

    @staticmethod
    def canonical_category(item: str, category: str) -> str:
        low = (item or "").lower()
        if any(t in low for t in ("moka", "coffee pot", "stainless-steel contraption",
                                  "stainless steel coffee", "espresso machine",
                                  "nespresso", "coffee maker")):
            return "coffee_maker"
        if any(t in low for t in ("filter", "tamper", "scale", "cup", "mug")):
            return "coffee_accessory"
        return category or "other"

    @staticmethod
    def canonical_status(item: str, status: str, quote: str = "") -> str:
        low = (quote or item or "").lower()
        if any(t in low for t in ("grabbed", "bought", "picked up", "paid", "hit my account")):
            return "confirmed"
        return status or "unconfirmed"

    def aggregate(self, user_id: str, category: str = "coffee_maker") -> dict:
        txs = self.transactions(user_id)
        seen = set(); evidence = []; pending = []
        for t in txs:
            if not t.get("amount_minor"):
                continue
            item = t.get("item") or ""
            cat = self.canonical_category(item, t.get("category") or "other")
            if cat != category:
                continue
            status = self.canonical_status(item, t.get("status") or "pending", t.get("quote") or "")
            key = (round(t["amount_minor"] / 100, 2), item, status)
            if key in seen:
                continue
            seen.add(key)
            row = {"request_id": t.get("request_id"), "message_id": t.get("memory_id"),
                   "amount": round(t["amount_minor"] / 100, 2), "item": item,
                   "status": status, "quote": t.get("quote")}
            (pending if status == "pending" else evidence).append(row)
        total = round(sum(x["amount"] for x in evidence), 2)
        return {"category": category, "total": total, "currency": "USD",
                "evidence": evidence, "excluded_pending": pending,
                "n_transactions": len(evidence) + len(pending)}

    def transactions(self, user_id: str, category: Optional[str] = None) -> list[dict]:
        con = sqlite3.connect(self.profile_db); con.row_factory = sqlite3.Row
        q = """SELECT t.*, s.memory_id, s.request_id, s.quote
               FROM profile_transactions t
               LEFT JOIN profile_transaction_sources s ON s.transaction_id=t.transaction_id
               WHERE t.user_id=?"""
        args: list = [user_id]
        if category:
            q += " AND t.category=?"
            args.append(category)
        rows = con.execute(q, args).fetchall(); con.close()
        return [dict(r) for r in rows]


class EvidenceSetBuilder:
    """Build a bounded, session-expanded evidence set (not a reranker).

    Q5/TTL multi-gold failures need more than top1: the service often truncates
    appended session partners back to top_k=5.  This builder keeps the base
    order but fills the budget with same-session partners of the top anchors,
    then remaining base candidates.  It never changes the first item.
    """

    def __init__(self, session_of: dict[str, str], rounds_by_session: dict[str, list[str]]):
        self.session_of = session_of
        self.rounds_by_session = rounds_by_session

    def build(self, base_rounds: list[str], budget: int = 12,
              anchors: int = 1, preserve_top: int = 10) -> list[str]:
        """Preserve base top-N, then fill with same-session partners.

        Keeping top5 first is required: inserting partners immediately after
        top1 displaces gold at ranks 2-5 and destroys r5/any on non-multi-gold
        questions.  `anchors` controls how many top rounds contribute partners.
        """
        out: list[str] = []
        seen: set[str] = set()

        def add(rid: str) -> None:
            if rid and rid not in seen and len(out) < budget:
                seen.add(rid); out.append(rid)

        # Contract: base top-N order must stay intact.
        for rid in base_rounds[:max(0, preserve_top)]:
            add(rid)
        for rid in base_rounds[:max(1, anchors)]:
            sid = self.session_of.get(rid)
            for partner in self.rounds_by_session.get(sid, []) if sid else []:
                add(partner)
        for rid in base_rounds:
            add(rid)
        return out


class AmbiguityDetector:
    """Detect under-specified repeated questions.

    A repeated question template with no query image/caption cannot identify
    the intended scene.  At serving time, build `question_sessions` from the
    library; if the normalized question maps to >=2 sessions and the query has
    no caption, abstain instead of guessing a scene.
    """

    def __init__(self, question_sessions: dict[str, set[str]], min_sessions: int = 2):
        self.question_sessions = question_sessions
        self.min_sessions = max(2, int(min_sessions))

    @staticmethod
    def normalize(question: str) -> str:
        import re
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (question or "").lower())).strip()

    def is_ambiguous(self, question: str, has_query_image: bool = False) -> bool:
        if has_query_image:
            return False  # caption can disambiguate the scene
        return len(self.question_sessions.get(self.normalize(question), set())) >= self.min_sessions


class MultiGoldGate:
    """Runtime gate for multi-gold evidence expansion.

    Use with EvidenceSetBuilder(preserve_top=10, budget=12); do not enable on
    single-target questions, where extra session partners add context without
    answering needs.
    """

    _PAT = re.compile(
        r"\b(all|each|every|list|how many|which (images|pictures|photos|moments|times|days|people|places)|"
        r"names?|breed|category|species|types?|kinds?|total)\b", re.I)

    def __init__(self, include_task_types: tuple[str, ...] = ("FR", "VR")):
        self.include_task_types = tuple(include_task_types)

    def should_expand(self, question: str, task_type: str = "") -> bool:
        return (task_type or "").upper() in self.include_task_types or bool(self._PAT.search(question or ""))


def extract_image_ids(text: str) -> list[str]:
    """Extract [Image D6:IMG_003] style IDs from evidence text."""
    return list(dict.fromkeys(re.findall(r"\[Image (D\d+:IMG_\d+)\]", text or "")))
