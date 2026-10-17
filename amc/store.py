import base64
import hashlib
import httpx
import json
import logging
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MAX_ADD_IMAGES, MAX_SINGLE_IMAGE
from .contract import ContractError
from .ids import memory_id
from .imaging import decode_data_uri, sniff_format

logger = logging.getLogger("amc.store")


class MemoryStore:
    def __init__(
        self,
        db_path,
        blob_dir,
        timezone_offset_hours: int = 8,
        reset: bool = False,
    ):
        self.db_path = str(db_path)
        self.blob_dir = Path(blob_dir)
        self.tz = timezone(timedelta(hours=timezone_offset_hours))
        self.lock = threading.RLock()
        self.version = defaultdict(int)

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            if reset:
                for suffix in ("", "-wal", "-shm"):
                    candidate = Path(self.db_path + suffix)
                    if candidate.exists():
                        candidate.unlink()

        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        with self.conn:
            self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                -- 单个 user 可能有上万条 memory（ATM-Bench: 1 user / 11k memories），
                -- ORDER BY 排序会退化为文件临时表；放内存里更快也更稳。
                PRAGMA temp_store=MEMORY;

                CREATE TABLE IF NOT EXISTS requests(
                    user_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, request_id)
                );

                CREATE TABLE IF NOT EXISTS memories(
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    msg_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    ts INTEGER,
                    created_at TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    index_text TEXT NOT NULL,
                    n_images INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id, ts, request_id, msg_index);
                """
            )

    def blob_path(self, sha256: str, fmt: str) -> Path:
        return self.blob_dir / sha256[:2] / f"{sha256}.{fmt}"

    def _store_image(self, url: str) -> Dict[str, Any]:
        # 支持两种形态：内联 data URI（首选）与 http(s) 外链（AML 契约允许 URL）。
        # 外链抓取失败时抛 422 并附上原因，便于定位是网络问题还是格式问题。
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            try:
                resp = httpx.get(url, timeout=20.0, follow_redirects=True)
                resp.raise_for_status()
                raw = resp.content
            except Exception as exc:
                raise ContractError(422, f"cannot fetch image url: {exc}") from exc
        else:
            try:
                raw = decode_data_uri(url)
            except Exception as exc:
                raise ContractError(422, f"invalid image: {exc}") from exc

        if len(raw) > MAX_SINGLE_IMAGE:
            raise ContractError(413, "single image exceeds 10 MiB")

        fmt = sniff_format(raw)
        if fmt is None:
            raise ContractError(422, "image is invalid or unsupported")

        sha256 = hashlib.sha256(raw).hexdigest()
        path = self.blob_path(sha256, fmt)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)

        return {
            "type": "image_ref",
            "sha256": sha256,
            "fmt": fmt,
            "bytes": len(raw),
        }

    def _timestamp_iso(self, timestamp: Optional[int]) -> str:
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()
        return datetime.fromtimestamp(timestamp / 1000, tz=self.tz).isoformat()

    def add_request(self, req: Dict[str, Any]) -> str:
        user_id = req["user_id"]
        request_id = req["request_id"]
        payload_hash = hashlib.sha256(
            json.dumps(
                req,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        with self.lock:
            old = self.conn.execute(
                "SELECT payload_hash FROM requests "
                "WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            ).fetchone()

            if old is not None:
                if old["payload_hash"] != payload_hash:
                    raise ContractError(
                        409,
                        "same request_id was previously used with a different payload",
                    )
                return "replayed"

            rows = []
            total_image_bytes = 0

            for index, message in enumerate(req["messages"]):
                normalized = []
                texts = []
                image_count = 0

                # 契约允许 content 为**纯字符串**（标准 chat 格式）或 ContentPart[]；
                # 未归一时用字符串当列表迭代会 TypeError → Add 直接 500。
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                for part in content:
                    if part.get("type") == "image_url":
                        ref = self._store_image(part["image_url"]["url"])
                        total_image_bytes += ref["bytes"]
                        image_count += 1
                        normalized.append(ref)
                    else:
                        normalized.append({"type": "text", "text": part["text"]})
                        texts.append(part["text"])

                if total_image_bytes > MAX_ADD_IMAGES:
                    raise ContractError(
                        413,
                        "decoded images in one Add request exceed 30 MiB",
                    )

                created_at = self._timestamp_iso(message.get("timestamp"))
                prefix = f"[{created_at[:10]} {message['role']}] "
                rows.append(
                    (
                        memory_id(user_id, request_id, index),
                        user_id,
                        req["session_id"],
                        request_id,
                        index,
                        message["role"],
                        message.get("timestamp"),
                        created_at,
                        json.dumps(normalized, ensure_ascii=False),
                        prefix + "\n".join(texts),
                        image_count,
                    )
                )

            with self.conn:
                self.conn.execute(
                    "INSERT INTO requests VALUES (?,?,?,?,?)",
                    (
                        user_id,
                        request_id,
                        req["session_id"],
                        payload_hash,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self.conn.executemany(
                    "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )

            self.version[user_id] += 1
            return "inserted"

    def list_memories(self, user_id: str) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE user_id=? "
                "ORDER BY COALESCE(ts, 0), request_id, msg_index",
                (user_id,),
            ).fetchall()

        output = []
        for row in rows:
            item = dict(row)
            item["content"] = json.loads(item.pop("content_json"))
            output.append(item)
        return output

    def render_content(
        self,
        content: List[Dict[str, Any]],
        hydrate: bool,
        remaining_budget: int,
        allow_image: bool = True,
    ):
        output = []
        used = 0

        for part in content:
            if part["type"] != "image_ref":
                output.append(part)
                continue

            if not hydrate or not allow_image:
                output.append({
                    "type": "text",
                    "text": "[Image: Visual details captured in context textual descriptions]",
                })
                continue

            if used + part["bytes"] > remaining_budget:
                output.append({
                    "type": "text",
                    "text": "[image omitted: response image budget]",
                })
                continue

            # 健壮性：blob 缺失/不可读时退化为文本占位，绝不因此让 /search 失败
            try:
                raw = self.blob_path(part["sha256"], part["fmt"]).read_bytes()
            except OSError:
                logger.warning("blob missing for %s; degrading to text placeholder",
                               part.get("sha256", "")[:12])
                output.append({
                    "type": "text",
                    "text": "[image unavailable: blob missing]",
                })
                continue
            output.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/{part['fmt']};base64,"
                        f"{base64.b64encode(raw).decode('ascii')}"
                    )
                },
            })
            used += len(raw)

        if not output:
            output = [{"type": "text", "text": "[empty memory]"}]
        return output, used

    def set_index_text(self, user_id: str, transform):
        memories = self.list_memories(user_id)
        rows = [(transform(memory), memory["id"]) for memory in memories]

        with self.lock, self.conn:
            self.conn.executemany(
                "UPDATE memories SET index_text=? WHERE id=?",
                rows,
            )
        self.version[user_id] += 1

    def purge_user(self, user_id: str):
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            self.conn.execute("DELETE FROM requests WHERE user_id=?", (user_id,))
        self.version[user_id] += 1

    def stats(self):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT COUNT(*) memories,
                       COUNT(DISTINCT user_id) users,
                       COALESCE(SUM(n_images), 0) image_refs
                FROM memories
                """
            ).fetchone()
            requests = self.conn.execute(
                "SELECT COUNT(*) FROM requests"
            ).fetchone()[0]

        return {
            "memories": row["memories"],
            "users": row["users"],
            "image_refs": row["image_refs"],
            "requests": requests,
            "blobs": sum(1 for path in self.blob_dir.rglob("*") if path.is_file()),
        }
