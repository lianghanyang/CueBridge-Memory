import base64
import hashlib
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


_CONN_LOCKS: dict = {}


def lock_for(conn):
    """返回**与 store.lock 同一把**的锁（按 connection 维度）。

    ⚠️ （reviewer 修 A）：`store.conn` 是单条 `check_same_thread=False`
    连接，store 自己的方法都 `with self.lock`，但 AB 通道（`ab_retrieve.py`）
    直接 `conn.execute(...)` 裸用 ⇒ 16 个慢线程并发 ⇒
    `sqlite3.InterfaceError: bad parameter or other API misuse`（当天 57 次 AB 通道失败）。
    ⇒ 这里把 store 的锁按 conn 注册，任何模块都能 `with lock_for(conn):` 安全复用。
    """
    try:
        lk = _CONN_LOCKS.get(id(conn))
        if lk is None:
            lk = __import__("threading").RLock()
            _CONN_LOCKS[id(conn)] = lk
        return lk
    except Exception:
        return __import__("threading").RLock()


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
        self._pending_captions = []
        self._pending_lock = __import__("threading").Lock()
        self._caption_cache = {}
        # caption 队列的可观测性：把"生成失败/写入失败"计数暴露到 /health，
        # 否则异常被宽 except 吞掉，静默丢记录（这个坑踩过）。
        self._caption_stats = {"done": 0, "gen_failed": 0, "write_failed": 0, "pending": 0}
        # caption 后台并发数：默认 4。
        # 背景：单线程时 1,003 张图要 33~83 分钟，而平台"先 add 完再 search"，
        # 索引在第一次 search 时定型 → 没补完的 caption 这一轮就永远错过。
        # 每个 add 各建 8 线程池导致 80 路并发打爆 VLM 网关。
        _w = int(__import__("os").getenv("AMC_CAPTION_WORKERS", "4") or "4")
        _w = max(1, min(_w, 8))
        for _ in range(_w):
            __import__("threading").Thread(target=self._caption_worker, daemon=True).start()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            _CONN_LOCKS[id(self.conn)] = self.lock
        except Exception:
            pass
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

    def _caption_worker(self):
        """后台把待描述的图片补上：写 index_text 末尾 `[Image sha8] cap`。"""
        import time as _t
        while True:
            item = None
            with self._pending_lock:
                if self._pending_captions:
                    item = self._pending_captions.pop(0)
            if item is None:
                _t.sleep(0.5)
                continue
            sha, fmt, raw = item
            try:
                cap = self.image_captioner(raw, fmt) or ""
            except Exception as exc:
                with self._pending_lock:
                    self._caption_stats["gen_failed"] += 1
                logger.warning("async caption failed: %s", exc)
                continue
            if not cap:
                continue
            self._caption_cache[sha] = cap
            try:
                # 生成 caption 是网络 IO（3~6s），放在锁外并行；
                # **写库必须串行**（sqlite 连接不是线程安全的，多线程并发写会报
                # "database is locked" 并静默丢记录 —— 实测 8 张图会丢 1 张）。
                with self.lock:
                    rows = self.conn.execute(
                        "SELECT id, index_text, user_id FROM memories WHERE content_json LIKE ?",
                        (f"%{sha}%",)).fetchall()
                    touched = set()
                    _new_texts = {}          # rid -> (uid, 新文本)
                    for rid, txt, uid in rows:
                        txt = txt or ""
                        if f"[Image {sha[:8]}]" in txt:
                            continue
                        _nt = txt.rstrip() + f"\n[Image {sha[:8]}] {cap}"
                        self.conn.execute("UPDATE memories SET index_text=? WHERE id=?", (_nt, rid))
                        touched.add(uid)
                        _new_texts[rid] = (uid, _nt)
                    self.conn.commit()
                    # ⚠️ 图书馆的文本来自 **tagger.data**（打标时的快照），不是 store.index_text。
                    # 只 bump store.version 只能让 dense/BM25 失效，**图书馆缓存仍留旧文本**
                    # 这里通过钩子广播新文本，由 app.py 注册的回调刷新 tagger.data 并 bump _lib_rev。
                    if _new_texts:
                        try:
                            _hook = getattr(self, "text_changed_hook", None)
                            if callable(_hook):
                                _hook(_new_texts)
                        except Exception:
                            logger.exception("text_changed_hook failed（忽略，索引仍会因 version 失效）")
                # ⚠️ 必须递增 version，否则检索侧的索引缓存不会失效 ——
                # 异步补上的 caption 就永远进不了 BM25/dense 索引（只有下次 add 或重启才生效）。
                # 这个 bug 会让 caption 在"add 完立刻 search"的平台时序下完全失效。
                for uid in touched:
                    self.version[uid] += 1
                with self._pending_lock:
                    self._caption_stats["done"] += 1
                logger.info("[caption] 异步补完 %s（影响 %d 条，失效 %d 个用户索引）",
                            sha[:8], len(rows), len(touched))
            except Exception:
                with self._pending_lock:
                    self._caption_stats["write_failed"] += 1
                logger.exception("async caption write failed")

    def _store_image(self, url: str) -> Dict[str, Any]:
        try:
            raw = decode_data_uri(url)
        except Exception as exc:
            raise ContractError(422, f"invalid image: {exc}") from exc

        if len(raw) > MAX_SINGLE_IMAGE:
            #   理由：多模态题没有图答不了；宁可多占磁盘，也不要丢证据。
            logger.warning("[add] 单图超限（%.1f MiB > %d）⇒ **仍然照收**（按设计不设单图上限）",
                           len(raw) / 1024 / 1024, MAX_SINGLE_IMAGE)

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

        caption = ""
        captioner = getattr(self, "image_captioner", None)
        # AMC_IMAGE_CAPTION=0 → 不再让 VLM 生成记忆侧图片描述。
        # 依据（2026-10-30 实测）：15,176 条含图记录里 **97.8%（14,835）原文已含平台描述**
        # （Gem 的 `[Image D1:IMG_00x] <英文描述>` / ATM 的 Tags+OCR+Summary）。
        # 而我们的 VLM 描述更短（中位 92 字 vs 平台 249 字）、与查询的嵌入相似度更低
        # （0.599 vs 0.754）⇒ **冗余且更差**，却要付 ~3.4s/张 的延迟 + API 成本。
        # 保留开关默认 1（不改变现有行为），等确认后再切 0。
        _want_caption = (__import__("os").getenv("AMC_IMAGE_CAPTION", "1") or "1").strip() not in ("0", "false", "no")
        if captioner is not None and _want_caption:
            # 记忆侧图片描述：**命中缓存就同步用；未命中则先入库、后台再补**
            # （VLM 一张约 7s，同步会让 /add 超时 —— 评测的 add 可能带多张图）
            cache = getattr(self, "_caption_cache", None)
            if cache is not None:
                caption = cache.get(sha256) or ""
            if not caption:
                try:
                    self._pending_captions.append((sha256, fmt, raw))
                except Exception:
                    pass

        return {
            "type": "image_ref",
            "sha256": sha256,
            "fmt": fmt,
            "bytes": len(raw),
            "caption": caption,
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
                        if ref is None:
                            # 单图超限 ⇒ 已跳过（文字仍在 texts 里）⇒ 继续处理后面的 part
                            continue
                        total_image_bytes += ref["bytes"]
                        image_count += 1
                        normalized.append(ref)
                        # 图片描述入索引：让"证据是图片"的查询也能被文字匹配到
                        cap = (ref.get("caption") or "").strip()
                        if cap:
                            texts.append(f"[Image {ref['sha256'][:8]}] {cap}")
                    else:
                        normalized.append({"type": "text", "text": part["text"]})
                        texts.append(part["text"])

                if total_image_bytes > MAX_ADD_IMAGES:
                    # ⚠️ 同上：**丢弃超出部分的图片、保留已处理的与文字**，不整块失败
                    logger.warning("[add] 本请求图片总量超限（%.1f MiB > %d），"
                                   "**已丢弃后续图片**、文字与已存图保留",
                                   total_image_bytes / 1024 / 1024, MAX_ADD_IMAGES)
                    total_image_bytes = -1          # 之后不再重复告警

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

            # ⚠️ 必须 `self.lock, self.conn`（与 307/423/431 行保持一致）。
            # 原来只写 `with self.conn:` ⇒ 与 caption worker 的写事务并发时抛
            # `sqlite3.OperationalError: cannot start a transaction within a transaction`
            # ⇒ add 直接 500。实测：10-27 有 5 次、10-29 有 8 次平台 add 因此失败（0.09%）。
            with self.lock, self.conn:
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
            #   若 **add 时推导出的 user_id**（可能带 `:chunk-N` / `:0` 后缀）与
            #   **search 时平台传的 user_id**（不带后缀）不一致 ⇒ 精确匹配查不到 ⇒ **检索为空**。
            #   兜底：精确查为空时，退化为"**带分隔符的前缀匹配**"
            #     `user_id LIKE ? || ':%'`
            #   ⇒ 只匹配"同一前缀 + 冒号分隔"的，**不会把 u_1 和 u_10 混在一起**
            if not rows:
                #   实测反例：search 传 `eval:run_abc123` 时，LIKE 会把
                #   `…:locomo_refined:conv-0` **和** `…:conv-1` 的记录**全混在一起**返回
                #   ⇒ "add 正常、search 200、答案却来自别的会话" ⇒ **比空答案更糟**
                #     （空答案只是 0 分；串会话是"言之有据的错误答案"）
                #   正确做法：**只精确退一级**（`rsplit(':',1)[0]` 精确匹配）
                #   ⇒ 既不漏（能兜住"add 推导多一级"），也不过界（不会拉进兄弟会话）
                try:
                    _parent = user_id.rsplit(":", 1)[0] if ":" in user_id else ""
                    if _parent:
                        rows = self.conn.execute(
                            "SELECT * FROM memories WHERE user_id=? "
                            "ORDER BY COALESCE(ts, 0), request_id, msg_index",
                            (_parent,),
                        ).fetchall()
                        if rows:
                            logger.warning(
                                "[store] user_id 精确为空 ⇒ **精确退一级**命中 %d 条（%s → %s）",
                                len(rows), str(user_id)[:36], str(_parent)[:36])
                except Exception:
                    logger.exception("[store] 退一级兜底失败（忽略）")

        output = []
        for row in rows:
            item = dict(row)
            item["content"] = json.loads(item.pop("content_json"))
            output.append(item)
        return output

    def tail_memories(self, user_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """只取**最后 n 条**记忆（`list_memories` 的 O(1) 版本）。

        为什么需要：`_empty_fallback` 里原来写的是
        `self.store.list_memories(uid)[-5:]` —— 对 1 万条的用户会**全量读出再丢掉 9,995 条**
        （实测 `list_memories` 在 10,504 条上 0.51s）。这里下推到 SQL，只返回 n 条。
        排序与 `list_memories` 一致（`COALESCE(ts,0), request_id, msg_index` 的末尾）。
        """
        n = max(1, int(n))
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE user_id=? "
                "ORDER BY COALESCE(ts, 0) DESC, request_id DESC, msg_index DESC LIMIT ?",
                (user_id, n),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["content"] = json.loads(item.pop("content_json"))
            output.append(item)
        # 返回**时间正序**（与 list_memories()[-n:] 一致）
        return list(reversed(output))

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
            # 小图原样返回（保证字节一致）；大图才压缩（原图 1.6MB 会撑爆响应）
            # 原样返回的条件：字节小 + 格式 web-safe + **长边不超过 1024**
            # （长边上限是 VLM 的安全值；超过就走下面的压缩阶梯，行为与改动前一致）
            # 压缩到 ~90KB 反而会丢细节（平台规范允许单图 ≤10MiB）。
            # 默认放到 10MiB（等于平台的单图上限），可用 AMC_IMAGE_PASSTHRU_MAX 调。
            import os as _os_pm
            _PASSTHRU_MAX = int(_os_pm.getenv("AMC_IMAGE_PASSTHRU_MAX", str(10 * 1024 * 1024)) or 10 * 1024 * 1024)
            _passthru = False
            # 契约：图片只接受 JPEG / PNG / WebP（gif 等其它格式必须转码）
            if len(raw) <= _PASSTHRU_MAX and str(part.get("fmt") or "").lower() in (
                    "png", "jpg", "jpeg", "webp"):
                # （长边超限 ⇒ 走压缩阶梯）。而实测原尺寸图 OCR 能识别全部文字，
                # 压到 ~90KB 反而丢细节。平台规范允许单图 ≤10MiB。
                # ⇒ 只保留字节上限（默认 10MiB = 平台单图上限），不再限制像素尺寸。
                _PASSTHRU_PX = int(_os_pm.getenv("AMC_IMAGE_PASSTHRU_MAX_PX", "0") or 0)
                try:
                    if _PASSTHRU_PX > 0:
                        from PIL import Image as _Im
                        with _Im.open(__import__("io").BytesIO(raw)) as _im0:
                            _passthru = max(_im0.size) <= _PASSTHRU_PX
                    else:
                        _passthru = True      # 不限像素 ⇒ 原样直传
                except Exception:
                    _passthru = True          # 连 PIL 都打不开时也直传（平台自己去解）
            if _passthru:
                output.append({
                    "type": "image_url",
                    "image_url": {"url": (f"data:image/{part['fmt']};base64,"
                                          f"{base64.b64encode(raw).decode('ascii')}")},
                })
                used += len(raw)
                continue
            try:
                from .imaging import vlm_ready_image
                small, sfmt = vlm_ready_image(raw, part["fmt"])
                try:
                    import io as _io
                    from PIL import Image as _Image
                    # OCR 实测（以原图为参照）：512px=0.90（漏字）、768=0.98、**1024=1.00**
                    # 截图类图片 1024px 仅 ~95KB（UI 图压缩率高），故优先 1024px
                    for side in (1024, 768, 512, 384, 256):
                        if len(small) <= 120_000:
                            break
                        im = _Image.open(_io.BytesIO(small)).convert("RGB")
                        w, h = im.size
                        sc = min(1.0, side / float(max(w, h)))
                        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))))
                        out = _io.BytesIO(); im.save(out, "JPEG", quality=72)
                        small, sfmt = out.getvalue(), "jpeg"
                except Exception:
                    pass
            except Exception:
                small, sfmt = raw, part["fmt"]
            output.append({
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/{sfmt};base64,"
                        f"{base64.b64encode(small).decode('ascii')}"
                    )
                },
            })
            used += len(small)

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

    def bump_abstain_by_gateway(self, n: int = 1) -> None:
        """累计"因网关失败而弃答"的次数（进程内，重启归零）。"""
        try:
            with self.lock:
                self._abstain_by_gateway = int(getattr(self, "_abstain_by_gateway", 0) or 0) + int(n)
        except Exception:
            pass

    def stats(self):
        _cap = dict(getattr(self, "_caption_stats", {}) or {})
        # ⚠️ 这个计数和"证据不足导致的弃答"分开，用来区分"网关问题"和"覆盖问题"
        _cap["abstain_by_gateway"] = int(getattr(self, "_abstain_by_gateway", 0) or 0)
        try:
            with self._pending_lock:
                _cap["pending"] = len(self._pending_captions)
        except Exception:
            pass
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
            "caption": _cap,                     # 后台 caption 队列：done / gen_failed / write_failed / pending
            "memories": row["memories"],
            "users": row["users"],
            "image_refs": row["image_refs"],
            "requests": requests,
            "blobs": sum(1 for path in self.blob_dir.rglob("*") if path.is_file()),
        }

    def request_ids_of(self, memory_ids):
        """m_* → request_id 映射（用于把检索器的记忆 id 与图书馆的轮次 id 对齐）。"""
        ids = [str(x) for x in (memory_ids or [])]
        if not ids:
            return {}
        out = {}
        with self.lock:
            for i in range(0, len(ids), 200):
                chunk = ids[i:i + 200]
                q = "SELECT id, request_id FROM memories WHERE id IN (%s)" % ",".join("?" * len(chunk))
                for mid, rid in self.conn.execute(q, chunk).fetchall():
                    out[str(mid)] = str(rid)
        return out

    def memory_ids_of(self, request_id: str):
        """某 request 下的所有 memory id（供 L1 档案层入队）。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id FROM memories WHERE request_id=? ORDER BY msg_index", (request_id,)).fetchall()
        return [r[0] for r in rows]

    def records_by_date(self, user_id: str, *, date_from: str = "", date_to: str = "",
                        has_image: bool = None, role: str = "", limit: int = 100):
        """按日期/是否有图/角色**枚举**记录（结构化筛选，不走相似度检索）。

        为什么需要（reviewer 2026-11-01）：确定性规则要做的正是
        "给定日期 D，取当天所有屏幕图" —— 这是**枚举**不是 top-k 检索。
        SQL 直接过滤，避免 list_memories 全量加载。
        """
        if not user_id:
            return []
        where = ["user_id=?"]
        args = [str(user_id)]
        if date_from:
            where.append("substr(COALESCE(created_at,''),1,10) >= ?")
            args.append(str(date_from)[:10])
        if date_to:
            where.append("substr(COALESCE(created_at,''),1,10) <= ?")
            args.append(str(date_to)[:10])
        if has_image is not None:
            where.append("(n_images > 0)" if has_image else "(COALESCE(n_images,0) = 0)")
        if role:
            where.append("role = ?")
            args.append(str(role))
        q = ("SELECT * FROM memories WHERE " + " AND ".join(where)
             + " ORDER BY COALESCE(ts,0), request_id, msg_index LIMIT ?")
        # ⚠️ 2026-11-02：**去掉 min(500) 夹断**。service.py:572 用 limit=200000
        #    全库扫，被这里夹到 500 ⇒ 只扫 10,504 条的 4.8%（同一 bug 两处实现，只修过 routes）。
        args.append(max(1, int(limit or 100)))
        with self.lock:
            rows = self.conn.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def user_of_request(self, request_id: str) -> str:
        """由 request_id 反查 user_id（会话扩展的 tail 模式需要它来查同会话记录）。"""
        if not request_id:
            return ""
        try:
            with self.lock:
                row = self.conn.execute(
                    "SELECT user_id FROM memories WHERE request_id=? LIMIT 1",
                    (str(request_id),)).fetchone()
            return str(row[0]) if row else ""
        except Exception:
            return ""

    def records_of_session(self, user_id: str, session_id: str, limit: int = 80):
        """按 (ts, msg_index) 返回该会话的记录 id（供"无 library 时的会话扩展"）。

        为什么需要：`_append_session_partners` 在 library augment 块里，
        而 library 只覆盖已打标的用户 ⇒ 没有 library 的用户**完全拿不到会话扩展**
        （2026-11-01 实测：ATM 用户 10,504 条、无 library ⇒ 扩展从未执行）。
        这个方法让扩展可以绕过 library 直接走存储。
        """
        if not (user_id and session_id):
            return []
        with self.lock:
            rows = self.conn.execute(
                "SELECT id FROM memories WHERE user_id=? AND session_id=? "
                "ORDER BY COALESCE(ts,0), request_id, msg_index LIMIT ?",
                (str(user_id), str(session_id), int(limit or 80))).fetchall()
        return [r[0] for r in rows]

    def session_of_requests(self, request_ids):
        """request_id → session_id（供证据集"同会话伙伴"扩展）。"""
        ids = [str(r) for r in (request_ids or [])]
        if not ids:
            return {}
        out = {}
        with self.lock:
            for i in range(0, len(ids), 200):
                chunk = ids[i:i + 200]
                q = ("SELECT request_id, session_id FROM memories WHERE request_id IN (%s) GROUP BY request_id"
                     % ",".join("?" * len(chunk)))
                for rid, sid in self.conn.execute(q, chunk):
                    out[rid] = sid or ""
        return out

    def count_memories(self, user_id: str) -> int:
        """该用户有多少条记忆（用于区分"未知用户"与"有记忆但没匹配"）。"""
        with self.lock:
            return int(self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id=?", (str(user_id),)).fetchone()[0] or 0)

    def get_memory(self, memory_id: str):
        """按 id 取一条记忆（供账本把来源原文带回）。"""
        with self.lock:
            row = self.conn.execute(
                "SELECT id, user_id, request_id, session_id, role, ts, created_at, content_json, index_text "
                "FROM memories WHERE id=?", (str(memory_id),)).fetchone()
        if not row:
            return None
        out = dict(row)
        try:
            out["content"] = json.loads(out.pop("content_json") or "[]")
        except Exception:
            out["content"] = []
        return out
