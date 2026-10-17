"""Embedding providers + persistent vector cache (NaN-safe)."""
import hashlib
import httpx
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms == 0)] = 1.0
    return matrix / norms


def finite_rows(matrix):
    """True for rows that are fully finite and non-zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    finite_mask = np.isfinite(matrix)
    finite = finite_mask.all(axis=1)
    norms = np.linalg.norm(np.where(finite_mask, matrix, 0.0), axis=1)
    return finite & (norms > 0)


def pick_device(requested):
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_dtype(requested, device):
    """auto: bf16 on Ampere+ (cc>=8), else fp32. fp16 is NOT auto-chosen
    because Qwen2/Qwen3 overflow to inf/NaN in fp16."""
    import torch
    name = (requested or "auto").strip().lower()
    table = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    if name in table:
        return table[name]
    if str(device).startswith("cuda"):
        try:
            major, _ = torch.cuda.get_device_capability(torch.device(device))
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float32
    return torch.float32


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


class VectorCache:
    """sha1(namespace||text) -> float32 vector in sqlite. Rejects NaN/zero vectors."""

    def __init__(self, path, namespace, dim=0):
        self.namespace = namespace
        self.dim = int(dim or 0)
        self.lock = threading.RLock()
        self.conn = None
        self.memory = None
        self.rejected = 0
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(path), check_same_thread=False)
                with self.conn:
                    self.conn.execute("PRAGMA journal_mode=WAL;")
                    self.conn.execute(
                        "CREATE TABLE IF NOT EXISTS vectors("
                        "key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
                    )
            except Exception as exc:
                logger.warning("[cache] sqlite disabled (%s), using memory", exc)
                self.conn = None
        if self.conn is None:
            self.memory = {}

    def key(self, text):
        return hashlib.sha1(f"{self.namespace}||{text}".encode("utf-8")).hexdigest()

    def _valid(self, vector):
        if self.dim and vector.shape[0] != self.dim:
            return False
        return bool(finite_rows(vector)[0])

    def get_many(self, keys):
        unique = list(dict.fromkeys(keys))
        found = {}
        with self.lock:
            if self.memory is not None:
                for key in unique:
                    vector = self.memory.get(key)
                    if vector is not None:
                        found[key] = vector
            else:
                for start in range(0, len(unique), 500):
                    chunk = unique[start:start + 500]
                    marks = ",".join("?" * len(chunk))
                    rows = self.conn.execute(
                        f"SELECT key, vec FROM vectors WHERE key IN ({marks})", chunk
                    ).fetchall()
                    for key, blob in rows:
                        vector = np.frombuffer(blob, dtype=np.float32)
                        if self._valid(vector):
                            found[key] = vector
                        else:
                            self.rejected += 1
        missing = [key for key in unique if key not in found]
        return found, missing

    def put_many(self, items):
        clean = []
        for key, vector in items:
            vector = np.asarray(vector, dtype=np.float32).ravel()
            if self._valid(vector):
                clean.append((key, vector))
            else:
                self.rejected += 1
        if not clean:
            return
        with self.lock:
            if self.memory is not None:
                for key, vector in clean:
                    self.memory[key] = vector
                return
            try:
                with self.conn:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO vectors(key, dim, vec) VALUES (?,?,?)",
                        [(key, int(v.shape[0]), v.tobytes()) for key, v in clean],
                    )
            except Exception as exc:
                logger.warning("[cache] write failed: %s", exc)


class BaseProvider:
    name = "base"
    dim = 0

    def encode_documents(self, texts):
        raise NotImplementedError

    def encode_query(self, text):
        raise NotImplementedError

    def describe(self):
        return {"provider": self.name, "dim": self.dim}


class LocalSTProvider(BaseProvider):
    def __init__(self, model_name, device="auto", batch=16, dim=0,
                 instruct="", strict=True, dtype="auto"):
        try:
            import torch  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "pip install 'sentence-transformers>=4.0' 'transformers>=4.51'"
            ) from exc

        self.device = pick_device(device)
        self.torch_dtype = resolve_dtype(dtype, self.device)
        self.batch = max(1, batch)
        self.instruct = instruct
        self.requested_model = model_name
        self.lock = threading.Lock()
        self.bad_rows = 0

        logger.info("[embed:local] loading %s on %s dtype=%s",
                    model_name, self.device, dtype_name(self.torch_dtype))
        started = time.time()
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs={"torch_dtype": self.torch_dtype},
                tokenizer_kwargs={"padding_side": "left"},
                truncate_dim=dim or None,
            )
        except TypeError:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model = self.model.to(self.torch_dtype)
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"failed to load embedding model '{model_name}': {exc}"
                ) from exc
            logger.error("[embed:local] load failed: %s -> fallback BAAI/bge-m3", exc)
            self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
            model_name = "BAAI/bge-m3"

        self.loaded_model = model_name
        self.name = f"local::{model_name}::d{dim or 0}"
        self.dim = int(self.model.get_sentence_embedding_dimension())
        prompts = getattr(self.model, "prompts", None) or {}
        self.has_query_prompt = "query" in prompts
        logger.info("[embed:local] ready model=%s dim=%d (%.1fs)",
                    model_name, self.dim, time.time() - started)

    def _encode(self, texts, is_query=False):
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        kwargs = {
            "batch_size": self.batch,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_tensor": True,
        }
        if is_query:
            if self.instruct:
                texts = [f"Instruct: {self.instruct}\nQuery: {t}" for t in texts]
            elif self.has_query_prompt:
                kwargs["prompt_name"] = "query"
        with self.lock:
            tensor = self.model.encode(texts, **kwargs)
        matrix = tensor.detach().float().cpu().numpy()
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        ok = finite_rows(matrix)
        if not ok.all():
            bad = int((~ok).sum())
            self.bad_rows += bad
            logger.warning("[embed:local] %d/%d non-finite vectors (dtype=%s)",
                           bad, len(texts), dtype_name(self.torch_dtype))
            matrix[~ok] = 0.0
        return l2_normalize(matrix)

    def encode_documents(self, texts):
        return self._encode(texts, is_query=False)

    def encode_query(self, text):
        return self._encode([text], is_query=True)[0]

    def describe(self):
        return {
            "provider": "local",
            "requested_model": self.requested_model,
            "loaded_model": self.loaded_model,
            "device": self.device,
            "dtype": dtype_name(self.torch_dtype),
            "dim": self.dim,
            "query_instruct": bool(self.instruct),
            "bad_rows": self.bad_rows,
        }


class OpenAIEmbeddingProvider(BaseProvider):
    """DashScope compatible-mode, or vLLM --task embed."""

    def __init__(self, model_name, base_url, api_key="", batch=10,
                 dim=0, instruct="", timeout=180, retries=6):
        import httpx
        self.model_name = model_name
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.batch = max(1, batch)
        self.dim = int(dim or 0)
        self.instruct = instruct
        self.retries = retries
        self.client = httpx.Client(timeout=timeout)
        logger.info("[embed:api] %s @ %s (batch=%d)", model_name, self.base_url, self.batch)
        self.failed_texts = []   # (index, text_preview, error)
        probe = self._call(["ping"])
        self.dim = int(probe.shape[1])
        self.name = f"openai::{model_name}::d{self.dim}"
        logger.info("[embed:api] ready dim=%d", self.dim)

    def _call(self, texts):
        """Call the embeddings endpoint. Retries on transient errors.

        On permanent errors (400 / 401) the full response body is logged
        once, then the call fails fast. On batch failures, the caller
        should fall back to per-item calls via `_call_with_fallback`.
        """
        payload = {"model": self.model_name, "input": list(texts),
                   "encoding_format": "float"}
        if self.dim:
            payload["dimensions"] = self.dim
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.post(
                    self.base_url + "/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                if response.status_code >= 400:
                    body = response.text[:2000]
                    # Permanent errors: fail fast, log body once
                    if response.status_code in (400, 401, 403, 404):
                        logger.error(
                            "[embed:api] PERMANENT %d (input size=%d): %s",
                            response.status_code, len(texts), body,
                        )
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}: {body}",
                            request=response.request,
                            response=response,
                        )
                    # Transient errors (429 / 5xx) -> retry with longer backoff
                    if response.status_code == 429:
                        wait = min(4 * attempt, 30)
                    else:
                        wait = min(2 ** (attempt - 1), 10)
                    logger.warning(
                        "[embed:api] attempt %d/%d HTTP %d, retry in %ss: %s",
                        attempt, self.retries, response.status_code, wait, body[:200],
                    )
                    last = f"HTTP {response.status_code}: {body[:200]}"
                    if attempt < self.retries:
                        time.sleep(wait)
                    continue

                rows = sorted(response.json()["data"], key=lambda r: r.get("index", 0))
                return l2_normalize([row["embedding"] for row in rows])
            except (httpx.HTTPStatusError, httpx.ReadTimeout,
                    httpx.ConnectTimeout, httpx.ConnectError,
                    httpx.RemoteProtocolError) as exc:
                last = exc
                # Permanent HTTP errors: do not retry
                if isinstance(exc, httpx.HTTPStatusError):
                    code = getattr(exc.response, "status_code", None)
                    if code in (400, 401, 403, 404):
                        raise
                if attempt == self.retries:
                    break
                wait = min(2 ** (attempt - 1), 15)
                logger.warning(
                    "[embed:api] attempt %d/%d failed (%s), retry in %ss",
                    attempt, self.retries,
                    exc.__class__.__name__, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                last = exc
                if attempt == self.retries:
                    break
                wait = min(2 ** (attempt - 1), 15)
                logger.warning(
                    "[embed:api] attempt %d/%d unexpected (%s), retry in %ss",
                    attempt, self.retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"embedding API failed after {self.retries} attempts: {last}")

    def _call_with_fallback(self, texts):
        """Try the batch. On permanent 400 failure, split into singles.

        Any single that still fails gets a zero vector and is recorded in
        self.failed_texts. This lets a long encoding run complete even when
        a handful of inputs are rejected (e.g. content inspection).
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            return self._call(texts)
        except httpx.HTTPStatusError as exc:
            code = getattr(exc.response, "status_code", None)
            if code != 400:
                raise
            logger.warning(
                "[embed:api] batch of %d failed with 400; splitting into singles",
                len(texts),
            )

        vectors = []
        for idx, text in enumerate(texts):
            try:
                vectors.append(self._call([text])[0])
            except Exception as exc:
                self.failed_texts.append((idx, str(text)[:120], str(exc)[:200]))
                logger.error(
                    "[embed:api] single item %d permanently rejected: %s",
                    idx, exc,
                )
                vectors.append(np.zeros(self.dim, dtype=np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)

    def encode_documents(self, texts):
        texts = list(texts)
        output = [self._call_with_fallback(texts[i:i + self.batch])
                  for i in range(0, len(texts), self.batch)]
        return np.vstack(output) if output else np.zeros((0, self.dim), dtype=np.float32)

    def encode_query(self, text):
        if self.instruct:
            text = f"Instruct: {self.instruct}\nQuery: {text}"
        return self._call_with_fallback([text])[0]

    def describe(self):
        return {
            "provider": "openai",
            "loaded_model": self.model_name,
            "base_url": self.base_url,
            "dim": self.dim,
            "query_instruct": bool(self.instruct),
            "failed_texts_count": len(self.failed_texts),
        }


def build_provider(settings):
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingProvider(
            model_name=settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            batch=settings.embedding_batch,
            dim=settings.embedding_dim,
            instruct=settings.embedding_instruct,
        )
    return LocalSTProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch=settings.embedding_batch,
        dim=settings.embedding_dim,
        instruct=settings.embedding_instruct,
        strict=settings.embedding_strict,
        dtype=settings.embedding_dtype,
    )