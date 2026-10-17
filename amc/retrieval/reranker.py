"""Rerankers (NaN-safe).

qwen3    : Qwen/Qwen3-Reranker-*, score = P(yes). bf16 by default (fp16 overflows).
vllm_gen : OpenAI-compatible chat endpoint, batched 0-100 scoring.
"""
import inspect
import json
import logging
import math
import re
import threading
import time

from .providers import dtype_name, pick_device, resolve_dtype

logger = logging.getLogger(__name__)

SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
PREFIX = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def doc_text(memory, limit=1500):
    text = memory.get("index_text") or ""
    if not text:
        text = "\n".join(
            part.get("text", "")
            for part in memory.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return (text or "[empty]")[:limit]


class Qwen3Reranker:
    def __init__(self, model_name="Qwen/Qwen3-Reranker-4B", device="auto",
                 batch=8, max_length=1024, instruct="", dtype="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = pick_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.batch = max(1, batch)
        self.max_length = max_length
        self.instruct = instruct or "Given a query, judge document relevance"
        self.model_name = model_name
        self.lock = threading.Lock()
        self.nan_count = 0
        self.oom_splits = 0

        logger.info("[rerank:qwen3] loading %s on %s dtype=%s",
                    model_name, self.device, dtype_name(self.dtype))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device).eval()

        params = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in params:
            self.keep_kwargs = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in params:
            self.keep_kwargs = {"num_logits_to_keep": 1}
        else:
            self.keep_kwargs = {}

        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.body_budget = max(64, max_length - len(self.prefix_ids) - len(self.suffix_ids))
        logger.info("[rerank:qwen3] ready yes=%s no=%s keep=%s",
                    self.yes_id, self.no_id, self.keep_kwargs)

    def _pairs(self, query, memories):
        return [
            f"<Instruct>: {self.instruct}\n<Query>: {query}\n<Document>: {doc_text(m)}"
            for m in memories
        ]

    def _forward(self, texts):
        torch = self.torch
        encoded = self.tokenizer(
            texts, padding=False, truncation="longest_first",
            max_length=self.body_budget, return_attention_mask=False,
        )
        input_ids = [self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids}, padding=True,
            return_attention_mask=True, return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with self.lock, torch.inference_mode():
            logits = self.model(**batch, **self.keep_kwargs).logits[:, -1, :].float()
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            probs = torch.softmax(pair, dim=1)[:, 1]
        return [float(x) for x in probs.cpu().tolist()]

    def _score(self, texts):
        try:
            return self._forward(texts)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(texts) == 1:
                raise
            self.oom_splits += 1
            logger.warning("[rerank:qwen3] OOM on batch=%d, splitting", len(texts))
            mid = len(texts) // 2
            return self._score(texts[:mid]) + self._score(texts[mid:])

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        memories = [memory for memory, _ in candidates]
        texts = self._pairs(query, memories)
        scores = []
        for start in range(0, len(texts), self.batch):
            scores.extend(self._score(texts[start:start + self.batch]))

        bad = [i for i, s in enumerate(scores) if not math.isfinite(s)]
        if bad and self.batch > 1:
            # 单条重算：没有 padding，排除 padding 引起的数值问题
            for i in bad:
                try:
                    scores[i] = self._score([texts[i]])[0]
                except Exception:
                    scores[i] = float("nan")
            bad = [i for i, s in enumerate(scores) if not math.isfinite(s)]
        if bad:
            self.nan_count += len(bad)
            logger.warning("[rerank:qwen3] %d/%d non-finite scores (dtype=%s), demoted",
                           len(bad), len(scores), dtype_name(self.dtype))
            for i in bad:
                scores[i] = -1.0 - 1e-6 * i

        ranked = sorted(zip(memories, scores), key=lambda item: item[1], reverse=True)
        return [(memory, float(score)) for memory, score in ranked[:top_k]]

    def describe(self):
        return {
            "reranker": "qwen3",
            "model": self.model_name,
            "device": self.device,
            "dtype": dtype_name(self.dtype),
            "nan_count": self.nan_count,
            "oom_splits": self.oom_splits,
        }


class VLLMGenerativeReranker:
    def __init__(self, base_url, model, api_key="EMPTY", batch=5,
                 max_tokens=128, timeout=60, retries=2, instruct=""):
        import httpx
        self.client = httpx.Client(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.batch = max(1, batch)
        self.max_tokens = max_tokens
        self.retries = retries
        self.instruct = instruct

    def _call(self, query, memories):
        passages = "\n".join(
            f"{i}. {doc_text(m, 900)}" for i, m in enumerate(memories, 1)
        )
        prompt = (
            f"Task: {self.instruct}\nQuery: {query}\n\nPassages:\n{passages}\n\n"
            f"Return ONLY a JSON array of {len(memories)} integers in 0-100, "
            "one relevance score per passage, in order."
        )
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.post(
                    self.base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You output only JSON arrays."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": self.max_tokens,
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                match = re.search(r"\[[\s0-9.,]+\]", content)
                if match:
                    values = json.loads(match.group())
                    if isinstance(values, list) and len(values) == len(memories):
                        return [float(v) / 100.0 for v in values]
                numbers = re.findall(r"\d{1,3}", content)
                if len(numbers) >= len(memories):
                    return [float(n) / 100.0 for n in numbers[:len(memories)]]
            except Exception as exc:
                logger.warning("[rerank:vllm] attempt %d failed: %s", attempt, exc)
                time.sleep(1)
        return None

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        scored = []
        for start in range(0, len(candidates), self.batch):
            chunk = candidates[start:start + self.batch]
            memories = [m for m, _ in chunk]
            scores = self._call(query, memories)
            if scores is None:
                scored.extend((m, float(b)) for m, b in chunk)
            else:
                scored.extend(zip(memories, scores))
        scored = [(m, s if math.isfinite(s) else -1.0) for m, s in scored]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(m, float(s)) for m, s in scored[:top_k]]

    def describe(self):
        return {"reranker": "vllm_gen", "model": self.model, "base_url": self.base_url}


def build_reranker(settings):
    if settings.reranker == "qwen3":
        return Qwen3Reranker(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
            batch=settings.reranker_batch,
            max_length=settings.reranker_max_len,
            instruct=settings.reranker_instruct,
            dtype=settings.reranker_dtype,
        )
    if settings.reranker == "vllm_gen":
        return VLLMGenerativeReranker(
            base_url=settings.reranker_base_url,
            model=settings.reranker_vllm_model,
            api_key=settings.reranker_api_key,
            batch=settings.reranker_batch,
            instruct=settings.reranker_instruct,
        )
    return None
