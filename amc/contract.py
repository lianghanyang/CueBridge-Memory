import math
from typing import Any, List

from .config import MAX_ADD_IMAGES, MAX_SINGLE_IMAGE
from .imaging import decode_data_uri

ADD_REQUIRED = {"request_id", "user_id", "session_id"}


class ContractError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def validate_content_parts(content: Any, where: str) -> List[str]:
    errors = []
    if isinstance(content, str):
        if not content.strip():
            errors.append(f"{where}: string content must be non-empty")
        return errors

    if not isinstance(content, list) or not content:
        return [f"{where}: content must be non-empty string or ContentPart[]"]

    total_image_bytes = 0
    for idx, part in enumerate(content):
        p_str = f"{where}[{idx}]"
        if not isinstance(part, dict):
            errors.append(f"{p_str}: must be an object")
            continue
        p_type = part.get("type")
        if p_type == "text":
            txt = part.get("text")
            if not isinstance(txt, str) or not txt.strip():
                errors.append(f"{p_str}.text must be non-empty string")
        elif p_type == "image_url":
            url_dict = part.get("image_url")
            url = url_dict.get("url") if isinstance(url_dict, dict) else None
            # AML 契约允许 image_url.url 是 "URL 或 base64 data URI"。
            # 只做**格式**校验（http(s) 链接也放行，避免误 422），
            # 真正取字节在内联时校验，外链则在入库时按需抓取（见 store.py）。
            if not isinstance(url, str) or not url.strip():
                errors.append(f"{p_str}.image_url.url must be a non-empty string")
            elif url.startswith(("http://", "https://")):
                pass                      # 允许外链，交由入库阶段抓取
            else:
                try:
                    raw_bytes = decode_data_uri(url)
                    total_image_bytes += len(raw_bytes)
                    if len(raw_bytes) > MAX_SINGLE_IMAGE:
                        errors.append(f"{p_str}: single image exceeds 10 MiB limit")
                except Exception as exc:
                    errors.append(f"{p_str}: invalid inline data URI ({exc})")
        else:
            errors.append(f"{p_str}.type must be 'text' or 'image_url'")

    if total_image_bytes > MAX_ADD_IMAGES:
        errors.append(f"{where}: total images exceed 30 MiB limit")
    return errors


def validate_add_request(req: Any) -> List[str]:
    if not isinstance(req, dict):
        return ["request body must be a JSON object"]
    errors = []
    for f in ADD_REQUIRED:
        val = req.get(f)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"field '{f}' is required and must be non-empty string")

    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("field 'messages' must be a non-empty array")
    else:
        for idx, msg in enumerate(messages):
            m_str = f"messages[{idx}]"
            if not isinstance(msg, dict):
                errors.append(f"{m_str} must be an object")
                continue
            if msg.get("role") not in {"user", "assistant", "system"}:
                errors.append(f"{m_str}.role is invalid")
            ts = msg.get("timestamp")
            if ts is not None and (not isinstance(ts, int) or isinstance(ts, bool)):
                errors.append(f"{m_str}.timestamp must be integer Unix ms")
            errors.extend(validate_content_parts(msg.get("content"), f"{m_str}.content"))
    return errors


def validate_search_request(req: Any) -> List[str]:
    if not isinstance(req, dict):
        return ["request body must be a JSON object"]
    errors = []
    user_id = req.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        errors.append("field 'user_id' is required and must be non-empty string")

    top_k = req.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        errors.append("field 'top_k' must be a positive integer")

    query = req.get("query")
    if isinstance(query, str):
        if not query.strip():
            errors.append("field 'query' string must be non-empty")
    elif isinstance(query, list):
        errors.extend(validate_content_parts(query, "query"))
    else:
        errors.append("field 'query' must be a non-empty string or ContentPart[]")

    options = req.get("options")
    if options is not None:
        if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
            errors.append("field 'options' must be an array of strings")
    return errors


def validate_search_response_contract(resp: Any, top_k: int) -> List[str]:
    """严格对照 AML 官方 Search Response 契约验证"""
    if not isinstance(resp, dict):
        return ["response must be a JSON object"]
    if "data" not in resp or not isinstance(resp["data"], list):
        return ["response must contain 'data' array (no top-level array or 'items' wrapper allowed)"]

    errors = []
    data = resp["data"]
    if len(data) > top_k:
        errors.append(f"returned records count ({len(data)}) exceeds requested top_k ({top_k})")

    seen_ids = set()
    prev_score = None

    for idx, item in enumerate(data):
        i_str = f"data[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{i_str} must be an object")
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid.strip():
            errors.append(f"{i_str}.id must be non-empty string")
        elif mid in seen_ids:
            errors.append(f"{i_str}.id '{mid}' is duplicated")
        seen_ids.add(mid)

        content = item.get("content")
        if content is None:
            errors.append(f"{i_str}.content is required")
        else:
            errors.extend(validate_content_parts(content, f"{i_str}.content"))

        score = item.get("score")
        if score is not None:
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                errors.append(f"{i_str}.score must be a finite numeric value")
            elif prev_score is not None and score > prev_score + 1e-6:
                errors.append(
                    f"{i_str}.score ({score}) > previous ({prev_score}); "
                    "scores must be non-increasing"
                )
            prev_score = score
    return errors
