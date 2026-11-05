import math
from typing import Any, List

from .config import MAX_ADD_IMAGES, MAX_SINGLE_IMAGE
from .imaging import decode_data_uri

# ⚠️🔴 （严重兼容性修复）：
#   原来把 `session_id` 也当成**必填** ⇒ 缺它直接 422。
#   但官方 Add 示例是：
#     {"request_id":"eval:run_abc123:locomo_refined:conv-0:chunk-0",
#      "messages":[{"role":"user","timestamp":1704…}]}
#     ⇒ **没有 session_id**（也没有 user_id）
#   ⇒ 若平台真按示例发 ⇒ **我们会对所有 add 返 422 ⇒ 整轮失败**
#   修法：只强制 `request_id` + `messages`；`user_id`/`session_id` 缺省时
#        **从 request_id 推导**（request_id 形如 `...:conv-0:chunk-0`），
#        并打 warning（可见、可统计），而不是拒绝请求。
ADD_REQUIRED = {"request_id"}


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
            try:
                raw_bytes = decode_data_uri(url or "")
                total_image_bytes += len(raw_bytes)
                #   多模态题没有图无法作答 ⇒ 超限也要收（落库时若真的过大再单独记日志）。
                #   需要恢复时设 AMC_MAX_SINGLE_IMAGE_MIB 调小 + 删掉下面这行注释。
                # if len(raw_bytes) > MAX_SINGLE_IMAGE:
                #     errors.append(f"{p_str}: single image exceeds 10 MiB limit")
                _ = raw_bytes  # 仍然解码一次，确保 data URI 合法
            except Exception as exc:
                errors.append(f"{p_str}: invalid inline data URI ({exc})")
        else:
            errors.append(f"{p_str}.type must be 'text' or 'image_url'")

    # ⚠️ （同上）：不再因"总图片体积"拒整条 add
    # if total_image_bytes > MAX_ADD_IMAGES:
    #     errors.append(f"{where}: total images exceed 30 MiB limit")
    _ = total_image_bytes
    return errors


def validate_add_request(req: Any) -> List[str]:
    if not isinstance(req, dict):
        return ["request body must be a JSON object"]
    errors = []
    for f in ADD_REQUIRED:
        val = req.get(f)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"field '{f}' is required and must be non-empty string")
    # `user_id` / `session_id`：**缺省时推导**，不再直接拒（见上面的兼容性说明）
    rid = str(req.get("request_id") or "")
    for f, derive in (("user_id", True), ("session_id", True)):
        val = req.get(f)
        if not isinstance(val, str) or not val.strip():
            if derive and rid:
                # request_id `eval:run_x:conv-0:chunk-3` ⇒ user/session 取到 `chunk` 之前的部分
                #   若 request_id **没有 `:chunk` 后缀**而是别的分块命名（`:0` / `:2`），
                #   兜底顺序：① `:chunk-N` 之前 ② 剥掉末尾 `:<纯数字>` ③ 整串
                d = rid.rsplit(":chunk", 1)[0] or rid
                _m = __import__("re").search(r"^(.*?):\d+$", d)
                if _m and _m.group(1):
                    d = _m.group(1)
                req[f] = d
            else:
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
