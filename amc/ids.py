import hashlib


def memory_id(user_id: str, request_id: str, message_index: int) -> str:
    raw = f"{user_id}|{request_id}|{message_index}".encode("utf-8")
    return "m_" + hashlib.sha256(raw).hexdigest()[:24]
