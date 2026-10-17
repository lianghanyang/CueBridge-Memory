"""AML 契约校验测试：请求/响应/分数单调性（不依赖网络与模型）。"""
import pytest

from amc.contract import (
    validate_add_request,
    validate_content_parts,
    validate_search_request,
    validate_search_response_contract,
)

VALID_ADD = {
    "user_id": "u_1",
    "session_id": "s_1",
    "request_id": "r_1",
    "messages": [
        {"role": "user", "content": "hello", "timestamp": 1760000000000},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}], "timestamp": 1760000001000},
    ],
}


def test_add_valid_passes():
    assert validate_add_request(VALID_ADD) == []


@pytest.mark.parametrize("field", ["user_id", "session_id", "request_id"])
def test_add_missing_required_field(field):
    bad = {k: v for k, v in VALID_ADD.items() if k != field}
    assert any(field in e for e in validate_add_request(bad))


def test_add_rejects_empty_messages():
    bad = dict(VALID_ADD, messages=[])
    assert validate_add_request(bad)


def test_add_rejects_bad_role_and_timestamp():
    bad = dict(VALID_ADD, messages=[{"role": "tool", "content": "x", "timestamp": 1}])
    assert validate_add_request(bad)
    bad2 = dict(VALID_ADD, messages=[{"role": "user", "content": "x", "timestamp": "y"}])
    assert validate_add_request(bad2)


def test_content_accepts_string_and_parts():
    assert validate_content_parts("plain text", "c") == []
    assert validate_content_parts([{"type": "text", "text": "a"}], "c") == []
    # 契约允许 image_url.url 是 URL 或 data URI：外链（http/https）必须放行，避免误 422
    assert validate_content_parts([{"type": "image_url", "image_url": {"url": "https://x/a.png"}}], "c") == []
    assert validate_content_parts([{"type": "image_url", "image_url": {"url": "http://x/a.png"}}], "c") == []
    assert validate_content_parts([{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}], "c") == []
    assert validate_content_parts([{"type": "image_url", "image_url": {"url": "not-a-url"}}], "c")
    assert validate_content_parts([], "c")           # 空数组不合法
    assert validate_content_parts(123, "c")          # 非字符串/数组不合法


def test_search_request():
    assert validate_search_request({"user_id": "u", "top_k": 5, "query": "q"}) == []
    assert validate_search_request({"user_id": "u", "top_k": 5,
                                    "query": [{"type": "text", "text": "q"}]}) == []
    assert validate_search_request({"top_k": 5, "query": "q"})          # 缺 user_id
    assert validate_search_request({"user_id": "u", "top_k": 0, "query": "q"})   # top_k 非正
    assert validate_search_request({"user_id": "u", "top_k": 5})        # 缺 query


def test_search_response_contract_monotonic_and_unique():
    good = {"data": [
        {"id": "m1", "content": "a", "score": 0.9, "created_at": "2026-01-01T00:00:00+08:00"},
        {"id": "m2", "content": "b", "score": 0.5, "created_at": "2026-01-01T00:00:00+08:00"},
    ]}
    assert validate_search_response_contract(good, 5) == []

    increasing = {"data": [
        {"id": "m1", "content": "a", "score": 0.1},
        {"id": "m2", "content": "b", "score": 0.9},
    ]}
    assert any("non-increasing" in e for e in validate_search_response_contract(increasing, 5))

    dup = {"data": [{"id": "m1", "content": "a", "score": 0.9},
                    {"id": "m1", "content": "b", "score": 0.8}]}
    assert validate_search_response_contract(dup, 5)

    too_many = {"data": [{"id": f"m{i}", "content": "x", "score": 1 - i * 0.1} for i in range(4)]}
    assert validate_search_response_contract(too_many, 3)

    assert validate_search_response_contract([], 3)      # 顶层必须是对象
