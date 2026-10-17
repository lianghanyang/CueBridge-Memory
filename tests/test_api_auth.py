"""端到端 API 测试：鉴权（Token/Bearer/裸 key/错误 key）+ 路径别名 + 契约转发。

用桩服务注入，**不加载 4B、不访问网络**，因此可在 CI 里秒级运行。
"""
import os

os.environ.setdefault("AMC_SKIP_APP_AUTOCREATE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from amc.app import create_app  # noqa: E402
from amc.contract import ContractError  # noqa: E402

KEY = "test-secret-key"


class _Store:
    def stats(self):
        return {"rows": 0}


class StubService:
    retriever = None

    def __init__(self):
        self.store = _Store()

    def add(self, body):
        if not body.get("user_id"):
            raise ContractError(422, "user_id is required")
        return {"success": True, "request_id": body.get("request_id")}

    def search(self, body, hydrate=True):
        return {"data": [{"id": "m1", "content": "hit", "score": 0.5,
                          "created_at": "2026-01-01T00:00:00+08:00"}]}


def _client():
    return TestClient(create_app(service=StubService(), system_key=KEY))


ADD = {"user_id": "u", "session_id": "s", "request_id": "r",
       "messages": [{"role": "user", "content": "hi", "timestamp": 1760000000000}]}


def test_health_and_aliases_are_public():
    c = _client()
    assert c.get("/health").status_code == 200
    assert c.get("/v1/memories/health").status_code == 200


def test_token_bearer_and_bare_key_all_accepted():
    c = _client()
    for header in (f"Token {KEY}", f"Bearer {KEY}", KEY):
        r = c.post("/v1/memories/add", json=ADD, headers={"Authorization": header})
        assert r.status_code == 200, header
        assert r.json()["success"] is True


def test_wrong_or_missing_key_is_401():
    c = _client()
    assert c.post("/v1/memories/add", json=ADD).status_code == 401
    assert c.post("/v1/memories/add", json=ADD,
                  headers={"Authorization": "Token wrong"}).status_code == 401
    assert c.post("/v1/memories/search", json={"user_id": "u", "top_k": 1, "query": "q"},
                  headers={"Authorization": "Bearer nope"}).status_code == 401


def test_legacy_and_v1_paths_are_equivalent():
    c = _client()
    h = {"Authorization": f"Token {KEY}"}
    body = {"user_id": "u", "top_k": 1, "query": "q"}
    a = c.post("/search", json=body, headers=h)
    b = c.post("/v1/memories/search", json=body, headers=h)
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()
    # 分数必须非递增（AML 契约）
    scores = [d["score"] for d in b.json()["data"]]
    assert scores == sorted(scores, reverse=True)


def test_contract_error_is_forwarded_as_422():
    c = _client()
    bad = {k: v for k, v in ADD.items() if k != "user_id"}
    r = c.post("/v1/memories/add", json=bad, headers={"Authorization": f"Token {KEY}"})
    assert r.status_code == 422 and "user_id" in r.json()["error"]
