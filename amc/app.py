import logging
from typing import Optional

import os
from fastapi import Body, FastAPI, Header
from fastapi.responses import JSONResponse

from .config import Settings
from .contract import ContractError
from .retrieval import build_retriever
from .retrieval.describe import describe_retriever
from .service import MemoryService
from .store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("amc")


def build_service(settings: Settings):
    errors = settings.validate()
    if errors:
        raise RuntimeError("invalid configuration:\n  - " + "\n  - ".join(errors))
    logger.info("retriever=%s", settings.retriever)
    store = MemoryStore(
        settings.db_path,
        settings.blob_dir,
        timezone_offset_hours=settings.timezone_offset_hours,
    )
    retriever = build_retriever(
        settings.retriever,
        store,
        min_score=settings.min_score,
        use_relative_time=settings.use_relative_time,
        settings=settings,
    )
    describer = None
    if getattr(settings, "query_vision", False):
        if not getattr(settings, "query_vision_base_url", ""):
            logger.warning("AMC_QUERY_VISION=1 but no base_url -> query images ignored")
        else:
            from .query_vision import QueryImageDescriber
            describer = QueryImageDescriber(
                base_url=settings.query_vision_base_url,
                api_key=settings.query_vision_api_key,
                model=settings.query_vision_model,
                max_images=settings.query_vision_max_images,
                timeout=settings.query_vision_timeout,
                cache_path=settings.query_vision_cache_path or None,
            )
            logger.info("query-vision enabled model=%s max_images=%d",
                        settings.query_vision_model, settings.query_vision_max_images)
    relation_selector = None
    if getattr(settings, "relation_select", False):
        if not getattr(settings, "relation_select_base_url", ""):
            logger.warning("AMC_RELATION_SELECT=1 but no base_url -> disabled")
        else:
            from .relation_select import RelationSelectClient
            relation_selector = RelationSelectClient(
                base_url=settings.relation_select_base_url,
                api_key=getattr(settings, "relation_select_api_key", "") or "EMPTY",
                model=settings.relation_select_model,
                pool=settings.relation_select_pool,
                timeout=settings.relation_select_timeout,
                workers=settings.relation_select_workers,
                pairwise=settings.relation_select_pairwise,
                requires_query_image=settings.relation_select_requires_query_image,
                cache_path=settings.relation_select_cache_path or None,
            )
            logger.info("relation-select enabled url=%s model=%s pool=%d pairwise=%s",
                        settings.relation_select_base_url, settings.relation_select_model,
                        settings.relation_select_pool, settings.relation_select_pairwise)

    vl_reranker = None
    if getattr(settings, "vl_rerank", False):
        if not getattr(settings, "vl_rerank_base_url", ""):
            logger.warning("AMC_VL_RERANK=1 but no base_url -> VL rerank disabled")
        else:
            from .vl_rerank import VLRerankClient
            vl_reranker = VLRerankClient(
                base_url=settings.vl_rerank_base_url,
                model=settings.vl_rerank_model,
                pool=settings.vl_rerank_pool,
                timeout=settings.vl_rerank_timeout,
                workers=settings.vl_rerank_workers,
                include_doc_image=settings.vl_rerank_include_doc_image,
                requires_query_image=settings.vl_rerank_requires_query_image,
                max_query_images=settings.vl_rerank_max_query_images,
                cache_path=settings.vl_rerank_cache_path or None,
            )
            logger.info("vl-rerank enabled url=%s model=%s pool=%d requires_image=%s",
                        settings.vl_rerank_base_url, settings.vl_rerank_model,
                        settings.vl_rerank_pool,
                        settings.vl_rerank_requires_query_image)
    return MemoryService(
        store,
        retriever,
        max_return=settings.max_return,
        settings=settings,
        describer=describer,
        vl_reranker=vl_reranker,
        relation_selector=relation_selector,
    )


def create_app(service=None, system_key: Optional[str] = None):
    settings = Settings.from_env()
    service = service or build_service(settings)
    key = settings.system_key if system_key is None else system_key

    app = FastAPI(title="CueBridge Memory — Add/Search", version="2.3.0")

    def unauthorized(authorization):
        """接受 `Authorization: Bearer <key>` 与 `Authorization: Token <key>`
        （AML 报名表要求 Token 形式）。留空或未配置 key 时放行。"""
        if not key:
            return None
        raw = (authorization or "").strip()
        if not raw:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        if raw == key:                                   # 裸 key
            return None
        for prefix in ("Bearer", "Token"):
            if raw.lower().startswith(prefix.lower() + " "):
                if raw[len(prefix) + 1:].strip() == key:
                    return None
                break
        return JSONResponse(status_code=401, content={"error": "unauthorized"})

    def safe_call(handler, body, authorization, label):
        error = unauthorized(authorization)
        if error:
            return error
        try:
            return handler(body)
        except ContractError as exc:
            return JSONResponse(status_code=exc.status, content={"error": str(exc)})
        except Exception:
            logger.exception("%s failed", label)
            return JSONResponse(status_code=500, content={"error": "internal server error"})

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "retriever": describe_retriever(service.retriever),
            "stats": service.store.stats(),
        }

    def add_impl(body, authorization):
        return safe_call(service.add, body, authorization, "add")

    def search_impl(body, authorization):
        return safe_call(lambda payload: service.search(payload, hydrate=True),
                         body, authorization, "search")

    @app.post("/add")
    def add(body: dict = Body(...), authorization: Optional[str] = Header(None)):
        return add_impl(body, authorization)

    @app.post("/search")
    def search(body: dict = Body(...), authorization: Optional[str] = Header(None)):
        return search_impl(body, authorization)

    # AML 报名表给的示例路径是 /v1/memories/*，这里补别名，两套都能用
    @app.post("/v1/memories/add")
    def add_v1(body: dict = Body(...), authorization: Optional[str] = Header(None)):
        return add_impl(body, authorization)

    @app.post("/v1/memories/search")
    def search_v1(body: dict = Body(...), authorization: Optional[str] = Header(None)):
        return search_impl(body, authorization)

    @app.get("/v1/memories/health")
    def health_v1():
        return health()

    return app


# 便于测试/离线导入：设 AMC_SKIP_APP_AUTOCREATE=1 时不在此处构建服务（否则 import 即加载 4B）
if os.getenv("AMC_SKIP_APP_AUTOCREATE", "0") != "1":
    app = create_app()
else:  # pragma: no cover
    app = None
