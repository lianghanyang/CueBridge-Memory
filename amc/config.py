import os
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024
MAX_SINGLE_IMAGE = 10 * MIB
MAX_ADD_IMAGES = 30 * MIB
MAX_SEARCH_IMAGES = 30 * MIB

DASHSCOPE_COMPATIBLE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

DEFAULT_INSTRUCT = (
    "Given a user question, retrieve the user's personal memory "
    "entries (chat turns, photos, emails) that help answer it"
)

DTYPES = {
    "auto",
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "half",
    "fp32",
    "float32",
}


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name, default):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass
class Settings:
    # Memory 服务
    db_path: Path = Path("./data/store/memory.db")
    blob_dir: Path = Path("./data/store/blobs")
    retriever: str = "hybrid"
    max_return: int = 50
    system_key: str = ""
    timezone_offset_hours: int = 8
    min_score: float = 0.0
    use_relative_time: bool = False

    # Embedding。默认通过 DashScope OpenAI-compatible API 调用。
    embedding_backend: str = "openai"  # openai | local
    embedding_model: str = "text-embedding-v4"
    embedding_base_url: str = DASHSCOPE_COMPATIBLE_URL
    embedding_api_key: str = ""
    embedding_dim: int = 1024
    embedding_device: str = "auto"
    embedding_dtype: str = "auto"
    embedding_batch: int = 5
    embedding_max_chars: int = 10000
    embedding_instruct: str = DEFAULT_INSTRUCT
    embedding_strict: bool = True
    embedding_cache_path: Path = Path("./data/store/embcache.db")
    embed_on_add: bool = True
    index_cache_users: int = 32

    # 可选 Reranker。默认关闭，不占用额外 GPU。
    # cross-encoder 的**打分单位**：turn=单条消息（历史默认）| round=整个会话轮
    # 实测（S1_S6_REPORT.md §2，text 路 1,040）：round 让 Hit@1 +0.0269 ✅、R@n_gold +0.0188 ✅，
    # 且不需要任何额外模型调用（只是把同轮另一条也拼进打分文本）。
    rerank_unit: str = "round"
    reranker: str = "none"  # none | qwen3 | vllm_gen
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_device: str = "auto"
    reranker_dtype: str = "auto"
    reranker_batch: int = 8
    reranker_max_len: int = 1024
    reranker_candidates: int = 30
    reranker_base_url: str = ""
    reranker_api_key: str = "EMPTY"
    reranker_vllm_model: str = ""
    reranker_instruct: str = DEFAULT_INSTRUCT
    rerank_alpha: float = 1.0
    rerank_prior: str = "rank"  # rank | score (用上游融合分数做先验)
    # 记忆服务内的 LLM 推理式重排（opt-in）。实测收益：
    #   Mem-Gallery +0.0174（CI [+0.0070,+0.0286]，机制验收门 PASS）、ATM-Bench +0.036
    # 代价：每次 /search 一次 LLM 调用（池 30 × 1400 字符 ≈ 6~12k input token）。
    # 默认关闭 —— 与 FlowGrid 的纪律一致：无成本预算确认的证据不默认启用。
    # 查询图理解（记忆侧）：/search 的 query 允许是 ContentPart[]，若含图片则用 VLM
    # 生成检索导向描述并并入查询文本。约束不变（embedding 仍是 text-embedding-v4），
    # answer 侧不受影响。实测（487 条带查询图样本，检索阶段）：R@1 +0.2141、Hit@1 +0.5914。
    # 默认关闭：需显式提供 VLM 端点；未配置时自动优雅退化为纯文本查询。
    query_vision: bool = False
    query_vision_base_url: str = ""
    query_vision_api_key: str = ""
    query_vision_model: str = "qwen2.5-vl-7b-instruct"
    query_vision_max_images: int = 2
    query_vision_timeout: float = 60.0
    query_vision_cache_path: str = ""
    # 专用 VL reranker（HTTP，记忆侧 /search 末段）。实测（VT_RERANK_RESULT.md §10）：
    #   查询含图：+0.0463 R@1（候选带图 +0.0510）；查询不含图：−0.0745（负收益）
    # ⇒ 默认带 requires_query_image 硬闸：仅当查询里确有图片时才重排。
    # 关系感知证据选择（结构化表示 + 通用关系规则 + 逐对图像证据关系）。
    # 实测（STRUCTURED_MEMORY_PROBE.md §8，全 image 路 487 条）：R@1 +0.0311 ✅、Hit@1 +0.0924 ✅、
    # R@n_gold +0.0614 ✅（vs 只上 4B）；全库换算 R@1 0.4396 → 0.4630。
    # 硬闸：仅在查询含图时启用（text 路未验证）；生效时**替代** vl_rerank。
    relation_select: bool = False
    relation_select_base_url: str = ""
    relation_select_api_key: str = ""
    relation_select_model: str = "qwen2.5-vl-7b-instruct"
    relation_select_pool: int = 14
    relation_select_timeout: float = 90.0
    relation_select_workers: int = 8
    relation_select_pairwise: bool = True
    relation_select_requires_query_image: bool = True
    relation_select_cache_path: str = ""
    vl_rerank: bool = False
    vl_rerank_base_url: str = ""
    vl_rerank_model: str = "Qwen3-VL-Reranker-8B"
    vl_rerank_pool: int = 5
    vl_rerank_timeout: float = 60.0
    vl_rerank_workers: int = 8
    vl_rerank_include_doc_image: bool = False
    vl_rerank_requires_query_image: bool = True
    vl_rerank_max_query_images: int = 1
    vl_rerank_cache_path: str = ""
    # 置信度门控（S1_S6_REPORT.md §4）：内层（4B）top1-top2 的分数间距 > 该阈值时**不调 LLM**，
    # 直接保留内层序——因为实测"改错"主要来自覆盖掉已对的 top-1。
    # 0 = 关闭门控（总是调 LLM）。2 折交叉验证下最优阈值落在 margin 的 40–60% 分位。
    llm_rerank_gate_margin: float = 0.0
    # LLM 档提示：round_rank=产线原提示（排序）| evidence_set=集合级多选
    # 实测（S1_S6_REPORT.md §2/§3）：evidence_set 必须配合门控才为正。
    llm_rerank_prompt: str = "round_rank"
    # 若查询文本含该标记则**跳过 LLM 档**（image 路已有自己的关系选择档，且 LLM 档在该路实测为负：
    # T4：4B→LLM 改对 61 / 改错 95）。设成 "[Query image]" 即可只在 text 路启用。
    llm_rerank_skip_marker: str = ""
    # 按 request 去重后再喂 LLM（修复：round_emit=adjacent + round 单位同分 →
    # 池窗口里一半槽位是同一轮的重复副本，稀释多选型提示）
    # 默认**开启**：这是正确性修复（LLM 不该看到同一轮的重复副本）；
    # 实测（S1_S6_REPORT/SUMMARY §1.11）：R@n_gold +0.0166 ✅、其余中性，无负向。
    llm_rerank_dedup_rounds: bool = True
    # Search 响应里 content 的形态：text=压平成字符串（默认，兼容性最好）；
    # parts=按 AML 规范返回 ContentPart[]（含图片时有用）。契约两种都允许。
    search_content_format: str = "text"
    # LLM 档单次调用超时（秒）：超时即退化为内层 4B 排序，绝不拖垮 /search
    llm_rerank_timeout: float = 20.0
    llm_rerank: bool = False
    llm_rerank_base_url: str = ""
    llm_rerank_api_key: str = ""
    llm_rerank_model: str = "qwen-flash"
    llm_rerank_pool: int = 30          # 交给 LLM 的候选数（4B 初筛后）
    llm_rerank_layout: str = "round"   # round（补同轮另一条，实测最优）| raw
    llm_rerank_doc_chars: int = 1400   # 证据长度：ATM 上 420 字符会 -0.044
    llm_rerank_topk_ask: int = 10

    # Hybrid / RRF
    retrieval_limit: int = 120
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    # Round-level (request-level) dense channel — preprocessing-stage R@1
    round_leg: bool = False          # add the concatenated user+assistant channel
    round_rrf_k: int = 20            # RRF k (only used when fusion='rrf')
    # 加权 RRF（借鉴 FlowGrid：特征权重 1.0 / 词法权重 0.1）。
    # 他们的扫描结论：词法权重越大 MRR 越高、但 Recall@20 越低；0.1 是唯一 Pareto 安全点。
    rrf_weight_feature: float = 1.0  # dense + round 两路（"特征"路）
    rrf_weight_lexical: float = 1.0  # sparse 路（BM25+anchor，"词法"路）
    # 聚合视图通道（window / session-segment）——FlowGrid 阶梯的 L1。
    # 视图只用于**召回打分**，命中后按 source_memory_ids 归因到原始消息；
    # 原始消息始终是返回的证据。
    # 视图索引方式：lexical（复用 BM25+anchor，零 embedding —— FlowGrid 用 FTS5）| dense
    view_mode: str = "lexical"
    view_window: bool = False
    view_window_size: int = 3
    view_window_overlap: int = 1
    view_segment: bool = False
    view_segment_max_messages: int = 12
    view_segment_max_gap_seconds: int = 1800
    view_weight: float = 1.0
    view_max_chars: int = 2000
    # 契约不变量：top_k 中保底留给**原始消息**的槽位比例（FlowGrid message_slot_ratio）。
    message_slot_ratio: float = 0.5
    # 最坏代价上界（界定单次 Search 的代价；FlowGrid max_candidates / max_query_tokens）
    max_candidates: int = 400
    max_query_tokens: int = 64
    round_weight: float = 1.0        # weight of the round-dense channel
    round_max_chars: int = 2000      # max chars per concatenated round document
    # 默认 auto：**不识别数据集**，只看可观测的 round 结构
    # （存在多消息轮 → 通道高度相关，rrf 的排名鲁棒性更优；单消息轮 → 只剩
    #  dense+sparse 两个异质通道，归一化分数融合更优）。因此它按 user 自适应，
    # 混合语料下每个 user 各自落到正确 regime，而不是把两套数据当同一分布。
    # 依据（result/fusion-mixed__*.json，混合池配对 bootstrap）：
    #   auto > blend > rrf；auto 在 Mem-Gallery 取 rrf(0.2899)、在 ATM 取 blend(0.4923)。
    fusion: str = "auto"             # auto | blend | znorm | rrf
    blend_beta: float = 0.5          # blend 中 znorm 分量的权重（混编语料默认）
    dense_leg_weight: float = 1.0    # turn-dense channel weight (fusion='znorm')
    sparse_leg_weight: float = 1.0   # BM25+anchor channel weight (fusion='znorm')
    fusion_pool: str = "topk"        # topk (normalise in each channel's head) | full
    round_emit: str = "auto"         # auto | best_first(无rerank) | adjacent(有rerank)
    # Memory-Side 优化配置 (完全符合比赛契约)
    search_image_limit: int = -1  # -1: 全量原图; 0: 纯文本(离线描述替代像素); >0: 最多返回N张图
    search_dedup: bool = False     # 内存侧近重复过滤
    search_id_priority: bool = False  # 内存侧结构化 ID 优先锚定

    @classmethod
    def from_env(cls):
        raw_backend = os.getenv("AMC_EMBEDDING_BACKEND", "openai").strip().lower()

        # 允许配置中写 dashscope/aliyun/bailian，内部统一走现有
        # OpenAI-compatible provider，避免增加重复实现。
        backend = {
            "dashscope": "openai",
            "aliyun": "openai",
            "bailian": "openai",
        }.get(raw_backend, raw_backend)

        if backend == "local":
            default_model = "Qwen/Qwen3-Embedding-0.6B"
            default_base_url = ""
            default_batch = 16
            default_dim = 0
        else:
            default_model = "text-embedding-v4"
            default_base_url = DASHSCOPE_COMPATIBLE_URL
            default_batch = 5
            default_dim = 1024

        embedding_api_key = (
            os.getenv("AMC_EMBEDDING_API_KEY", "").strip()
            or os.getenv("DASHSCOPE_API_KEY", "").strip()
        )

        return cls(
            db_path=Path(os.getenv("AMC_DB_PATH", "./data/store/memory.db")),
            blob_dir=Path(os.getenv("AMC_BLOB_DIR", "./data/store/blobs")),
            retriever=os.getenv("AMC_RETRIEVER", "hybrid"),
            max_return=env_int("AMC_MAX_RETURN", 50),
            system_key=os.getenv("MEMORY_SYSTEM_KEY", ""),
            timezone_offset_hours=env_int("AMC_TIMEZONE_OFFSET_HOURS", 8),
            min_score=env_float("AMC_MIN_SCORE", 0.0),
            use_relative_time=env_bool("AMC_USE_RELATIVE_TIME", False),

            embedding_backend=backend,
            embedding_model=os.getenv("AMC_EMBEDDING_MODEL", default_model),
            embedding_base_url=os.getenv("AMC_EMBEDDING_BASE_URL", default_base_url),
            embedding_api_key=embedding_api_key,
            embedding_dim=env_int("AMC_EMBEDDING_DIM", default_dim),
            embedding_device=os.getenv("AMC_EMBEDDING_DEVICE", "auto"),
            embedding_dtype=os.getenv("AMC_EMBEDDING_DTYPE", "auto").strip().lower(),
            embedding_batch=env_int("AMC_EMBEDDING_BATCH", default_batch),
            embedding_max_chars=env_int("AMC_EMBEDDING_MAX_CHARS", 10000),
            embedding_instruct=os.getenv("AMC_EMBEDDING_INSTRUCT", DEFAULT_INSTRUCT),
            embedding_strict=env_bool("AMC_EMBEDDING_STRICT", True),
            embedding_cache_path=Path(
                os.getenv("AMC_EMBED_CACHE_PATH", "./data/store/embcache.db")
            ),
            embed_on_add=env_bool("AMC_EMBED_ON_ADD", True),
            index_cache_users=env_int("AMC_INDEX_CACHE_USERS", 32),

            reranker=os.getenv("AMC_RERANKER", "none").strip().lower(),
            reranker_model=os.getenv(
                "AMC_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B"
            ),
            reranker_device=os.getenv("AMC_RERANKER_DEVICE", "auto"),
            reranker_dtype=os.getenv("AMC_RERANKER_DTYPE", "auto").strip().lower(),
            reranker_batch=env_int("AMC_RERANKER_BATCH", 8),
            reranker_max_len=env_int("AMC_RERANKER_MAX_LEN", 1024),
            reranker_candidates=env_int("AMC_RERANKER_CANDIDATES", 30),
            reranker_base_url=os.getenv(
                "AMC_RERANKER_BASE_URL",
                os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8002/v1"),
            ),
            reranker_api_key=os.getenv(
                "AMC_RERANKER_API_KEY",
                os.getenv("OPENAI_API_KEY", "EMPTY"),
            ),
            reranker_vllm_model=os.getenv(
                "AMC_RERANKER_VLLM_MODEL",
                os.getenv("OPENAI_MODEL", ""),
            ),
            reranker_instruct=os.getenv("AMC_RERANKER_INSTRUCT", DEFAULT_INSTRUCT),
            rerank_alpha=env_float("AMC_RERANK_ALPHA", 1.0),
            rerank_prior=os.getenv("AMC_RERANK_PRIOR", "rank").strip().lower(),
            rrf_weight_feature=env_float("AMC_RRF_W_FEATURE", 1.0),
            rrf_weight_lexical=env_float("AMC_RRF_W_LEXICAL", 1.0),
            view_mode=os.getenv("AMC_VIEW_MODE", "lexical").strip().lower(),
            view_window=env_bool("AMC_VIEW_WINDOW", False),
            view_window_size=env_int("AMC_VIEW_WINDOW_SIZE", 3),
            view_window_overlap=env_int("AMC_VIEW_WINDOW_OVERLAP", 1),
            view_segment=env_bool("AMC_VIEW_SEGMENT", False),
            view_segment_max_messages=env_int("AMC_VIEW_SEGMENT_MAX_MESSAGES", 12),
            view_segment_max_gap_seconds=env_int("AMC_VIEW_SEGMENT_MAX_GAP_SECONDS", 1800),
            view_weight=env_float("AMC_VIEW_WEIGHT", 1.0),
            view_max_chars=env_int("AMC_VIEW_MAX_CHARS", 2000),
            message_slot_ratio=env_float("AMC_MESSAGE_SLOT_RATIO", 0.5),
            max_candidates=env_int("AMC_MAX_CANDIDATES", 400),
            max_query_tokens=env_int("AMC_MAX_QUERY_TOKENS", 64),
            relation_select=env_bool("AMC_RELATION_SELECT", False),
            relation_select_base_url=os.getenv("AMC_RELATION_SELECT_BASE_URL", ""),
            relation_select_api_key=os.getenv("AMC_RELATION_SELECT_API_KEY",
                                              os.getenv("OPENAI_API_KEY",
                                                        os.getenv("AMC_LLM_RERANK_API_KEY",
                                                                  os.getenv("AMC_EMBEDDING_API_KEY", "")))),
            relation_select_model=os.getenv("AMC_RELATION_SELECT_MODEL",
                                            "qwen2.5-vl-7b-instruct"),
            relation_select_pool=env_int("AMC_RELATION_SELECT_POOL", 10),
            relation_select_timeout=float(os.getenv("AMC_RELATION_SELECT_TIMEOUT", "90")),
            relation_select_workers=env_int("AMC_RELATION_SELECT_WORKERS", 8),
            relation_select_pairwise=env_bool("AMC_RELATION_SELECT_PAIRWISE", True),
            relation_select_requires_query_image=env_bool(
                "AMC_RELATION_SELECT_REQUIRES_IMAGE", True),
            relation_select_cache_path=os.getenv("AMC_RELATION_SELECT_CACHE", ""),
            vl_rerank=env_bool("AMC_VL_RERANK", False),
            vl_rerank_base_url=os.getenv("AMC_VL_RERANK_BASE_URL", ""),
            vl_rerank_model=os.getenv("AMC_VL_RERANK_MODEL", "Qwen3-VL-Reranker-8B"),
            vl_rerank_pool=env_int("AMC_VL_RERANK_POOL", 5),
            vl_rerank_timeout=float(os.getenv("AMC_VL_RERANK_TIMEOUT", "60")),
            vl_rerank_workers=env_int("AMC_VL_RERANK_WORKERS", 8),
            vl_rerank_include_doc_image=env_bool("AMC_VL_RERANK_DOC_IMAGE", False),
            vl_rerank_requires_query_image=env_bool("AMC_VL_RERANK_REQUIRES_IMAGE", True),
            vl_rerank_max_query_images=env_int("AMC_VL_RERANK_MAX_QUERY_IMAGES", 1),
            vl_rerank_cache_path=os.getenv("AMC_VL_RERANK_CACHE", ""),
            query_vision=env_bool("AMC_QUERY_VISION", False),
            query_vision_base_url=os.getenv("AMC_QUERY_VISION_BASE_URL",
                                            os.getenv("AMC_LLM_RERANK_BASE_URL",
                                                      os.getenv("AMC_EMBEDDING_BASE_URL", ""))),
            query_vision_api_key=os.getenv("AMC_QUERY_VISION_API_KEY",
                                           os.getenv("OPENAI_API_KEY",
                                                     os.getenv("AMC_LLM_RERANK_API_KEY",
                                                               os.getenv("AMC_EMBEDDING_API_KEY", "")))),
            query_vision_model=os.getenv("AMC_QUERY_VISION_MODEL", "qwen2.5-vl-7b-instruct"),
            query_vision_max_images=env_int("AMC_QUERY_VISION_MAX_IMAGES", 2),
            query_vision_timeout=float(os.getenv("AMC_QUERY_VISION_TIMEOUT", "60")),
            query_vision_cache_path=os.getenv("AMC_QUERY_VISION_CACHE", ""),
            llm_rerank_gate_margin=float(os.getenv("AMC_LLM_RERANK_GATE_MARGIN", "0")),
            llm_rerank_prompt=os.getenv("AMC_LLM_RERANK_PROMPT", "round_rank"),
            llm_rerank_skip_marker=os.getenv("AMC_LLM_RERANK_SKIP_MARKER", ""),
            llm_rerank_dedup_rounds=env_bool("AMC_LLM_RERANK_DEDUP_ROUNDS", True),
            search_content_format=os.getenv("AMC_SEARCH_CONTENT_FORMAT", "text").strip().lower(),
            llm_rerank_timeout=float(os.getenv("AMC_LLM_RERANK_TIMEOUT", "20")),
            llm_rerank=env_bool("AMC_LLM_RERANK", False),
            # 比赛要求 LLM 相关组件用 gpt-4o-mini（Reranker 不限）。
            # 密钥回退链：AMC_LLM_RERANK_API_KEY -> OPENAI_API_KEY -> 嵌入用的 key。
            llm_rerank_base_url=os.getenv("AMC_LLM_RERANK_BASE_URL",
                                          os.getenv("OPENAI_BASE_URL",
                                                    os.getenv("AMC_EMBEDDING_BASE_URL", ""))),
            llm_rerank_api_key=os.getenv("AMC_LLM_RERANK_API_KEY",
                                         os.getenv("OPENAI_API_KEY",
                                                   os.getenv("AMC_EMBEDDING_API_KEY", ""))),
            llm_rerank_model=os.getenv("AMC_LLM_RERANK_MODEL",
                                       os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            llm_rerank_pool=env_int("AMC_LLM_RERANK_POOL", 30),
            llm_rerank_layout=os.getenv("AMC_LLM_RERANK_LAYOUT", "round").strip().lower(),
            llm_rerank_doc_chars=env_int("AMC_LLM_RERANK_DOC_CHARS", 1400),
            llm_rerank_topk_ask=env_int("AMC_LLM_RERANK_TOPK_ASK", 10),

            retrieval_limit=env_int("AMC_RETRIEVAL_LIMIT", 120),
            rrf_k=env_int("AMC_RRF_K", 60),
            dense_weight=env_float("AMC_DENSE_WEIGHT", 1.0),
            sparse_weight=env_float("AMC_SPARSE_WEIGHT", 1.0),
            round_leg=env_bool("AMC_ROUND_LEG", False),
            round_rrf_k=env_int("AMC_ROUND_RRF_K", 20),
            round_weight=env_float("AMC_ROUND_WEIGHT", 1.0),
            round_max_chars=env_int("AMC_ROUND_MAX_CHARS", 2000),
            fusion=os.getenv("AMC_FUSION", "auto").strip().lower(),
            blend_beta=env_float("AMC_BLEND_BETA", 0.5),
            dense_leg_weight=env_float("AMC_DENSE_LEG_WEIGHT", 1.0),
            sparse_leg_weight=env_float("AMC_SPARSE_LEG_WEIGHT", 1.0),
            fusion_pool=os.getenv("AMC_FUSION_POOL", "topk").strip().lower(),
            round_emit=os.getenv("AMC_ROUND_EMIT", "auto").strip().lower(),
            search_image_limit=env_int("AMC_SEARCH_IMAGE_LIMIT", -1),
            search_dedup=env_bool("AMC_SEARCH_DEDUP", False),
            search_id_priority=env_bool("AMC_SEARCH_ID_PRIORITY", False),
        )

    def validate(self):
        errors = []
        known = {
            "tfidf", "bm25", "bm25_anchor", "anchor",
            "dense", "embedding", "hybrid", "hybrid_rrf", "hybrid_round",
        }

        base = self.retriever.lower().replace("+", "_").replace("-", "_")
        if base.endswith("_rerank"):
            base = base[:-len("_rerank")]
        base = {
            "anchor": "bm25_anchor",
            "embedding": "dense",
            "hybrid_rrf": "hybrid",
        }.get(base, base)

        if base not in known:
            errors.append(f"unknown AMC_RETRIEVER={self.retriever}")

        if self.view_mode not in {"lexical", "dense"}:
            errors.append(f"bad AMC_VIEW_MODE={self.view_mode} (lexical|dense)")

        if self.llm_rerank_layout not in {"round", "raw"}:
            errors.append(f"bad AMC_LLM_RERANK_LAYOUT={self.llm_rerank_layout}")

        if self.rerank_prior not in {"rank", "score"}:
            errors.append(f"bad AMC_RERANK_PRIOR={self.rerank_prior}")

        if self.round_emit not in {"auto", "best_first", "adjacent"}:
            errors.append(f"bad AMC_ROUND_EMIT={self.round_emit}")

        if self.fusion_pool not in {"topk", "full"}:
            errors.append(f"bad AMC_FUSION_POOL={self.fusion_pool}")

        if self.fusion not in {"blend", "znorm", "rrf", "auto"}:
            errors.append(f"bad AMC_FUSION={self.fusion} (blend|znorm|rrf|auto)")
        if not 0.0 <= self.blend_beta <= 1.0:
            errors.append(f"bad AMC_BLEND_BETA={self.blend_beta}")

        if self.embedding_backend not in {"local", "openai"}:
            errors.append(f"bad AMC_EMBEDDING_BACKEND={self.embedding_backend}")

        if self.reranker not in {"none", "qwen3", "vllm_gen"}:
            errors.append(f"bad AMC_RERANKER={self.reranker}")

        for label, dtype in (
            ("AMC_EMBEDDING_DTYPE", self.embedding_dtype),
            ("AMC_RERANKER_DTYPE", self.reranker_dtype),
        ):
            if dtype not in DTYPES:
                errors.append(f"bad {label}={dtype}")

        if not 0.0 <= self.rerank_alpha <= 1.0:
            errors.append("AMC_RERANK_ALPHA must be in [0, 1]")

        for label, value in (
            ("AMC_MAX_RETURN", self.max_return),
            ("AMC_RETRIEVAL_LIMIT", self.retrieval_limit),
            ("AMC_INDEX_CACHE_USERS", self.index_cache_users),
            ("AMC_EMBEDDING_BATCH", self.embedding_batch),
        ):
            if value <= 0:
                errors.append(f"{label} must be > 0")

        if self.embedding_dim < 0:
            errors.append("AMC_EMBEDDING_DIM must be >= 0")

        uses_dense = base in {"dense", "hybrid"}
        if uses_dense and self.embedding_backend == "openai":
            if not self.embedding_base_url:
                errors.append("AMC_EMBEDDING_BASE_URL is required for API embedding")

            is_dashscope = (
                "dashscope.aliyuncs.com" in self.embedding_base_url
                or "dashscope-intl.aliyuncs.com" in self.embedding_base_url
            )
            if is_dashscope and not self.embedding_api_key:
                errors.append(
                    "DASHSCOPE_API_KEY or AMC_EMBEDDING_API_KEY is required for "
                    "text-embedding-v4. Alibaba Cloud RAM AccessKey ID/Secret "
                    "cannot be used directly as the DashScope inference "
                    "Bearer token."
                )

        if (
            self.embedding_backend == "local"
            and self.embedding_model == "text-embedding-v4"
        ):
            errors.append(
                "text-embedding-v4 is a DashScope API model, not a local "
                "HuggingFace repository. Use AMC_EMBEDDING_BACKEND=openai."
            )

        if self.reranker != "none" and self.reranker_candidates <= 0:
            errors.append("AMC_RERANKER_CANDIDATES must be > 0")

        if self.reranker == "vllm_gen" and not self.reranker_vllm_model:
            errors.append("AMC_RERANKER_VLLM_MODEL / OPENAI_MODEL is required")

        return errors
