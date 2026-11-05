"""L1 用户档案/事实库的 DDL。

设计原则（来自需求）：
- 结构化档案是**派生层**，每条事实/交易都必须能回溯到原始消息（source_message_ids）
- 金额用**最小货币单位整数**（amount_minor），避免浮点误差
- 交易日期（occurred_at）与消息写入日期（created_at）分开
- 同一笔消费多次提及 → 一个事件 + 多个来源（多对多）
"""
from __future__ import annotations

DDL = [
    # 原始证据引用（多对多）：同一事件可被多条消息提及
    """
    CREATE TABLE IF NOT EXISTS profile_transaction_sources (
        transaction_id TEXT NOT NULL,
        memory_id      TEXT NOT NULL,          -- 指向 memories.id
        request_id     TEXT,
        quote          TEXT,                   -- 支撑该事件的原文片段
        created_at     TEXT,
        PRIMARY KEY (transaction_id, memory_id)
    )
    """,
    # 消费事件
    """
    CREATE TABLE IF NOT EXISTS profile_transactions (
        transaction_id  TEXT PRIMARY KEY,
        user_id         TEXT NOT NULL,
        occurred_at     TEXT,                  -- 实际消费日期（可空）
        merchant        TEXT,
        item            TEXT,                  -- 商品原文
        category        TEXT,                  -- 归一化类别，如 coffee_maker
        amount_minor    INTEGER,               -- 最小货币单位（如美分）
        currency        TEXT,                  -- USD / GBP / SGD ...
        status          TEXT NOT NULL DEFAULT 'pending',  -- pending|confirmed|refunded|void
        confidence      REAL,
        extractor       TEXT,                  -- 提取器版本
        created_at      TEXT,
        updated_at      TEXT
    )
    """,
    # 非交易事实：偏好/兴趣/事件/承诺
    """
    CREATE TABLE IF NOT EXISTS profile_facts (
        fact_id          TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        fact_type        TEXT NOT NULL,        -- preference|interest|event|commitment|entity|relation
        subject          TEXT,                 -- 通常 'user'
        predicate        TEXT,                 -- enjoys / plans / lives_in ...
        object           TEXT,                 -- 规范化值
        polarity         TEXT,                 -- positive|negative|neutral
        confidence       REAL,
        evidence_type    TEXT,                 -- explicit_statement|inferred|purchase
        source_memory_ids TEXT,                -- JSON 数组
        valid_from       TEXT,
        valid_to         TEXT,
        status           TEXT NOT NULL DEFAULT 'active',  -- active|superseded|retracted
        created_at       TEXT,
        updated_at       TEXT
    )
    """,
    #    问题：query_expand 把"那次/上次/当时"当**停用词删掉**了（只有词面匹配 + 时间窗）
    #    ⇒ relational_association(1019) + implicit_preference(122) = **1,141 题**缺锚点
    #    抽取只依赖 5 个字段：text / role / created_at / session_id / id（ev_id 可空）
    """
    CREATE TABLE IF NOT EXISTS profile_refs (
        user_id          TEXT NOT NULL,
        phrase           TEXT NOT NULL,   -- 指代短语（"那次"/"上次"/"当时"…）
        date             TEXT,            -- 该记录日期 YYYY-MM-DD
        anchor_memory_id TEXT,            -- 锚点记录 id
        session_id       TEXT,
        snippet          TEXT,            -- 该短语后 ~160 字上下文（供检索时判断）
        score            REAL DEFAULT 1.0,
        created_at       TEXT,
        PRIMARY KEY (user_id, anchor_memory_id, phrase)
    )
    """,
    """CREATE INDEX IF NOT EXISTS idx_refs_user_phrase ON profile_refs(user_id, phrase)""",
    # 处理进度/覆盖状态（用于 coverage_status）
    """
    CREATE TABLE IF NOT EXISTS profile_ingest (
        user_id      TEXT NOT NULL,
        memory_id    TEXT NOT NULL,
        status       TEXT NOT NULL,            -- pending|done|failed
        extractor    TEXT,
        updated_at   TEXT,
        PRIMARY KEY (user_id, memory_id)
    )
    """,
    # 可增长的类别表：/add 时由 LLM 决定"复用现有类别"还是"新建类别"
    """
    CREATE TABLE IF NOT EXISTS profile_categories (
        user_id     TEXT NOT NULL,
        name        TEXT NOT NULL,            -- 细粒度类别名（如 hotel_aberdeen / coffee_maker）
        description TEXT,                     -- 一句话描述，供检索时匹配问题
        parent      TEXT,                     -- 粗粒度分组（travel/food/...），用于兜底
        n_tx        INTEGER DEFAULT 0,
        created_at  TEXT,
        updated_at  TEXT,
        PRIMARY KEY (user_id, name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tx_user_cat ON profile_transactions(user_id, category, status)",
    "CREATE INDEX IF NOT EXISTS idx_tx_user_time ON profile_transactions(user_id, occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_fact_user_type ON profile_facts(user_id, fact_type, status)",
    "CREATE INDEX IF NOT EXISTS idx_ingest_status ON profile_ingest(user_id, status)",
]


def ensure_schema(conn) -> None:
    for stmt in DDL:
        conn.execute(stmt)
    conn.commit()
