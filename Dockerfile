# CueBridge Memory — 记忆侧 Add/Search 服务
# 说明：服务需要 GPU 跑 Qwen3-Reranker-4B；纯 API 部署可把 AMC_RETRIEVER 调成不含本地 reranker 的组合。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，充分利用镜像层缓存
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 再拷代码（包 amc 在仓库根，靠 PYTHONPATH 暴露）
COPY pyproject.toml README.md ./
COPY amc/ ./amc/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# 数据与缓存目录（生产请挂卷持久化）
RUN mkdir -p /app/data/store /app/data/reports/deploy
VOLUME ["/app/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

# 生产环境用 .env / -e 注入密钥；不要写进镜像
CMD ["python", "-m", "scripts.serve_memory", "--host", "0.0.0.0", "--port", "8000"]
