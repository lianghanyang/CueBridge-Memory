#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# 配置来源优先级：显式 AMC_ENV_FILE > 仓库 .env > 本地 .env.alibaba（兼容主仓开发）
ENV_FILE="${AMC_ENV_FILE:-${ROOT}/.env}"
[[ -f "${ENV_FILE}" ]] || ENV_FILE="${AMC_ENV_FILE:-${ROOT}/.env.alibaba}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi


# 2. 若重排器被显式设为 none，强制校正为开启（本系统严重依赖轮级精排）
if [[ "${AMC_RERANKER:-none}" == "none" ]]; then
  export AMC_RERANKER="qwen3"
fi

# ── 统一新配置（见 SUMMARY_REPORT.md §1.14）───────────────────────────────
# 旧默认是 `hybrid`（无 round 通道），实测 round 通道在 Mem-Gallery 上 R@1 +0.0258 ✅。
# 若 env 里写的是旧的 `hybrid`，这里**升级为 hybrid_round**；要保留旧行为设
# AMC_RETRIEVER_ALLOW_LEGACY=1。
if [[ "${AMC_RETRIEVER_ALLOW_LEGACY:-0}" != "1" && "${AMC_RETRIEVER:-hybrid}" == "hybrid" ]]; then
  echo "[config] AMC_RETRIEVER=hybrid -> hybrid_round（统一新配置；设 AMC_RETRIEVER_ALLOW_LEGACY=1 可保留旧行为）"
  export AMC_RETRIEVER="hybrid_round"
fi
if [[ "${AMC_RETRIEVER:-}" != *"rerank"* ]]; then
  export AMC_RETRIEVER="${AMC_RETRIEVER:-hybrid_round}_rerank"
fi

# 3. 自动探测或配置 Reranker 模型权重
# 统一新配置用 **4B**（阶梯 L6 实测 +0.0907 ✅）；旧的探测逻辑会优先 0.6B，是生产/评测差异之一。
LOCAL_4B_SNAP="/home/hi/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5"
if [[ -z "${AMC_RERANKER_MODEL:-}" || "${AMC_RERANKER_MODEL}" == "Qwen/Qwen3-Reranker-0.6B" ]]; then
  if [[ -d "${LOCAL_4B_SNAP}" ]]; then
    export AMC_RERANKER_MODEL="${LOCAL_4B_SNAP}"
  elif [[ -d "/mnt/newdisk/model/Qwen3-Reranker-4B" ]]; then
    export AMC_RERANKER_MODEL="/mnt/newdisk/model/Qwen3-Reranker-4B"
  else
    export AMC_RERANKER_MODEL="Qwen/Qwen3-Reranker-4B"
  fi
fi

# ── 统一新配置的其余开关（都可被 env 覆盖）──────────────────────────────
export AMC_FUSION="${AMC_FUSION:-auto}"                    # 只按可观测的轮结构路由
export AMC_RERANK_UNIT="${AMC_RERANK_UNIT:-round}"         # S2：按"轮"打分（0 额外调用）
export AMC_LLM_RERANK="${AMC_LLM_RERANK:-1}"               # LLM 档（text 路）
export AMC_LLM_RERANK_BASE_URL="${AMC_LLM_RERANK_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export AMC_LLM_RERANK_MODEL="${AMC_LLM_RERANK_MODEL:-qwen-flash}"
export AMC_LLM_RERANK_PROMPT="${AMC_LLM_RERANK_PROMPT:-round_rank}"
export AMC_LLM_RERANK_GATE_MARGIN="${AMC_LLM_RERANK_GATE_MARGIN:-0}"      # 门控关闭
export AMC_LLM_RERANK_DEDUP_ROUNDS="${AMC_LLM_RERANK_DEDUP_ROUNDS:-1}"    # 窗口按轮去重
export AMC_LLM_RERANK_POOL="${AMC_LLM_RERANK_POOL:-12}"
export AMC_LLM_RERANK_SKIP_MARKER="${AMC_LLM_RERANK_SKIP_MARKER:-[Query image]}"
# image 路：查询图桥 + 关系感知证据选择（需本地 VLM 端点；不可用时优雅退化）
export AMC_QUERY_VISION="${AMC_QUERY_VISION:-1}"
export AMC_QUERY_VISION_BASE_URL="${AMC_QUERY_VISION_BASE_URL:-http://127.0.0.1:8002/v1}"
export AMC_QUERY_VISION_MODEL="${AMC_QUERY_VISION_MODEL:-/mnt/newdisk/model/Qwen2.5-VL-7B-Instruct}"
export AMC_RELATION_SELECT="${AMC_RELATION_SELECT:-1}"
export AMC_RELATION_SELECT_BASE_URL="${AMC_RELATION_SELECT_BASE_URL:-http://127.0.0.1:8002/v1}"
export AMC_RELATION_SELECT_MODEL="${AMC_RELATION_SELECT_MODEL:-/mnt/newdisk/model/Qwen2.5-VL-7B-Instruct}"
export AMC_RELATION_SELECT_POOL="${AMC_RELATION_SELECT_POOL:-14}"
export AMC_RELATION_SELECT_PAIRWISE="${AMC_RELATION_SELECT_PAIRWISE:-1}"
export AMC_RELATION_SELECT_REQUIRES_IMAGE="${AMC_RELATION_SELECT_REQUIRES_IMAGE:-1}"

export AMC_RERANKER_DEVICE="cuda"
export AMC_RERANKER_DTYPE="${AMC_RERANKER_DTYPE:-bfloat16}"

# 基础配置
export AMC_EMBEDDING_BACKEND="${AMC_EMBEDDING_BACKEND:-dashscope}"
export AMC_EMBEDDING_MODEL="${AMC_EMBEDDING_MODEL:-text-embedding-v4}"
export AMC_EMBEDDING_BASE_URL="${AMC_EMBEDDING_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export AMC_EMBEDDING_DIM="${AMC_EMBEDDING_DIM:-1024}"
export AMC_EMBEDDING_BATCH="${AMC_EMBEDDING_BATCH:-10}"
export AMC_EMBED_ON_ADD="${AMC_EMBED_ON_ADD:-1}"
export MEMORY_SYSTEM_KEY="${MEMORY_SYSTEM_KEY:-dev-memory-key}"

EMBEDDING_KEY="${AMC_EMBEDDING_API_KEY:-${DASHSCOPE_API_KEY:-}}"
if [[ -z "${EMBEDDING_KEY}" || "${EMBEDDING_KEY}" == __FILL_ME_* ]]; then
  echo "[error] 未设置有效的 DASHSCOPE_API_KEY / AMC_EMBEDDING_API_KEY" >&2
  exit 2
fi
export AMC_EMBEDDING_API_KEY="${EMBEDDING_KEY}"

HOST="${AMC_HOST:-127.0.0.1}"
PORT="${AMC_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CURRENT_PID=""
cleanup() {
  echo -e "\n[supervisor] Shutting down Memory supervisor..."
  if [[ -n "${CURRENT_PID}" ]]; then
    kill -TERM "${CURRENT_PID}" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup SIGINT SIGTERM

echo "=========================================================="
echo "[supervisor] Memory Node starting on GPU: ${CUDA_VISIBLE_DEVICES}"
echo "[supervisor] Retriever: ${AMC_RETRIEVER} | Reranker: ${AMC_RERANKER}"
echo "[supervisor] Reranker Model: ${AMC_RERANKER_MODEL}"
echo "[supervisor] rerank_unit=${AMC_RERANK_UNIT} fusion=${AMC_FUSION} | llm=${AMC_LLM_RERANK} prompt=${AMC_LLM_RERANK_PROMPT} gate=${AMC_LLM_RERANK_GATE_MARGIN} dedup=${AMC_LLM_RERANK_DEDUP_ROUNDS}"
echo "[supervisor] query_vision=${AMC_QUERY_VISION} relation_select=${AMC_RELATION_SELECT} pool=${AMC_RELATION_SELECT_POOL}"
echo "[supervisor] Service Endpoint: http://${HOST}:${PORT}"
echo "[supervisor] Auto-redeploy on GPU crash: ENABLED"
echo "=========================================================="

# 崩溃自愈守护循环：崩溃时不退化，自动重新加载权重部署
RESTART_COUNT=0
while true; do
  echo "[supervisor] Launching Memory service (attempt $((RESTART_COUNT + 1)))..."
  "${PYTHON_BIN}" -m scripts.serve_memory --host "${HOST}" --port "${PORT}" &
  CURRENT_PID=$!
  
  set +e
  wait "${CURRENT_PID}"
  EXIT_STATUS=$?
  set -e

  RESTART_COUNT=$((RESTART_COUNT + 1))
  echo "[supervisor] WARNING: Memory process exited with code ${EXIT_STATUS}."
  echo "[supervisor] Cleaning GPU ${CUDA_VISIBLE_DEVICES} state and redeploying in 3 seconds..."
  
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 3
done
