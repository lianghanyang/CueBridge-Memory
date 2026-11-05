#!/usr/bin/env bash

# ⚠️ （修正版）：只清理"真正的服务进程"，**不再按字符串匹配 shell**
#   上一版用 `ps | grep 脚本名` ⇒ 会把我自己的 `bash -c '...serve_memory...'` 也杀掉。
#   这版只 kill 两类：
#     ① 直接执行 `python -m scripts.serve_memory` 的进程（cmdline 第 2 段是 python 且含 serve_memory）
#     ② 占用 8000/8020 端口的进程（ss -ltnp）
if [ "${AMC_START_GUARD:-1}" != "0" ]; then
  _SELF=$$; _PARENT=$PPID
  _OLD=$(ps -eo pid,args --no-headers 2>/dev/null | awk -v s="$_SELF" -v p="$_PARENT" '
    $1==s || $1==p {next}
    ($2 ~ /python/ && $0 ~ /scripts\.serve_memory/) {print $1}')
  _PORT=$(ss -ltnp 2>/dev/null | awk '/:8000|:8020/ {match($0,/pid=([0-9]+)/,m); if(m[1]!="") print m[1]}' | sort -u | tr '\n' ' ')
  _ALL=$(echo "$_OLD $_PORT" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
  if [ -n "${_ALL// /}" ]; then
    echo "[guard] 清理旧实例: $_ALL"
    for _p in $_ALL; do kill -9 "$_p" 2>/dev/null; done
    sleep 2
  fi
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

# 生产默认使用 .env.prod（由运行中的服务导出并包含对齐离线最优管线的开关）；
# 不存在时回退到 .env.alibaba（兼容旧流程）。
if [[ -n "${AMC_ENV_FILE:-}" ]]; then
  ENV_FILE="${AMC_ENV_FILE}"
elif [[ -f "${ROOT}/.env.prod" ]]; then
  ENV_FILE="${ROOT}/.env.prod"
else
  ENV_FILE="${ROOT}/.env.alibaba"
fi
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi


# 2. 解决 .env.alibaba 覆盖问题：如果被设为 none 或 hybrid，强制校正为开启 reranker
if [[ "${AMC_RERANKER:-none}" == "none" ]]; then
  export AMC_RERANKER="qwen3"
fi

# ── 生产对齐（2026-10-19）：text 路线上 = 离线 harness ─────────────────
#   AMC_RETRIEVER=hybrid（与离线 orig 池一致，需 AMC_RETRIEVER_ALLOW_LEGACY=1 才不被升级）
#   AMC_LLM_RERANK_POOL=70（离线 p70）
#   AMC_LLM_RERANK_PROMPT=evidence_set / RERANK_UNIT=round / GATE=0 / DEDUP=1
#   实测：text 路 Hit@1 = 0.786（与离线完全一致）；image 路差异源于离线用了
#   接口外的"查询图注"，线上不可复现（详见 SUMMARY_REPORT §1.15）
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
LOCAL_4B_SNAP="${HF_HOME:-~/.cache/huggingface}/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5"
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
export AMC_QUERY_VISION_BASE_URL="${AMC_QUERY_VISION_BASE_URL:-http://127.0.0.1:8000"
export AMC_QUERY_VISION_MODEL="${AMC_QUERY_VISION_MODEL:-/mnt/newdisk/model/Qwen2.5-VL-7B-Instruct}"
export AMC_RELATION_SELECT="${AMC_RELATION_SELECT:-1}"
export AMC_RELATION_SELECT_BASE_URL="${AMC_RELATION_SELECT_BASE_URL:-http://127.0.0.1:8000"
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

# ── 公网隧道自愈（Cloudflare named tunnel）───────────────────────────────
#   背景：tunnel 会静默死掉，公网变 530/1033，而本地一切正常 → 评测直接判 0。
#   这里由 supervisor 统一守护：启动时拉起、每 30s 探测、退出时回收。
TUNNEL_UUID="${AMC_TUNNEL_UUID:-1180c59c-85f6-4188-9169-7f80e4eb5d93}"
TUNNEL_BIN="${AMC_TUNNEL_BIN:-/usr/local/bin/cloudflared}"
TUNNEL_PROTOCOL="${AMC_TUNNEL_PROTOCOL:-http2}"
# 校园网实测：到 CF 边缘 **TCP/443 over IPv4 可用**；**UDP/7844（QUIC）被拦**。
# 所以正确组合是 http2 + IPv4（单独用 IPv4 配 QUIC 会连不上）。
TUNNEL_EDGE_IP="${AMC_TUNNEL_EDGE_IP:-4}"
# 第二连接器：**也用 http2**（实测 QUIC 掉线次数是 http2 的两倍，已弃用）
# ⚠️ 2026-11-02：默认改为**空**（原 http2 与主连接器相同 ⇒ 变成同协议冗余）
# 校园网封 UDP/7844(QUIC)，双协议 HA 用不了 ⇒ 只跑一个连接器即可。
TUNNEL_PROTOCOL_B="${AMC_TUNNEL_PROTOCOL_B-}"
TUNNEL_LOG="${ROOT}/data/reports/deploy/tunnel.log"
TUNNEL_AUTOSTART="${AMC_TUNNEL_AUTOSTART:-1}"
# ⚠️ 2026-11-02：隧道指向 **restart_guard**（而非 app 本身）。
# 理由：重启窗口里 app 会 refused，而 guard 常驻 ⇒ 返回 503+Retry-After，平台可重试。
# guard 由本脚本的 ensure_guard() 守护（监听 GUARD_PORT，转发到 app 的 PORT）。
GUARD_PORT="${AMC_GUARD_PORT:-8020}"
GUARD_ENABLED="${AMC_GUARD_ENABLED:-1}"
# 隧道目标：guard 启用时指向 GUARD_PORT，否则退回 app 的 PORT
TUNNEL_TARGET_PORT="${GUARD_PORT}"
[[ "${GUARD_ENABLED}" == "1" ]] || TUNNEL_TARGET_PORT="${PORT}"
WATCHDOG_PID=""

# 用 /proc/<pid>/cmdline 判断，避免 pgrep -f 误匹配执行命令的 shell 自身
cloudflared_has_protocol() {
  local proto="$1" p
  for p in $(pgrep -x cloudflared 2>/dev/null); do
    if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--protocol $proto"; then
      return 0
    fi
  done
  return 1
}

tunnel_alive() {
  cloudflared_has_protocol "${TUNNEL_PROTOCOL}"
}

start_tunnel() {
  [[ "${TUNNEL_AUTOSTART}" == "1" ]] || return 0
  [[ -x "${TUNNEL_BIN}" ]] || { echo "[tunnel] 未找到 ${TUNNEL_BIN}，跳过"; return 0; }
  [[ -f "${HOME}/.cloudflared/cert.pem" ]] || { echo "[tunnel] 无 ~/.cloudflared 凭据，跳过"; return 0; }
  if tunnel_alive; then echo "[tunnel] 已在运行"; return 0; fi
  mkdir -p "$(dirname "${TUNNEL_LOG}")"
  # 校园网实测：到 Cloudflare 边缘的 **IPv4 不通**（curl -4 超时），IPv6 正常。
  # 因此用 http2(TCP) + IPv4：实测 TCP/443 over IPv4 可连，而 UDP/7844(QUIC) 被拦。
  echo "[tunnel] 启动 ${TUNNEL_UUID} → http://${HOST}:${PORT}（protocol=${TUNNEL_PROTOCOL} edge-ip=${TUNNEL_EDGE_IP:-auto}）"
  setsid nohup "${TUNNEL_BIN}" tunnel --protocol "${TUNNEL_PROTOCOL}" \
    ${TUNNEL_EDGE_IP:+--edge-ip-version "${TUNNEL_EDGE_IP}"} \
    --url "http://${HOST}:${TUNNEL_TARGET_PORT}" run "${TUNNEL_UUID}" \
    >> "${TUNNEL_LOG}" 2>&1 < /dev/null &
  sleep 2
  tunnel_alive && echo "[tunnel] 已连接" || echo "[tunnel] 启动失败，详见 ${TUNNEL_LOG}"
}

guard_alive() {
  ss -ltn 2>/dev/null | grep -q ":${GUARD_PORT} " && return 0
  return 1
}

start_guard() {
  [[ "${GUARD_ENABLED}" == "1" ]] || return 0
  [[ -f "${ROOT}/data/reports/gold_sim/restart_guard.py" ]] || { echo "[guard] 脚本缺失，跳过"; return 0; }
  guard_alive && return 0
  echo "[guard] 启动 :${GUARD_PORT} → :${PORT}（重启窗口返回 503+Retry-After）"
  mkdir -p "${ROOT}/logs"
  setsid nohup python3 "${ROOT}/data/reports/gold_sim/restart_guard.py" \
    --listen "${GUARD_PORT}" --upstream "${PORT}" --retry-after 3 --health-always-200 \
    >> "${ROOT}/logs/guard.log" 2>&1 < /dev/null &
  sleep 2
  guard_alive && echo "[guard] 已监听" || echo "[guard] 启动失败，详见 logs/guard.log"
}

ensure_guard() {
  [[ "${GUARD_ENABLED}" == "1" ]] || return 0
  if ! guard_alive; then
    echo "[guard] 检测到 guard 断开，自动重连"
    start_guard
  fi
}

ensure_tunnel() {
  [[ "${TUNNEL_AUTOSTART}" == "1" ]] || return 0
  if ! tunnel_alive; then
    echo "[tunnel] 检测到隧道已断开，自动重连"
    start_tunnel
  fi
}

tunnel_alive_b() {
  # 目标：同协议 **两个** 实例（互为冗余）
  [[ -n "${TUNNEL_PROTOCOL_B}" ]] || return 1
  local n=0 p
  for p in $(pgrep -x cloudflared 2>/dev/null); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--protocol ${TUNNEL_PROTOCOL_B}" && n=$((n+1))
  done
  [[ ${n} -ge 2 ]]
}

start_tunnel_b() {
  # 第二个连接器（默认 QUIC），与主连接器（http2）并存 → 同一隧道双协议 HA
  [[ "${TUNNEL_AUTOSTART}" == "1" ]] || return 0
  [[ -x "${TUNNEL_BIN}" ]] || return 0
  [[ -f "${HOME}/.cloudflared/cert.pem" ]] || return 0
  tunnel_alive_b && return 0
  echo "[tunnel-b] 启动第二连接器（protocol=${TUNNEL_PROTOCOL_B}）"
  setsid nohup "${TUNNEL_BIN}" tunnel --protocol "${TUNNEL_PROTOCOL_B}" \
    --url "http://${HOST}:${TUNNEL_TARGET_PORT}" run "${TUNNEL_UUID}" \
    >> "${ROOT}/data/reports/deploy/tunnel2.log" 2>&1 < /dev/null &
  sleep 2
}

ensure_tunnel() {
  [[ "${TUNNEL_AUTOSTART}" == "1" ]] || return 0
  if ! tunnel_alive; then
    echo "[tunnel] 检测到主连接器断开，自动重连"
    start_tunnel
  fi
  if [[ -n "${TUNNEL_PROTOCOL_B}" ]]; then
    if ! tunnel_alive_b; then
      echo "[tunnel-b] 检测到第二连接器断开，自动重连"
      start_tunnel_b
    fi
  fi
}

tunnel_watchdog() {
  # 只做**本地进程**守护：不向公网发探测请求（Cloudflare 免费额度有流量限制）
  while true; do sleep 30; ensure_guard; ensure_tunnel; done
}

cleanup() {
  echo -e "\n[supervisor] Shutting down Memory supervisor..."
  if [[ -n "${CURRENT_PID}" ]]; then
    kill -TERM "${CURRENT_PID}" 2>/dev/null || true
  fi
  if [[ -n "${WATCHDOG_PID}" ]]; then
    kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
  fi
  if [[ "${AMC_TUNNEL_KEEP_ON_EXIT:-0}" != "1" ]] && tunnel_alive; then
    echo "[tunnel] 关闭隧道"
    pkill -f "cloudflared tunnel .*${TUNNEL_UUID}" 2>/dev/null || true
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

# 公网隧道：启动 + 独立看门狗
start_tunnel
if [[ -n "${TUNNEL_PROTOCOL_B}" ]]; then start_tunnel_b; fi
if [[ "${TUNNEL_AUTOSTART}" == "1" ]]; then
  tunnel_watchdog &
  WATCHDOG_PID=$!
  echo "[tunnel] 看门狗 pid=${WATCHDOG_PID}（每 30s 探测）"
fi

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
  ensure_guard
  ensure_tunnel
  sleep 3
done
