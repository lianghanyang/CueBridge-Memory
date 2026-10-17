#!/usr/bin/env bash
# 上线前自检（preflight）：把"能不能提交"变成一条命令。
#
# 覆盖：
#   1. 必需环境变量（嵌入 key / gpt-4o-mini key / 对外鉴权 key）
#   2. Embedding 端点：text-embedding-v4 可用且维度=1024
#   3. gpt-4o-mini：/chat/completions 可用（比赛硬要求）
#   4. gpt-4o-mini：**视觉**输入是否可用（决定 image 路走哪个 profile）
#   5. GPU：是否有 ≥9GB 空闲显存放 4B cross-encoder
#   6. 记忆库：data/store/memory.db 存在且有数据
#
# 用法：bash scripts/aml_preflight.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

# 优先用比赛配置，其次用主仓配置
for f in .env.aml .env.alibaba; do
  if [[ -f "$f" ]]; then set -a; source "$f" 2>/dev/null || true; set +a; echo "[preflight] 载入 $f"; fi
done

EMB_BASE="${AMC_EMBEDDING_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
EMB_KEY="${AMC_EMBEDDING_API_KEY:-${DASHSCOPE_API_KEY:-}}"
LLM_BASE="${AMC_LLM_RERANK_BASE_URL:-https://9527.codes/v1}"
LLM_KEY="${AMC_LLM_RERANK_API_KEY:-${OPENAI_API_KEY:-}}"
LLM_MODEL="${AMC_LLM_RERANK_MODEL:-gpt-4o-mini}"

PASS=0; FAIL=0; WARN=0
ok(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
bad(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }
warn(){ echo "  ⚠️  $1"; WARN=$((WARN+1)); }

echo "== CueBridge Memory · 上线前自检 =="

# 1) 环境变量
[[ -n "${EMB_KEY}" ]] && ok "嵌入 key 已配置" || bad "缺 DASHSCOPE_API_KEY / AMC_EMBEDDING_API_KEY"
[[ -n "${LLM_KEY}" ]] && ok "LLM key 已配置" || bad "缺 OPENAI_API_KEY / AMC_LLM_RERANK_API_KEY"
[[ -n "${MEMORY_SYSTEM_KEY:-}" ]] && ok "对外鉴权 key 已配置（MEMORY_SYSTEM_KEY）" || warn "未配置 MEMORY_SYSTEM_KEY（服务将不校验鉴权）"

# 2) 嵌入
emb_code=$(curl -s -m 30 -o /tmp/pf_emb.json -w '%{http_code}' "${EMB_BASE%/}/embeddings" \
  -H "Authorization: Bearer ${EMB_KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${AMC_EMBEDDING_MODEL:-text-embedding-v4}\",\"input\":\"ping\"}" 2>/dev/null)
if [[ "${emb_code}" == "200" ]]; then
  dim=$(python3 -c "import json;print(len(json.load(open('/tmp/pf_emb.json'))['data'][0]['embedding']))" 2>/dev/null || echo '?')
  [[ "${dim}" == "1024" ]] && ok "text-embedding-v4 可用（维度 ${dim}）" || warn "嵌入可用但维度=${dim}（期望 1024）"
else
  bad "嵌入端点 -> ${emb_code} $(head -c 120 /tmp/pf_emb.json 2>/dev/null)"
fi

# 3) gpt-4o-mini 文本
chat_code=$(curl -s -m 40 -o /tmp/pf_chat.json -w '%{http_code}' "${LLM_BASE%/}/chat/completions" \
  -H "Authorization: Bearer ${LLM_KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":5}" 2>/dev/null)
if [[ "${chat_code}" == "200" ]]; then
  ok "gpt-4o-mini 文本调用可用"
else
  bad "gpt-4o-mini 文本调用 -> ${chat_code} $(head -c 150 /tmp/pf_chat.json 2>/dev/null)"
  # 关键诊断：该 token 的**分组**里到底有没有这个模型
  curl -s -m 20 "${LLM_BASE%/}/models" -H "Authorization: Bearer ${LLM_KEY}" -o /tmp/pf_models.json 2>/dev/null
  python3 - "${LLM_MODEL}" <<'PY' || true
import json, sys
want = sys.argv[1]
try:
    d = json.load(open('/tmp/pf_models.json'))
    ids = [m.get('id') for m in d.get('data', [])]
except Exception:
    print("         （无法读取 /models 列表）"); sys.exit(0)
if not ids:
    print("         （该 token 看不到任何模型）"); sys.exit(0)
if want in ids:
    print(f"         ⇒ 模型在清单里，但通道不可用（网关侧渠道无健康上游/额度）")
else:
    print(f"         ⇒ **该 token 的分组里没有 {want}**（可见 {len(ids)} 个：{', '.join(ids[:6])}{' …' if len(ids)>6 else ''}）")
    print(f"           需在网关侧把 {want} 的渠道加入该 token 所属分组（错误信息里的 group）")
PY
fi

# 4) gpt-4o-mini 视觉（1x1 PNG）
PNG_B64="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAhklEQVR4nO3awQmDQBBA0RgsIUVYguVYmuVYhCWkiJxzU0J4LPx3noX5zHWnc1keI3vqBX5VgFaAVoBWgFaAVoBWgDbffbC+9j+s8eV4b9eHh79AAVoBWgFaAVoBWgFaAVoBWgFaAVoBWgFaAVoBWgHa8AFTH56wArQCtAK0ArQCtAK04QM+fPQF2cAL6igAAAAASUVORK5CYII="
vis_code=$(curl -s -m 60 -o /tmp/pf_vis.json -w '%{http_code}' "${LLM_BASE%/}/chat/completions" \
  -H "Authorization: Bearer ${LLM_KEY}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${LLM_MODEL}\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what color?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${PNG_B64}\"}}]}]}" 2>/dev/null)
if [[ "${vis_code}" == "200" ]]; then
  ok "gpt-4o-mini 视觉输入可用 ⇒ image 路可走严格合规 profile"
else
  warn "gpt-4o-mini 视觉 -> ${vis_code}（image 路建议用本地 VLM profile）"
fi

# 5) GPU：先**逐卡实测**能否做 bf16 矩阵乘（有的卡查得到显存但一算就崩），再看空闲显存
gpu_probe() {  # $1=index -> "OK" / "FAIL:原因"
  CUDA_VISIBLE_DEVICES="$1" python3 - <<'PY' 2>/dev/null || echo "FAIL"
import torch
# 用接近真实负载的尺寸（小矩阵可能掩盖 kernel/驱动问题）
a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
float(c.float().sum().item())
print("OK")
PY
}
if command -v nvidia-smi >/dev/null 2>&1; then
  usable=""; maxfree=0
  while IFS=, read -r idx total used; do
    idx="${idx// /}"; free=$(( total - used ))
    [[ ${free} -gt ${maxfree} ]] && maxfree=${free}
    res="$(gpu_probe "${idx}")"
    if [[ "${res}" == "OK" ]]; then
      ok "GPU ${idx}：计算可用，空闲 ${free} MiB"
      [[ ${free} -ge 9000 && -z "${usable}" ]] && usable="${idx}"
    else
      warn "GPU ${idx}：**计算不可用**（bf16 matmul 失败），空闲 ${free} MiB 也不能用"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null)
  if [[ -n "${usable}" ]]; then
    ok "可部署 4B 的 GPU：index=${usable}（显存 ≥9GB 且计算可用）⇒ 用 CUDA_VISIBLE_DEVICES=${usable}"
  else
    warn "没有"计算可用且 ≥9GB 空闲"的 GPU（最大空闲 ${maxfree} MiB）⇒ 先用 AMC_RERANKER=none 跑通链路"
  fi
else
  warn "未检测到 nvidia-smi（无 GPU 环境）"
fi

# 6) 记忆库
DB="${AMC_DB_PATH:-./data/store/memory.db}"
if [[ -f "${DB}" ]]; then
  n=$(python3 -c "import sqlite3;print(sqlite3.connect('file:${DB}?mode=ro',uri=True).execute('select count(*) from memories').fetchone()[0])" 2>/dev/null || echo 0)
  [[ "${n}" -gt 0 ]] && ok "记忆库 ${DB}（${n} 条）" || bad "记忆库为空：${DB}"
else
  bad "记忆库不存在：${DB}"
fi

echo
echo "== 结果：通过 ${PASS}，失败 ${FAIL}，警告 ${WARN} =="
echo "  提示：gpt-4o-mini 若为 503 No available channel，需在网关侧启用通道；"
echo "        显存不足时用 AMC_RERANKER=none 仅验证 API 链路，指标会下降。"
[[ "${FAIL}" == "0" ]]
