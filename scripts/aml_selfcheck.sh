#!/usr/bin/env bash
# AML 提交自检：对着**公网地址**验证 Add/Search 契约、Token 认证与分数单调性。
#
# 用法：
#   bash scripts/aml_selfcheck.sh                       # 用 data/reports/deploy/public_url.txt
#   BASE=https://xxx.trycloudflare.com KEY=xxx bash scripts/aml_selfcheck.sh
#
# 检查项：
#   1) GET  /v1/memories/health            -> 200
#   2) POST /v1/memories/add               -> 200（Authorization: Token）
#   3) POST /v1/memories/search            -> 200 且返回 data 数组
#   4) 每条含 id/content/score/created_at，score 非递增
#   5) 错误 key                            -> 401
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"

URL_FILE="${ROOT}/data/reports/deploy/public_url.txt"
if [[ -z "${BASE:-}" && -f "${URL_FILE}" ]]; then
  # shellcheck disable=SC1090
  BASE="$(grep -m1 '^PUBLIC_BASE=' "${URL_FILE}" | cut -d= -f2-)"
fi
if [[ -z "${BASE:-}" ]]; then
  BASE="http://127.0.0.1:${AMC_PORT:-8000}"
  echo "[warn] 未找到公网地址，改测本地 ${BASE}"
fi
if [[ -z "${KEY:-}" ]]; then
  KEY="$(awk -F= '/^MEMORY_SYSTEM_KEY=/{v=substr($0,index($0,"=")+1); gsub(/["'"'"']/,"",v)} END{print v}' .env.alibaba 2>/dev/null)"
fi
[[ -z "${KEY}" ]] && KEY="dev-memory-key"

PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# 每次先清掉临时响应文件：否则请求失败时会把**上一次的旧响应**当成结果打印，误导排查
rm -f /tmp/aml_health.json /tmp/aml_add.json /tmp/aml_search.json 2>/dev/null || true

echo "== CueBridge Memory · AML 自检：${BASE} =="

# 1) health
code=$(curl -s -m 20 -o /tmp/aml_health.json -w '%{http_code}' "${BASE}/v1/memories/health" 2>/dev/null)
[[ "${code}" == "200" ]] && ok "health -> 200" || bad "health -> ${code}"

UID_="aml_selfcheck_$(date +%s)"
RID_="req_${UID_}"

# 2) add（Authorization: Token，报名表要求的形式）
TS_MS=$(( $(date +%s) * 1000 ))
add_body=$(cat <<JSON
{"user_id":"${UID_}","session_id":"${UID_}","request_id":"${RID_}","messages":[
 {"role":"user","content":"My favourite operating system is Debian GNU/Linux.","timestamp":${TS_MS}},
 {"role":"assistant","content":"Noted: you prefer Debian GNU/Linux.","timestamp":$((TS_MS + 5000))}]}
JSON
)
code=$(curl -s -m 60 -o /tmp/aml_add.json -w '%{http_code}' -X POST "${BASE}/v1/memories/add" \
  -H "Authorization: Token ${KEY}" -H 'Content-Type: application/json' -d "${add_body}" 2>/dev/null)
[[ "${code}" == "200" ]] && ok "add (Token) -> 200" || bad "add (Token) -> ${code} $(head -c 160 /tmp/aml_add.json)"

# 3) search
code=$(curl -s -m 90 -o /tmp/aml_search.json -w '%{http_code}' -X POST "${BASE}/v1/memories/search" \
  -H "Authorization: Token ${KEY}" -H 'Content-Type: application/json' \
  -d "{\"query\":\"Which operating system does the user prefer?\",\"user_id\":\"${UID_}\",\"top_k\":5}" 2>/dev/null)
if [[ "${code}" == "200" ]]; then
  ok "search (Token) -> 200"
else
  bad "search (Token) -> ${code} $(head -c 160 /tmp/aml_search.json)"
fi

# 4) 契约校验
python3 - <<'PY' && ok "响应契约（字段 + 分数非递增）" || bad "响应契约不满足"
import json, sys
try:
    d = json.load(open('/tmp/aml_search.json'))
except Exception as e:
    print('   解析失败:', e); sys.exit(1)
data = d.get('data')
if not isinstance(data, list):
    print('   缺少 data 数组'); sys.exit(1)
prev = None
for it in data:
    for f in ('id', 'content', 'score', 'created_at'):
        if f not in it:
            print(f'   条目缺字段 {f}: {it}'); sys.exit(1)
    s = float(it['score'])
    if prev is not None and s > prev + 1e-9:
        print('   score 非递增被破坏'); sys.exit(1)
    prev = s
print(f'   data 条数={len(data)}')
sys.exit(0)
PY

# 5) 错误 key 必须 401
code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -X POST "${BASE}/v1/memories/search" \
  -H "Authorization: Token definitely-wrong" -H 'Content-Type: application/json' \
  -d "{\"query\":\"x\",\"user_id\":\"${UID_}\",\"top_k\":1}" 2>/dev/null)
[[ "${code}" == "401" ]] && ok "错误 key -> 401" || bad "错误 key -> ${code}（应 401）"

echo
echo "== 结果：通过 ${PASS} 项，失败 ${FAIL} 项 =="
[[ "${FAIL}" == "0" ]]
