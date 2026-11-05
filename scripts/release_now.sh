#!/usr/bin/env bash
# 放行脚本：删 hold → 记录 T2 → 探针自测 → 输出 marker
# 用法：bash scripts/release_now.sh
# 注意：本脚本**不改 amc/ 代码**，不改变 BUILD
set -uo pipefail
cd ~/agent_memory
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
PY=python3
HOLD=data/store/search_hold.txt

echo "=== 0) 放行前状态 ==="
[ -f "$HOLD" ] && echo "  hold: **存在**（/search 503）" || echo "  hold: 不存在（可能已放行）"
printf "  BUILD: %s\n" "$($PY -c "import sys;sys.path.insert(0,'.');from amc.build_id import BUILD;print(BUILD)" 2>/dev/null)"
printf "  marker(放行前): %s\n" "$(wc -l < data/logs/serve_supervisor.log)"

# 放行前先确认：hold 真的在拦（不带 x-internal-test 应为 503）
PRE=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST http://127.0.0.1:8020/v1/memories/search \
  -H 'Authorization: Token dev-memory-key' -H 'Content-Type: application/json' \
  -d '{"query":"hold-check","user_id":"eval:local-v1:atmbench:owner","top_k":5}' 2>/dev/null)
printf "  放行前不带 internal-test 的请求: %s（期望 503）\n" "$PRE"

echo "=== 1) 删 hold ==="
if [ "$DRY" = "1" ]; then
  echo "  [--dry] **不删 hold**（演练模式：后续探针会因 hold 而 503，属预期）"
else
  rm -f "$HOLD"
fi
T2=$(date '+%Y-%m-%dT%H:%M:%S%z')
[ "$DRY" = "1" ] || echo "$T2" > data/store/release_t2.txt
[ "$DRY" = "1" ] && echo "  （演练）未删除 hold" || echo "  ✅ 已删除 search_hold.txt"
[ "$DRY" = "1" ] && echo "  T2（演练，不作数）= $T2" || echo "  **T2（服务就绪时间）= $T2**"
printf "  marker(放行后): %s\n" "$(wc -l < data/logs/serve_supervisor.log)"

echo "=== 2) 探针 A：**不带 x-internal-test**（唯一能证明 hold 已删的探针）==="
PA=/tmp/rel_probe_A.json
CA=$(curl -s -o $PA -w '%{http_code}' -m 150 -X POST http://127.0.0.1:8020/v1/memories/search \
  -H 'Authorization: Token dev-memory-key' -H 'Content-Type: application/json' \
  -d '{"query":"When did I start working at Louisa Coffee?","user_id":"eval:local-v1:atmbench:owner","top_k":100}' 2>/dev/null)
$PY - "$PA" "$CA" <<'PYEOF'
import json,sys
p,code=sys.argv[1],sys.argv[2]
try: d=json.load(open(p)).get('data') or []
except Exception: d=[]
ids=[str(x.get('id')) for x in d]
first=ids[0] if ids else ''
ok_first = first.startswith('m_') or (first=='amc_aggregate_note' and len(ids)>1 and ids[1].startswith('m_'))
rids=[str(x.get('request_id') or x.get('id')) for x in d]
print('  探针A: %s ｜ n=%-3d ｜ 首条 %-26s ｜ 首条合规=%s ｜ request_id 唯一=%s' % (
    code, len(d), first[:26], ok_first, len(rids)==len(set(rids))))
print('  ⇒ 期望：**200**（若 503 说明 hold 未删成功）')
PYEOF

echo "=== 3) 探针 B/C：带 x-internal-test（验证服务面稳定）==="
for i in 1 2; do
  PB=/tmp/rel_probe_$i.json
  CB=$(curl -s -o $PB -w '%{http_code}' -m 150 -X POST http://127.0.0.1:8020/v1/memories/search \
    -H 'Authorization: Token dev-memory-key' -H 'Content-Type: application/json' -H 'x-internal-test: 1' \
    -d '{"query":"When did I start working at Louisa Coffee?","user_id":"eval:local-v1:atmbench:owner","top_k":100}' 2>/dev/null)
  $PY - "$PB" "$CB" "$i" <<'PYEOF'
import json,sys
p,code,i=sys.argv[1],sys.argv[2],sys.argv[3]
try: d=json.load(open(p)).get('data') or []
except Exception: d=[]
ids=[str(x.get('id')) for x in d]
first=ids[0] if ids else ''
ok_first = first.startswith('m_') or (first=='amc_aggregate_note' and len(ids)>1 and ids[1].startswith('m_'))
rids=[str(x.get('request_id') or x.get('id')) for x in d]
print('  探针B%s: %s ｜ n=%-3d ｜ 首条 %-26s ｜ 首条合规=%s ｜ 无重复=%s' % (
    i, code, len(d), first[:26], ok_first, len(rids)==len(set(rids))))
PYEOF
done

echo "=== 4) 公网可达（不带 internal-test，应为 200）==="
printf "  公网 health: %s\n" "$(curl -s -o /dev/null -w '%{http_code}' -m 25 https://your-domain.example/v1/memories/health 2>/dev/null || echo DOWN)"

echo "=== 5) 结果判读 ==="
echo "  · 探针A=200 且 首条合规=m_*  ⇒ **放行成功**"
echo "  · 探针A=503                 ⇒ hold 未删成功，检查 $HOLD"
echo "  · 探针A=200 但 n=0          ⇒ 该用户不在库（换已知有数据的用户）"
echo "=== 完成 ==="
