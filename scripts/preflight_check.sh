#!/usr/bin/env bash
# 复测前自检（只读 + 可选的清理）。用法：bash scripts/preflight_check.sh [--fix]
set -uo pipefail
FIX=0; [ "${1:-}" = "--fix" ] && FIX=1
ok=0; bad=0
chk() { # chk <描述> <命令> <期望>
  local d="$1"; shift; local exp="$1"; shift
  local got; got=$(eval "$@" 2>/dev/null || echo ERR)
  if [ "$got" = "$exp" ]; then printf "  ✅ %-34s %s\n" "$d" "$got"; ok=$((ok+1));
  else printf "  ❌ %-34s 期望=%s 实际=%s\n" "$d" "$exp" "$got"; bad=$((bad+1)); fi
}
echo "=== 1. 进程唯一性 ==="
N=$(ps -eo args --no-headers | awk '/scripts\.serve_memory/ && !/awk/ {c++} END{print c+0}')
printf "  %-36s %s\n" "serve_memory 进程数" "$N"
[ "$N" = "1" ] && ok=$((ok+1)) || { bad=$((bad+1)); [ "$FIX" = "1" ] && bash /tmp/cleanup_wm.sh; }
NS=$(ps -eo args --no-headers | awk '/start_memory_api\.sh/ && !/awk/ {c++} END{print c+0}')
printf "  %-36s %s\n" "start_memory_api 进程数" "$NS"
[ "$NS" -le 2 ] && ok=$((ok+1)) || bad=$((bad+1))

echo "=== 2. 健康与端口 ==="
chk "8000 /health" "200" "curl -s -o /dev/null -w '%{http_code}' -m 15 http://127.0.0.1:8000/health"
chk "8020 /health" "200" "curl -s -o /dev/null -w '%{http_code}' -m 15 http://127.0.0.1:8020/health"
chk "公网 /health" "200" "curl -s -o /dev/null -w '%{http_code}' -m 20 https://your-domain.example/v1/memories/health"

echo "=== 3. BUILD 与 env ==="
B=$(cd ~/agent_memory && python3 -c "import sys;sys.path.insert(0,'.');from amc.build_id import BUILD;print(BUILD)" 2>/dev/null)
printf "  %-36s %s\n" "盘上 BUILD" "$B"
chk "CUDA_VISIBLE_DEVICES" "1" "grep -oP '(?<=^export CUDA_VISIBLE_DEVICES=)\S+' .env.prod"
chk "ROUND_ITEMS" "1" "grep -oP '(?<=^export AMC_ROUND_ITEMS=)\S+' .env.prod"
chk "FINAL_DEDUP_ROUND" "1" "grep -oP '(?<=^export AMC_FINAL_DEDUP_ROUND=)\S+' .env.prod"
chk "SLOW_RETURN_S" "85" "grep -oP '(?<=^export AMC_SLOW_RETURN_S=)\S+' .env.prod"
chk "SLOW_FALLBACK" "1" "grep -oP '(?<=^export AMC_SLOW_FALLBACK=)\S+' .env.prod || echo 1"
chk "merge_rerank.on 不存在" "absent" "[ -f data/store/merge_rerank.on ] && echo present || echo absent"

echo "=== 4. hold 与数据 ==="
chk "search_hold.txt" "present" "[ -f data/store/search_hold.txt ] && echo present || echo absent"
chk "slow_gate_s.txt 不存在" "absent" "[ -f data/store/slow_gate_s.txt ] && echo present || echo absent"
M=$(cd ~/agent_memory && python3 -c "
import sqlite3;c=sqlite3.connect('file:data/store/memory.db?mode=ro',uri=True)
print(c.execute('select count(*) from memories').fetchone()[0])" 2>/dev/null)
printf "  %-36s %s\n" "memories 条数" "$M"

echo "=== 5. GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sed -n '2p')
RUNNING=$(ps -eo args --no-headers | awk '/scripts\.serve_memory/ && !/awk/ {c++} END{print c+0}')
printf "  %-36s %s MiB（我们的服务%s）\n" "GPU1 空闲" "${FREE:-?}" "$([ "$RUNNING" -gt 0 ] && echo 已在跑，含我们的显存 || echo 未运行)"
if [ "$RUNNING" -gt 0 ]; then
  echo "  ✅ GPU1 已承载我们的服务（16~17GB 已用属正常）"; ok=$((ok+1))
elif [ "${FREE:-0}" -gt 10000 ]; then
  echo "  ✅ GPU1 空闲 ${FREE} MiB ≥ 10GB，可启动"; ok=$((ok+1))
else
  echo "  ⚠️ GPU1 空闲 ${FREE} MiB < 10GB ⇒ 启动可能 OOM"; bad=$((bad+1))
fi
# GPU2 已知不可用（device-side assert），提醒不要切
echo "  ⚠️ 注意：GPU2 实测会触发 CUDA device-side assert ⇒ 不要切过去"

echo
echo "=== 结果：通过 $ok ｜ 问题 $bad ==="
[ "$bad" -eq 0 ] && echo "✅ 可以放行" || echo "❌ 先修复上面的问题"
