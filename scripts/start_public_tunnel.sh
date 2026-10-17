#!/usr/bin/env bash
# CueBridge Memory：把本机记忆服务（默认 127.0.0.1:8000）暴露到公网，输出可登记的 HTTPS 地址。
#
# 四种模式（AMC_TUNNEL=…）：
#   cloudflared       默认。quick tunnel，**域名每次重启都变**，仅适合临时联调。
#   serveo            免账号，但需**一次性**用浏览器注册本机 SSH 公钥后，才可固定子域。
#                     首次连接会打印注册链接（console.serveo.net/ssh/keys?add=SHA256:…）。
#                     用法：AMC_TUNNEL=serveo AMC_TUNNEL_SUBDOMAIN=cuebridge bash <本脚本>
#                     注册后得到：https://cuebridge.serveo.net
#   cloudflared-named 正式固定域名（需自有域名 + 一次性 `cloudflared tunnel login`）
#                     用法：AMC_TUNNEL=cloudflared-named AMC_TUNNEL_NAME=cuebridge \
#                           AMC_PUBLIC_DOMAIN=memory.example.com bash <本脚本>
#   ngrok             需 authtoken（可选保留域名）
#                     用法：AMC_TUNNEL=ngrok AMC_NGROK_AUTHTOKEN=xxx [AMC_NGROK_DOMAIN=xxx.ngrok-free.app]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT}"
PORT="${AMC_PORT:-8000}"
TUNNEL="${AMC_TUNNEL:-cloudflared}"
SUB="${AMC_TUNNEL_SUBDOMAIN:-cuebridge}"
OUT_DIR="${ROOT}/data/reports/deploy"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/tunnel.log"
URL_FILE="${OUT_DIR}/public_url.txt"
SSH_KEY="${AMC_TUNNEL_SSH_KEY:-$HOME/.ssh/id_ed25519}"

: > "${LOG}"; : > "${URL_FILE}"

extract_url() {
  # 排除 serveo 的**控制台/提示**域名，只取真正转发的地址；
  # 优先固定子域（<sub>.serveo.net），其次随机地址与其它隧道。
  local hits
  hits=$(grep -oE 'https?://[a-zA-Z0-9._-]+\.(trycloudflare\.com|lhr\.life|localhost\.run|serveo\.net|serveousercontent\.com|ngrok-free\.app|ngrok\.io)' \
    "${LOG}" 2>/dev/null | grep -v 'console\.serveo\.net' | grep -v 'serveo\.net/ssh' | sort -u)
  [[ -z "${hits}" ]] && return 0
  echo "${hits}" | grep -E "\.serveo\.net$" | head -1 && return 0
  echo "${hits}" | head -1
}
wait_for_url() {
  local deadline=$(( $(date +%s) + $1 ))
  while [[ $(date +%s) -lt ${deadline} ]]; do
    local u; u="$(extract_url)"; [[ -n "${u}" ]] && { echo "${u}"; return 0; }
    sleep 2
  done
  return 1
}

case "${TUNNEL}" in
  cloudflared)
    command -v cloudflared >/dev/null 2>&1 || { echo "[error] 未安装 cloudflared" >&2; exit 2; }
    echo "[tunnel] cloudflared quick tunnel -> http://127.0.0.1:${PORT}（域名临时）"
    setsid nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" \
      >>"${LOG}" 2>&1 < /dev/null &
    ;;
  serveo)
    command -v ssh >/dev/null 2>&1 || { echo "[error] 未安装 ssh" >&2; exit 2; }
    if [[ ! -f "${SSH_KEY}" ]]; then
      mkdir -p "$(dirname "${SSH_KEY}")"
      ssh-keygen -t ed25519 -N '' -f "${SSH_KEY}" -q
      echo "[serveo] 已生成 SSH key：${SSH_KEY}.pub"
    fi
    echo "[tunnel] serveo -> http://127.0.0.1:${PORT}，请求子域 ${SUB}"
    # 用 autossh（若可用）自动重连：比赛期隧道断了要能自愈
    SSH_BIN=ssh
    if command -v autossh >/dev/null 2>&1 && [[ "${AMC_TUNNEL_AUTOSSH:-1}" == "1" ]]; then
      SSH_BIN=autossh
      echo "[tunnel] 使用 autossh 自动重连（AMC_TUNNEL_AUTOSSH=0 可关闭）"
    fi
    setsid nohup "${SSH_BIN}" -M 0 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -i "${SSH_KEY}" -R "${SUB}:80:localhost:${PORT}" serveo.net \
      >>"${LOG}" 2>&1 < /dev/null &
    echo "[serveo] 若日志提示需注册公钥：打开 https://console.serveo.net/ssh/keys 用 Google/GitHub 登录，"
    echo "         把本机公钥 $( [[ -f ${SSH_KEY}.pub ]] && cut -d' ' -f1-2 ${SSH_KEY}.pub || echo ${SSH_KEY}.pub ) 加进去（一次性）。"
    ;;
  cloudflared-named)
    command -v cloudflared >/dev/null 2>&1 || { echo "[error] 未安装 cloudflared" >&2; exit 2; }
    : "${AMC_TUNNEL_NAME:?需要 AMC_TUNNEL_NAME}"
    : "${AMC_PUBLIC_DOMAIN:?需要 AMC_PUBLIC_DOMAIN}"
    if [[ ! -f "$HOME/.cloudflared/cert.pem" ]]; then
      echo "[error] 尚未登录 Cloudflare：请先执行 cloudflared tunnel login（浏览器一次性登录）" >&2
      exit 2
    fi
    if ! cloudflared tunnel info "${AMC_TUNNEL_NAME}" >/dev/null 2>&1; then
      echo "[tunnel] 创建命名隧道 ${AMC_TUNNEL_NAME}"
      cloudflared tunnel create "${AMC_TUNNEL_NAME}" >>"${LOG}" 2>&1 || true
      cloudflared tunnel route dns "${AMC_TUNNEL_NAME}" "${AMC_PUBLIC_DOMAIN}" >>"${LOG}" 2>&1 || true
    fi
    echo "[tunnel] cloudflared 命名隧道 ${AMC_TUNNEL_NAME} -> https://${AMC_PUBLIC_DOMAIN}"
    setsid nohup cloudflared tunnel --no-autoupdate run --url "http://127.0.0.1:${PORT}" \
      "${AMC_TUNNEL_NAME}" >>"${LOG}" 2>&1 < /dev/null &
    ;;
  ngrok)
    command -v ngrok >/dev/null 2>&1 || { echo "[error] 未安装 ngrok" >&2; exit 2; }
    [[ -n "${AMC_NGROK_AUTHTOKEN:-}" ]] && ngrok config add-authtoken "${AMC_NGROK_AUTHTOKEN}" >/dev/null 2>&1 || true
    args=(http "${PORT}" --log=stdout)
    [[ -n "${AMC_NGROK_DOMAIN:-}" ]] && args+=(--domain "${AMC_NGROK_DOMAIN}")
    echo "[tunnel] ngrok -> http://127.0.0.1:${PORT}${AMC_NGROK_DOMAIN:+（固定域名 ${AMC_NGROK_DOMAIN}）}"
    setsid nohup ngrok "${args[@]}" >>"${LOG}" 2>&1 < /dev/null &
    ;;
  *)
    echo "[error] 未知 AMC_TUNNEL=${TUNNEL}（cloudflared|serveo|cloudflared-named|ngrok）" >&2; exit 2
    ;;
esac

if [[ "${TUNNEL}" == "cloudflared-named" ]]; then
  BASE="https://${AMC_PUBLIC_DOMAIN}"
else
  BASE="$(wait_for_url "${AMC_TUNNEL_WAIT:-90}")" || {
    echo "[error] 90s 内没拿到公网地址，日志尾部：" >&2; tail -20 "${LOG}" >&2; exit 1
  }
fi

{
  echo "PUBLIC_BASE=${BASE}"
  echo "ADD_URL=${BASE}/v1/memories/add"
  echo "SEARCH_URL=${BASE}/v1/memories/search"
  echo "HEALTH_URL=${BASE}/v1/memories/health"
  echo "TUNNEL_MODE=${TUNNEL}"
  echo "GENERATED_AT=$(date -Iseconds)"
} | tee "${URL_FILE}"

echo
echo "[tunnel] 已写入 ${URL_FILE}"
case "${TUNNEL}" in
  cloudflared) echo "[tunnel] ⚠️ quick tunnel 域名重启会变；正式提交请用 serveo / cloudflared-named / ngrok" ;;
  serveo)      echo "[tunnel] 注册公钥后地址为 https://${SUB}.serveo.net（名字固定；隧道进程需常驻）" ;;
  cloudflared-named) echo "[tunnel] 固定域名 https://${AMC_PUBLIC_DOMAIN}（可登记到报名表）" ;;
esac
