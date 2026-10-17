# 运维手册（Runbook）

## 1. 启停

```bash
# 启动（带崩溃自愈守护；日志在 stdout）
bash scripts/start_memory_api.sh

# 停止
pkill -f "scripts.serve_mem"      # 守护循环也会随之退出

# 单独起（无守护，便于调试）
PYTHONPATH=. python -m scripts.serve_memory --host 0.0.0.0 --port 8000
```

## 2. 健康检查与自检

```bash
curl -s localhost:8000/health | head -c 300
bash scripts/aml_selfcheck.sh                 # 本地
bash scripts/aml_selfcheck.sh                 # 自动用 public_url.txt 里的公网地址
```

自检失败时的排查顺序：
1. `add` 返回 **422** → 请求体字段不符契约（需 `user_id`/`session_id`/`request_id` + `messages`，`timestamp` 为整数毫秒）。
2. `add`/`search` 返回 **500** → 看服务日志；常见原因：嵌入 key 缺失（`AMC_EMBEDDING_API_KEY`）、图片超 30 MiB。
3. `search` 返回 **401** → 调用方需带 `Authorization: Token <MEMORY_SYSTEM_KEY>`。
4. `search` 慢 → 4B 精排是主要耗时；可临时 `AMC_RERANKER=none` 做链路验证。

## 3. 公网接入

### 3.1 临时（推荐用于评测期快速上线）

```bash
bash scripts/start_public_tunnel.sh           # cloudflared quick tunnel
# 或
AMC_TUNNEL=ssh bash scripts/start_public_tunnel.sh
```

- 产出 `data/reports/deploy/public_url.txt`（含 add/search/health 完整地址）。
- ⚠️ **临时域名重启会变**，每次重启都要同步更新报名表 URL。

### 3.2 固定域名（长期评测建议）

在校园网出口做端口映射 / 反向代理（Nginx、Caddy）到 `127.0.0.1:8000`，
或用 cloudflared **命名隧道**（需 Cloudflare 账号，域名固定）：

```bash
cloudflared tunnel login
cloudflared tunnel create roundweave
cloudflared tunnel route dns roundweave memory.example.com
cloudflared tunnel run --url http://127.0.0.1:8000 roundweave
```

## 4. 合规检查清单（每次上线前过一遍）

- [ ] `AMC_EMBEDDING_MODEL=text-embedding-v4`，且**没有**其它向量模型参与检索。
- [ ] 所有 LLM 调用都指向 `gpt-4o-mini`（证据选择档 / 查询图描述 / 关系选择）。
- [ ] Reranker 仍是 cross-encoder（`AMC_RERANKER=qwen3`）。
- [ ] 未改动 answer 侧任何文件（`loopx/`、answer 服务）。
- [ ] `bash scripts/aml_selfcheck.sh` 全绿。
- [ ] 报名表的 Add/Search URL 与 `public_url.txt` 一致。

## 5. 已知故障与处置

| 现象 | 原因 | 处置 |
|---|---|---|
| `add` 500 `string indices must be integers` | 旧版本未处理"纯字符串 content" | 已修（`amc/store.py` 归一化为 text part） |
| 查询图档静默失效（image 路指标骤降） | VLM 输入图片无像素上限 → 上下文溢出 HTTP 400 | 已修（`amc/imaging.py::vlm_ready_image`，长边 ≤1024 / ≤0.8 MB） |
| `search` 某条 content 为占位文本 | 记忆的图片 blob 缺失 | 设计如此（优雅降级），不影响排序 |
| 服务反复重启 | CUDA 致命错误 | 启动脚本会强制退出并重部署（不静默退化）；检查显存是否被其它任务挤占 |
| gpt-4o-mini 返回 503 `No available channel` | 网关侧无可用通道 | 需在网关侧启用通道；期间 LLM 档会自动退化为纯 4B 排序（不影响可用性） |

## 6. 资源需求

- GPU：4B cross-encoder ≈ 9 GB 显存（bfloat16）；无 GPU 时可用 `AMC_RERANKER=none` 跑通链路。
- CPU/内存：8 核 / 16 GB 足够（检索为 CPU + 缓存密集）。
- 网络：DashScope（嵌入）与 `gpt-4o-mini` 网关出网。
