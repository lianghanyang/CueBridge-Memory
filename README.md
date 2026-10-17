# CueBridge Memory

一个面向 Agent Memory Leaderboard 的长期记忆检索服务，提供 AML 兼容的同步
Add/Search API。系统以「对话轮」为第一级单位索引与打分：`text-embedding-v4`
做稠密检索，`Qwen3-Reranker-4B` 做精排，LLM 组件统一使用 `gpt-4o-mini`，
并把查询图片转成检索用文字描述接入纯文本链路。

当前版本：`v1.0.0`

## 系统边界

项目包含两条互相独立、但共用同一份 `text-embedding-v4` 的链路：

```text
赛事服务：Add -> SQLite 持久化（消息 + 图片 blob + 向量）-> user_id 隔离 -> Search -> 记忆证据列表

离线评测：公开集 JSONL -> 检索/重排/LLM 证据选择 -> Top-k -> 指标与配对显著性检验
```

赛事服务只返回记忆证据，**不生成最终答案**；官方平台负责 Answer 和 Eval。


项目**为记忆侧**，合规约束如下：

| 要求 | 本系统做法 |
|---|---|
| Embedding **必须** `text-embedding-v4` | 全链路唯一向量模型（1024 维），未替换、未混用；图片不进入向量层 |
| LLM 相关组件**必须** `gpt-4o-mini` | 证据选择档、查询图描述、关系选择三处调用统一走 `gpt-4o-mini` |
| Reranker **不限制** | 使用 `Qwen3-Reranker-4B`（cross-encoder，轮级打分） |

## 服务器要求

- Linux 服务器；
- Python 3.10 至 3.12，或 Docker；
- **一张 ≥9 GB 显存的 GPU** 用于本地 4B 重排器（纯 API 部署可换成不含本地重排器的组合）；

## Python 部署

```bash
git clone https://github.com/lianghanyang/CueBridge-Memory.git
cd CueBridge-Memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 要先设置 DASHSCOPE_API_KEY / gpt-4o-mini KEY / MEMORY_SYSTEM_KEY
bash scripts/start_memory_api.sh      
curl -s localhost:8000/health | head -c 200
```

## Docker 部署

```bash
docker build -t cuebridge-memory:1.0.0 .
docker run --rm -p 8000:8000 \
  -e AMC_EMBEDDING_API_KEY='sk-...' \
  -e AMC_LLM_RERANK_API_KEY='sk-...' \
  -e MEMORY_SYSTEM_KEY='replace-with-a-long-random-secret' \
  -e AMC_RETRIEVER='hybrid_round_rerank' \
  -e AMC_LLM_RERANK_MODEL='gpt-4o-mini' \
  -e AMC_LLM_RERANK_PROMPT='evidence_set' \
  -v "$(pwd)/data:/app/data" \
  cuebridge-memory:1.0.0
```

```bash
docker compose up --build -d
docker compose logs -f cuebridge-memory
```

## 赛事 API

### Health

```bash
curl -s https://memory.cuebridge.top/v1/memories/health
# {"status":"ok","retriever":{...},"stats":{"rows":...}}
```

### Add

```http
POST /v1/memories/add       
Authorization: Token <MEMORY_SYSTEM_KEY>
Content-Type: application/json

{
  "user_id":    "u_123",               
  "session_id": "s_456",                
  "request_id": "r_789",               
  "messages": [
    {"role": "user",      "content": "…",                      "timestamp": 1760000000000},
    {"role": "assistant", "content": [{"type": "text", "text": "…"}], "timestamp": 1760000005000}
  ]
}
```

- `content` 支持纯字符串或 `ContentPart[]`（`text` / `image_url` 两种 part）；
- `image_url.url` 支持内联 base64 data URI（JPEG/PNG/WebP）与 http(s) 外链；
- `timestamp` 为整数 Unix 毫秒；
- 单张图片上限 10 MiB，单次 Add 解码后图片总量上限 30 MiB。

### Search

```http
POST /v1/memories/search       （等价路径：/search）
Authorization: Token <MEMORY_SYSTEM_KEY>
Content-Type: application/json

{"query": "用户偏好哪个操作系统？", "user_id": "u_123", "top_k": 10}
```

- `query` 支持纯字符串或 `ContentPart[]`（可内含 `image_url`，即多模态查询）；
- 响应保证 `data` 按 `score` 非递增，每条含 `id/content/score/created_at`：

```json
{"data": [{"id": "…", "content": "…", "score": 0.83, "created_at": "2026-10-16T10:00:00+08:00"}]}
```

鉴权接受 `Authorization: Token <key>`、`Authorization: Bearer <key>` 与裸 key 三种形式；
key 错误或缺失返回 `401`。

## 配置项

全部通过环境变量控制，模板见 `.env.example`。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEMORY_SYSTEM_KEY` | 空 | 对外鉴权 key |
| `AMC_EMBEDDING_MODEL` | `text-embedding-v4` | 比赛要求 |
| `AMC_RETRIEVER` | `hybrid_round_rerank` | dense + 轮通道 + `bm25_anchor` 融合 |
| `AMC_FUSION` | `auto` | 可观测的轮结构做路由 |
| `AMC_RERANKER_MODEL` | 空 | 本地 4B 重排器路径（`Qwen3-Reranker-4B`） |
| `AMC_RERANK_UNIT` | `round` | 以整轮为打分单位 |
| `AMC_LLM_RERANK_MODEL` | `gpt-4o-mini` | 比赛要求 |
| `AMC_RELATION_SELECT` | `0` | 关系感知证据选择（结构化卡片） |

## 公开数据验证

1.0.0 冻结前在**两个公开数据集**上做了配对验证，全部经官方 `MemoryService.search`（与线上服务同一路径），
两个系统使用同一批记忆、同一 `text-embedding-v4`（1024 维）与同一 `top_k=100` 的检索口径，
唯一差别是基线不启用 LLM 证据选择档。该实验没有读取或分析任何官方 Smoke 私有负载。

| 数据集（n） | 指标 | 基线 | CueBridge Memory 1.0.0 | 配对 Δ [95% CI] |
|---|---|---|---|---|
| **Mem-Gallery（1,527）** | R@1 | 0.3788 | **0.4081** | +0.0293 [+0.0169, +0.0425] |
| | Hit@1 | 0.5946 | **0.6398** | +0.0452 [+0.0269, +0.0648] |
| | R@n_gold | 0.5381 | **0.5836** | +0.0455 [+0.0315, +0.0596] |
| **ATM-Bench（1,044）** | R@1 | 0.5734 | **0.6023** | +0.0289 [+0.0120, +0.0465] |
| | Hit@1 | 0.6648 | **0.7059** | +0.0412 [+0.0211, +0.0603] |
| | R@n_gold | 0.6502 | **0.6864** | +0.0362 [+0.0185, +0.0542] |

逐题比较（Mem-Gallery 全量）：1.0.0 胜 151、平 1,294、负 82（R@1）；
ATM-Bench：胜 79、平 929、负 36。以上所有 95% 置信区间均不含 0。
以上结果仅用于公开数据回归和工程选型，不是官方榜单成绩。

> 按路拆分：文本路 1,040 条 R@1 0.5147 → 0.5531（+0.0384）；图像证据路 487 条 0.0886 → 0.0986（+0.0099）。
> 图像证据路的检索本身很困难（R@1 约 0.10），是后续最有提升空间的方向。

## 测试

```bash
PYTHONPATH=. python -m pytest tests/ -q      # 20 例，约 3 秒
```

覆盖 AML 契约（请求/响应/分数非递增）、鉴权（Token/Bearer/裸 key/错误 key）、
路径别名、LLM 输出解析（含代码块包裹、尾随逗号、截断数组）。测试**不下载模型、
不访问网络**，可在 CI 中直接运行。

上线前与部署后各跑一次检查：

```bash
bash scripts/aml_preflight.sh     # key / 嵌入维度 1024 / LLM 文本 / 视觉 / GPU 显存 / 记忆库
bash scripts/aml_selfcheck.sh     # 公网端到端：health -> add -> search -> 契约 -> 错误 key 401
```

## 正式评测前检查

1. 固定 Git Commit、镜像标签、依赖、模型与配置；
2. `pytest` 与本地 Smoke 全部通过；
3. 重启恢复、并发、长时间运行通过；
4. 公开数据集无缺失样本，跨用户泄漏率为零；
5. 公网 HTTPS 的 Health/Add/Search 可从外部访问（含**慢图像查询**，确认无中间层超时）；
6. 使用官方 Smoke 修正集成差异；
7. 最后才启动 Full。

## 交付物与文档

| 路径 | 说明 |
|---|---|
| `amc/` | 记忆服务实现（契约、存储、检索、重排、服务装配） |
| `scripts/` | 启动、隧道、上线前检查、公网自检 |
| `tests/` | 离线单元测试 |
| `Dockerfile`、`compose.yaml` | 容器化部署 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构与各模块收益（局部 + 整体） |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | 启停、健康检查、故障处理、合规注意 |
| [docs/DEPLOY_STATUS.md](docs/DEPLOY_STATUS.md) | 当前部署状态、实测指标与踩坑记录 |
| [MANIFEST.md](MANIFEST.md) | 交付物清单与复现路径 |


