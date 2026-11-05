# CueBridge Memory

一个面向 Agent Memory Leaderboard 的长期记忆检索服务，提供 AML 兼容的同步
Add/Search API。系统以「对话轮」为第一级单位索引与打分：`text-embedding-v4`
做稠密检索，`Qwen3-Reranker-4B` 做精排，LLM 组件统一使用 `gpt-4o-mini`。

当前版本：`v1.1.0`

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
| Search 响应的 `content` part **只能是** `text` / `image_url` | 出口统一物化：库里存的 `image_ref` 全部转成 `image_url`；**未能物化的直接丢弃**，绝不外泄非法类型 |
| 单张图片 ≤ 10 MiB，单响应 ≤ 30 MiB | 图片预算按**原始图片字节**累加（当前 21 MiB）；契约限的是响应 JSON ≤30 MiB，base64 膨胀约 4/3 ⇒ 21 MiB 原始 ≈ 28.7 MB JSON；预算用尽即停止附图（**宁少给，不超限**）|

## 服务器要求

- Linux 服务器；
- Python 3.10 至 3.12，或 Docker；
- **一张 ≥9 GB 显存的 GPU** 用于本地 4B 重排器（纯 API 部署可换成不含本地重排器的组合）；

## Python 部署

```bash
git clone https://github.com/lianghanyang/CueBridge-Memory.git
cd CueBridge-Memory
git checkout v1.1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 要先设置 AMC_EMBEDDING_API_KEY / AMC_LLM_RERANK_API_KEY / MEMORY_SYSTEM_KEY
bash scripts/start_memory_api.sh
curl -s localhost:8000/health | head -c 200
```

启动脚本带**重复进程守卫**：只允许一个 `scripts.serve_memory` 与一个端口占用者，
避免"两个 supervisor 互相抢 GPU / 抢端口"导致的崩溃循环。

## Docker 部署

```bash
docker build -t cuebridge-memory:1.1.0 .
docker run --rm -p 8000:8000 \
  -e AMC_EMBEDDING_API_KEY='sk-...' \
  -e AMC_LLM_RERANK_API_KEY='sk-...' \
  -e MEMORY_SYSTEM_KEY='replace-with-a-long-random-secret' \
  -e AMC_RETRIEVER='hybrid_round_rerank' \
  -e AMC_LLM_RERANK_MODEL='gpt-4o-mini' \
  -e AMC_LLM_RERANK_PROMPT='evidence_set' \
  -e AMC_ROUND_LAYOUT='interleave' \
  -v "$(pwd)/data:/app/data" \
  cuebridge-memory:1.1.0
```

```bash
docker compose up --build -d
docker compose logs -f cuebridge-memory
```

## 赛事 API

### Health

```bash
curl -s https://your-domain.example/v1/memories/health
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
- 单张图片上限 10 MiB，单次 Add 解码后图片总量上限 30 MiB；
- 字段不合规时**不返回 4xx**（契约要求），而是 200 + `success:true` + `note` 说明"已接收但未入库的字段"。

### Search

```http
POST /v1/memories/search       （等价路径：/search）
Authorization: Token <MEMORY_SYSTEM_KEY>
Content-Type: application/json

{"query": "用户偏好哪个操作系统？", "user_id": "u_123", "top_k": 10}
```

- `query` 支持纯字符串或 `ContentPart[]`（可内含 `image_url`，即多模态查询）；
- 响应 `data` 每条含 `id / content / score / created_at`；`score` **可选、不承诺单调**；
- **`content` 是按消息顺序交错的 part 序列**：`text`（该条消息正文，含 caption）与
  `image_url`（该条消息的图）交替出现，**图片紧跟在它自己的描述之后**——
  这让答案端能把"题面说的 wider view"对到具体那张图，而不是面对一堆无归属的图；
- 图片以 base64 data URI 返回（前 2 张不压缩，其后压缩），单响应 ≤ 30 MiB：

```json
{"data": [{"id": "…",
           "content": [{"type": "text", "text": "Here's a wider view of the same test. Image Caption: …"},
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}],
           "score": 0.83, "created_at": "2026-10-16T10:00:00+08:00"}]}
```

**超时语义**：一次 Search 最多跑 85 秒（`AMC_SLOW_RETURN_S`）。若在预算内无法完成
**全部**检索环节（含最后的重排合并），返回 **`408`**（平台可重试 = 加时间），
**绝不返回"缺环节的 200"**——缺环节的 200 会直接丢分。

鉴权接受 `Authorization: Token <key>`、`Authorization: Bearer <key>`、裸 key 与 `X-Api-Key`；
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
| `AMC_ROUND_LAYOUT` | `interleave` | `interleave`=按消息原样交错（图跟着它的 caption）；`join`=整轮文本合并（旧行为，可回退） |
| `AMC_ROUND_ITEMS` | `1` | 每轮返回一条代表记录（内容为该轮全文/全图） |
| `AMC_RETURN_IMAGES` | `1` | 是否在响应里返回原图 |
| `AMC_RETURN_IMAGES_MAX` | `50` | 每次响应最多返回多少张图（答案端对图数敏感：>50 张会被上游 400） |
| `AMC_RETURN_IMAGES_MIB` | `21` | 图片体积预算，按**原始图片字节**累加（21 MiB 原始 ≈ 28.7 MB JSON，< 30 MiB 契约上限）|
| `AMC_MERGE_RERANK` | 文件开关 | 最终重排合并（`data/store/merge_rerank.on` 存在即开启） |
| `AMC_MERGE_SYNC_WAIT_S` | `35` | 等待合并重排的上限；时间不足时**压缩等待，不跳过** |
| `AMC_SLOW_RETURN_S` | `85` | 单次 Search 的总时间闸；超时返回 `408` 而非不完整 200 |
| `AMC_LIBRARY` | `1` | 记忆图书馆（离线打标 + 按标签补召回） |
| `AMC_RELATION_SELECT` | `0` | 关系感知证据选择（结构化卡片） |

## v1.1 变更（相对 v1.0.0）

| 变更 | 解决的问题 |
|---|---|
| **轮级证据完整性**：每轮返回整轮全文（不限长度），同轮文本并入而非丢弃 | 旧版整轮只留一条消息，答案句常在 msg#4/5 被去重删掉 |
| **图片按消息交错返回**（`AMC_ROUND_LAYOUT=interleave`） | 旧版把整轮图片全堆在末尾，"哪张图属于哪句 caption"丢失，答案端无法按题面线索选图 |
| **契约合规**：出口把 `image_ref` 全部物化成 `image_url`，未物化的直接丢弃 | 旧版响应里残留 `image_ref`（非法 type），答案端会**整请求 400** |
| **体积预算下调到 21 MiB（原始字节）** | 旧版 28 MiB 原始，base64 膨胀后 JSON 达 **33.7 MB**，**超 30 MiB 契约上限**；现按 30/1.37≈21 取值 |
| **重排合并同步等待 + 超时 408** | 旧版超时后"跳过合并直接 200"，等于丢一个环节；现改为压缩等待、等不到就 408（平台可重试） |
| **重排输入不截断** | 旧版对每条证据截断 1400 字，长轮里第 17/18 条的答案会被切掉 |
| **启动守卫** | 旧版可能同时存在两个 supervisor，互相抢 GPU/端口导致崩溃循环 |
| 移除死代码（`cecr.py`、`ab_answer.py`） | 零引用的模块，避免读者误以为在生产链路里 |

## 公开数据验证

> ⚠️ **先看声明**：下表是 **v1.0.0** 的配对对照结果；**v1.1 未在这两个集上重跑**，
> 因此它**不构成 v1.1 的成绩**，只作为"本项目做过配对实验"的基线记录。

### v1.0.0 的对照实验（保留作为基线记录）

我们在 Mem-Gallery 与 ATM-Bench 上开展配对对照实验，两组使用完全相同的记忆库、
text-embedding-v4（1024 维）向量模型与 top_k=100 的粗检索配置。基线为
text-embedding-v4 稠密召回 + Qwen3-Reranker-4B 轮级精排，不启用 LLM 证据选择；
实验组在基线上叠加关系感知证据选择。

| 数据集（n） | 指标 | 基线 | CueBridge Memory 1.0.0 | 配对 Δ [95% CI] |
|---|---|---|---|---|
| **Mem-Gallery（1,527）** | R@1 | 0.3788 | **0.4081** | +0.0293 [+0.0169, +0.0425] |
| | Hit@1 | 0.5946 | **0.6398** | +0.0452 [+0.0269, +0.0648] |
| | R@n_gold | 0.5381 | **0.5836** | +0.0455 [+0.0315, +0.0596] |
| **ATM-Bench（1,044）** | R@1 | 0.5734 | **0.6023** | +0.0289 [+0.0120, +0.0465] |
| | Hit@1 | 0.6648 | **0.7059** | +0.0412 [+0.0211, +0.0603] |
| | R@n_gold | 0.6502 | **0.6864** | +0.0362 [+0.0185, +0.0542] |

### v1.1.0 的验收指标

v1.1 的改动集中在**契约合规与轮级证据完整性**，而非召回排序，因此验收用工程指标：

| 指标 | 结果 |
|---|---|
| 响应 `content` 只含 `text` / `image_url` | 100%（`image_ref` 计数为 0） |
| 单响应图片数 / 体积 | ≤50 张 / 实测 8.16 MB（h02，50 张） |
| 单响应体积 ≤ 30 MiB | 100%（实测 0.8~3.6 MB） |
| 每张返回图片都能对到它的 caption | 100%（交错布局；旧版为 0%，图全堆在末尾） |
| 超时行为 | 只返回 `408`，不返回缺环节的 `200` |
| 每轮返回时长（整轮全文，不截断） | p50 12~20 s，p90 25~35 s（视题而定） |
| `data` 数组 / `id` 非空 / part 类型只含 `text`+`image_url` | 通过（含边界：空用户 → `{"data": []}`）|
| 图片 base64 data URI / 单图 ≤10 MiB / 单响应 ≤30 MiB | 通过（实测单响应 ≤6.3 MB，50 张图时 8.16 MB）|
| 返回条数 ≤ `top_k` | 通过（top_k=5→5、10→10、100→9~65）|
| `/health` 未鉴权可访问；错误 key → `401` | 通过 |
| `created_at` 为合法时间戳 | 通过 |

> 以上 v1.1 指标是**工程验收**，不是榜单成绩；v1.1 未在 Mem-Gallery / ATM-Bench 上重跑全量，
> 因此上表 v1.0.0 的检索指标仍是本项目唯一的公开集对照结果。

## 测试

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

覆盖 AML 契约（请求/响应）、鉴权（Token/Bearer/裸 key/错误 key）、路径别名、
LLM 输出解析。测试**不下载模型、不访问网络**，可在 CI 中直接运行。

上线前与部署后各跑一次检查：

```bash
bash scripts/preflight_check.sh    # 进程唯一 / 端口 / GPU / env / 数据量
bash scripts/release_now.sh        # 放行：删 hold -> 记 T2 -> 探针（含契约与鉴权）
```

## 已知限制

- **同轮/多轮图片可能互相矛盾**：实测某一轮有 7 张图，其中一张"凳子在色块左侧"、
  另一张"凳子在色块右侧"，而题目问的是后者；给了全部图，答案模型可能按前者回答；
- **自动 caption 未必描述题目所问的细节，而且这是普遍问题、不是个例**：
  实测某题 caption 只写 "Testing a warm terracotta paint color on a living room wall"
  而没提凳子 ⇒ 视觉题的答案端需自行甄别图与文本，**我们只保证结构与图的可获得性**，
  不保证"看图就能答对"；
- **图片多时会被体积预算截断**：预算（21 MiB 原始 ≈ 28.7 MB JSON）用尽即停止附图，
  宁少给、不超契约；
- **答案端表现不在本项目的承诺范围内**：我们只保证检索证据的结构与可获取性，
  不承诺第三方答案模型能据此答对（内部代理评测受模型空间判断能力限制）；

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
| `amc/` | 记忆服务实现（契约、存储、检索、重排、轮级装配、图片物化） |
| `scripts/` | 启动、隧道、上线前检查、放行 |
| `tests/` | 离线单元测试 |
| `Dockerfile`、`compose.yaml` | 容器化部署 |
| [MANIFEST.md](MANIFEST.md) | 交付物清单与复现路径 |
